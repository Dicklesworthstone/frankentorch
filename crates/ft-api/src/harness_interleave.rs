//! Arm interleaving for head-to-head harnesses (`frankentorch-6atx2`).
//!
//! # The defect this module exists to remove
//!
//! `gauntlet_lane_sweep_h2h` used to run its **entire** PyTorch arm to
//! completion before the first FrankenTorch lane started. Every ratio it
//! printed was same-invocation and same-host but **not interleaved**: the two
//! arms were sampled tens of seconds apart, so any load shift in that gap landed
//! entirely, and undetectably, in the ratio.
//!
//! A contention preflight does not cover this. It certifies only that nothing
//! heavy sat on the placement CPUs at the *instant* sampling began; it cannot see
//! a peer job starting one second later, and it cannot see page-cache or thermal
//! history at all. During the REPS-16 re-bank that was not hypothetical — a
//! neighbouring project's oracle cycled between idle and 4000-6900% CPU while
//! load average moved between 6 and 69 across 36 runs.
//!
//! Repetition plus a median *averages the gap's effect down*; nothing bounds it.
//! Interleaving removes the defect instead of averaging it.
//!
//! # Why the incumbent becomes a co-process
//!
//! You cannot interleave with a child that computes everything and exits. The
//! incumbent arm is therefore driven as a **request/response co-process**: it
//! sets up, warms up, announces [`READY_MARKER`], and then returns exactly one
//! timed sample per [`REQUEST_PREFIX`] line. The harness alternates — one
//! incumbent sample, then our samples for that same lane, within one round.
//!
//! # The estimator must not move
//!
//! Interleaving is a change to *when* samples are taken, not *how many* or
//! *which statistic* summarises them. A harness that also switched the
//! incumbent's estimator (say min-of-7 to min-of-16) would shift the ratio's
//! level, and its before/after set could no longer isolate the effect of
//! interleaving. [`incumbent_sample_rounds`] exists for exactly this: it spreads
//! a **fixed** sample count evenly across the rounds, so the estimator is
//! bit-for-bit the one the banked set used while the samples are now interleaved.

/// The line the incumbent co-process prints once setup and warmups are done.
///
/// The driver blocks on this. Without it the first request would race the
/// child's import and warmup cost straight into the first sample.
pub const READY_MARKER: &str = "PT_READY";

/// Prefix of a single timed sample returned by the incumbent co-process.
pub const SAMPLE_MARKER: &str = "PT_SAMPLE ";

/// Prefix the driver writes to ask the co-process for one sample of a lane.
pub const REQUEST_PREFIX: &str = "SAMPLE ";

/// The line that asks the co-process to exit its request loop.
pub const QUIT_REQUEST: &str = "QUIT";

/// One balanced-square round: incumbent (`true`) and FrankenTorch (`false`).
///
/// Every arm occupies two slots in each half and every slot position equally,
/// so a monotonic drift or a transient peer load is common-mode rather than a
/// bias toward either implementation.  Consumers still gate each arm's
/// first-half/second-half A/A null before quoting the paired ratio.
pub const BALANCED_SQUARE: [bool; 8] = [true, false, false, true, true, false, false, true];

/// One timed measurement returned by the incumbent arm.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct IncumbentSample<'a> {
    /// Lane the sample belongs to, echoed back so a reply cannot be misfiled.
    pub lane: &'a str,
    /// Wall time for this single forward+backward, in milliseconds.
    pub milliseconds: f64,
    /// Gradient checksum, carried so parity is checked on the same run that
    /// produced the timing rather than on a separate trusting pass.
    pub gradient_checksum: f64,
}

/// Rounds on which the incumbent arm contributes a sample.
///
/// Returns exactly `min(samples, rounds)` round indices, spread as evenly as
/// integer arithmetic allows across `0..rounds`.
///
/// # Why not simply sample every round
///
/// Because that silently changes the incumbent's estimator. The banked set
/// summarised the incumbent as min-of-7; taking 16 samples and reporting
/// min-of-16 is a *different, lower* statistic, which would move the ratio for
/// reasons unrelated to interleaving and destroy the before/after comparison
/// this change has to survive. Fixing the count and spreading it keeps the
/// estimator identical.
///
/// # Why not sample the first `samples` rounds
///
/// That is the original defect in miniature — the incumbent would finish before
/// our arm's later rounds ever ran, reintroducing the very gap being removed.
#[must_use]
pub fn incumbent_sample_rounds(rounds: usize, samples: usize) -> Vec<usize> {
    if rounds == 0 || samples == 0 {
        return Vec::new();
    }
    let samples = samples.min(rounds);
    // Bresenham-style: advance a `samples/rounds` accumulator and emit a round
    // whenever it crosses an integer boundary. Telescoping makes the count
    // exactly `samples`.
    (0..rounds)
        .filter(|&round| (round * samples) / rounds != ((round + 1) * samples) / rounds)
        .collect()
}

/// The request line asking the co-process for one sample of `lane`.
#[must_use]
pub fn sample_request(lane: &str) -> String {
    format!("{REQUEST_PREFIX}{lane}")
}

/// Parse one [`SAMPLE_MARKER`] line from the co-process.
///
/// Returns [`None`] for any line that is not a well-formed sample, including a
/// sample whose time is negative or non-finite — those are broken measurements,
/// and folding them into a median would corrupt the ratio silently.
#[must_use]
pub fn parse_sample_line(line: &str) -> Option<IncumbentSample<'_>> {
    let mut fields = line.trim().strip_prefix(SAMPLE_MARKER)?.split_whitespace();
    let lane = fields.next()?;
    if lane.is_empty() {
        return None;
    }
    let milliseconds: f64 = fields.next()?.parse().ok()?;
    let gradient_checksum: f64 = fields.next()?.parse().ok()?;
    // A negative or NaN duration is not a slow sample, it is a broken one.
    if !milliseconds.is_finite() || milliseconds < 0.0 {
        return None;
    }
    Some(IncumbentSample {
        lane,
        milliseconds,
        gradient_checksum,
    })
}

/// Widest A/A null CI that can still certify a calm sample
/// (`frankentorch-8ieqm`).
///
/// # Why a width veto is needed at all
///
/// The old gate passed a row whenever the bootstrap median-ratio CI bracketed
/// 1.0. But the null is FrankenTorch-vs-FrankenTorch, so any disturbance that
/// scales both arms cancels exactly. What contention actually does to the null
/// is **widen** its CI — and a wider CI brackets 1.0 more easily. The gate
/// therefore got *easier* to pass as the host got noisier: anti-conservative in
/// precisely the condition it is trusted to detect. One banked run measured
/// `max_pool3d` at 29.22x, its FT arm spiking to 3.8x its own median, and the
/// gate PASSED it on `[0.528,1.359]`.
///
/// # Where this number comes from
///
/// It is not fitted to make any particular row pass. The null compares identical
/// code against itself, so its CI should sit tight around unity; 0.60 admits at
/// most a ±30% band. The two observed populations bracket it with margin on both
/// sides: the clean reference run's CI was 0.468 wide, and the disturbed 29.22x
/// row's was 0.831. 0.60 sits between them, ~28% clear of each, and is a round
/// number rather than either endpoint.
pub const MAX_NULL_CI_WIDTH: f64 = 0.60;

/// What an A/A null CI can support about the sample it came from.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NullVerdict {
    /// Centred on unity and tight: the sample is calm enough to quote.
    Calm,
    /// Too noisy to conclude anything — neither certifies nor condemns.
    TooWide,
    /// Tight, but the two identical arms did not agree: a real position or
    /// ordering effect, which is the failure the null exists to catch.
    OffCentre,
}

impl NullVerdict {
    /// Short label for the harness's gate column.
    #[must_use]
    pub fn label(self) -> &'static str {
        match self {
            Self::Calm => "PASS",
            Self::TooWide => "WIDE",
            Self::OffCentre => "FAIL",
        }
    }

    /// Whether a row carrying this verdict may be quoted as a measurement.
    #[must_use]
    pub fn is_quotable(self) -> bool {
        matches!(self, Self::Calm)
    }
}

/// The A/A null's own resolution, measured from the run's own slots — `frankentorch-mdsmm.1`.
///
/// # Why a fixed band cannot work here
///
/// The null is a median over rounds of one warm slot divided by another. Under the hypothesis it
/// tests — "this arm has no within-round position effect" — the warm slots of a round are
/// EXCHANGEABLE, so permuting them destroys any position effect while preserving the lane, the
/// host, the round-to-round variation and the heavy tail. Recomputing the statistic on permuted
/// slots therefore samples it under the null being tested, and every excursion is a FALSE
/// rejection by construction.
///
/// Measured over 864 retained slot dumps (18 guard-gated invocations, two lanes, paired and both
/// isolation modes), the permuted statistic centres at 0.9999 and spans **[0.937, 1.071] at
/// p05-p95**. Against that, the fixed `+/-0.02` band rejects **41% of provably clean arms**, and
/// 0.97-1.03 rejects 27%. That is a floor no host quality and no extra rounds within a run can
/// beat: it is the estimator's own resolution at this round count.
///
/// 38 of those 48 observed nulls sat inside their own permutation band, so after the slot-0 fix
/// there is little systematic left — the residual excursions the rotors kept hitting (0.966,
/// 1.024) are what a clean arm produces at 24 rounds.
///
/// # What this returns instead
///
/// The interval the statistic occupies when there is nothing to detect, for THIS run's rounds.
/// A band derived from the data adapts to round count and lane noise automatically: more rounds
/// narrows it, a noisy lane widens it, and no constant has to be chosen or defended.
///
/// DETERMINISTIC BY CONSTRUCTION. A gate that consults a random number generator is not a gate —
/// the same slots must always produce the same verdict, or a row cannot be re-adjudicated. The
/// permutation order comes from a counter-based hash of the slot values themselves, so it is
/// reproducible from the retained dump alone, with no seed to record.
#[must_use]
pub fn permutation_null_band(rounds: &[[f64; 4]], first_warm: usize, draws: usize) -> (f64, f64) {
    if rounds.is_empty() || draws == 0 {
        return (f64::NEG_INFINITY, f64::INFINITY);
    }
    // Seed from the data: same slots in, same band out, forever.
    let mut seed: u64 = 0x9E37_79B9_7F4A_7C15;
    for r in rounds {
        for v in r {
            seed = seed.rotate_left(5) ^ v.to_bits();
            seed = seed.wrapping_mul(0x1000_0000_01B3);
        }
    }
    let mut nulls = Vec::with_capacity(draws);
    let mut state = seed | 1;
    let mut next = || {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        state
    };
    let mut per_round: Vec<f64> = Vec::with_capacity(rounds.len());
    for _ in 0..draws {
        per_round.clear();
        for r in rounds {
            // Shuffle the WARM slots only; slot 0 is excluded from the statistic upstream when
            // round_warmup is 0, and is a legitimate warm sample above it.
            let mut w: Vec<f64> = r[first_warm..].to_vec();
            for i in (1..w.len()).rev() {
                let j = (next() % (i as u64 + 1)) as usize;
                w.swap(i, j);
            }
            let last = *w.last().expect("non-empty");
            if last > 0.0 && w[0].is_finite() {
                per_round.push(w[0] / last);
            }
        }
        if per_round.is_empty() {
            continue;
        }
        per_round.sort_by(f64::total_cmp);
        let mid = per_round.len() / 2;
        nulls.push(if per_round.len() % 2 == 1 {
            per_round[mid]
        } else {
            f64::midpoint(per_round[mid - 1], per_round[mid])
        });
    }
    if nulls.is_empty() {
        return (f64::NEG_INFINITY, f64::INFINITY);
    }
    nulls.sort_by(f64::total_cmp);
    // 2.5/97.5, i.e. a 5% false-rejection rate by construction — the number the band is CHOSEN
    // to deliver rather than one discovered afterwards.
    let lo = nulls[((nulls.len() as f64) * 0.025) as usize];
    let hi = nulls[(((nulls.len() as f64) * 0.975) as usize).min(nulls.len() - 1)];
    (lo, hi)
}

/// How many permutation draws the gate uses. 999 makes the 2.5/97.5 percentiles stable to the
/// third decimal, and costs microseconds against a lane that takes milliseconds per sample.
pub const NULL_PERMUTATION_DRAWS: usize = 999;

/// Adjudicate an A/A null from its bootstrap CI.
///
/// # Why width is checked before centring
///
/// Because at large widths the bracketing test is close to arbitrary in *both*
/// directions. A banked `conv3d` row failed on `[1.011,4.453]` — a CI 3.44 wide
/// that failed only because its lower bound cleared 1.0 by 0.011. Calling that a
/// detected position effect is no more defensible than calling the 0.831-wide
/// row calm. A sample that noisy supports no verdict at all, so width is
/// answered first and such rows come out [`NullVerdict::TooWide`] rather than
/// being scored either way.
///
/// Non-finite or inverted bounds are treated as [`NullVerdict::TooWide`]: the
/// gate fails closed rather than certifying a row it could not evaluate.
#[must_use]
pub fn adjudicate_null(low: f64, high: f64, max_width: f64) -> NullVerdict {
    if !low.is_finite() || !high.is_finite() || high < low {
        return NullVerdict::TooWide;
    }
    if high - low > max_width {
        return NullVerdict::TooWide;
    }
    if low <= 1.0 && high >= 1.0 {
        NullVerdict::Calm
    } else {
        NullVerdict::OffCentre
    }
}

/// The steps a timed region covers, in order (`frankentorch-574cu`).
///
/// # Why this is a shared constant and not a comment in each arm
///
/// The gauntlet harness used to stop its FrankenTorch timer AFTER computing the
/// gradient checksum, while the PyTorch arm stopped BEFORE its equivalent —
/// `return (time.perf_counter()-s)*1e3, x.grad.sum().item()` evaluates the
/// elapsed term first, so torch's grad sum was outside its measurement. The two
/// arms were therefore not timing the same work. On the `avg_pool2d` lane that
/// checksum — a serial dependent-add chain over 2M f64 — was 1.599 ms, 24% of a
/// 6.562 ms session, and every lane paid a term like it on one arm only.
///
/// A bias present in every repetition of ONE arm does not average out. So the
/// region is named here, both arms declare what they timed, and
/// [`timed_region_disagreement`] fails the run if they disagree.
pub const TIMED_STEPS: &[&str] = &["forward", "loss_sum", "backward"];

/// Work that must sit OUTSIDE the timer on both arms.
///
/// The checksum exists to prove the two sides computed the same thing; it is
/// verification, not op work, and timing it on one side measures a reduction
/// nobody is comparing.
pub const UNTIMED_TEARDOWN: &[&str] = &["gradient_checksum"];

/// The line the incumbent prints to declare the region it timed.
pub const TIMED_STEPS_MARKER: &str = "PT_TIMED_STEPS ";

/// Parse the incumbent's declared timed region.
#[must_use]
pub fn parse_timed_steps(stdout: &str) -> Option<Vec<&str>> {
    stdout.lines().find_map(|line| {
        let value = line.trim().strip_prefix(TIMED_STEPS_MARKER)?.trim();
        if value.is_empty() {
            return None;
        }
        Some(value.split(',').map(str::trim).collect())
    })
}

/// Check that both arms timed the same region.
///
/// Order matters: two arms that time the same steps in a different order are not
/// obviously comparable either, and saying so costs nothing.
///
/// Returns `None` when they agree, or the message to fail with when they do not.
#[must_use]
pub fn timed_region_disagreement(ours: &[&str], incumbent: &[&str]) -> Option<String> {
    if ours == incumbent {
        return None;
    }
    let extra_ours: Vec<&str> = ours
        .iter()
        .filter(|s| !incumbent.contains(*s))
        .copied()
        .collect();
    let extra_theirs: Vec<&str> = incumbent
        .iter()
        .filter(|s| !ours.contains(*s))
        .copied()
        .collect();
    Some(format!(
        "the two arms did not time the same region: ours timed {ours:?}, the incumbent timed \
         {incumbent:?} (only ours: {extra_ours:?}; only theirs: {extra_theirs:?}). A step timed on \
         one arm and not the other biases every repetition in the same direction, so no ratio from \
         this run is quotable."
    ))
}

/// Env var that re-enables the legacy block-arm ordering (`frankentorch-6atx2`).
///
/// # Why a bespoke name rather than the repo's generic `FT_ORIG`
///
/// `FT_ORIG` is this repo's usual "run the pre-change path" A/B gate, and using
/// it here would be consistent. It is also a footgun *here specifically*: this
/// harness is the one that publishes campaign vs-PyTorch ratios, and `FT_ORIG`
/// left exported in a shell for some other example's A/B would silently
/// downgrade it to the arm ordering this bead exists to remove — producing
/// numbers that look normal and are biased. The name is therefore unmistakable
/// and names this harness.
pub const LEGACY_BLOCK_ARMS_ENV: &str = "FT_H2H_LEGACY_BLOCK_ARMS";

/// How the two arms are ordered against each other.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ArmOrdering {
    /// Each round takes an incumbent sample immediately beside ours for the same
    /// lane. The default, and the only ordering whose ratios are quotable.
    Interleaved,
    /// The whole incumbent arm runs to completion before our first lane — the
    /// pre-`6atx2` behaviour, kept ONLY so the two orderings can be measured
    /// against each other in one window on identical code.
    LegacyBlock,
}

impl ArmOrdering {
    /// Label for the harness's provenance block.
    #[must_use]
    pub fn label(self) -> &'static str {
        match self {
            Self::Interleaved => "INTERLEAVED per round (default)",
            Self::LegacyBlock => "LEGACY BLOCK — whole incumbent arm first (NOT quotable)",
        }
    }

    /// Whether ratios from this ordering may be quoted.
    #[must_use]
    pub fn is_quotable(self) -> bool {
        matches!(self, Self::Interleaved)
    }
}

/// Decide the arm ordering from [`LEGACY_BLOCK_ARMS_ENV`]'s value.
///
/// Interleaved is the default and every ambiguous input resolves to it: the
/// legacy ordering is a deliberate opt-in, so an unset, empty, negative or
/// unrecognised value must never silently select the biased mode. A typo'd
/// `FT_H2H_LEGACY_BLOCK_ARMS=ture` gets the correct ordering, not the broken one.
#[must_use]
pub fn arm_ordering_from_env(value: Option<&str>) -> ArmOrdering {
    match value.map(str::trim) {
        Some(v) if v.eq_ignore_ascii_case("1") => ArmOrdering::LegacyBlock,
        Some(v) if v.eq_ignore_ascii_case("true") => ArmOrdering::LegacyBlock,
        Some(v) if v.eq_ignore_ascii_case("yes") => ArmOrdering::LegacyBlock,
        _ => ArmOrdering::Interleaved,
    }
}

/// Arguments for launching the incumbent co-process with `script`.
///
/// # Why this exists rather than a literal in the harness
///
/// **`python -` deadlocks this protocol.** That mode reads the *program* from
/// stdin until EOF, so an interpreter launched that way blocks forever waiting
/// for a close that can never come — stdin has to stay open to carry sample
/// requests. A block-mode harness gets away with `-` precisely because it closes
/// stdin immediately and never talks to the child again; converting it to a
/// co-process without changing the launcher hangs with no output and no error.
///
/// This returns the `-c` form, which leaves the child's stdin free.
#[must_use]
pub fn interpreter_args(script: &str) -> [&str; 2] {
    ["-c", script]
}

/// The incumbent co-process request loop, in Python.
///
/// Lives beside the parser for the same reason [`crate::harness_provenance`]
/// keeps its probe beside its marker: if the two drift, the harness deadlocks
/// waiting for a reply that will never come in the shape it expects. The lanes
/// dict and the `run` function are supplied by the calling harness; this is only
/// the ready/serve/quit protocol.
/// Co-process program for ISOLATION MODE (`FT_H2H_NO_INCUMBENT`), which imports NOTHING.
///
/// # Why a stub exists at all
///
/// In isolation mode the driver never sends a `SAMPLE` request — the incumbent's slots stay in the
/// balanced square but no incumbent work runs in them. The child was nevertheless the FULL torch
/// program: it imported torch, built every lane's tensors, and ran 32 warm-up iterations PER LANE
/// before printing `PT_READY`. So an "isolated" run was preceded by minutes of heavy torch work on
/// the same box, warming the same caches and pulling the same clocks, and the mode's own banner
/// claiming "the co-process is the only variable" was not quite true.
///
/// It also made the mode useless where it is most needed. `frankentorch-rayon-pool-width-qq8as`
/// asks whether the pool-width finding generalises past THIS host, and the machines available to
/// answer that — rch build workers — do not have torch installed. An arm-internal measurement that
/// cannot run without torch cannot visit a second machine.
///
/// The stub speaks the same protocol and does none of the work: version marker, timed-steps
/// declaration, `PT_READY`, then it waits for `QUIT`.
///
/// # Honesty of the two markers
///
/// The version is reported as a stub identifier rather than a torch version, so a row taken this
/// way can never be mistaken for one with a live incumbent — the provenance block says
/// `incumbent=NONE` and the run carries no vs-PyTorch claim, which is exactly what isolation mode
/// already declares about its PT columns.
///
/// The timed-steps line echoes the driver's own list. That check exists to prove both arms timed
/// the same region; with no incumbent work there is no second region to disagree with, and echoing
/// keeps the driver's enforcement path identical rather than adding a branch that could rot.
pub const ISOLATION_STUB_PY: &str = r#"
import sys
print('PT_TORCH_VERSION none-isolation-stub-no-torch', flush=True)
print('PT_TIMED_STEPS forward,loss_sum,backward', flush=True)
print('PT_READY', flush=True)
for _line in sys.stdin:
    if _line.strip() == 'QUIT':
        break
"#;

pub const SAMPLE_LOOP_PY: &str = r#"
import sys, os
# frankentorch-6atx2: warm every lane BEFORE announcing readiness, so no lane's
# first interleaved sample carries its own warmup cost.
# frankentorch-uilzh: the COUNT is env-tunable and DEFAULTS TO 4, unchanged, so
# every row already banked stays comparable. It is tunable because four is not
# enough for torch once a lane is large: at [32,64,56,56] the group_norm incumbent
# null reads 0.867-0.900 -- its second half FASTER than its first, i.e. still
# warming up inside the measurement. The Rust arm reads the SAME variable and must
# stay matched: an asymmetric warmup has a bias whose direction depends on which
# arm is faster, which is a property no instrument may have.
for _name, (_base, _fn) in LANES.items():
    for _ in range(int(os.environ.get('FT_H2H_WARMUP', '32'))):
        run(_base, _fn)
print('PT_READY', flush=True)
for _line in sys.stdin:
    _line = _line.strip()
    if _line == 'QUIT':
        break
    if _line.startswith('SAMPLE '):
        _lane = _line[len('SAMPLE '):]
        _base, _fn = LANES[_lane]
        _ms, _g = run(_base, _fn)
        print('PT_SAMPLE %s %.6f %.12g' % (_lane, _ms, _g), flush=True)
"#;

#[cfg(test)]
mod tests {
    use super::*;

    /// The stub must satisfy every handshake the driver enforces, or an isolation run dies before
    /// its first lane — and it must import nothing, or it cannot run on a machine without torch,
    /// which is the entire reason it exists.
    #[test]
    fn isolation_stub_speaks_the_protocol_and_imports_nothing() {
        assert!(ISOLATION_STUB_PY.contains(READY_MARKER));
        assert!(ISOLATION_STUB_PY.contains(QUIT_REQUEST));
        assert!(ISOLATION_STUB_PY.contains(crate::harness_provenance::VERSION_MARKER.trim_end()));
        assert!(ISOLATION_STUB_PY.contains(TIMED_STEPS_MARKER.trim_end()));
        for step in TIMED_STEPS {
            assert!(
                ISOLATION_STUB_PY.contains(step),
                "the stub must declare every timed step the driver checks for, missing {step}"
            );
        }
        // The assertion is about the IMPORT, not the substring: the protocol's own marker is
        // `PT_TORCH_VERSION` and the stub's identifier says `no-torch`, so a bare `contains("torch")`
        // fails on the two lines that exist to be honest about there being no torch.
        assert!(
            !ISOLATION_STUB_PY.contains("import torch"),
            "importing torch would defeat the point: the stub exists so an arm-internal run can \
             visit a machine that has none"
        );
        for line in ISOLATION_STUB_PY.lines() {
            let line = line.trim();
            assert!(
                !(line.starts_with("import ") && line != "import sys"),
                "the stub may import nothing but sys, found: {line}"
            );
            assert!(
                !line.starts_with("from "),
                "the stub may import nothing but sys, found: {line}"
            );
        }
        assert!(ISOLATION_STUB_PY.contains("flush=True"));
    }

    #[test]
    fn balanced_square_gives_each_arm_every_half_and_slot_position_equally() {
        let incumbent_slots = BALANCED_SQUARE
            .iter()
            .enumerate()
            .filter_map(|(index, incumbent)| incumbent.then_some(index))
            .collect::<Vec<_>>();
        assert_eq!(incumbent_slots, vec![0, 3, 4, 7]);
        assert_eq!(BALANCED_SQUARE[..4].iter().filter(|slot| **slot).count(), 2);
        assert_eq!(BALANCED_SQUARE[4..].iter().filter(|slot| **slot).count(), 2);
        assert_eq!(BALANCED_SQUARE.iter().filter(|slot| !**slot).count(), 4);
    }

    #[test]
    fn schedule_returns_exactly_the_requested_sample_count() {
        assert_eq!(incumbent_sample_rounds(16, 7).len(), 7);
        assert_eq!(incumbent_sample_rounds(16, 16).len(), 16);
        assert_eq!(incumbent_sample_rounds(18, 7).len(), 7);
        assert_eq!(incumbent_sample_rounds(100, 3).len(), 3);
    }

    /// The count must hold across the whole grid, not just the configured pair —
    /// this is what lets `REPS` change without silently changing the estimator.
    #[test]
    fn schedule_count_holds_for_every_rounds_sample_pair() {
        for rounds in 1..64_usize {
            for samples in 1..=rounds {
                let schedule = incumbent_sample_rounds(rounds, samples);
                assert_eq!(
                    schedule.len(),
                    samples,
                    "rounds={rounds} samples={samples} produced {schedule:?}"
                );
                assert!(
                    schedule.iter().all(|&r| r < rounds),
                    "rounds={rounds} samples={samples} produced an out-of-range round"
                );
                assert!(
                    schedule.windows(2).all(|w| w[0] < w[1]),
                    "rounds={rounds} samples={samples} must be strictly increasing"
                );
            }
        }
    }

    /// NEGATIVE CASE: **this is the 6atx2 defect itself.** A schedule that puts
    /// every incumbent sample in the leading rounds means the incumbent arm
    /// finishes before our arm's later rounds run — arm-then-arm with extra
    /// steps. The samples must reach into the final rounds.
    #[test]
    fn schedule_is_not_front_loaded() {
        let rounds = 16;
        let schedule = incumbent_sample_rounds(rounds, 7);
        let front_loaded: Vec<usize> = (0..7).collect();
        assert_ne!(
            schedule, front_loaded,
            "sampling the first N rounds reintroduces the arm-then-arm gap"
        );
        let last = *schedule.last().expect("schedule must be non-empty");
        assert!(
            last >= rounds - rounds / 7,
            "the incumbent must still be sampling near the end of the run; last={last}"
        );
    }

    /// The spread must be bounded, or a "spread" schedule could still leave one
    /// long unsampled stretch that a load shift hides in.
    #[test]
    fn schedule_gaps_are_bounded() {
        for rounds in 2..64_usize {
            for samples in 1..=rounds {
                let schedule = incumbent_sample_rounds(rounds, samples);
                // Ceiling of rounds/samples, plus one for the boundary rounding.
                let bound = rounds.div_ceil(samples) + 1;
                for pair in schedule.windows(2) {
                    assert!(
                        pair[1] - pair[0] <= bound,
                        "rounds={rounds} samples={samples} gap {} exceeds {bound}",
                        pair[1] - pair[0]
                    );
                }
            }
        }
    }

    #[test]
    fn degenerate_schedules_are_empty_not_panics() {
        assert!(incumbent_sample_rounds(0, 7).is_empty());
        assert!(incumbent_sample_rounds(16, 0).is_empty());
        assert!(incumbent_sample_rounds(0, 0).is_empty());
    }

    /// Asking for more samples than rounds clamps rather than looping forever or
    /// emitting duplicate rounds.
    #[test]
    fn more_samples_than_rounds_clamps_to_one_per_round() {
        let schedule = incumbent_sample_rounds(4, 99);
        assert_eq!(schedule, vec![0, 1, 2, 3]);
    }

    #[test]
    fn parses_a_well_formed_sample() {
        let sample = parse_sample_line("PT_SAMPLE avg_pool2d 12.345600 -1.5e3")
            .expect("well-formed sample must parse");
        assert_eq!(sample.lane, "avg_pool2d");
        assert!((sample.milliseconds - 12.345_6).abs() < 1e-9);
        assert!((sample.gradient_checksum + 1500.0).abs() < 1e-9);
    }

    #[test]
    fn tolerates_surrounding_whitespace() {
        let sample = parse_sample_line("   PT_SAMPLE conv3d 5.5 -9.9  \n")
            .expect("must tolerate whitespace");
        assert_eq!(sample.lane, "conv3d");
    }

    /// NEGATIVE CASE: other chatter on the child's stdout — warnings, the version
    /// line, the ready line — must not be mistaken for a measurement.
    #[test]
    fn non_sample_lines_are_rejected() {
        assert_eq!(parse_sample_line("PT_READY"), None);
        assert_eq!(parse_sample_line("PT_TORCH_VERSION 2.12.1+cpu"), None);
        assert_eq!(parse_sample_line("some warning from numpy"), None);
        assert_eq!(parse_sample_line(""), None);
        // The old block-mode line shape must NOT parse as an interleaved sample.
        assert_eq!(parse_sample_line("PT avg_pool2d 12.3456 -1.5"), None);
    }

    /// NEGATIVE CASE: a truncated reply must not yield a half-built sample.
    #[test]
    fn truncated_samples_are_rejected() {
        assert_eq!(parse_sample_line("PT_SAMPLE "), None);
        assert_eq!(parse_sample_line("PT_SAMPLE conv3d"), None);
        assert_eq!(parse_sample_line("PT_SAMPLE conv3d 5.5"), None);
        assert_eq!(parse_sample_line("PT_SAMPLE conv3d notanumber -1.0"), None);
    }

    /// NEGATIVE CASE: a broken duration is not a slow measurement. Folding a NaN
    /// or negative into the incumbent's median corrupts the ratio without ever
    /// failing loudly, which is the whole failure class this harness fights.
    #[test]
    fn non_finite_or_negative_durations_are_rejected() {
        assert_eq!(parse_sample_line("PT_SAMPLE conv3d nan -1.0"), None);
        assert_eq!(parse_sample_line("PT_SAMPLE conv3d inf -1.0"), None);
        assert_eq!(parse_sample_line("PT_SAMPLE conv3d -0.5 -1.0"), None);
    }

    /// The request the driver writes must be the one the Python loop matches, or
    /// the harness hangs waiting for a reply that is never produced.
    #[test]
    fn request_matches_the_python_loop_prefix() {
        assert_eq!(sample_request("conv3d"), "SAMPLE conv3d");
        assert!(SAMPLE_LOOP_PY.contains("startswith('SAMPLE ')"));
        assert!(sample_request("conv3d").starts_with(REQUEST_PREFIX));
    }

    /// The Python loop must emit exactly the markers the Rust side waits on and
    /// parses. This is the deadlock guard: a drifted marker means the driver
    /// blocks forever on a ready line or a sample that never arrives.
    #[test]
    fn python_loop_emits_the_markers_the_driver_expects() {
        assert!(
            SAMPLE_LOOP_PY.contains(READY_MARKER),
            "loop must announce readiness"
        );
        assert!(
            SAMPLE_LOOP_PY.contains(SAMPLE_MARKER.trim_end()),
            "loop must emit the sample marker"
        );
        assert!(
            SAMPLE_LOOP_PY.contains(QUIT_REQUEST),
            "loop must honour the quit request"
        );
        assert!(
            SAMPLE_LOOP_PY.contains("flush=True"),
            "unflushed stdout deadlocks the driver on a pipe"
        );
    }

    /// **THE NAMED REGRESSION CASE (`frankentorch-8ieqm`).** A banked run
    /// measured `max_pool3d` at 29.22x with its FT arm spiked to 3.8x its own
    /// median, and the old bracketing gate PASSED it on `[0.528,1.359]`. That
    /// row must be undecidable, or the gate is still certifying disturbed
    /// samples.
    #[test]
    fn the_contended_null_that_used_to_pass_is_now_undecidable() {
        let (low, high) = (0.528, 1.359);
        assert!(
            low <= 1.0 && high >= 1.0,
            "precondition: the old bracketing rule passed this row"
        );
        assert_eq!(
            adjudicate_null(low, high, MAX_NULL_CI_WIDTH),
            NullVerdict::TooWide,
            "the 29.22x contended run must not certify a calm sample"
        );
        assert!(!adjudicate_null(low, high, MAX_NULL_CI_WIDTH).is_quotable());
    }

    /// The clean reference run must survive, or the veto is not a tightening but
    /// a blanket refusal that makes the gate useless.
    #[test]
    fn the_clean_reference_run_still_certifies() {
        assert_eq!(
            adjudicate_null(0.798, 1.266, MAX_NULL_CI_WIDTH),
            NullVerdict::Calm
        );
        // A representative tight row from the banked set.
        assert_eq!(
            adjudicate_null(0.951, 1.067, MAX_NULL_CI_WIDTH),
            NullVerdict::Calm
        );
    }


    /// THE ENVELOPE IS NOT THE GATED QUANTITY — `frankentorch-strict-aa-gate-band-v8rpa`.
    ///
    /// The bead reported this exact row as a gate-integrity defect:
    ///
    ///     prelu_noshortcut  PT PASS [0.966,1.018]  FT PASS [0.948,1.036]
    ///
    /// reasoning that both envelopes leave the 0.97..=1.03 band and so both verdicts must be FAIL.
    /// They must not. `adjudicate_null` asks two things of an ENVELOPE — that it is narrower than
    /// `MAX_NULL_CI_WIDTH` and that it brackets unity — and both envelopes satisfy both. The
    /// ±0.02 band is applied to the POINT ESTIMATE by the caller, which is a different number.
    ///
    /// Holding a bootstrap CI inside ±0.03 would be a far stricter protocol than this one: at the
    /// round counts this board uses, a calm null's CI is routinely 0.05-0.13 wide, so that rule
    /// would fail nearly every honest row. This test pins the actual contract so the next reader
    /// who conflates the two finds an assertion instead of writing a bug report.
    /// The band must be REPRODUCIBLE from the slots alone — a gate that consults an unseeded RNG
    /// cannot re-adjudicate a retained row.
    #[test]
    fn the_permutation_band_is_deterministic_for_the_same_slots() {
        let rounds: Vec<[f64; 4]> = (0..24)
            .map(|i| {
                let f = f64::from(i);
                [10.0 + f * 0.1, 5.0 + f * 0.02, 5.1 - f * 0.01, 4.9 + f * 0.03]
            })
            .collect();
        let a = permutation_null_band(&rounds, 1, NULL_PERMUTATION_DRAWS);
        let b = permutation_null_band(&rounds, 1, NULL_PERMUTATION_DRAWS);
        assert_eq!(a.0.to_bits(), b.0.to_bits());
        assert_eq!(a.1.to_bits(), b.1.to_bits());
        // Different data must give a different band, or the seed is not reading the slots.
        let mut other = rounds.clone();
        other[0][1] *= 1.5;
        assert_ne!(permutation_null_band(&other, 1, NULL_PERMUTATION_DRAWS), a);
    }

    /// The band must BRACKET UNITY: permuting exchangeable slots cannot create a direction.
    /// If it did not, the gate would reject every clean arm on one side.
    #[test]
    fn the_permutation_band_brackets_unity_on_exchangeable_slots() {
        let rounds: Vec<[f64; 4]> = (0..24)
            .map(|i| {
                let f = f64::from(i % 5);
                [99.0, 5.0 + f * 0.1, 5.0 - f * 0.05, 5.0 + f * 0.02]
            })
            .collect();
        let (lo, hi) = permutation_null_band(&rounds, 1, NULL_PERMUTATION_DRAWS);
        assert!(lo <= 1.0 && hi >= 1.0, "band [{lo},{hi}] must contain unity");
        // Slot 0 is 20x the warm slots here and must not reach the band at all.
        assert!(hi < 2.0, "slot 0 leaked into the permutation band: [{lo},{hi}]");
    }

    /// MORE ROUNDS MUST NARROW IT. That is the property a fixed constant cannot have, and it is
    /// why the gate is self-calibrating rather than a new magic number.
    #[test]
    fn the_permutation_band_narrows_with_more_rounds() {
        let make = |n: i32| -> Vec<[f64; 4]> {
            (0..n)
                .map(|i| {
                    let f = f64::from(i % 7) * 0.1;
                    [9.0, 5.0 + f, 5.0 - f * 0.5, 5.0 + f * 0.25]
                })
                .collect()
        };
        let (l8, h8) = permutation_null_band(&make(8), 1, NULL_PERMUTATION_DRAWS);
        let (l96, h96) = permutation_null_band(&make(96), 1, NULL_PERMUTATION_DRAWS);
        assert!(
            (h96 - l96) < (h8 - l8),
            "96 rounds gave [{l96},{h96}], wider than 8 rounds' [{l8},{h8}]"
        );
    }

    /// A REAL position effect must still be caught: if the last warm slot is systematically
    /// slower, the observed null falls outside its own permutation band.
    #[test]
    fn a_real_position_effect_lands_outside_the_permutation_band() {
        let rounds: Vec<[f64; 4]> = (0..24)
            .map(|i| {
                let f = f64::from(i % 3) * 0.01;
                [9.0, 5.0 + f, 5.5 + f, 6.0 + f] // a monotone rise across the warm slots
            })
            .collect();
        let (lo, hi) = permutation_null_band(&rounds, 1, NULL_PERMUTATION_DRAWS);
        let observed: Vec<f64> = rounds.iter().map(|r| r[1] / r[3]).collect();
        let mut v = observed.clone();
        v.sort_by(f64::total_cmp);
        let point = v[v.len() / 2];
        assert!(
            point < lo || point > hi,
            "a 20% rise across the warm slots read {point} inside [{lo},{hi}]"
        );
    }

    #[test]
    fn the_reported_envelopes_are_calm_because_the_band_is_not_an_envelope_test() {
        assert_eq!(adjudicate_null(0.966, 1.018, MAX_NULL_CI_WIDTH), NullVerdict::Calm);
        assert_eq!(adjudicate_null(0.948, 1.036, MAX_NULL_CI_WIDTH), NullVerdict::Calm);
    }

    /// What DOES make an envelope fail: not bracketing unity, at any width.
    #[test]
    fn an_envelope_that_misses_unity_is_off_centre_however_tight() {
        assert_eq!(adjudicate_null(1.004, 1.006, MAX_NULL_CI_WIDTH), NullVerdict::OffCentre);
        assert_eq!(adjudicate_null(0.994, 0.996, MAX_NULL_CI_WIDTH), NullVerdict::OffCentre);
        // ...and an envelope that brackets unity but is too wide supports no verdict at all.
        assert_eq!(adjudicate_null(0.60, 1.30, MAX_NULL_CI_WIDTH), NullVerdict::TooWide);
    }

    /// **THE DEFECT ITSELF, as a property.** Under the old rule a null centred on
    /// unity passed no matter how wide it got, so contention made the gate
    /// EASIER to pass. Widening must never improve a verdict.
    #[test]
    fn widening_a_centred_null_never_makes_it_easier_to_pass() {
        let mut seen_too_wide = false;
        let mut half = 0.01_f64;
        while half < 4.0 {
            let verdict = adjudicate_null(1.0 - half, 1.0 + half, MAX_NULL_CI_WIDTH);
            // Every one of these brackets 1.0, so the OLD rule passed them all.
            assert!(1.0 - half <= 1.0 && 1.0 + half >= 1.0);
            if verdict == NullVerdict::TooWide {
                seen_too_wide = true;
            } else {
                assert!(
                    !seen_too_wide,
                    "verdict improved back to {verdict:?} at half-width {half} after going wide"
                );
            }
            half += 0.01;
        }
        assert!(
            seen_too_wide,
            "a centred null must eventually become undecidable as it widens"
        );
    }

    /// A wide CI that just barely clears 1.0 is not a detected position effect.
    /// The banked `conv3d` row failed on a CI 3.44 wide whose lower bound cleared
    /// unity by 0.011 — arbitrary in the other direction, and now undecidable.
    #[test]
    fn a_hugely_wide_off_centre_null_is_undecidable_not_a_failure() {
        assert_eq!(
            adjudicate_null(1.011, 4.453, MAX_NULL_CI_WIDTH),
            NullVerdict::TooWide,
            "a 3.44-wide CI supports no verdict, in either direction"
        );
    }

    /// A genuine position effect — tight AND off unity — must still FAIL, or the
    /// width veto would have swallowed the failure the null exists to catch.
    #[test]
    fn a_tight_off_centre_null_still_fails() {
        assert_eq!(
            adjudicate_null(1.05, 1.20, MAX_NULL_CI_WIDTH),
            NullVerdict::OffCentre
        );
        assert_eq!(
            adjudicate_null(0.80, 0.95, MAX_NULL_CI_WIDTH),
            NullVerdict::OffCentre
        );
        assert!(!NullVerdict::OffCentre.is_quotable());
    }

    /// The boundary is inclusive: a CI exactly at the limit still certifies.
    ///
    /// Deliberately uses exact binary fractions (0.75 and 1.25 differ by exactly
    /// 0.5). Decimal-looking pairs do NOT work here — `1.3 - 0.7` evaluates to
    /// 0.6000000000000001, so a test written that way fails on an ULP and says
    /// nothing about the boundary rule it meant to check.
    #[test]
    fn the_width_boundary_is_inclusive() {
        assert_eq!(1.25_f64 - 0.75_f64, 0.5_f64, "chosen bounds must be exact");
        assert_eq!(adjudicate_null(0.75, 1.25, 0.5), NullVerdict::Calm);
        // Just over the limit is undecidable.
        assert_eq!(adjudicate_null(0.75, 1.3, 0.5), NullVerdict::TooWide);
    }

    /// NEGATIVE CASE: the gate must fail closed on input it cannot evaluate,
    /// never certify it.
    #[test]
    fn unevaluable_bounds_fail_closed() {
        for (low, high) in [
            (f64::NAN, 1.2),
            (0.8, f64::NAN),
            (f64::NEG_INFINITY, 1.2),
            (0.8, f64::INFINITY),
            (1.2, 0.8),
        ] {
            assert_eq!(
                adjudicate_null(low, high, MAX_NULL_CI_WIDTH),
                NullVerdict::TooWide,
                "[{low},{high}] must not certify"
            );
        }
    }

    #[test]
    fn only_calm_is_quotable() {
        assert!(NullVerdict::Calm.is_quotable());
        assert!(!NullVerdict::TooWide.is_quotable());
        assert_eq!(NullVerdict::Calm.label(), "PASS");
        assert_eq!(NullVerdict::TooWide.label(), "WIDE");
        assert_eq!(NullVerdict::OffCentre.label(), "FAIL");
    }

    /// **THE PLANTED ASYMMETRY (`frankentorch-574cu`).** This is the exact
    /// pre-fix configuration: the FrankenTorch arm timed the gradient checksum,
    /// the PyTorch arm did not. Run against the code as it stood before the fix,
    /// this case is what the harness actually did — so it must be rejected, and
    /// the message must name the offending step or it is not actionable.
    #[test]
    fn the_pre_fix_asymmetric_region_is_rejected() {
        let ours_pre_fix = ["forward", "loss_sum", "backward", "gradient_checksum"];
        let incumbent = ["forward", "loss_sum", "backward"];
        let message = timed_region_disagreement(&ours_pre_fix, &incumbent)
            .expect("the pre-fix asymmetry MUST be caught");
        assert!(
            message.contains("gradient_checksum"),
            "must name the offending step: {message}"
        );
        assert!(message.contains("quotable"), "{message}");
    }

    /// The shipped configuration must agree with itself — this is the assertion
    /// that flips from FAIL to PASS with the fix, because `TIMED_STEPS` no longer
    /// contains the checksum.
    #[test]
    fn the_shipped_timed_region_excludes_teardown() {
        for step in UNTIMED_TEARDOWN {
            assert!(
                !TIMED_STEPS.contains(step),
                "`{step}` is teardown and must not be inside the timed region: {TIMED_STEPS:?}"
            );
        }
        assert_eq!(TIMED_STEPS, &["forward", "loss_sum", "backward"]);
        assert!(timed_region_disagreement(TIMED_STEPS, TIMED_STEPS).is_none());
    }

    /// NEGATIVE CASE: an incumbent that times MORE than us is just as wrong, and
    /// biases the ratio the other way. The check must not be one-sided.
    #[test]
    fn an_incumbent_timing_extra_work_is_also_a_disagreement() {
        let message = timed_region_disagreement(
            &["forward", "loss_sum", "backward"],
            &["forward", "loss_sum", "backward", "gradient_checksum"],
        )
        .expect("an incumbent timing extra work must be caught too");
        assert!(message.contains("gradient_checksum"), "{message}");
    }

    /// Same steps in a different order are not obviously comparable either.
    #[test]
    fn reordered_steps_are_a_disagreement() {
        assert!(
            timed_region_disagreement(
                &["forward", "loss_sum", "backward"],
                &["forward", "backward", "loss_sum"],
            )
            .is_some()
        );
    }

    #[test]
    fn parses_the_incumbents_declared_region() {
        let stdout = "PT_TORCH_VERSION 2.12.1+cpu\nPT_TIMED_STEPS forward,loss_sum,backward\n";
        assert_eq!(
            parse_timed_steps(stdout),
            Some(vec!["forward", "loss_sum", "backward"])
        );
        // Whitespace around the separators must not fabricate a disagreement.
        assert_eq!(
            parse_timed_steps("PT_TIMED_STEPS  forward , loss_sum , backward \n"),
            Some(vec!["forward", "loss_sum", "backward"])
        );
    }

    /// NEGATIVE CASE: an arm that never declares its region must not be treated
    /// as agreeing. Silence is not symmetry.
    #[test]
    fn an_undeclared_region_is_none_not_agreement() {
        assert_eq!(parse_timed_steps("PT_TORCH_VERSION 2.12.1\n"), None);
        assert_eq!(parse_timed_steps("PT_TIMED_STEPS \n"), None);
        assert_eq!(parse_timed_steps(""), None);
    }

    /// The declaration the Python arm emits must round-trip into agreement with
    /// the Rust constant, or every run hard-fails on a formatting difference.
    #[test]
    fn the_python_declaration_agrees_with_the_rust_constant() {
        let declared = format!("{TIMED_STEPS_MARKER}{}", TIMED_STEPS.join(","));
        let parsed = parse_timed_steps(&declared).expect("our own declaration must parse");
        assert!(timed_region_disagreement(TIMED_STEPS, &parsed).is_none());
    }

    /// **THE DEFAULT MUST BE INTERLEAVED.** An unset variable is the case that
    /// covers every ordinary run of this harness, including CI and anyone who
    /// just types the command from the module header.
    #[test]
    fn the_default_ordering_is_interleaved() {
        assert_eq!(arm_ordering_from_env(None), ArmOrdering::Interleaved);
        assert!(arm_ordering_from_env(None).is_quotable());
    }

    /// The legacy ordering requires an EXPLICIT affirmative. Everything else —
    /// empty, negative, or unrecognised — resolves to interleaved, because the
    /// biased mode must never be reachable by accident.
    #[test]
    fn only_an_explicit_affirmative_selects_the_legacy_ordering() {
        for on in ["1", "true", "TRUE", "True", "yes", "YES", " 1 ", "  true  "] {
            assert_eq!(
                arm_ordering_from_env(Some(on)),
                ArmOrdering::LegacyBlock,
                "{on:?} should opt in"
            );
        }
        for off in [
            "",
            "  ",
            "0",
            "false",
            "no",
            "off",
            "ture",
            "2",
            "interleaved",
            "-1",
        ] {
            assert_eq!(
                arm_ordering_from_env(Some(off)),
                ArmOrdering::Interleaved,
                "{off:?} must NOT select the biased ordering"
            );
        }
    }

    /// NEGATIVE CASE: a typo must not silently buy the biased ordering. This is
    /// the whole reason the match is an allowlist rather than "anything truthy".
    #[test]
    fn a_typo_does_not_select_the_legacy_ordering() {
        assert_eq!(
            arm_ordering_from_env(Some("ture")),
            ArmOrdering::Interleaved
        );
        assert_eq!(
            arm_ordering_from_env(Some("yess")),
            ArmOrdering::Interleaved
        );
        assert_eq!(arm_ordering_from_env(Some("1 1")), ArmOrdering::Interleaved);
    }

    /// Only the interleaved ordering may be quoted, and the legacy label has to
    /// say so where a reader will see it.
    #[test]
    fn the_legacy_ordering_is_labelled_unquotable() {
        assert!(!ArmOrdering::LegacyBlock.is_quotable());
        assert!(ArmOrdering::LegacyBlock.label().contains("NOT quotable"));
        assert!(ArmOrdering::Interleaved.label().contains("INTERLEAVED"));
    }

    /// The gate must not collide with the repo's generic `FT_ORIG` A/B variable,
    /// or an unrelated example's A/B session silently downgrades the harness that
    /// publishes campaign ratios.
    #[test]
    fn the_gate_does_not_reuse_the_generic_ab_variable() {
        assert_ne!(LEGACY_BLOCK_ARMS_ENV, "FT_ORIG");
        assert!(LEGACY_BLOCK_ARMS_ENV.contains("LEGACY"));
    }

    /// NEGATIVE CASE: the deadlock that actually happened. Launching the child as
    /// `python -` makes the interpreter read its program from stdin until EOF, so
    /// it never starts serving while the driver waits for `PT_READY` — the whole
    /// harness hangs with no output and no error. The launcher must keep stdin
    /// free for requests.
    #[test]
    fn interpreter_is_launched_in_a_mode_that_leaves_stdin_free() {
        let args = interpreter_args("print(1)");
        assert_eq!(args[0], "-c", "must pass the program as an argument");
        assert_ne!(
            args[0], "-",
            "`python -` reads the program from stdin until EOF and deadlocks the request loop"
        );
        assert_eq!(args[1], "print(1)");
        assert!(
            !args.contains(&"-"),
            "no argument may put the interpreter in read-program-from-stdin mode"
        );
    }

    /// A line the Python loop actually formats must round-trip through the
    /// parser — testing the two halves separately can leave a format mismatch.
    #[test]
    fn python_format_string_round_trips_through_the_parser() {
        // The exact format the loop uses, rendered as Python would render it.
        let rendered = format!(
            "PT_SAMPLE {} {:.6} {:.12}",
            "max_pool3d", 7.25_f64, -1.5_f64
        );
        let sample = parse_sample_line(&rendered).expect("the loop's own format must parse");
        assert_eq!(sample.lane, "max_pool3d");
        assert!((sample.milliseconds - 7.25).abs() < 1e-9);
        assert!((sample.gradient_checksum + 1.5).abs() < 1e-9);
    }
}

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

/// The incumbent co-process request loop, in Python.
///
/// Lives beside the parser for the same reason [`crate::harness_provenance`]
/// keeps its probe beside its marker: if the two drift, the harness deadlocks
/// waiting for a reply that will never come in the shape it expects. The lanes
/// dict and the `run` function are supplied by the calling harness; this is only
/// the ready/serve/quit protocol.
pub const SAMPLE_LOOP_PY: &str = r#"
import sys
# frankentorch-6atx2: warm every lane BEFORE announcing readiness, so no lane's
# first interleaved sample carries its own warmup cost.
for _name, (_base, _fn) in LANES.items():
    for _ in range(4):
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
        let sample =
            parse_sample_line("   PT_SAMPLE conv3d 5.5 -9.9  \n").expect("must tolerate whitespace");
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

    /// A line the Python loop actually formats must round-trip through the
    /// parser — testing the two halves separately can leave a format mismatch.
    #[test]
    fn python_format_string_round_trips_through_the_parser() {
        // The exact format the loop uses, rendered as Python would render it.
        let rendered = format!("PT_SAMPLE {} {:.6} {:.12}", "max_pool3d", 7.25_f64, -1.5_f64);
        let sample = parse_sample_line(&rendered).expect("the loop's own format must parse");
        assert_eq!(sample.lane, "max_pool3d");
        assert!((sample.milliseconds - 7.25).abs() < 1e-9);
        assert!((sample.gradient_checksum + 1.5).abs() < 1e-9);
    }
}

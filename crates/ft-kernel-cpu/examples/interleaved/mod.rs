//! `frankentorch-stale-tuning-constants-lzku6` — ledger 293's trust rules, as one instrument.
//!
//! # Why this module exists
//!
//! Ledger 292i recorded a geqrf panel width that won ALL TWELVE kernel cells at a median ~1.17x
//! and then measured ~8% SLOWER at the paired lane gate — a 25-point isolation-to-lane swing on
//! evidence carrying every quality marker this repo asks for. Ledger 293 found the cause, and it
//! was not the lane:
//!
//!   1. **BLOCK ORDERING.** Every sweep here was written `for candidate { for rep }`, so the
//!      incumbent is measured FIRST in every pass and any drift across the pass is confounded
//!      with the candidate. The h2h lane harness interleaves arms within each round and takes
//!      per-round paired ratios, which cancels exactly that — which is why the lane was right.
//!   2. **EFFECT SMALLER THAN NOISE.** The incumbent's own cell varied 1.34-1.47x across passes
//!      while the effect being claimed was 1.17-1.24x. Effect/noise 0.36-0.72, all below 1. Run
//!      interleaved, the incumbent's WITHIN-RUN spread at n=256 was 1.59-1.80x: a baseline that
//!      moves 1.8x cannot resolve a 1.1x effect in ANY harness, however many passes are averaged.
//!
//! # The rule this module enforces
//!
//! An isolation sweep is trustworthy only when
//!
//!   (a) arms are INTERLEAVED per rep with alternating order — enforced structurally by [`run`],
//!       which is the only way to sample an arm here;
//!   (b) the effect EXCEEDS the incumbent's within-run spread — MEASURED and printed, never
//!       assumed, and the gate in [`verdict`];
//!   (c) the SIGN TEST supports it — 9-13 of 21 is a coin flip whatever the median says, so an
//!       exact two-sided binomial p-value is computed rather than eyeballed;
//!   (d) both estimators agree — paired (median of per-rep ratios) and marginal (ratio of
//!       medians) disagreeing is itself the finding, per `feedback_estimator_and_provenance`,
//!       which records a 1.512x disagreement between them on identical work.
//!
//! A cell failing any of (b), (c), (d) prints UNRESOLVED **with the arithmetic that failed**. That
//! is not a softer verdict than a number; it is the honest one, and it is what the block-ordered
//! sweeps could not say.
//!
//! # Two further hazards this shape closes
//!
//! * The incumbent arm is the SHIPPED path with its knob unset, never the shipped value re-fed
//!   through the override. `feedback_unset_knob_means_forced_off` and
//!   `feedback_one_knob_is_secretly_two` both record straw-man incumbents built exactly that way.
//! * An A/A NULL arm — a second copy of the incumbent, placed last so the order reversal gives it
//!   the maximum position contrast — rides in every sweep. It costs one arm's time and it is the
//!   only thing that can say whether the harness itself can resolve anything today.

#![allow(dead_code)]

/// Median of a timing sample. Panics on NaN, which in a timing vector is a bug, not a datum.
#[must_use]
pub fn median(v: &[f64]) -> f64 {
    if v.is_empty() {
        return f64::NAN;
    }
    let mut s = v.to_vec();
    s.sort_by(|a, b| a.partial_cmp(b).expect("no NaN in a timing sample"));
    let n = s.len();
    if n % 2 == 1 {
        s[n / 2]
    } else {
        f64::midpoint(s[n / 2 - 1], s[n / 2])
    }
}

fn sorted(v: &[f64]) -> Vec<f64> {
    let mut s = v.to_vec();
    s.sort_by(|a, b| a.partial_cmp(b).expect("no NaN in a timing sample"));
    s
}

fn quantile(s: &[f64], q: f64) -> f64 {
    if s.is_empty() {
        return f64::NAN;
    }
    let pos = q * (s.len() - 1) as f64;
    let lo = pos.floor() as usize;
    let hi = pos.ceil() as usize;
    let frac = pos - pos.floor();
    s[lo].mul_add(1.0 - frac, s[hi] * frac)
}

/// `(min, max, max/min)` — the WITHIN-RUN spread of one arm. The ratio is ledger 293's gate.
#[must_use]
pub fn spread(v: &[f64]) -> (f64, f64, f64) {
    let lo = v.iter().copied().fold(f64::INFINITY, f64::min);
    let hi = v.iter().copied().fold(0.0_f64, f64::max);
    (lo, hi, hi / lo)
}

/// `p75/p25` — a robust companion to [`spread`], printed as a DIAGNOSTIC only. The gate stays
/// max/min because that is what 293 states; this says how much of it is one outlying rep.
#[must_use]
pub fn iqr_ratio(v: &[f64]) -> f64 {
    let s = sorted(v);
    quantile(&s, 0.75) / quantile(&s, 0.25)
}

/// Exact two-sided sign test against `p = 0.5`.
#[must_use]
pub fn sign_test_p(wins: usize, losses: usize) -> f64 {
    let n = wins + losses;
    if n == 0 {
        return 1.0;
    }
    let k = wins.max(losses);
    let mut tail = 0.0f64;
    let mut c = 1.0f64; // C(n, 0)
    for i in 0..=n {
        if i >= k {
            tail += c;
        }
        c = c * (n - i) as f64 / (i + 1) as f64;
    }
    (2.0 * tail / 2.0f64.powi(i32::try_from(n).unwrap_or(i32::MAX))).min(1.0)
}

/// The rep count this module will actually honour, and the only one a caller should print.
///
/// Rounded UP TO EVEN, because the order reversal happens on odd reps: at an odd rep count the
/// forward order runs one more time than the reversed one, every arm's mean slot position is
/// therefore skewed toward its position in the forward order, and the correction the alternation
/// exists to make is left half-applied. At an even count each arm's mean slot is exactly
/// `(arms-1)/2` and the first-order position effect cancels for all of them at once.
///
/// Floored at 6, because an exact two-sided sign test cannot reach p<0.05 below it: a clean sweep
/// of 5 of 5 is p=0.0625 and would print UNRESOLVED however large the effect. Asking for fewer
/// reps than the gate can resolve is asking for a table of UNRESOLVED rows.
#[must_use]
pub fn reps_for(requested: usize) -> usize {
    let r = requested.max(6);
    r + r % 2
}

/// The fewest same-direction reps out of `reps` that can clear [`SIGN_ALPHA`]. Printed so the
/// sign-test column can be read without recomputing a binomial tail in your head.
#[must_use]
pub fn wins_needed(reps: usize) -> usize {
    (reps / 2..=reps)
        .find(|&w| sign_test_p(w, reps - w) < SIGN_ALPHA)
        .unwrap_or(reps + 1)
}

/// Runs `reps` INTERLEAVED rounds over `arms` arms, reversing the arm order on odd rounds so no
/// arm sits permanently in the warmer slot, and taking the MIN OF TWO samples per arm per round.
///
/// This is the only sampling entry point in the module on purpose: a caller cannot accidentally
/// reintroduce the `for candidate { for rep }` ordering that ledger 293 indicted, because the rep
/// loop is on the inside of this function and the arm index is handed to the caller, not chosen
/// by it. `sample(i)` must time arm `i` and nothing else.
///
/// Returns per-arm vectors of per-round times, aligned by round index, which is what makes the
/// PAIRED estimator paired.
pub fn run<F>(arms: usize, reps: usize, warm: usize, mut sample: F) -> Vec<Vec<f64>>
where
    F: FnMut(usize) -> f64,
{
    for _ in 0..warm {
        for i in 0..arms {
            std::hint::black_box(sample(i));
        }
    }
    let mut times: Vec<Vec<f64>> = vec![Vec::with_capacity(reps); arms];
    for rep in 0..reps {
        let mut order: Vec<usize> = (0..arms).collect();
        if rep % 2 == 1 {
            order.reverse();
        }
        for i in order {
            let first = sample(i);
            let second = sample(i);
            times[i].push(first.min(second));
        }
    }
    times
}

/// One arm's verdict under ledger 293's four conditions.
pub struct Verdict {
    /// Median of the per-rep ratios `incumbent/candidate`. >1 means the candidate is faster.
    pub paired: f64,
    /// Ratio of the medians. Disagreement with `paired` is itself a finding.
    pub marginal: f64,
    pub wins: usize,
    pub losses: usize,
    /// Exact two-sided sign-test p-value.
    pub p: f64,
    /// True only when (b), (c) and (d) all hold.
    pub resolved: bool,
    /// The arithmetic behind the verdict — the failing terms, or the direction if it passed.
    pub note: String,
}

/// Estimator agreement tolerance, matching ledger 274c/275b and the 293 re-run.
pub const ESTIMATOR_TOL: f64 = 0.05;
/// Sign-test significance level.
pub const SIGN_ALPHA: f64 = 0.05;

/// Scores `cand` against `base` under conditions (b), (c) and (d). `gate` is the incumbent's
/// within-run spread ratio from [`spread`] — the effect must exceed it, which is condition (b).
#[must_use]
pub fn verdict(base: &[f64], cand: &[f64], gate: f64) -> Verdict {
    let ratios: Vec<f64> = base.iter().zip(cand).map(|(b, c)| b / c).collect();
    let paired = median(&ratios);
    let marginal = median(base) / median(cand);
    let wins = ratios.iter().filter(|&&r| r > 1.0).count();
    let losses = ratios.iter().filter(|&&r| r < 1.0).count();
    let p = sign_test_p(wins, losses);

    let effect = (paired - 1.0).abs();
    let noise = gate - 1.0;
    let disagreement = (paired - marginal).abs();

    let mut fails: Vec<String> = Vec::new();
    if effect <= noise {
        fails.push(format!("effect {effect:.4} <= incumbent spread {noise:.4}"));
    }
    if p >= SIGN_ALPHA {
        fails.push(format!("sign test p={p:.4}"));
    }
    if disagreement > ESTIMATOR_TOL {
        fails.push(format!("estimators disagree by {disagreement:.4}"));
    }

    let resolved = fails.is_empty();
    let note = if resolved {
        if paired > 1.0 {
            "TRUSTED WIN".to_owned()
        } else {
            "TRUSTED LOSS".to_owned()
        }
    } else {
        format!("UNRESOLVED: {}", fails.join("; "))
    };
    Verdict {
        paired,
        marginal,
        wins,
        losses,
        p,
        resolved,
        note,
    }
}

impl Verdict {
    /// The trust columns, formatted to line up under [`Verdict::header`].
    #[must_use]
    pub fn row(&self) -> String {
        format!(
            "{:>9.4} {:>9.4} {:>8} {:>8.4}  {}",
            self.paired,
            self.marginal,
            format!("{}/{}", self.wins, self.wins + self.losses),
            self.p,
            self.note,
        )
    }

    /// Column titles for [`Verdict::row`].
    #[must_use]
    pub fn header() -> String {
        format!(
            "{:>9} {:>9} {:>8} {:>8}  {}",
            "PAIRED", "MARGIN", "SIGN", "p", "VERDICT"
        )
    }
}

/// Prints the provenance line every measured row in this repo is required to carry, plus the
/// rules being enforced. Call once at the top of a sweep.
pub fn banner(what: &str, reps: usize) {
    let host = std::fs::read_to_string("/etc/hostname").unwrap_or_else(|_| "unknown\n".to_owned());
    let load = std::fs::read_to_string("/proc/loadavg").unwrap_or_else(|_| "unknown".to_owned());
    println!(
        "PROV host={} nproc={} rayon={} reps={reps} loadavg={}",
        host.trim(),
        std::thread::available_parallelism().map_or(0, std::num::NonZero::get),
        rayon::current_num_threads(),
        load.split_whitespace()
            .take(3)
            .collect::<Vec<_>>()
            .join(","),
    );
    println!(
        "METHOD ({what}): arms INTERLEAVED within every rep, order REVERSED on odd reps, per-rep \
         min of 2, PAIRED = median of per-rep ratios. Ledger 293 — a block-ordered sweep confounds \
         drift with the candidate, and that is how geqrf b=64 won 12/12 cells and lost its lane 8%."
    );
    println!(
        "GATE: a cell is TRUSTED only if (b) the effect exceeds the incumbent's within-run spread, \
         (c) the exact two-sided sign test clears p<{SIGN_ALPHA} — which at {reps} reps means at \
         least {} of {reps} in one direction — and (d) the two estimators agree within \
         {ESTIMATOR_TOL}. Anything else prints UNRESOLVED with the arithmetic that failed.",
        wins_needed(reps),
    );
    println!(
        "CONTROLS: every table carries an A/A arm (the incumbent, duplicated) and, where the \
         candidate grid contains the shipped width, a `*` arm running that width through the KNOB \
         against the incumbent's unset path. Both should read ~1.0x on a coin-flip sign test; how \
         far they miss is this harness's own floor on the day, and no candidate row means more \
         than that floor allows."
    );
    println!(
        "READING: a TRUSTED WIN here is NECESSARY AND NOT SUFFICIENT — ledger 291 has a bit-exact \
         1.28-1.40x isolation win that moved no lane, and 292i has one that inverted. Ship only \
         after an all-cells win AND a paired lane certification. Blocking changes are NOT \
         bit-exact (they reassociate): gate on reconstruction and the oracle, never to_bits()."
    );
}

//! `frankentorch-f32-inplace-accessor-gap-5fxq2` — is the f32 in-place path actually paying for
//! its extra passes, or is the pass count a distinction without a difference?
//!
//! WHAT THIS COMPARES AND WHY IT IS NOT A PERF CLAIM. Both arms are FrankenTorch, so by the
//! standing rule this is diagnosis, not a win. It is worth running anyway because the two arms
//! differ in a way that makes the result INTERPRETABLE WITHOUT AN INCUMBENT:
//!
//!   f64 `exp_`  ->  `update_tensor_values_with` -> ONE in-place `par_iter_mut` pass
//!   f32 `exp_`  ->  `values_f32()` CLONE -> `par_iter().map(..).collect()` -> writeback memcpy
//!                   = THREE passes, and each element is computed as `transform(v as f64) as f32`
//!
//! f32 moves HALF the bytes of f64. On a bandwidth-bound elementwise op the f32 arm should
//! therefore be roughly twice as FAST. If it is merely level with f64 — let alone slower — the
//! only place that advantage can have gone is the two extra passes and the per-element dtype round
//! trip. **The prediction is registered here before the run**: f32/f64 near 0.5 means the extra
//! passes are free and the bead is moot; near or above 1.0 means they are not.
//!
//! `exp_` is chosen deliberately over `neg_`: it routes through the same helper but is
//! compute-heavy, so if the two arms come out level on a transcendental the memory-traffic
//! explanation is the surviving one.
//!
//! METHOD, per the campaign's measurement rules. Arms ALTERNATE per rep rather than running as a
//! fixed block — `feedback_alternate_the_square` records a fixed ABBA putting one arm in the two
//! middle slots every rep, which an A/A null taken on the outer slots cannot see. Each rep takes
//! min-of-2 per arm; the reported figure is the MEDIAN OF PER-REP RATIOS with the marginal
//! (median-of-mins) printed beside it, because `feedback_estimator_and_provenance` records a
//! 1.512x disagreement between those two estimators on identical work. A sign test is printed and
//! disagreement between the estimators is to be read as an unmeasured cell, not as a soft result.
//!
//! An A/A null runs the SAME dtype in both slots, which is what says whether the alternation
//! itself is biased.
//!
//! Config is by ARGV, never env: `rch exec` does not forward the environment, and a probe that
//! reads env silently measures its default on a worker and exits 0
//! (`feedback_build_invocation_rules`).
//!
//!   cargo run --release -p frankentorch-api --example inplace_f32_dtype_probe -- \
//!       [numel] [reps] [threads]

use ft_api::FrankenTorchSession;
use ft_core::ExecutionMode;
use std::time::Instant;

/// One `exp_` on a fresh leaf of `numel` elements, timed. The leaf is built OUTSIDE the clock,
/// matching every other harness in this campaign — allocation is not what is being priced.
fn timed_exp_f64(values: &[f64]) -> (f64, f64) {
    let mut session = FrankenTorchSession::new(ExecutionMode::Strict);
    let x = session
        .tensor_variable(values.to_vec(), vec![values.len()], false)
        .expect("f64 leaf");
    let started = Instant::now();
    session.tensor_exp_(x).expect("exp_ f64");
    let elapsed = started.elapsed().as_secs_f64() * 1_000.0;
    // Teardown, after the clock, on both arms alike.
    let checksum = session
        .tensor_values(x)
        .expect("values")
        .iter()
        .map(|v| v.abs())
        .sum::<f64>();
    (elapsed, checksum)
}

fn timed_exp_f32(values: &[f32]) -> (f64, f64) {
    let mut session = FrankenTorchSession::new(ExecutionMode::Strict);
    let x = session
        .tensor_variable_f32(values.to_vec(), vec![values.len()], false)
        .expect("f32 leaf");
    let started = Instant::now();
    session.tensor_exp_(x).expect("exp_ f32");
    let elapsed = started.elapsed().as_secs_f64() * 1_000.0;
    let checksum = session
        .tensor_values_f32(x)
        .expect("values")
        .iter()
        .map(|v| f64::from(v.abs()))
        .sum::<f64>();
    (elapsed, checksum)
}

fn median(mut xs: Vec<f64>) -> f64 {
    xs.sort_by(|a, b| a.partial_cmp(b).expect("no NaN in timings"));
    let n = xs.len();
    if n == 0 {
        return f64::NAN;
    }
    if n % 2 == 1 {
        xs[n / 2]
    } else {
        f64::midpoint(xs[n / 2 - 1], xs[n / 2])
    }
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let numel: usize = args.get(1).and_then(|v| v.parse().ok()).unwrap_or(4 << 20);
    let reps: usize = args.get(2).and_then(|v| v.parse().ok()).unwrap_or(12);
    let threads: usize = args.get(3).and_then(|v| v.parse().ok()).unwrap_or(16);

    rayon::ThreadPoolBuilder::new()
        .num_threads(threads)
        .build_global()
        .expect("rayon pool");

    // PROVENANCE. A row without the machine and the pool width is not comparable to any other row
    // (`feedback_measurement_host_identity`), so it is printed whether or not anyone reads it.
    let host = std::fs::read_to_string("/etc/hostname").unwrap_or_else(|_| "unknown\n".to_owned());
    let loadavg = std::fs::read_to_string("/proc/loadavg").unwrap_or_else(|_| "unknown".to_owned());
    println!(
        "PROV host={} nproc={} rayon={} numel={} reps={} loadavg={}",
        host.trim(),
        std::thread::available_parallelism().map_or(0, std::num::NonZero::get),
        threads,
        numel,
        reps,
        loadavg.split_whitespace().take(3).collect::<Vec<_>>().join(","),
    );
    println!(
        "PREDICTION f32/f64 ~0.5 if the extra passes are free (half the bytes); \
         >=1.0 if they are not"
    );

    // Values in a range where exp is finite in BOTH dtypes: an f32 overflow to inf would make the
    // checksums incomparable and the timings unrepresentative of real work.
    let base_f64: Vec<f64> = (0..numel)
        .map(|i| (i % 1000) as f64 * 0.001 - 0.5)
        .collect();
    #[allow(clippy::cast_possible_truncation)]
    let base_f32: Vec<f32> = base_f64.iter().map(|&v| v as f32).collect();

    // Warm both arms before any timed rep. A cold first pass reads systematically slow, and
    // `feedback_drift_gate_measures_sweep_length` records discarding the first pass of a sweep as
    // worth 1.23x on the median.
    for _ in 0..3 {
        timed_exp_f64(&base_f64);
        timed_exp_f32(&base_f32);
    }

    let mut f64_mins = Vec::new();
    let mut f32_mins = Vec::new();
    let mut ratios = Vec::new();
    let (mut ck64, mut ck32) = (0.0_f64, 0.0_f64);

    for rep in 0..reps {
        // ALTERNATE which dtype occupies the first slot, so neither arm is systematically in the
        // warmer position.
        let (a, b) = if rep % 2 == 0 {
            let a = timed_exp_f64(&base_f64).0.min(timed_exp_f64(&base_f64).0);
            let (b, c) = timed_exp_f32(&base_f32);
            ck32 = c;
            (a, b.min(timed_exp_f32(&base_f32).0))
        } else {
            let (b, c) = timed_exp_f32(&base_f32);
            ck32 = c;
            let b = b.min(timed_exp_f32(&base_f32).0);
            let (a, d) = timed_exp_f64(&base_f64);
            ck64 = d;
            (a.min(timed_exp_f64(&base_f64).0), b)
        };
        if ck64 == 0.0 {
            ck64 = timed_exp_f64(&base_f64).1;
        }
        f64_mins.push(a);
        f32_mins.push(b);
        ratios.push(b / a);
        println!("REP {rep:2} f64 {a:8.3} ms   f32 {b:8.3} ms   f32/f64 {:.4}", b / a);
    }

    let paired = median(ratios.clone());
    let marginal = median(f32_mins.clone()) / median(f64_mins.clone());
    let f32_faster = ratios.iter().filter(|&&r| r < 1.0).count();

    println!();
    println!("f64 median-of-mins {:8.3} ms", median(f64_mins));
    println!("f32 median-of-mins {:8.3} ms", median(f32_mins));
    println!("PAIRED   (median of per-rep ratios) f32/f64 {paired:.4}");
    println!("MARGINAL (ratio of medians)         f32/f64 {marginal:.4}");
    println!("SIGN TEST f32 faster in {f32_faster} of {reps} reps");
    println!("CHECKSUM f64 {ck64:.6e}   f32 {ck32:.6e}");
    // BRANCH SENTINEL. The f32 block is gated at 8192 elements and this probe runs 4M, so reading
    // the source says "parallel". If the serial count is nonzero the source read is wrong, and if
    // the parallel count is nonzero the 10x has to be explained by something other than the gate.
    let (par, ser) = ft_api::take_inplace_unary_f32_branches();
    println!("F32 BRANCH parallel={par} serial={ser}");
    // The pre-fix decomposition, banked here so the comparison is legible without digging up the
    // old log: read(clone) 10.653 ms 69.2% | map 3.109 ms 20.2% | write(back) 1.640 ms 10.6%,
    // summing to 15.402 ms of a 19.374 ms arm. The clone and the writeback are gone now, so the
    // counters that produced those shares were removed rather than left reading zero.
    println!(
        "PRE-FIX FRAMES (banked, ledger 289): read 69.2% | map 20.2% | write 10.6% \
         — the clone alone was 10.653 ms"
    );
    if (paired - marginal).abs() > 0.05 {
        println!(
            "ESTIMATORS DISAGREE by {:.4} — treat this cell as UNMEASURED (274c/275b)",
            (paired - marginal).abs()
        );
    }
    println!(
        "READING: f32 moves half the bytes, so ~0.5 means the extra passes cost nothing; \
         near 1.0 means the 3-pass clone/map/writeback and the per-element f64 round trip have \
         eaten the entire dtype advantage."
    );

    // A/A NULL: the same dtype in both slots, through the identical alternation. This is what
    // says whether the harness itself is biased; a paired figure is not readable without it.
    let mut null_ratios = Vec::new();
    for rep in 0..reps {
        let first = timed_exp_f64(&base_f64).0.min(timed_exp_f64(&base_f64).0);
        let second = timed_exp_f64(&base_f64).0.min(timed_exp_f64(&base_f64).0);
        null_ratios.push(if rep % 2 == 0 {
            second / first
        } else {
            first / second
        });
    }
    let null = median(null_ratios);
    println!(
        "A/A NULL f64-vs-f64 {null:.4} ({})",
        if (0.97..=1.03).contains(&null) {
            "PASS"
        } else {
            "FAIL — the paired figure above is not readable"
        }
    );
}

//! `frankentorch-stale-tuning-constants-lzku6` — WHY did geqrf invert isolation-to-lane?
//!
//! # The question
//!
//! The geqrf panel-width sweep won ALL TWELVE cells for b=64 (median ~1.17x) and the paired lane
//! certification then read ~8% SLOWER with all four readings outside the null (ledger 292i).
//! Cholesky's and getrf's sweeps survived the same gate (292d, 292e). What is different?
//!
//! # The hypothesis, from arithmetic on the sweep's own output
//!
//! The incumbent b=32 cell varied by **1.34-1.47x across passes at every size**, while the effect
//! being measured was 1.17-1.24x. **The effect was smaller than the baseline's own run-to-run
//! spread** (effect/noise 0.36 at n=256, 0.52 at n=512, 0.72 at n=1024 — all below 1). The lane
//! harness, by contrast, showed an incumbent spread of only 1.130x: about three times steadier.
//!
//! And there is a structural reason. **My sweep measures arms in BLOCKS** — `for b { for rep }` —
//! so b=32 is always measured first, every pass, and any drift over the pass is confounded with the
//! candidate. The lane harness INTERLEAVES arms within each round and takes per-round paired
//! ratios, which cancels exactly that. All three of my sweeps (cholesky, getrf, geqrf) share the
//! block ordering; cholesky's and getrf's effects were simply large enough to survive it.
//!
//! # The test
//!
//! Same shapes, same widths, same everything — but arms INTERLEAVED per rep, with the order
//! reversed on odd reps so neither candidate sits permanently in the warmer slot, per-rep min, and
//! the reported figure the MEDIAN OF PER-REP RATIOS. Both estimators are printed, because
//! `feedback_estimator_and_provenance` records a 1.512x disagreement between them on identical
//! work, and disagreement here is itself the answer.
//!
//! **Registered prediction:** if block ordering explains the inversion, the interleaved estimate
//! for b=64 collapses toward or below 1.0 and the lane is vindicated. If b=64 still wins
//! interleaved, the inversion lives in what the LANE adds (session allocation, the torch
//! co-process's cache pressure) and not in how I measured.
//!
//!   cargo run --release -p frankentorch-kernel-cpu --example geqrf_nb_interleaved -- [n] [reps]

fn median(mut v: Vec<f64>) -> f64 {
    v.sort_by(|a, b| a.partial_cmp(b).expect("no NaN"));
    let n = v.len();
    if n == 0 {
        return f64::NAN;
    }
    if n % 2 == 1 { v[n / 2] } else { f64::midpoint(v[n / 2 - 1], v[n / 2]) }
}

fn time_once(a: &[f64], n: usize, b: usize) -> f64 {
    let prev = ft_kernel_cpu::set_householder_panel_width(b);
    let started = std::time::Instant::now();
    let (packed, tau) = ft_kernel_cpu::geqrf_blocked_nb_f64(a, n, n, b, 2, None);
    let ms = started.elapsed().as_secs_f64() * 1_000.0;
    std::hint::black_box((&packed, &tau));
    ft_kernel_cpu::set_householder_panel_width(prev);
    ms
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let n: usize = args.get(1).and_then(|v| v.parse().ok()).unwrap_or(256);
    let reps: usize = args.get(2).and_then(|v| v.parse().ok()).unwrap_or(21);

    let host = std::fs::read_to_string("/etc/hostname").unwrap_or_else(|_| "unknown\n".to_owned());
    let load = std::fs::read_to_string("/proc/loadavg").unwrap_or_else(|_| "unknown".to_owned());
    println!(
        "PROV host={} rayon={} n={n} reps={reps} loadavg={}",
        host.trim(),
        rayon::current_num_threads(),
        load.split_whitespace().take(3).collect::<Vec<_>>().join(","),
    );
    println!(
        "PREDICTION REGISTERED: if BLOCK ORDERING explains the isolation-to-lane inversion, the \
         interleaved b=64 estimate collapses toward or below 1.0. If b=64 still wins here, the \
         inversion lives in what the LANE adds, not in how I measured."
    );

    let a: Vec<f64> = (0..n * n)
        .map(|idx| {
            let (i, j) = (idx / n, idx % n);
            (((i * 31 + j * 17) % 97) as f64 - 48.0) / 24.0
        })
        .collect();

    for _ in 0..3 {
        std::hint::black_box(time_once(&a, n, 32));
        std::hint::black_box(time_once(&a, n, 64));
    }

    let (mut base, mut cand, mut ratios, mut null_ratios) = (Vec::new(), Vec::new(), Vec::new(), Vec::new());
    for rep in 0..reps {
        // ALTERNATE the order so neither width sits permanently in the warmer slot.
        let (b32, b64) = if rep % 2 == 0 {
            let x = time_once(&a, n, 32).min(time_once(&a, n, 32));
            let y = time_once(&a, n, 64).min(time_once(&a, n, 64));
            (x, y)
        } else {
            let y = time_once(&a, n, 64).min(time_once(&a, n, 64));
            let x = time_once(&a, n, 32).min(time_once(&a, n, 32));
            (x, y)
        };
        ratios.push(b32 / b64);
        base.push(b32);
        cand.push(b64);
        // A/A: the same width in both slots, through the identical alternation.
        let p = time_once(&a, n, 32).min(time_once(&a, n, 32));
        let q = time_once(&a, n, 32).min(time_once(&a, n, 32));
        null_ratios.push(if rep % 2 == 0 { p / q } else { q / p });
    }

    let paired = median(ratios.clone());
    let marginal = median(base.clone()) / median(cand.clone());
    let null = median(null_ratios);
    let wins = ratios.iter().filter(|&&r| r > 1.0).count();
    let (lo, hi) = (
        base.iter().copied().fold(f64::INFINITY, f64::min),
        base.iter().copied().fold(0.0_f64, f64::max),
    );

    println!("\nINTERLEAVED (arms alternate within each rep)");
    println!("  b=32 median {:.4} ms   b=64 median {:.4} ms", median(base), median(cand));
    println!("  PAIRED   (median of per-rep ratios, b32/b64)  {paired:.4}x");
    println!("  MARGINAL (ratio of medians)                   {marginal:.4}x");
    println!("  SIGN TEST b=64 faster in {wins} of {reps} reps");
    println!("  A/A NULL (b=32 vs b=32, same alternation)     {null:.4}x");
    println!(
        "  incumbent spread WITHIN this run: {:.4}-{:.4} ms = {:.3}x",
        lo,
        hi,
        hi / lo
    );
    if (paired - marginal).abs() > 0.05 {
        println!(
            "  ESTIMATORS DISAGREE by {:.4} — treat as UNMEASURED (274c/275b)",
            (paired - marginal).abs()
        );
    }
    println!(
        "\nCOMPARE: the BLOCK-ordered sweep reported b=64 at ~1.17x median for this width pair. \
         The lane certification reported 0.92x. Whichever of those this interleaved figure sits \
         beside is the one whose method was sound."
    );
}

//! `frankentorch-stale-tuning-constants-lzku6` — the geqrf panel-width sweep.
//!
//! # Converted to INTERLEAVED arms — ledger 293
//!
//! **This is the sweep that made the rule.** Block-ordered, it gave b=64 all twelve cells at a
//! median ~1.17x; the paired lane certification then read ~8% SLOWER with all four readings
//! outside the null (292i). Re-run interleaved at the same shapes and widths, b=64 at n=256
//! collapsed to sign tests of 11/21 and 9/21 — coin flips — landing beside the LANE's 0.92x and
//! not beside this sweep's 1.17x. **The lane was right and the harness was wrong** (293).
//!
//! Two defects, and both are now structurally impossible here rather than merely avoided:
//!
//!   1. `for b { for rep }` measured the incumbent FIRST in every pass, confounding drift with the
//!      candidate. Sampling now goes through `interleaved::run`, which owns the rep loop and hands
//!      the arm index out — a caller cannot re-block the ordering.
//!   2. The effect was smaller than the noise. The incumbent's WITHIN-RUN spread at n=256 was
//!      1.59-1.80x against a claimed 1.17x effect. That spread is now measured, printed, and is
//!      the gate: a cell whose effect does not exceed it prints UNRESOLVED with the arithmetic.
//!
//! # The confound this harness already existed to avoid
//!
//! torch:4's lane brief is the reason this file is not a copy of the cholesky sweep:
//!
//!   * **NB=32 is not one knob.** `geqrf_blocked_f64`, the public QR wrapper, `orgqr_blocked_f64`
//!     and `ormqr_blocked_f64` each hard-code their own 32. This sweep parameterises ONLY geqrf,
//!     via the existing `geqrf_blocked_nb_f64`, and touches no default.
//!   * **`HOUSEHOLDER_PANEL_WIDTH` is not the panel width.** It is an exact-shape skinny-split
//!     predicate: `m == WIDTH || k == WIDTH`, other dimension >= 512, flops >= 2^23. So at b=32 and
//!     n>=512 the panel GEMMs get a special parallel route that **a candidate b=16 or b=48 would
//!     silently lose**. Measuring that would be "b changed AND the split turned off" —
//!     `feedback_one_knob_is_secretly_two` exactly.
//!
//! **So this sweep moves the split width WITH the candidate**, and prints the admission census so
//! eligibility is PROVEN equalised rather than assumed. A cell whose admitted count does not track
//! its neighbours is a confounded cell and is called out as such. The incumbent arm leaves the
//! override UNSET and reads its width back from the kernel, so arm0 is the shipped route and not
//! the shipped number re-fed through the knob (`feedback_unset_knob_means_forced_off`).
//!
//! Leaf is held at 2 throughout: the brief is explicit that NB and leaf must not move together, and
//! commit 526267af already showed they interact (NB16 won n512 and lost n1024 at one leaf, NB32
//! won both at another).
//!
//! # Model (from the brief, not fitted here)
//!
//! Forward geqrf trailing work sums to `2/3(n^3 - n*b^2) + 1/2(b*n^2 - b^2*n)` MACs: the leading
//! `2/3 n^3` is effectively b-invariant, so smaller b raises outer-GEMM work slightly toward that
//! limit while lowering the middle T-multiply. Panel factor and T build are `Theta(b*n^2)` overall,
//! and `dlarft` scans all m per dot, so its serial component is ~`m*b^2/2` per panel — **smaller b
//! shrinks the serial term but multiplies panel count, allocations and GEMM launches.** At n=256 the
//! row-major path also pays R staging ~`n^3/(2b)`, favouring LARGER b; at n>=512 the column-major
//! path's two whole-matrix transposes are b-invariant and that pressure disappears. So the optimum
//! is expected to differ across the row-major -> column-major boundary, which is why n=256 is in the
//! sweep rather than assumed to behave like its neighbours.
//!
//! NOT BIT-EXACT across b — changing panel boundaries changes reduction association. Certify by
//! reconstruction and the oracle, never `to_bits()`.
//!
//!   cargo run --release -p frankentorch-kernel-cpu --example geqrf_nb_sweep -- [reps]

mod interleaved;

/// One arm's fastest sample: `(wall_ms, stage timings, admitted, rejected)`.
type Cell = (f64, ft_kernel_cpu::QrStageTimings, u64, u64);

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let reps = interleaved::reps_for(args.get(1).and_then(|v| v.parse().ok()).unwrap_or(10));

    interleaved::banner("geqrf panel width", reps);
    println!(
        "SPLIT EQUALISATION: the skinny-split width moves WITH the candidate b, so no b loses the \
         route the incumbent gets. The admitted/rejected census below is the PROOF, not the intent."
    );
    let shipped_b = ft_kernel_cpu::householder_panel_width();
    println!(
        "INCUMBENT: b={shipped_b}, read back from the kernel, run with the split override UNSET."
    );

    for n in [256usize, 512, 1024] {
        let a: Vec<f64> = (0..n * n)
            .map(|idx| {
                let (i, j) = (idx / n, idx % n);
                (((i * 31 + j * 17) % 97) as f64 - 48.0) / 24.0
            })
            .collect();

        let bs: Vec<usize> = [32usize, 48, 64, 96, 128, 192]
            .into_iter()
            .filter(|&b| b <= n)
            .collect();
        // Arm 0 is the incumbent; arms 1..=bs.len() are the candidates; the LAST arm is a second
        // copy of the incumbent — the A/A null, placed last so the odd-rep order reversal gives it
        // the maximum position contrast and therefore the least flattering reading.
        let arms = bs.len() + 2;
        let aa = arms - 1;
        let width = |i: usize| {
            if i == 0 || i == aa {
                shipped_b
            } else {
                bs[i - 1]
            }
        };
        let label = |i: usize| {
            if i == 0 {
                format!("shipped(b={shipped_b})")
            } else if i == aa {
                "A/A".to_owned()
            } else if bs[i - 1] == shipped_b {
                // CONTROL: the incumbent's own width, driven through the split-width KNOB instead
                // of left unset. It should read 1.0x; how far it misses is this run's floor, and
                // it is also the only thing that proves setting the override to the shipped value
                // is neutral rather than a route change.
                format!("b={}*", bs[i - 1])
            } else {
                format!("b={}", bs[i - 1])
            }
        };

        let mut best: Vec<Cell> = vec![
            (
                f64::INFINITY,
                ft_kernel_cpu::QrStageTimings::default(),
                0,
                0
            );
            arms
        ];
        let times = interleaved::run(arms, reps, 2, |i| {
            let b = width(i);
            // EQUALISE ELIGIBILITY: the predicate matches an exact width, so it must follow b.
            // The incumbent arms leave it UNSET, which is the shipped route by construction.
            let prev =
                ft_kernel_cpu::set_householder_panel_width(if i == 0 || i == aa { 0 } else { b });
            let _ = ft_kernel_cpu::householder_skinny_census_take();
            let mut t = ft_kernel_cpu::QrStageTimings::default();
            let started = std::time::Instant::now();
            let (packed, tau) = ft_kernel_cpu::geqrf_blocked_nb_f64(&a, n, n, b, 2, Some(&mut t));
            let wall = started.elapsed().as_secs_f64() * 1_000.0;
            std::hint::black_box((&packed, &tau));
            let (adm, rej) = ft_kernel_cpu::householder_skinny_census_take();
            ft_kernel_cpu::set_householder_panel_width(prev);
            if wall < best[i].0 {
                best[i] = (wall, t, adm, rej);
            }
            wall
        });
        ft_kernel_cpu::set_householder_panel_width(0);

        let (lo, hi, gate) = interleaved::spread(&times[0]);
        println!("\nn={n}   (leaf=2 fixed; row-major path at n=256, column-major at n>=512)");
        println!(
            "  incumbent WITHIN-RUN spread {lo:.4}-{hi:.4} ms = {gate:.3}x  (IQR {:.3}x) \
             — THE GATE: an effect at or below {:.4} is UNRESOLVED, not a result",
            interleaved::iqr_ratio(&times[0]),
            gate - 1.0
        );
        println!(
            "  {:>15} {:>9} {:>8} {:>8} {:>8} {:>8} {:>8} {:>8} {:>11} {}",
            "arm",
            "median",
            "copy",
            "panelT",
            "tbuild",
            "trailR",
            "pack",
            "gemm",
            "adm/rej",
            interleaved::Verdict::header(),
        );
        for i in 0..arms {
            let (_, t, adm, rej) = best[i];
            let ms = |v: u128| v as f64 / 1e6;
            let trust = if i == 0 {
                format!("{:>9} {:>9} {:>8} {:>8}  incumbent", "-", "-", "-", "-")
            } else {
                interleaved::verdict(&times[0], &times[i], gate).row()
            };
            println!(
                "  {:>15} {:>9.4} {:>8.4} {:>8.4} {:>8.4} {:>8.4} {:>8.4} {:>8.4} {:>11} {trust}",
                label(i),
                interleaved::median(&times[i]),
                ms(t.copy_zeroing_ns),
                ms(t.panel_and_t_ns),
                ms(t.panel_t_build_ns),
                ms(t.trailing_r_ns),
                ms(t.trailing_pack_ns),
                ms(t.trailing_gemm_ns),
                format!("{adm}/{rej}"),
            );
        }
    }
    ft_kernel_cpu::set_householder_panel_width(0);
    println!(
        "\nREADING: check the split census FIRST — a cell whose admitted count does not track its \
         neighbours measured a different dispatch, not a different b. Then the A/A row: if the \
         harness's own null is not ~1.0 on a coin-flip sign test, nothing below it is a \
         measurement. Only then the verdicts, and only an ALL-CELLS TRUSTED WIN followed by a \
         paired lane certification is a reason to move the default. This exact sweep, \
         block-ordered, already produced one all-cells win that the lane refuted by 25 points."
    );
}

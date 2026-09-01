//! `frankentorch-stale-tuning-constants-lzku6` — the geqrf panel-width sweep.
//!
//! # The confound this harness exists to avoid
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
//! its neighbours is a confounded cell and is called out as such.
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

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let reps: usize = args.get(1).and_then(|v| v.parse().ok()).unwrap_or(7);

    let host = std::fs::read_to_string("/etc/hostname").unwrap_or_else(|_| "unknown\n".to_owned());
    let load = std::fs::read_to_string("/proc/loadavg").unwrap_or_else(|_| "unknown".to_owned());
    println!(
        "PROV host={} nproc={} rayon={} reps={reps} loadavg={}",
        host.trim(),
        std::thread::available_parallelism().map_or(0, std::num::NonZero::get),
        rayon::current_num_threads(),
        load.split_whitespace().take(3).collect::<Vec<_>>().join(","),
    );
    println!(
        "SPLIT EQUALISATION: the skinny-split width moves WITH the candidate b, so no b loses the \
         route b=32 gets. The admitted/rejected census below is the PROOF, not the intent."
    );

    for n in [256usize, 512, 1024] {
        let a: Vec<f64> = (0..n * n)
            .map(|idx| {
                let (i, j) = (idx / n, idx % n);
                (((i * 31 + j * 17) % 97) as f64 - 48.0) / 24.0
            })
            .collect();
        println!("\nn={n}   (leaf=2 fixed; row-major path at n=256, column-major at n>=512)");
        println!(
            "  {:>4} {:>9} {:>9} {:>9} {:>9} {:>9} {:>9} {:>8} {:>13}",
            "b", "copy", "panelT", "tbuild", "trailR", "pack", "gemm", "TOTAL", "split adm/rej"
        );
        let mut cells: Vec<(usize, f64)> = Vec::new();
        for b in [32usize, 48, 64, 96, 128, 192] {
            if b > n {
                continue;
            }
            // EQUALISE ELIGIBILITY: the predicate matches an exact width, so it must follow b.
            let prev_width = ft_kernel_cpu::set_householder_panel_width(b);
            for _ in 0..2 {
                let _ = ft_kernel_cpu::geqrf_blocked_nb_f64(&a, n, n, b, 2, None);
            }
            let mut best = (f64::INFINITY, ft_kernel_cpu::QrStageTimings::default(), 0u64, 0u64);
            for _ in 0..reps {
                let _ = ft_kernel_cpu::householder_skinny_census_take();
                let mut t = ft_kernel_cpu::QrStageTimings::default();
                let started = std::time::Instant::now();
                let (packed, tau) = ft_kernel_cpu::geqrf_blocked_nb_f64(&a, n, n, b, 2, Some(&mut t));
                let wall = started.elapsed().as_secs_f64() * 1_000.0;
                std::hint::black_box((&packed, &tau));
                let (adm, rej) = ft_kernel_cpu::householder_skinny_census_take();
                if wall < best.0 {
                    best = (wall, t, adm, rej);
                }
            }
            ft_kernel_cpu::set_householder_panel_width(prev_width);
            let (wall, t, adm, rej) = best;
            let ms = |v: u128| v as f64 / 1e6;
            cells.push((b, wall));
            println!(
                "  {b:>4} {:>9.4} {:>9.4} {:>9.4} {:>9.4} {:>9.4} {:>9.4} {:>8.4} {:>13}",
                ms(t.copy_zeroing_ns),
                ms(t.panel_and_t_ns),
                ms(t.panel_t_build_ns),
                ms(t.trailing_r_ns),
                ms(t.trailing_pack_ns),
                ms(t.trailing_gemm_ns),
                wall,
                format!("{adm}/{rej}"),
            );
        }
        let shipped = cells
            .iter()
            .find(|(b, _)| *b == 32)
            .map_or(f64::NAN, |(_, ms)| *ms);
        let best = cells
            .iter()
            .copied()
            .fold((0usize, f64::INFINITY), |acc, c| if c.1 < acc.1 { c } else { acc });
        println!(
            "  BEST b={} at {:.4} ms; incumbent b=32 {:.4} ms  ->  {:.4}x",
            best.0,
            best.1,
            shipped,
            shipped / best.1
        );
    }
    ft_kernel_cpu::set_householder_panel_width(0);
    println!(
        "\nREADING: ship only on an ALL-CELLS win across the three sizes, then a paired lane \
         certification. And check the split census FIRST — a cell whose admitted count does not \
         track its neighbours measured a different dispatch, not a different b."
    );
}

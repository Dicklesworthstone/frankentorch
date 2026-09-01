//! `frankentorch-valnx` — the LU (getrf) NB sweep, both phases counted, at three sizes.
//!
//! # Why this is NOT the cholesky change repeated
//!
//! Cholesky's NB went 128 -> 64 for 1.264x (ledger 292d). Applying the same direction here would be
//! a REGRESSION: LU's NB was already retuned 64 -> 128 by `lu_factor_contiguous_f64`'s own ladder,
//! winning 1.213x / 1.133x / 1.165x at n=512/520/1024. **The two ops' optima move in OPPOSITE
//! directions, and the panel-MAC model says why.**
//!
//!     cholesky panel is nb x nb   ->  MACs ~ n*nb^2/6   QUADRATIC in nb
//!     LU       panel is m  x nb   ->  MACs ~ n^2*nb/4   LINEAR    in nb
//!
//! Halving cholesky's nb cuts its panel work FOURFOLD, which is why small nb wins there. Halving
//! LU's only halves it, so the trailing GEMM's preference for a larger `k` dominates instead. LU's
//! panel is also `lu_factor_panel_recursive_f64`, which turns its internal updates into GEMM — it
//! is not the pure scalar chain cholesky's panel is, so it runs at a far better rate and tolerates
//! being larger.
//!
//! **So the transferable thing is the METHOD, not the constant.** That is the whole point of
//! re-deriving the model per op instead of propagating a number.
//!
//! # What is actually open here
//!
//! The existing ladder swept {16, 32, 48, 64, 96, 128} and **128 won at the TOP of that grid**. A
//! winner at the edge of a grid is a winner against nothing above it. This file's own ledger
//! records the same mistake with the sign reversed: the SVD panel width sat at 16 because the grid
//! was {16,32,64} and never held 8, and 8 then won 15 of 16 cells. So this sweeps 192, 256 and 384
//! as well, and reports both phases per cell so the ANSWER comes with its mechanism.
//!
//! # Reported
//!
//! Per (n, nb): panel / solve / trailing from the in-kernel counters, the residual, the wall time,
//! and the panel MAC count computed EXACTLY (not from the asymptotic form) so the measured panel
//! rate can be compared across nb — a model checked only where it was fitted is not a model.
//!
//! NOT BIT-EXACT across nb: blocking reassociates. LU is tolerance-validated for the same reason
//! cholesky is.
//!
//!   cargo run --release -p frankentorch-kernel-cpu --example lu_nb_sweep -- [reps]

use ft_core::{DType, Device, TensorMeta};

/// EXACT panel multiply-accumulates for a blocked getrf: panel `p` has `m = n - p*nb` rows, and
/// its unblocked factorisation does `(m-j-1)*(nb-j-1)` MACs for each column `j`.
fn lu_panel_macs(n: usize, nb: usize) -> f64 {
    let mut total = 0.0f64;
    let mut jb = 0usize;
    while jb < n {
        let width = nb.min(n - jb);
        let m = n - jb;
        for j in 0..width {
            let rows = m.saturating_sub(j + 1);
            let cols = width.saturating_sub(j + 1);
            total += (rows * cols) as f64;
        }
        jb += nb;
    }
    total
}

fn diag_dominant(n: usize) -> Vec<f64> {
    (0..n * n)
        .map(|idx| {
            let (i, j) = (idx / n, idx % n);
            let v = (((i * 7 + j * 13) % 101) as f64 - 50.0) / 25.0;
            if i == j { v + n as f64 } else { v }
        })
        .collect()
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let reps: usize = args.get(1).and_then(|v| v.parse().ok()).unwrap_or(9);

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
        "MODEL: LU panel MACs ~ n^2*nb/4, LINEAR in nb (cholesky's is n*nb^2/6, QUADRATIC) — which \
         is why the two ops' optima move in OPPOSITE directions. Shipped LU nb is 128, and it won \
         at the TOP of a {{16..128}} grid, so 192/256/384 are the untested side."
    );

    for n in [256usize, 512, 1024] {
        let a = diag_dominant(n);
        let meta = TensorMeta::from_shape(vec![n, n], DType::F64, Device::Cpu);
        println!(
            "\nn={n}   total getrf MACs ~ n^3/3 = {:.3e}",
            (n as f64).powi(3) / 3.0
        );
        println!(
            "  {:>5} {:>9} {:>9} {:>9} {:>9} {:>9} {:>11} {:>10}",
            "nb", "panel", "solve", "trailing", "resid", "TOTAL", "panel MACs", "panel GF/s"
        );
        let mut best = (f64::INFINITY, 0usize);
        let mut shipped = f64::INFINITY;
        for nb in [32usize, 64, 96, 128, 192, 256, 384] {
            if nb > n {
                continue;
            }
            for _ in 0..2 {
                let _ = ft_kernel_cpu::lu_factor_contiguous_nb_f64(&a, &meta, nb).expect("warm");
            }
            let mut cell = (f64::INFINITY, 0u64, 0u64, 0u64);
            for _ in 0..reps {
                let _ = ft_kernel_cpu::lu_stage_take_ns();
                let started = std::time::Instant::now();
                let f = ft_kernel_cpu::lu_factor_contiguous_nb_f64(&a, &meta, nb).expect("lu");
                let wall = started.elapsed().as_secs_f64() * 1_000.0;
                std::hint::black_box(&f);
                let (p, s, t) = ft_kernel_cpu::lu_stage_take_ns();
                if wall < cell.0 {
                    cell = (wall, p, s, t);
                }
            }
            let (wall, p, s, t) = cell;
            let ms = |v: u64| v as f64 / 1e6;
            let resid = wall - (ms(p) + ms(s) + ms(t));
            let macs = lu_panel_macs(n, nb);
            println!(
                "  {nb:>5} {:>9.4} {:>9.4} {:>9.4} {:>9.4} {:>9.4} {macs:>11.3e} {:>10.2}",
                ms(p),
                ms(s),
                ms(t),
                resid,
                wall,
                2.0 * macs / (ms(p) / 1000.0) / 1e9
            );
            if wall < best.0 {
                best = (wall, nb);
            }
            if nb == 128 {
                shipped = wall;
            }
        }
        println!(
            "  BEST nb={} at {:.4} ms; SHIPPED (nb=128) {:.4} ms  ->  {:.4}x",
            best.1,
            best.0,
            shipped,
            shipped / best.0
        );
    }
    println!(
        "\nREADING: ship only on an ALL-CELLS win across the three sizes AND a paired lane \
         certification, per the cholesky template (292d). A winner in one cell of a noisy sweep is \
         the friendliest reading, not the answer."
    );
}

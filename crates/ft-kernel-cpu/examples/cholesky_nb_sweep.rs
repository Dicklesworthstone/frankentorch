//! `frankentorch-valnx` — the Cholesky NB sweep, with BOTH phases counted, at three sizes.
//!
//! # The model, stated before the run
//!
//! Panel MACs over a whole factorisation are `(n/nb)·nb³/6 = n·nb²/6`. At n=512, nb=128 that
//! predicts 1,398,101 against a measured census of 1,398,016 (ledger 292a), so the model is EXACT
//! rather than fitted. The trailing update always does ~`n³/3` MACs whatever nb is; only its `k`
//! changes. Measured rates: panel **0.85 GF/s**, trailing **~90 GF/s** — a 100x gap per FLOP.
//!
//! So shrinking nb trades a QUADRATIC panel term against a shape change in a term that is two
//! orders of magnitude faster. At n=512 the prediction is:
//!
//!     nb    panel MACs   panel ms   trailing ms (if it holds 90 GF/s)
//!     128    1,398,101     3.29            0.99
//!      64      349,525     0.82            0.99
//!      32       87,381     0.21            0.99
//!
//! Halving nb should remove ~2.5 ms of a 5.7 ms lane. **The entire risk is one unmeasured
//! quantity: the trailing update's rate at smaller `k`.** The break-even is a trailing rate of
//! 25.9 GF/s — the swap pays unless k=64 costs that GEMM a 3.5x degradation from ~90.
//!
//! # Why three sizes
//!
//! The optimum is a CURVE in n, not a constant: the panel term scales as `n·nb²` and the trailing
//! term as `n³`, so their ratio moves with n and an nb tuned at one size is a gate fitted to one
//! dataset (item 279 — the size curve IS the result). n=256/512/1024.
//!
//! # What is reported
//!
//! Every phase from the in-kernel counters — panel, TRSM, trailing, upper-zero — plus the wall
//! time and the residual, per (n, nb), so the accounting CLOSES at each cell rather than only in
//! aggregate. The predicted panel ms is printed beside the measured one, because a model that is
//! only checked where it was fitted is not a model.
//!
//! NOT BIT-EXACT ACROSS nb, and it does not need to be: blocking reassociates the trailing sums,
//! which is why the blocked path is already tolerance-validated rather than bitwise. Correctness
//! across widths is pinned by `cholesky_nb_knob_is_neutral_at_default_and_correct_elsewhere`.
//!
//!   cargo run --release -p frankentorch-kernel-cpu --example cholesky_nb_sweep -- [reps]

use ft_core::{DType, Device, TensorMeta};

fn spd(n: usize) -> Vec<f64> {
    let mut a = vec![0.0f64; n * n];
    for i in 0..n {
        for j in 0..i {
            let v = ((i * 17 + j * 31) % 23) as f64 * 0.01 - 0.1;
            a[i * n + j] = v;
            a[j * n + i] = v;
        }
        a[i * n + i] = n as f64;
    }
    a
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
        "MODEL (registered): panel MACs = n*nb^2/6 at ~0.85 GF/s; trailing = n^3/3 MACs at ~90 \
         GF/s regardless of nb. Predicted panel ms is printed beside the measured one."
    );

    for n in [256usize, 512, 1024] {
        let a = spd(n);
        let meta = TensorMeta::from_shape(vec![n, n], DType::F64, Device::Cpu);
        println!(
            "\nn={n}   trailing MACs ~ n^3/3 = {:.3e}  (nb-INDEPENDENT)",
            (n as f64).powi(3) / 3.0
        );
        println!(
            "  {:>5} {:>9} {:>9} {:>9} {:>9} {:>9} {:>9} {:>8} {:>10}",
            "nb", "panel", "pred", "TRSM", "trailing", "zero", "resid", "TOTAL", "trail GF/s"
        );
        let mut best = (f64::INFINITY, 0usize);
        for nb in [16usize, 32, 64, 128, 256] {
            if nb > n {
                continue;
            }
            ft_kernel_cpu::set_cholesky_nb(nb);
            for _ in 0..2 {
                let _ = ft_kernel_cpu::cholesky_contiguous_f64(&a, &meta, false).expect("warm");
            }
            let mut cell = (f64::INFINITY, 0u64, 0u64, 0u64, 0u64);
            for _ in 0..reps {
                let _ = ft_kernel_cpu::cholesky_stage_take_ns();
                let started = std::time::Instant::now();
                let f = ft_kernel_cpu::cholesky_contiguous_f64(&a, &meta, false).expect("chol");
                let wall = started.elapsed().as_secs_f64() * 1_000.0;
                std::hint::black_box(&f);
                let (p, t, tr, z) = ft_kernel_cpu::cholesky_stage_take_ns();
                if wall < cell.0 {
                    cell = (wall, p, t, tr, z);
                }
            }
            let (wall, p, t, tr, z) = cell;
            let ms = |v: u64| v as f64 / 1e6;
            let resid = wall - (ms(p) + ms(t) + ms(tr) + ms(z));
            let pred_panel = 2.0 * (n as f64) * (nb as f64) * (nb as f64) / 6.0 / 0.85e9 * 1000.0;
            let trail_gf = 2.0 * (n as f64).powi(3) / 3.0 / (ms(tr) / 1000.0) / 1e9;
            println!(
                "  {nb:>5} {:>9.4} {:>9.4} {:>9.4} {:>9.4} {:>9.4} {:>9.4} {:>8.4} {:>10.1}",
                ms(p),
                pred_panel,
                ms(t),
                ms(tr),
                ms(z),
                resid,
                wall,
                trail_gf
            );
            if wall < best.0 {
                best = (wall, nb);
            }
        }
        ft_kernel_cpu::set_cholesky_nb(0);
        let shipped = {
            let mut w = f64::INFINITY;
            for _ in 0..reps {
                let started = std::time::Instant::now();
                let f = ft_kernel_cpu::cholesky_contiguous_f64(&a, &meta, false).expect("chol");
                w = w.min(started.elapsed().as_secs_f64() * 1_000.0);
                std::hint::black_box(&f);
            }
            w
        };
        println!(
            "  BEST nb={} at {:.4} ms; SHIPPED (nb=128) {:.4} ms  ->  {:.4}x",
            best.1,
            best.0,
            shipped,
            shipped / best.0
        );
    }
    ft_kernel_cpu::set_cholesky_nb(0);
    println!(
        "\nREADING: a winning nb here is necessary and not sufficient — item 291 recorded a \
         bit-exact 1.28-1.40x isolation win that moved no lane. Any nb change is also NOT \
         bit-exact (blocking reassociates the trailing sums), so it needs reconstruction and the \
         oracle, never to_bits()."
    );
}

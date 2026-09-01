//! `frankentorch-stale-tuning-constants-lzku6` lane 2 — the getri (`inv`) NB sweep.
//!
//! # The census picked this constant, not the one the lane list started with
//!
//! Lane 2 was opened against `lu_solve_contiguous_f64`'s NB. A census killed that: neither slogdet
//! NOR inv executes `lu_solve` — its counters read 0.0000 ms for both — because `inv` goes through
//! `lu_inverse_from_factor_f64`, a different function with its own `NB`, and `lu_solve` serves
//! `linalg.solve` instead. Re-tuning the first constant would have measured nothing, which is
//! ledger 292's finding restated (cholesky never calls `dgemm_sub_into` either).
//!
//! **So the constant swept here is the one `inv` actually runs**, and `inv` is a board lane, which
//! is what makes it certifiable.
//!
//! # The model
//!
//! getri here is two blocked triangular solves against the identity: FORWARD `L y = I` restricted
//! to the columns that can be nonzero (the identity-structure restriction shipped under 37sxo), and
//! BACKWARD `U x = y`. Each block of width nb does a small triangular solve on the diagonal block
//! plus a GEMM update of the remaining columns — so the per-block scalar work grows with nb while
//! the GEMM's `k` grows with it too. That is the same competing pair as the factorisations, but the
//! solve's scalar half is a TRSM rather than a panel factorisation, so its exponent is different
//! again and the optimum has to be measured rather than inherited from 292d/292e.
//!
//! Both halves are counted in-call (`lu_inverse_half_take_ns`) so the sweep reports WHERE the time
//! moves, not just that it moved.
//!
//! # Rules
//!
//! Three sizes; sweep both sides of the incumbent 64 (the getrf ladder's winner sat at the TOP of
//! its grid and was a winner against nothing above it); ship only on an ALL-CELLS win; then a
//! paired lane certification on `inv`. NOT bit-exact — blocking reassociates.
//!
//!   cargo run --release -p frankentorch-kernel-cpu --example inv_nb_sweep -- [reps]

use ft_core::{DType, Device, TensorMeta};

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
        "CENSUS THAT PICKED THIS CONSTANT: slogdet and inv both read 0.0000 ms of lu_solve, so the \
         NB that `inv` executes is lu_inverse_from_factor_f64's, not lu_solve_contiguous_f64's."
    );

    for n in [256usize, 512, 1024] {
        let a: Vec<f64> = (0..n * n)
            .map(|idx| {
                let (i, j) = (idx / n, idx % n);
                let v = (((i * 7 + j * 13) % 101) as f64 - 50.0) / 25.0;
                if i == j { v + n as f64 } else { v }
            })
            .collect();
        let meta = TensorMeta::from_shape(vec![n, n], DType::F64, Device::Cpu);
        println!("\nn={n}");
        println!(
            "  {:>5} {:>10} {:>10} {:>10} {:>10} {:>9}",
            "nb", "forward", "backward", "resid", "TOTAL", "vs 64"
        );
        let mut cells: Vec<(usize, f64)> = Vec::new();
        for nb in [16usize, 32, 64, 96, 128, 192] {
            if nb > n {
                continue;
            }
            ft_kernel_cpu::set_lu_inv_nb(nb);
            for _ in 0..2 {
                let _ = ft_kernel_cpu::inv_tensor_contiguous_f64(&a, &meta).expect("warm");
            }
            let mut cell = (f64::INFINITY, 0u64, 0u64);
            for _ in 0..reps {
                let _ = ft_kernel_cpu::lu_inverse_half_take_ns();
                let started = std::time::Instant::now();
                let v = ft_kernel_cpu::inv_tensor_contiguous_f64(&a, &meta).expect("inv");
                let wall = started.elapsed().as_secs_f64() * 1_000.0;
                std::hint::black_box(&v);
                let (f, b) = ft_kernel_cpu::lu_inverse_half_take_ns();
                if wall < cell.0 {
                    cell = (wall, f, b);
                }
            }
            let (wall, f, b) = cell;
            let ms = |v: u64| v as f64 / 1e6;
            cells.push((nb, wall));
            println!(
                "  {nb:>5} {:>10.4} {:>10.4} {:>10.4} {:>10.4} {:>9}",
                ms(f),
                ms(b),
                wall - (ms(f) + ms(b)),
                wall,
                "-"
            );
        }
        ft_kernel_cpu::set_lu_inv_nb(0);
        let shipped = cells
            .iter()
            .find(|(nb, _)| *nb == 64)
            .map_or(f64::NAN, |(_, ms)| *ms);
        let best = cells
            .iter()
            .copied()
            .fold((0usize, f64::INFINITY), |acc, c| if c.1 < acc.1 { c } else { acc });
        println!(
            "  BEST nb={} at {:.4} ms; incumbent nb=64 {:.4} ms  ->  {:.4}x",
            best.0,
            best.1,
            shipped,
            shipped / best.1
        );
    }
    ft_kernel_cpu::set_lu_inv_nb(0);
    println!(
        "\nREADING: ship only on an ALL-CELLS win plus a paired lane certification on `inv`. \
         A kernel win is not a lane win — ledger 291."
    );
}

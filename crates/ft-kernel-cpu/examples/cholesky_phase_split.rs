//! `frankentorch-valnx` — where does a blocked Cholesky actually spend?
//!
//! # Why this run exists
//!
//! Two lanes now say the trailing-update GEMM is NOT where this family spends. The 2-D
//! `dgemm_sub_into` arm is bit-exact and wins 1.28-1.40x in isolation, and it moved neither
//! slogdet (5 of 31 calls qualified) nor inv (19 of 45, 42%) — ledger 291/291a. Coverage was not
//! the problem, so the target is.
//!
//! This does not propose a lever. It measures the phase split so the NEXT lever is chosen from a
//! decomposition rather than from the assumption that a factorisation is its GEMM.
//!
//! # Method
//!
//! `cholesky_stage_take_ns()` already reports `(panel, trsm, trailing, upper_zero)` from INSIDE
//! the kernel — counters in the call, never a subtraction between arms, which is the rule item
//! 277a was written for. This adds two things they cannot give on their own:
//!
//!   * ACCOUNTING CLOSURE. The four phases are compared against the wall time of the same call.
//!     Whatever is left is GLUE — allocation, the symmetric copy-in, bounds work — and naming it
//!     as a residual is honest only if the residual is shown rather than assumed away.
//!   * The `dgemm_sub_into` CENSUS for this op, so the trailing share can be read together with
//!     how many of its calls would even qualify for the 2-D arm.
//!
//! Reported as MIN over reps (the phase counters accumulate, so they are drained per rep and the
//! rep with the smallest wall time is the one quoted — mixing a min wall with summed counters
//! across reps would attribute one rep's phases to another's total).
//!
//! Config by ARGV, never env: `rch exec` does not forward the environment.
//!
//!   cargo run --release -p frankentorch-kernel-cpu --example cholesky_phase_split -- [n] [reps]

use ft_core::{DType, Device, TensorMeta};

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let n: usize = args.get(1).and_then(|v| v.parse().ok()).unwrap_or(512);
    let reps: usize = args.get(2).and_then(|v| v.parse().ok()).unwrap_or(9);

    let host = std::fs::read_to_string("/etc/hostname").unwrap_or_else(|_| "unknown\n".to_owned());
    let load = std::fs::read_to_string("/proc/loadavg").unwrap_or_else(|_| "unknown".to_owned());
    println!(
        "PROV host={} nproc={} rayon={} n={n} reps={reps} loadavg={}",
        host.trim(),
        std::thread::available_parallelism().map_or(0, std::num::NonZero::get),
        rayon::current_num_threads(),
        load.split_whitespace().take(3).collect::<Vec<_>>().join(","),
    );

    // Diagonally dominant so the factorisation is well conditioned and every panel does real work.
    let mut a = vec![0.0f64; n * n];
    for i in 0..n {
        for j in 0..i {
            let v = ((i * 17 + j * 31) % 23) as f64 * 0.01 - 0.1;
            a[i * n + j] = v;
            a[j * n + i] = v;
        }
        a[i * n + i] = n as f64;
    }
    let meta = TensorMeta::from_shape(vec![n, n], DType::F64, Device::Cpu);

    // The fine-grained panel census is OPT-IN because it costs two `Instant` pairs per COLUMN.
    // Its own overhead therefore lands INSIDE the panel total it reports, so it is quantified
    // below by measuring the same phase with the census off rather than hoped to be negligible.
    ft_kernel_cpu::set_cholesky_panel_census_enabled(false);
    for _ in 0..2 {
        let _ = ft_kernel_cpu::cholesky_contiguous_f64(&a, &meta, false).expect("warmup");
    }
    let mut panel_uninstrumented = f64::INFINITY;
    for _ in 0..reps {
        let _ = ft_kernel_cpu::cholesky_stage_take_ns();
        let f = ft_kernel_cpu::cholesky_contiguous_f64(&a, &meta, false).expect("cholesky");
        std::hint::black_box(&f);
        let (panel, _, _, _) = ft_kernel_cpu::cholesky_stage_take_ns();
        panel_uninstrumented = panel_uninstrumented.min(panel as f64 / 1e6);
    }
    ft_kernel_cpu::set_cholesky_panel_census_enabled(true);

    let mut best = (f64::INFINITY, 0u64, 0u64, 0u64, 0u64, 0u64, 0u64);
    let mut panel_census = (0u64, 0u64, 0u64, 0u64);
    for _ in 0..reps {
        // Drain BOTH counter families so this rep's numbers cannot inherit the last rep's.
        let _ = ft_kernel_cpu::cholesky_stage_take_ns();
        let _ = ft_kernel_cpu::dgemm_sub_arm_hits_take();
        let _ = ft_kernel_cpu::cholesky_panel_census_take();
        let started = std::time::Instant::now();
        let factor = ft_kernel_cpu::cholesky_contiguous_f64(&a, &meta, false).expect("cholesky");
        let wall = started.elapsed().as_secs_f64() * 1_000.0;
        std::hint::black_box(&factor);
        let (panel, trsm, trailing, zero) = ft_kernel_cpu::cholesky_stage_take_ns();
        let (tiled, col) = ft_kernel_cpu::dgemm_sub_arm_hits_take();
        let census = ft_kernel_cpu::cholesky_panel_census_take();
        if wall < best.0 {
            best = (wall, panel, trsm, trailing, zero, tiled, col);
            panel_census = census;
        }
    }

    let (wall, panel, trsm, trailing, zero, tiled, col) = best;
    let ms = |v: u64| v as f64 / 1e6;
    let pct = |v: u64| 100.0 * ms(v) / wall;
    let accounted = ms(panel) + ms(trsm) + ms(trailing) + ms(zero);
    let glue = wall - accounted;

    println!("\nBLOCKED CHOLESKY n={n}, min-wall rep of {reps}");
    println!("  {:<22} {:>9} {:>8}", "phase", "ms", "% lane");
    println!("  {:<22} {:>9.4} {:>7.1}%", "panel factor", ms(panel), pct(panel));
    println!("  {:<22} {:>9.4} {:>7.1}%", "TRSM (panel solve)", ms(trsm), pct(trsm));
    println!("  {:<22} {:>9.4} {:>7.1}%", "trailing update", ms(trailing), pct(trailing));
    println!("  {:<22} {:>9.4} {:>7.1}%", "strict-upper zero", ms(zero), pct(zero));
    println!(
        "  {:<22} {:>9.4} {:>7.1}%   <- residual, NOT a measured phase",
        "glue (residual)", glue, 100.0 * glue / wall
    );
    println!("  {:<22} {:>9.4} {:>7.1}%", "TOTAL (wall)", wall, 100.0);
    println!(
        "\nACCOUNTING CLOSURE: phases {accounted:.4} ms of {wall:.4} ms wall = {:.1}%. \
         The remainder is glue and is shown, not assumed away.",
        100.0 * accounted / wall
    );
    println!(
        "dgemm_sub_into CENSUS for this op: (tiled_2d, column_split) = ({tiled}, {col}) — \
         with the 2-D arm DEFAULT OFF, so `tiled` is expected to read 0 and `col` is the \
         trailing update's call count."
    );
    // ---- inside the panel -------------------------------------------------------------------
    let (diag_ns, rows_ns, cols, macs) = panel_census;
    let panel_ms = ms(panel);
    let inner = ms(diag_ns) + ms(rows_ns);
    println!("\nINSIDE THE PANEL (census ON; its own cost is quantified below)");
    println!("  {:<26} {:>9} {:>8}", "sub-phase", "ms", "% panel");
    println!(
        "  {:<26} {:>9.4} {:>7.1}%",
        "diagonal dot + sqrt",
        ms(diag_ns),
        100.0 * ms(diag_ns) / panel_ms
    );
    println!(
        "  {:<26} {:>9.4} {:>7.1}%",
        "sub-diagonal row dots",
        ms(rows_ns),
        100.0 * ms(rows_ns) / panel_ms
    );
    println!(
        "  {:<26} {:>9.4} {:>7.1}%   <- residual, NOT a measured sub-phase",
        "loop/index residual",
        panel_ms - inner,
        100.0 * (panel_ms - inner) / panel_ms
    );
    println!("  {:<26} {:>9.4} {:>7.1}%", "PANEL total", panel_ms, 100.0);
    println!(
        "  closure: sub-phases {inner:.4} of {panel_ms:.4} ms = {:.1}%",
        100.0 * inner / panel_ms
    );
    println!(
        "\n  columns {cols}, multiply-accumulates {macs}  ->  the panel achieves {:.2} GFLOP/s",
        2.0 * macs as f64 / (panel_ms / 1000.0) / 1e9
    );
    println!(
        "  the TRAILING update does ~n^3/3 = {:.3e} MACs in {:.4} ms  ->  {:.2} GFLOP/s",
        f64::from(n as u32).powi(3) / 3.0,
        ms(trailing),
        2.0 * f64::from(n as u32).powi(3) / 3.0 / (ms(trailing) / 1000.0) / 1e9
    );
    println!(
        "\nINSTRUMENT COST: the panel reads {panel_ms:.4} ms with the census ON and \
         {panel_uninstrumented:.4} ms with it OFF ({:+.1}%). The sub-phase shares are shares of \
         the INSTRUMENTED panel.",
        100.0 * (panel_ms - panel_uninstrumented) / panel_uninstrumented
    );
    println!(
        "\nREADING: this names the next lever's TARGET, it does not propose one. Compare the \
         trailing share against ledger 291a's reopen predicate — a lane only justifies reopening \
         the 2-D arm if its QUALIFYING-call TIME share dominates, and call share is not time share."
    );
}

//! `frankentorch-stale-tuning-constants-lzku6` — where does `inv` actually spend?
//!
//! # This run exists to correct my own claim
//!
//! Lane 2 reported a "residual" of ~45-55% for `inv`, computed as `wall - (getri_fwd + getri_back)`
//! and described as identity setup, allocation and the final permutation. **That was wrong in a way
//! worth recording**: `inv_tensor_contiguous_f64` runs `lu_factor_contiguous_f64` (getrf) BEFORE
//! `lu_inverse_from_factor_f64` (getri), so that subtraction had the ENTIRE LU FACTORISATION inside
//! it — a known, already-counted O(n^3) phase, hidden inside a word that implied glue.
//!
//! A subtraction is only a phase if you know what is inside it. Ledger 277a says the same thing
//! about a 6 ms frame invented the same way; this is that error committed by me, one lane later.
//!
//! # The decomposition, with closure
//!
//! getrf's three stages come from `lu_stage_take_ns`, getri's four from `lu_inverse_half_take_ns`
//! and `lu_inverse_extra_take_ns`. All seven are compared against the wall time of the same call,
//! and whatever is left is named RESIDUAL and SHOWN — not assumed away, and not called glue until
//! something has been subtracted that is actually known.
//!
//!     getrf   panel + solve + trailing
//!     getri   setup (Z=I + scratch) + forward + backward + permutation (+ n*n output alloc)
//!
//! Counters are drained PER REP and quoted against that rep's own wall, because they accumulate:
//! a min wall beside counters summed over reps attributes one rep's phases to another's total.
//!
//!   cargo run --release -p frankentorch-kernel-cpu --example inv_phase_split -- [n] [reps]

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

    let a: Vec<f64> = (0..n * n)
        .map(|idx| {
            let (i, j) = (idx / n, idx % n);
            let v = (((i * 7 + j * 13) % 101) as f64 - 50.0) / 25.0;
            if i == j { v + n as f64 } else { v }
        })
        .collect();
    let meta = TensorMeta::from_shape(vec![n, n], DType::F64, Device::Cpu);

    for _ in 0..2 {
        let _ = ft_kernel_cpu::inv_tensor_contiguous_f64(&a, &meta).expect("warm");
    }

    let mut best = (f64::INFINITY, [0u64; 7]);
    for _ in 0..reps {
        let _ = ft_kernel_cpu::lu_stage_take_ns();
        let _ = ft_kernel_cpu::lu_inverse_half_take_ns();
        let _ = ft_kernel_cpu::lu_inverse_extra_take_ns();
        let started = std::time::Instant::now();
        let v = ft_kernel_cpu::inv_tensor_contiguous_f64(&a, &meta).expect("inv");
        let wall = started.elapsed().as_secs_f64() * 1_000.0;
        std::hint::black_box(&v);
        let (gp, gs, gt) = ft_kernel_cpu::lu_stage_take_ns();
        let (fwd, back) = ft_kernel_cpu::lu_inverse_half_take_ns();
        let (setup, perm) = ft_kernel_cpu::lu_inverse_extra_take_ns();
        if wall < best.0 {
            best = (wall, [gp, gs, gt, setup, fwd, back, perm]);
        }
    }

    let (wall, p) = best;
    let ms = |v: u64| v as f64 / 1e6;
    let names = [
        "getrf panel",
        "getrf solve",
        "getrf trailing",
        "getri setup (Z=I)",
        "getri forward",
        "getri backward",
        "getri permutation",
    ];
    let accounted: f64 = p.iter().map(|&v| ms(v)).sum();

    println!("\ninv n={n}, min-wall rep of {reps}");
    println!("  {:<22} {:>9} {:>8}", "phase", "ms", "% call");
    for (name, &v) in names.iter().zip(p.iter()) {
        println!("  {name:<22} {:>9.4} {:>7.1}%", ms(v), 100.0 * ms(v) / wall);
    }
    println!(
        "  {:<22} {:>9.4} {:>7.1}%   <- residual, NOT a measured phase",
        "residual",
        wall - accounted,
        100.0 * (wall - accounted) / wall
    );
    println!("  {:<22} {:>9.4} {:>7.1}%", "TOTAL (wall)", wall, 100.0);
    println!(
        "\nCLOSURE: seven counted phases {accounted:.4} ms of {wall:.4} ms = {:.1}%.",
        100.0 * accounted / wall
    );
    let getrf: f64 = p[..3].iter().map(|&v| ms(v)).sum();
    let getri: f64 = p[3..].iter().map(|&v| ms(v)).sum();
    println!(
        "  getrf {getrf:.4} ms ({:.1}%)   getri {getri:.4} ms ({:.1}%)",
        100.0 * getrf / wall,
        100.0 * getri / wall
    );
    println!(
        "\nCORRECTION THIS RUN EXISTS FOR: lane 2 called `wall - (fwd + back)` a residual and \
         implied glue. That subtraction contained getrf, printed above as its own three stages. \
         A subtraction is only a phase if you know what is inside it."
    );
}

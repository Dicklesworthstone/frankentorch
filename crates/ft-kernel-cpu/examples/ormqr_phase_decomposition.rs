//! `frankentorch-g0wpj` — where ORMQR's time actually goes, on the kernel, at three sizes.
//!
//! # Why this exists as a kernel example
//!
//! The only in-tree caller of `ormqr_stage_take_ns` is the h2h harness, which needs a live torch
//! arm and therefore a measurement window that includes a peer-free host. The DECOMPOSITION does
//! not need torch at all — it is FT-vs-FT phase attribution — so it should not be gated behind the
//! hardest instrument in the repo. `feedback_attribute_the_lane_not_the_kernel` cuts the other way
//! for a LANE claim, but this is deliberately not one: it is the mechanism step that decides which
//! lever is worth building.
//!
//! # Reading the profiler honestly
//!
//! The stage counters became trustworthy only at ledger 293e: they are process-global, and before
//! that a concurrent test's ORMQR call recorded into them. Profiling is now owned by the thread
//! that enables it, so a single-threaded driver like this one gets its own work and nobody else's.
//! Before that fix these numbers would have been unfalsifiable.
//!
//! `total` is measured around the whole call, and the phases are measured inside it, so
//! `total - sum(phases)` is the UNATTRIBUTED remainder — allocation, the `ordered` panel
//! collection, loop overhead. Printing it is the point: ledger 292g records a subtraction that was
//! called a residual while containing an entire LU factorisation, and a decomposition whose parts
//! do not add up is a decomposition that has not found the op yet.
//!
//! # What the shares are for
//!
//! torch:4's decomposition (bead g0wpj, 03:57) put the direct subtract at 9% (n=512) and 26-42%
//! (n=1024). That pass materialises `upd = V*(T V^T C)` and then walks it subtracting into C. The
//! fused `dgemm_sub_into` (alpha=-1, beta=1 into strided C) deletes both, and the same swap is
//! ALREADY SHIPPED in geqrf's trailing update where it won 1.082x. This file measures the share
//! that lever would be aiming at, on this host, before the lever is written.
//!
//!   cargo run --release -p frankentorch-kernel-cpu --example ormqr_phase_decomposition -- [reps]

fn median(v: &mut Vec<f64>) -> f64 {
    v.sort_by(|a, b| a.partial_cmp(b).expect("no NaN"));
    let n = v.len();
    if n == 0 {
        return f64::NAN;
    }
    if n % 2 == 1 { v[n / 2] } else { f64::midpoint(v[n / 2 - 1], v[n / 2]) }
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
        "PROFILER OWNERSHIP: the ORMQR stage counters are process-global and became thread-owned \
         at ledger 293e; this driver enables profiling on its own thread, so the shares below are \
         its own work. Before that fix a concurrent caller could contribute to them."
    );
    println!(
        "TARGET UNDER TEST: `subtract` is the pass that materialises upd = V*(T V^T C) and then \
         walks it into C. `dgemm_sub_into` deletes both the buffer and the pass, and won 1.082x \
         doing exactly that in geqrf's trailing update."
    );

    for n in [256usize, 512, 1024] {
        let a: Vec<f64> = (0..n * n)
            .map(|idx| {
                let (i, j) = (idx / n, idx % n);
                (((i * 31 + j * 17) % 97) as f64 - 48.0) / 24.0
            })
            .collect();
        let (packed, tau) = ft_kernel_cpu::geqrf_blocked_f64(&a, n, n);
        let c0: Vec<f64> = (0..n * n).map(|i| ((i as f64) * 0.019).cos()).collect();

        // Warm, then profile. Profiling is enabled around the whole measured set so the enable
        // itself is never inside a timed region.
        for _ in 0..2 {
            let mut c = c0.clone();
            ft_kernel_cpu::ormqr_blocked_f64(&packed, &tau, n, n, n, &mut c, n, n, true, false);
        }

        let previous = ft_kernel_cpu::set_ormqr_stage_profile_enabled(true);
        let mut totals: Vec<f64> = Vec::with_capacity(reps);
        let mut acc = [0.0f64; 7];
        for _ in 0..reps {
            let mut c = c0.clone();
            let _ = ft_kernel_cpu::ormqr_stage_take_ns();
            ft_kernel_cpu::ormqr_blocked_f64(&packed, &tau, n, n, n, &mut c, n, n, true, false);
            let (panel, transpose, workspace, vt_c, t_w, v_w, subtract, total) =
                ft_kernel_cpu::ormqr_stage_take_ns();
            std::hint::black_box(&c);
            let ms = |v: u64| v as f64 / 1e6;
            totals.push(ms(total));
            for (slot, value) in acc
                .iter_mut()
                .zip([panel, transpose, workspace, vt_c, t_w, v_w, subtract])
            {
                *slot += ms(value);
            }
        }
        ft_kernel_cpu::set_ormqr_stage_profile_enabled(previous);

        let total_med = median(&mut totals);
        let mean_total: f64 = acc.iter().sum::<f64>() / reps as f64;
        let names = [
            "panel build",
            "T transpose",
            "workspace",
            "V^T C",
            "T W",
            "V W",
            "subtract",
        ];
        println!("\nn={n}  left apply, non-transpose  —  median total {total_med:.4} ms");
        println!("  {:>12} {:>10} {:>8}", "phase", "ms/call", "share");
        for (name, value) in names.iter().zip(acc.iter()) {
            let per_call = value / reps as f64;
            println!("  {name:>12} {per_call:>10.4} {:>7.1}%", 100.0 * per_call / total_med);
        }
        let unattributed = total_med - mean_total;
        println!(
            "  {:>12} {unattributed:>10.4} {:>7.1}%   <- allocation, panel collection, loop \
             overhead. A decomposition whose parts do not add up has not found the op yet (292g).",
            "UNATTRIBUTED",
            100.0 * unattributed / total_med
        );
    }
    println!(
        "\nREADING: `subtract` plus the part of UNATTRIBUTED that is the per-panel `upd` allocation \
         is what a fused dgemm_sub_into removes; `V W` is the GEMM it fuses INTO and stays. If \
         subtract is small here, the lever is not worth building and this file has done its job."
    );
}

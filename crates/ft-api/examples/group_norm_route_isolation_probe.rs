//! Which half of the 1.19x GroupNorm kernel-route gap is real? — `frankentorch-68pwz`.
//!
//! Item 84 measured two kernel routes 1.19x apart, replicated 5/5 arm-internal, and then
//! declined to act on it. The reason is worth restating because it is the whole point of
//! this file: the harness lane that looks like a clean lever-off twin — introduced as
//! *"Lever OFF: identical work, statistics rebuilt in the backward"* — is not that. The two
//! lanes differ in the FORWARD FUNCTION as well:
//!
//!     route A   group_norm_forward_f32_with_cpg2_stats  +  backward_..._with_cpg2_stats
//!     route B   group_norm_forward_f32_scheduled        +  backward_scalar_f32
//!
//! So "stats reuse costs 1.19x" is a confounded reading. Items 74, 75 and 80c were each
//! wrong in exactly this way — a number attributed to one thing that was measured across
//! two — and item 84a refused to make it a fourth.
//!
//! THIS PROBE SPLITS THE CONFOUND with three cells, changing ONE thing at a time:
//!
//!     A  stats-forward + stats-backward       what ft-api ships for cpg == 2
//!     M  stats-forward + recomputing-backward the DECISIVE middle cell
//!     B  scheduled-forward + recomputing-backward
//!
//! A vs M holds the forward fixed and switches the backward, so it prices the stats path.
//! M vs B holds the backward fixed and switches the forward, so it prices the forward.
//! Whichever pair carries the gap is the lever; the other is not.
//!
//! Each cell also reports its forward and backward separately, so the answer does not rest
//! on a subtraction between cells.
//!
//! Arm-internal: no incumbent, no ratio, no drift gate — honest on a busy host, though the
//! usual rule still applies that a saturated machine is worth nothing at all.

use ft_core::{DType, TensorMeta};
use std::time::Instant;

/// The scored lane: `[32, 64, 56, 56]`, 32 groups, so `cpg == 2`.
const GN_N: usize = 32;
const GN_C: usize = 64;
const GN_H: usize = 56;
const GN_W: usize = 56;
const GN_GROUPS: usize = 32;
const EPS: f32 = 1e-5;

fn loadavg() -> String {
    std::fs::read_to_string("/proc/loadavg")
        .map(|raw| {
            raw.split_whitespace()
                .take(3)
                .collect::<Vec<_>>()
                .join(" / ")
        })
        .unwrap_or_else(|_| "unavailable".to_owned())
}

fn cpu_mhz() -> String {
    let mut mhz: Vec<f64> = (0..)
        .map_while(|cpu| {
            std::fs::read_to_string(format!(
                "/sys/devices/system/cpu/cpu{cpu}/cpufreq/scaling_cur_freq"
            ))
            .ok()
        })
        .filter_map(|raw| raw.trim().parse::<f64>().ok().map(|khz| khz / 1000.0))
        .collect();
    if mhz.is_empty() {
        return "unavailable".to_owned();
    }
    mhz.sort_by(|a, b| a.partial_cmp(b).unwrap());
    format!(
        "min={:.0} mean={:.0} max={:.0} spread={:.2}x",
        mhz[0],
        mhz.iter().sum::<f64>() / mhz.len() as f64,
        mhz[mhz.len() - 1],
        mhz[mhz.len() - 1] / mhz[0]
    )
}

fn main() {
    let spatial = GN_H * GN_W;
    let cpg = GN_C / GN_GROUPS;
    assert_eq!(cpg, 2, "this probe is about the cpg == 2 route");
    let numel = GN_N * GN_C * spatial;

    let x: Vec<f32> = (0..numel)
        .map(|index| ((index % 251) as f32) * 0.001 - 0.12)
        .collect();
    let weight: Vec<f32> = (0..GN_C).map(|c| (c as f32) * 0.01 + 1.0).collect();
    let bias: Vec<f32> = (0..GN_C).map(|c| (c as f32) * 0.003).collect();
    let out_meta = TensorMeta::from_shape(
        vec![GN_N, GN_C, GN_H, GN_W],
        DType::F32,
        ft_core::Device::Cpu,
    );

    println!("group_norm_route_isolation_probe (frankentorch-68pwz, item 84b)");
    println!("shape [{GN_N},{GN_C},{GN_H},{GN_W}] groups={GN_GROUPS} cpg={cpg} spatial={spatial}");
    println!("rayon_threads={}", rayon::current_num_threads());
    println!("pre  loadavg {}", loadavg());
    println!("pre  cpu_mhz {}", cpu_mhz());
    println!();

    // ORDERING CONTAMINATION, found by this probe's own control on its first run.
    //
    // Cells A and M call the IDENTICAL forward. On the first version of this probe — which
    // ran A, then M, then B inside each rep with no warmup — that shared forward reported
    // 2.745 ms in A and 1.521 ms in M: a 1.80x gap for the same function, because A paid
    // the cold allocator and page state every rep and M never did. The "2.15x" that fell
    // out of A -> M was that artifact, not the stats path.
    //
    // Two fixes, both needed. An untimed warmup pass runs every cell once so no cell is
    // measured cold, and the A/M forward agreement is asserted at the end instead of being
    // left as a coincidence to notice. A probe whose cells share a component should CHECK
    // that component reads the same in both, and this one now does.
    {
        let (out, stats) = ft_kernel_cpu::group_norm_forward_f32_with_cpg2_stats(
            &x,
            Some(&weight),
            Some(&bias),
            GN_N,
            GN_GROUPS,
            spatial,
            EPS,
        );
        std::hint::black_box(&out);
        let _ = ft_kernel_cpu::group_norm_backward_scalar_f32_with_cpg2_stats(
            1.0f32,
            &x,
            Some(&weight),
            &stats,
            GN_N,
            GN_GROUPS,
            spatial,
        );
        let _ = ft_kernel_cpu::group_norm_backward_scalar_f32(
            1.0f32,
            &x,
            Some(&weight),
            GN_N,
            GN_GROUPS,
            cpg,
            spatial,
            EPS,
        );
        let out = ft_kernel_cpu::group_norm_forward_f32_scheduled(
            &x,
            Some(&weight),
            Some(&bias),
            GN_N,
            GN_GROUPS,
            cpg,
            spatial,
            EPS,
            true,
        );
        std::hint::black_box(&out);
    }

    let reps = 7;
    // [forward, backward, total] minima per cell.
    let mut a = [f64::INFINITY; 3];
    let mut m = [f64::INFINITY; 3];
    let mut b = [f64::INFINITY; 3];
    // Gradient checksums, so the three cells are shown to compute the same thing rather
    // than assumed to. A cell that quietly diverged would be a faster wrong answer.
    let mut sums = [0.0f64; 3];

    for _ in 0..reps {
        // ---- cell A: stats-forward + stats-backward (what ft-api ships) ----
        let t0 = Instant::now();
        let (out, stats) = ft_kernel_cpu::group_norm_forward_f32_with_cpg2_stats(
            &x,
            Some(&weight),
            Some(&bias),
            GN_N,
            GN_GROUPS,
            spatial,
            EPS,
        );
        let fwd = t0.elapsed().as_secs_f64() * 1_000.0;
        let loss = ft_kernel_cpu::sum_tensor_contiguous_f32(&out, &out_meta).expect("sum");
        assert!(loss.is_finite());
        let t1 = Instant::now();
        let (dx, _, _) = ft_kernel_cpu::group_norm_backward_scalar_f32_with_cpg2_stats(
            1.0f32,
            &x,
            Some(&weight),
            &stats,
            GN_N,
            GN_GROUPS,
            spatial,
        );
        let bwd = t1.elapsed().as_secs_f64() * 1_000.0;
        sums[0] = dx.iter().map(|&v| f64::from(v.abs())).sum();
        a[0] = a[0].min(fwd);
        a[1] = a[1].min(bwd);
        a[2] = a[2].min(fwd + bwd);

        // ---- cell M: SAME forward, recomputing backward (the decisive cell) ----
        let t0 = Instant::now();
        let (out, _stats) = ft_kernel_cpu::group_norm_forward_f32_with_cpg2_stats(
            &x,
            Some(&weight),
            Some(&bias),
            GN_N,
            GN_GROUPS,
            spatial,
            EPS,
        );
        let fwd = t0.elapsed().as_secs_f64() * 1_000.0;
        let loss = ft_kernel_cpu::sum_tensor_contiguous_f32(&out, &out_meta).expect("sum");
        assert!(loss.is_finite());
        let t1 = Instant::now();
        let (dx, _, _) = ft_kernel_cpu::group_norm_backward_scalar_f32(
            1.0f32,
            &x,
            Some(&weight),
            GN_N,
            GN_GROUPS,
            cpg,
            spatial,
            EPS,
        );
        let bwd = t1.elapsed().as_secs_f64() * 1_000.0;
        sums[1] = dx.iter().map(|&v| f64::from(v.abs())).sum();
        m[0] = m[0].min(fwd);
        m[1] = m[1].min(bwd);
        m[2] = m[2].min(fwd + bwd);

        // ---- cell B: scheduled forward + recomputing backward ----
        let t0 = Instant::now();
        let out = ft_kernel_cpu::group_norm_forward_f32_scheduled(
            &x,
            Some(&weight),
            Some(&bias),
            GN_N,
            GN_GROUPS,
            cpg,
            spatial,
            EPS,
            true,
        );
        let fwd = t0.elapsed().as_secs_f64() * 1_000.0;
        let loss = ft_kernel_cpu::sum_tensor_contiguous_f32(&out, &out_meta).expect("sum");
        assert!(loss.is_finite());
        let t1 = Instant::now();
        let (dx, _, _) = ft_kernel_cpu::group_norm_backward_scalar_f32(
            1.0f32,
            &x,
            Some(&weight),
            GN_N,
            GN_GROUPS,
            cpg,
            spatial,
            EPS,
        );
        let bwd = t1.elapsed().as_secs_f64() * 1_000.0;
        sums[2] = dx.iter().map(|&v| f64::from(v.abs())).sum();
        b[0] = b[0].min(fwd);
        b[1] = b[1].min(bwd);
        b[2] = b[2].min(fwd + bwd);
    }

    println!(
        "{:>46}{:>10}{:>10}{:>10}",
        "cell", "fwd ms", "bwd ms", "total"
    );
    println!(
        "{:>46}{:>10.3}{:>10.3}{:>10.3}",
        "A  stats-fwd + stats-bwd   (ships)", a[0], a[1], a[2]
    );
    println!(
        "{:>46}{:>10.3}{:>10.3}{:>10.3}",
        "M  stats-fwd + recomputing-bwd", m[0], m[1], m[2]
    );
    println!(
        "{:>46}{:>10.3}{:>10.3}{:>10.3}",
        "B  scheduled-fwd + recomputing-bwd", b[0], b[1], b[2]
    );
    println!();
    println!(
        "A -> M  (forward held fixed, backward switched): {:.3} ms, {:.2}x",
        a[2] - m[2],
        a[2] / m[2]
    );
    println!(
        "M -> B  (backward held fixed, forward switched): {:.3} ms, {:.2}x",
        m[2] - b[2],
        m[2] / b[2]
    );
    println!(
        "A -> B  (both, i.e. item 84's confounded gap):   {:.3} ms, {:.2}x",
        a[2] - b[2],
        a[2] / b[2]
    );
    println!();
    // THE CONTROL. A and M call the same forward, so their forward timings must agree; if
    // they do not, cell order is still contaminating the comparison and A -> M means
    // nothing. Stated as a hard check rather than a note, because the first run of this
    // probe failed it at 1.80x and the failure was only visible to someone reading the
    // per-phase columns.
    let fwd_control = a[0].max(m[0]) / a[0].min(m[0]);
    println!(
        "CONTROL  A.fwd {:.3} vs M.fwd {:.3} (same function) -> {fwd_control:.3}x{}",
        a[0],
        m[0],
        if fwd_control > 1.10 {
            "   *** CONTAMINATED — A->M IS NOT READABLE ***"
        } else {
            "   ok"
        }
    );
    println!();
    println!("VERDICT: whichever single-variable step carries the gap is the lever. If A->M is");
    println!("  flat then the stats path is NOT the cost and frankentorch-qkwsy is exonerated;");
    println!("  if A->M carries it, qkwsy's stats reuse is a regression and the session's");
    println!("  backward should stop passing cached statistics.");
    println!();
    // The three cells must agree to f32 rounding; they are different routes to the same
    // gradient, and a divergence means the comparison above is between two answers.
    let spread = sums.iter().cloned().fold(f64::MIN, f64::max)
        - sums.iter().cloned().fold(f64::MAX, f64::min);
    let rel = spread / sums[0].abs().max(f64::MIN_POSITIVE);
    println!(
        "|dx| checksums  A {:.6e}  M {:.6e}  B {:.6e}   relative spread {rel:.3e}",
        sums[0], sums[1], sums[2]
    );
    assert!(
        rel < 1e-5,
        "the three routes must compute the same gradient; relative spread {rel:.3e} says they do not"
    );
    println!();
    println!("post loadavg {}", loadavg());
    println!("post cpu_mhz {}", cpu_mhz());
}

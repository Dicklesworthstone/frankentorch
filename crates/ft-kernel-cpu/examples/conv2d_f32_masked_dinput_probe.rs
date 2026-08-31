//! Where does the f32 MASKED backward's extra time go? — `frankentorch-hi9r6`.
//!
//! WHY THIS PROBE. NEGATIVE_EVIDENCE item 264 left the f32 gap in one place: the summed route is
//! at parity with PyTorch or better (`conv2d_f32` read 1.01-1.02x FASTER, both gates PASS) while
//! the masked route is 2.34-2.39x SLOWER. At the lane's shape those two differ only in WHICH
//! backward runs — the 3x3 stride-1 all-ones adjoint versus the generic fused-mask entry — and
//! the lane-level difference is ~31 ms.
//!
//! That 31 ms is a LANE difference, so it still contains the session, the mask multiply and the
//! sum. This times the two KERNEL entries directly, at the lane's exact shape, so the kernel-level
//! gap is a number rather than a subtraction. Item 141's lesson is the reason: a residual is not a
//! measurement of whatever you name it, and this bead has already had three attributions
//! overturned by timing the thing directly instead of subtracting around it.
//!
//! It also splits the fused entry by `output_mask`, which is free and decides the next lever's
//! target: `[true,false,false]` is dinput alone, `[false,true,false]` is dweight alone. If dinput
//! dominates, the lever is the direct dinput kernel; if the two are comparable, the 31 ms is not
//! one phase and a kernel rewrite would be aimed at the wrong half.
//!
//! ARM-INTERNAL: no incumbent, no ratio, no drift gate. It says which phase to attack and is not
//! a standing. Needs no PyTorch, so it runs on any rch worker.
//!
//! Everything goes to STDERR so a remote runner returns it.

use std::time::Instant;

// The f32 lane's shape, copied from `C2F32_N` and the `C2_*` constants so the probe and the lane
// describe one workload.
const BATCH: usize = 160;
const IN_CH: usize = 32;
const OUT_CH: usize = 32;
const H: usize = 32;
const W: usize = 32;
const K: usize = 3;
const PH: usize = H + 2;
const PW: usize = W + 2;

fn main() {
    let reps: usize = std::env::var("FT_REPS")
        .ok()
        .and_then(|t| t.trim().parse().ok())
        .unwrap_or(7);

    let padded: Vec<f32> = (0..BATCH * IN_CH * PH * PW)
        .map(|i| ((i % 37) as f32) * 0.013 - 0.21)
        .collect();
    let weight: Vec<f32> = (0..OUT_CH * IN_CH * K * K)
        .map(|i| ((i % 11) as f32) * 0.0625 - 0.3125)
        .collect();
    let ones: Vec<f32> = vec![1.0; BATCH * OUT_CH * H * W];
    let mask: Vec<f32> = (0..BATCH * OUT_CH * H * W)
        .map(|i| ((i % 23) as f32) * 0.019 - 0.19)
        .collect();

    eprintln!(
        "F32_DINPUT b={BATCH} ci={IN_CH} co={OUT_CH} {H}x{W} k{K}x{K} reps={reps} threads={} \
         (all arms inside each rep, adjacent; min after discarding the first)",
        rayon::current_num_threads()
    );

    // THE SEPARATING ARM — `frankentorch-hi9r6`. The 13.8x between the adjoint and the generic
    // entry has two candidate sources and the earlier probe could not tell them apart: the 3x3
    // stride-1 STRUCTURAL specialisation, and `dout == 1` COLLAPSING the work (with ones, every
    // dweight row is the same row and dinput is a fixed stencil). Only the first would transfer
    // to a general-dout 3x3 kernel, so building one is justified only if the first dominates.
    //
    // This arm runs the GENERIC fused entry with an ALL-ONES mask. Same code path, same shape,
    // same 3x3 stride-1 structure — the only thing that changes is whether the values happen to
    // be 1.0. If it matches the non-uniform-mask arm, the generic path extracts NOTHING from
    // ones, which means the adjoint's 4.121 ms is it doing LESS WORK rather than the same work
    // faster, and the 13.8x does not transfer.
    let ones_mask: Vec<f32> = vec![1.0; BATCH * OUT_CH * H * W];
    let mut fused_ones_mask = f64::INFINITY;
    // THE ROUTE ARM — `frankentorch-hi9r6`. Item 265 left the dinput half at ~28 ms and the
    // back-of-envelope says it is neither compute- nor bandwidth-bound: ~3.02 GFLOP in ~28 ms is
    // ~108 GFLOP/s against an AVX2 f32 ceiling several times that, while ~45 MB of traffic in the
    // same 28 ms is ~1.6 GB/s against a DRAM ceiling ~30x that. A phase that is at 12% of compute
    // AND 3% of bandwidth is bound by neither, which points at the SCATTER's access pattern.
    //
    // This arm forces the LEGACY panel + col2im route (`conv2d_dinput_panel_legacy`) against the
    // shipped direct one, dinput only, both inside each rep. The two share the col2im-style
    // scatter and differ in whether a `flat x patch_width` panel is materialised first. If they
    // land close, the panel is not the cost and the scatter is — which would say the remaining
    // dinput lever is the scatter's memory order, not another panel-elimination.
    let mut legacy_dinput = f64::INFINITY;

    let mut adjoint = f64::INFINITY;
    let mut fused_both = f64::INFINITY;
    let mut fused_dinput = f64::INFINITY;
    let mut fused_dweight = f64::INFINITY;

    for rep in 0..reps {
        // The specialised all-ones 3x3 stride-1 adjoint — what the SUMMED lane reaches.
        let start = Instant::now();
        let a = ft_kernel_cpu::conv2d_backward_f32(
            &ones, &padded, &weight, BATCH, IN_CH, PH, PW, K, K, H, W, 1, 1, OUT_CH, false,
        );
        let t_adjoint = start.elapsed().as_secs_f64() * 1e3;
        std::hint::black_box(&a);

        // The generic fused-mask entry — what the MASKED lane reaches. Same dout shape, but a
        // non-uniform mask, so the all-ones route is not taken.
        let start = Instant::now();
        let b = ft_kernel_cpu::conv2d_backward_mask_fused_f32(
            &ones, &mask, &padded, &weight, BATCH, IN_CH, PH, PW, K, K, H, W, 1, 1, OUT_CH,
            [true, true, false],
        );
        let t_both = start.elapsed().as_secs_f64() * 1e3;
        std::hint::black_box(&b);

        let start = Instant::now();
        let c = ft_kernel_cpu::conv2d_backward_mask_fused_f32(
            &ones, &mask, &padded, &weight, BATCH, IN_CH, PH, PW, K, K, H, W, 1, 1, OUT_CH,
            [true, false, false],
        );
        let t_din = start.elapsed().as_secs_f64() * 1e3;
        std::hint::black_box(&c);

        let start = Instant::now();
        let d = ft_kernel_cpu::conv2d_backward_mask_fused_f32(
            &ones, &mask, &padded, &weight, BATCH, IN_CH, PH, PW, K, K, H, W, 1, 1, OUT_CH,
            [false, true, false],
        );
        let t_dw = start.elapsed().as_secs_f64() * 1e3;
        std::hint::black_box(&d);

        let previous_legacy = ft_kernel_cpu::set_conv2d_dinput_panel_legacy(true);
        let start = Instant::now();
        let f = ft_kernel_cpu::conv2d_backward_mask_fused_f32(
            &ones, &mask, &padded, &weight, BATCH, IN_CH, PH, PW, K, K, H, W, 1, 1, OUT_CH,
            [true, false, false],
        );
        let t_legacy = start.elapsed().as_secs_f64() * 1e3;
        std::hint::black_box(&f);
        ft_kernel_cpu::set_conv2d_dinput_panel_legacy(previous_legacy);

        let start = Instant::now();
        let e = ft_kernel_cpu::conv2d_backward_mask_fused_f32(
            &ones, &ones_mask, &padded, &weight, BATCH, IN_CH, PH, PW, K, K, H, W, 1, 1, OUT_CH,
            [true, true, false],
        );
        let t_ones_mask = start.elapsed().as_secs_f64() * 1e3;
        std::hint::black_box(&e);

        // Discard the first rep on every arm: allocator and page-fault costs the rest do not pay.
        if rep > 0 {
            adjoint = adjoint.min(t_adjoint);
            fused_both = fused_both.min(t_both);
            fused_dinput = fused_dinput.min(t_din);
            fused_dweight = fused_dweight.min(t_dw);
            fused_ones_mask = fused_ones_mask.min(t_ones_mask);
            legacy_dinput = legacy_dinput.min(t_legacy);
        }
    }

    eprintln!("F32_DINPUT all-ones adjoint (summed route)   {adjoint:8.3} ms");
    eprintln!("F32_DINPUT fused mask, dinput+dweight        {fused_both:8.3} ms");
    eprintln!("F32_DINPUT fused mask, dinput ONLY           {fused_dinput:8.3} ms");
    eprintln!("F32_DINPUT fused mask, dweight ONLY          {fused_dweight:8.3} ms");
    eprintln!("F32_DINPUT fused mask, ALL-ONES mask (separator) {fused_ones_mask:8.3} ms");
    eprintln!("F32_DINPUT dinput ONLY, LEGACY panel+col2im route  {legacy_dinput:8.3} ms");
    eprintln!(
        "F32_DINPUT ROUTE: legacy / direct = {:.3}x. ~1.0 means materialising the panel is NOT \
         the dinput cost and the SCATTER is, so the remaining lever is memory order rather than \
         another panel elimination.",
        legacy_dinput / fused_dinput
    );
    eprintln!(
        "F32_DINPUT kernel-level gap (fused both - adjoint) {:8.3} ms; dinput is {:.0}% of the \
         fused entry",
        fused_both - adjoint,
        100.0 * fused_dinput / fused_both
    );
    eprintln!(
        "F32_DINPUT SEPARATOR: generic-with-ones / generic-with-mask = {:.3}x. ~1.0 means the \
         generic path extracts NOTHING from ones, so the adjoint's speed is LESS WORK and the \
         13.8x does NOT transfer to a general-dout 3x3 kernel.",
        fused_ones_mask / fused_both
    );
}

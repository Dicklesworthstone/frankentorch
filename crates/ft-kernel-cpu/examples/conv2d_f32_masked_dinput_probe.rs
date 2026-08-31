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

use rayon::prelude::*;
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
    // ARGV, not env: `rch exec` does not forward the caller's environment, so an env-configured
    // run silently executes the DEFAULT at the worker's full width (ledger 273c). These arms are
    // thread-sensitive, so the width is an argument and the probe echoes it back.
    let argv: Vec<String> = std::env::args().skip(1).collect();
    let reps: usize = argv.first().and_then(|t| t.parse().ok()).unwrap_or(7);
    let threads: usize = argv.get(1).and_then(|t| t.parse().ok()).unwrap_or(0);
    if threads > 0 {
        rayon::ThreadPoolBuilder::new()
            .num_threads(threads)
            .build_global()
            .expect("rayon pool width");
    }
    eprintln!(
        "PROV host={} nproc={} rayon={} loadavg={}",
        std::fs::read_to_string("/etc/hostname").unwrap_or_default().trim(),
        std::thread::available_parallelism().map_or(0, std::num::NonZero::get),
        rayon::current_num_threads(),
        std::fs::read_to_string("/proc/loadavg").unwrap_or_default().trim(),
    );

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
    // THE FORWARD ARM — `frankentorch-hi9r6`. A top-3 frame summary for `conv2d_f32_masked` is
    // only sound if every frame comes from the SAME host: the lane figures are thinkstation1 and
    // every kernel figure here is hetzner2, and `feedback_measurement_host_identity` forbids
    // splicing those into one decomposition. Timing the forward here puts all three frames on one
    // machine, so the shares are a decomposition rather than a cross-host estimate.
    let mut forward = f64::INFINITY;
    // THE SCATTER ARM — `frankentorch-hi9r6`, item 267's "isolate a phase INSIDE the kernel".
    //
    // `conv2d_backward_dinput_blocked_rows_f32` has exactly two frames: a per-row-block
    // `sgemm`, and a fully SCALAR accumulate `dpb[irow + kc] += block[...]` nested over
    // in_ch x kh x kw. At this shape that scatter is 160 * 1024 * 288 = 47.2M read-modify-writes.
    // `gemm` is private to the crate so the GEMM cannot be timed from an example, but
    // `conv2d_col2im_f32` IS public and is structurally the same scatter over the same total
    // work — so timing it bounds the frame without instrumenting shipped code.
    //
    // READ IT AS AN UPPER BOUND, not as the blocked route's scatter cost: standalone col2im
    // streams a 189 MB panel from DRAM, while the blocked route scatters from a cache-resident
    // block. That asymmetry is the whole point of the blocking, so if col2im alone is COMPARABLE
    // to the entire direct route then the scatter is the dominant frame; if it is far larger,
    // the blocking is already doing the work and the GEMM is what remains.
    let dpanel: Vec<f32> = (0..BATCH * H * W * IN_CH * K * K)
        .map(|i| ((i % 19) as f32) * 0.007 - 0.06)
        .collect();
    let mut col2im = f64::INFINITY;
    // THE GATHER-DIRECTION ARM — `frankentorch-hi9r6`, item 268's promoted candidate probed the
    // cheap way BEFORE writing a kernel.
    //
    // Item 268 measured the scatter as 84-95% of the direct dinput route and latency-bound on
    // scattered read-modify-write, which promotes the GATHER INVERSION: compute each `dpadded`
    // element from the contributions reaching it, so every output is WRITTEN ONCE instead of
    // accumulated into 9 times. That is a real kernel, so its premise gets tested first.
    //
    // THE PREMISE, ISOLATED WITHOUT WRITING IT: `conv2d_im2col_f32` and `conv2d_col2im_f32` move
    // the SAME 47.2M elements between the SAME two buffers at the SAME stencil geometry, in
    // opposite directions. im2col writes each of its 47.2M outputs exactly once; col2im
    // accumulates into 5.92M outputs 9 times each. That is precisely the difference the gather
    // inversion is proposing to buy.
    //
    // WHY THIS AND NOT A MICROBENCHMARK: `feedback_insitu_over_standalone` records a standalone
    // ladder INVERTING in situ — a predicted 5.7x that measured as a 1.118x REGRESSION once
    // allocator warmth was real. Both arms here are SHIPPED kernels called through their public
    // entries in the same process as everything else in this probe, so there is no standalone
    // artefact to be fooled by.
    //
    // REFUTES THE CANDIDATE IF ~1.0: same volume, same geometry, and the write-once direction
    // buys nothing, so the inversion would not either.
    let mut im2col = f64::INFINITY;
    // THE dweight ROUTE ARM — `frankentorch-hi9r6`. Every probe since item 267 aimed at dinput,
    // so this prices the half nobody looked inside.
    //
    // WHICH HALF IS LARGER IS HOST-DEPENDENT AND IS NOT SETTLED. fixmydocuments read dweight
    // 27.3-35.2 against dinput 17.1-18.2 (dweight 60-65%); hetzner2 read 25.3-31.6 against
    // 27.8-28.5 (dweight 48-53%). I briefly wrote "dweight is the larger half" here on the
    // fixmydocuments numbers alone and it does not survive the second host. The defensible
    // statement is that the two halves are COMPARABLE and neither is negligible.
    //
    // It forces the streamed dweight OFF against the shipped ON, dweight-only, both inside each
    // rep. That does two things at once: it prices my own lever at the KERNEL level (the 1.74x
    // in item 263 is a LANE figure), and it says how much of the remaining 27-35 ms is the panel
    // the lever already removed versus arithmetic that no panel change can touch.
    let mut dweight_legacy = f64::INFINITY;

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

        let previous_stream = ft_kernel_cpu::set_conv2d_dweight_streamed(false);
        let start = Instant::now();
        let dwl = ft_kernel_cpu::conv2d_backward_mask_fused_f32(
            &ones, &mask, &padded, &weight, BATCH, IN_CH, PH, PW, K, K, H, W, 1, 1, OUT_CH,
            [false, true, false],
        );
        let t_dw_legacy = start.elapsed().as_secs_f64() * 1e3;
        std::hint::black_box(&dwl);
        ft_kernel_cpu::set_conv2d_dweight_streamed(previous_stream);

        let start = Instant::now();
        let ic = ft_kernel_cpu::conv2d_im2col_f32(
            &padded, BATCH, IN_CH, PH, PW, K, K, H, W, 1, 1,
        );
        let t_im2col = start.elapsed().as_secs_f64() * 1e3;
        std::hint::black_box(&ic);

        let start = Instant::now();
        let sc = ft_kernel_cpu::conv2d_col2im_f32(
            &dpanel, BATCH, IN_CH, PH, PW, K, K, H, W, 1, 1,
        );
        let t_col2im = start.elapsed().as_secs_f64() * 1e3;
        std::hint::black_box(&sc);

        let start = Instant::now();
        let fw = ft_kernel_cpu::conv2d_forward_f32(
            &padded, &weight, None, BATCH, IN_CH, PH, PW, K, K, H, W, 1, 1, OUT_CH,
        );
        let t_fw = start.elapsed().as_secs_f64() * 1e3;
        std::hint::black_box(&fw);

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
            forward = forward.min(t_fw);
            col2im = col2im.min(t_col2im);
            im2col = im2col.min(t_im2col);
            dweight_legacy = dweight_legacy.min(t_dw_legacy);
        }
    }

    // THE BUDGET SWEEP — `frankentorch-hi9r6`. `block_rows` is `BUDGET_BYTES / (patch_width*4)`
    // with a shipped 576 KiB, i.e. 512 rows here. That constant governs the scatter's resident
    // working set, and items 268/269 put the scatter at 77-95% of this route — so it governs the
    // frame carrying the f32 deficit, and it has never been swept. Swept BELOW the shipped value
    // as well as above, per `feedback_tuning_grid_missing_the_winner`.
    let mut budget_rows: Vec<(usize, f64)> = Vec::new();
    for &budget in &[128 * 1024usize, 256 * 1024, 576 * 1024, 1024 * 1024, 2048 * 1024, 4096 * 1024]
    {
        let previous = ft_kernel_cpu::set_conv2d_dinput_budget_bytes_f32(budget);
        let mut best = f64::INFINITY;
        for rep in 0..reps {
            let start = Instant::now();
            let r = ft_kernel_cpu::conv2d_backward_mask_fused_f32(
                &ones, &mask, &padded, &weight, BATCH, IN_CH, PH, PW, K, K, H, W, 1, 1, OUT_CH,
                [true, false, false],
            );
            let t = start.elapsed().as_secs_f64() * 1e3;
            std::hint::black_box(&r);
            if rep > 0 {
                best = best.min(t);
            }
        }
        ft_kernel_cpu::set_conv2d_dinput_budget_bytes_f32(previous);
        budget_rows.push((budget, best));
    }
    for (budget, ms) in &budget_rows {
        let shipped = if *budget == 576 * 1024 { "  <- SHIPPED" } else { "" };
        eprintln!(
            "F32_DINPUT BUDGET {:>5} KiB -> block_rows {:>5}  dinput {:8.3} ms{}",
            budget / 1024,
            budget / (IN_CH * K * K * 4),
            ms,
            shipped
        );
    }

    eprintln!("F32_DINPUT forward                             {forward:8.3} ms");
    eprintln!("F32_DINPUT col2im scatter alone (upper bound) {col2im:8.3} ms");
    eprintln!("F32_DINPUT im2col gather alone (same volume)  {im2col:8.3} ms");
    eprintln!("F32_DINPUT all-ones adjoint (summed route)   {adjoint:8.3} ms");
    eprintln!("F32_DINPUT fused mask, dinput+dweight        {fused_both:8.3} ms");
    eprintln!("F32_DINPUT fused mask, dinput ONLY           {fused_dinput:8.3} ms");
    eprintln!("F32_DINPUT fused mask, dweight ONLY          {fused_dweight:8.3} ms");
    eprintln!("F32_DINPUT dweight ONLY, LEGACY panel route   {dweight_legacy:8.3} ms");
    eprintln!("F32_DINPUT fused mask, ALL-ONES mask (separator) {fused_ones_mask:8.3} ms");
    eprintln!("F32_DINPUT dinput ONLY, LEGACY panel+col2im route  {legacy_dinput:8.3} ms");

    // ---------------------------------------------------------------------------------------
    // THE PANEL-FREE ARM — `frankentorch-t1gph`.
    //
    // Item 269 closed every identified lever on this frame and left ONE direction: "the remaining
    // distance to PyTorch is almost certainly that PyTorch does not materialise a panel at all."
    // That is a different backward, not a faster version of this one, and before anyone writes it
    // there are two things worth knowing, because the evidence already in the tree points BOTH
    // ways.
    //
    // AGAINST: this codebase already has a panel-free direct 3x3 convolution
    // (`conv3d_forward_direct_3x3s1_f64`), and item 68 measured it winning 1.134x at in_ch=8 and
    // LOSING 0.658x at in_ch=12, degrading to a 1.5-3.3x pessimization above. This lane runs at
    // in_ch=32 — four times past the measured crossover. Packed-GEMM microkernels beat stencil
    // loops once the channel depth gives them enough reuse, and that is not shape-specific
    // folklore, it is measured here.
    //
    // FOR: item 68 measured a FORWARD, where the panel is built once and consumed once. In THIS
    // backward the panel round trip is most of the cost — item 269b has the direct dinput route at
    // 16.983/17.117 ms with the col2im scatter alone at 13.086/13.266 ms, so ~77% of the frame is
    // spent writing 189 MB of panel and reading it back to accumulate into 23.7 MB. A panel-free
    // kernel deletes that entire round trip and writes each output ONCE. So it could lose badly on
    // arithmetic throughput and still win on traffic.
    //
    // Arithmetic alone cannot settle which effect dominates, so this measures it — WITHOUT
    // touching a shipping path, because going panel-free is also a PARITY decision: it accumulates
    // over `oc` in a different order than the GEMM does, which would break
    // `conv2d_dinput_direct_f32_matches_panel_col2im_bitwise` and the fused-vs-materialised bitwise
    // test. That is a policy call, and it should be made against a number rather than a hope.
    //
    // The kernel below is deliberately STRAIGHTFORWARD BUT NOT STUPID: per-channel weights are
    // hoisted into a 9 x out_ch block that stays in L1, and the innermost loop is a contiguous
    // length-`out_ch` dot product over `dout`, which vectorises. If a fair-but-unoptimised direct
    // kernel lands anywhere near the shipping route, the lever is alive and worth real blocking
    // work; if it is an order of magnitude off, item 68's crossover governs and this direction is
    // dead for the same reason conv3d's was.
    // ---------------------------------------------------------------------------------------
    // THE dout_flat TRANSPOSE ARMS — `frankentorch-t1gph`.
    //
    // The budget sweep above times `conv2d_backward_dinput_direct_f32` ALONE at 8.4-8.9 ms while
    // "fused mask, dinput ONLY" reads 27.026 ms for the same dinput work. That ~18 ms sits in the
    // fused entry's `dout_flat` construction, and a residual is not a measurement of whatever you
    // name it (item 141), so both forms are timed DIRECTLY here.
    //
    // What the shipping build does: `par_chunks_mut(out_ch)` hands out one 32-float row per task
    // and each row gathers its 32 sources from `(n*out_ch + oc)*patch_count + patch` — addresses
    // `patch_count * 4` = 4 KiB apart. Every one of the 5.24M reads therefore lands on its own
    // cache line, and the task granularity is 128 bytes. That is the same shape as the strided
    // column gather item 274 removed from `lu_solve`.
    //
    // The candidate tiles it: for a block of patches, loop `oc` OUTSIDE and `patch` INSIDE, so the
    // source read is contiguous along `patch` and the strided write lands inside an 8 KiB
    // L1-resident tile. Same products, same values, pure data movement — bit-exact by
    // construction.
    let patch_count_t = H * W;
    let mut dout_build_shipped = f64::INFINITY;
    let mut dout_build_tiled = f64::INFINITY;
    let mut tiled_matches = true;
    for rep in 0..reps {
        let start = Instant::now();
        let mut shipped = vec![0.0f32; BATCH * patch_count_t * OUT_CH];
        shipped
            .par_chunks_mut(OUT_CH)
            .enumerate()
            .for_each(|(row, destination)| {
                let n = row / patch_count_t;
                let patch = row % patch_count_t;
                for (out_channel, slot) in destination.iter_mut().enumerate() {
                    let source = (n * OUT_CH + out_channel) * patch_count_t + patch;
                    *slot = ones[source] * mask[source];
                }
            });
        let t_shipped = start.elapsed().as_secs_f64() * 1e3;

        const PBLK: usize = 64;
        let start = Instant::now();
        let mut tiled = vec![0.0f32; BATCH * patch_count_t * OUT_CH];
        tiled
            .par_chunks_mut(patch_count_t * OUT_CH)
            .enumerate()
            .for_each(|(n, plane)| {
                let mut p0 = 0;
                while p0 < patch_count_t {
                    let p1 = (p0 + PBLK).min(patch_count_t);
                    for oc in 0..OUT_CH {
                        let base = (n * OUT_CH + oc) * patch_count_t;
                        for patch in p0..p1 {
                            plane[patch * OUT_CH + oc] =
                                ones[base + patch] * mask[base + patch];
                        }
                    }
                    p0 = p1;
                }
            });
        let t_tiled = start.elapsed().as_secs_f64() * 1e3;

        if rep == 0 {
            tiled_matches = shipped
                .iter()
                .zip(&tiled)
                .all(|(a, b)| a.to_bits() == b.to_bits());
        }
        if rep > 0 {
            dout_build_shipped = dout_build_shipped.min(t_shipped);
            dout_build_tiled = dout_build_tiled.min(t_tiled);
        }
        std::hint::black_box(&shipped);
        std::hint::black_box(&tiled);
    }
    eprintln!("F32_DINPUT dout_flat build, SHIPPED row-gather    {dout_build_shipped:8.3} ms");
    eprintln!(
        "F32_DINPUT dout_flat build, TILED transpose       {dout_build_tiled:8.3} ms  ({:.4}x, bitwise match {tiled_matches})",
        dout_build_shipped / dout_build_tiled
    );

    // ---------------------------------------------------------------------------------------
    // THE dweight M-BLOCK SWEEP -- `frankentorch-t1gph`.
    //
    // In-entry counters put dweight at 22.501 ms of a 33.889 ms fused backward: 66%, and more than
    // twice dinput. Every lever and refutation on this bead so far aimed at dinput, which is now
    // the smaller half; this is the first arm pointed at the frame that actually dominates.
    //
    // `mb` is capped at 8 by a hardcoded constant and has never been swept. It matters because it
    // trades two streams against each other: `ptile` depends only on `ni`, so panel columns are
    // re-gathered once per m-block, while `atile` depends only on `mi`, so `dout_flat` is re-read
    // once per n-block. At this shape the totals move in OPPOSITE directions -- mb=8 gives
    // 4x189 + 4x21 = 840 MB, mb=16 gives 546 MB, mb=32 gives 525 MB -- so the SHIPPED value is
    // predicted to be the worst of the three by ~1.6x on traffic.
    //
    // PREDICTION RECORDED BEFORE THE RUN: mb=32 wins, by LESS than the 1.6x traffic ratio, because
    // raising mb shrinks nb from 72 to 18 and a narrower N costs something in the microkernel. If
    // mb=32 LOSES, the microkernel term dominates the traffic term, and that is the useful result:
    // it would say this frame is bound by GEMM shape rather than by the redundant gather.
    //
    // mb=4 is included because grids that only look upward from the shipped value are how the SVD
    // panel width sat at 16 while 8 won 15 of 16 cells.
    //
    // The dweight frame is read from the entry's OWN counter, so this is not a subtraction.
    // GATHER vs GEMM INSIDE THE dweight FRAME -- `frankentorch-06csx`.
    //
    // The GEMM shape probe put one thread's 188.7 MFLOP share at ~3.8 ms (49.6 GFLOP/s, which is
    // 75% of this box's square-GEMM ceiling) against a 22.4 ms dweight frame. That says the GEMM
    // is not the frame -- but it says it by subtracting across two measurements, which on this
    // campaign already invented a 6 ms frame that did not exist (ledger 277a). Counters instead.
    //
    // Both counters are summed across rayon workers, so they are CPU time, not wall time: their
    // RATIO is the quantity, and neither absolute should be compared to the frame.
    {
        let mut best_g = f64::INFINITY;
        let mut best_m = f64::INFINITY;
        for rep in 0..reps {
            let _ = ft_kernel_cpu::conv2d_dweight_split_take_ns();
            let out = ft_kernel_cpu::conv2d_backward_mask_fused_f32(
                &ones, &mask, &padded, &weight, BATCH, IN_CH, PH, PW, K, K, H, W, 1, 1, OUT_CH,
                [false, true, false],
            );
            std::hint::black_box(&out);
            let (g_ns, m_ns) = ft_kernel_cpu::conv2d_dweight_split_take_ns();
            if rep > 0 {
                best_g = best_g.min(g_ns as f64 / 1e6);
                best_m = best_m.min(m_ns as f64 / 1e6);
            }
        }
        eprintln!(
            "F32_DWSPLIT panel gather {best_g:8.3} ms CPU   GEMM {best_m:8.3} ms CPU   gather is {:.1}% of the two",
            100.0 * best_g / (best_g + best_m)
        );
    }

    // ---------------------------------------------------------------------------------------
    // THE f64 TWIN -- `frankentorch-06csx`.
    //
    // The f32 twin ships an n-split-first tiling worth 1.5468x paired (ledger 278). The f64 twin
    // still runs the old 4x4 tiling, and the gather-repetition argument behind the win is
    // dtype-independent -- but its MAGNITUDE is not, which is why this is measured rather than
    // inherited. The f64 lane's batch is 8 against the f32 lane's 160, so `flat` is 8192 not
    // 163840: 32 k-blocks per tile instead of 640, which makes the fixed per-tile costs (building
    // `runs`, allocating `ptile`) a ~20x larger share of the frame.
    //
    // PREDICTION RECORDED BEFORE THE RUN: (mb=32, min_nb=18) wins, in the same direction as f32 but
    // by LESS than 1.5468x, because the fixed per-tile costs it cannot remove are a bigger fraction
    // here. If it does not win at all, the gather repetition is being dominated by those fixed
    // costs at this batch, and the f64 twin should keep its heuristic -- which would itself be the
    // useful result, since it would mean the f32 win does NOT generalise by shape argument alone.
    {
        const B64: usize = 8; // C2_N, the f64 conv2d lane's batch
        let padded64: Vec<f64> = (0..B64 * IN_CH * PH * PW)
            .map(|i| ((i % 37) as f64) * 0.013 - 0.21)
            .collect();
        let weight64: Vec<f64> = (0..OUT_CH * IN_CH * K * K)
            .map(|i| ((i % 11) as f64) * 0.0625 - 0.3125)
            .collect();
        let ones64: Vec<f64> = vec![1.0; B64 * OUT_CH * H * W];
        let mask64: Vec<f64> = (0..B64 * OUT_CH * H * W)
            .map(|i| ((i % 23) as f64) * 0.019 - 0.19)
            .collect();

        // PAIRED ROW, f64 fused backward. OFF restores the OLD tiling exactly -- (mb=8,
        // min_nb=72) gives m_blocks 4, n_blocks min(16, 288/72)=4, 16 tiles, nb=72 -- rather than
        // (8, 64), which under the new n-split code would yield 20 tiles on 16 threads and be a
        // straw man. That mistake inflated the f32 row from 1.5468x to 2.2969x before it was
        // caught, and the tell was the OFF arm's ABSOLUTE disagreeing with banked history.
        {
            let once = |on: bool| -> f64 {
                let (pm, pn) = if on {
                    (
                        ft_kernel_cpu::set_conv2d_dweight_mb_f64(0),
                        ft_kernel_cpu::set_conv2d_dweight_min_nb_f64(0),
                    )
                } else {
                    (
                        ft_kernel_cpu::set_conv2d_dweight_mb_f64(8),
                        ft_kernel_cpu::set_conv2d_dweight_min_nb_f64(72),
                    )
                };
                let start = Instant::now();
                let out = ft_kernel_cpu::conv2d_backward_mask_fused_f64(
                    &ones64, &mask64, &padded64, &weight64, B64, IN_CH, PH, PW, K, K, H, W, 1, 1,
                    OUT_CH,
                    [true, true, false],
                );
                let ms = start.elapsed().as_secs_f64() * 1e3;
                std::hint::black_box(&out);
                ft_kernel_cpu::set_conv2d_dweight_mb_f64(pm);
                ft_kernel_cpu::set_conv2d_dweight_min_nb_f64(pn);
                ms
            };
            let mut off_v = Vec::new();
            let mut on_v = Vec::new();
            let mut nulls = Vec::new();
            for rep in 0..reps {
                let r = if rep % 2 == 0 {
                    let a = [once(false), once(true), once(true), once(false)];
                    [a[0], a[1], a[2], a[3]]
                } else {
                    let a = [once(true), once(false), once(false), once(true)];
                    [a[1], a[0], a[3], a[2]]
                };
                if rep == 0 {
                    continue;
                }
                off_v.push(r[0].min(r[3]));
                on_v.push(r[1].min(r[2]));
                nulls.push(r[0] / r[3]);
            }
            let median = |v: &mut Vec<f64>| -> f64 {
                v.sort_by(f64::total_cmp);
                if v.is_empty() { f64::NAN } else { v[v.len() / 2] }
            };
            let mut ratios: Vec<f64> = off_v.iter().zip(&on_v).map(|(a, b)| a / b).collect();
            let paired = median(&mut ratios);
            let null = median(&mut nulls.clone());
            let wins = off_v.iter().zip(&on_v).filter(|(o, n)| n < o).count();
            let off_m = median(&mut off_v.clone());
            let on_m = median(&mut on_v.clone());
            eprintln!(
                "F64_NSPLIT fused backward  OFF (old tiling 4x4, nb=72) {off_m:7.3} ms   ON (n-split) {on_m:7.3} ms"
            );
            eprintln!(
                "F64_NSPLIT   marginal {:.4}x   paired {paired:.4}x   SIGN TEST {wins}/{}   A/A null {null:.4} {}",
                off_m / on_m,
                off_v.len(),
                if (0.97..=1.03).contains(&null) { "PASS" } else { "FAIL -- discard this row" }
            );
        }

        for (mb, min_nb) in [(0usize, 0usize), (32, 18), (32, 9), (16, 18), (32, 32)] {
            let pm = ft_kernel_cpu::set_conv2d_dweight_mb_f64(mb);
            let pn = ft_kernel_cpu::set_conv2d_dweight_min_nb_f64(min_nb);
            let mut best = f64::INFINITY;
            for rep in 0..reps {
                let start = Instant::now();
                let out = ft_kernel_cpu::conv2d_backward_mask_fused_f64(
                    &ones64, &mask64, &padded64, &weight64, B64, IN_CH, PH, PW, K, K, H, W, 1, 1,
                    OUT_CH,
                    [false, true, false],
                );
                let ms = start.elapsed().as_secs_f64() * 1e3;
                std::hint::black_box(&out);
                if rep > 0 {
                    best = best.min(ms);
                }
            }
            ft_kernel_cpu::set_conv2d_dweight_mb_f64(pm);
            ft_kernel_cpu::set_conv2d_dweight_min_nb_f64(pn);
            eprintln!(
                "F64_DWTILE mb={mb:2} min_nb={min_nb:2}   dweight entry {best:7.3} ms{}",
                if mb == 0 { "   <- SHIPPED" } else { "" }
            );
        }
    }

    // PAIRED ROW FOR THE n-SPLIT TILING -- `frankentorch-06csx`.
    //
    // OFF restores the previous heuristic exactly (mb=8, min_nb=64); ON is the shipped default.
    // Whole fused backward, alternating square per rep, per-rep min-of-2, median of per-rep
    // ratios, A/A null from the two same-arm samples of one rep, marginal and sign test printed
    // beside it (ledger 274c/275b).
    {
        let once = |on: bool| -> f64 {
            let (pm, pn) = if on {
                (ft_kernel_cpu::set_conv2d_dweight_mb(0), ft_kernel_cpu::set_conv2d_dweight_min_nb(0))
            } else {
                // min_nb=72, NOT 64. Under the new n-split code, min_nb=64 gives
                // n_blocks = min(16, ceil(288/64)) = 5 and so 4 x 5 = 20 tiles on 16 threads --
                // a ragged second round the OLD heuristic never had, because it derived
                // n_blocks = threads.div_ceil(m_blocks) = 4 and produced exactly 16 tiles.
                // Measuring against that straw man read 2.2969x. 288/72 = 4 reproduces the
                // incumbent tiling exactly: m_blocks 4, n_blocks 4, nb 72, 16 tiles.
                (ft_kernel_cpu::set_conv2d_dweight_mb(8), ft_kernel_cpu::set_conv2d_dweight_min_nb(72))
            };
            let start = Instant::now();
            let out = ft_kernel_cpu::conv2d_backward_mask_fused_f32(
                &ones, &mask, &padded, &weight, BATCH, IN_CH, PH, PW, K, K, H, W, 1, 1, OUT_CH,
                [true, true, false],
            );
            let ms = start.elapsed().as_secs_f64() * 1e3;
            std::hint::black_box(&out);
            ft_kernel_cpu::set_conv2d_dweight_mb(pm);
            ft_kernel_cpu::set_conv2d_dweight_min_nb(pn);
            ms
        };
        let mut off_v = Vec::new();
        let mut on_v = Vec::new();
        let mut nulls = Vec::new();
        for rep in 0..reps {
            let r = if rep % 2 == 0 {
                let a = [once(false), once(true), once(true), once(false)];
                [a[0], a[1], a[2], a[3]]
            } else {
                let a = [once(true), once(false), once(false), once(true)];
                [a[1], a[0], a[3], a[2]]
            };
            if rep == 0 {
                continue;
            }
            off_v.push(r[0].min(r[3]));
            on_v.push(r[1].min(r[2]));
            nulls.push(r[0] / r[3]);
        }
        let median = |v: &mut Vec<f64>| -> f64 {
            v.sort_by(f64::total_cmp);
            if v.is_empty() { f64::NAN } else { v[v.len() / 2] }
        };
        let mut ratios: Vec<f64> = off_v.iter().zip(&on_v).map(|(a, b)| a / b).collect();
        let paired = median(&mut ratios);
        let null = median(&mut nulls.clone());
        let wins = off_v.iter().zip(&on_v).filter(|(o, n)| n < o).count();
        let off_m = median(&mut off_v.clone());
        let on_m = median(&mut on_v.clone());
        eprintln!(
            "F32_NSPLIT fused backward  OFF (incumbent tiling: 4x4=16 tiles, nb=72) {off_m:7.3} ms   ON (n-split) {on_m:7.3} ms"
        );
        eprintln!(
            "F32_NSPLIT   marginal {:.4}x   paired {paired:.4}x   SIGN TEST ON faster in {wins}/{} reps   A/A null {null:.4} {}",
            off_m / on_m,
            off_v.len(),
            if (0.97..=1.03).contains(&null) { "PASS" } else { "FAIL -- discard this row" }
        );
    }

    // THE CELL THE CLAMP HID -- `frankentorch-06csx`. The mb sweep showed the gather scaling with
    // m_blocks while the frame did not, because `n_blocks` is clamped to patch_width/64 = 5 and so
    // raising mb converts saved gather work into idle cores. This sweeps (mb, min_nb) together so
    // occupancy is held at 16 tiles while m_blocks falls -- the combination the one-dimensional
    // sweep could not reach.
    //
    // PREDICTION RECORDED BEFORE THE RUN: (mb=32, min_nb=18) should be the best cell. It gathers
    // once instead of four times (82.5 vs 298.7 ms CPU in the frame's dominant half) at full
    // occupancy. The risk is that nb=18 is a poor GEMM width -- the shape probe measured only
    // 1.12x between a good and a bad shape, so the gather term should dominate. If it does NOT
    // win, GEMM width matters more than the shape probe suggested and the frame is closed.
    for (mb, min_nb) in [(8usize, 0usize), (32, 64), (32, 18), (32, 9), (16, 18), (32, 32)] {
        let prev_nb = ft_kernel_cpu::set_conv2d_dweight_min_nb(min_nb);
        let previous = ft_kernel_cpu::set_conv2d_dweight_mb(mb);
        let mut best = f64::INFINITY;
        let mut best_g = f64::INFINITY;
        let mut best_m = f64::INFINITY;
        for rep in 0..reps {
            let _ = ft_kernel_cpu::masked_frame_take_ns();
            let _ = ft_kernel_cpu::conv2d_dweight_split_take_ns();
            let out = ft_kernel_cpu::conv2d_backward_mask_fused_f32(
                &ones, &mask, &padded, &weight, BATCH, IN_CH, PH, PW, K, K, H, W, 1, 1, OUT_CH,
                [false, true, false],
            );
            std::hint::black_box(&out);
            let (_, w_ns, _) = ft_kernel_cpu::masked_frame_take_ns();
            let (g_ns, m_ns) = ft_kernel_cpu::conv2d_dweight_split_take_ns();
            if rep > 0 {
                best = best.min(w_ns as f64 / 1e6);
                best_g = best_g.min(g_ns as f64 / 1e6);
                best_m = best_m.min(m_ns as f64 / 1e6);
            }
        }
        ft_kernel_cpu::set_conv2d_dweight_mb(previous);
        ft_kernel_cpu::set_conv2d_dweight_min_nb(prev_nb);
        let m_blocks = OUT_CH.div_ceil(mb);
        eprintln!(
            "F32_DWNB mb={mb:2} min_nb={min_nb:2} -> m_blocks {m_blocks}   frame {best:7.3} ms   gather {best_g:8.3} + GEMM {best_m:7.3} ms CPU{}",
            if mb == 8 && min_nb == 0 { "   <- SHIPPED" } else { "" }
        );
    }

    for mb in [4usize, 8, 16, 32] {
        let previous = ft_kernel_cpu::set_conv2d_dweight_mb(mb);
        let mut best = f64::INFINITY;
        let mut best_g = f64::INFINITY;
        let mut best_m = f64::INFINITY;
        for rep in 0..reps {
            let _ = ft_kernel_cpu::masked_frame_take_ns();
            let _ = ft_kernel_cpu::conv2d_dweight_split_take_ns();
            let out = ft_kernel_cpu::conv2d_backward_mask_fused_f32(
                &ones, &mask, &padded, &weight, BATCH, IN_CH, PH, PW, K, K, H, W, 1, 1, OUT_CH,
                [false, true, false],
            );
            std::hint::black_box(&out);
            let (_, w_ns, _) = ft_kernel_cpu::masked_frame_take_ns();
            let (g_ns, m_ns) = ft_kernel_cpu::conv2d_dweight_split_take_ns();
            if rep > 0 {
                best = best.min(w_ns as f64 / 1e6);
                best_g = best_g.min(g_ns as f64 / 1e6);
                best_m = best_m.min(m_ns as f64 / 1e6);
            }
        }
        ft_kernel_cpu::set_conv2d_dweight_mb(previous);
        let m_blocks = OUT_CH.div_ceil(mb);
        // Report the gather/GEMM split PER mb. The counters say the gather is 82% of this frame,
        // and the tiling says raising mb should cut the redundant gather proportionally -- yet the
        // frame barely moved between mb=8 and mb=16. Both cannot be right, and the split measured
        // at each mb says which.
        eprintln!(
            "F32_DWMB mb={mb:2} -> m_blocks {m_blocks}   dweight frame {best:7.3} ms   gather {best_g:8.3} + GEMM {best_m:7.3} ms CPU ({:.1}% gather){}",
            100.0 * best_g / (best_g + best_m),
            if mb == 8 { "   <- SHIPPED" } else { "" }
        );
    }

    // ---------------------------------------------------------------------------------------
    // WHERE THE ~6 ms ACTUALLY IS — `frankentorch-t1gph`, ledger 276d.
    //
    // The deficit map left a residual I refused to name: the tiled `dout_flat` build (2.668 ms)
    // plus the direct dinput kernel (~11.3 ms) is ~14.0 ms, against a measured 20.037 ms for the
    // fused dinput-only entry. Six milliseconds unaccounted, and the tempting stories — first
    // touch on the fresh 21 MB `dout_flat`, or on the 23.7 MB output — are exactly the kind of
    // plausible attribution that item 276a already punished on this bead (18 ms claimed, 4 ms
    // real).
    //
    // So this reads COUNTERS INSIDE THE ENTRY instead of subtracting arms. The three frames are
    // timed where they happen, and the entry's own wall clock is timed around all of them. What
    // the frames do not account for is then genuinely OUTSIDE all three — the `Vec` allocations
    // and their first touch, and the return — which is a different lever from any of the three and
    // cannot be reached by tuning them.
    //
    // Both `output_mask` shapes are run because they answer different questions: dinput-only is
    // the arm the 20.037 ms figure came from, and dinput+dweight is the arm the lane actually
    // executes.
    for (label, om) in [
        ("dinput ONLY      ", [true, false, false]),
        ("dinput + dweight ", [true, true, false]),
    ] {
        let mut entry = f64::INFINITY;
        let mut best_dout = f64::INFINITY;
        let mut best_dw = f64::INFINITY;
        let mut best_di = f64::INFINITY;
        for rep in 0..reps {
            let _ = ft_kernel_cpu::masked_frame_take_ns();
            let start = Instant::now();
            let out = ft_kernel_cpu::conv2d_backward_mask_fused_f32(
                &ones, &mask, &padded, &weight, BATCH, IN_CH, PH, PW, K, K, H, W, 1, 1, OUT_CH,
                om,
            );
            let ms = start.elapsed().as_secs_f64() * 1e3;
            std::hint::black_box(&out);
            let (d_ns, w_ns, i_ns) = ft_kernel_cpu::masked_frame_take_ns();
            if rep > 0 {
                entry = entry.min(ms);
                best_dout = best_dout.min(d_ns as f64 / 1e6);
                best_dw = best_dw.min(w_ns as f64 / 1e6);
                best_di = best_di.min(i_ns as f64 / 1e6);
            }
        }
        let accounted = best_dout + best_dw + best_di;
        eprintln!(
            "F32_FRAMES {label} entry {entry:7.3} ms = dout_flat {best_dout:6.3} + dweight {best_dw:7.3} + dinput {best_di:7.3}  -> accounted {:5.1}%, OUTSIDE all three {:6.3} ms",
            100.0 * accounted / entry,
            entry - accounted
        );
    }

    // ---------------------------------------------------------------------------------------
    // THE PAIRED ROW FOR THE TILED dout_flat LEVER — `frankentorch-t1gph`.
    //
    // The tiled build shipped on FRAME evidence (4.017 -> 2.784 ms on the build loop, bitwise
    // identical) and I said plainly at the time that the LANE ratio was not measured. This is that
    // measurement, against the live incumbent through the real fused entry.
    //
    // Same harness as the levers on x6wc3/37sxo, and the shape is not optional: OFF/ON/ON/OFF with
    // the square ALTERNATED per rep (a fixed square parks one arm in the two middle slots forever
    // and the A/A null, comparing the OUTER slots, is structurally blind to it — ledger 274c),
    // per-rep min-of-2 per arm, median of per-rep ratios, and the marginal ratio plus a SIGN TEST
    // printed beside it so estimator disagreement shows up as the harness defect it is (275b).
    //
    // PREDICTION, recorded before the run so it cannot be fitted afterwards: the frame is ~4 ms of
    // a ~22.7 ms dinput-only arm and the tiling saves ~1.2 ms of it, so the entry should move
    // ~1.05x — at or below what this harness resolves. A null here does NOT retract the frame
    // measurement; it bounds what the frame is worth at the entry, which is the honest thing to
    // put on the thread.
    let mut tile_off = Vec::new();
    let mut tile_on = Vec::new();
    let mut tile_nulls = Vec::new();
    {
        let once = |tiled: bool| -> f64 {
            let previous = ft_kernel_cpu::set_masked_dout_tiled(tiled);
            let start = Instant::now();
            let out = ft_kernel_cpu::conv2d_backward_mask_fused_f32(
                &ones, &mask, &padded, &weight, BATCH, IN_CH, PH, PW, K, K, H, W, 1, 1, OUT_CH,
                [true, true, false],
            );
            let ms = start.elapsed().as_secs_f64() * 1e3;
            std::hint::black_box(&out);
            ft_kernel_cpu::set_masked_dout_tiled(previous);
            ms
        };
        for rep in 0..reps {
            let r = if rep % 2 == 0 {
                let a = [once(false), once(true), once(true), once(false)];
                [a[0], a[1], a[2], a[3]]
            } else {
                let a = [once(true), once(false), once(false), once(true)];
                [a[1], a[0], a[3], a[2]]
            };
            if rep == 0 {
                continue;
            }
            tile_off.push(r[0].min(r[3]));
            tile_on.push(r[1].min(r[2]));
            tile_nulls.push(r[0] / r[3]);
        }
    }
    {
        let median = |v: &mut Vec<f64>| -> f64 {
            v.sort_by(f64::total_cmp);
            if v.is_empty() { f64::NAN } else { v[v.len() / 2] }
        };
        let mut ratios: Vec<f64> = tile_off.iter().zip(&tile_on).map(|(a, b)| a / b).collect();
        let paired = median(&mut ratios);
        let null = median(&mut tile_nulls.clone());
        let wins = tile_off.iter().zip(&tile_on).filter(|(o, n)| n < o).count();
        let off_m = median(&mut tile_off.clone());
        let on_m = median(&mut tile_on.clone());
        eprintln!("F32_TILEAB fused backward (dinput+dweight), OFF row-gather {off_m:8.3} ms   ON tiled {on_m:8.3} ms");
        eprintln!(
            "F32_TILEAB   marginal {:.4}x   paired {paired:.4}x   SIGN TEST ON faster in {wins}/{} reps   A/A null {null:.4} {}",
            off_m / on_m,
            tile_off.len(),
            if (0.97..=1.03).contains(&null) { "PASS" } else { "FAIL — discard this row" }
        );
    }

    // `incoming * mask` in the `[flat][out_ch]` layout the fused entry forms internally, so the
    // panel-free arm consumes exactly what the shipping route consumes. `ones` stands in for
    // `incoming` here, matching the arms above.
    let patch_count_probe = H * W;
    let mut dout_flat = vec![0.0f32; BATCH * patch_count_probe * OUT_CH];
    for n in 0..BATCH {
        for oc in 0..OUT_CH {
            let src = (n * OUT_CH + oc) * patch_count_probe;
            for patch in 0..patch_count_probe {
                dout_flat[(n * patch_count_probe + patch) * OUT_CH + oc] =
                    ones[src + patch] * mask[src + patch];
            }
        }
    }

    let mut panel_free = f64::INFINITY;
    let mut panel_free_worst_rel = 0.0f64;
    for rep in 0..reps {
        let start = Instant::now();
        let direct = {
            let patch_width = IN_CH * K * K;
            let patch_count = H * W;
            let mut out = vec![0.0f32; BATCH * IN_CH * PH * PW];
            out.par_chunks_mut(IN_CH * PH * PW)
                .enumerate()
                .for_each(|(b, plane)| {
                    // 9 x OUT_CH of weights for one input channel: 288 floats, L1-resident, and
                    // laid out so the inner loop reads it contiguously.
                    let mut w9 = vec![0.0f32; K * K * OUT_CH];
                    for c in 0..IN_CH {
                        for tap in 0..K * K {
                            for oc in 0..OUT_CH {
                                w9[tap * OUT_CH + oc] =
                                    weight[oc * patch_width + c * K * K + tap];
                            }
                        }
                        for ih in 0..PH {
                            for iw in 0..PW {
                                let mut acc = 0.0f32;
                                for kr in 0..K {
                                    // Wrapping sub: an out-of-range origin becomes huge and the
                                    // bound check rejects it, so no signed arithmetic is needed.
                                    let oh_i = ih.wrapping_sub(kr);
                                    if oh_i >= H {
                                        continue;
                                    }
                                    for kc in 0..K {
                                        let ow_i = iw.wrapping_sub(kc);
                                        if ow_i >= W {
                                            continue;
                                        }
                                        let pc = oh_i * W + ow_i;
                                        let drow = &dout_flat
                                            [(b * patch_count + pc) * OUT_CH..][..OUT_CH];
                                        let wrow = &w9[(kr * K + kc) * OUT_CH..][..OUT_CH];
                                        for (d, w) in drow.iter().zip(wrow) {
                                            acc += *d * *w;
                                        }
                                    }
                                }
                                plane[c * PH * PW + ih * PW + iw] = acc;
                            }
                        }
                    }
                });
            out
        };
        let ms = start.elapsed().as_secs_f64() * 1e3;
        if rep > 0 {
            panel_free = panel_free.min(ms);
        }
        if rep == 0 {
            // Tolerance, not bits: summing `oc` sequentially is a different association than the
            // GEMM's. Checking it at all is the point — a fast kernel computing the wrong thing is
            // the failure mode a timing probe cannot see.
            let reference = ft_kernel_cpu::conv2d_backward_dinput_direct_f32(
                &dout_flat, &weight, BATCH, IN_CH, PH, PW, K, K, H, W, 1, 1, OUT_CH,
            );
            for (a, b) in reference.iter().zip(&direct) {
                let rel = f64::from((a - b).abs()) / (1.0 + f64::from(a.abs()));
                if rel > panel_free_worst_rel {
                    panel_free_worst_rel = rel;
                }
            }
        }
        std::hint::black_box(&direct);
    }
    eprintln!("F32_DINPUT dinput ONLY, PANEL-FREE direct kernel   {panel_free:8.3} ms  (worst rel vs shipping route {panel_free_worst_rel:.3e})");
    let gflop = 2.0 * (BATCH * H * W * IN_CH * K * K * OUT_CH) as f64 / 1e9;
    eprintln!(
        "F32_DINPUT   {gflop:.3} GFLOP -> shipping {:.1} GFLOP/s, panel-free {:.1} GFLOP/s, ratio {:.4}x",
        gflop / (legacy_dinput.min(fused_dinput) / 1e3),
        gflop / (panel_free / 1e3),
        fused_dinput / panel_free
    );
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
    // The masked lane runs weight_grad = false, so its backward is the dinput-ONLY arm.
    let masked_kernels = forward + fused_dinput;
    eprintln!(
        "F32_DINPUT MASKED-LANE KERNELS (one host): forward {:.3} ({:.0}%) + backward {:.3} \
         ({:.0}%) = {:.3} ms. The SUMMED lane's kernels are forward + adjoint = {:.3} ms, and \
         that lane measures at PARITY with PyTorch — so the forward is not the deficit.",
        forward,
        100.0 * forward / masked_kernels,
        fused_dinput,
        100.0 * fused_dinput / masked_kernels,
        masked_kernels,
        forward + adjoint
    );
    eprintln!(
        "F32_DINPUT SCATTER: col2im alone / direct dinput = {:.3}x. Comparable means the SCATTER \
         is the dominant frame inside the direct route; much larger means the blocking already \
         beats it and the per-block GEMM is what remains.",
        col2im / fused_dinput
    );
    eprintln!(
        "F32_DINPUT GATHER-DIRECTION: col2im (scatter) / im2col (gather) = {:.3}x on identical \
         volume and geometry. Much greater than 1 means writing each output ONCE is the win the \
         gather inversion would buy; ~1.0 REFUTES it before the kernel is written.",
        col2im / im2col
    );
    eprintln!(
        "F32_DINPUT DWEIGHT LEVER (kernel level): legacy / streamed = {:.3}x. dweight is {:.0}% \
         of the dinput+dweight pair on THIS host; the split is host-dependent (48-65% observed) \
         and the two halves are comparable.",
        dweight_legacy / fused_dweight,
        100.0 * fused_dweight / (fused_dweight + fused_dinput)
    );
}

//! Where does the f32 TRAINING lane spend its time, outside the kernels? — `frankentorch-t1gph`.
//!
//! WHY THIS EXISTS. Every conv2d lever this campaign shipped or refuted was measured against
//! `conv2d_backward_mask_fused_f32` called DIRECTLY. But the board's lane is not a kernel call: it
//! is a `FrankenTorchSession` doing
//!
//!     functional_conv2d -> tensor_mul(mask) -> tensor_sum -> tensor_backward
//!
//! with the input leaf requiring grad, and the timed region covers all of it. The kernel backward
//! I have been optimising is only part of what that lane measures, and the rest — the retained
//! `tensor_pad` node the grad path creates, the mask multiply, the sum, the tape walk and the
//! gradient allocation — has never been attributed at all.
//!
//! ROUTING, WHICH IS WHY THIS PROBE TARGETS THE TRAIN LANE SPECIFICALLY. `conv2d_f32_masked`
//! passes `weight_grad = false`, so it never builds dweight and the four dweight levers (263, 277,
//! 278, 280) are routed away from it (ledger 282a). The lane those levers actually reach is
//! `conv2d_f32_masked_train`, reproduced here: input leaf requires grad AND weight requires grad,
//! so the fused backward runs with `output_mask = [true, true, false]`.
//!
//! NO TORCH ARM, DELIBERATELY. The h2h harness cannot certify a row on a quiet host right now —
//! its drift gate needs `start_load >= 4 * self_load * (1 - exp(-T/60))`, about 86 at rayon=16
//! plus torch's 8 threads, which a 32-core box cannot supply, and waiting for quiet lowers
//! `start_load` and makes it worse (four voided attempts, ledger 282d). This probe is FT-only, so
//! it runs on any rch worker and answers "which frame of the training lane is worth a lever"
//! without needing that window. It makes NO vs-PyTorch claim.
//!
//! Everything goes to STDERR so a remote runner returns it.

use ft_api::FrankenTorchSession;
use ft_core::ExecutionMode;
use std::time::Instant;

// The f32 lane's shape, from the harness's C2F32_N and C2_* constants.
const BATCH: usize = 160;
const IN_CH: usize = 32;
const OUT_CH: usize = 32;
const H: usize = 32;
const W: usize = 32;
const K: usize = 3;

fn main() {
    // ARGV, not env: `rch exec` does not forward the caller's environment (ledger 273c).
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
        "PROV host={} nproc={} rayon={} reps={reps} loadavg={}",
        std::fs::read_to_string("/etc/hostname").unwrap_or_default().trim(),
        std::thread::available_parallelism().map_or(0, std::num::NonZero::get),
        rayon::current_num_threads(),
        std::fs::read_to_string("/proc/loadavg").unwrap_or_default().trim(),
    );

    let values: Vec<f32> = (0..BATCH * IN_CH * H * W)
        .map(|i| ((i % 37) as f32) * 0.013 - 0.21)
        .collect();
    let weights: Vec<f32> = (0..OUT_CH * IN_CH * K * K)
        .map(|i| ((i % 11) as f32) * 0.0625 - 0.3125)
        .collect();
    let mask: Vec<f32> = (0..BATCH * OUT_CH * H * W)
        .map(|i| ((i % 23) as f32) * 0.019 - 0.19)
        .collect();

    // BIT-EXACTNESS FIRST, then the ratio. The reuse path replaces a recomputed convolution with
    // the one `functional_conv2d` already produced; both are `conv2d_forward_f32` on the same
    // plan inputs, so the forward VALUE and every gradient must match to the bit. Checked here
    // rather than only in a unit test because this probe drives the real session, which is where
    // a tape-level mistake (a stale `precomputed`, a wrong node) would actually show.
    {
        let run = |reuse: bool| -> (Vec<f32>, f64) {
            let prev = ft_api::set_fuse_conv2d_reuse_f32(reuse);
            let mut session = FrankenTorchSession::new(ExecutionMode::Strict);
            let x = session
                .tensor_variable_f32(values.clone(), vec![BATCH, IN_CH, H, W], true)
                .expect("leaf x");
            let w = session
                .tensor_variable_f32(weights.clone(), vec![OUT_CH, IN_CH, K, K], true)
                .expect("leaf w");
            let m = session
                .tensor_variable_f32(mask.clone(), vec![BATCH, OUT_CH, H, W], false)
                .expect("leaf mask");
            let out = session
                .functional_conv2d(x, w, None, (1, 1), (1, 1))
                .expect("conv2d");
            let scored = session.tensor_mul(out, m).expect("mask multiply");
            let loss = session.tensor_sum(scored).expect("sum");
            let report = session.tensor_backward(loss).expect("backward");
            let gx = report.gradient(x).expect("grad x").to_vec();
            let gw: f64 = report
                .gradient(w)
                .expect("grad w")
                .iter()
                .map(|g| g.abs())
                .sum();
            ft_api::set_fuse_conv2d_reuse_f32(prev);
            (gx.iter().map(|&v| v as f32).collect(), gw)
        };
        let _ = ft_api::take_fuse_conv2d_fwd_calls();
        let (gx_off, gw_off) = run(false);
        let (calls_off, recomp_off) = ft_api::take_fuse_conv2d_fwd_calls();
        let (gx_on, gw_on) = run(true);
        let (calls_on, recomp_on) = ft_api::take_fuse_conv2d_fwd_calls();
        eprintln!(
            "TRAIN   FWD-CLOSURE  OFF calls {calls_off} recomputes {recomp_off}   ON calls {calls_on} recomputes {recomp_on}   (a recompute on the ON arm is the cached value having been consumed already)"
        );
        let mismatches = gx_off
            .iter()
            .zip(&gx_on)
            .filter(|(a, b)| a.to_bits() != b.to_bits())
            .count();
        eprintln!(
            "TRAIN   PARITY reuse-vs-recompute: dx bitwise mismatches {mismatches}/{}   dw |sum| {:.9e} vs {:.9e}",
            gx_off.len(),
            gw_off,
            gw_on
        );
        assert_eq!(mismatches, 0, "reuse path changed the input gradient");
        assert_eq!(gw_off.to_bits(), gw_on.to_bits(), "reuse path changed the weight gradient");
    }

    let mut total = f64::INFINITY;
    let mut t_build = f64::INFINITY;
    let mut t_fwd = f64::INFINITY;
    let mut t_mul = f64::INFINITY;
    let mut t_sum = f64::INFINITY;
    let mut t_bwd = f64::INFINITY;
    let mut t_grad = f64::INFINITY;
    let mut t_kernel = f64::INFINITY;
    let mut t_down = f64::INFINITY;
    let mut t_widen = f64::INFINITY;

    for rep in 0..reps {
        // Leaf construction is timed separately and EXCLUDED from the lane total below, because
        // the board's lane builds its leaves before starting its clock. Reporting it anyway: a
        // 20 MB + 23 MB pair of `to_vec()` calls is not free, and if it turns out to dominate this
        // process then the frames measured after it are being read on a warmed allocator.
        let start = Instant::now();
        let mut session = FrankenTorchSession::new(ExecutionMode::Strict);
        let x = session
            .tensor_variable_f32(values.clone(), vec![BATCH, IN_CH, H, W], true)
            .expect("leaf x");
        let w = session
            .tensor_variable_f32(weights.clone(), vec![OUT_CH, IN_CH, K, K], true)
            .expect("leaf w");
        let m = session
            .tensor_variable_f32(mask.clone(), vec![BATCH, OUT_CH, H, W], false)
            .expect("leaf mask");
        let build = start.elapsed().as_secs_f64() * 1e3;

        // THE LANE, matching `timed_conv2d_f32(..., weight_grad = true)`.
        let lane_start = Instant::now();

        let f0 = Instant::now();
        let out = session
            .functional_conv2d(x, w, None, (1, 1), (1, 1))
            .expect("conv2d");
        let fwd = f0.elapsed().as_secs_f64() * 1e3;

        let m0 = Instant::now();
        let scored = session.tensor_mul(out, m).expect("mask multiply");
        let mul = m0.elapsed().as_secs_f64() * 1e3;

        let s0 = Instant::now();
        let loss = session.tensor_sum(scored).expect("sum");
        let sum = s0.elapsed().as_secs_f64() * 1e3;

        // ATTRIBUTE THE BACKWARD -- `frankentorch-t1gph`. tensor_backward is 63% of this lane and
        // only the conv2d kernel inside it has ever been attributed; the tape walk, the gradient
        // allocation and dsum never have. The counters live INSIDE
        // conv2d_backward_mask_fused_f32, so draining them around this call splits the backward
        // in-context — no subtraction across two different call sites, which is the trap that
        // invented a 6 ms frame in ledger 277a.
        let _ = ft_kernel_cpu::masked_frame_take_ns();
        let _ = ft_api::take_fuse_bwd_frames_ns();
        let b0 = Instant::now();
        let report = session.tensor_backward(loss).expect("backward");
        let bwd = b0.elapsed().as_secs_f64() * 1e3;
        let (k_dout, k_dweight, k_dinput) = ft_kernel_cpu::masked_frame_take_ns();
        let kernel_ms = (k_dout + k_dweight + k_dinput) as f64 / 1e6;
        let (bw_down_ns, bw_widen_ns) = ft_api::take_fuse_bwd_frames_ns();
        let down_ms = bw_down_ns as f64 / 1e6;
        let widen_ms = bw_widen_ns as f64 / 1e6;

        let lane = lane_start.elapsed().as_secs_f64() * 1e3;

        // The board's lane also reads the input gradient out to checksum it; that read walks a
        // 20 MB buffer and belongs to the lane, so it is measured rather than assumed free.
        let g0 = Instant::now();
        let checksum: f64 = report
            .gradient(x)
            .expect("grad")
            .iter()
            .map(|g| g.abs())
            .sum::<f64>();
        let grad = g0.elapsed().as_secs_f64() * 1e3;
        std::hint::black_box(checksum);

        if rep > 0 {
            total = total.min(lane);
            t_build = t_build.min(build);
            t_fwd = t_fwd.min(fwd);
            t_mul = t_mul.min(mul);
            t_sum = t_sum.min(sum);
            t_bwd = t_bwd.min(bwd);
            t_grad = t_grad.min(grad);
            t_kernel = t_kernel.min(kernel_ms);
            t_down = t_down.min(down_ms);
            t_widen = t_widen.min(widen_ms);
        }
    }

    let accounted = t_fwd + t_mul + t_sum + t_bwd;
    eprintln!("TRAIN b={BATCH} ci={IN_CH} co={OUT_CH} {H}x{W} k{K}x{K}  (min after discarding the first)");
    eprintln!("TRAIN   lane total (fwd+mul+sum+bwd)   {total:8.3} ms");
    eprintln!("TRAIN     functional_conv2d (pad+fwd)  {t_fwd:8.3} ms  ({:5.1}%)", 100.0 * t_fwd / total);
    eprintln!("TRAIN     tensor_mul (mask)            {t_mul:8.3} ms  ({:5.1}%)", 100.0 * t_mul / total);
    eprintln!("TRAIN     tensor_sum                   {t_sum:8.3} ms  ({:5.1}%)", 100.0 * t_sum / total);
    eprintln!("TRAIN     tensor_backward (tape walk)  {t_bwd:8.3} ms  ({:5.1}%)", 100.0 * t_bwd / total);
    eprintln!(
        "TRAIN   accounted {:5.1}%   unattributed {:8.3} ms",
        100.0 * accounted / total,
        total - accounted
    );
    eprintln!(
        "TRAIN     of that backward: conv2d fused KERNEL {t_kernel:8.3} ms  ({:5.1}% of the backward, {:5.1}% of the lane)",
        100.0 * t_kernel / t_bwd,
        100.0 * t_kernel / total
    );
    eprintln!(
        "TRAIN     backward NON-kernel total {:8.3} ms  ({:5.1}% of the lane), decomposed:",
        t_bwd - t_kernel,
        100.0 * (t_bwd - t_kernel) / total
    );
    eprintln!(
        "TRAIN       f64->f32 DOWNCAST of the incoming grad  {t_down:8.3} ms  ({:5.1}% of the lane)  [SERIAL .iter().map()]",
        100.0 * t_down / total
    );
    eprintln!(
        "TRAIN       f32->f64 WIDEN of dpadded/dweight       {t_widen:8.3} ms  ({:5.1}% of the lane)  [already par_iter]",
        100.0 * t_widen / total
    );
    eprintln!(
        "TRAIN       REMAINDER (tape walk, dsum, pad bwd, grad alloc) {:8.3} ms  ({:5.1}% of the lane)",
        t_bwd - t_kernel - t_down - t_widen,
        100.0 * (t_bwd - t_kernel - t_down - t_widen) / total
    );
    eprintln!("TRAIN   [outside the lane clock] leaf build {t_build:8.3} ms | grad read+checksum {t_grad:8.3} ms");

    // THE ALLOCATOR-WARMTH TEST -- `frankentorch-t1gph`, ledger 283d.
    //
    // The reuse lever sheds ~11 ms from `tensor_mul` and the BACKWARD gains ~10 ms, replicated
    // across two 40-sample windows. My first explanation (the tape re-invokes the forward closure
    // and the one-shot cache falls through) was REFUTED by a counter: the closure runs exactly
    // once in each arm. The standing hypothesis is allocator warmth -- the recompute arm allocates
    // and drops a transient 21 MB `out` buffer inside the closure, leaving a block of exactly that
    // size on the free list for the backward to reuse, while the reuse arm never creates it.
    //
    // THIS TESTS THE HYPOTHESIS WITHOUT WRITING A LEVER FOR IT. A third arm runs the reuse path
    // but allocates and drops an equivalent dummy buffer first, priming the free list the same way
    // the recompute incidentally did. Three pre-specified outcomes:
    //   * dummy ~= recompute  -> allocator warmth CONFIRMED, and a backward-side buffer reuse is
    //     worth ~10 ms of a ~95 ms lane;
    //   * dummy ~= reuse      -> REFUTED, the free list is not the mechanism and the backward's
    //     extra time is something else;
    //   * in between          -> partial, and the remainder still needs a name.
    // A few lines either way, versus a kernel's worth of work to find out by shipping.
    {
        let once = |reuse: bool, prime: bool| -> f64 {
            let prev = ft_api::set_fuse_conv2d_reuse_f32(reuse);
            let mut session = FrankenTorchSession::new(ExecutionMode::Strict);
            let x = session
                .tensor_variable_f32(values.clone(), vec![BATCH, IN_CH, H, W], true)
                .expect("leaf x");
            let w = session
                .tensor_variable_f32(weights.clone(), vec![OUT_CH, IN_CH, K, K], true)
                .expect("leaf w");
            let m = session
                .tensor_variable_f32(mask.clone(), vec![BATCH, OUT_CH, H, W], false)
                .expect("leaf mask");
            let lane0 = Instant::now();
            let out = session
                .functional_conv2d(x, w, None, (1, 1), (1, 1))
                .expect("conv2d");
            let scored = session.tensor_mul(out, m).expect("mask multiply");
            if prime {
                // Same element count as the `out` buffer the recompute arm drops. Touched so the
                // pages are really faulted in, then dropped, leaving it on the free list.
                let mut dummy = vec![0.0f32; BATCH * OUT_CH * H * W];
                dummy[0] = 1.0;
                dummy[BATCH * OUT_CH * H * W - 1] = 1.0;
                std::hint::black_box(&dummy);
                drop(dummy);
            }
            let loss = session.tensor_sum(scored).expect("sum");
            let report = session.tensor_backward(loss).expect("backward");
            let lane = lane0.elapsed().as_secs_f64() * 1e3;
            std::hint::black_box(report.gradient(x).expect("grad").len());
            ft_api::set_fuse_conv2d_reuse_f32(prev);
            lane
        };
        let mut recompute = Vec::new();
        let mut plain = Vec::new();
        let mut primed = Vec::new();
        for rep in 0..reps {
            // Rotate the order each rep so no arm keeps a fixed slot (ledger 274c).
            let (a, b, c) = match rep % 3 {
                0 => (once(false, false), once(true, false), once(true, true)),
                1 => {
                    let x = once(true, false);
                    let y = once(true, true);
                    let z = once(false, false);
                    (z, x, y)
                }
                _ => {
                    let x = once(true, true);
                    let y = once(false, false);
                    let z = once(true, false);
                    (y, z, x)
                }
            };
            if rep == 0 {
                continue;
            }
            recompute.push(a);
            plain.push(b);
            primed.push(c);
        }
        let med = |v: &mut Vec<f64>| -> f64 {
            v.sort_by(f64::total_cmp);
            if v.is_empty() { f64::NAN } else { v[v.len() / 2] }
        };
        let r = med(&mut recompute.clone());
        let pl = med(&mut plain.clone());
        let pr = med(&mut primed.clone());
        eprintln!(
            "TRAIN_PRIME lane  recompute {r:8.3} ms | reuse {pl:8.3} ms | reuse+primed {pr:8.3} ms"
        );
        eprintln!(
            "TRAIN_PRIME   priming recovered {:+.3} ms of the {:+.3} ms the reuse arm gives up vs recompute in the backward  ({:.0}%)",
            pl - pr,
            pl - r,
            if (pl - r).abs() > 1e-9 { 100.0 * (pl - pr) / (pl - r) } else { 0.0 }
        );
    }

    // WHICH LANES ACTUALLY REACH A NARROW -- `frankentorch-dwto7`.
    //
    // Before claiming anything for a site, find out whether a lane executes it. On t1gph FOUR
    // shipped levers turned out to be routed away from that bead's own headline lane (282a), and
    // the fix there was to read the harness flags first. Here the equivalent check is cheaper: the
    // shared helper counts its own calls, so routing a site through it makes the site
    // self-reporting.
    //
    // Three lane shapes, matching the board:
    //   plain      timed_conv2d_f32(.., mask=None, weight_grad=false)  -> "conv2d_f32"
    //   masked     mask, weight_grad=false                             -> "conv2d_f32_masked"
    //   train      mask, weight_grad=true                              -> "conv2d_f32_masked_train"
    // The plain lane is the one that can reach `functional_conv2d`'s OWN f32 backward, because the
    // masked lanes are intercepted by the conv2d/mask fusion. It may still skip the narrow: the
    // all-ones adjoint above it (item 185) handles a sum-loss dout without converting anything.
    {
        let probe = |with_mask: bool, weight_grad: bool| -> (u64, u64) {
            ft_api::reset_narrow_counts();
            let mut session = FrankenTorchSession::new(ExecutionMode::Strict);
            let x = session
                .tensor_variable_f32(values.clone(), vec![BATCH, IN_CH, H, W], true)
                .expect("leaf x");
            let w = session
                .tensor_variable_f32(weights.clone(), vec![OUT_CH, IN_CH, K, K], weight_grad)
                .expect("leaf w");
            let out = session
                .functional_conv2d(x, w, None, (1, 1), (1, 1))
                .expect("conv2d");
            let scored = if with_mask {
                let m = session
                    .tensor_variable_f32(mask.clone(), vec![BATCH, OUT_CH, H, W], false)
                    .expect("leaf mask");
                session.tensor_mul(out, m).expect("mask multiply")
            } else {
                out
            };
            let loss = session.tensor_sum(scored).expect("sum");
            let report = session.tensor_backward(loss).expect("backward");
            std::hint::black_box(report.gradient(x).expect("grad").len());
            ft_api::narrow_counts()
        };
        for (name, with_mask, wg) in [
            ("conv2d_f32        (no mask, no wgrad)", false, false),
            ("conv2d_f32_masked (mask, no wgrad)   ", true, false),
            ("conv2d_f32_masked_train (mask+wgrad) ", true, true),
        ] {
            let (calls, elems) = probe(with_mask, wg);
            eprintln!(
                "NARROW_ROUTE {name}  narrow calls {calls}  elements {elems}{}",
                if calls == 0 { "   <- UNREACHED: gate is correctness only, no perf claim" } else { "" }
            );
        }
    }

    // PAIRED A/B for the PARALLEL NARROW -- `frankentorch-t1gph`, ledger 286.
    //
    // The tape is f64 and the kernel is f32, so every backward narrows 5.24M elements. The widen
    // twin beside it is already par_iter and runs 3.604 ms on MORE elements; this serial narrow
    // runs 14.776 ms, 16% of the training lane. The lever is the one-line symmetry fix.
    //
    // PREDICTION RECORDED BEFORE THE RUN: the downcast frame falls toward the widen's rate
    // (~3-4 ms), so ~11 ms leaves a ~92 ms lane and the lane moves ~1.12x. If the frame falls but
    // the LANE does not, that is displacement again -- this campaign has seen it three times -- and
    // the per-frame counters will localise it rather than leave it inferred.
    {
        let once = |par: bool| -> (f64, f64) {
            // The shared helper's twin takes force_SERIAL, so the sense is inverted: par => false.
            let prev = ft_api::set_gradient_narrow_serial(!par);
            let mut session = FrankenTorchSession::new(ExecutionMode::Strict);
            let x = session
                .tensor_variable_f32(values.clone(), vec![BATCH, IN_CH, H, W], true)
                .expect("leaf x");
            let w = session
                .tensor_variable_f32(weights.clone(), vec![OUT_CH, IN_CH, K, K], true)
                .expect("leaf w");
            let m = session
                .tensor_variable_f32(mask.clone(), vec![BATCH, OUT_CH, H, W], false)
                .expect("leaf mask");
            let lane0 = Instant::now();
            let out = session
                .functional_conv2d(x, w, None, (1, 1), (1, 1))
                .expect("conv2d");
            let scored = session.tensor_mul(out, m).expect("mask multiply");
            let loss = session.tensor_sum(scored).expect("sum");
            let _ = ft_api::take_fuse_bwd_frames_ns();
            let report = session.tensor_backward(loss).expect("backward");
            let lane = lane0.elapsed().as_secs_f64() * 1e3;
            let (d_ns, _) = ft_api::take_fuse_bwd_frames_ns();
            std::hint::black_box(report.gradient(x).expect("grad").len());
            ft_api::set_gradient_narrow_serial(prev);
            (lane, d_ns as f64 / 1e6)
        };
        let mut off_l = Vec::new();
        let mut on_l = Vec::new();
        let mut off_d = Vec::new();
        let mut on_d = Vec::new();
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
            off_l.push(r[0].0.min(r[3].0));
            on_l.push(r[1].0.min(r[2].0));
            off_d.push(r[0].1.min(r[3].1));
            on_d.push(r[1].1.min(r[2].1));
            nulls.push(r[0].0 / r[3].0);
        }
        let median = |v: &mut Vec<f64>| -> f64 {
            v.sort_by(f64::total_cmp);
            if v.is_empty() { f64::NAN } else { v[v.len() / 2] }
        };
        let mut ratios: Vec<f64> = off_l.iter().zip(&on_l).map(|(a, b)| a / b).collect();
        let paired = median(&mut ratios);
        let null = median(&mut nulls.clone());
        let wins = off_l.iter().zip(&on_l).filter(|(o, n)| n < o).count();
        let olm = median(&mut off_l.clone());
        let onm = median(&mut on_l.clone());
        let odm = median(&mut off_d.clone());
        let ondm = median(&mut on_d.clone());
        eprintln!(
            "TRAIN_NARROW lane OFF (serial) {olm:8.3} ms   ON (par_iter) {onm:8.3} ms   |   downcast frame {odm:7.3} -> {ondm:7.3} ms ({:.4}x)",
            odm / ondm
        );
        eprintln!(
            "TRAIN_NARROW   marginal {:.4}x   paired {paired:.4}x   SIGN TEST {wins}/{}   A/A null {null:.4} {}",
            olm / onm,
            off_l.len(),
            if (0.97..=1.03).contains(&null) { "PASS" } else { "FAIL -- discard this row" }
        );
    }

    // PAIRED A/B for the reuse lever, alternating square, per-rep min-of-2, median of per-rep
    // ratios, A/A null from the two same-arm samples of one rep (ledger 274c/275b). Both the LANE
    // and the tensor_mul FRAME are reported: the lever removes a duplicate convolution from the
    // frame, and the lane is what the board would see.
    {
        // Returns (lane, mul, fwd, bwd). The extra two frames exist to LOCALISE the displacement:
        // the reuse lever sheds ~11 ms from `mul` and the lane only moves ~1 ms, so ~9.6 ms
        // reappears somewhere, and "somewhere" was inferred from (lane - mul) rather than
        // measured. On this campaign an inferred residual has already been wrong by 4x (277a), so
        // the neighbouring frames get their own clocks.
        let once = |reuse: bool| -> (f64, f64, f64, f64) {
            let prev = ft_api::set_fuse_conv2d_reuse_f32(reuse);
            let mut session = FrankenTorchSession::new(ExecutionMode::Strict);
            let x = session
                .tensor_variable_f32(values.clone(), vec![BATCH, IN_CH, H, W], true)
                .expect("leaf x");
            let w = session
                .tensor_variable_f32(weights.clone(), vec![OUT_CH, IN_CH, K, K], true)
                .expect("leaf w");
            let m = session
                .tensor_variable_f32(mask.clone(), vec![BATCH, OUT_CH, H, W], false)
                .expect("leaf mask");
            let lane0 = Instant::now();
            let f0 = Instant::now();
            let out = session
                .functional_conv2d(x, w, None, (1, 1), (1, 1))
                .expect("conv2d");
            let fwd = f0.elapsed().as_secs_f64() * 1e3;
            let m0 = Instant::now();
            let scored = session.tensor_mul(out, m).expect("mask multiply");
            let mul = m0.elapsed().as_secs_f64() * 1e3;
            let loss = session.tensor_sum(scored).expect("sum");
            let b0 = Instant::now();
            let report = session.tensor_backward(loss).expect("backward");
            let bwd = b0.elapsed().as_secs_f64() * 1e3;
            let lane = lane0.elapsed().as_secs_f64() * 1e3;
            std::hint::black_box(report.gradient(x).expect("grad").len());
            ft_api::set_fuse_conv2d_reuse_f32(prev);
            (lane, mul, fwd, bwd)
        };
        let mut off_l = Vec::new();
        let mut on_l = Vec::new();
        let mut off_m = Vec::new();
        let mut on_m = Vec::new();
        let mut off_f = Vec::new();
        let mut on_f = Vec::new();
        let mut off_b = Vec::new();
        let mut on_b = Vec::new();
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
            off_l.push(r[0].0.min(r[3].0));
            on_l.push(r[1].0.min(r[2].0));
            off_m.push(r[0].1.min(r[3].1));
            on_m.push(r[1].1.min(r[2].1));
            off_f.push(r[0].2.min(r[3].2));
            on_f.push(r[1].2.min(r[2].2));
            off_b.push(r[0].3.min(r[3].3));
            on_b.push(r[1].3.min(r[2].3));
            nulls.push(r[0].0 / r[3].0);
        }
        let median = |v: &mut Vec<f64>| -> f64 {
            v.sort_by(f64::total_cmp);
            if v.is_empty() { f64::NAN } else { v[v.len() / 2] }
        };
        let mut ratios: Vec<f64> = off_l.iter().zip(&on_l).map(|(a, b)| a / b).collect();
        let paired = median(&mut ratios);
        let null = median(&mut nulls.clone());
        let wins = off_l.iter().zip(&on_l).filter(|(o, n)| n < o).count();
        let olm = median(&mut off_l.clone());
        let onm = median(&mut on_l.clone());
        let omm = median(&mut off_m.clone());
        let onmm = median(&mut on_m.clone());
        eprintln!(
            "TRAIN_AB lane OFF (recompute) {olm:8.3} ms   ON (reuse) {onm:8.3} ms   |   tensor_mul frame {omm:8.3} -> {onmm:8.3} ms ({:.4}x)",
            omm / onmm
        );
        eprintln!(
            "TRAIN_AB   marginal {:.4}x   paired {paired:.4}x   SIGN TEST {wins}/{}   A/A null {null:.4} {}",
            olm / onm,
            off_l.len(),
            if (0.97..=1.03).contains(&null) { "PASS" } else { "FAIL -- discard this row" }
        );
        // WHERE THE SHED TIME WENT. Each neighbouring frame is measured, not inferred.
        let ofm = median(&mut off_f.clone());
        let onf = median(&mut on_f.clone());
        let obm = median(&mut off_b.clone());
        let onb = median(&mut on_b.clone());
        eprintln!(
            "TRAIN_AB   DISPLACEMENT  mul {:+.3} ms | fwd {:+.3} ms | bwd {:+.3} ms | lane {:+.3} ms   (ON minus OFF; mul is the shed, the rest is where it went)",
            onmm - omm,
            onf - ofm,
            onb - obm,
            onm - olm
        );
        eprintln!(
            "TRAIN_AB   frames OFF fwd {ofm:7.3} mul {omm:7.3} bwd {obm:7.3}   ON fwd {onf:7.3} mul {onmm:7.3} bwd {onb:7.3}"
        );
    }
}

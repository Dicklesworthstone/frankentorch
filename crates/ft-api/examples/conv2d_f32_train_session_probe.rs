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
        let (gx_off, gw_off) = run(false);
        let (gx_on, gw_on) = run(true);
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

        let b0 = Instant::now();
        let report = session.tensor_backward(loss).expect("backward");
        let bwd = b0.elapsed().as_secs_f64() * 1e3;

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
    eprintln!("TRAIN   [outside the lane clock] leaf build {t_build:8.3} ms | grad read+checksum {t_grad:8.3} ms");

    // PAIRED A/B for the reuse lever, alternating square, per-rep min-of-2, median of per-rep
    // ratios, A/A null from the two same-arm samples of one rep (ledger 274c/275b). Both the LANE
    // and the tensor_mul FRAME are reported: the lever removes a duplicate convolution from the
    // frame, and the lane is what the board would see.
    {
        let once = |reuse: bool| -> (f64, f64) {
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
            let m0 = Instant::now();
            let scored = session.tensor_mul(out, m).expect("mask multiply");
            let mul = m0.elapsed().as_secs_f64() * 1e3;
            let loss = session.tensor_sum(scored).expect("sum");
            let report = session.tensor_backward(loss).expect("backward");
            let lane = lane0.elapsed().as_secs_f64() * 1e3;
            std::hint::black_box(report.gradient(x).expect("grad").len());
            ft_api::set_fuse_conv2d_reuse_f32(prev);
            (lane, mul)
        };
        let mut off_l = Vec::new();
        let mut on_l = Vec::new();
        let mut off_m = Vec::new();
        let mut on_m = Vec::new();
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
    }
}

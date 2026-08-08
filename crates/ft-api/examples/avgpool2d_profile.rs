//! Where does op work actually go inside the session? — `frankentorch-k1h8g`,
//! `frankentorch-we7ry`.
//!
//! Started as an `avg_pool2d` partition (hence the file name) and grew into the
//! measurement bench for the no-grad borrow vein. It is an INTERNAL partition,
//! never a head-to-head: no PyTorch arm, and nothing it prints is a vs-upstream
//! ratio.
//!
//! # What it times
//!
//! | phase | what it isolates |
//! |---|---|
//! | `raw fwd` / `raw bwd` / `raw fwd+bwd` | the pooling kernels alone, on an already-materialised slice |
//! | `nograd fwd` | the same kernel through the session with the tape skipped — `nograd fwd - raw fwd` is the session wrapper |
//! | `raw fwd f32` / `nograd fwd f32` | the same question one dtype over, with its own control |
//! | `nograd sum` | the fused `tensor_sum(avg_pool2d(x))` shortcut |
//! | `bn raw` / `bn raw cold` / `bn nograd` | BatchNorm2d, with TWO raw controls — see the allocator note |
//! | `tensor_variable` | materialising a fresh 16 MiB node, alone |
//! | `bn1sum raw` / `bn1sum nograd` | `batch_norm1d_sum`, a cheap reduction over a large activation |
//! | `session` + its four steps | the full forward+backward, split fwd / loss_sum / backward / grad fetch |
//!
//! # Two traps this harness exists to avoid, both of which caught me
//!
//! **The grad checksum is teardown, not op work** (`frankentorch-574cu`). It is
//! reported separately and excluded from the op-work total; counting it inflated
//! a published lane ratio.
//!
//! **`min` is the wrong statistic when the arms differ in allocation
//! behaviour.** `bn raw` drops its 16 MiB output each iteration so the allocator
//! recycles warm pages; `bn raw cold` keeps every output alive, which is the
//! session's real pattern. They are indistinguishable at the MINIMUM (1.82 vs
//! 1.81 ms) and differ 3x at the MEDIAN (2.98 vs 8.83 ms). So:
//!
//! - for a BORROW A/B — one phase against itself, a single edit as the only
//!   variable — `min` is fine, because both arms share an allocation pattern;
//! - for a session-minus-kernel DECOMPOSITION, use the median and match the
//!   allocation pattern, or the wrapper term is inflated by the allocator.
//!
//! Run (local; no PyTorch needed):
//! ```text
//! cargo run --release -p ft-api --features fair-alloc --example avgpool2d_profile
//! ```

use std::time::Instant;

use ft_api::FrankenTorchSession;
use ft_core::ExecutionMode;

// Shape lifted verbatim from the lane under investigation.
const N: usize = 8;
const C: usize = 64;
const H: usize = 64;
const W: usize = 64;
const KH: usize = 2;
const KW: usize = 2;
const SH: usize = 2;
const SW: usize = 2;
const OH: usize = H / SH;
const OW: usize = W / SW;

const REPS: usize = 12;

fn seq(n: usize) -> Vec<f64> {
    (0..n).map(|i| ((i % 251) as f64) * 0.001 - 0.12).collect()
}

fn report(label: &str, samples: &mut [f64]) {
    samples.sort_by(f64::total_cmp);
    println!(
        "  {label:<14} min {:7.3} ms   median {:7.3} ms",
        samples[0],
        samples[samples.len() / 2]
    );
}

fn main() {
    let input = seq(N * C * H * W);
    let dout = seq(N * C * OH * OW);

    println!("avg_pool2d profile — [{N},{C},{H},{W}] k({KH},{KW}) s({SH},{SW}) f64");
    println!(
        "input {} elems ({} MiB), output {} elems ({} MiB), reps={REPS}\n",
        input.len(),
        input.len() * 8 / (1024 * 1024),
        dout.len(),
        dout.len() * 8 / (1024 * 1024)
    );

    // --- raw forward kernel ------------------------------------------------
    let mut fwd = Vec::with_capacity(REPS);
    for _ in 0..REPS {
        let started = Instant::now();
        let out = ft_kernel_cpu::avg_pool2d_forward_f64(
            &input, N, C, H, W, KH, KW, OH, OW, SH, SW, 0, 0, H, W, true,
        );
        fwd.push(started.elapsed().as_secs_f64() * 1e3);
        std::hint::black_box(&out);
    }

    // --- raw backward kernel -----------------------------------------------
    let mut bwd = Vec::with_capacity(REPS);
    for _ in 0..REPS {
        let started = Instant::now();
        let dp = ft_kernel_cpu::avg_pool2d_backward_f64(
            &dout, N, C, H, W, KH, KW, OH, OW, SH, SW, 0, 0, H, W, true,
        );
        bwd.push(started.elapsed().as_secs_f64() * 1e3);
        std::hint::black_box(&dp);
    }

    // --- both kernels back to back: the floor a perfect tape would hit ------
    let mut raw_both = Vec::with_capacity(REPS);
    for _ in 0..REPS {
        let started = Instant::now();
        let out = ft_kernel_cpu::avg_pool2d_forward_f64(
            &input, N, C, H, W, KH, KW, OH, OW, SH, SW, 0, 0, H, W, true,
        );
        let dp = ft_kernel_cpu::avg_pool2d_backward_f64(
            &dout, N, C, H, W, KH, KW, OH, OW, SH, SW, 0, 0, H, W, true,
        );
        raw_both.push(started.elapsed().as_secs_f64() * 1e3);
        std::hint::black_box((&out, &dp));
    }

    // --- no-grad session forward: the SAME kernel through the session, but the
    // tape is skipped entirely (functional_avg_pool2d takes its no-grad fast
    // path). session_nograd - raw_fwd is therefore the cost of the session
    // wrapper alone, and fwd_call - session_nograd is what the autograd GRAPH
    // adds on top. Without this split "tape overhead" is one undifferentiated
    // number and a lever cannot be aimed.
    let mut nograd = Vec::with_capacity(REPS);
    for _ in 0..REPS {
        let mut session = FrankenTorchSession::new(ExecutionMode::Strict);
        let x = session
            .tensor_variable(input.clone(), vec![N, C, H, W], false)
            .expect("leaf");
        let started = Instant::now();
        let out = session
            .functional_avg_pool2d(x, (KH, KW), (SH, SW), (0, 0), false, true)
            .expect("avg_pool2d");
        nograd.push(started.elapsed().as_secs_f64() * 1e3);
        std::hint::black_box(out);
    }

    // --- f32 no-grad forward: the same question one dtype over --------------
    // The f32 branch of functional_avg_pool2d is written the same way as the f64
    // one, so if the f64 no-grad path was copying its input the f32 path very
    // likely is too. "Very likely" is not a measurement, hence this phase.
    let input_f32: Vec<f32> = input.iter().map(|v| *v as f32).collect();
    // Raw f32 kernel, so "is the f32 session path also copying its input?" is a
    // measured question and not an argument by analogy with the f64 one.
    let mut raw_f32 = Vec::with_capacity(REPS);
    for _ in 0..REPS {
        let started = Instant::now();
        let out = ft_kernel_cpu::avg_pool2d_forward_f32(
            &input_f32, N, C, H, W, KH, KW, OH, OW, SH, SW, 0, 0, H, W, true,
        );
        raw_f32.push(started.elapsed().as_secs_f64() * 1e3);
        std::hint::black_box(&out);
    }
    let mut nograd_f32 = Vec::with_capacity(REPS);
    for _ in 0..REPS {
        let mut session = FrankenTorchSession::new(ExecutionMode::Strict);
        let x = session
            .tensor_variable_f32(input_f32.clone(), vec![N, C, H, W], false)
            .expect("f32 leaf");
        let started = Instant::now();
        let out = session
            .functional_avg_pool2d(x, (KH, KW), (SH, SW), (0, 0), false, true)
            .expect("avg_pool2d f32");
        nograd_f32.push(started.elapsed().as_secs_f64() * 1e3);
        std::hint::black_box(out);
    }

    // --- no-grad sum shortcut: tensor_sum(avg_pool2d(x)) ---------------------
    // frankentorch-we7ry: the fused *_sum path has its own no-grad branch, which
    // reads the padded activation OWNED. Timed here as a straight before/after of
    // that one edit — no raw-kernel control needed, because the edit is the only
    // variable between the two runs of this phase.
    let mut nograd_sum = Vec::with_capacity(REPS);
    for _ in 0..REPS {
        let mut session = FrankenTorchSession::new(ExecutionMode::Strict);
        let x = session
            .tensor_variable(input.clone(), vec![N, C, H, W], false)
            .expect("leaf");
        let started = Instant::now();
        let out = session
            .functional_avg_pool2d(x, (KH, KW), (SH, SW), (0, 0), false, true)
            .expect("avg_pool2d");
        let s = session.tensor_sum(out).expect("sum");
        nograd_sum.push(started.elapsed().as_secs_f64() * 1e3);
        std::hint::black_box(s);
    }

    // --- no-grad BatchNorm2d: raw kernels vs the session path ---------------
    // frankentorch-we7ry. NOT a vs-PyTorch lane: the interleaved H2H harness has
    // no BatchNorm lane and builds its leaves with requires_grad=true, so it
    // takes the GRAD path and cannot observe a no-grad borrow at all. This is the
    // FT-vs-FT wrapper term, which is the thing the change actually moves.
    let bn_c = C;
    let bn_spatial = H * W;
    let bn_w: Vec<f64> = (0..bn_c).map(|j| 1.0 + (j as f64) * 0.01).collect();
    let bn_b: Vec<f64> = (0..bn_c).map(|j| (j as f64) * 0.02 - 0.3).collect();

    // Two raw controls, because the allocator makes them differ. `bn raw (warm)`
    // drops its 16 MiB output each iteration, so the allocator hands the same
    // block back and the pages are already faulted in. `bn raw (cold)` KEEPS every
    // output alive, forcing a fresh mapping per call — which is the allocation
    // pattern the session path actually has, since it holds the input and output
    // live at once. Comparing the session against the warm control alone
    // overstates the wrapper by exactly this difference.
    let mut bn_raw = Vec::with_capacity(REPS);
    for _ in 0..REPS {
        let started = Instant::now();
        let (mean, var) = ft_kernel_cpu::batch_norm_stats_f64(&input, N, bn_c, bn_spatial);
        let out = ft_kernel_cpu::batch_norm_apply_f64(
            &input,
            &mean,
            &var,
            Some(&bn_w),
            Some(&bn_b),
            N,
            bn_c,
            bn_spatial,
            1e-5,
        );
        bn_raw.push(started.elapsed().as_secs_f64() * 1e3);
        std::hint::black_box(&out);
    }

    let mut bn_raw_cold = Vec::with_capacity(REPS);
    let mut kept: Vec<Vec<f64>> = Vec::with_capacity(REPS);
    for _ in 0..REPS {
        let started = Instant::now();
        let (mean, var) = ft_kernel_cpu::batch_norm_stats_f64(&input, N, bn_c, bn_spatial);
        let out = ft_kernel_cpu::batch_norm_apply_f64(
            &input,
            &mean,
            &var,
            Some(&bn_w),
            Some(&bn_b),
            N,
            bn_c,
            bn_spatial,
            1e-5,
        );
        bn_raw_cold.push(started.elapsed().as_secs_f64() * 1e3);
        kept.push(out);
    }
    std::hint::black_box(&kept);

    let mut bn_nograd = Vec::with_capacity(REPS);
    for _ in 0..REPS {
        let mut session = FrankenTorchSession::new(ExecutionMode::Strict);
        let x = session
            .tensor_variable(input.clone(), vec![N, C, H, W], false)
            .expect("bn leaf");
        let wt = session
            .tensor_variable(bn_w.clone(), vec![bn_c], false)
            .expect("bn weight");
        let bt = session
            .tensor_variable(bn_b.clone(), vec![bn_c], false)
            .expect("bn bias");
        let started = Instant::now();
        let (out, _, _) = session
            .functional_batch_norm2d(x, None, None, Some(wt), Some(bt), true, 0.1, 1e-5)
            .expect("batch_norm2d");
        bn_nograd.push(started.elapsed().as_secs_f64() * 1e3);
        std::hint::black_box(out);
    }

    // --- what IS the BatchNorm2d residual wrapper? --------------------------
    // The borrow removed the input copy but left ~6.4 ms of wrapper on a 2.2 ms
    // kernel. Leading hypothesis: the OUTPUT node. batch_norm_apply_f64 already
    // allocates and fills a 16 MiB Vec (that cost is inside `bn raw`); handing it
    // to `tensor_variable` may pay a SECOND 16 MiB write into tape storage. Timed
    // alone here, on a freshly-built Vec each iteration so the allocation is not
    // recycled — which is what makes it a cold-page cost rather than a memcpy.
    let mut mk_var = Vec::with_capacity(REPS);
    for _ in 0..REPS {
        let fresh = seq(N * C * H * W);
        let mut session = FrankenTorchSession::new(ExecutionMode::Strict);
        let started = Instant::now();
        let t = session
            .tensor_variable(fresh, vec![N, C, H, W], false)
            .expect("materialise");
        mk_var.push(started.elapsed().as_secs_f64() * 1e3);
        std::hint::black_box(t);
    }

    // --- no-grad batch_norm1d_sum: same 16 MiB buffer viewed as [N, C, L] ----
    // frankentorch-we7ry. Again FT-vs-FT: there is no BatchNorm lane in the
    // interleaved H2H harness and its leaves are requires_grad=true.
    let bn1_l = (N * C * H * W) / (N * bn_c);
    let mut bn1_raw = Vec::with_capacity(REPS);
    for _ in 0..REPS {
        let started = Instant::now();
        let (mean, var) = ft_kernel_cpu::batch_norm_stats_f64(&input, N, bn_c, bn1_l);
        let s = ft_kernel_cpu::batch_norm_sum_forward_f64(
            &input,
            &mean,
            &var,
            Some(&bn_w),
            Some(&bn_b),
            N,
            bn_c,
            bn1_l,
            1e-5,
        );
        bn1_raw.push(started.elapsed().as_secs_f64() * 1e3);
        std::hint::black_box(s);
    }

    let mut bn1_nograd = Vec::with_capacity(REPS);
    for _ in 0..REPS {
        let mut session = FrankenTorchSession::new(ExecutionMode::Strict);
        let x = session
            .tensor_variable(input.clone(), vec![N, bn_c, bn1_l], false)
            .expect("bn1 leaf");
        let wt = session
            .tensor_variable(bn_w.clone(), vec![bn_c], false)
            .expect("bn1 weight");
        let bt = session
            .tensor_variable(bn_b.clone(), vec![bn_c], false)
            .expect("bn1 bias");
        let started = Instant::now();
        let (out, _, _) = session
            .functional_batch_norm1d_sum(x, None, None, Some(wt), Some(bt), true, 0.1, 1e-5)
            .expect("batch_norm1d_sum");
        bn1_nograd.push(started.elapsed().as_secs_f64() * 1e3);
        std::hint::black_box(out);
    }

    // --- full session forward+backward, exactly as the H2H harness times it -
    //
    // Also split into its four steps. Callgrind is useless for naming the frame
    // here: it serialises threads, so idle rayon workers spinning in steal loops
    // are counted as retired instructions and `Stealer::steal` swamps the profile
    // (84% Ir) whether or not it costs any wall clock. Direct phase timing is the
    // honest instrument for a rayon-backed path.
    let mut session_ms = Vec::with_capacity(REPS);
    let (mut t_fwd, mut t_sum, mut t_bwd, mut t_grad) = (
        Vec::with_capacity(REPS),
        Vec::with_capacity(REPS),
        Vec::with_capacity(REPS),
        Vec::with_capacity(REPS),
    );
    for _ in 0..REPS {
        let mut session = FrankenTorchSession::new(ExecutionMode::Strict);
        let x = session
            .tensor_variable(input.clone(), vec![N, C, H, W], true)
            .expect("leaf");
        let started = Instant::now();

        let mark = Instant::now();
        let out = session
            .functional_avg_pool2d(x, (KH, KW), (SH, SW), (0, 0), false, true)
            .expect("avg_pool2d");
        t_fwd.push(mark.elapsed().as_secs_f64() * 1e3);

        let mark = Instant::now();
        let loss = session.tensor_sum(out).expect("sum");
        t_sum.push(mark.elapsed().as_secs_f64() * 1e3);

        let mark = Instant::now();
        let report_ = session.tensor_backward(loss).expect("backward");
        t_bwd.push(mark.elapsed().as_secs_f64() * 1e3);

        let mark = Instant::now();
        let checksum = report_.gradient(x).expect("grad").iter().sum::<f64>();
        t_grad.push(mark.elapsed().as_secs_f64() * 1e3);

        session_ms.push(started.elapsed().as_secs_f64() * 1e3);
        std::hint::black_box(checksum);
    }

    report("raw fwd", &mut fwd);
    report("raw bwd", &mut bwd);
    report("raw fwd+bwd", &mut raw_both);
    report("nograd fwd", &mut nograd);
    report("raw fwd f32", &mut raw_f32);
    report("nograd fwd f32", &mut nograd_f32);
    report("nograd sum", &mut nograd_sum);
    report("bn raw", &mut bn_raw);
    report("bn raw cold", &mut bn_raw_cold);
    report("bn nograd", &mut bn_nograd);
    report("tensor_variable", &mut mk_var);
    report("bn1sum raw", &mut bn1_raw);
    report("bn1sum nograd", &mut bn1_nograd);
    println!(
        "  BN1sum wrapper (nograd - raw) = {:.3} ms",
        bn1_nograd[0] - bn1_raw[0]
    );
    println!(
        "  BN wrapper (bn nograd - bn raw) = {:.3} ms\n",
        bn_nograd[0] - bn_raw[0]
    );
    report("session", &mut session_ms);
    println!("\n  session broken down:");
    report("  fwd call", &mut t_fwd);
    report("  sum", &mut t_sum);
    report("  backward", &mut t_bwd);
    report("  grad fetch", &mut t_grad);

    // frankentorch-574cu: the gradient checksum is TEARDOWN, not op work — the
    // PyTorch arm never times its equivalent. Op work is forward + loss_sum +
    // backward, so the overhead accounting below excludes the fetch. Counting it
    // was what inflated this lane's published ratio in the first place.
    let raw_floor = raw_both[0];
    let op_work = t_fwd[0] + t_sum[0] + t_bwd[0];
    println!("\n  op work (fwd+sum+backward, checksum EXCLUDED) = {op_work:.3} ms");
    println!(
        "  raw kernels           = {:.3} ms ({:.0}% of op work)",
        raw_floor,
        raw_floor / op_work * 100.0
    );
    println!(
        "  session+tape overhead = {:.3} ms ({:.0}% of op work)",
        op_work - raw_floor,
        (op_work - raw_floor) / op_work * 100.0
    );
    println!(
        "    of which session wrapper (nograd fwd - raw fwd) = {:.3} ms",
        nograd[0] - fwd[0]
    );
    println!(
        "    of which autograd graph, forward side (fwd call - nograd fwd) = {:.3} ms",
        t_fwd[0] - nograd[0]
    );
    println!(
        "    of which backward side (backward - raw bwd) = {:.3} ms",
        t_bwd[0] - bwd[0]
    );
    println!(
        "\n  grad fetch (teardown, NOT op work) = {:.3} ms — excluded above",
        t_grad[0]
    );
}

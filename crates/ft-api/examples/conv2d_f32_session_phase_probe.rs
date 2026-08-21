//! Where do the 47 ms of session cost go? — `frankentorch-qif1n`.
//!
//! # What is already known
//!
//! A kernels-only twin priced the surround exactly. Both lanes in one h2h invocation, all gates
//! PASS, parity match, `PT(kernels)/PT(session) = 1.012`:
//!
//! ```text
//! conv2d_f32          FT 77.487 ms   PT 25.570 ms   3.03x SLOWER
//! conv2d_f32_kernels  FT 30.428 ms   PT 25.874 ms   1.18x SLOWER
//! session overhead  = 47.06 ms = 60.7% of the lane, 1.55x the kernel time itself
//! ```
//!
//! So our f32 conv2d KERNELS are 1.18x behind PyTorch's whole op, and the session and tape layer
//! adds more than the kernels cost. This probe splits that 47 ms.
//!
//! # What it measures
//!
//! The h2h lane times `forward + loss_sum + backward` as ONE region, which is the right contract
//! against an incumbent but useless for attribution. Here the same three calls are timed separately,
//! plus the two the lane keeps outside its timer (leaf construction, gradient read-back) so nothing
//! is hidden by the choice of region.
//!
//! Both dtypes run the same decomposition at the same batch: the f64 side is the control, because
//! the standing being explained is "our f32 is 1.28x SLOWER per sample than our own f64" and a phase
//! that is slow in BOTH dtypes cannot be what makes f32 the worse of the two.
//!
//! # Honesty about what this is
//!
//! Arm-internal. There is no incumbent and no ratio against PyTorch — this answers "which phase" and
//! nothing about "are we fast". A fresh session per repetition, because the tape is per-session and
//! reusing one would let node accumulation leak across measurements (`project_gmuml_tape_retention`
//! records a session tape that never frees, degrading later ops).
//!
//! MIN over repetitions, per this campaign's estimator convention on a shared host, with one untimed
//! warm-up repetition first (NEGATIVE_EVIDENCE item 247: a first pass can run up to 8x slow).

use std::time::Instant;

use ft_api::FrankenTorchSession;
use ft_core::ExecutionMode;

const CI: usize = 32;
const CO: usize = 32;
const H: usize = 32;
const W: usize = 32;
const K: usize = 3;

#[derive(Default, Clone, Copy)]
struct Phases {
    leaf: f64,
    forward: f64,
    sum: f64,
    backward: f64,
    grad_read: f64,
}

fn main() {
    let batch: usize = std::env::var("PROBE_N")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(160);
    let reps: usize = std::env::var("PROBE_REPS")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(5);

    let x64: Vec<f64> = (0..batch * CI * H * W)
        .map(|i| ((i % 97) as f64) * 0.01 - 0.5)
        .collect();
    let w64: Vec<f64> = (0..CO * CI * K * K)
        .map(|i| ((i % 89) as f64) * 0.01 - 0.4)
        .collect();
    let x32: Vec<f32> = x64.iter().map(|&v| v as f32).collect();
    let w32: Vec<f32> = w64.iter().map(|&v| v as f32).collect();

    println!(
        "conv2d SESSION phase split, f32 vs f64 (frankentorch-qif1n)\n  \
         batch={batch} CI={CI} CO={CO} H={H} W={W} K={K}  rayon_threads={}  reps={reps}\n  \
         arm-internal: no incumbent, no ratio against PyTorch. MIN over reps, fresh session each.",
        rayon::current_num_threads()
    );

    let _ = run_f32(batch, &x32, &w32);
    let _ = run_f64(batch, &x64, &w64);
    sentinel(batch, &x32, &w32, &x64, &w64);
    sentinel(batch, &x32, &w32, &x64, &w64);
    pad_gate_ab(batch, &x32, reps);
    pad_backward_probe(batch, &x32, reps);

    // POOLED-vs-UNPOOLED A/B — `frankentorch-ymhld`, and it must be PAIRED inside one process.
    //
    // The first attempt to price the pool hook compared two separate runs and the host moved 20%
    // between them (load 21 -> 41), which swamps anything a buffer recycle can be worth
    // (`project_buffer_pool_contention`: a pool HIT is worth <=1.06x). Toggling inside one process
    // in palindrome order ON/OFF/OFF/ON makes host drift common-mode and lands it symmetrically on
    // both arms, per item 51.
    let mut p32: Vec<Phases> = Vec::with_capacity(reps);
    let mut p64: Vec<Phases> = Vec::with_capacity(reps);
    let mut pooled: Vec<f64> = Vec::with_capacity(reps * 2);
    let mut unpooled: Vec<f64> = Vec::with_capacity(reps * 2);
    for _ in 0..reps {
        p32.push(run_f32(batch, &x32, &w32));
        p64.push(run_f64(batch, &x64, &w64));
        p64.push(run_f64(batch, &x64, &w64));
        p32.push(run_f32(batch, &x32, &w32));

        ft_core::buffer_pool::set_enabled(true);
        pooled.push(run_f32(batch, &x32, &w32).backward);
        ft_core::buffer_pool::set_enabled(false);
        unpooled.push(run_f32(batch, &x32, &w32).backward);
        unpooled.push(run_f32(batch, &x32, &w32).backward);
        ft_core::buffer_pool::set_enabled(true);
        pooled.push(run_f32(batch, &x32, &w32).backward);
    }
    ft_core::buffer_pool::set_enabled(true);
    let min = |v: &[f64]| v.iter().copied().fold(f64::INFINITY, f64::min);
    println!(
        "\n  BUFFER-POOL A/B on the f32 BACKWARD (paired, palindrome ON/OFF/OFF/ON in ONE process)\n             pooled   min {:8.3} ms\n    unpooled min {:8.3} ms\n             unpooled/pooled {:.3}x   (>1 = the pool helps; a HIT is worth <=1.06x per\n             project_buffer_pool_contention, so treat anything under that as noise)\n             pool stats: {:?}",
        min(&pooled),
        min(&unpooled),
        min(&unpooled) / min(&pooled),
        ft_core::buffer_pool::stats()
    );

    let min_of = |v: &[Phases], pick: fn(&Phases) -> f64| -> f64 {
        v.iter().map(pick).fold(f64::INFINITY, f64::min)
    };
    let rows: [(&str, fn(&Phases) -> f64); 5] = [
        ("leaf build", |p| p.leaf),
        ("forward", |p| p.forward),
        ("loss sum", |p| p.sum),
        ("backward", |p| p.backward),
        ("grad read", |p| p.grad_read),
    ];

    println!(
        "\n  {:<12}{:>12}{:>12}{:>12}",
        "phase", "f32 ms", "f64 ms", "f64/f32"
    );
    let (mut timed32, mut timed64) = (0.0, 0.0);
    for (label, pick) in rows {
        let a = min_of(&p32, pick);
        let b = min_of(&p64, pick);
        println!("  {label:<12}{a:>12.3}{b:>12.3}{:>12.2}", b / a);
        // The h2h lane's timed region is forward + loss sum + backward, and only those.
        if matches!(label, "forward" | "loss sum" | "backward") {
            timed32 += a;
            timed64 += b;
        }
    }
    println!(
        "  {:<12}{timed32:>12.3}{timed64:>12.3}{:>12.2}   <- the h2h lane's timed region",
        "LANE TOTAL",
        timed64 / timed32
    );
    println!(
        "\n  Read the f64/f32 column: a phase below 1.0 is one where f32 is SLOWER than f64, which \
         is the direction that has to be explained. A phase near or above 1.5 is behaving as f32 \
         should (the raw kernels measure 1.53-1.55x)."
    );
}

/// SENTINEL: does the f32 all-ones adjoint actually FIRE at the board's shape, and what does it
/// cost against the f64 entry the session uses?
///
/// The phase split put the whole f32 deficit in the backward. Reading the source explains only half
/// of it: both dtypes share `conv2d_ones_dout_route`, and the board's shape (ph=34, kh=kw=3, oh=32,
/// stride 1) selects `ThreeByThreeStride1` for both — so the f32 fast path SHOULD fire. What
/// differs is the entry: ft-api calls `conv2d_backward_f32_ones_dout_from_f64_grad` as a
/// pre-check on the tape's f64 gradient, while the f64 side calls `conv2d_backward_masked_f64` and
/// lets it dispatch internally.
///
/// `feedback_sentinel_before_fixing`: prove which path EXECUTES before changing anything — source
/// reading gave three confident wrong answers on one bug. `Some`/`None` here is that proof, and the
/// timing beside it says whether firing is enough.
fn sentinel(batch: usize, x32: &[f32], w32: &[f32], x64: &[f64], w64: &[f64]) {
    let ph = H + 2;
    let pw = W + 2;
    let mut padded32 = vec![0.0f32; batch * CI * ph * pw];
    let mut padded64 = vec![0.0f64; batch * CI * ph * pw];
    for bc in 0..batch * CI {
        for row in 0..H {
            let from = bc * H * W + row * W;
            let to = bc * ph * pw + (row + 1) * pw + 1;
            padded32[to..to + W].copy_from_slice(&x32[from..from + W]);
            padded64[to..to + W].copy_from_slice(&x64[from..from + W]);
        }
    }
    // The tape hands the backward an f64 gradient in BOTH dtypes: the f32 entry takes `dout_f64`
    // by design, deciding in f64 because `1.0f64 as f32` is exactly `1.0f32`.
    let dout64 = vec![1.0f64; batch * CO * H * W];

    let t = Instant::now();
    let fired = ft_kernel_cpu::conv2d_backward_f32_ones_dout_from_f64_grad(
        &dout64, &padded32, w32, batch, CI, ph, pw, K, K, H, W, 1, 1, CO, false,
    );
    let f32_ms = ms(t);
    let verdict = if fired.is_some() {
        "FIRED"
    } else {
        "DID NOT FIRE"
    };
    std::hint::black_box(&fired);

    let t = Instant::now();
    let out64 = ft_kernel_cpu::conv2d_backward_masked_f64(
        &dout64,
        &padded64,
        w64,
        batch,
        CI,
        ph,
        pw,
        K,
        K,
        H,
        W,
        1,
        1,
        CO,
        [true, true, false],
    );
    let f64_ms = ms(t);
    std::hint::black_box(&out64);

    println!(
        "\n  SENTINEL (what ft-api actually calls, all-ones adjoint, same shape):\n             f32 ones-entry  {f32_ms:8.3} ms   {verdict}\n             f64 masked-entry{f64_ms:8.3} ms   (dispatches to its adjoint internally)\n             f64/f32 {:.2}x   (>1 = f32 faster; the raw generic kernels measure 1.5x)",
        f64_ms / f32_ms
    );
}

/// PRICE THE `tensor_pad` GRAD GATE — `frankentorch-wb7vt`.
///
/// `tensor_pad`'s block-copy fast path is gated on `!tensor_requires_grad(input)`, so an inference
/// pad gets a memset-plus-row-copy while a TRAINING pad falls through to a per-output-element
/// O(ndim) coordinate decode — which that function's own comment measures at 3.7x slower than
/// torch. Every conv2d on the board's training route pads an input that requires grad.
///
/// That is a mechanism from READING, and on `frankentorch-qif1n` five such mechanisms produced four
/// refutations. So it gets priced before anyone acts on it: same tensor, same shape, same padding,
/// same process, `requires_grad` the ONLY difference, palindrome GRAD/NOGRAD/NOGRAD/GRAD so host
/// drift is common-mode (item 51).
///
/// The pad is timed alone rather than inside conv2d, because conv2d's grad and no-grad routes
/// differ in more than the pad and the comparison would not isolate the gate.
fn pad_gate_ab(batch: usize, values: &[f32], reps: usize) {
    let shape = vec![batch, CI, H, W];
    let pad = [1usize, 1, 1, 1];
    let mut grad_ms: Vec<f64> = Vec::new();
    let mut nograd_ms: Vec<f64> = Vec::new();
    let once = |requires_grad: bool| -> f64 {
        let mut session = FrankenTorchSession::new(ExecutionMode::Strict);
        let x = session
            .tensor_variable_f32(values.to_vec(), shape.clone(), requires_grad)
            .expect("pad leaf");
        let t = Instant::now();
        let padded = session.tensor_pad(x, &pad, 0.0).expect("pad");
        let elapsed = ms(t);
        std::hint::black_box(padded);
        elapsed
    };
    let _ = once(true);
    let _ = once(false);
    for _ in 0..reps {
        grad_ms.push(once(true));
        nograd_ms.push(once(false));
        nograd_ms.push(once(false));
        grad_ms.push(once(true));
    }
    let min = |v: &[f64]| v.iter().copied().fold(f64::INFINITY, f64::min);
    println!(
        "\n  tensor_pad GRAD-GATE A/B (frankentorch-wb7vt), [{batch},{CI},{H},{W}] pad 1 on each \
         spatial side\n    \
         requires_grad=true   min {:8.3} ms   <- per-element O(ndim) decode\n    \
         requires_grad=false  min {:8.3} ms   <- block-copy fast path\n    \
         grad/nograd {:.2}x   (1.0 = the gate costs nothing; >1 = training pays for it)",
        min(&grad_ms),
        min(&nograd_ms),
        min(&grad_ms) / min(&nograd_ms)
    );
}

/// ISOLATE THE PAD'S BACKWARD — `frankentorch-ymhld`.
///
/// The backward residue (backward phase minus the kernel entry) is ~30 ms at batch 160 in BOTH
/// dtypes, and a batch sweep showed it scales 2.90x for a 2.50x batch increase — so it is
/// per-element data movement, not per-call bookkeeping. The crop is the named suspect: conv2d's
/// input is padded on the tape, so the backward must crop `dpadded` (5.9M elements) back to the
/// input extent.
///
/// This times a pad-only tape: leaf -> pad -> sum -> backward. Nothing else is in that backward, so
/// whatever it costs IS the crop plus the sum's adjoint, and the sum's adjoint is a fill.
///
/// Reported beside the conv2d backward residue from the same run, because the question is not "is
/// the crop slow" in the abstract but "is the crop the 30 ms". Two batches, so it can be checked
/// against the same 2.5x scaling the residue showed.
fn pad_backward_probe(batch: usize, values: &[f32], reps: usize) {
    let shape = vec![batch, CI, H, W];
    let pad = [1usize, 1, 1, 1];
    let once = || -> f64 {
        let mut session = FrankenTorchSession::new(ExecutionMode::Strict);
        let x = session
            .tensor_variable_f32(values.to_vec(), shape.clone(), true)
            .expect("pad leaf");
        let padded = session.tensor_pad(x, &pad, 0.0).expect("pad");
        let loss = session.tensor_sum(padded).expect("sum");
        let t = Instant::now();
        let report = session.tensor_backward(loss).expect("backward");
        let elapsed = ms(t);
        std::hint::black_box(report.gradient(x).map(<[f64]>::len));
        elapsed
    };
    let _ = once();
    let mut v: Vec<f64> = (0..reps).map(|_| once()).collect();
    v.sort_by(f64::total_cmp);
    println!(
        "\n  PAD-ONLY BACKWARD (frankentorch-ymhld), [{batch},{CI},{H},{W}] pad 1 each spatial \
         side\n    leaf -> pad -> sum -> backward, backward timed alone: min {:8.3} ms (n={})\n    \
         Compare against the conv2d backward RESIDUE at this batch. If the crop is the residue \
         these are the same order; if it is a fraction, the residue is elsewhere.",
        v[0],
        v.len()
    );
}

fn run_f32(batch: usize, values: &[f32], weights: &[f32]) -> Phases {
    let mut p = Phases::default();
    let mut session = FrankenTorchSession::new(ExecutionMode::Strict);
    let t = Instant::now();
    let x = session
        .tensor_variable_f32(values.to_vec(), vec![batch, CI, H, W], true)
        .expect("leaf");
    let w = session
        .tensor_variable_f32(weights.to_vec(), vec![CO, CI, K, K], false)
        .expect("weight");
    p.leaf = ms(t);
    let t = Instant::now();
    let out = session
        .functional_conv2d(x, w, None, (1, 1), (1, 1))
        .expect("conv2d");
    p.forward = ms(t);
    let t = Instant::now();
    let loss = session.tensor_sum(out).expect("sum");
    p.sum = ms(t);
    let t = Instant::now();
    let report = session.tensor_backward(loss).expect("backward");
    p.backward = ms(t);
    let t = Instant::now();
    let g = report.gradient(x).expect("grad");
    let checksum: f64 = g.iter().map(|v| v.abs()).sum();
    p.grad_read = ms(t);
    std::hint::black_box(checksum);
    p
}

fn run_f64(batch: usize, values: &[f64], weights: &[f64]) -> Phases {
    let mut p = Phases::default();
    let mut session = FrankenTorchSession::new(ExecutionMode::Strict);
    let t = Instant::now();
    let x = session
        .tensor_variable(values.to_vec(), vec![batch, CI, H, W], true)
        .expect("leaf");
    let w = session
        .tensor_variable(weights.to_vec(), vec![CO, CI, K, K], false)
        .expect("weight");
    p.leaf = ms(t);
    let t = Instant::now();
    let out = session
        .functional_conv2d(x, w, None, (1, 1), (1, 1))
        .expect("conv2d");
    p.forward = ms(t);
    let t = Instant::now();
    let loss = session.tensor_sum(out).expect("sum");
    p.sum = ms(t);
    let t = Instant::now();
    let report = session.tensor_backward(loss).expect("backward");
    p.backward = ms(t);
    let t = Instant::now();
    let g = report.gradient(x).expect("grad");
    let checksum: f64 = g.iter().map(|v| v.abs()).sum();
    p.grad_read = ms(t);
    std::hint::black_box(checksum);
    p
}

fn ms(started: Instant) -> f64 {
    started.elapsed().as_secs_f64() * 1_000.0
}

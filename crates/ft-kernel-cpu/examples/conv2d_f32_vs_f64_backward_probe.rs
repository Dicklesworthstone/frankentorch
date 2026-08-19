//! Is the f32 conv2d backward paying for a SERIAL transpose-GEMM? — `frankentorch-qif1n`.
//!
//! # The question this exists to answer
//!
//! `conv2d_f32_masked` is a certified 7.34x loss against PyTorch (FT 185.434 ms vs PT 25.273 ms, 16
//! threads, both A/A nulls PASS, parity match). Reading the two GEMMs side by side gives a
//! mechanism: `gemm::dgemm_tb_scaled` dispatches three ways — column-parallel (item 119), a 2-D
//! tile path (item 130), else serial — while `gemm::sgemm_tb_scaled` is ONE unconditional serial
//! `sgemm_mm` call at every shape. `sgemm_tb` is the f32 weight-gradient GEMM.
//!
//! That is a mechanism, not a diagnosis, and the bead says so. This probe is the cheapest
//! discriminator I can build from OUTSIDE the crate: `gemm` is a private module, but
//! `conv2d_backward_f32` and `conv2d_backward_f64` are both public and take the SAME arguments in
//! the same order, differing only in dtype.
//!
//! # What the two outcomes mean
//!
//! f32 moves half the bytes of f64 for identical arithmetic shape, so on a bandwidth-bound kernel
//! with equal parallelism the f32 side should be FASTER — not equal.
//!
//!   * f32 clearly faster than f64  → the f32 path is not obviously handicapped; the 7.34x lives
//!     somewhere this probe does not look, and the serial-GEMM story needs demoting.
//!   * f32 at or worse than f64     → the f32 path is doing the same work with less parallelism,
//!     which is exactly what a serial `sgemm_tb` against a parallel `dgemm_tb` predicts.
//!
//! It CANNOT attribute a share of the 7.34x to the GEMM — for that the module would have to be
//! reachable, or the crate would have to grow a phase counter. It can say whether the hypothesis
//! survives contact, which is what the bead is missing.
//!
//! # Why the arms are interleaved
//!
//! The host this runs on is shared and frequently at run-queue 50+. An A/B/B/A order per repetition
//! makes contention common-mode: both arms see the same neighbours, and the first arm of the pair
//! is not systematically the cold one (NEGATIVE_EVIDENCE item 247 measured a first-pass penalty of
//! up to 8x at sweep granularity, so ordering is not a detail). Times are reported as MIN over
//! repetitions, which is the estimator this campaign uses for a contended host, with the median
//! alongside so a divergence between them is visible rather than hidden.
//!
//! Arm-internal by construction: there is no incumbent here and no ratio against PyTorch. This
//! answers "which of our two dtypes is handicapped", not "are we fast".

use std::time::Instant;

/// `[N, C, H, W]` with a 3x3 stride-1 pad-1 kernel — the shape family the `conv2d_f32*` board lanes
/// use, at a batch that keeps a single call in the tens of milliseconds so the timer is not
/// resolution-bound.
/// Batch is env-tunable (`PROBE_N`) because the first run of this probe was at 32 while the board's
/// `conv2d_f32` lane runs 160, and "you measured the wrong shape" is the first objection any
/// negative result here deserves. Defaults to the board's batch so the two agree unless told
/// otherwise.
fn batch() -> usize {
    std::env::var("PROBE_N")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(160)
}

const CI: usize = 32;
const CO: usize = 32;
const H: usize = 32;
const W: usize = 32;
const K: usize = 3;
const PAD: usize = 1;

fn main() {
    let reps: usize = std::env::var("PROBE_REPS")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(5);

    let n = batch();
    let oh = H + 2 * PAD - K + 1;
    let ow = W + 2 * PAD - K + 1;
    let ph = H + 2 * PAD;
    let pw = W + 2 * PAD;

    // Deterministic, bounded, and identical between dtypes up to the f32 rounding of the same
    // decimal values — the arms must differ in dtype, not in data.
    let seq64 = |n: usize| -> Vec<f64> { (0..n).map(|i| ((i % 97) as f64) * 0.01 - 0.5).collect() };
    let dout64 = seq64(n * CO * oh * ow);
    let padded64 = seq64(n * CI * ph * pw);
    let weight64 = seq64(CO * CI * K * K);
    let dout32: Vec<f32> = dout64.iter().map(|&v| v as f32).collect();
    let padded32: Vec<f32> = padded64.iter().map(|&v| v as f32).collect();
    let weight32: Vec<f32> = weight64.iter().map(|&v| v as f32).collect();

    println!(
        "conv2d backward f32-vs-f64, same shape, same data (frankentorch-qif1n)\n  \
         N={n} CI={CI} CO={CO} H={H} W={W} K={K} pad={PAD}  oh={oh} ow={ow}\n  \
         rayon_threads={}  reps={reps}  (A/B/B/A per rep; MIN and median reported)",
        rayon::current_num_threads()
    );

    let mut t32: Vec<f64> = Vec::with_capacity(reps * 2);
    let mut t64: Vec<f64> = Vec::with_capacity(reps * 2);

    // One untimed warm-up of each arm: item 247 measured the first pass of a sweep running up to
    // 8x slow, and a probe that charges that to whichever dtype happens to go first is not a probe.
    let _ = run_f32(n, &dout32, &padded32, &weight32, ph, pw, oh, ow);
    let _ = run_f64(n, &dout64, &padded64, &weight64, ph, pw, oh, ow);

    let mut f32_fwd: Vec<f64> = Vec::with_capacity(reps * 2);
    let mut f64_fwd: Vec<f64> = Vec::with_capacity(reps * 2);
    let _ = run_fwd_f32(n, &padded32, &weight32, ph, pw, oh, ow);
    let _ = run_fwd_f64(n, &padded64, &weight64, ph, pw, oh, ow);

    for _ in 0..reps {
        t32.push(run_f32(n, &dout32, &padded32, &weight32, ph, pw, oh, ow));
        t64.push(run_f64(n, &dout64, &padded64, &weight64, ph, pw, oh, ow));
        t64.push(run_f64(n, &dout64, &padded64, &weight64, ph, pw, oh, ow));
        t32.push(run_f32(n, &dout32, &padded32, &weight32, ph, pw, oh, ow));

        f32_fwd.push(run_fwd_f32(n, &padded32, &weight32, ph, pw, oh, ow));
        f64_fwd.push(run_fwd_f64(n, &padded64, &weight64, ph, pw, oh, ow));
        f64_fwd.push(run_fwd_f64(n, &padded64, &weight64, ph, pw, oh, ow));
        f32_fwd.push(run_fwd_f32(n, &padded32, &weight32, ph, pw, oh, ow));
    }

    let report = |label: &str, mut v: Vec<f64>| -> (f64, f64) {
        v.sort_by(f64::total_cmp);
        let min = v[0];
        let med = v[v.len() / 2];
        println!(
            "  {label:<4} min {min:8.3} ms   median {med:8.3} ms   n={}",
            v.len()
        );
        (min, med)
    };
    println!("  -- BACKWARD --");
    let (min32, med32) = report("f32", t32);
    let (min64, med64) = report("f64", t64);
    println!("  -- FORWARD --");
    let (fmin32, _) = report("f32", f32_fwd);
    let (fmin64, _) = report("f64", f64_fwd);
    println!(
        "  forward f64/f32 by MIN {:.2}x   (>1 means f32 is faster)",
        fmin64 / fmin32
    );

    println!(
        "\n  f64/f32 by MIN    {:.2}x   (>1 means f32 is faster, which is what half the bytes \
         should buy)\n  f64/f32 by MEDIAN {:.2}x",
        min64 / min32,
        med64 / med32
    );
    if min64 / min32 >= 1.30 {
        println!(
            "  READING: f32 is clearly faster. The serial-sgemm_tb story does NOT survive this \
             probe as the dominant cause, and frankentorch-qif1n should be re-aimed."
        );
    } else {
        println!(
            "  READING: f32 is NOT buying what half the bytes should buy. Consistent with the f32 \
             path doing the same work at less parallelism -- which is what a serial sgemm_tb \
             against a three-way-dispatching dgemm_tb predicts. NOT a share of the 7.34x: this \
             probe cannot attribute one, only say the hypothesis survives."
        );
    }
}

/// FORWARD arm, added after the backward result came back the wrong way for the hypothesis.
///
/// The backward probe said f32 is 1.62x FASTER than f64, yet the board's f32 conv2d LANE is 1.28x
/// slower per sample than the f64 lane (both certified). Those two facts can only be reconciled if
/// the f32 cost sits outside the backward, and the forward is the first place to look because it is
/// the other half of the same kernel pair and is equally reachable from here.
fn run_fwd_f32(
    n: usize,
    padded: &[f32],
    weight: &[f32],
    ph: usize,
    pw: usize,
    oh: usize,
    ow: usize,
) -> f64 {
    let started = Instant::now();
    let out = ft_kernel_cpu::conv2d_forward_f32(
        padded, weight, None, n, CI, ph, pw, K, K, oh, ow, 1, 1, CO,
    );
    let elapsed = started.elapsed().as_secs_f64() * 1_000.0;
    std::hint::black_box(&out);
    elapsed
}

fn run_fwd_f64(
    n: usize,
    padded: &[f64],
    weight: &[f64],
    ph: usize,
    pw: usize,
    oh: usize,
    ow: usize,
) -> f64 {
    let started = Instant::now();
    let out = ft_kernel_cpu::conv2d_forward_f64(
        padded, weight, None, n, CI, ph, pw, K, K, oh, ow, 1, 1, CO,
    );
    let elapsed = started.elapsed().as_secs_f64() * 1_000.0;
    std::hint::black_box(&out);
    elapsed
}

fn run_f32(
    n: usize,
    dout: &[f32],
    padded: &[f32],
    weight: &[f32],
    ph: usize,
    pw: usize,
    oh: usize,
    ow: usize,
) -> f64 {
    let started = Instant::now();
    let out = ft_kernel_cpu::conv2d_backward_f32(
        dout, padded, weight, n, CI, ph, pw, K, K, oh, ow, 1, 1, CO, false,
    );
    let elapsed = started.elapsed().as_secs_f64() * 1_000.0;
    // Consume the result so the call cannot be elided; `feedback_insitu_over_standalone` records a
    // standalone ladder inverting in situ, and a dead-code-eliminated arm is the crudest version of
    // that error.
    std::hint::black_box(&out);
    elapsed
}

fn run_f64(
    n: usize,
    dout: &[f64],
    padded: &[f64],
    weight: &[f64],
    ph: usize,
    pw: usize,
    oh: usize,
    ow: usize,
) -> f64 {
    let started = Instant::now();
    let out = ft_kernel_cpu::conv2d_backward_f64(
        dout, padded, weight, n, CI, ph, pw, K, K, oh, ow, 1, 1, CO, false,
    );
    let elapsed = started.elapsed().as_secs_f64() * 1_000.0;
    std::hint::black_box(&out);
    elapsed
}

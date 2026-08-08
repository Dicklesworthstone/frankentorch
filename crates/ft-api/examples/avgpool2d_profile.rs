//! Where does `avg_pool2d`'s op work actually go? — `frankentorch-k1h8g`.
//!
//! The lane measures 4-6x slower than PyTorch on `[8,64,64,64]` k(2,2) s(2,2)
//! f64, forward+backward, leaf built outside the timer. The f64 path is already
//! a single fused custom op (one windowed-mean forward, a geometry-only
//! backward, no saved input), so "compose overhead" is not the explanation and
//! the cost has to be partitioned before any lever is chosen.
//!
//! This splits the same workload into the pieces that can actually be blamed:
//!
//!   raw fwd      `avg_pool2d_forward_f64` alone, on an already-materialised slice
//!   raw bwd      `avg_pool2d_backward_f64` alone
//!   raw fwd+bwd  the two kernels back to back — the floor a perfect tape would hit
//!   session      the full `FrankenTorchSession` forward+backward the harness times
//!
//! `session - (raw fwd + raw bwd)` is tape/`apply_function` overhead: input
//! clones, node allocation, gradient plumbing. If that term dominates, the lever
//! is in ft-api; if the raw kernels dominate, it is in ft-kernel-cpu. Reporting
//! both stops a lever being aimed at the wrong crate.
//!
//! Run (local; no PyTorch needed — this is an internal partition, not a H2H):
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
    report("session", &mut session_ms);
    println!("\n  session broken down:");
    report("  fwd call", &mut t_fwd);
    report("  sum", &mut t_sum);
    report("  backward", &mut t_bwd);
    report("  grad fetch", &mut t_grad);

    let raw_floor = raw_both[0];
    let session_min = session_ms[0];
    println!(
        "\n  tape/apply_function overhead = session - raw = {:.3} ms ({:.0}% of session)",
        session_min - raw_floor,
        (session_min - raw_floor) / session_min * 100.0
    );
    println!(
        "  raw kernels = {:.0}% of session",
        raw_floor / session_min * 100.0
    );
}

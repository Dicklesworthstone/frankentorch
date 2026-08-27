//! Deterministic instruction-count driver for the SVD lane.
//!
//! WHY THIS EXISTS. The SVD loss (2.40x at n=512, 3.10x at n=1024 vs PyTorch) has been
//! measured only by wall clock, and this host has refused window after window: peer
//! criterion/PyTorch/callgrind runs put loadavg between 15 and 99 for hours, and the
//! standing A/A null on this lane reads 1.049-1.203x, which is wider than several of the
//! effects people want to claim. Retired instructions do not care about load. They are a
//! deterministic count of work performed, so unlike a wall-time ratio they are comparable
//! ACROSS runs and across a busy host.
//!
//! THE ESTIMATOR IS A DIFFERENCE, not a total. Run this at `--iters 1` and at `--iters N`;
//! per-iteration instructions are `(I_N - I_1) / (N - 1)`. That cancels process startup,
//! the allocator's first touch, the fixture build, and the binary's own load - none of which
//! are the SVD. Reporting a raw total instead would fold all of that into the figure and
//! flatter or damn whichever side has the cheaper startup (PyTorch's is enormous: `import
//! torch` alone is billions of instructions).
//!
//! SINGLE THREAD BY DEFAULT, and that is not a limitation to apologise for. A retired-
//! instruction count over a thread pool includes the pool's spinning, which varies with host
//! load and would reintroduce exactly the nondeterminism this driver exists to escape. One
//! thread on each side measures WORK. It deliberately does not measure how well either side
//! parallelises - that is a separate question and a wall-clock one.
//!
//! The matrix is built from a fixed integer recurrence, not an RNG: the same bits on every
//! run and on both arms, so a count difference is a code difference.

use ft_api::FrankenTorchSession;
use ft_core::ExecutionMode;

fn arg(name: &str, default: usize) -> usize {
    let mut it = std::env::args().skip(1);
    while let Some(a) = it.next() {
        if a == name {
            return it
                .next()
                .and_then(|v| v.parse().ok())
                .unwrap_or_else(|| panic!("{name} needs a number"));
        }
    }
    default
}

/// Deterministic, well-conditioned-enough fixture. A fixed LCG rather than `rand` so the
/// values are identical on every run, every host and every build.
fn fixture_bumped(n: usize, bump: f64) -> Vec<f64> {
    let mut state: u64 = 0x2545_F491_4F6C_DD1D;
    let mut out = Vec::with_capacity(n * n);
    for _ in 0..n * n {
        state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        // top 26 bits -> [-1, 1), exactly representable, no transcendental in the fixture
        let v = ((state >> 38) as f64) / (f64::from(1u32 << 25)) - 1.0;
        out.push(v);
    }
    // Push the diagonal up so the matrix is not near-singular; a rank-deficient input can
    // change how many QR sweeps the bidiagonal solver runs, which would make the count
    // depend on the fixture rather than on the code.
    for i in 0..n {
        out[i * n + i] += bump;
    }
    out
}

fn main() {
    let n = arg("--n", 512);
    let iters = arg("--iters", 1);
    let values_only = std::env::var("FT_VALUES_ONLY").is_ok();

    let bump = arg("--bump", 4) as f64;
    let data = fixture_bumped(n, bump);
    let mut checksum = 0.0f64;

    for _ in 0..iters {
        let mut session = FrankenTorchSession::new(ExecutionMode::Strict);
        let a = session
            .tensor_variable(data.clone(), vec![n, n], false)
            .expect("fixture tensor");
        if values_only {
            let s = session.tensor_linalg_svdvals(a).expect("svdvals");
            checksum += session.tensor_values(s).expect("values")[0];
        } else {
            let (_u, s, _vh) = session.tensor_linalg_svd(a, false).expect("svd");
            checksum += session.tensor_values(s).expect("values")[0];
        }
    }

    // Consume the result so nothing above is dead-code-eliminated. LLVM will happily delete
    // a factorisation whose output is never read, and that failure mode looks exactly like a
    // fast implementation.
    println!("n={n} iters={iters} checksum={checksum:.12e}");

    // Read the SAME phase counters the h2h harness quotes, on THIS path. The harness reports
    // the sweep at ~0.26 ms while a perf instruction profile of this driver puts 78% of
    // retired instructions inside the function that timer wraps; those cannot both describe
    // the same work, and printing both instruments from one run is what tells them apart.
    if std::env::var("FT_PHASES").is_ok() {
        let (reduction, form_pq, sweep) = ft_kernel_cpu::svd_reduction_sweep_ns_take();
        let (dl_qr, dl_gemm, dl_assemble) = ft_kernel_cpu::svd_deferred_left_phase_ns_take();
        let hits = ft_kernel_cpu::svd_deferred_left_hits_take();
        let ms = |v: u64| v as f64 / 1e6;
        println!(
            "phases_ms reduction={:.3} form_pq={:.3} sweep={:.3} | deferred_left qr={:.3} \
             gemm={:.3} assemble={:.3} hits={hits}",
            ms(reduction),
            ms(form_pq),
            ms(sweep),
            ms(dl_qr),
            ms(dl_gemm),
            ms(dl_assemble)
        );
    }
}

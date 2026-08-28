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

/// The h2h lane's DEFAULT fixture, `_mk(n, False)` — reproduced here bit-for-bit so the
/// instruction ratio can be read on the exact matrix the banked wall-clock rows were taken on.
///
/// It is `3*I` plus a low-rank modular term: at n=512, 495 of its 512 singular values are
/// exactly 3.0. `frankentorch-gqmws`.
fn fixture_mk(n: usize) -> Vec<f64> {
    let mut a = vec![0.0_f64; n * n];
    for r in 0..n {
        for c in 0..n {
            let v = ((((r + 2) * (c + 3)) % 17) as f64 - 8.0) * 0.05;
            a[r * n + c] = v + if r == c { 3.0 } else { 0.0 };
        }
    }
    a
}

/// The h2h lane's `FT_FIXTURE=generic` matrix, reproduced bit-for-bit (integer arithmetic, a
/// power-of-two scale, exact `+16` diagonal). 512 of 512 distinct singular values, cond 97.4.
fn fixture_generic(n: usize) -> Vec<f64> {
    let mut a = vec![0.0_f64; n * n];
    for r in 0..n {
        for c in 0..n {
            let h = (r * 73 + c * 151 + (r * c) % 257) % 2048;
            a[r * n + c] = (h as f64) / 2048.0 - 1.0 + if r == c { 16.0 } else { 0.0 };
        }
    }
    a
}

/// Deterministic, well-conditioned-enough fixture. A fixed LCG rather than `rand` so the
/// values are identical on every run, every host and every build.
fn fixture_bumped(n: usize, bump: f64) -> Vec<f64> {
    let mut state: u64 = 0x2545_F491_4F6C_DD1D;
    let mut out = Vec::with_capacity(n * n);
    for _ in 0..n * n {
        state = state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
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

    // FT_FIXTURE mirrors the h2h harness's own selector by name, so the two tools cannot drift
    // apart on which matrix they mean. Default stays the LCG fixture the earlier rows used.
    let kind = std::env::var("FT_FIXTURE").unwrap_or_else(|_| "lcg".to_owned());
    let bump = arg("--bump", 4) as f64;
    let data = match kind.as_str() {
        "mk" => fixture_mk(n),
        "generic" => fixture_generic(n),
        _ => fixture_bumped(n, bump),
    };
    eprintln!("fixture={kind}");

    // FT_REPLAY=0 selects the pre-i040z row-major replay for both the SVD U/V stream and the
    // eigh tql2 stream. Instruction counts are load-immune, so a same-binary A/B across this
    // toggle is valid on a busy host where a wall-clock one is not.
    if let Ok(v) = std::env::var("FT_REPLAY") {
        let transposed = v.trim() != "0";
        ft_kernel_cpu::set_svd_replay_transposed(transposed);
        eprintln!("replay_transposed={transposed}");
    }

    // FT_OP=gemm measures what OUR OWN GEMM achieves at this size, so the blocking ceiling for
    // the BLAS-2 phases is a measured number rather than an invented one.
    //
    // eigh's backtransform does ~4n^3/3 flops (a GEMV plus a rank-1 update per step, i^2
    // multiply-adds each) and the certified single-window figure is 17.839 ms at n=512, i.e.
    // ~10.0 GFLOP/s. The reduce is the same BLAS-2 shape. What blocking can buy is bounded by
    // the rate the GEMM path actually reaches on this machine, in this build, at this size —
    // which is what this measures. Same public route the phases use internally.
    if std::env::var("FT_OP").map(|v| v == "gemm").unwrap_or(false) {
        // FT_GEMM_SHAPE="m,k,n" measures a RECTANGULAR shape instead of n x n x n. The blocked
        // backtransform's panel GEMMs are skinny — (nb, m, m), (nb, nb, m) and (m, nb, m) with
        // nb = 8..64 against m up to 512 — and the standing hypothesis for why that lever came
        // out flat (9c3b3e1b) is that skinny GEMMs do not reach the density a square one does.
        // That hypothesis was recorded as UNTESTED; this measures it.
        let shape: Vec<usize> = std::env::var("FT_GEMM_SHAPE")
            .ok()
            .map(|s| s.split(',').filter_map(|t| t.trim().parse().ok()).collect())
            .unwrap_or_default();
        let (gm, gk, gn) = if shape.len() == 3 {
            (shape[0], shape[1], shape[2])
        } else {
            (n, n, n)
        };
        let mut session = FrankenTorchSession::new(ExecutionMode::Strict);
        let lhs: Vec<f64> = (0..gm * gk).map(|i| data[i % data.len()]).collect();
        let rhs: Vec<f64> = (0..gk * gn).map(|i| data[i % data.len()]).collect();
        let a = session
            .tensor_variable(lhs, vec![gm, gk], false)
            .expect("gemm lhs");
        let b = session
            .tensor_variable(rhs, vec![gk, gn], false)
            .expect("gemm rhs");
        let _ = session.tensor_matmul(a, b).expect("warm rect matmul");
        let rounds_r = iters.max(5);
        let mut best_r = f64::INFINITY;
        for _ in 0..rounds_r {
            let started = std::time::Instant::now();
            let out = session.tensor_matmul(a, b).expect("rect matmul");
            let elapsed = started.elapsed().as_secs_f64();
            std::hint::black_box(&out);
            best_r = best_r.min(elapsed);
        }
        #[allow(clippy::cast_precision_loss)]
        let rflops = 2.0 * (gm as f64) * (gk as f64) * (gn as f64);
        println!(
            "gemm m={gm} k={gk} n={gn} min={:.4} ms  {:.1} GFLOP/s  (min of {rounds_r})",
            best_r * 1e3,
            rflops / best_r / 1e9
        );
        return;
        #[allow(unreachable_code)]
        {
            let _ = (a, b);
        }
    }
    if false {
        let mut session = FrankenTorchSession::new(ExecutionMode::Strict);
        let a = session
            .tensor_variable(data.clone(), vec![n, n], false)
            .expect("gemm lhs");
        let b = session
            .tensor_variable(data.clone(), vec![n, n], false)
            .expect("gemm rhs");
        // Warm: first call pays allocator first-touch, which is not the GEMM.
        let _ = session.tensor_matmul(a, b).expect("warm matmul");
        let rounds = iters.max(3);
        let mut best = f64::INFINITY;
        for _ in 0..rounds {
            let started = std::time::Instant::now();
            let out = session.tensor_matmul(a, b).expect("matmul");
            let elapsed = started.elapsed().as_secs_f64();
            std::hint::black_box(&out);
            best = best.min(elapsed);
        }
        #[allow(clippy::cast_precision_loss)]
        let flops = 2.0 * (n as f64) * (n as f64) * (n as f64);
        println!(
            "gemm n={n} min={:.3} ms  {:.1} GFLOP/s  (min of {rounds})",
            best * 1e3,
            flops / best / 1e9
        );
        return;
    }

    // FT_BT_NB prices the blocked backtransform (frankentorch-wjrqt, shipped default-OFF).
    // 0 = the unblocked loop; >0 = panel width. Instruction counts are load-immune, so this
    // same-binary A/B is valid on a busy host where a wall-clock one is not.
    if let Ok(v) = std::env::var("FT_BT_NB") {
        let nb: usize = v.trim().parse().unwrap_or(0);
        ft_kernel_cpu::set_eigh_backtransform_nb(nb);
        eprintln!("backtransform_nb={nb}");
    }

    // FT_OP=eigh re-takes the eigh phase map. The banked one (reduce 42.7% / backtransform
    // 31.8% / tql2 24.8%) was measured on the DEFAULT fixture, whose symmetrised form has 496
    // of 512 eigenvalues exactly equal — the tridiagonal QL iteration deflates on equal
    // eigenvalues, so tql2 is understated there for the same reason the SVD sweep read 0%.
    // Same defect, second op, so the map has to be re-taken before it picks a lever.
    if std::env::var("FT_OP").map(|v| v == "eigh").unwrap_or(false) {
        // eigh reads one triangle, so hand it the symmetrised matrix both arms use.
        let mut sym = vec![0.0f64; n * n];
        for r in 0..n {
            for c in 0..n {
                sym[r * n + c] = (data[r * n + c] + data[c * n + r]) * 0.5;
            }
        }
        // FT_DTYPE=f32 exercises eigh_tql2_z_deferred_f32 instead. It is a SEPARATE function
        // from the f64 replay, so the f64 numbers say nothing about it — measuring the twin is
        // the point of having fixed the twin.
        if std::env::var("FT_DTYPE").map(|v| v == "f32").unwrap_or(false) {
            let sym32: Vec<f32> = sym.iter().map(|&v| v as f32).collect();
            let meta = ft_core::TensorMeta::from_shape(
                vec![n, n],
                ft_core::DType::F32,
                ft_core::Device::Cpu,
            );
            let out = ft_kernel_cpu::eigh_contiguous_f32(&sym32, &meta).expect("f32 eigh");
            println!(
                "eigh_f32 n={n} fixture={kind} lambda0={:.6e}",
                out.eigenvalues[0]
            );
            return;
        }
        let _ = ft_kernel_cpu::eigh_stage_profile_f64(&sym, n); // warm up

        // FT_TRED2_SWEEP=384,128,64,32 prices the reduction's parallel gate INSIDE one process,
        // interleaved round-robin with min-of-N per gate. Separate invocations would compare
        // two different windows on a host that has moved 1.94x between runs of one ELF; a
        // round-robin sweep with a min estimator is the form that survives drift.
        //
        // The question it answers: eigh's reduce is 52.9% of the op at 8 threads on a generic
        // spectrum and scales only 1.12x. For SVD the equivalent phase turned out memory-bound,
        // and forcing its parallel branch measured 34% WORSE — so this is not a widen, it is a
        // test of whether eigh's reduce is thread-limited or bandwidth-limited. The default
        // gate is TRED2_PAR_MIN_L_DEFAULT = 384, i.e. only rows with l >= 384 go parallel.
        if let Ok(spec) = std::env::var("FT_TRED2_SWEEP") {
            let gates: Vec<usize> = spec
                .split(',')
                .filter_map(|t| t.trim().parse().ok())
                .collect();
            let rounds: usize = std::env::var("FT_TRED2_ROUNDS")
                .ok()
                .and_then(|v| v.parse().ok())
                .unwrap_or(3);
            let mut best = vec![(u128::MAX, u128::MAX, u128::MAX); gates.len()];
            for _ in 0..rounds {
                for (slot, &gate) in best.iter_mut().zip(gates.iter()) {
                    let (r, b, t) = ft_kernel_cpu::eigh_stage_profile_gated_f64(&sym, n, gate);
                    slot.0 = slot.0.min(r);
                    slot.1 = slot.1.min(b);
                    slot.2 = slot.2.min(t);
                }
            }
            let ms = |v: u128| v as f64 / 1e6;
            for (&gate, &(r, b, t)) in gates.iter().zip(best.iter()) {
                println!(
                    "tred2_gate={gate} n={n} fixture={kind} reduce={:.3}ms backtransform={:.3}ms \
                     tql2={:.3}ms total={:.3}ms",
                    ms(r),
                    ms(b),
                    ms(t),
                    ms(r + b + t)
                );
            }
            return;
        }

        let (reduce, back, tql2) = ft_kernel_cpu::eigh_stage_profile_f64(&sym, n);
        let total = (reduce + back + tql2).max(1);
        let ms = |v: u128| v as f64 / 1e6;
        let pct = |v: u128| (v as f64 / total as f64) * 100.0;
        println!(
            "eigh_phases n={n} fixture={kind} reduce={:.3}ms ({:.1}%) backtransform={:.3}ms \
             ({:.1}%) tql2={:.3}ms ({:.1}%)",
            ms(reduce),
            pct(reduce),
            ms(back),
            pct(back),
            ms(tql2),
            pct(tql2)
        );
        return;
    }
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

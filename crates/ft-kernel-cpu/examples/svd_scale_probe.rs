//! Scale-dependence and full-U cost probes for `svd_contiguous_f64`.
//!
//! Two findings share this file because they share a call and a build:
//!
//! * `frankentorch-qpe2n` — `svd_tall`'s rank test compares a singular value
//!   against an ABSOLUTE `tol = 1e-15`. Singular values scale linearly with the
//!   matrix, so uniformly scaling a well-conditioned matrix DOWN should trip that
//!   test on every column at once and replace U with Gram-Schmidt basis vectors,
//!   while S and Vh stay correct. The prediction is a sharp cliff between a
//!   `1e-15` and a `1e-16` scaling of one matrix, with nothing else changed.
//!   PyTorch 2.12.1+cpu is scale-invariant on the same matrix down to `1e-30`
//!   (condition number 3.8293 throughout), so there is a known right answer.
//!
//! * `frankentorch-264q0` — the orthonormal-basis completion is O(m^4) for
//!   tall + `full_matrices`. This prints the full-vs-reduced cost curve so the
//!   exponent can be read off rather than argued about.
//!
//! Deliberately an example and not a unit test: `src/lib.rs` is under a peer's
//! exclusive reservation, and both questions are answerable through the public
//! kernel API without touching it.
//!
//! Run: `cargo run -q --release -p frankentorch-kernel-cpu --example svd_scale_probe`

use ft_core::{DType, Device, TensorMeta};
use ft_kernel_cpu::{
    svd_contiguous_f64, svd_deferred_left_hits_take, svd_deferred_left_phase_ns_take,
    svd_reduction_sweep_ns_take,
};

/// Well-conditioned deterministic fill, same xorshift family as `svd_golden`.
fn deterministic_matrix(m: usize, n: usize, seed: u64) -> Vec<f64> {
    let mut s = seed;
    let mut a = vec![0.0f64; m * n];
    for x in a.iter_mut() {
        s ^= s << 13;
        s ^= s >> 7;
        s ^= s << 17;
        *x = (s >> 11) as f64 / (1u64 << 53) as f64 - 0.5;
    }
    a
}

/// max |U diag(S) Vh - A| / max |A|, and max |U^T U - I|.
fn reconstruction_error(
    a: &[f64],
    m: usize,
    n: usize,
    u: &[f64],
    s: &[f64],
    vh: &[f64],
) -> (f64, f64) {
    let k = s.len();
    let u_cols = u.len() / m;
    let mut worst = 0.0f64;
    let anorm = a.iter().fold(0.0f64, |acc, v| acc.max(v.abs()));
    for i in 0..m {
        for j in 0..n {
            let mut acc = 0.0f64;
            for t in 0..k {
                acc += u[i * u_cols + t] * s[t] * vh[t * n + j];
            }
            worst = worst.max((acc - a[i * n + j]).abs());
        }
    }
    let rel = if anorm > 0.0 { worst / anorm } else { worst };

    let mut orth = 0.0f64;
    for p in 0..u_cols {
        for q in p..u_cols {
            let mut dot = 0.0f64;
            for i in 0..m {
                dot += u[i * u_cols + p] * u[i * u_cols + q];
            }
            let want = if p == q { 1.0 } else { 0.0 };
            orth = orth.max((dot - want).abs());
        }
    }
    (rel, orth)
}

fn scale_probe() {
    println!("== qpe2n: absolute tol=1e-15 vs uniform scaling (8x4, well-conditioned) ==");
    println!(
        "{:>10}  {:>12}  {:>18}  {:>12}  {}",
        "scale", "s_min", "max|USVt-A|/|A|", "max|UtU-I|", "verdict"
    );
    let (m, n) = (8usize, 4usize);
    let base = deterministic_matrix(m, n, 0x9e37_79b9_7f4a_7c15);
    let meta = TensorMeta::from_shape(vec![m, n], DType::F64, Device::Cpu);
    for exp in [0i32, -8, -12, -14, -15, -16, -17, -20, -30] {
        let scale = 10.0f64.powi(exp);
        let a: Vec<f64> = base.iter().map(|v| v * scale).collect();
        match svd_contiguous_f64(&a, &meta, false) {
            Ok(r) => {
                let (rel, orth) = reconstruction_error(&a, m, n, &r.u, &r.s, &r.vh);
                let s_min = r.s.iter().copied().fold(f64::INFINITY, f64::min);
                // A correct reduced SVD reconstructs to ~1e-15 RELATIVE regardless
                // of scale. Anything above 1e-9 relative is the failure this bead
                // predicts, not rounding.
                let verdict = if rel > 1e-9 { "BROKEN" } else { "ok" };
                println!("{scale:>10.0e}  {s_min:>12.4e}  {rel:>18.3e}  {orth:>12.3e}  {verdict}");
            }
            Err(e) => println!("{scale:>10.0e}  {:>12}  svd failed: {e:?}", "-"),
        }
    }
}

fn full_u_cost_probe() {
    println!();
    println!("== 264q0: full-vs-reduced cost, tall shapes (min of 3, interleaved) ==");
    println!(
        "{:>10}  {:>14}  {:>14}  {:>14}  {:>10}",
        "shape", "reduced ns", "full ns", "full-U ns", "full/reduced"
    );
    for &(m, n) in &[(32usize, 8usize), (64, 8), (96, 8), (128, 8)] {
        let a = deterministic_matrix(m, n, 0x1234_5678_9abc_def0);
        let meta = TensorMeta::from_shape(vec![m, n], DType::F64, Device::Cpu);
        let mut red = u128::MAX;
        let mut full = u128::MAX;
        for _ in 0..3 {
            let t = std::time::Instant::now();
            let _ = svd_contiguous_f64(&a, &meta, false);
            red = red.min(t.elapsed().as_nanos());
            let t = std::time::Instant::now();
            let _ = svd_contiguous_f64(&a, &meta, true);
            full = full.min(t.elapsed().as_nanos());
        }
        println!(
            "{:>10}  {red:>14}  {full:>14}  {:>14}  {:>10.1}x",
            format!("{m}x{n}"),
            full.saturating_sub(red),
            full as f64 / red.max(1) as f64
        );
    }
    println!();
    println!("If the completion is O(m^4), doubling m at fixed n multiplies the");
    println!("full-U term by ~16 while the reduced call grows ~linearly in m.");
}

/// Exit non-zero if any scale reconstructs badly, so this file is a regression
/// gate and not merely a report. `frankentorch-qpe2n` made every scale at or below
/// 1e-15 fail; a rebuilt binary that still fails them should not look like a pass.
fn scale_regression_gate() -> bool {
    let (m, n) = (8usize, 4usize);
    let base = deterministic_matrix(m, n, 0x9e37_79b9_7f4a_7c15);
    let meta = TensorMeta::from_shape(vec![m, n], DType::F64, Device::Cpu);
    let mut worst_scale = None;
    for exp in [0i32, -8, -12, -14, -15, -16, -17, -20, -30, -60, -150] {
        let scale = 10.0f64.powi(exp);
        let a: Vec<f64> = base.iter().map(|v| v * scale).collect();
        let Ok(r) = svd_contiguous_f64(&a, &meta, false) else {
            worst_scale = Some(exp);
            continue;
        };
        let (rel, orth) = reconstruction_error(&a, m, n, &r.u, &r.s, &r.vh);
        if rel > 1e-9 || orth > 1e-9 {
            worst_scale = Some(exp);
        }
    }
    match worst_scale {
        None => {
            println!("qpe2n gate: PASS -- reconstruction holds at every scale down to 1e-150");
            true
        }
        Some(exp) => {
            println!("qpe2n gate: FAIL -- reconstruction breaks at scale 1e{exp}");
            false
        }
    }
}

/// `frankentorch-v09ms`. Both square fast paths bail on `if full_matrices || m != n
/// || n < 64`, so a SQUARE matrix gets them at `full_matrices=false` and gets
/// neither at `full_matrices=true` — even though for m == n the two calls are the
/// same decomposition and U is m x m either way. This prices that gate. If the two
/// read the same, the gate costs nothing and the bead should be closed rather than
/// acted on.
fn square_fast_path_gate_probe() {
    println!();
    println!("== v09ms: square full_matrices vs reduced, at and below the n>=64 gate ==");
    println!(
        "{:>8}  {:>14}  {:>14}  {:>12}  {}",
        "n", "reduced ns", "full ns", "full/reduced", "fast paths eligible"
    );
    for &n in &[48usize, 64, 96, 128] {
        let a = deterministic_matrix(n, n, 0x0bad_c0ff_ee12_3456);
        let meta = TensorMeta::from_shape(vec![n, n], DType::F64, Device::Cpu);
        let mut red = u128::MAX;
        let mut full = u128::MAX;
        for _ in 0..3 {
            let t = std::time::Instant::now();
            let _ = svd_contiguous_f64(&a, &meta, false);
            red = red.min(t.elapsed().as_nanos());
            let t = std::time::Instant::now();
            let _ = svd_contiguous_f64(&a, &meta, true);
            full = full.min(t.elapsed().as_nanos());
        }
        // n < 64 is the control: there NEITHER call is eligible, so any ratio away
        // from 1.0 at n=48 is measuring something other than the gate.
        let eligible = if n >= 64 {
            "reduced only"
        } else {
            "neither (control)"
        };
        println!(
            "{n:>8}  {red:>14}  {full:>14}  {:>12.2}x  {eligible}",
            full as f64 / red.max(1) as f64
        );
    }
}

/// `frankentorch-r7jdo.1`. The square phase split says the VECTORS phase is 57-77%
/// of a square SVD and rising with n. Before aiming a lever at it, name the path
/// that actually runs: the deferred-left fast path computes U as a single parallel
/// `A*V/S` GEMM, whereas the general Golub-Reinsch path accumulates left Givens
/// rotations, and those two have completely different levers. Source reading has
/// been wrong about this before, so this counts takes instead of assuming.
fn deferred_left_sentinel_probe() {
    println!();
    println!("== r7jdo.1: which path does a square full SVD actually take? ==");
    println!(
        "{:>8}  {:>14}  {:>14}  {}",
        "n", "full ns", "hits", "path taken"
    );
    for &n in &[128usize, 256, 384] {
        let a = deterministic_matrix(n, n, 0x9e37_79b9_7f4a_7c15);
        let meta = TensorMeta::from_shape(vec![n, n], DType::F64, Device::Cpu);
        let _ = svd_deferred_left_hits_take();
        let t = std::time::Instant::now();
        let _ = svd_contiguous_f64(&a, &meta, true);
        let ns = t.elapsed().as_nanos();
        let hits = svd_deferred_left_hits_take();
        let path = if hits > 0 {
            "deferred-left fast path (A*V/S GEMM)"
        } else {
            "GENERAL Golub-Reinsch (left-Givens accumulation)"
        };
        println!("{n:>8}  {ns:>14}  {hits:>14}  {path}");
    }
}

/// Sub-split of the vectors phase, `frankentorch-r7jdo.1`. Knowing vectors are
/// 57-77% of a square SVD does not say which lever to build; the deferred-left path
/// has three parts with three completely different remedies, so this prices them.
/// Min of 3 by re-running and taking the cheapest whole call's split.
fn deferred_left_phase_split() {
    println!();
    println!("== r7jdo.1: inside the vectors phase, deferred-left path ==");
    println!(
        "{:>6}  {:>13}  {:>13}  {:>13}  {:>13}  {}",
        "n", "bidiag+V ns", "A*V gemm ns", "assemble ns", "total ns", "dominant term"
    );
    for &n in &[128usize, 256, 384] {
        let a = deterministic_matrix(n, n, 0x9e37_79b9_7f4a_7c15);
        let meta = TensorMeta::from_shape(vec![n, n], DType::F64, Device::Cpu);
        let mut best = (u64::MAX, 0u64, 0u64, 0u64);
        for _ in 0..3 {
            let _ = svd_deferred_left_phase_ns_take();
            let _ = svd_contiguous_f64(&a, &meta, true);
            let (qr, gemm, asm) = svd_deferred_left_phase_ns_take();
            let total = qr + gemm + asm;
            if total < best.0 {
                best = (total, qr, gemm, asm);
            }
        }
        let (total, qr, gemm, asm) = best;
        let dominant = if qr >= gemm && qr >= asm {
            "bidiag+V accumulation"
        } else if gemm >= asm {
            "A*V gemm"
        } else {
            "assemble (SERIAL O(n^2))"
        };
        #[allow(clippy::cast_precision_loss)]
        let pct = |x: u64| 100.0 * x as f64 / total.max(1) as f64;
        println!(
            "{n:>6}  {qr:>13}  {gemm:>13}  {asm:>13}  {total:>13}  {dominant} \
             ({:.1}% / {:.1}% / {:.1}%)",
            pct(qr),
            pct(gemm),
            pct(asm)
        );
    }
}

/// `frankentorch-ga99y` step 2. V's row orthogonality across RANK x SCALE.
///
/// Known so far: rank-2-of-64 at 1e-20 gives orthoV 1.839e-2, while full-rank at
/// 1e-20 is clean at 9.5e-15 — so the defect needs both. This sweep says which
/// variable actually drives it. If orthoV degrades smoothly as rank falls it is a
/// deficiency effect; if it switches on sharply at some scale it is a threshold
/// effect, which is what the near-threshold-convergence hypothesis predicts.
fn v_orthogonality_rank_scale_sweep() {
    println!();
    println!("== ga99y: V row orthogonality across rank x scale (n=64) ==");
    print!("{:>6}", "rank");
    for exp in [0i32, -5, -10, -15, -20] {
        print!("  {:>12}", format!("1e{exp}"));
    }
    println!();

    let n = 64usize;
    for &rank in &[2usize, 4, 8, 32, 64] {
        // `rank` independent columns; the rest repeat them cyclically.
        let mut z = 0x51ca_7e51_ca7e_u64;
        let mut base = vec![0.0f64; n * n];
        for row in 0..n {
            for col in 0..n {
                z ^= z << 13;
                z ^= z >> 7;
                z ^= z << 17;
                #[allow(clippy::cast_precision_loss)]
                let v = ((z >> 11) as f64) / ((1u64 << 53) as f64) - 0.5;
                base[row * n + col] = if col < rank {
                    v
                } else {
                    base[row * n + (col % rank)]
                };
            }
        }
        print!("{rank:>6}");
        for exp in [0i32, -5, -10, -15, -20] {
            let scale = 10.0f64.powi(exp);
            let a: Vec<f64> = base.iter().map(|v| v * scale).collect();
            let meta = TensorMeta::from_shape(vec![n, n], DType::F64, Device::Cpu);
            match svd_contiguous_f64(&a, &meta, false) {
                Ok(r) => {
                    let e = svd_rows_orthogonality_error_pub(&r.vh, r.k, r.n);
                    print!("  {e:>12.3e}");
                }
                Err(_) => print!("  {:>12}", "err"),
            }
        }
        println!();
    }
    println!("clean is ~1e-14; the bead's failing cell is rank 2 @ 1e-20 = 1.839e-2");
}

/// Local copy of the orthogonality metric so the example does not depend on a
/// private helper.
fn svd_rows_orthogonality_error_pub(vh: &[f64], k: usize, n: usize) -> f64 {
    let mut worst = 0.0f64;
    for p in 0..k {
        for q in p..k {
            let mut dot = 0.0f64;
            for j in 0..n {
                dot += vh[p * n + j] * vh[q * n + j];
            }
            let want = if p == q { 1.0 } else { 0.0 };
            worst = worst.max((dot - want).abs());
        }
    }
    worst
}

/// `frankentorch-r7jdo.1`, third level. The vectors phase is 94.5-99.1% of a
/// square SVD and all of it is inside `golub_reinsch_svd_*`, which has exactly two
/// O(n^3) halves: the Householder reduction to bidiagonal form, and the
/// bidiagonal-QR sweep that accumulates V. Those have completely different levers
/// — the reduction is already blocked (dgebrd-shape, GEMM trailing updates),
/// while the sweep applies Givens rotations one at a time. This says which one
/// holds the time before anything is built against either.
fn reduction_versus_sweep_split() {
    println!();
    println!("== r7jdo.1: reduction vs V-accumulating sweep (square, min of 3) ==");
    println!(
        "{:>6}  {:>13}  {:>13}  {:>13}  {:>13}  {}",
        "n", "reduce ns", "formP ns", "sweep ns", "total ns", "split"
    );
    for &n in &[128usize, 192, 256] {
        let a = deterministic_matrix(n, n, 0x9e37_79b9_7f4a_7c15);
        let meta = TensorMeta::from_shape(vec![n, n], DType::F64, Device::Cpu);
        let mut best = (u64::MAX, 0u64, 0u64, 0u64);
        for _ in 0..3 {
            let _ = svd_reduction_sweep_ns_take();
            let _ = svd_contiguous_f64(&a, &meta, true);
            let (rd, fp, sw) = svd_reduction_sweep_ns_take();
            if rd + fp + sw < best.0 {
                best = (rd + fp + sw, rd, fp, sw);
            }
        }
        let (total, rd, fp, sw) = best;
        #[allow(clippy::cast_precision_loss)]
        let pct = |x: u64| 100.0 * x as f64 / total.max(1) as f64;
        println!(
            "{n:>6}  {rd:>13}  {fp:>13}  {sw:>13}  {total:>13}  \
             reduce {:.1}% / formP {:.1}% / sweep {:.1}%",
            pct(rd),
            pct(fp),
            pct(sw)
        );
    }
    println!("the reduction is already blocked/GEMM-bound; the sweep is per-rotation");
}

fn main() {
    reduction_versus_sweep_split();
    v_orthogonality_rank_scale_sweep();
    scale_probe();
    full_u_cost_probe();
    square_fast_path_gate_probe();
    deferred_left_sentinel_probe();
    deferred_left_phase_split();
    println!();
    if !scale_regression_gate() {
        std::process::exit(1);
    }
}

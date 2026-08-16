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
use ft_kernel_cpu::svd_contiguous_f64;

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

fn main() {
    scale_probe();
    full_u_cost_probe();
}

//! Honest geev gap probe (frankentorch-l9xod). Times `eigvals`/`eig` on a
//! RANDOM matrix (the workload torch's LAPACK dgeev actually faces) rather than
//! the near-triangular fast-converging bench matrix. The matrix is built from a
//! deterministic LCG so the IDENTICAL matrix can be fed to torch for a fair
//! head-to-head (see scripts comparison in the session log).
//!
//!   rch exec -- cargo run --release -q -p ft-kernel-cpu --example eig_random_gap
//!
//! LCG: x_{k+1} = (6364136223846793005*x + 1442695040888963407) mod 2^64,
//! element = (x >> 11) as f64 / 2^53 * 2.0 - 1.0  (uniform in [-1, 1)).

use ft_core::{DType, Device, TensorMeta};
use ft_kernel_cpu::{eig_contiguous_f64, eigvals_contiguous_f64};
use std::time::Instant;

fn build(n: usize) -> Vec<f64> {
    let mut a = vec![0.0f64; n * n];
    let mut x: u64 = 0x9E3779B97F4A7C15; // fixed seed
    for slot in a.iter_mut() {
        x = x
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        let u = (x >> 11) as f64 / 9007199254740992.0; // 2^53
        *slot = u * 2.0 - 1.0;
    }
    a
}

fn bench<F: FnMut()>(mut f: F, it: usize) -> f64 {
    f();
    let t = Instant::now();
    for _ in 0..it {
        f();
    }
    t.elapsed().as_secs_f64() * 1e3 / it as f64
}

fn main() {
    println!("threads={}", rayon::current_num_threads());
    for &n in &[128usize, 256, 512] {
        let a = build(n);
        let m = TensorMeta::from_shape(vec![n, n], DType::F64, Device::Cpu);
        let it = if n <= 256 { 4 } else { 2 };
        let ev = bench(
            || {
                let _ = eigvals_contiguous_f64(&a, &m).unwrap();
            },
            it,
        );
        let eg = bench(
            || {
                let _ = eig_contiguous_f64(&a, &m).unwrap();
            },
            it,
        );
        // checksum of eigenvalue magnitudes so we can sanity-check vs torch
        let evals = eigvals_contiguous_f64(&a, &m).unwrap();
        let mut trace_re = 0.0f64;
        for i in 0..n {
            trace_re += evals[2 * i];
        }
        println!(
            "n={n:<5} eigvals={ev:9.2}ms  eig={eg:9.2}ms  vec_machinery={:.2}ms  sum_re(eigs)={trace_re:.6}",
            eg - ev
        );
    }
}

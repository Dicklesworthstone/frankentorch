//! Same-process A/B: BLOCKED compact-WY dsytrd reduction vs the unblocked packed
//! dsytd2 reference for `eigvalsh`, measured in ONE process so the two paths see
//! the SAME worker (no cross-`rch-exec` worker-variance confound). Reports the
//! speedup ratio and the max sorted-eigenvalue diff (tolerance parity, qgce4).
//!
//! Run: cargo run --release -j1 -p ft-kernel-cpu --example eigvalsh_blocked_ab

use ft_core::{DType, Device, TensorMeta};
use ft_kernel_cpu::{eigvalsh_blocked_f64, eigvalsh_contiguous_f64};
use std::time::Instant;

fn sym_matrix(n: usize) -> Vec<f64> {
    let mut a = vec![0.0f64; n * n];
    for i in 0..n {
        for j in 0..n {
            let bij = ((i * 31 + j * 17) % 97) as f64 * 0.013 - 0.5;
            let bji = ((j * 31 + i * 17) % 97) as f64 * 0.013 - 0.5;
            a[i * n + j] = 0.5 * (bij + bji);
        }
        a[i * n + i] += n as f64;
    }
    a
}

fn bench<F: FnMut()>(mut f: F, iters: usize) -> f64 {
    f(); // warm
    let t = Instant::now();
    for _ in 0..iters {
        f();
    }
    t.elapsed().as_secs_f64() * 1e3 / iters as f64
}

fn main() {
    println!(
        "eigvalsh: blocked dsytrd vs packed dsytd2 (same process, threads={})",
        rayon::current_num_threads()
    );
    for &n in &[256usize, 512, 768, 1024] {
        let a = sym_matrix(n);
        let meta = TensorMeta::from_shape(vec![n, n], DType::F64, Device::Cpu);
        let iters = if n <= 512 { 6 } else { 3 };
        // Interleave A and B to share any throttling/turbo state evenly.
        let packed = bench(
            || {
                let _ = eigvalsh_contiguous_f64(&a, &meta).unwrap();
            },
            iters,
        );
        let blocked = bench(
            || {
                let _ = eigvalsh_blocked_f64(&a, &meta).unwrap();
            },
            iters,
        );
        let mut p = eigvalsh_contiguous_f64(&a, &meta).unwrap();
        let mut b = eigvalsh_blocked_f64(&a, &meta).unwrap();
        p.sort_by(f64::total_cmp);
        b.sort_by(f64::total_cmp);
        let maxdiff = p
            .iter()
            .zip(&b)
            .map(|(x, y)| (x - y).abs())
            .fold(0.0f64, f64::max);
        println!(
            "n={n:<5} packed={packed:8.3}ms blocked={blocked:8.3}ms  ratio={:.2}x  maxdiff={maxdiff:.2e}",
            packed / blocked
        );
    }
}

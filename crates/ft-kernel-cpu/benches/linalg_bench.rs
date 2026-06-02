//! LU-factorization (det) bench. det routes through lu_factor_contiguous_f64,
//! whose O(n^3) trailing-submatrix update is the parallelized hotspot. Toggle:
//!   baseline:  rch exec -- env RAYON_NUM_THREADS=1 cargo bench -p ft-kernel-cpu --bench linalg_bench
//!   optimized: rch exec -- cargo bench -p ft-kernel-cpu --bench linalg_bench

use criterion::{Criterion, black_box, criterion_group, criterion_main};
use ft_core::{DType, Device, TensorMeta};
use ft_kernel_cpu::det_contiguous_f64;

fn bench_lu(c: &mut Criterion) {
    for &n in &[768usize, 1536usize] {
        // Diagonally dominant -> well-conditioned, no near-singular short-circuit.
        let mut a = vec![0.0_f64; n * n];
        for i in 0..n {
            for j in 0..n {
                a[i * n + j] = ((i * 31 + j * 17) % 97) as f64 * 0.013 - 0.5;
            }
            a[i * n + i] += n as f64;
        }
        let meta = TensorMeta::from_shape(vec![n, n], DType::F64, Device::Cpu);
        c.bench_function(&format!("det_lu_f64_{n}x{n}"), |b| {
            b.iter(|| black_box(det_contiguous_f64(black_box(&a), &meta).unwrap()))
        });
    }
}

criterion_group!(benches, bench_lu);
criterion_main!(benches);

//! LU-factorization (det) bench. det routes through lu_factor_contiguous_f64,
//! whose O(n^3) trailing-submatrix update is the parallelized hotspot. Toggle:
//!   baseline:  rch exec -- env RAYON_NUM_THREADS=1 cargo bench -p ft-kernel-cpu --bench linalg_bench
//!   optimized: rch exec -- cargo bench -p ft-kernel-cpu --bench linalg_bench

use criterion::{Criterion, black_box, criterion_group, criterion_main};
use ft_core::{DType, Device, TensorMeta};
use ft_kernel_cpu::{cholesky_contiguous_f64, det_contiguous_f64, eigh_contiguous_f64};

fn bench_eigh(c: &mut Criterion) {
    for &n in &[128usize, 256usize] {
        // Symmetric, well-conditioned: A = (B + B^T)/2 + n*I.
        let mut a = vec![0.0_f64; n * n];
        for i in 0..n {
            for j in 0..n {
                let bij = ((i * 31 + j * 17) % 97) as f64 * 0.013 - 0.5;
                let bji = ((j * 31 + i * 17) % 97) as f64 * 0.013 - 0.5;
                a[i * n + j] = 0.5 * (bij + bji);
            }
            a[i * n + i] += n as f64;
        }
        let meta = TensorMeta::from_shape(vec![n, n], DType::F64, Device::Cpu);
        c.bench_function(&format!("eigh_f64_{n}x{n}"), |bch| {
            bch.iter(|| black_box(eigh_contiguous_f64(black_box(&a), &meta).unwrap()))
        });
    }
}

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

fn bench_cholesky(c: &mut Criterion) {
    for &n in &[768usize, 1536usize] {
        // SPD: A = B^T B + n*I (well-conditioned, positive definite).
        let mut b = vec![0.0_f64; n * n];
        for i in 0..n {
            for j in 0..n {
                b[i * n + j] = ((i * 31 + j * 17) % 97) as f64 * 0.013 - 0.5;
            }
        }
        let mut a = vec![0.0_f64; n * n];
        for i in 0..n {
            for j in 0..n {
                let mut s = 0.0_f64;
                for k in 0..n {
                    s += b[k * n + i] * b[k * n + j];
                }
                a[i * n + j] = s;
            }
            a[i * n + i] += n as f64;
        }
        let meta = TensorMeta::from_shape(vec![n, n], DType::F64, Device::Cpu);
        c.bench_function(&format!("cholesky_f64_{n}x{n}"), |bch| {
            bch.iter(|| black_box(cholesky_contiguous_f64(black_box(&a), &meta, false).unwrap()))
        });
    }
}

criterion_group!(benches, bench_lu, bench_cholesky, bench_eigh);
criterion_main!(benches);

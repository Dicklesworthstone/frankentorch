//! LU-factorization (det) bench. det routes through lu_factor_contiguous_f64,
//! whose O(n^3) trailing-submatrix update is the parallelized hotspot. Toggle:
//!   baseline:  rch exec -- env RAYON_NUM_THREADS=1 cargo bench -p ft-kernel-cpu --bench linalg_bench
//!   optimized: rch exec -- cargo bench -p ft-kernel-cpu --bench linalg_bench

use criterion::{BatchSize, Criterion, black_box, criterion_group, criterion_main};
use ft_core::{DType, Device, TensorMeta};
use ft_kernel_cpu::{
    cholesky_contiguous_f64, det_contiguous_f64, eig_contiguous_f64, eigh_contiguous_f64,
    eigvals_contiguous_f64, eigvalsh_contiguous_f64, inv_tensor_contiguous_f64,
    lobpcg_contiguous_f64, matrix_exp_contiguous_f64, qr_contiguous_f64, svd_contiguous_f64,
    svd_lowrank_contiguous_f64, svdvals_contiguous_f64, symmetric_rank2k_lower_update_f64,
};

fn symmetric_rank2k_lower_update_scalar(n: usize, k: usize, v: &[f64], w: &[f64], a: &mut [f64]) {
    for row in 0..n {
        for col in 0..=row {
            let mut update = 0.0_f64;
            for p in 0..k {
                update += v[row * k + p] * w[col * k + p] + w[row * k + p] * v[col * k + p];
            }
            a[row * n + col] -= update;
        }
    }
}

fn bench_lobpcg(c: &mut Criterion) {
    for &n in &[256usize, 512usize] {
        // Symmetric, well-separated spectrum (distinct diagonal + small off-diag).
        let mut a = vec![0.0_f64; n * n];
        for i in 0..n {
            for j in 0..n {
                a[i * n + j] = (((i * 7 + j * 3 + 1) % 13) as f64) * 0.02;
            }
        }
        for i in 0..n {
            for j in 0..i {
                let s = 0.5 * (a[i * n + j] + a[j * n + i]);
                a[i * n + j] = s;
                a[j * n + i] = s;
            }
            a[i * n + i] += i as f64;
        }
        let meta = TensorMeta::from_shape(vec![n, n], DType::F64, Device::Cpu);
        c.bench_function(&format!("lobpcg_f64_{n}x{n}_k8"), |bch| {
            bch.iter(|| {
                black_box(lobpcg_contiguous_f64(black_box(&a), &meta, 8, true, 100, 1e-9).unwrap())
            })
        });
    }
}

fn bench_svd_lowrank(c: &mut Criterion) {
    for &n in &[256usize, 512usize] {
        // Low-rank-plus-tiny-noise n x n (effective rank ~16): the regime where
        // randomized SVD (O(n^2 k)) dwarfs the full O(n^3) SVD.
        let r = 16usize;
        let mut b = vec![0.0_f64; n * r];
        let mut cm = vec![0.0_f64; r * n];
        for i in 0..n {
            for j in 0..r {
                b[i * r + j] = ((i * 7 + j * 3 + 1) % 23) as f64 * 0.01 - 0.11;
            }
        }
        for i in 0..r {
            for j in 0..n {
                cm[i * n + j] = ((i * 5 + j * 2 + 4) % 19) as f64 * 0.01 - 0.09;
            }
        }
        let mut a = vec![0.0_f64; n * n];
        for i in 0..n {
            for j in 0..n {
                let mut s = 0.0;
                for k in 0..r {
                    s += b[i * r + k] * cm[k * n + j];
                }
                a[i * n + j] = s + (((i + j) % 7) as f64) * 1e-6;
            }
        }
        let meta = TensorMeta::from_shape(vec![n, n], DType::F64, Device::Cpu);
        c.bench_function(&format!("svd_lowrank_f64_{n}x{n}_q16"), |bch| {
            bch.iter(|| black_box(svd_lowrank_contiguous_f64(black_box(&a), &meta, 16, 2).unwrap()))
        });
    }
}

fn bench_eig_general(c: &mut Criterion) {
    for &n in &[128usize, 256usize] {
        // Non-symmetric with WELL-SEPARATED real eigenvalues (distinct diagonal
        // + small off-diagonal perturbation) so the shifted QR iteration
        // converges in a few steps per eigenvalue rather than hitting max_iter.
        let mut a = vec![0.0_f64; n * n];
        for i in 0..n {
            for j in 0..n {
                a[i * n + j] = ((i * 41 + j * 13 + 5) % 17) as f64 * 0.01 - 0.08;
            }
            a[i * n + i] = (i as f64) + 1.0;
        }
        let meta = TensorMeta::from_shape(vec![n, n], DType::F64, Device::Cpu);
        c.bench_function(&format!("eig_f64_{n}x{n}"), |bch| {
            bch.iter(|| black_box(eig_contiguous_f64(black_box(&a), &meta).unwrap()))
        });
        c.bench_function(&format!("eigvals_f64_{n}x{n}"), |bch| {
            bch.iter(|| black_box(eigvals_contiguous_f64(black_box(&a), &meta).unwrap()))
        });
    }
}

fn bench_qr(c: &mut Criterion) {
    // Householder QR (Q and R). Each reflection's apply to R (per-column factor +
    // row update) and to Q (per-row) is the O(n^3) compute-bound hotspot,
    // bit-exactly parallelized over columns/rows.
    for &n in &[512usize, 768usize] {
        let mut a = vec![0.0f64; n * n];
        for i in 0..n {
            for j in 0..n {
                a[i * n + j] = (((i * 53 + j * 31) % 97) as f64 - 48.0) * 0.1;
            }
            a[i * n + i] += n as f64;
        }
        let meta = TensorMeta::from_shape(vec![n, n], DType::F64, Device::Cpu);
        c.bench_function(&format!("qr_f64_{n}x{n}"), |bch| {
            bch.iter(|| black_box(qr_contiguous_f64(black_box(&a), &meta, true).unwrap()))
        });
    }
}

fn bench_inv(c: &mut Criterion) {
    // Matrix inverse = LU factor (already parallel) + solve against the n-column
    // identity; the matrix-RHS triangular solve is the parallelization target.
    for &n in &[256usize, 512usize] {
        // Diagonally dominant -> well-conditioned, non-singular.
        let mut a = vec![0.0_f64; n * n];
        for i in 0..n {
            for j in 0..n {
                a[i * n + j] = ((i * 31 + j * 17) % 97) as f64 * 0.013 - 0.5;
            }
            a[i * n + i] += n as f64;
        }
        let meta = TensorMeta::from_shape(vec![n, n], DType::F64, Device::Cpu);
        c.bench_function(&format!("inv_f64_{n}x{n}"), |bch| {
            bch.iter(|| black_box(inv_tensor_contiguous_f64(black_box(&a), &meta).unwrap()))
        });
    }
}

fn bench_matrix_exp(c: &mut Criterion) {
    // Scaling-and-squaring matrix exponential: dominated by the n x n matmuls.
    for &n in &[128usize, 256usize] {
        let mut a = vec![0.0_f64; n * n];
        for i in 0..n {
            for j in 0..n {
                a[i * n + j] = (((i * 41 + j * 23 + 5) % 89) as f64) * 0.011 - 0.5;
            }
        }
        let meta = TensorMeta::from_shape(vec![n, n], DType::F64, Device::Cpu);
        c.bench_function(&format!("matrix_exp_f64_{n}x{n}"), |bch| {
            bch.iter(|| black_box(matrix_exp_contiguous_f64(black_box(&a), &meta).unwrap()))
        });
    }
}

fn bench_svdvals(c: &mut Criterion) {
    // Values-only SVD: a bidiagonalize-then-bidiagonal-QR path can skip all
    // U/V accumulation. Tall shapes maximize the saved O(m n^2) U work.
    for &(m, n) in &[(256usize, 128usize), (384usize, 128usize)] {
        let mut a = vec![0.0_f64; m * n];
        for i in 0..m {
            for j in 0..n {
                a[i * n + j] = (((i * 53 + j * 131 + 7) % 251) as f64) * 0.01 - 1.25
                    + ((i as f64) * 0.013).sin();
            }
        }
        let meta = TensorMeta::from_shape(vec![m, n], DType::F64, Device::Cpu);
        c.bench_function(&format!("svdvals_f64_{m}x{n}"), |bch| {
            bch.iter(|| black_box(svdvals_contiguous_f64(black_box(&a), &meta).unwrap()))
        });
    }
}

fn bench_svd(c: &mut Criterion) {
    // Tall and square cases; SVD reduces to a symmetric eigenproblem on A^T A.
    for &(m, n) in &[(256usize, 128usize), (256usize, 256usize)] {
        let mut a = vec![0.0_f64; m * n];
        for i in 0..m {
            for j in 0..n {
                a[i * n + j] = ((i * 31 + j * 17) % 97) as f64 * 0.013 - 0.5;
            }
        }
        let meta = TensorMeta::from_shape(vec![m, n], DType::F64, Device::Cpu);
        c.bench_function(&format!("svd_f64_{m}x{n}"), |bch| {
            bch.iter(|| black_box(svd_contiguous_f64(black_box(&a), &meta, false).unwrap()))
        });
    }
}

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
        c.bench_function(&format!("eigvalsh_f64_{n}x{n}"), |bch| {
            bch.iter(|| black_box(eigvalsh_contiguous_f64(black_box(&a), &meta).unwrap()))
        });
    }
}

fn bench_symmetric_rank2k_update(c: &mut Criterion) {
    let (n, k) = (256usize, 32usize);
    let v: Vec<f64> = (0..n * k)
        .map(|i| ((i % 37) as f64 - 18.0) * 0.009 + (i as f64) * 1e-8)
        .collect();
    let w: Vec<f64> = (0..n * k)
        .map(|i| ((i % 29) as f64 - 14.0) * 0.011 - (i as f64) * 1e-8)
        .collect();
    let a: Vec<f64> = (0..n * n)
        .map(|i| ((i % 53) as f64 - 26.0) * 0.003)
        .collect();

    c.bench_function("sym_rank2k_lower_scalar_f64_256x32", |bch| {
        bch.iter_batched(
            || a.clone(),
            |mut work| {
                symmetric_rank2k_lower_update_scalar(n, k, &v, &w, &mut work);
                black_box(work)
            },
            BatchSize::SmallInput,
        );
    });
    c.bench_function("sym_rank2k_lower_gemm_f64_256x32", |bch| {
        bch.iter_batched(
            || a.clone(),
            |mut work| {
                symmetric_rank2k_lower_update_f64(n, k, &v, &w, &mut work).unwrap();
                black_box(work)
            },
            BatchSize::SmallInput,
        );
    });
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

criterion_group!(
    benches,
    bench_lu,
    bench_cholesky,
    bench_eigh,
    bench_symmetric_rank2k_update,
    bench_lobpcg,
    bench_eig_general,
    bench_svd,
    bench_svd_lowrank,
    bench_svdvals,
    bench_matrix_exp,
    bench_inv,
    bench_qr
);
criterion_main!(benches);

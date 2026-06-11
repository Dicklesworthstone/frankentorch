//! LU-factorization (det) bench. det routes through lu_factor_contiguous_f64,
//! whose O(n^3) trailing-submatrix update is the parallelized hotspot. Toggle:
//!   baseline:  rch exec -- env RAYON_NUM_THREADS=1 cargo bench -p ft-kernel-cpu --bench linalg_bench
//!   optimized: rch exec -- cargo bench -p ft-kernel-cpu --bench linalg_bench

use criterion::{BatchSize, Criterion, black_box, criterion_group, criterion_main};
use ft_core::{DType, Device, TensorMeta};
use ft_kernel_cpu::{
    banded_to_tridiagonal_f64, cholesky_contiguous_f32, cholesky_contiguous_f64,
    cholesky_solve_contiguous_f32, cholesky_solve_contiguous_f64, det_contiguous_f64,
    eig_contiguous_f64, eigh_contiguous_f64, eigvals_contiguous_f64, eigvalsh_contiguous_f64,
    eigvalsh_two_stage_f64, inv_tensor_contiguous_f64, lobpcg_contiguous_f64,
    lu_factor_contiguous_f64, lu_solve_contiguous_f32, lu_solve_contiguous_f64,
    lu_solve_mixed_refine_contiguous_f64, matrix_exp_contiguous_f32, matrix_exp_contiguous_f64,
    qr_contiguous_f64, qr_contiguous_f64_stage_profile, svd_contiguous_f64,
    svd_lowrank_contiguous_f64, svdvals_contiguous_f64, symmetric_rank2k_lower_update_f64,
    symmetric_to_banded_f64,
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
        if n == 512 {
            c.bench_function(&format!("qr_f64_{n}x{n}_stage_profile"), |bch| {
                bch.iter(|| {
                    black_box(qr_contiguous_f64_stage_profile(black_box(&a), &meta, true).unwrap())
                })
            });
        }
    }
    // Tall reduced QR (m >> n) — QR's dominant use case (least squares). The
    // reverse-dorgqr blocked path builds Q at m×k instead of the full m×m, so the
    // Q accumulation is O(m·k²) not O(m²·k). frankentorch-ct2yy.
    for &(m, ncol) in &[(2048usize, 128usize), (4096usize, 128usize)] {
        let mut a = vec![0.0f64; m * ncol];
        for i in 0..m {
            for j in 0..ncol {
                a[i * ncol + j] = (((i * 53 + j * 31) % 97) as f64 - 48.0) * 0.1;
            }
        }
        for j in 0..ncol {
            a[j * ncol + j] += m as f64;
        }
        let meta = TensorMeta::from_shape(vec![m, ncol], DType::F64, Device::Cpu);
        c.bench_function(&format!("qr_f64_tall_{m}x{ncol}"), |bch| {
            bch.iter(|| black_box(qr_contiguous_f64(black_box(&a), &meta, true).unwrap()))
        });
        if m == 2048 && ncol == 128 {
            c.bench_function(&format!("qr_f64_tall_{m}x{ncol}_stage_profile"), |bch| {
                bch.iter(|| {
                    black_box(qr_contiguous_f64_stage_profile(black_box(&a), &meta, true).unwrap())
                })
            });
        }
    }
}

fn bench_inv(c: &mut Criterion) {
    // Matrix inverse = LU factor (already parallel) + solve against the n-column
    // identity; the matrix-RHS triangular solve is the parallelization target.
    // 1024 included so the column-parallel lu_solve win (where the RHS panel
    // spills cache) is visible — small n stays serial (cache-optimal).
    for &n in &[256usize, 512usize, 1024usize] {
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

fn bench_lu_solve(c: &mut Criterion) {
    let (n, rhs) = (512usize, 32usize);
    let mut a = vec![0.0_f64; n * n];
    for i in 0..n {
        for j in 0..n {
            a[i * n + j] = ((i * 31 + j * 17) % 97) as f64 * 0.013 - 0.5;
        }
        a[i * n + i] += n as f64;
    }
    let b: Vec<f64> = (0..n * rhs)
        .map(|idx| ((idx * 19 + 11) % 101) as f64 * 0.007 - 0.33)
        .collect();
    let meta_a = TensorMeta::from_shape(vec![n, n], DType::F64, Device::Cpu);
    let meta_b = TensorMeta::from_shape(vec![n, rhs], DType::F64, Device::Cpu);

    c.bench_function("lu_solve_f64_512x512_rhs32", |bch| {
        bch.iter(|| {
            let factor = lu_factor_contiguous_f64(black_box(&a), &meta_a).unwrap();
            black_box(lu_solve_contiguous_f64(&factor, black_box(&b), &meta_b).unwrap())
        })
    });

    c.bench_function("lu_solve_mixed_refine_512x512_rhs32", |bch| {
        bch.iter(|| {
            black_box(
                lu_solve_mixed_refine_contiguous_f64(
                    black_box(&a),
                    &meta_a,
                    black_box(&b),
                    &meta_b,
                )
                .unwrap(),
            )
        })
    });

    // Native f32 LU solve (frankentorch-b3o90) vs f64, SOLVE-ONLY (factor
    // precomputed), n-RHS = the inverse workload, serial regime (n²·nrhs < 2^26).
    // A triangular solve streams the factor -> f32 halves the bytes -> ~2x.
    {
        let n2 = 384usize;
        let mut a2 = vec![0.0_f64; n2 * n2];
        for i in 0..n2 {
            for j in 0..n2 {
                a2[i * n2 + j] = ((i * 31 + j * 17) % 97) as f64 * 0.013 - 0.5;
            }
            a2[i * n2 + i] += n2 as f64;
        }
        let meta_a2 = TensorMeta::from_shape(vec![n2, n2], DType::F64, Device::Cpu);
        let factor = lu_factor_contiguous_f64(&a2, &meta_a2).unwrap();
        let mut id = vec![0.0_f64; n2 * n2];
        for i in 0..n2 {
            id[i * n2 + i] = 1.0;
        }
        let id_meta = TensorMeta::from_shape(vec![n2, n2], DType::F64, Device::Cpu);
        c.bench_function("lu_solve_f64_384x384_nrhs", |bch| {
            bch.iter(|| {
                black_box(lu_solve_contiguous_f64(&factor, black_box(&id), &id_meta).unwrap())
            })
        });
        let lu32: Vec<f32> = factor.lu.iter().map(|&v| v as f32).collect();
        let id32: Vec<f32> = id.iter().map(|&v| v as f32).collect();
        let id_meta32 = TensorMeta::from_shape(vec![n2, n2], DType::F32, Device::Cpu);
        c.bench_function("lu_solve_f32_384x384_nrhs", |bch| {
            bch.iter(|| {
                black_box(
                    lu_solve_contiguous_f32(
                        &lu32,
                        &factor.pivots,
                        n2,
                        black_box(&id32),
                        &id_meta32,
                    )
                    .unwrap(),
                )
            })
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
        // Native f32 path (frankentorch-b3o90): all square n^3 GEMMs, so f32 gets
        // the full ~2-3x sgemm speedup vs the current f32->f64 upcast.
        let a32: Vec<f32> = a.iter().map(|&v| v as f32).collect();
        let meta32 = TensorMeta::from_shape(vec![n, n], DType::F32, Device::Cpu);
        c.bench_function(&format!("matrix_exp_f32_{n}x{n}"), |bch| {
            bch.iter(|| black_box(matrix_exp_contiguous_f32(black_box(&a32), &meta32).unwrap()))
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
    // Tall and square cases. NOTE: the `(i*31+j*17)%97` matrix is RANK-DEFICIENT
    // (rank <= 97), which triggers a clustered/zero singular-value convergence
    // pathology in the Golub-Reinsch implicit-QR sweep — svd_f64_256x256 here is
    // ~12x slower than a well-conditioned 256x256 (frankentorch-yyylo). The
    // `_wellcond` case below measures representative SVD perf so the
    // bidiagonalization-vs-sweep balance (and any parallelization win) is visible
    // instead of being swamped by the pathological sweep on the degenerate input.
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

    // Well-conditioned full-rank square case (deterministic xorshift), the
    // representative real-world SVD workload.
    {
        let (m, n) = (256usize, 256usize);
        let mut a = vec![0.0_f64; m * n];
        let mut s: u64 = 0x9e37_79b9_7f4a_7c15;
        for x in a.iter_mut() {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            *x = (s >> 11) as f64 / (1u64 << 53) as f64 - 0.5;
        }
        let meta = TensorMeta::from_shape(vec![m, n], DType::F64, Device::Cpu);
        c.bench_function(&format!("svd_f64_{m}x{n}_wellcond"), |bch| {
            bch.iter(|| black_box(svd_contiguous_f64(black_box(&a), &meta, false).unwrap()))
        });
        // full_matrices=true (torch default): U and V are both accumulated in the
        // bidiagonal-QR sweep (no deferred-left GEMM), so this exercises the
        // sweep's left+right Givens streams directly.
        c.bench_function(&format!("svd_full_f64_{m}x{n}_wellcond"), |bch| {
            bch.iter(|| black_box(svd_contiguous_f64(black_box(&a), &meta, true).unwrap()))
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

fn bench_eigvalsh_two_stage(c: &mut Criterion) {
    for &(n, b) in &[(128usize, 16usize), (256usize, 32usize)] {
        // Same matrix family as `bench_eigh`, so public live and two-stage rows
        // compare the eigensolver algorithm rather than input conditioning.
        let mut a = vec![0.0_f64; n * n];
        for i in 0..n {
            for j in 0..n {
                let bij = ((i * 31 + j * 17) % 97) as f64 * 0.013 - 0.5;
                let bji = ((j * 31 + i * 17) % 97) as f64 * 0.013 - 0.5;
                a[i * n + j] = 0.5 * (bij + bji);
            }
            a[i * n + i] += n as f64;
        }
        c.bench_function(&format!("eigvalsh_two_stage_f64_{n}x{n}_b{b}"), |bch| {
            bch.iter(|| black_box(eigvalsh_two_stage_f64(black_box(&a), n, b).unwrap()))
        });
        c.bench_function(&format!("sym_to_banded_f64_{n}x{n}_b{b}"), |bch| {
            bch.iter(|| black_box(symmetric_to_banded_f64(black_box(&a), n, b).unwrap()))
        });
        let (band, _q) = symmetric_to_banded_f64(&a, n, b).unwrap();
        c.bench_function(&format!("banded_to_tridiag_f64_{n}x{n}_b{b}"), |bch| {
            bch.iter(|| black_box(banded_to_tridiagonal_f64(black_box(&band), n, b)))
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
        // Native f32 path (frankentorch-b3o90): same SPD matrix in f32. torch f32
        // cholesky uses f32 LAPACK, so this is the parity-correct dtype and ~2x
        // faster than the current f32->f64 upcast (which runs the f64 kernel).
        let a32: Vec<f32> = a.iter().map(|&v| v as f32).collect();
        let meta32 = TensorMeta::from_shape(vec![n, n], DType::F32, Device::Cpu);
        c.bench_function(&format!("cholesky_f32_{n}x{n}"), |bch| {
            bch.iter(|| {
                black_box(cholesky_contiguous_f32(black_box(&a32), &meta32, false).unwrap())
            })
        });
    }
}

fn bench_cholesky_solve(c: &mut Criterion) {
    // n-RHS Cholesky solve = the cholesky_inverse workload; the column-parallel
    // triangular-solve target (matches the lu_solve/inv win, frankentorch-otbok).
    for &n in &[512usize, 1024usize] {
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
        let factor = cholesky_contiguous_f64(&a, &meta, false).unwrap();
        let mut id = vec![0.0_f64; n * n];
        for i in 0..n {
            id[i * n + i] = 1.0;
        }
        let id_meta = TensorMeta::from_shape(vec![n, n], DType::F64, Device::Cpu);
        c.bench_function(&format!("cholesky_solve_f64_{n}x{n}_nrhs"), |bch| {
            bch.iter(|| {
                black_box(
                    cholesky_solve_contiguous_f64(&factor, black_box(&id), &id_meta, false)
                        .unwrap(),
                )
            })
        });
        // Native f32 path (frankentorch-b3o90): a triangular solve streams the
        // factor, so f32 halves the memory traffic -> ~2x vs the f64-upcast path.
        let factor32: Vec<f32> = factor.factor.iter().map(|&v| v as f32).collect();
        let id32: Vec<f32> = id.iter().map(|&v| v as f32).collect();
        let id_meta32 = TensorMeta::from_shape(vec![n, n], DType::F32, Device::Cpu);
        c.bench_function(&format!("cholesky_solve_f32_{n}x{n}_nrhs"), |bch| {
            bch.iter(|| {
                black_box(
                    cholesky_solve_contiguous_f32(
                        &factor32,
                        n,
                        black_box(&id32),
                        &id_meta32,
                        false,
                    )
                    .unwrap(),
                )
            })
        });
    }
}

criterion_group!(
    benches,
    bench_lu,
    bench_lu_solve,
    bench_cholesky,
    bench_cholesky_solve,
    bench_eigh,
    bench_eigvalsh_two_stage,
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

//! GEMM throughput bench. Run single-threaded vs multi-threaded with the same
//! binary by toggling `RAYON_NUM_THREADS`:
//!   baseline:  RAYON_NUM_THREADS=1 cargo bench -p ft-kernel-cpu --bench gemm_bench
//!   optimized: cargo bench -p ft-kernel-cpu --bench gemm_bench
//! The matmul kernel only splits across rows when more than one rayon thread is
//! available, so RAYON_NUM_THREADS=1 exercises the original single-call path.

use criterion::{Criterion, black_box, criterion_group, criterion_main};
use ft_core::{DType, Device, TensorMeta};
use ft_kernel_cpu::{matmul_tensor_contiguous_f32, matmul_tensor_contiguous_f64};

fn bench_gemm(c: &mut Criterion) {
    for &(m, k, n) in &[(512usize, 512usize, 512usize), (1024, 1024, 1024)] {
        let lhs: Vec<f64> = (0..m * k).map(|i| (i % 100) as f64 * 0.01).collect();
        let rhs: Vec<f64> = (0..k * n).map(|i| (i % 100) as f64 * 0.02).collect();
        let lm = TensorMeta::from_shape(vec![m, k], DType::F64, Device::Cpu);
        let rm = TensorMeta::from_shape(vec![k, n], DType::F64, Device::Cpu);
        c.bench_function(&format!("matmul_f64_{m}x{k}x{n}"), |b| {
            b.iter(|| {
                black_box(
                    matmul_tensor_contiguous_f64(black_box(&lhs), black_box(&rhs), &lm, &rm)
                        .unwrap(),
                )
            })
        });

        let lhs32: Vec<f32> = (0..m * k).map(|i| (i % 100) as f32 * 0.01).collect();
        let rhs32: Vec<f32> = (0..k * n).map(|i| (i % 100) as f32 * 0.02).collect();
        let lm32 = TensorMeta::from_shape(vec![m, k], DType::F32, Device::Cpu);
        let rm32 = TensorMeta::from_shape(vec![k, n], DType::F32, Device::Cpu);
        c.bench_function(&format!("matmul_f32_{m}x{k}x{n}"), |b| {
            b.iter(|| {
                black_box(
                    matmul_tensor_contiguous_f32(black_box(&lhs32), black_box(&rhs32), &lm32, &rm32)
                        .unwrap(),
                )
            })
        });
    }
}

criterion_group!(benches, bench_gemm);
criterion_main!(benches);

//! Compute-bound elementwise bench (pow / powf). Toggle RAYON_NUM_THREADS=1 for
//! the single-threaded baseline vs default (all cores) for the parallel path:
//!   baseline:  rch exec -- env RAYON_NUM_THREADS=1 cargo bench -p ft-kernel-cpu --bench elementwise_bench
//!   optimized: rch exec -- cargo bench -p ft-kernel-cpu --bench elementwise_bench

use criterion::{Criterion, black_box, criterion_group, criterion_main};
use ft_core::{DType, Device, TensorMeta};
use ft_kernel_cpu::{pow_tensor_contiguous_f32, pow_tensor_contiguous_f64};

fn bench_pow(c: &mut Criterion) {
    let numel = 1usize << 20; // 1M elements
    let data64: Vec<f64> = (0..numel)
        .map(|i| (i % 97) as f64 * 0.013 + 0.001)
        .collect();
    let meta64 = TensorMeta::from_shape(vec![numel], DType::F64, Device::Cpu);
    c.bench_function("pow_f64_1m_exp2.5", |b| {
        b.iter(|| {
            black_box(
                pow_tensor_contiguous_f64(black_box(&data64), &meta64, black_box(2.5))
                    .expect("valid f64 pow benchmark input"),
            )
        })
    });

    let data32: Vec<f32> = (0..numel)
        .map(|i| (i % 89) as f32 * 0.017 + 0.001)
        .collect();
    let meta32 = TensorMeta::from_shape(vec![numel], DType::F32, Device::Cpu);
    c.bench_function("pow_f32_1m_exp2.5", |b| {
        b.iter(|| {
            black_box(
                pow_tensor_contiguous_f32(black_box(&data32), &meta32, black_box(2.5))
                    .expect("valid f32 pow benchmark input"),
            )
        })
    });
}

criterion_group!(benches, bench_pow);
criterion_main!(benches);

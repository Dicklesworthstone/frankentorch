//! softmax / log_softmax dim-reduction bench. Toggle RAYON_NUM_THREADS=1 for the
//! single-threaded baseline vs default (all cores) for the parallel path:
//!   baseline:  rch exec -- env RAYON_NUM_THREADS=1 cargo bench -p ft-kernel-cpu --bench softmax_bench
//!   optimized: rch exec -- cargo bench -p ft-kernel-cpu --bench softmax_bench

use criterion::{Criterion, black_box, criterion_group, criterion_main};
use ft_core::{DType, Device, TensorMeta};
use ft_kernel_cpu::{
    log_softmax_dim_tensor_contiguous_f32, log_softmax_dim_tensor_contiguous_f64,
    softmax_dim_tensor_contiguous_f32, softmax_dim_tensor_contiguous_f64,
};

fn bench_softmax(c: &mut Criterion) {
    // Cross-entropy / classifier shape: [batch, vocab], reduce over last dim.
    let (rows, cols) = (8192usize, 1024usize);
    let data: Vec<f64> = (0..rows * cols)
        .map(|i| ((i % 257) as f64 - 128.0) * 0.05)
        .collect();
    let meta = TensorMeta::from_shape(vec![rows, cols], DType::F64, Device::Cpu);
    c.bench_function("log_softmax_f64_8192x1024_dim1", |b| {
        b.iter(|| {
            black_box(
                log_softmax_dim_tensor_contiguous_f64(black_box(&data), &meta, 1)
                    .expect("valid log_softmax input"),
            )
        })
    });

    let data32: Vec<f32> = (0..rows * cols)
        .map(|i| ((i % 257) as f32 - 128.0) * 0.05)
        .collect();
    let meta32 = TensorMeta::from_shape(vec![rows, cols], DType::F32, Device::Cpu);
    c.bench_function("log_softmax_f32_8192x1024_dim1", |b| {
        b.iter(|| {
            black_box(
                log_softmax_dim_tensor_contiguous_f32(black_box(&data32), &meta32, 1)
                    .expect("valid log_softmax input"),
            )
        })
    });
    c.bench_function("softmax_f32_8192x1024_dim1", |b| {
        b.iter(|| {
            black_box(
                softmax_dim_tensor_contiguous_f32(black_box(&data32), &meta32, 1)
                    .expect("valid softmax input"),
            )
        })
    });

    // Strided (general) path: softmax over a middle dim.
    let (a, m, inner) = (4096usize, 32usize, 8usize);
    let data_s: Vec<f64> = (0..a * m * inner)
        .map(|i| ((i % 211) as f64 - 100.0) * 0.04)
        .collect();
    let meta_s = TensorMeta::from_shape(vec![a, m, inner], DType::F64, Device::Cpu);
    c.bench_function("softmax_f64_strided_4096x32x8_dim1", |b| {
        b.iter(|| {
            black_box(
                softmax_dim_tensor_contiguous_f64(black_box(&data_s), &meta_s, 1)
                    .expect("valid softmax input"),
            )
        })
    });
}

criterion_group!(benches, bench_softmax);
criterion_main!(benches);

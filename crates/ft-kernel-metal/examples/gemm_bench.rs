//! GEMM micro-bench at franken_whisper turbo encoder shapes.
//!
//! Measures the fused `matmul_bias` kernel's effective GFLOPS at the exact
//! shapes the large-v3-turbo encoder dispatches (seq=1500, d_model=1280,
//! d_ff=5120), so a kernel change has a same-harness before/after number.
//!
//! Usage: cargo run --release -p ft-kernel-metal --example gemm_bench [reps]

use ft_kernel_metal::fused::{Batch, GpuTensor};

fn bench_shape_w16(m: usize, k: usize, n: usize, reps: usize) {
    use ft_kernel_metal::fused::GpuWeightF16;
    let a: Vec<f32> = (0..m * k)
        .map(|i| ((i % 13) as f32) * 0.01 - 0.05)
        .collect();
    let w: Vec<f32> = (0..k * n).map(|i| ((i % 7) as f32) * 0.02 - 0.06).collect();
    let bias: Vec<f32> = (0..n).map(|i| (i % 3) as f32 * 0.1).collect();
    let ga = GpuTensor::upload(&a, m, k).expect("upload a");
    let gw = GpuWeightF16::upload(&w, k, n).expect("upload w16");
    let gb = GpuTensor::upload(&bias, 1, n).expect("upload bias");

    // Warmup + one-element validation (f16-rounded weights).
    let b = Batch::new().expect("batch");
    let out = b.matmul_bias_w16(&ga, &gw, Some(&gb));
    b.finish();
    let got = out.download();
    let mut want = bias[0];
    for e in 0..k {
        let wh = f32::from(half::f16::from_f32(w[e * n]));
        let ah = f32::from(half::f16::from_f32(a[e]));
        want += ah * wh;
    }
    assert!(
        (got[0] - want).abs() <= 5e-2 * (1.0 + want.abs()),
        "matmul_bias_w16[0,0]: {} vs {}",
        got[0],
        want
    );

    let flops = 2.0 * m as f64 * k as f64 * n as f64;
    let mut best_ms = f64::MAX;
    let mut total_ms = 0.0;
    for _ in 0..reps {
        let t = std::time::Instant::now();
        let b = Batch::new().expect("batch");
        let _out = b.matmul_bias_w16(&ga, &gw, Some(&gb));
        b.finish();
        let ms = t.elapsed().as_secs_f64() * 1e3;
        best_ms = best_ms.min(ms);
        total_ms += ms;
    }
    println!(
        "w16 [{m:>5}x{k:>5}]x[{k:>5}x{n:>5}]  best {best_ms:8.2} ms  mean {:8.2} ms  best {:7.1} GFLOPS",
        total_ms / reps as f64,
        flops / (best_ms / 1e3) / 1e9,
    );
}

fn bench_shape(m: usize, k: usize, n: usize, reps: usize) {
    let a: Vec<f32> = (0..m * k)
        .map(|i| ((i % 13) as f32) * 0.01 - 0.05)
        .collect();
    let w: Vec<f32> = (0..k * n).map(|i| ((i % 7) as f32) * 0.02 - 0.06).collect();
    let bias: Vec<f32> = (0..n).map(|i| (i % 3) as f32 * 0.1).collect();
    let ga = GpuTensor::upload(&a, m, k).expect("upload a");
    let gw = GpuTensor::upload(&w, k, n).expect("upload w");
    let gb = GpuTensor::upload(&bias, 1, n).expect("upload bias");

    // Warmup (also validates one output element against a CPU dot).
    let b = Batch::new().expect("batch");
    let out = b.matmul_bias(&ga, &gw, Some(&gb));
    b.finish();
    let got = out.download();
    let mut want = bias[0];
    for e in 0..k {
        want += a[e] * w[e * n];
    }
    assert!(
        (got[0] - want).abs() <= 1e-2 * (1.0 + want.abs()),
        "matmul_bias[0,0]: {} vs {}",
        got[0],
        want
    );

    let flops = 2.0 * m as f64 * k as f64 * n as f64;
    let mut best_ms = f64::MAX;
    let mut total_ms = 0.0;
    for _ in 0..reps {
        let t = std::time::Instant::now();
        let b = Batch::new().expect("batch");
        let _out = b.matmul_bias(&ga, &gw, Some(&gb));
        b.finish();
        let ms = t.elapsed().as_secs_f64() * 1e3;
        best_ms = best_ms.min(ms);
        total_ms += ms;
    }
    println!(
        "[{m:>5}x{k:>5}]x[{k:>5}x{n:>5}]  best {best_ms:8.2} ms  mean {:8.2} ms  best {:7.1} GFLOPS",
        total_ms / reps as f64,
        flops / (best_ms / 1e3) / 1e9,
    );
}

fn main() {
    let reps: usize = std::env::args()
        .nth(1)
        .and_then(|v| v.parse().ok())
        .unwrap_or(5);
    if !ft_kernel_metal::fused::is_available() {
        eprintln!("fused Metal pipelines unavailable on this host");
        std::process::exit(1);
    }
    println!("fused matmul_bias @ turbo encoder shapes, reps={reps}");
    // QKV / attn-out projections.
    bench_shape(1500, 1280, 1280, reps);
    // MLP fc / proj.
    bench_shape(1500, 1280, 5120, reps);
    bench_shape(1500, 5120, 1280, reps);
    // The production layer path (f16-resident weights).
    bench_shape_w16(1500, 1280, 1280, reps);
    bench_shape_w16(1500, 1280, 5120, reps);
    bench_shape_w16(1500, 5120, 1280, reps);
}

//! Same-process, same-worker A/B for the plane-gated batched-transpose materialize lever:
//! OLD = `vec![0.0; batch*plane]` (serial first-touch) + par_chunks_mut transpose (current main);
//! NEW = `ft_kernel_cpu::transpose_batched_materialize_*` (uninit first-touch for small planes,
//! pre-faulted path UNCHANGED for large planes → can't regress). Bit-for-bit equality checked.
//! Run: cargo run --release -p ft-api --example transpose_uninit_ab

use std::time::Instant;

fn old_transpose_f32(src: &[f32], batch: usize, rows: usize, cols: usize) -> Vec<f32> {
    use rayon::prelude::*;
    let plane = rows * cols;
    let total = batch * plane;
    let mut dst = vec![0.0f32; total];
    dst.par_chunks_mut(plane).enumerate().for_each(|(b, dpl)| {
        let so = b * plane;
        ft_kernel_cpu::transpose_2d_into_f32(&src[so..so + plane], dpl, rows, cols);
    });
    dst
}

fn bench<F: Fn() -> usize>(f: F) -> f64 {
    let mut best = f64::INFINITY;
    for _ in 0..9 {
        let t = Instant::now();
        let s = f();
        let el = t.elapsed().as_secs_f64() * 1e3;
        std::hint::black_box(s);
        if el < best {
            best = el;
        }
    }
    best
}

fn main() {
    // (label, batch, rows, cols) — plane = rows*cols. Gate is plane <= 1<<20.
    let cases: Vec<(&str, usize, usize, usize)> = vec![
        ("attn.mT [b=256,256x64]", 256, 256, 64), // plane=16K <=1M -> uninit (WIN expected)
        ("mid [b=64,512x512]", 64, 512, 512),     // plane=262K <=1M -> uninit (neutral)
        ("single [b=1,8192x8192]", 1, 8192, 8192), // plane=64M >1M -> old path (== 1.0x)
        ("few [b=4,4096x4096]", 4, 4096, 4096),   // plane=16M >1M -> old path (== 1.0x, regression gated out)
    ];
    println!("case                       OLD(ms)  NEW(ms)  NEW/OLD   bitmatch   [f32, min-9]");
    for (label, batch, rows, cols) in cases {
        let total = batch * rows * cols;
        let src: Vec<f32> = (0..total).map(|i| (i % 1009) as f32 + 0.25).collect();
        let a = old_transpose_f32(&src, batch, rows, cols);
        let b = ft_kernel_cpu::transpose_batched_materialize_f32(&src, batch, rows, cols);
        let bitmatch = a == b;
        let old_ms = bench(|| old_transpose_f32(&src, batch, rows, cols).len());
        let new_ms =
            bench(|| ft_kernel_cpu::transpose_batched_materialize_f32(&src, batch, rows, cols).len());
        let ratio = old_ms / new_ms;
        println!("  {label:<24} {old_ms:7.3} {new_ms:7.3}   {ratio:5.2}x   bitmatch={bitmatch}");
    }
}

//! A/B for tensor_nonzero. OLD = serial clone+scan+decompose replica; NEW = sess.tensor_nonzero
//! (parallel stream-compaction fast path, numel >= NONZERO_PARALLEL_MIN). bitmatch verifies the
//! parallel per-chunk compaction reproduces the exact serial row-major order.
//! Run: cargo run --release -p ft-api --example nonzero_ab

use ft_api::FrankenTorchSession;
use ft_core::ExecutionMode;
use std::time::Instant;

// Replica of the serial path: for each nonzero flat index, push its ndim decomposed indices.
fn old_nonzero(vals: &[f64], shape: &[usize]) -> Vec<f64> {
    let ndim = shape.len();
    let mut strides = vec![1usize; ndim];
    for i in (0..ndim.saturating_sub(1)).rev() {
        strides[i] = strides[i + 1] * shape[i + 1];
    }
    let mut indices = Vec::new();
    for (flat_idx, &val) in vals.iter().enumerate() {
        if val != 0.0 || val.is_nan() {
            let mut remaining = flat_idx;
            for &s in &strides {
                indices.push((remaining / s) as f64);
                remaining %= s;
            }
        }
    }
    indices
}

fn bench<F: FnMut() -> usize>(mut f: F) -> f64 {
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
    println!("tensor_nonzero f64, min-9:  OLD=serial clone+scan+decompose  NEW=parallel compaction");
    let cases: [(&str, Vec<usize>, usize); 4] = [
        ("2d 4000x4000 ~50%", vec![4000, 4000], 2),
        ("3d 256x256x256 ~50%", vec![256, 256, 256], 2),
        ("2d 8000x2000 ~25%", vec![8000, 2000], 4),
        ("2d 2000x2000 ~50%", vec![2000, 2000], 2),
    ];
    for (label, shape, frac) in cases {
        let numel: usize = shape.iter().product();
        // ~1/frac nonzero, deterministic (nonzero value varies so decompose isn't trivially skipped).
        let vals: Vec<f64> = (0..numel)
            .map(|i| if i % frac == 0 { (i % 97) as f64 + 1.0 } else { 0.0 })
            .collect();

        let mut sess = FrankenTorchSession::new(ExecutionMode::Strict);
        let t = sess.tensor_variable(vals.clone(), shape.clone(), false).unwrap();
        let nz = sess.tensor_nonzero(t).unwrap();
        let new_out = sess.tensor_values(nz).unwrap();
        let old_out = old_nonzero(&vals, &shape);
        let bitmatch = new_out == old_out;
        let nnz = new_out.len() / shape.len();

        let old_ms = bench(|| old_nonzero(&vals, &shape).len());
        let new_ms = bench(|| sess.tensor_nonzero(t).unwrap().0);
        println!(
            "  {label:<22} ({:>4}MB)  OLD {:8.3}  NEW {:8.3}  = {:.2}x  bitmatch={}  nnz={}",
            numel * 8 / (1 << 20),
            old_ms,
            new_ms,
            old_ms / new_ms,
            bitmatch,
            nnz
        );
    }
}

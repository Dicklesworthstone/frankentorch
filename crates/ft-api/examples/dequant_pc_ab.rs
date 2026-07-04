//! A/B for tensor_dequantize_per_channel F64. OLD = exact replica of the pre-fix path (CLONE via
//! to_vec then SERIAL nested dequant loop); NEW = sess.tensor_dequantize_per_channel (borrow + parallel).
//! NOT an apply_function op, so the clone+serial replica faithfully models the real ORIG.
//! Run: cargo run --release -p ft-api --example dequant_pc_ab

use ft_api::FrankenTorchSession;
use ft_core::ExecutionMode;
use std::time::Instant;

fn old_dequant(
    q: &[f64],
    scales: &[f64],
    zps: &[i64],
    stride_before: usize,
    channel_size: usize,
    stride_after: usize,
) -> Vec<f64> {
    let cloned = q.to_vec();
    let mut out = vec![0.0; cloned.len()];
    for before in 0..stride_before {
        for c in 0..channel_size {
            let scale = scales[c];
            let zp = zps[c] as f64;
            for after in 0..stride_after {
                let idx = before * channel_size * stride_after + c * stride_after + after;
                out[idx] = (cloned[idx] - zp) * scale;
            }
        }
    }
    out
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
    println!("tensor_dequantize_per_channel f64, min-9:  OLD=clone + serial  NEW=borrow + parallel");
    let cases: [(&str, usize, usize, usize); 3] =
        [("256x128x1024", 256, 128, 1024), ("128x64x2048", 128, 64, 2048), ("512x256x256", 512, 256, 256)];
    for (label, before, channels, after) in cases {
        let numel = before * channels * after;
        let qd: Vec<f64> = (0..numel).map(|i| ((i % 255) as f64) - 128.0).collect();
        let scales: Vec<f64> = (0..channels).map(|c| 0.02 + (c % 7) as f64 * 0.001).collect();
        let zps: Vec<i64> = (0..channels).map(|c| (c % 5) as i64 - 2).collect();

        let mut sess = FrankenTorchSession::new(ExecutionMode::Strict);
        let it = sess.tensor_variable(qd.clone(), vec![before, channels, after], false).unwrap();
        let out = sess.tensor_dequantize_per_channel(it, &scales, &zps, 1).unwrap();
        let new_out = sess.tensor_values(out).unwrap();
        let old_out = old_dequant(&qd, &scales, &zps, before, channels, after);
        let bitmatch = new_out == old_out;

        let old_ms = bench(|| old_dequant(&qd, &scales, &zps, before, channels, after).len());
        let new_ms = bench(|| sess.tensor_dequantize_per_channel(it, &scales, &zps, 1).unwrap().0);
        println!(
            "  {label:<14} ({:>3}MB)  OLD {:8.3}  NEW {:8.3}  = {:.2}x  bitmatch={}",
            numel * 8 / (1 << 20),
            old_ms,
            new_ms,
            old_ms / new_ms,
            bitmatch
        );
    }
}

// roi_align serial-vs-parallel A/B: run with RAYON_NUM_THREADS=1 (serial-equivalent)
// vs many threads (parallel) in the same process to measure the parallelize-over-
// output-elements speedup. Realistic Faster R-CNN shape. Inputs materialized BEFORE
// Instant::now() so the tensor_variable copy never lands in the timed region.
use ft_api::FrankenTorchSession;
use ft_core::ExecutionMode;
use std::time::Instant;

fn main() {
    let (n, c, h, w) = (1usize, 256usize, 50usize, 50usize);
    let k = 512usize; // proposals
    let (out_h, out_w) = (7usize, 7usize);
    let sampling_ratio = 2usize;
    let spatial_scale = 1.0f64;

    let feat: Vec<f64> = (0..n * c * h * w)
        .map(|i| {
            let x = (i as u64).wrapping_mul(0x9e3779b97f4a7c15).wrapping_add(3);
            (x >> 40) as f64 / (1u64 << 24) as f64
        })
        .collect();
    let mut boxes = Vec::with_capacity(k * 5);
    for r in 0..k {
        let x1 = (r % 40) as f64;
        let y1 = ((r * 7) % 40) as f64;
        boxes.push(0.0); // batch_idx
        boxes.push(x1);
        boxes.push(y1);
        boxes.push(x1 + 8.0 + (r % 5) as f64);
        boxes.push(y1 + 8.0 + (r % 4) as f64);
    }

    let threads = rayon::current_num_threads();
    let mut best = f64::INFINITY;
    for _ in 0..7 {
        let mut s = FrankenTorchSession::new(ExecutionMode::Strict);
        let ff = s.tensor_variable(feat.clone(), vec![n, c, h, w], false).unwrap();
        let bb = s.tensor_variable(boxes.clone(), vec![k, 5], false).unwrap();
        let t0 = Instant::now();
        let ra = s
            .roi_align(ff, bb, (out_h, out_w), spatial_scale, sampling_ratio)
            .unwrap();
        let ms = t0.elapsed().as_secs_f64() * 1e3;
        best = best.min(ms);
        std::hint::black_box((&s, ra));
    }
    println!("[roi_align N={n} C={c} H={h} W={w} K={k} out={out_h}x{out_w} sr={sampling_ratio}] threads={threads}: {best:.2} ms");
}

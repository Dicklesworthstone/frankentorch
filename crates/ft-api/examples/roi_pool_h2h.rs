// roi_pool + ps_roi_pool serial-vs-parallel A/B (RAYON_NUM_THREADS 1 vs many, same
// process). Inputs materialized BEFORE Instant::now().
use ft_api::FrankenTorchSession;
use ft_core::ExecutionMode;
use std::time::Instant;

fn main() {
    let threads = rayon::current_num_threads();

    // roi_pool: [1,256,50,50], 512 ROIs, 7x7 out.
    {
        let (n, c, h, w) = (1usize, 256usize, 50usize, 50usize);
        let k = 512usize;
        let (out_h, out_w) = (7usize, 7usize);
        let feat: Vec<f64> = (0..n * c * h * w)
            .map(|i| ((i as u64).wrapping_mul(0x9e3779b97f4a7c15) >> 40) as f64 / (1u64 << 24) as f64)
            .collect();
        let mut boxes = Vec::with_capacity(k * 5);
        for r in 0..k {
            let x1 = (r % 40) as f64;
            let y1 = ((r * 7) % 40) as f64;
            boxes.extend_from_slice(&[0.0, x1, y1, x1 + 8.0 + (r % 5) as f64, y1 + 8.0 + (r % 4) as f64]);
        }
        let mut best = f64::INFINITY;
        for _ in 0..7 {
            let mut s = FrankenTorchSession::new(ExecutionMode::Strict);
            let ff = s.tensor_variable(feat.clone(), vec![n, c, h, w], false).unwrap();
            let bb = s.tensor_variable(boxes.clone(), vec![k, 5], false).unwrap();
            let t0 = Instant::now();
            let rp = s.roi_pool(ff, bb, (out_h, out_w), 1.0).unwrap();
            best = best.min(t0.elapsed().as_secs_f64() * 1e3);
            std::hint::black_box((&s, rp));
        }
        println!("[roi_pool C={c} H={h} W={w} K={k} out={out_h}x{out_w}] threads={threads}: {best:.2} ms");
    }

    // ps_roi_pool: [1, 21*49, 50, 50], 512 ROIs, output_size 7.
    {
        let (n, h, w) = (1usize, 50usize, 50usize);
        let output_size = 7usize;
        let num_classes = 21usize;
        let c = num_classes * output_size * output_size;
        let k = 512usize;
        let feat: Vec<f64> = (0..n * c * h * w)
            .map(|i| ((i as u64).wrapping_mul(0x9e3779b97f4a7c15) >> 40) as f64 / (1u64 << 24) as f64)
            .collect();
        let mut boxes = Vec::with_capacity(k * 5);
        for r in 0..k {
            let x1 = (r % 40) as f64;
            let y1 = ((r * 7) % 40) as f64;
            boxes.extend_from_slice(&[0.0, x1, y1, x1 + 8.0 + (r % 5) as f64, y1 + 8.0 + (r % 4) as f64]);
        }
        let mut best = f64::INFINITY;
        for _ in 0..7 {
            let mut s = FrankenTorchSession::new(ExecutionMode::Strict);
            let ff = s.tensor_variable(feat.clone(), vec![n, c, h, w], false).unwrap();
            let bb = s.tensor_variable(boxes.clone(), vec![k, 5], false).unwrap();
            let t0 = Instant::now();
            let ps = s.ps_roi_pool(ff, bb, output_size, 1.0).unwrap();
            best = best.min(t0.elapsed().as_secs_f64() * 1e3);
            std::hint::black_box((&s, ps));
        }
        println!("[ps_roi_pool C={c} K={k} osz={output_size} ncls={num_classes}] threads={threads}: {best:.2} ms");
    }
}

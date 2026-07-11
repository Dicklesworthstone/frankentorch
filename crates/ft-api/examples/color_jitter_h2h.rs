// color_jitter serial-vs-parallel A/B (RAYON_NUM_THREADS 1 vs many, same process).
// Batch of images; per-image RNG factors, per-pixel color transform. hue!=0 to
// exercise the heavy HSV path. Inputs materialized BEFORE Instant::now().
use ft_api::FrankenTorchSession;
use ft_core::ExecutionMode;
use std::time::Instant;

fn main() {
    let (b, h, w) = (32usize, 224usize, 224usize);
    let data: Vec<f64> = (0..b * 3 * h * w)
        .map(|i| ((i as u64).wrapping_mul(0x9e3779b97f4a7c15) >> 40) as f64 / (1u64 << 24) as f64)
        .collect();

    let threads = rayon::current_num_threads();
    let mut best = f64::INFINITY;
    for _ in 0..7 {
        let mut s = FrankenTorchSession::new(ExecutionMode::Strict);
        let imgs = s
            .tensor_variable(data.clone(), vec![b, 3, h, w], false)
            .unwrap();
        let t0 = Instant::now();
        let cj = s.color_jitter(imgs, 0.4, 0.4, 0.4, 0.1).unwrap();
        best = best.min(t0.elapsed().as_secs_f64() * 1e3);
        std::hint::black_box((&s, cj));
    }
    println!("[color_jitter B={b} 3x{h}x{w}] threads={threads}: {best:.2} ms");
}

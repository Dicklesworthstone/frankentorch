// farthest_point_sampling serial-vs-parallel A/B (batch-parallel; RAYON_NUM_THREADS
// 1 vs many, same process). Inputs materialized BEFORE Instant::now().
use ft_api::FrankenTorchSession;
use ft_core::ExecutionMode;
use std::time::Instant;

fn main() {
    let (batch, n_points, num_samples) = (16usize, 4096usize, 1024usize);
    let data: Vec<f64> = (0..batch * n_points * 3)
        .map(|i| ((i as u64).wrapping_mul(0x9e3779b97f4a7c15) >> 40) as f64 / (1u64 << 24) as f64)
        .collect();

    let threads = rayon::current_num_threads();
    let mut best = f64::INFINITY;
    for _ in 0..5 {
        let mut s = FrankenTorchSession::new(ExecutionMode::Strict);
        let pts = s
            .tensor_variable(data.clone(), vec![batch, n_points, 3], false)
            .unwrap();
        let t0 = Instant::now();
        let fps = s.farthest_point_sampling(pts, num_samples).unwrap();
        best = best.min(t0.elapsed().as_secs_f64() * 1e3);
        std::hint::black_box((&s, fps));
    }
    println!("[fps B={batch} N={n_points} S={num_samples}] threads={threads}: {best:.2} ms");
}

// group_points serial-vs-parallel A/B (RAYON_NUM_THREADS 1 vs many, same process).
// PointNet++ grouping shape. Inputs materialized BEFORE Instant::now().
use ft_api::FrankenTorchSession;
use ft_core::ExecutionMode;
use std::time::Instant;

fn main() {
    let (batch, n_points, c, m, k) = (16usize, 4096usize, 128usize, 512usize, 32usize);
    let pts: Vec<f64> = (0..batch * n_points * c)
        .map(|i| ((i as u64).wrapping_mul(0x9e3779b97f4a7c15) >> 40) as f64 / (1u64 << 24) as f64)
        .collect();
    let idxs: Vec<f64> = (0..batch * m * k)
        .map(|i| (((i as u64).wrapping_mul(0x9e3779b97f4a7c15) >> 40) % n_points as u64) as f64)
        .collect();

    let threads = rayon::current_num_threads();
    let mut best = f64::INFINITY;
    for _ in 0..7 {
        let mut s = FrankenTorchSession::new(ExecutionMode::Strict);
        let pt = s
            .tensor_variable(pts.clone(), vec![batch, n_points, c], false)
            .unwrap();
        let it = s
            .tensor_variable(idxs.clone(), vec![batch, m, k], false)
            .unwrap();
        let t0 = Instant::now();
        let gp = s.group_points(pt, it).unwrap();
        best = best.min(t0.elapsed().as_secs_f64() * 1e3);
        std::hint::black_box((&s, gp));
    }
    println!(
        "[group_points B={batch} N={n_points} C={c} M={m} K={k}] threads={threads}: {best:.2} ms"
    );
}

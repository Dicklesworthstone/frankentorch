// ball_query serial-vs-parallel A/B (RAYON_NUM_THREADS 1 vs many, same process).
// Inputs materialized BEFORE Instant::now().
use ft_api::FrankenTorchSession;
use ft_core::ExecutionMode;
use std::time::Instant;

fn main() {
    let batch = 16usize;
    let n_points = 4096usize;
    let n_queries = 1024usize;
    let max_samples = 32usize;
    let radius = 0.1f64;

    let pts: Vec<f64> = (0..batch * n_points * 3)
        .map(|i| ((i as u64).wrapping_mul(0x9e3779b97f4a7c15) >> 40) as f64 / (1u64 << 24) as f64)
        .collect();
    let qrs: Vec<f64> = (0..batch * n_queries * 3)
        .map(|i| {
            ((i as u64).wrapping_mul(0x9e3779b97f4a7c15).wrapping_add(777) >> 40) as f64
                / (1u64 << 24) as f64
        })
        .collect();

    let threads = rayon::current_num_threads();
    let mut best = f64::INFINITY;
    for _ in 0..7 {
        let mut s = FrankenTorchSession::new(ExecutionMode::Strict);
        let pt = s.tensor_variable(pts.clone(), vec![batch, n_points, 3], false).unwrap();
        let qt = s.tensor_variable(qrs.clone(), vec![batch, n_queries, 3], false).unwrap();
        let t0 = Instant::now();
        let bq = s.ball_query(pt, qt, radius, max_samples).unwrap();
        best = best.min(t0.elapsed().as_secs_f64() * 1e3);
        std::hint::black_box((&s, bq));
    }
    println!("[ball_query B={batch} N={n_points} M={n_queries} ms_samples={max_samples} r={radius}] threads={threads}: {best:.2} ms");
}

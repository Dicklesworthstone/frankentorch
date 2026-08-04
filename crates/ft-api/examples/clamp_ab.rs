//! Same-worker RAYON_NUM_THREADS A/B for the clamp kernel parallelization ([4000,4000] f64
//! no-grad, NO torch). Run twice: RAYON_NUM_THREADS=1 (≈ old serial) vs =64 (parallel); the
//! 1t/Nt ratio is the parallelization win, robust to a contended worker.
//! Run: RAYON_NUM_THREADS=N cargo run --release -p ft-api --example clamp_ab

use std::time::Instant;

use ft_api::FrankenTorchSession;
use ft_core::ExecutionMode;

const R: usize = 4000;
const C: usize = 4000;

fn main() {
    let data: Vec<f64> = (0..R * C).map(|i| ((i % 19) as f64) - 9.0).collect();
    let mut best = f64::INFINITY;
    for _ in 0..9 {
        let mut s = FrankenTorchSession::new(ExecutionMode::Strict);
        let x = s.tensor_variable(data.clone(), vec![R, C], false).unwrap();
        let t = Instant::now();
        let _ = s.tensor_clamp(x, -3.0, 3.0).unwrap();
        let el = t.elapsed().as_secs_f64() * 1e3;
        if el < best {
            best = el;
        }
    }
    println!(
        "clamp [{R},{C}] f64 no-grad: {best:.3} ms  (threads={})",
        rayon::current_num_threads()
    );
}

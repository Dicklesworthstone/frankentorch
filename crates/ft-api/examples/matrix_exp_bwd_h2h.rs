use ft_api::FrankenTorchSession;
use ft_core::ExecutionMode;
use std::time::Instant;

fn main() {
    let n = 400usize;
    let data: Vec<f64> = (0..n * n)
        .map(|i| {
            let x = (i as u64).wrapping_mul(6364136223846793005).wrapping_add(1);
            ((x >> 33) as f64 / (1u64 << 31) as f64 - 1.0) * 0.05
        })
        .collect();
    let mut best_fwd = f64::INFINITY;
    let mut best_bwd = f64::INFINITY;
    for _ in 0..6 {
        let mut s = FrankenTorchSession::new(ExecutionMode::Strict);
        let a = s.tensor_variable(data.clone(), vec![n, n], true).unwrap();
        let t0 = Instant::now();
        let y = s.tensor_matrix_exp(a).unwrap();
        let loss = s.tensor_sum(y).unwrap();
        let t1 = Instant::now();
        s.tensor_backward(loss).unwrap();
        let t2 = Instant::now();
        let fwd = (t1 - t0).as_secs_f64() * 1e3;
        let bwd = (t2 - t1).as_secs_f64() * 1e3;
        if fwd < best_fwd {
            best_fwd = fwd;
        }
        if bwd < best_bwd {
            best_bwd = bwd;
        }
        std::hint::black_box(&s);
    }
    println!("[FT] matrix_exp [400,400] f64: fwd {best_fwd:.1} ms  bwd {best_bwd:.1} ms");
}

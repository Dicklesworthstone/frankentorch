use std::error::Error;
use std::time::Instant;

use ft_api::FrankenTorchSession;
use ft_core::ExecutionMode;

fn boxed<E: std::fmt::Debug>(err: E) -> std::io::Error {
    std::io::Error::other(format!("{err:?}"))
}

// Batch of well-conditioned matrices [B,n,n]: A = M Mᵀ + n·I (SPD, invertible).
fn spd_batch(b: usize, n: usize) -> Vec<f64> {
    let mut out = vec![0.0_f64; b * n * n];
    for bi in 0..b {
        let base = bi * n * n;
        let m: Vec<f64> = (0..n * n).map(|i| (((i + bi) % 13) as f64 - 6.0) * 0.1).collect();
        for r in 0..n {
            for c in 0..n {
                let mut acc = 0.0;
                for k in 0..n {
                    acc += m[r * n + k] * m[c * n + k];
                }
                out[base + r * n + c] = acc + if r == c { n as f64 } else { 0.0 };
            }
        }
    }
    out
}

fn run_ft(b: usize, n: usize) -> Result<f64, Box<dyn Error>> {
    let mut best = f64::INFINITY;
    for _ in 0..3 {
        let a = spd_batch(b, n);
        let mut s = FrankenTorchSession::new(ExecutionMode::Strict);
        let av = s.tensor_variable(a, vec![b, n, n], false).map_err(boxed)?;
        let start = Instant::now();
        let _inv = s.tensor_linalg_inv(av).map_err(boxed)?;
        let elapsed_ms = start.elapsed().as_secs_f64() * 1e3;
        if elapsed_ms < best { best = elapsed_ms; }
    }
    Ok(best)
}

fn main() -> Result<(), Box<dyn Error>> {
    for (b, n) in [(400usize, 64usize), (1000usize, 64usize)] {
        let ft_ms = run_ft(b, n)?;
        println!("B={b} n={n}: FT linalg_inv {ft_ms:.1} ms");
    }
    Ok(())
}

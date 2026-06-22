use std::error::Error;
use std::time::Instant;

use ft_api::FrankenTorchSession;
use ft_core::ExecutionMode;

fn boxed<E: std::fmt::Debug>(err: E) -> std::io::Error {
    std::io::Error::other(format!("{err:?}"))
}

fn fill(batch: usize, m: usize, n: usize) -> Vec<f64> {
    let mut a = vec![0.0_f64; batch * m * n];
    for plane in 0..batch {
        for r in 0..m {
            for c in 0..n {
                let v = ((((plane + 1) * (r + 2) * (c + 3)) % 17) as f64 - 8.0) * 0.05;
                a[plane * m * n + r * n + c] = v + if r == c { 3.0 } else { 0.0 };
            }
        }
    }
    a
}

fn run_ft(batch: usize, m: usize, n: usize) -> Result<f64, Box<dyn Error>> {
    let mut best = f64::INFINITY;
    for _ in 0..4 {
        let a_data = fill(batch, m, n);
        let b_data: Vec<f64> = (0..batch * m * 4).map(|i| ((i % 11) as f64) * 0.1 - 0.5).collect();
        let mut s = FrankenTorchSession::new(ExecutionMode::Strict);
        let a = s.tensor_variable(a_data, vec![batch, m, n], false).map_err(boxed)?;
        let b = s.tensor_variable(b_data, vec![batch, m, 4], false).map_err(boxed)?;
        let start = Instant::now();
        let _x = s.tensor_linalg_lstsq(a, b).map_err(boxed)?;
        let elapsed_ms = start.elapsed().as_secs_f64() * 1e3;
        if elapsed_ms < best { best = elapsed_ms; }
    }
    Ok(best)
}

fn main() -> Result<(), Box<dyn Error>> {
    for (batch, m, n) in [(2000usize, 32usize, 16usize), (1000usize, 64usize, 32usize), (500usize, 96usize, 48usize)] {
        let ft_ms = run_ft(batch, m, n)?;
        println!("B={batch} m={m} n={n}: FT {ft_ms:.1} ms");
    }
    Ok(())
}

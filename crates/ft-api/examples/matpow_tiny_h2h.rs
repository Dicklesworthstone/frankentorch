use std::error::Error;
use std::time::Instant;

use ft_api::FrankenTorchSession;
use ft_core::ExecutionMode;

fn boxed<E: std::fmt::Debug>(err: E) -> std::io::Error {
    std::io::Error::other(format!("{err:?}"))
}

fn fill(batch: usize, n: usize) -> Vec<f64> {
    let mut a = vec![0.0_f64; batch * n * n];
    for plane in 0..batch {
        for r in 0..n {
            for c in 0..n {
                let v = ((((plane + 1) * (r + 2) * (c + 3)) % 17) as f64 - 8.0) * 0.01;
                a[plane * n * n + r * n + c] = v + if r == c { 1.0 } else { 0.0 };
            }
        }
    }
    a
}

fn run_ft(batch: usize, n: usize, k: u64) -> Result<f64, Box<dyn Error>> {
    let mut best = f64::INFINITY;
    for _ in 0..4 {
        let data = fill(batch, n);
        let mut s = FrankenTorchSession::new(ExecutionMode::Strict);
        let a = s
            .tensor_variable(data, vec![batch, n, n], false)
            .map_err(boxed)?;
        let start = Instant::now();
        let _c = s.tensor_matrix_power(a, k as i32).map_err(boxed)?;
        let elapsed_ms = start.elapsed().as_secs_f64() * 1e3;
        if elapsed_ms < best {
            best = elapsed_ms;
        }
    }
    Ok(best)
}

fn main() -> Result<(), Box<dyn Error>> {
    for (batch, n) in [
        (10000usize, 32usize),
        (5000usize, 64usize),
        (2000usize, 96usize),
    ] {
        let ft_ms = run_ft(batch, n, 16)?;
        println!("B={batch} n={n} k=16: FT {ft_ms:.1} ms");
    }
    Ok(())
}

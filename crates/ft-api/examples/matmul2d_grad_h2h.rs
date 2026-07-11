use std::error::Error;
use std::time::Instant;

use ft_api::FrankenTorchSession;
use ft_core::ExecutionMode;

fn boxed<E: std::fmt::Debug>(err: E) -> std::io::Error {
    std::io::Error::other(format!("{err:?}"))
}

fn fill(n: usize, salt: usize) -> Vec<f64> {
    (0..n)
        .map(|i| (((i + salt) % 17) as f64 - 8.0) * 0.02)
        .collect()
}

fn run_ft(m: usize, k: usize, n: usize) -> Result<f64, Box<dyn Error>> {
    let mut best = f64::INFINITY;
    for _ in 0..3 {
        let ad = fill(m * k, 0);
        let bd = fill(k * n, 7);
        let mut s = FrankenTorchSession::new(ExecutionMode::Strict);
        let a = s.tensor_variable(ad, vec![m, k], true).map_err(boxed)?;
        let b = s.tensor_variable(bd, vec![k, n], true).map_err(boxed)?;
        let start = Instant::now();
        let c = s.tensor_matmul(a, b).map_err(boxed)?;
        let csq = s.tensor_mul(c, c).map_err(boxed)?; // non-uniform incoming grad (2C) -> naive else path
        let loss = s.tensor_sum(csq).map_err(boxed)?;
        s.tensor_backward(loss).map_err(boxed)?;
        let elapsed_ms = start.elapsed().as_secs_f64() * 1e3;
        if elapsed_ms < best {
            best = elapsed_ms;
        }
    }
    Ok(best)
}

fn main() -> Result<(), Box<dyn Error>> {
    for (m, k, n) in [(2048usize, 2048usize, 2048usize), (4096, 1024, 4096)] {
        let ft_ms = run_ft(m, k, n)?;
        println!("M={m} K={k} N={n}: FT {ft_ms:.1} ms");
    }
    Ok(())
}

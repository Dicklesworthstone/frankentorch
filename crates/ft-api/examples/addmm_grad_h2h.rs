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
        let bias = fill(m * n, 3);
        let xd = fill(m * k, 0);
        let wd = fill(k * n, 7);
        let mut s = FrankenTorchSession::new(ExecutionMode::Strict);
        let b = s.tensor_variable(bias, vec![m, n], true).map_err(boxed)?;
        let x = s.tensor_variable(xd, vec![m, k], true).map_err(boxed)?;
        let w = s.tensor_variable(wd, vec![k, n], true).map_err(boxed)?;
        let start = Instant::now();
        let out = s.tensor_addmm(b, x, w, 1.0, 1.0).map_err(boxed)?;
        let sq = s.tensor_mul(out, out).map_err(boxed)?;
        let loss = s.tensor_sum(sq).map_err(boxed)?;
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

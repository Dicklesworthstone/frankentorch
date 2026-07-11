use std::error::Error;
use std::time::Instant;

use ft_api::FrankenTorchSession;
use ft_core::ExecutionMode;

fn boxed<E: std::fmt::Debug>(err: E) -> std::io::Error {
    std::io::Error::other(format!("{err:?}"))
}

fn fill(n: usize, salt: usize) -> Vec<f64> {
    (0..n)
        .map(|i| (((i + salt) % 17) as f64 - 8.0) * 0.05)
        .collect()
}

fn run_ft(b: usize, m: usize, n: usize) -> Result<f64, Box<dyn Error>> {
    let mut best = f64::INFINITY;
    for _ in 0..3 {
        let ad = fill(b * m * n, 0);
        let mut s = FrankenTorchSession::new(ExecutionMode::Strict);
        let a = s.tensor_variable(ad, vec![b, m, n], false).map_err(boxed)?;
        let start = Instant::now();
        let _r = s.tensor_linalg_matrix_rank(a, None).map_err(boxed)?;
        let elapsed_ms = start.elapsed().as_secs_f64() * 1e3;
        if elapsed_ms < best {
            best = elapsed_ms;
        }
    }
    Ok(best)
}

fn main() -> Result<(), Box<dyn Error>> {
    {
        let (b, m, n) = (200usize, 96usize, 64usize);
        let ft_ms = run_ft(b, m, n)?;
        println!("B={b} m={m} n={n}: FT matrix_rank {ft_ms:.1} ms");
    }
    Ok(())
}

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

fn run_ft(b: usize, n: usize) -> Result<f64, Box<dyn Error>> {
    let mut best = f64::INFINITY;
    for _ in 0..3 {
        let ad = fill(b * n * n, 0);
        let td = fill(b * n, 3);
        let cd = fill(b * n * n, 5);
        let mut s = FrankenTorchSession::new(ExecutionMode::Strict);
        let a = s.tensor_variable(ad, vec![b, n, n], false).map_err(boxed)?;
        let tau = s.tensor_variable(td, vec![b, n], false).map_err(boxed)?;
        let c = s.tensor_variable(cd, vec![b, n, n], false).map_err(boxed)?;
        let start = Instant::now();
        let _q = s.tensor_ormqr(a, tau, c, true, false).map_err(boxed)?;
        let elapsed_ms = start.elapsed().as_secs_f64() * 1e3;
        if elapsed_ms < best {
            best = elapsed_ms;
        }
    }
    Ok(best)
}

fn main() -> Result<(), Box<dyn Error>> {
    for (b, n) in [(300usize, 64usize), (1000usize, 64usize)] {
        let ft_ms = run_ft(b, n)?;
        println!("B={b} n={n}: FT {ft_ms:.1} ms");
    }
    Ok(())
}

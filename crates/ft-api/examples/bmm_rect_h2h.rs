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

fn run_ft(bh: usize, m: usize, k: usize, n: usize) -> Result<f64, Box<dyn Error>> {
    let mut best = f64::INFINITY;
    for _ in 0..4 {
        let ad = fill(bh * m * k, 0);
        let bd = fill(bh * k * n, 7);
        let mut s = FrankenTorchSession::new(ExecutionMode::Strict);
        let a = s
            .tensor_variable(ad, vec![bh, m, k], false)
            .map_err(boxed)?;
        let b = s
            .tensor_variable(bd, vec![bh, k, n], false)
            .map_err(boxed)?;
        let start = Instant::now();
        let _c = s.tensor_matmul(a, b).map_err(boxed)?;
        let elapsed_ms = start.elapsed().as_secs_f64() * 1e3;
        if elapsed_ms < best {
            best = elapsed_ms;
        }
    }
    Ok(best)
}

fn main() -> Result<(), Box<dyn Error>> {
    for (bh, m, k, n) in [
        (4096usize, 128usize, 64usize, 128usize),
        (2048, 256, 64, 256),
        (4096, 64, 128, 64),
    ] {
        let ft_ms = run_ft(bh, m, k, n)?;
        println!("BH={bh} {m}x{k}x{n}: FT {ft_ms:.1} ms");
    }
    Ok(())
}

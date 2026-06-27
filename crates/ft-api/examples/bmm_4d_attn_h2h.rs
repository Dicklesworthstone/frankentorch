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

fn run_ft(b: usize, h: usize, s_: usize, d: usize) -> Result<f64, Box<dyn Error>> {
    let mut best = f64::INFINITY;
    for _ in 0..4 {
        let ad = fill(b * h * s_ * d, 0);
        let bd = fill(b * h * d * s_, 7);
        let mut sess = FrankenTorchSession::new(ExecutionMode::Strict);
        let a = sess
            .tensor_variable(ad, vec![b, h, s_, d], false)
            .map_err(boxed)?;
        let bb = sess
            .tensor_variable(bd, vec![b, h, d, s_], false)
            .map_err(boxed)?;
        let start = Instant::now();
        let _c = sess.tensor_matmul(a, bb).map_err(boxed)?;
        let elapsed_ms = start.elapsed().as_secs_f64() * 1e3;
        if elapsed_ms < best {
            best = elapsed_ms;
        }
    }
    Ok(best)
}

fn main() -> Result<(), Box<dyn Error>> {
    for (b, h, s_, d) in [
        (64usize, 16usize, 128usize, 64usize),
        (32, 16, 256, 64),
        (128, 8, 128, 64),
    ] {
        let ft_ms = run_ft(b, h, s_, d)?;
        println!("[{b},{h},{s_},{d}]: FT {ft_ms:.1} ms");
    }
    Ok(())
}

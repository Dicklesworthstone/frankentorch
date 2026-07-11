use ft_api::FrankenTorchSession;
use ft_core::ExecutionMode;
use std::error::Error;
use std::time::Instant;
fn boxed<E: std::fmt::Debug>(e: E) -> std::io::Error {
    std::io::Error::other(format!("{e:?}"))
}
fn fill(n: usize, salt: usize) -> Vec<f64> {
    (0..n)
        .map(|i| (((i + salt) % 17) as f64 - 8.0) * 0.05)
        .collect()
}
fn main() -> Result<(), Box<dyn Error>> {
    let (b, n) = (150usize, 96usize);
    let mut best = f64::INFINITY;
    for _ in 0..3 {
        let ad = fill(b * n * n, 0);
        let mut s = FrankenTorchSession::new(ExecutionMode::Strict);
        let a = s.tensor_variable(ad, vec![b, n, n], false).map_err(boxed)?;
        let start = Instant::now();
        let _r = s.tensor_linalg_eigvals(a).map_err(boxed)?;
        let ms = start.elapsed().as_secs_f64() * 1e3;
        if ms < best {
            best = ms;
        }
    }
    println!("B={b} n={n}: FT eigvals {best:.1} ms");
    Ok(())
}

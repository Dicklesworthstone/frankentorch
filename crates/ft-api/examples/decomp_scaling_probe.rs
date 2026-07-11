use ft_api::FrankenTorchSession;
use ft_core::ExecutionMode;
use std::error::Error;
use std::time::Instant;
fn boxed<E: std::fmt::Debug>(e: E) -> std::io::Error {
    std::io::Error::other(format!("{e:?}"))
}
fn fill(n: usize) -> Vec<f64> {
    (0..n)
        .map(|i| (((i * 2654435761usize) % 100003) as f64 / 100003.0) - 0.5)
        .collect()
}
fn main() -> Result<(), Box<dyn Error>> {
    let (b, n) = (150usize, 96usize);
    for op in ["qr", "svd", "eigh"] {
        let mut best = f64::INFINITY;
        for _ in 0..3 {
            let ad = fill(b * n * n);
            let mut s = FrankenTorchSession::new(ExecutionMode::Strict);
            let a = s.tensor_variable(ad, vec![b, n, n], false).map_err(boxed)?;
            let st = Instant::now();
            match op {
                "qr" => {
                    s.tensor_linalg_qr(a, true).map_err(boxed)?;
                }
                "svd" => {
                    s.tensor_linalg_svd(a, false).map_err(boxed)?;
                }
                _ => {
                    s.tensor_linalg_eigh(a).map_err(boxed)?;
                }
            };
            let ms = st.elapsed().as_secs_f64() * 1e3;
            if ms < best {
                best = ms;
            }
        }
        println!("{op}: FT {best:.1} ms");
    }
    Ok(())
}

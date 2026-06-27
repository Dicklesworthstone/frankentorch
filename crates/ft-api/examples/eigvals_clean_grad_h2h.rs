use std::error::Error;
use std::time::Instant;

use ft_api::FrankenTorchSession;
use ft_core::ExecutionMode;

fn boxed<E: std::fmt::Debug>(err: E) -> std::io::Error {
    std::io::Error::other(format!("{err:?}"))
}

// Upper-triangular with diagonal 1..N (real, distinct eigenvalues), matching the torch fixture.
fn fill(batch: usize, n: usize) -> Vec<f64> {
    let mut a = vec![0.0_f64; batch * n * n];
    for plane in 0..batch {
        for r in 0..n {
            for c in 0..n {
                if c > r {
                    let v = ((((plane + 1) * (r + 2) * (c + 3)) % 13) as f64 - 6.0) * 0.05;
                    a[plane * n * n + r * n + c] = v;
                } else if c == r {
                    a[plane * n * n + r * n + c] = (r + 1) as f64;
                }
            }
        }
    }
    a
}

fn run_ft(batch: usize, n: usize) -> Result<f64, Box<dyn Error>> {
    let mut best = f64::INFINITY;
    for _ in 0..3 {
        let data = fill(batch, n);
        let mut s = FrankenTorchSession::new(ExecutionMode::Strict);
        let a = s
            .tensor_variable(data, vec![batch, n, n], true)
            .map_err(boxed)?;
        let start = Instant::now();
        let ev = s.tensor_linalg_eigvals(a).map_err(boxed)?;
        let sq = s.tensor_mul(ev, ev).map_err(boxed)?;
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
    for (batch, n) in [
        (2000usize, 16usize),
        (1000usize, 32usize),
        (500usize, 64usize),
    ] {
        let ft_ms = run_ft(batch, n)?;
        println!("B={batch} n={n}: FT {ft_ms:.1} ms");
    }
    Ok(())
}

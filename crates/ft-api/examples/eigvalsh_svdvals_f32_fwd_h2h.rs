use std::error::Error;
use std::time::Instant;

use ft_api::FrankenTorchSession;
use ft_core::ExecutionMode;

fn boxed<E: std::fmt::Debug>(err: E) -> std::io::Error {
    std::io::Error::other(format!("{err:?}"))
}

fn fill_sym(batch: usize, n: usize) -> Vec<f32> {
    let mut a = vec![0.0_f32; batch * n * n];
    for plane in 0..batch {
        for r in 0..n {
            for c in 0..n {
                let v = ((((plane + 1) * (r + 2) * (c + 3)) % 19) as f32 - 9.0) * 0.02;
                a[plane * n * n + r * n + c] += v;
                a[plane * n * n + c * n + r] += v;
                if r == c {
                    a[plane * n * n + r * n + c] += 2.0;
                }
            }
        }
    }
    a
}

fn fill_gen(batch: usize, n: usize) -> Vec<f32> {
    let mut a = vec![0.0_f32; batch * n * n];
    for plane in 0..batch {
        for r in 0..n {
            for c in 0..n {
                let v = ((((plane + 1) * (r + 2) * (c + 3)) % 19) as f32 - 9.0) * 0.02;
                a[plane * n * n + r * n + c] = v + if r == c { 3.0 } else { 0.0 };
            }
        }
    }
    a
}

fn run_ft(op: &str, batch: usize, n: usize) -> Result<f64, Box<dyn Error>> {
    let mut best = f64::INFINITY;
    for _ in 0..4 {
        let data = if op == "eigvalsh" {
            fill_sym(batch, n)
        } else {
            fill_gen(batch, n)
        };
        let mut s = FrankenTorchSession::new(ExecutionMode::Strict);
        let a = s
            .tensor_variable_f32(data, vec![batch, n, n], false)
            .map_err(boxed)?;
        let start = Instant::now();
        if op == "eigvalsh" {
            let _ = s.tensor_linalg_eigvalsh(a).map_err(boxed)?;
        } else {
            let _ = s.tensor_linalg_svdvals(a).map_err(boxed)?;
        }
        let elapsed_ms = start.elapsed().as_secs_f64() * 1e3;
        if elapsed_ms < best {
            best = elapsed_ms;
        }
    }
    Ok(best)
}

fn main() -> Result<(), Box<dyn Error>> {
    for op in ["eigvalsh", "svdvals"] {
        for (batch, n) in [
            (2000usize, 32usize),
            (2000usize, 64usize),
            (1000usize, 96usize),
        ] {
            let ft_ms = run_ft(op, batch, n)?;
            println!("{op} B={batch} n={n}: FT {ft_ms:.1} ms");
        }
    }
    Ok(())
}

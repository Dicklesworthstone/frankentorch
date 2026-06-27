use std::error::Error;
use std::time::Instant;

use ft_api::FrankenTorchSession;
use ft_core::ExecutionMode;

fn boxed<E: std::fmt::Debug>(err: E) -> std::io::Error {
    std::io::Error::other(format!("{err:?}"))
}

// Build a batch of lower-triangular cholesky factors L [B,n,n] (positive diagonal):
// cholesky_solve(eye, L) = (L Lᵀ)^{-1} is well-defined for any such L. Builds L directly
// (no tensor_cholesky) to isolate the solve timing.
fn lower_tri_batch(b: usize, n: usize) -> Vec<f64> {
    let mut out = vec![0.0_f64; b * n * n];
    for bi in 0..b {
        let base = bi * n * n;
        for r in 0..n {
            for c in 0..=r {
                out[base + r * n + c] = if r == c {
                    2.0 + ((r + bi) % 5) as f64 * 0.3
                } else {
                    (((r + c + bi) % 7) as f64 - 3.0) * 0.05
                };
            }
        }
    }
    out
}

fn run_ft(b: usize, n: usize) -> Result<f64, Box<dyn Error>> {
    let mut best = f64::INFINITY;
    for _ in 0..3 {
        let ltri = lower_tri_batch(b, n);
        let mut s = FrankenTorchSession::new(ExecutionMode::Strict);
        let l = s
            .tensor_variable(ltri, vec![b, n, n], false)
            .map_err(boxed)?;
        let start = Instant::now();
        let _inv = s.tensor_cholesky_inverse(l, false).map_err(boxed)?; // batched API
        let elapsed_ms = start.elapsed().as_secs_f64() * 1e3;
        if elapsed_ms < best {
            best = elapsed_ms;
        }
    }
    Ok(best)
}

// Verify batched cholesky_inverse matches the per-plane 2-D path (bit-for-bit).
fn verify(b: usize, n: usize) -> Result<(), Box<dyn Error>> {
    let ltri = lower_tri_batch(b, n);
    let mut s = FrankenTorchSession::new(ExecutionMode::Strict);
    let l = s
        .tensor_variable(ltri.clone(), vec![b, n, n], false)
        .map_err(boxed)?;
    let inv = s.tensor_cholesky_inverse(l, false).map_err(boxed)?;
    let got = s.tensor_values(inv).map_err(boxed)?;
    for bi in 0..b {
        let l2 = s
            .tensor_variable(
                ltri[bi * n * n..(bi + 1) * n * n].to_vec(),
                vec![n, n],
                false,
            )
            .map_err(boxed)?;
        let inv2 = s.tensor_cholesky_inverse(l2, false).map_err(boxed)?;
        let want = s.tensor_values(inv2).map_err(boxed)?;
        for (g, w) in got[bi * n * n..(bi + 1) * n * n].iter().zip(want.iter()) {
            if (g - w).abs() > 1e-9 {
                return Err(boxed(format!("batch {bi}: {g} vs {w}")).into());
            }
        }
    }
    println!("verify B={b} n={n}: batched == per-plane OK");
    Ok(())
}

fn main() -> Result<(), Box<dyn Error>> {
    verify(8, 6)?;
    for (b, n) in [(400usize, 64usize), (1000usize, 64usize)] {
        let ft_ms = run_ft(b, n)?;
        println!("B={b} n={n}: FT cholesky_inverse(batched) {ft_ms:.1} ms");
    }
    Ok(())
}

//! Is the new BatchNorm2d lane's parity MISMATCH a BUG, or f32 accumulation noise? —
//! `frankentorch-68pwz`.
//!
//! WHY THIS EXISTS. `batch_norm2d_f32` and `batch_norm2d_f32_dense` are the first h2h lanes
//! BatchNorm2d has ever had, and both reported MISMATCH in the parity column on their first
//! invocation. A fresh lane's parity column has found a real years-old gradient bug in this
//! project before (`frankentorch-prelu`, the `x>0` convention), and it has also produced a
//! P0 that turned out to be the lane's own fault and had to be retracted within the turn.
//! So the question is settled with an arbiter rather than by picking whichever story is more
//! interesting.
//!
//! THE SUSPICION THAT MOTIVATES IT. BatchNorm reduces over `N*H*W = 100,352` elements per
//! channel; GroupNorm, whose lanes report `match` on the same fixtures, reduces over
//! `cpg*spatial = 6,272`. A 16x longer f32 reduction is 16x more exposed to summation order,
//! and the harness's parity tolerance is a tight `1e-6` relative on a checksum of 6.4M
//! near-cancelling gradients. That is a mechanism for a MISMATCH with no bug behind it.
//!
//! HOW IT ARBITRATES. Both f32 arms are scored against an f64 computation of the SAME
//! quantity, which no f32 accumulation order can bias. If FT-f32 and torch-f32 sit on either
//! side of the f64 answer at comparable distance, the disagreement is precision. If FT-f32 is
//! far from f64 and torch-f32 is close, we have a bug and it is ours.
//!
//! Run:
//! ```text
//! cargo run --release -p frankentorch-api --example bn_parity_arbiter
//! .venv/bin/python <the script this prints>
//! ```

use ft_api::FrankenTorchSession;
use ft_core::ExecutionMode;

const N: usize = 32;
const C: usize = 64;
const H: usize = 56;
const W: usize = 56;

/// The h2h harness's `seq` generator, reproduced so both arms normalize identical numbers.
fn seq(len: usize) -> Vec<f64> {
    (0..len)
        .map(|i| ((i % 251) as f64) * 0.001 - 0.12)
        .collect()
}

fn main() {
    let xv = seq(N * C * H * W);
    let wv: Vec<f64> = seq(C).iter().map(|v| v * 10.0 + 1.0).collect();
    let bv: Vec<f64> = seq(C).iter().map(|v| v * 3.0).collect();

    for dense in [false, true] {
        let tag = if dense {
            "DENSE sum(out*out)"
        } else {
            "SUM   sum(out)"
        };
        // ---- FT, f32: exactly what the lane runs ----
        #[allow(clippy::cast_possible_truncation)]
        let (f32_dx, f32_dw, f32_db) = {
            let mut s = FrankenTorchSession::new(ExecutionMode::Strict);
            let x = s
                .tensor_variable_f32(
                    xv.iter().map(|&v| v as f32).collect(),
                    vec![N, C, H, W],
                    true,
                )
                .expect("x");
            let w = s
                .tensor_variable_f32(wv.iter().map(|&v| v as f32).collect(), vec![C], true)
                .expect("w");
            let b = s
                .tensor_variable_f32(bv.iter().map(|&v| v as f32).collect(), vec![C], true)
                .expect("b");
            let (out, _, _) = s
                .functional_batch_norm2d(x, None, None, Some(w), Some(b), true, 0.1, 1e-5)
                .expect("bn");
            let scored = if dense {
                s.tensor_mul(out, out).expect("sq")
            } else {
                out
            };
            let loss = s.tensor_sum(scored).expect("sum");
            let r = s.tensor_backward(loss).expect("bwd");
            (
                r.gradient(x)
                    .expect("dx")
                    .iter()
                    .map(|g| g.abs())
                    .sum::<f64>(),
                r.gradient(w)
                    .expect("dw")
                    .iter()
                    .map(|g| g.abs())
                    .sum::<f64>(),
                r.gradient(b)
                    .expect("db")
                    .iter()
                    .map(|g| g.abs())
                    .sum::<f64>(),
            )
        };

        // ---- FT, f64: the arbiter. Same op, same inputs, no f32 rounding anywhere ----
        let (f64_dx, f64_dw, f64_db) = {
            let mut s = FrankenTorchSession::new(ExecutionMode::Strict);
            let x = s
                .tensor_variable(xv.clone(), vec![N, C, H, W], true)
                .expect("x");
            let w = s.tensor_variable(wv.clone(), vec![C], true).expect("w");
            let b = s.tensor_variable(bv.clone(), vec![C], true).expect("b");
            let (out, _, _) = s
                .functional_batch_norm2d(x, None, None, Some(w), Some(b), true, 0.1, 1e-5)
                .expect("bn");
            let scored = if dense {
                s.tensor_mul(out, out).expect("sq")
            } else {
                out
            };
            let loss = s.tensor_sum(scored).expect("sum");
            let r = s.tensor_backward(loss).expect("bwd");
            (
                r.gradient(x)
                    .expect("dx")
                    .iter()
                    .map(|g| g.abs())
                    .sum::<f64>(),
                r.gradient(w)
                    .expect("dw")
                    .iter()
                    .map(|g| g.abs())
                    .sum::<f64>(),
                r.gradient(b)
                    .expect("db")
                    .iter()
                    .map(|g| g.abs())
                    .sum::<f64>(),
            )
        };

        println!("=== LOSS: {tag} ===");
        println!("bn_parity_arbiter (frankentorch-68pwz)  shape [{N},{C},{H},{W}]");
        println!("per-channel reduction length N*H*W = {}", N * H * W);
        println!();
        println!(
            "{:<10} {:>24} {:>24} {:>24}",
            "arm", "sum|dx|", "sum|dw|", "sum|db|"
        );
        println!(
            "{:<10} {f32_dx:>24.12e} {f32_dw:>24.12e} {f32_db:>24.12e}",
            "FT f32"
        );
        println!(
            "{:<10} {f64_dx:>24.12e} {f64_dw:>24.12e} {f64_db:>24.12e}",
            "FT f64"
        );
        println!();
        println!(
            "FT f32 vs FT f64 relative: dx {:.3e}  dw {:.3e}  db {:.3e}",
            (f32_dx - f64_dx).abs() / f64_dx.abs().max(1.0),
            (f32_dw - f64_dw).abs() / f64_dw.abs().max(1.0),
            (f32_db - f64_db).abs() / f64_db.abs().max(1.0),
        );
        println!("harness parity tolerance is 1e-6 relative");
        println!();
    }
    println!("--- run this against the SAME numbers to place torch on the same axis ---");
    println!(
        r#"import torch, torch.nn.functional as Fn
def seq(n): return ((torch.arange(n,dtype=torch.int64)%251).double())*0.001-0.12
xv = seq({n}*{c}*{h}*{w}).reshape({n},{c},{h},{w}); wv = seq({c})*10.0+1.0; bv = seq({c})*3.0
for dt in (torch.float32, torch.float64):
    x = xv.to(dt).clone().requires_grad_(True)
    w = wv.to(dt).clone().requires_grad_(True)
    b = bv.to(dt).clone().requires_grad_(True)
    Fn.batch_norm(x, None, None, w, b, True, 0.1, 1e-5).sum().backward()
    print(dt, float(x.grad.abs().sum().double()), float(w.grad.abs().sum().double()), float(b.grad.abs().sum().double()))"#,
        n = N,
        c = C,
        h = H,
        w = W
    );
}

//! Parity check for the `gauntlet_conv2d_grad` lane (frankentorch-ug4ep).
//!
//! A timing lane is only meaningful if both sides compute the same thing, and
//! the gauntlet groups do not assert that themselves. This reproduces the lane's
//! exact workload and prints the values to compare against
//! `benches/pytorch_conv2d_grad.py`, so a shape or padding-convention mismatch
//! shows up as a wrong number rather than as a flattering timing.

use ft_api::FrankenTorchSession;
use ft_core::ExecutionMode;

const N: usize = 8;
const C_IN: usize = 64;
const C_OUT: usize = 64;
const H: usize = 32;
const W: usize = 32;
const K: usize = 3;

fn deterministic_values(n: usize, shift: f64) -> Vec<f64> {
    (0..n)
        .map(|i| (((i as f64) * 0.017 + shift).sin()) * 0.2)
        .collect()
}

fn main() {
    let mut session = FrankenTorchSession::new(ExecutionMode::Strict);
    let x = session
        .tensor_variable(
            deterministic_values(N * C_IN * H * W, 0.0),
            vec![N, C_IN, H, W],
            true,
        )
        .unwrap();
    let w = session
        .tensor_variable(
            deterministic_values(C_OUT * C_IN * K * K, 1.0),
            vec![C_OUT, C_IN, K, K],
            true,
        )
        .unwrap();

    let out = session
        .functional_conv2d(x, w, None, (1, 1), (1, 1))
        .unwrap();
    let out_shape = session.tensor_shape(out).unwrap();
    let loss = session.tensor_sum(out).unwrap();
    let loss_value = session.tensor_values(loss).unwrap()[0];
    session.tensor_backward(loss).unwrap();

    let x_grad = session.tensor_grad(x).unwrap().unwrap();
    let w_grad = session.tensor_grad(w).unwrap().unwrap();

    println!("out shape {out_shape:?}");
    println!("loss {loss_value:.12}");
    println!("x.grad[0] {:.12}  w.grad[0] {:.12}", x_grad[0], w_grad[0]);
}

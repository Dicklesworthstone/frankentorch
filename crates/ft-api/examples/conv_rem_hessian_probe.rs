//! Status probe for the remaining conv double-backward variants vs torch
//! (frankentorch-lboou): conv_transpose1d / conv_transpose2d (dilated) /
//! conv2d (dilated+grouped). Prints input-Hessian diag or ERR.
//!   cargo run -q --release -p ft-api --example conv_rem_hessian_probe
use ft_api::FrankenTorchSession;
use ft_autograd::{AutogradError, TensorNodeId};
use ft_core::ExecutionMode;

fn run(
    name: &str,
    xv: Vec<f64>,
    xs: Vec<usize>,
    wv: Vec<f64>,
    ws: Vec<usize>,
    f: impl Fn(&mut FrankenTorchSession, TensorNodeId, TensorNodeId) -> Result<TensorNodeId, AutogradError>,
    k: usize,
) {
    let mut s = FrankenTorchSession::new(ExecutionMode::Strict);
    let n: usize = xs.iter().product();
    let x = s.tensor_variable(xv, xs, true).unwrap();
    let w = s.tensor_variable(wv, ws, false).unwrap();
    match (|| {
        let y = f(&mut s, x, w)?;
        let sq = s.tensor_mul(y, y)?;
        let loss = s.tensor_sum(sq)?;
        s.tensor_functional_hessian(loss, x)
    })() {
        Ok(h) => {
            let d: Vec<f64> = (0..k.min(n)).map(|i| (h[i * n + i] * 1e5).round() / 1e5).collect();
            println!("{name}: {d:?}");
        }
        Err(e) => println!("{name}: ERR {e:?}"),
    }
}

fn main() {
    run(
        "ct1d",
        vec![0.1, 0.2, -0.1],
        vec![1, 1, 3],
        vec![0.2, -0.4, 0.1],
        vec![1, 1, 3],
        |s, x, w| s.functional_conv_transpose1d(x, w, None, 1, 0, 0),
        3,
    );
    run(
        "ct2d_dil",
        vec![0.1, 0.2, -0.1, 0.3],
        vec![1, 1, 2, 2],
        vec![0.2, -0.4, 0.1, 0.3],
        vec![1, 1, 2, 2],
        |s, x, w| s.functional_conv_transpose2d_dilated(x, w, None, (1, 1), (0, 0), (0, 0), (2, 2), 1),
        4,
    );
    run(
        "c2d_dil_grp",
        (0..50).map(|i| (i % 7) as f64 * 0.1 - 0.3).collect(),
        vec![1, 2, 5, 5],
        (0..8).map(|i| (i % 5) as f64 * 0.2 - 0.4).collect(),
        vec![2, 1, 2, 2],
        |s, x, w| s.functional_conv2d_dilated(x, w, None, (1, 1), (0, 0), (2, 2), 2),
        4,
    );
}

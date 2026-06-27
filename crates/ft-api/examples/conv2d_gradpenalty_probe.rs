//! WGAN-GP-style double-backward probe for conv2d vs torch (frankentorch-j4uio).
//! loss = sum(conv2d(x,w,b)^2); g = grad(loss, x, create_graph=True);
//! pen = sum(g^2); then grad(pen, [w, b]). Exercises the create_graph cg path's
//! dweight + dbias adjoints (the input-Hessian probe only covers dpadded).
//!   cargo run -q --release -p ft-api --example conv2d_gradpenalty_probe
use ft_api::FrankenTorchSession;
use ft_autograd::BackwardOptions;
use ft_core::ExecutionMode;

fn main() {
    let xv: Vec<f64> = (0..2 * 5 * 5).map(|i| (i % 7) as f64 * 0.1 - 0.3).collect();
    let wv: Vec<f64> = (0..3 * 3 * 3).map(|i| (i % 5) as f64 * 0.2 - 0.4).collect();
    let bv: Vec<f64> = vec![0.1, -0.2, 0.3];
    let mut s = FrankenTorchSession::new(ExecutionMode::Strict);
    let x = s.tensor_variable(xv, vec![2, 1, 5, 5], true).unwrap();
    let w = s.tensor_variable(wv, vec![3, 1, 3, 3], true).unwrap();
    let b = s.tensor_variable(bv, vec![3], true).unwrap();

    let y = s.functional_conv2d(x, w, Some(b), (1, 1), (0, 0)).unwrap();
    let sq = s.tensor_mul(y, y).unwrap();
    let loss = s.tensor_sum(sq).unwrap();

    let opts = BackwardOptions::for_mode(s.mode())
        .with_create_graph(true)
        .with_retain_graph(true);
    let report1 = s.tensor_backward_with_options(loss, opts).unwrap();
    let g = report1.gradient_node(x).expect("loss depends on x");

    let g2 = s.tensor_mul(g, g).unwrap();
    let pen = s.tensor_sum(g2).unwrap();
    let pen_val = s.tensor_values(pen).unwrap()[0];

    let grads = s
        .tensor_autograd_grad(&[pen], &[w, b], None, false, false)
        .unwrap();
    let gw = grads[0].as_ref().unwrap();
    let gb = grads[1].as_ref().unwrap();
    let gw_sum: f64 = gw.iter().sum();
    println!("pen={pen_val:.6}");
    println!(
        "gw_sum={:.6} gw[0,0,0,0]={:.6} gw[2,0,2,2]={:.6}",
        gw_sum, gw[0], gw[26]
    );
    println!("gb=[{:.4}, {:.4}, {:.4}]", gb[0], gb[1], gb[2]);
}

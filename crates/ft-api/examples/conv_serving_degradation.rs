//! Conv2d reused-session serving degradation (BlackThrush). gmuml is conv-SPECIFIC
//! (SDPA serving was flat at 2.4GB, 2026-06-21an). The conv-pad mitigation shipped, but
//! "conv reshape nodes (input_4d/weight/output) STILL leak" — so conv2d serving may still
//! degrade where SDPA didn't. Measure per-inference time early vs late in ONE session.
//!
//! Run: cargo run --release -p ft-api --example conv_serving_degradation

use std::time::Instant;

use ft_api::FrankenTorchSession;
use ft_core::ExecutionMode;

const N_B: usize = 8;
const C_IN: usize = 32;
const C_OUT: usize = 32;
const H: usize = 64;
const W: usize = 64;
const K: usize = 3;

fn vals(n: usize, shift: f64) -> Vec<f64> {
    (0..n)
        .map(|i| ((i % 251) as f64) * 0.001 - 0.12 + shift)
        .collect()
}

fn main() {
    let n: usize = std::env::var("N")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(100);
    let x = vals(N_B * C_IN * H * W, 0.0);
    let w = vals(C_OUT * C_IN * K * K, 0.1);
    let x_shape = vec![N_B, C_IN, H, W];
    let w_shape = vec![C_OUT, C_IN, K, K];

    let mut s = FrankenTorchSession::new(ExecutionMode::Strict);
    let mut per_iter = Vec::with_capacity(n);
    for _ in 0..n {
        let t = Instant::now();
        let xn = s
            .tensor_variable(x.clone(), x_shape.clone(), false)
            .unwrap();
        let wn = s
            .tensor_variable(w.clone(), w_shape.clone(), false)
            .unwrap();
        let out = s.functional_conv2d(xn, wn, None, (1, 1), (1, 1)).unwrap();
        let _: f64 = s.tensor_values(out).unwrap().iter().sum();
        per_iter.push(t.elapsed().as_secs_f64() * 1e3);
    }

    let avg = |sl: &[f64]| sl.iter().sum::<f64>() / sl.len() as f64;
    let first10 = avg(&per_iter[..10.min(n)]);
    let last10 = avg(&per_iter[n.saturating_sub(10)..]);
    println!(
        "reused-session conv2d serving [{N_B},{C_IN}->{C_OUT},{H},{W},k{K}], {n} inferences in ONE session:"
    );
    println!("  iter[0..10] avg   : {first10:8.3} ms");
    println!(
        "  iter[{}..{}] avg : {:8.3} ms",
        n.saturating_sub(10),
        n,
        last10
    );
    println!(
        "  degradation       : {:.2}x (last10/first10)",
        last10 / first10
    );
    if last10 > 1.5 * first10 {
        println!(
            "  => CONV SERVING DEGRADATION confirmed (gmuml conv reshape-node leak). Lever: conv-pad-recipe for the reshape nodes (build in local Vec, lazy session node)."
        );
    } else {
        println!(
            "  => conv serving FLAT — conv gmuml is tamed (pad mitigation suffices; reshape nodes don't degrade at this scale)."
        );
    }
}

//! rrelu no-grad inference A/B (cc): borrowed-inputs vs the prior full-input clone. rrelu(eval)
//! is a trivially cheap op so the clone dominated. Inputs built once; time op+read per iter.
//! Run: cargo run --release -p ft-api --example rrelu_nograd_ab
use ft_api::FrankenTorchSession;
use ft_core::ExecutionMode;
use std::time::Instant;
const ROWS: usize = 2048;
const COLS: usize = 1024;
fn main() {
    let iters: usize = std::env::var("ITERS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(80);
    let mut s = FrankenTorchSession::new(ExecutionMode::Strict);
    let v: Vec<f64> = (0..ROWS * COLS)
        .map(|i| ((i as f64) * 0.001).sin())
        .collect();
    let x = s.tensor_variable(v, vec![ROWS, COLS], false).unwrap();
    let op = |s: &mut FrankenTorchSession| -> f64 {
        let o = s.tensor_rrelu(x, 0.125, 0.333).unwrap();
        s.tensor_values(o).unwrap().iter().map(|v| v.abs()).sum()
    };
    for _ in 0..3 {
        let _ = op(&mut s);
    }
    let mut t = Vec::with_capacity(iters);
    let mut chk = 0.0;
    for _ in 0..iters {
        let s0 = Instant::now();
        chk = op(&mut s);
        t.push(s0.elapsed().as_secs_f64() * 1e3);
    }
    t.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let tot: f64 = t.iter().sum();
    println!(
        "rrelu no-grad [{ROWS}x{COLS}] {iters} iters: min {:.3} ms  total {tot:.1} ms  checksum {chk:.6e}",
        t[0]
    );
}

//! A/B for tensor_prelu F64 (scalar weight). OLD = replica of the pre-fix fallback (CLONE input via
//! to_vec — matches tensor_values(input) — then the serial prelu); NEW = sess.tensor_prelu (F64
//! borrows input + weight). bitmatch verifies the borrow path matches.
//! Run: cargo run --release -p ft-api --example prelu_ab

use ft_api::FrankenTorchSession;
use ft_core::ExecutionMode;
use std::time::Instant;

fn old_prelu(input: &[f64], w: f64) -> Vec<f64> {
    let cloned = input.to_vec(); // old fallback materialized the whole input via tensor_values
    cloned.iter().map(|&x| if x >= 0.0 { x } else { w * x }).collect()
}

fn bench<F: FnMut() -> usize>(mut f: F) -> f64 {
    let mut best = f64::INFINITY;
    for _ in 0..9 {
        let t = Instant::now();
        let s = f();
        let el = t.elapsed().as_secs_f64() * 1e3;
        std::hint::black_box(s);
        if el < best {
            best = el;
        }
    }
    best
}

fn main() {
    println!("tensor_prelu f64 (scalar w), min-9:  OLD=clone input + serial prelu  NEW=borrow input");
    let cases: [(&str, usize); 3] = [("8M", 8_000_000), ("16M", 16_000_000), ("32M", 32_000_000)];
    for (label, numel) in cases {
        let w = 0.25_f64;
        let input: Vec<f64> = (0..numel).map(|i| ((i % 211) as f64 - 100.0) * 0.01).collect();

        let mut sess = FrankenTorchSession::new(ExecutionMode::Strict);
        let it = sess.tensor_variable(input.clone(), vec![numel], false).unwrap();
        let wt = sess.tensor_variable(vec![w], vec![1], false).unwrap();
        let out = sess.tensor_prelu(it, wt).unwrap();
        let new_out = sess.tensor_values(out).unwrap();
        let old_out = old_prelu(&input, w);
        let bitmatch = new_out == old_out;

        let old_ms = bench(|| old_prelu(&input, w).len());
        let new_ms = bench(|| sess.tensor_prelu(it, wt).unwrap().0);
        println!(
            "  {label:<6} ({:>3}MB)  OLD {:8.3}  NEW {:8.3}  = {:.2}x  bitmatch={}",
            numel * 8 / (1 << 20),
            old_ms,
            new_ms,
            old_ms / new_ms,
            bitmatch
        );
    }
}

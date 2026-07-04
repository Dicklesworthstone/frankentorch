//! A/B for tensor_bitwise_not F64. OLD = to_vec (serial first-touch clone) + serial map = exact ORIG
//! model (no apply_function); NEW = sess.tensor_bitwise_not (borrow + par_map).
//! Run: cargo run --release -p ft-api --example bitwise_not_ab

use ft_api::FrankenTorchSession;
use ft_core::ExecutionMode;
use std::time::Instant;

fn old_bitwise_not(input: &[f64]) -> Vec<f64> {
    let cloned = input.to_vec(); // old path materialized input via tensor_values
    cloned.iter().map(|&a| (!(a as i64)) as f64).collect()
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
    println!("tensor_bitwise_not f64, min-9:  OLD=clone + serial map  NEW=borrow + par_map");
    let cases: [(&str, usize); 3] = [("8M", 8_000_000), ("16M", 16_000_000), ("32M", 32_000_000)];
    for (label, numel) in cases {
        let input: Vec<f64> = (0..numel).map(|i| (i % 100_003) as f64).collect();

        let mut sess = FrankenTorchSession::new(ExecutionMode::Strict);
        let it = sess.tensor_variable(input.clone(), vec![numel], false).unwrap();
        let out = sess.tensor_bitwise_not(it).unwrap();
        let new_out = sess.tensor_values(out).unwrap();
        let old_out = old_bitwise_not(&input);
        let bitmatch = new_out == old_out;

        let old_ms = bench(|| old_bitwise_not(&input).len());
        let new_ms = bench(|| sess.tensor_bitwise_not(it).unwrap().0);
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

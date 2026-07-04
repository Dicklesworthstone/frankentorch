//! A/B for tensor_bitwise_and F64 (representative of the shared bitwise_binary_par_f64 helper: and/or/
//! xor/shifts all route through it). OLD = clone both inputs (serial first-touch) + serial zip_map =
//! exact ORIG model (no apply_function); NEW = real op (borrow both + par_zip_map).
//! Run: cargo run --release -p ft-api --example bitwise_and_ab

use ft_api::FrankenTorchSession;
use ft_core::ExecutionMode;
use std::time::Instant;

fn old_bitwise_and(a: &[f64], b: &[f64]) -> Vec<f64> {
    let ca = a.to_vec();
    let cb = b.to_vec();
    ca.iter().zip(cb.iter()).map(|(&x, &y)| ((x as i64) & (y as i64)) as f64).collect()
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
    println!("tensor_bitwise_and f64, min-9:  OLD=clone both + serial zip  NEW=borrow both + par_zip");
    let cases: [(&str, usize); 3] = [("8M", 8_000_000), ("16M", 16_000_000), ("32M", 32_000_000)];
    for (label, numel) in cases {
        let a: Vec<f64> = (0..numel).map(|i| (i % 100_003) as f64).collect();
        let b: Vec<f64> = (0..numel).map(|i| (i % 97) as f64).collect();

        let mut sess = FrankenTorchSession::new(ExecutionMode::Strict);
        let at = sess.tensor_variable(a.clone(), vec![numel], false).unwrap();
        let bt = sess.tensor_variable(b.clone(), vec![numel], false).unwrap();
        let out = sess.tensor_bitwise_and(at, bt).unwrap();
        let new_out = sess.tensor_values(out).unwrap();
        let old_out = old_bitwise_and(&a, &b);
        let bitmatch = new_out == old_out;

        let old_ms = bench(|| old_bitwise_and(&a, &b).len());
        let new_ms = bench(|| sess.tensor_bitwise_and(at, bt).unwrap().0);
        println!(
            "  {label:<6} ({:>3}MB x2)  OLD {:8.3}  NEW {:8.3}  = {:.2}x  bitmatch={}",
            numel * 8 / (1 << 20),
            old_ms,
            new_ms,
            old_ms / new_ms,
            bitmatch
        );
    }
}

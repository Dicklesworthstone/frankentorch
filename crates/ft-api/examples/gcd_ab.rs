//! A/B for tensor_gcd F64. OLD = clone both (storage to_vec) + serial gcd map = exact ORIG model
//! (no apply_function); NEW = real op (clone both + PARALLEL gcd map). The per-element Euclidean gcd
//! is compute-heavy, so parallelizing the map is the win. Run: cargo run --release -p ft-api --example gcd_ab

use ft_api::FrankenTorchSession;
use ft_core::ExecutionMode;
use std::time::Instant;

fn gcd(mut a: i64, mut b: i64) -> i64 {
    a = a.abs();
    b = b.abs();
    while b != 0 {
        let t = b;
        b = a % b;
        a = t;
    }
    a
}

fn old_gcd(a: &[f64], b: &[f64]) -> Vec<f64> {
    let ca = a.to_vec();
    let cb = b.to_vec();
    ca.iter().zip(cb.iter()).map(|(&x, &y)| gcd(x as i64, y as i64) as f64).collect()
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
    println!("tensor_gcd f64, min-9:  OLD=clone both + serial gcd  NEW=clone both + par gcd");
    let cases: [(&str, usize); 3] = [("4M", 4_000_000), ("8M", 8_000_000), ("16M", 16_000_000)];
    for (label, numel) in cases {
        // Mixed large-ish values so the Euclidean loop runs several iterations.
        let a: Vec<f64> = (0..numel).map(|i| ((i * 2_654_435_761usize) % 1_000_003) as f64).collect();
        let b: Vec<f64> = (0..numel).map(|i| ((i * 40_503usize) % 999_983 + 1) as f64).collect();

        let mut sess = FrankenTorchSession::new(ExecutionMode::Strict);
        let at = sess.tensor_variable(a.clone(), vec![numel], false).unwrap();
        let bt = sess.tensor_variable(b.clone(), vec![numel], false).unwrap();
        let out = sess.tensor_gcd(at, bt).unwrap();
        let new_out = sess.tensor_values(out).unwrap();
        let old_out = old_gcd(&a, &b);
        let bitmatch = new_out == old_out;

        let old_ms = bench(|| old_gcd(&a, &b).len());
        let new_ms = bench(|| sess.tensor_gcd(at, bt).unwrap().0);
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

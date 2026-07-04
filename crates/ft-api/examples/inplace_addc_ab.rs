//! Real-op A/B for in-place addcmul_ (ternary, 3-operand) F64 fast path. OLD = clone-ALL-3 + serial
//! map (HEAD generic); NEW = borrow-all-3 + parallel. Run: cargo run --release -p ft-api --example inplace_addc_ab

use ft_api::FrankenTorchSession;
use ft_core::ExecutionMode;
use std::time::Instant;

fn old_addcmul(t: &[f64], t1: &[f64], t2: &[f64], value: f64) -> Vec<f64> {
    let tv = t.to_vec();
    let av = t1.to_vec();
    let bv = t2.to_vec();
    tv.iter()
        .zip(av.iter())
        .zip(bv.iter())
        .map(|((&t, &a), &b)| t + value * a * b)
        .collect()
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
    println!("tensor_addcmul_ (in-place, ternary) f64, min-9:  OLD=clone-3+serial  NEW=borrow-3+parallel");
    let value = 0.7;
    for &n in &[1usize << 24, 1 << 26] {
        let t: Vec<f64> = (0..n).map(|i| (i % 211) as f64).collect();
        let a: Vec<f64> = (0..n).map(|i| (i % 173) as f64 + 0.5).collect();
        let b: Vec<f64> = (0..n).map(|i| (i % 97) as f64 + 0.25).collect();
        let mut s = FrankenTorchSession::new(ExecutionMode::Strict);
        let tt = s.tensor_variable(t.clone(), vec![n], false).unwrap();
        let at = s.tensor_variable(a.clone(), vec![n], false).unwrap();
        let bt = s.tensor_variable(b.clone(), vec![n], false).unwrap();
        s.tensor_addcmul_(tt, at, bt, value).unwrap();
        let bitmatch = s.tensor_values(tt).unwrap() == old_addcmul(&t, &a, &b, value);

        let old_ms = bench(|| old_addcmul(&t, &a, &b, value).len());
        let new_ms = bench(|| {
            // tt already mutated; re-running addcmul_ on it does the same read+compute+write work.
            s.tensor_addcmul_(tt, at, bt, value).unwrap();
            s.tensor_values(tt).unwrap().len()
        });
        println!(
            "  n={:>10} ({:>4}MB)  OLD {:8.3}  NEW {:8.3}  = {:.2}x  bitmatch={}",
            n,
            n * 8 / (1 << 20),
            old_ms,
            new_ms,
            old_ms / new_ms,
            bitmatch
        );
    }
}

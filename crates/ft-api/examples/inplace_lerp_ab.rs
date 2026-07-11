//! Real-op A/B for in-place lerp_ F64 fast path. OLD = clone-both + serial `s+w*(e-s)` (HEAD generic);
//! NEW = borrow-both + parallel (same naive formula). Run: cargo run --release -p ft-api --example inplace_lerp_ab

use ft_api::FrankenTorchSession;
use ft_core::ExecutionMode;
use std::time::Instant;

fn old_lerp(t: &[f64], e: &[f64], w: f64) -> Vec<f64> {
    let tv = t.to_vec();
    let ev = e.to_vec();
    tv.iter()
        .zip(ev.iter())
        .map(|(&s, &e)| s + w * (e - s))
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
    println!(
        "tensor_lerp_ (in-place) f64, min-9:  OLD=clone-both+serial  NEW=borrow-both+parallel"
    );
    let w = 0.3;
    for &n in &[1usize << 24, 1 << 26] {
        let t: Vec<f64> = (0..n).map(|i| (i % 211) as f64).collect();
        let e: Vec<f64> = (0..n).map(|i| (i % 173) as f64 + 0.5).collect();
        let mut s = FrankenTorchSession::new(ExecutionMode::Strict);
        let tt = s.tensor_variable(t.clone(), vec![n], false).unwrap();
        let et = s.tensor_variable(e.clone(), vec![n], false).unwrap();
        s.tensor_lerp_(tt, et, w).unwrap();
        let bitmatch = s.tensor_values(tt).unwrap() == old_lerp(&t, &e, w);
        let old_ms = bench(|| old_lerp(&t, &e, w).len());
        let new_ms = bench(|| {
            s.tensor_lerp_(tt, et, w).unwrap();
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

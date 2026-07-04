//! Real-op A/B: in-place tensor_maximum_ on f64. OLD = HEAD's generic path (clone BOTH + serial map
//! + update), replicated inline; NEW = `s.tensor_maximum_` (added F64 borrow-both+parallel fast path).
//! Same process, min-9. Run: cargo run --release -p ft-api --example inplace_maxmin_ab

use ft_api::FrankenTorchSession;
use ft_core::ExecutionMode;
use std::time::Instant;

// Replica of HEAD's generic maximum_ compute (clone both + serial map). Returns the new target vals.
fn old_maximum(target: &[f64], other: &[f64]) -> Vec<f64> {
    let tv = target.to_vec(); // lossy_f64 clone
    let ov = other.to_vec(); // lossy_f64 clone
    tv.iter().zip(ov.iter()).map(|(&a, &b)| a.max(b)).collect()
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
    println!("tensor_maximum_ (in-place) f64, min-9:  OLD=clone-both+serial  NEW=borrow-both+parallel");
    for &n in &[1usize << 22, 1 << 24, 1 << 26] {
        let a: Vec<f64> = (0..n).map(|i| (i % 211) as f64 - 100.0).collect();
        let b: Vec<f64> = (0..n).map(|i| (i % 173) as f64 - 80.0).collect();
        // correctness: NEW in-place result == OLD replica
        let mut s = FrankenTorchSession::new(ExecutionMode::Strict);
        let at = s.tensor_variable(a.clone(), vec![n], false).unwrap();
        let bt = s.tensor_variable(b.clone(), vec![n], false).unwrap();
        s.tensor_maximum_(at, bt).unwrap();
        let new_out = s.tensor_values(at).unwrap();
        let old_out = old_maximum(&a, &b);
        let bitmatch = new_out == old_out;

        let old_ms = bench(|| old_maximum(&a, &b).len());
        // NEW: rebuild target each iter (maximum_ mutates it) OUTSIDE... but maximum_(x, y) with
        // x already >= is idempotent-ish; to keep it honest, re-set target to `a` each iter via a
        // fresh tensor is costly. Instead measure repeated maximum_(at, bt): after first call at is
        // max(a,b); subsequent calls are max(max,b)==max — same work (full read+compute+write).
        let new_ms = bench(|| {
            s.tensor_maximum_(at, bt).unwrap();
            s.tensor_values(at).unwrap().len()
        });
        let ratio = old_ms / new_ms;
        println!(
            "  n={:>10} ({:>4}MB)  OLD {:8.3}  NEW {:8.3}  = {:.2}x  bitmatch={}",
            n,
            n * 8 / (1 << 20),
            old_ms,
            new_ms,
            ratio,
            bitmatch
        );
    }
}

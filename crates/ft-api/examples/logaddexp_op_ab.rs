//! Real-op A/B: tensor_logaddexp on f64. OLD = HEAD's generic path (clone BOTH + ~9-op compose:
//! max/sub/exp/add/log), replicated inline; NEW = `s.tensor_logaddexp` (added F64 fused fast path:
//! borrow both + one parallel pass). bitmatch checks the fused == compose (finite). min-9.
//! Run: cargo run --release -p ft-api --example logaddexp_op_ab

use ft_api::FrankenTorchSession;
use ft_core::ExecutionMode;
use std::time::Instant;

// Replica of HEAD's generic f64 compose (max + sub + exp + add + ln), element-wise, for FINITE inputs.
fn old_logaddexp(a: &[f64], b: &[f64]) -> Vec<f64> {
    let av = a.to_vec(); // lossy_f64 clone
    let bv = b.to_vec(); // lossy_f64 clone
    av.iter()
        .zip(bv.iter())
        .map(|(&x, &y)| {
            let m = x.max(y);
            m + ((x - m).exp() + (y - m).exp()).ln()
        })
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
    println!("tensor_logaddexp f64 (finite), min-9:  OLD=clone-both+9op-compose  NEW=borrow-both+fused-parallel");
    for &n in &[1usize << 22, 1 << 24, 1 << 26] {
        let a: Vec<f64> = (0..n).map(|i| ((i % 211) as f64 - 100.0) * 0.1).collect();
        let b: Vec<f64> = (0..n).map(|i| ((i % 173) as f64 - 80.0) * 0.1).collect();
        let mut s = FrankenTorchSession::new(ExecutionMode::Strict);
        let at = s.tensor_variable(a.clone(), vec![n], false).unwrap();
        let bt = s.tensor_variable(b.clone(), vec![n], false).unwrap();
        let node = s.tensor_logaddexp(at, bt).unwrap();
        let new_out = s.tensor_values(node).unwrap();
        let old_out = old_logaddexp(&a, &b);
        let bitmatch = new_out == old_out;

        let old_ms = bench(|| old_logaddexp(&a, &b).len());
        let new_ms = bench(|| {
            let node = s.tensor_logaddexp(at, bt).unwrap();
            s.tensor_values(node).unwrap().len()
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

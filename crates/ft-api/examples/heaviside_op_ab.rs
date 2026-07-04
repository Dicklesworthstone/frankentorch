//! Real-op A/B: tensor_heaviside on f64. OLD = HEAD's generic f64 path (clone BOTH operands + SERIAL
//! step), replicated inline; NEW = `s.tensor_heaviside` (added F64 fast path: borrow both + parallel).
//! Same process, min-9. Run: cargo run --release -p ft-api --example heaviside_op_ab

use ft_api::FrankenTorchSession;
use ft_core::ExecutionMode;
use std::time::Instant;

fn old_heaviside(x: &[f64], v: &[f64]) -> Vec<f64> {
    let xv = x.to_vec(); // lossy_f64 clone
    let vv = v.to_vec(); // lossy_f64 clone (both operands)
    xv.iter()
        .zip(vv.iter())
        .map(|(&x, &v)| {
            if x > 0.0 {
                1.0
            } else if x == 0.0 {
                v
            } else {
                0.0
            }
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
    println!("tensor_heaviside f64, min-9:  OLD=clone-both+serial  NEW=borrow-both+parallel");
    for &n in &[1usize << 22, 1 << 24, 1 << 26] {
        let x: Vec<f64> = (0..n).map(|i| (i % 7) as f64 - 3.0).collect();
        let v: Vec<f64> = (0..n).map(|i| (i % 5) as f64).collect();
        let mut s = FrankenTorchSession::new(ExecutionMode::Strict);
        let xt = s.tensor_variable(x.clone(), vec![n], false).unwrap();
        let vt = s.tensor_variable(v.clone(), vec![n], false).unwrap();
        let node = s.tensor_heaviside(xt, vt).unwrap();
        let new_out = s.tensor_values(node).unwrap();
        let old_out = old_heaviside(&x, &v);
        let bitmatch = new_out == old_out;

        let old_ms = bench(|| old_heaviside(&x, &v).len());
        let new_ms = bench(|| {
            let node = s.tensor_heaviside(xt, vt).unwrap();
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

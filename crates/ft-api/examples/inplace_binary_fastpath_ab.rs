//! A/B for the in-place-binary F64 fast path added to atan2_/logaddexp_/logaddexp2_/gcd_/lcm_
//! (they previously ran the SERIAL tensor_values_lossy_f64 clone-both + serial map + writeback for
//! ALL dtypes; now f64+contiguous routes through try_inplace_binary_f64 = borrow both + parallel map).
//! Representative = atan2_ (transcendental -> compute-dominated). OLD models the old serial path:
//! clone(target)+clone(other) -> SERIAL atan2 map -> writeback. NEW = real op (borrow both + par map).
//! atan2 range (-pi,pi] so repeated in-place atan2 stays bounded (stable timing).
//! Run PLAIN (no pipe): cargo run --release -p ft-api --example inplace_binary_fastpath_ab

use ft_api::FrankenTorchSession;
use ft_core::ExecutionMode;
use std::time::Instant;

fn old_inplace_atan2(target: &[f64], other: &[f64]) -> Vec<f64> {
    let mut buf = target.to_vec(); // lossy_f64(target) clone
    let other_c = other.to_vec(); // lossy_f64(other) clone
    let mapped: Vec<f64> = buf
        .iter()
        .zip(other_c.iter())
        .map(|(&y, &x)| y.atan2(x))
        .collect(); // serial
    buf.copy_from_slice(&mapped); // update_for_float writeback
    buf
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
        "in-place binary fastpath (atan2_) f64, min-9:  OLD=clone both+serial atan2+writeback  NEW=borrow both+par"
    );
    let cases: [(&str, usize); 3] = [("4M", 4_000_000), ("8M", 8_000_000), ("16M", 16_000_000)];
    for (label, numel) in cases {
        let target: Vec<f64> = (0..numel)
            .map(|i| ((i % 2000) as f64 - 1000.0) * 0.01)
            .collect();
        let other: Vec<f64> = (0..numel)
            .map(|i| ((i % 1500) as f64 - 700.0) * 0.01 + 0.3)
            .collect();

        let mut sess = FrankenTorchSession::new(ExecutionMode::Strict);
        let tt = sess
            .tensor_variable(target.clone(), vec![numel], false)
            .unwrap();
        let ot = sess
            .tensor_variable(other.clone(), vec![numel], false)
            .unwrap();
        // bitmatch: one application of the real op vs the old replica on the same inputs.
        sess.tensor_atan2_(tt, ot).unwrap();
        let new_once = sess.tensor_values(tt).unwrap();
        let old_once = old_inplace_atan2(&target, &other);
        let bitmatch = new_once == old_once;

        let new_ms = bench(|| {
            sess.tensor_atan2_(tt, ot).unwrap();
            numel
        });
        let old_ms = bench(|| old_inplace_atan2(&target, &other).len());
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

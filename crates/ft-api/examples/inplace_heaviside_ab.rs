//! A/B for the in-place-binary F64 fast path added to heaviside_ (was: serial clone-both
//! (lossy_f64(target)+tensor_values(values)) + serial select + writeback for all dtypes; now
//! f64+contiguous borrows both + parallel select via try_inplace_binary_f64). OLD models the old serial
//! path; NEW = real op. Run PLAIN (no pipe): cargo run --release -p ft-api --example inplace_heaviside_ab

use ft_api::FrankenTorchSession;
use ft_core::ExecutionMode;
use std::time::Instant;

fn heaviside(x: f64, v: f64) -> f64 {
    if x > 0.0 {
        1.0
    } else if x == 0.0 {
        v
    } else {
        0.0
    }
}

fn old_inplace_heaviside(target: &[f64], values: &[f64]) -> Vec<f64> {
    let mut buf = target.to_vec(); // lossy_f64(target) clone
    let vals_c = values.to_vec(); // tensor_values(values) clone
    let mapped: Vec<f64> = buf.iter().zip(vals_c.iter()).map(|(&x, &v)| heaviside(x, v)).collect();
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
    println!("in-place heaviside_ f64, min-9:  OLD=clone both+serial select+writeback  NEW=borrow both+par");
    let cases: [(&str, usize); 3] = [("4M", 4_000_000), ("8M", 8_000_000), ("16M", 16_000_000)];
    for (label, numel) in cases {
        // Mix of negative / zero / positive so all three heaviside branches are exercised.
        let target: Vec<f64> = (0..numel).map(|i| ((i % 3) as f64 - 1.0)).collect();
        let values: Vec<f64> = (0..numel).map(|i| (i % 7) as f64 * 0.5).collect();

        let mut sess = FrankenTorchSession::new(ExecutionMode::Strict);
        let tt = sess.tensor_variable(target.clone(), vec![numel], false).unwrap();
        let vt = sess.tensor_variable(values.clone(), vec![numel], false).unwrap();
        // bitmatch: one application of the real op vs the old replica on the same inputs.
        sess.tensor_heaviside_(tt, vt).unwrap();
        let new_once = sess.tensor_values(tt).unwrap();
        let old_once = old_inplace_heaviside(&target, &values);
        let bitmatch = new_once == old_once;

        let new_ms = bench(|| {
            sess.tensor_heaviside_(tt, vt).unwrap();
            numel
        });
        let old_ms = bench(|| old_inplace_heaviside(&target, &values).len());
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

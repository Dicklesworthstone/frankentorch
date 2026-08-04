//! A/B for apply_tensor_binary_in_place (in-place add_/sub_/mul_/div_ share it). Representative = mul_
//! (other ~= 1.0 so repeated in-place multiply keeps target bounded -> timing stable). OLD models the
//! old body: clone(target) + clone(other) -> SERIAL zip-map(a*b) -> writeback into the clone buffer
//! (4 passes, 2 allocs). NEW = real op (clone other once + in-place par_iter_mut on target = 2 passes,
//! parallel). Run PLAIN (no pipe): cargo run --release -p ft-api --example inplace_binary_ab

use ft_api::FrankenTorchSession;
use ft_core::ExecutionMode;
use std::time::Instant;

fn old_inplace_mul(target: &[f64], other: &[f64]) -> Vec<f64> {
    let mut buf = target.to_vec(); // values(target) clone
    let other_c = other.to_vec(); // values(other) clone
    let mapped: Vec<f64> = buf
        .iter()
        .zip(other_c.iter())
        .map(|(&a, &b)| a * b)
        .collect(); // serial map
    buf.copy_from_slice(&mapped); // writeback into existing storage
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
    let mut report = String::from(
        "in-place binary (mul_) f64, min-9:  OLD=clone both+serial zip+writeback  NEW=clone other+in-place par\n",
    );
    print!("{report}");
    let cases: [(&str, usize); 3] = [("4M", 4_000_000), ("8M", 8_000_000), ("16M", 16_000_000)];
    for (label, numel) in cases {
        let target: Vec<f64> = (0..numel)
            .map(|i| ((i % 1000) as f64 - 500.0) * 0.01)
            .collect();
        // other in [0.99, 1.01] so repeated in-place multiply stays bounded (stable timing).
        let other: Vec<f64> = (0..numel).map(|i| 0.99 + (i % 21) as f64 * 0.001).collect();

        let mut sess = FrankenTorchSession::new(ExecutionMode::Strict);
        let tt = sess
            .tensor_variable(target.clone(), vec![numel], false)
            .unwrap();
        let ot = sess
            .tensor_variable(other.clone(), vec![numel], false)
            .unwrap();
        // bitmatch: one application of the real op vs the old replica on the same inputs.
        sess.tensor_mul_(tt, ot).unwrap();
        let new_once = sess.tensor_values(tt).unwrap();
        let old_once = old_inplace_mul(&target, &other);
        let bitmatch = new_once == old_once;

        let new_ms = bench(|| {
            sess.tensor_mul_(tt, ot).unwrap();
            numel
        });
        let old_ms = bench(|| old_inplace_mul(&target, &other).len());
        let line = format!(
            "  {label:<6} ({:>3}MB x2)  OLD {:8.3}  NEW {:8.3}  = {:.2}x  bitmatch={}\n",
            numel * 8 / (1 << 20),
            old_ms,
            new_ms,
            old_ms / new_ms,
            bitmatch
        );
        print!("{line}");
        report.push_str(&line);
    }
    let _ = report;
}

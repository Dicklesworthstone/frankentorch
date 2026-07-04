//! A/B for apply_tensor_unary_in_place (73 in-place unary ops share it: exp_/log_/erf_/lgamma_/
//! sigmoid_/gelu_/silu_/i0_/... all route through it). Representative op = sigmoid_ (compute-heavy,
//! values stay bounded in (0,1) so repeated timing is stable). OLD models the helper's OLD path:
//! clone (values()) -> SERIAL map(sigmoid) -> writeback-copy (3 serial passes). NEW = real op
//! (in-place update_tensor_values_with + par_iter_mut, ONE pass). The OLD replica reuses the clone
//! buffer for the writeback (2 allocs) to match the real old cost, so this is not inflated.
//! Run: cargo run --release -p ft-api --example inplace_unary_ab

use ft_api::FrankenTorchSession;
use ft_core::ExecutionMode;
use std::time::Instant;

fn sigmoid(x: f64) -> f64 {
    1.0 / (1.0 + (-x).exp())
}

fn old_inplace_sigmoid(input: &[f64]) -> Vec<f64> {
    let mut buf = input.to_vec(); // values() clone (alloc + read-copy)
    let mapped: Vec<f64> = buf.iter().map(|&x| sigmoid(x)).collect(); // serial map (alloc + compute)
    buf.copy_from_slice(&mapped); // writeback into existing storage (copy, no alloc)
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
        "in-place unary (sigmoid_) f64, min-9:  OLD=clone+serial map+writeback  NEW=in-place par_iter_mut\n",
    );
    print!("{report}");
    let cases: [(&str, usize); 3] = [("4M", 4_000_000), ("8M", 8_000_000), ("16M", 16_000_000)];
    for (label, numel) in cases {
        let input: Vec<f64> = (0..numel).map(|i| ((i % 2000) as f64 - 1000.0) * 0.01).collect();

        let mut sess = FrankenTorchSession::new(ExecutionMode::Strict);
        let it = sess.tensor_variable(input.clone(), vec![numel], false).unwrap();
        // bitmatch: one application of the real op vs the old replica on the same input.
        sess.tensor_sigmoid_(it).unwrap();
        let new_once = sess.tensor_values(it).unwrap();
        let old_once = old_inplace_sigmoid(&input);
        let bitmatch = new_once == old_once;

        // NEW timing: repeated in-place sigmoid_ (values stay in (0,1) -> numel exp() each call).
        let new_ms = bench(|| {
            sess.tensor_sigmoid_(it).unwrap();
            numel
        });
        // OLD timing: the 3-pass serial replica on the original input.
        let old_ms = bench(|| old_inplace_sigmoid(&input).len());
        let line = format!(
            "  {label:<6} ({:>3}MB)  OLD {:8.3}  NEW {:8.3}  = {:.2}x  bitmatch={}\n",
            numel * 8 / (1 << 20),
            old_ms,
            new_ms,
            old_ms / new_ms,
            bitmatch
        );
        print!("{line}");
        report.push_str(&line);
    }
    // rch drops child stdout intermittently; also write the report under CARGO_TARGET_DIR,
    // which rch reliably syncs back to the local machine.
    if let Ok(dir) = std::env::var("CARGO_TARGET_DIR") {
        let _ = std::fs::write(format!("{dir}/inplace_ab_result.txt"), &report);
    }
}

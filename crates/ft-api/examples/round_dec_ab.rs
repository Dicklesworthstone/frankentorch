//! A/B for tensor_round_decimals F64. NEW = real op (no-grad fused 1-pass round_ties_even(x*f)/f);
//! OLD = a 3-pass PARALLEL compose replica (mul -> round -> div, 2 intermediates), CONSERVATIVE (it
//! omits the full(factor) alloc + tape-node overhead the real compose also pays, so the real win is
//! >= this ratio). Run: cargo run --release -p ft-api --example round_dec_ab

use ft_api::FrankenTorchSession;
use ft_core::ExecutionMode;
use rayon::prelude::*;
use std::time::Instant;

fn old_round_decimals(input: &[f64], factor: f64) -> Vec<f64> {
    // Models the mul -> round -> div compose (each a parallel kernel pass), 2 full intermediates.
    let scaled: Vec<f64> = input.par_iter().map(|&x| x * factor).collect();
    let rounded: Vec<f64> = scaled.par_iter().map(|&x| x.round_ties_even()).collect();
    rounded.par_iter().map(|&x| x / factor).collect()
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
        "tensor_round_decimals f64 (dec=3), min-9:  OLD=3-pass parallel compose  NEW=fused 1-pass"
    );
    let cases: [(&str, usize); 3] = [("8M", 8_000_000), ("16M", 16_000_000), ("32M", 32_000_000)];
    for (label, numel) in cases {
        let decimals = 3;
        let factor = 10.0_f64.powi(decimals);
        let input: Vec<f64> = (0..numel).map(|i| (i % 100_003) as f64 * 0.0007).collect();

        let mut sess = FrankenTorchSession::new(ExecutionMode::Strict);
        let it = sess
            .tensor_variable(input.clone(), vec![numel], false)
            .unwrap();
        let out = sess.tensor_round_decimals(it, decimals).unwrap();
        let new_out = sess.tensor_values(out).unwrap();
        let old_out = old_round_decimals(&input, factor);
        let bitmatch = new_out == old_out;

        let old_ms = bench(|| old_round_decimals(&input, factor).len());
        let new_ms = bench(|| sess.tensor_round_decimals(it, decimals).unwrap().0);
        println!(
            "  {label:<6} ({:>3}MB)  OLD {:8.3}  NEW {:8.3}  = {:.2}x  bitmatch={}",
            numel * 8 / (1 << 20),
            old_ms,
            new_ms,
            old_ms / new_ms,
            bitmatch
        );
    }
}

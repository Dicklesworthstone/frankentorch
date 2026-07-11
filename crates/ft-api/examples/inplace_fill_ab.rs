//! A/B for fill_inplace_scalar (in-place zero_/ones_/fill_ share it). Representative = ones_ (non-zero
//! fill, so no calloc shortcut -> the clearest case). OLD models the old body: vec![1.0; numel] (serial
//! memset that page-faults the fresh Vec) + copy_from_slice into the existing (warm) storage = the
//! update_tensor_values_for_float writeback. NEW = real op (in-place par_iter_mut fill, one pass, no
//! intermediate Vec). Run PLAIN (no pipe): cargo run --release -p ft-api --example inplace_fill_ab

use ft_api::FrankenTorchSession;
use ft_core::ExecutionMode;
use std::time::Instant;

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
    println!("in-place fill (ones_) f64, min-9:  OLD=vec![1.0;n]+writeback  NEW=in-place par fill");
    let cases: [(&str, usize); 3] = [("4M", 4_000_000), ("8M", 8_000_000), ("16M", 16_000_000)];
    for (label, numel) in cases {
        let init: Vec<f64> = (0..numel).map(|i| (i % 97) as f64 * 0.01).collect();

        let mut sess = FrankenTorchSession::new(ExecutionMode::Strict);
        let tt = sess
            .tensor_variable(init.clone(), vec![numel], false)
            .unwrap();
        // bitmatch: one application of the real op -> all 1.0.
        sess.tensor_ones_(tt).unwrap();
        let new_vals = sess.tensor_values(tt).unwrap();
        let bitmatch = new_vals.len() == numel && new_vals.iter().all(|&x| x == 1.0);

        // OLD scratch storage: allocated + faulted ONCE (like the tensor's existing storage), so the
        // timed cost is the fresh vec![1.0;n] memset + the copy_from_slice writeback (the real old path).
        let mut scratch = vec![0.0f64; numel];
        let old_ms = bench(|| {
            let v = vec![1.0f64; numel];
            scratch.copy_from_slice(&v);
            scratch.len()
        });
        let new_ms = bench(|| {
            sess.tensor_ones_(tt).unwrap();
            numel
        });
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

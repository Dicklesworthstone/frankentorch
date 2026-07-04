//! A/B for tensor_tril F64. OLD = apply_function-path replica (CLONE input via to_vec, then the
//! same parallel per-row mask fill); NEW = sess.tensor_tril (F64 borrows the input). bitmatch verifies
//! the borrow path matches. Run: cargo run --release -p ft-api --example tril_ab

use ft_api::FrankenTorchSession;
use ft_core::ExecutionMode;
use rayon::prelude::*;
use std::time::Instant;

fn old_tril(input: &[f64], m: usize, n: usize, diagonal: i64) -> Vec<f64> {
    let cloned = input.to_vec(); // apply_function materializes inputs
    let mut result = vec![0.0f64; m * n];
    result.par_chunks_mut(n).enumerate().for_each(|(i, row)| {
        let lim = (i as i64) + diagonal;
        let src = &cloned[i * n..i * n + n];
        for (j, slot) in row.iter_mut().enumerate() {
            if (j as i64) <= lim {
                *slot = src[j];
            }
        }
    });
    result
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
    println!("tensor_tril f64, min-9:  OLD=clone input + parallel fill  NEW=borrow input + parallel fill");
    let cases: [(&str, usize, usize); 3] = [("4000x4000", 4000, 4000), ("8000x2000", 8000, 2000), ("2000x8000", 2000, 8000)];
    for (label, m, n) in cases {
        let diagonal = 0_i64;
        let input: Vec<f64> = (0..m * n).map(|i| (i % 251) as f64 * 0.5 + 1.0).collect();

        let mut sess = FrankenTorchSession::new(ExecutionMode::Strict);
        let it = sess.tensor_variable(input.clone(), vec![m, n], false).unwrap();
        let out = sess.tensor_tril(it, diagonal).unwrap();
        let new_out = sess.tensor_values(out).unwrap();
        let old_out = old_tril(&input, m, n, diagonal);
        let bitmatch = new_out == old_out;

        let old_ms = bench(|| old_tril(&input, m, n, diagonal).len());
        let new_ms = bench(|| sess.tensor_tril(it, diagonal).unwrap().0);
        println!(
            "  {label:<12} ({:>3}MB)  OLD {:8.3}  NEW {:8.3}  = {:.2}x  bitmatch={}",
            m * n * 8 / (1 << 20),
            old_ms,
            new_ms,
            old_ms / new_ms,
            bitmatch
        );
    }
}

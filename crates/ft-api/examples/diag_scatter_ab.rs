//! A/B for tensor_diagonal_scatter F64 (offset 0). OLD = to_vec (serial memcpy) + serial diagonal
//! overwrite = exact ORIG model (no apply_function); NEW = sess.tensor_diagonal_scatter (borrow +
//! PARALLEL copy). Run: cargo run --release -p ft-api --example diag_scatter_ab

use ft_api::FrankenTorchSession;
use ft_core::ExecutionMode;
use std::time::Instant;

fn old_diag_scatter(input: &[f64], src: &[f64], n: usize) -> Vec<f64> {
    let mut result = input.to_vec();
    for (i, &v) in src.iter().enumerate() {
        result[i * n + i] = v;
    }
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
    println!(
        "tensor_diagonal_scatter f64 offset0, min-9:  OLD=to_vec + serial  NEW=borrow + parallel copy"
    );
    let cases: [(&str, usize); 3] = [
        ("4000x4000", 4000),
        ("5000x5000", 5000),
        ("6000x6000", 6000),
    ];
    for (label, n) in cases {
        let numel = n * n;
        let input: Vec<f64> = (0..numel).map(|i| (i % 251) as f64 * 0.5).collect();
        let src: Vec<f64> = (0..n).map(|i| (i % 97) as f64 + 1000.0).collect();

        let mut sess = FrankenTorchSession::new(ExecutionMode::Strict);
        let it = sess
            .tensor_variable(input.clone(), vec![n, n], false)
            .unwrap();
        let st = sess.tensor_variable(src.clone(), vec![n], false).unwrap();
        let out = sess.tensor_diagonal_scatter(it, st, 0).unwrap();
        let new_out = sess.tensor_values(out).unwrap();
        let old_out = old_diag_scatter(&input, &src, n);
        let bitmatch = new_out == old_out;

        let old_ms = bench(|| old_diag_scatter(&input, &src, n).len());
        let new_ms = bench(|| sess.tensor_diagonal_scatter(it, st, 0).unwrap().0);
        println!(
            "  {label:<12} ({:>3}MB)  OLD {:8.3}  NEW {:8.3}  = {:.2}x  bitmatch={}",
            numel * 8 / (1 << 20),
            old_ms,
            new_ms,
            old_ms / new_ms,
            bitmatch
        );
    }
}

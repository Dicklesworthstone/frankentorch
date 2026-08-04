//! A/B for tensor_select_scatter F64 (dim0). OLD = to_vec (serial memcpy) + serial overwrite = exact
//! ORIG model (no apply_function); NEW = sess.tensor_select_scatter (borrow + PARALLEL copy).
//! Run: cargo run --release -p ft-api --example select_scatter_ab

use ft_api::FrankenTorchSession;
use ft_core::ExecutionMode;
use std::time::Instant;

fn old_select_scatter(input: &[f64], src: &[f64], row_len: usize, index: usize) -> Vec<f64> {
    let mut result = input.to_vec();
    let dst = index * row_len;
    result[dst..dst + row_len].copy_from_slice(src);
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
        "tensor_select_scatter f64 dim0, min-9:  OLD=to_vec + serial  NEW=borrow + parallel copy"
    );
    let cases: [(&str, usize, usize); 3] = [
        ("8000x2000", 8000, 2000),
        ("4000x4000", 4000, 4000),
        ("16000x1000", 16000, 1000),
    ];
    for (label, rows, cols) in cases {
        let numel = rows * cols;
        let input: Vec<f64> = (0..numel).map(|i| (i % 251) as f64 * 0.5).collect();
        let index = rows / 3;
        let src: Vec<f64> = (0..cols).map(|i| (i % 97) as f64 + 1000.0).collect();

        let mut sess = FrankenTorchSession::new(ExecutionMode::Strict);
        let it = sess
            .tensor_variable(input.clone(), vec![rows, cols], false)
            .unwrap();
        let st = sess
            .tensor_variable(src.clone(), vec![cols], false)
            .unwrap();
        let out = sess.tensor_select_scatter(it, st, 0, index as i64).unwrap();
        let new_out = sess.tensor_values(out).unwrap();
        let old_out = old_select_scatter(&input, &src, cols, index);
        let bitmatch = new_out == old_out;

        let old_ms = bench(|| old_select_scatter(&input, &src, cols, index).len());
        let new_ms = bench(|| {
            sess.tensor_select_scatter(it, st, 0, index as i64)
                .unwrap()
                .0
        });
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

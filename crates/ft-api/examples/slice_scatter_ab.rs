//! A/B for tensor_slice_scatter F64. OLD = exact replica of the pre-fix path (CLONE input via to_vec
//! [serial memcpy] then the serial scatter); NEW = sess.tensor_slice_scatter (borrow + PARALLEL copy
//! + scatter). NOT an apply_function op, so the clone+serial replica models the real ORIG.
//! Run: cargo run --release -p ft-api --example slice_scatter_ab

use ft_api::FrankenTorchSession;
use ft_core::ExecutionMode;
use std::time::Instant;

// dim=0 replica: result = input.clone(); overwrite rows [start,end) with src.
fn old_slice_scatter(
    input: &[f64],
    src: &[f64],
    row_len: usize,
    start: usize,
    slice_len: usize,
) -> Vec<f64> {
    let mut result = input.to_vec();
    for s in 0..slice_len {
        let dst = (start + s) * row_len;
        let sr = s * row_len;
        result[dst..dst + row_len].copy_from_slice(&src[sr..sr + row_len]);
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
        "tensor_slice_scatter f64 dim0, min-9:  OLD=to_vec + serial  NEW=borrow + parallel copy"
    );
    // (label, rows, cols, slice rows replaced)
    let cases: [(&str, usize, usize, usize); 3] = [
        ("8000x2000 slice256", 8000, 2000, 256),
        ("4000x4000 slice128", 4000, 4000, 128),
        ("16000x1000 slice512", 16000, 1000, 512),
    ];
    for (label, rows, cols, slice_len) in cases {
        let numel = rows * cols;
        let input: Vec<f64> = (0..numel).map(|i| (i % 251) as f64 * 0.5).collect();
        let start = rows / 4;
        let src: Vec<f64> = (0..slice_len * cols)
            .map(|i| (i % 97) as f64 + 1000.0)
            .collect();

        let mut sess = FrankenTorchSession::new(ExecutionMode::Strict);
        let it = sess
            .tensor_variable(input.clone(), vec![rows, cols], false)
            .unwrap();
        let st = sess
            .tensor_variable(src.clone(), vec![slice_len, cols], false)
            .unwrap();
        let out = sess
            .tensor_slice_scatter(
                it,
                st,
                0,
                Some(start as i64),
                Some((start + slice_len) as i64),
                1,
            )
            .unwrap();
        let new_out = sess.tensor_values(out).unwrap();
        let old_out = old_slice_scatter(&input, &src, cols, start, slice_len);
        let bitmatch = new_out == old_out;

        let old_ms = bench(|| old_slice_scatter(&input, &src, cols, start, slice_len).len());
        let new_ms = bench(|| {
            sess.tensor_slice_scatter(
                it,
                st,
                0,
                Some(start as i64),
                Some((start + slice_len) as i64),
                1,
            )
            .unwrap()
            .0
        });
        println!(
            "  {label:<20} ({:>3}MB)  OLD {:8.3}  NEW {:8.3}  = {:.2}x  bitmatch={}",
            numel * 8 / (1 << 20),
            old_ms,
            new_ms,
            old_ms / new_ms,
            bitmatch
        );
    }
}

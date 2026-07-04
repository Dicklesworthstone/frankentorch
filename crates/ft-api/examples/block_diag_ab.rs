//! A/B for tensor_block_diag F64. OLD = apply_function-path replica (CLONE each block via to_vec,
//! then calloc output + copy rows — matches the orig's `inputs` materialization); NEW =
//! sess.tensor_block_diag (F64 native fast path, borrow blocks zero-copy). bitmatch verifies the
//! fast path matches the composed positional copies. Run: cargo run --release -p ft-api --example block_diag_ab

use ft_api::FrankenTorchSession;
use ft_core::ExecutionMode;
use std::time::Instant;

// Replica of the apply_function F64 path: clone each block (as the orig materializes `inputs`),
// then copy positional rows into a zeroed output.
fn old_block_diag(blocks: &[(Vec<f64>, usize, usize)]) -> Vec<f64> {
    let total_rows: usize = blocks.iter().map(|b| b.1).sum();
    let total_cols: usize = blocks.iter().map(|b| b.2).sum();
    let mut data = vec![0.0_f64; total_rows * total_cols];
    let mut row_off = 0;
    let mut col_off = 0;
    for (vals, rows, cols) in blocks {
        let cloned = vals.to_vec(); // orig clones each block via tensor_values
        for r in 0..*rows {
            let dst = (row_off + r) * total_cols + col_off;
            let src = r * cols;
            data[dst..dst + cols].copy_from_slice(&cloned[src..src + cols]);
        }
        row_off += rows;
        col_off += cols;
    }
    data
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
    println!("tensor_block_diag f64, min-9:  OLD=apply_fn replica (clone blocks)  NEW=native fast path (borrow)");
    let cases: [(&str, usize, usize); 3] = [("3x 2048^2", 3, 2048), ("4x 1500^2", 4, 1500), ("2x 3000^2", 2, 3000)];
    for (label, nblocks, side) in cases {
        let blocks: Vec<(Vec<f64>, usize, usize)> = (0..nblocks)
            .map(|b| {
                let v: Vec<f64> = (0..side * side).map(|i| ((i + b) % 251) as f64 * 0.25).collect();
                (v, side, side)
            })
            .collect();
        let total = nblocks * side;
        let out_mb = total * total * 8 / (1 << 20);

        let mut sess = FrankenTorchSession::new(ExecutionMode::Strict);
        let ids: Vec<_> = blocks
            .iter()
            .map(|(v, r, c)| sess.tensor_variable(v.clone(), vec![*r, *c], false).unwrap())
            .collect();
        let out = sess.tensor_block_diag(&ids).unwrap();
        let new_out = sess.tensor_values(out).unwrap();
        let old_out = old_block_diag(&blocks);
        let bitmatch = new_out == old_out;

        let old_ms = bench(|| old_block_diag(&blocks).len());
        let new_ms = bench(|| sess.tensor_block_diag(&ids).unwrap().0);
        println!(
            "  {label:<12} (out {:>4}MB)  OLD {:8.3}  NEW {:8.3}  = {:.2}x  bitmatch={}",
            out_mb, old_ms, new_ms, old_ms / new_ms, bitmatch
        );
    }
}

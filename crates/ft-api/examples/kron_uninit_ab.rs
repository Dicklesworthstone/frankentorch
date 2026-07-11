//! Same-process, same-worker A/B for the kron materialize lever:
//! OLD = `par_zeroed` (parallel-collect 0.0 into every element = a dead FIRST write) + parallel
//! row fill; NEW = `ft_kernel_cpu::build_uninit` + the same parallel row fill (fill is the sole
//! writer → N writes not 2N; kron's fill is SEQUENTIAL per row so no strided-fault penalty).
//! Bit-for-bit equality checked. Run: cargo run --release -p ft-api --example kron_uninit_ab

use rayon::prelude::*;
use std::time::Instant;

const PARALLEL_FIRST_TOUCH_MIN: usize = 1 << 18;

fn par_zeroed_f32(n: usize) -> Vec<f32> {
    if n >= PARALLEL_FIRST_TOUCH_MIN {
        (0..n).into_par_iter().map(|_| 0.0f32).collect()
    } else {
        vec![0.0f32; n]
    }
}

// kron 2-D fill (mirrors tensor_kron f32): out row i = a_row·b_rows + b_row; each a-col writes
// the block a[a_row,a_col]·B[b_row,:] at columns [a_col·b_cols ..]. Sequential per row.
fn fill(
    result: &mut [f32],
    a_vals: &[f32],
    b_vals: &[f32],
    a_cols: usize,
    b_rows: usize,
    b_cols: usize,
    out_cols: usize,
) {
    let fill_row = |i: usize, out_row: &mut [f32]| {
        let a_row = i / b_rows;
        let b_row = i % b_rows;
        let brow = &b_vals[b_row * b_cols..b_row * b_cols + b_cols];
        for a_col in 0..a_cols {
            let a_val = a_vals[a_row * a_cols + a_col];
            let dst = &mut out_row[a_col * b_cols..a_col * b_cols + b_cols];
            for (d, &bv) in dst.iter_mut().zip(brow) {
                *d = a_val * bv;
            }
        }
    };
    result
        .par_chunks_mut(out_cols)
        .enumerate()
        .for_each(|(i, row)| fill_row(i, row));
}

fn bench<F: Fn() -> usize>(f: F) -> f64 {
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
    // (label, a_rows, a_cols, b_rows, b_cols)
    let cases: Vec<(&str, usize, usize, usize, usize)> = vec![
        ("[64,64]⊗[64,64]", 64, 64, 64, 64), // out 4096x4096 = 16.8M, 64MB
        ("[512,512]⊗[8,8]", 512, 512, 8, 8), // out 4096x4096
        ("[8,8]⊗[512,512]", 8, 8, 512, 512), // out 4096x4096
        ("[128,128]⊗[32,32]", 128, 128, 32, 32), // out 4096x4096
    ];
    println!("case                    OLD(ms)  NEW(ms)  NEW/OLD   bitmatch   [f32, min-9]");
    for (label, a_rows, a_cols, b_rows, b_cols) in cases {
        let out_rows = a_rows * b_rows;
        let out_cols = a_cols * b_cols;
        let total = out_rows * out_cols;
        let a_vals: Vec<f32> = (0..a_rows * a_cols)
            .map(|i| (i % 97) as f32 + 0.5)
            .collect();
        let b_vals: Vec<f32> = (0..b_rows * b_cols)
            .map(|i| (i % 89) as f32 + 0.25)
            .collect();
        let mut a = par_zeroed_f32(total);
        fill(&mut a, &a_vals, &b_vals, a_cols, b_rows, b_cols, out_cols);
        let b = ft_kernel_cpu::build_uninit(total, |r: &mut [f32]| {
            fill(r, &a_vals, &b_vals, a_cols, b_rows, b_cols, out_cols)
        });
        let bitmatch = a == b;
        let old_ms = bench(|| {
            let mut r = par_zeroed_f32(total);
            fill(&mut r, &a_vals, &b_vals, a_cols, b_rows, b_cols, out_cols);
            r.len()
        });
        let new_ms = bench(|| {
            ft_kernel_cpu::build_uninit(total, |r: &mut [f32]| {
                fill(r, &a_vals, &b_vals, a_cols, b_rows, b_cols, out_cols)
            })
            .len()
        });
        let ratio = old_ms / new_ms;
        println!("  {label:<21} {old_ms:7.3} {new_ms:7.3}   {ratio:5.2}x   bitmatch={bitmatch}");
    }
}

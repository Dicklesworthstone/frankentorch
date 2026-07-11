//! Same-process, same-worker A/B for the generic (elem!=1) permute materialize lever:
//! OLD = `(0..numel).into_par_iter().map(|_| src[0]).collect()` (par-collect first-touch = a
//! full SECOND write of the output) + TILE transpose; NEW = `ft_kernel_cpu::build_uninit` +
//! TILE transpose (transpose is the sole writer, N writes not 2N). Bit-for-bit equality checked.
//! Run: cargo run --release -p ft-api --example permute_uninit_ab

use rayon::prelude::*;
use std::time::Instant;

const TILE: usize = 16;

// One plane's cache-blocked [a_dim, b_dim] transpose of elem-runs (mirrors permute_slice).
fn transpose_plane(sgn: &[f32], dgn: &mut [f32], a_dim: usize, b_dim: usize, elem: usize) {
    let mut ii = 0;
    while ii < a_dim {
        let i_end = (ii + TILE).min(a_dim);
        let mut jj = 0;
        while jj < b_dim {
            let j_end = (jj + TILE).min(b_dim);
            for i in ii..i_end {
                for j in jj..j_end {
                    let s_off = (i * b_dim + j) * elem;
                    let d_off = (j * a_dim + i) * elem;
                    dgn[d_off..d_off + elem].copy_from_slice(&sgn[s_off..s_off + elem]);
                }
            }
            jj += TILE;
        }
        ii += TILE;
    }
}

fn old_permute(src: &[f32], batch: usize, a_dim: usize, b_dim: usize, elem: usize) -> Vec<f32> {
    let plane = a_dim * b_dim * elem;
    let numel = batch * plane;
    // par-collect first-touch (the CURRENT path): full SECOND write of the output.
    let mut dst: Vec<f32> = (0..numel).into_par_iter().map(|_| src[0]).collect();
    dst.par_chunks_mut(plane)
        .zip(src[..numel].par_chunks(plane))
        .for_each(|(dgn, sgn)| transpose_plane(sgn, dgn, a_dim, b_dim, elem));
    dst
}

fn new_permute(src: &[f32], batch: usize, a_dim: usize, b_dim: usize, elem: usize) -> Vec<f32> {
    let plane = a_dim * b_dim * elem;
    let numel = batch * plane;
    ft_kernel_cpu::build_uninit(numel, |dst| {
        dst.par_chunks_mut(plane)
            .zip(src[..numel].par_chunks(plane))
            .for_each(|(dgn, sgn)| transpose_plane(sgn, dgn, a_dim, b_dim, elem));
    })
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
    // (label, batch, a_dim, b_dim, elem) — elem!=1 (trailing kept dims), the permute_slice case.
    let cases: Vec<(&str, usize, usize, usize, usize)> = vec![
        ("[8,512,512,16] swapWH", 8, 512, 512, 16), // plane=4.2M, 8 planes, 134MB
        ("[64,256,256,8] swapWH", 64, 256, 256, 8), // plane=524K, 64 planes, 134MB
        ("[16,512,512,4] swapWH", 16, 512, 512, 4), // plane=1M, 16 planes, 134MB
    ];
    println!("case                     OLD(ms)  NEW(ms)  NEW/OLD   bitmatch   [f32, min-9]");
    for (label, batch, a_dim, b_dim, elem) in cases {
        let total = batch * a_dim * b_dim * elem;
        let src: Vec<f32> = (0..total).map(|i| (i % 1013) as f32 + 0.5).collect();
        let a = old_permute(&src, batch, a_dim, b_dim, elem);
        let b = new_permute(&src, batch, a_dim, b_dim, elem);
        let bitmatch = a == b;
        let old_ms = bench(|| old_permute(&src, batch, a_dim, b_dim, elem).len());
        let new_ms = bench(|| new_permute(&src, batch, a_dim, b_dim, elem).len());
        let ratio = old_ms / new_ms;
        println!("  {label:<22} {old_ms:7.3} {new_ms:7.3}   {ratio:5.2}x   bitmatch={bitmatch}");
    }
}

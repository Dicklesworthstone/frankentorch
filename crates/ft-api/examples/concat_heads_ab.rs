//! A/B for the attention-head-merge (concat_attention_heads) pattern: [B,H,S,D] -> [B,S,H*D].
//! OLD = serial nested-loop gather; NEW = par-over-output-rows gather (strided reads). Tests the
//! DRAM-wall hypothesis (cf. window_partition regression). Run: cargo run --release -p ft-api --example concat_heads_ab

use rayon::prelude::*;
use std::time::Instant;

// Serial (mirrors ft-nn concat_attention_heads): read src[b,h,seq,:] contiguous, write strided.
fn old_concat(src: &[f64], b: usize, h: usize, s: usize, d: usize) -> Vec<f64> {
    let embed = h * d;
    let mut out = vec![0.0; b * s * embed];
    for batch in 0..b {
        for head in 0..h {
            for seq in 0..s {
                let so = batch * h * s * d + head * s * d + seq * d;
                let dof = batch * s * embed + seq * embed + head * d;
                out[dof..dof + d].copy_from_slice(&src[so..so + d]);
            }
        }
    }
    out
}

// Parallel over output rows (b*s): each row gathers H strided blocks from src (strided READ).
fn new_concat(src: &[f64], b: usize, h: usize, s: usize, d: usize) -> Vec<f64> {
    let embed = h * d;
    let mut out = vec![0.0; b * s * embed];
    out.par_chunks_mut(embed).enumerate().for_each(|(row, orow)| {
        let batch = row / s;
        let seq = row % s;
        for head in 0..h {
            let so = batch * h * s * d + head * s * d + seq * d;
            orow[head * d..head * d + d].copy_from_slice(&src[so..so + d]);
        }
    });
    out
}

fn bench<F: Fn() -> usize>(f: F) -> f64 {
    let mut best = f64::INFINITY;
    for _ in 0..9 {
        let t = Instant::now();
        let x = f();
        let el = t.elapsed().as_secs_f64() * 1e3;
        std::hint::black_box(x);
        if el < best {
            best = el;
        }
    }
    best
}

fn main() {
    println!("concat_attention_heads [B,H,S,D]->[B,S,H*D] f64, min-9:  OLD=serial  NEW=par-over-rows");
    let cases = [
        ("B32 H8 S512 D64", 32usize, 8, 512, 64),
        ("B16 H16 S256 D64", 16, 16, 256, 64),
        ("B8 H12 S1024 D64", 8, 12, 1024, 64),
    ];
    for (label, b, h, s, d) in cases {
        let n = b * h * s * d;
        let src: Vec<f64> = (0..n).map(|i| (i % 1009) as f64 + 0.5).collect();
        let bitmatch = old_concat(&src, b, h, s, d) == new_concat(&src, b, h, s, d);
        let o = bench(|| old_concat(&src, b, h, s, d).len());
        let nw = bench(|| new_concat(&src, b, h, s, d).len());
        println!(
            "  {label:<18} ({:>3}MB)  OLD {:8.3}  NEW {:8.3}  = {:.2}x  bitmatch={}",
            n * 8 / (1 << 20),
            o,
            nw,
            o / nw,
            bitmatch
        );
    }
}

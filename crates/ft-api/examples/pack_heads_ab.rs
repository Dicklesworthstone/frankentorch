//! A/B for pack_attention_heads (head-split [B,S,H*D]->[B,H,S,D]). OLD=serial nested loop;
//! NEW=par-over-(batch,head)-planes. Run: cargo run --release -p ft-api --example pack_heads_ab

use rayon::prelude::*;
use std::time::Instant;

fn old_pack(values: &[f64], b: usize, s: usize, h: usize, d: usize) -> Vec<f64> {
    let embed = h * d;
    let batch_stride = s * embed;
    let head_stride = s * d;
    let mut packed = vec![0.0; values.len()];
    for batch in 0..b {
        for seq in 0..s {
            for head in 0..h {
                let src = batch * batch_stride + seq * embed + head * d;
                let dst = (batch * h + head) * head_stride + seq * d;
                packed[dst..dst + d].copy_from_slice(&values[src..src + d]);
            }
        }
    }
    packed
}

fn new_pack(values: &[f64], b: usize, s: usize, h: usize, d: usize) -> Vec<f64> {
    let embed = h * d;
    let batch_stride = s * embed;
    let head_stride = s * d;
    let mut packed = vec![0.0; values.len()];
    packed.par_chunks_mut(head_stride).enumerate().for_each(|(bh, plane)| {
        let batch = bh / h;
        let head = bh % h;
        for seq in 0..s {
            let src = batch * batch_stride + seq * embed + head * d;
            plane[seq * d..seq * d + d].copy_from_slice(&values[src..src + d]);
        }
    });
    packed
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
    println!("pack_attention_heads [B,S,H*D]->[B,H,S,D] f64, min-9:  OLD=serial  NEW=par-over-planes");
    let cases = [("B32 H8 S512 D64", 32usize, 512, 8, 64), ("B16 H16 S256 D64", 16, 256, 16, 64), ("B8 H12 S1024 D64", 8, 1024, 12, 64)];
    for (label, b, s, h, d) in cases {
        let n = b * s * h * d;
        let src: Vec<f64> = (0..n).map(|i| (i % 1009) as f64 + 0.5).collect();
        let bm = old_pack(&src, b, s, h, d) == new_pack(&src, b, s, h, d);
        let o = bench(|| old_pack(&src, b, s, h, d).len());
        let nw = bench(|| new_pack(&src, b, s, h, d).len());
        println!("  {label:<18} ({:>3}MB)  OLD {:8.3}  NEW {:8.3}  = {:.2}x  bitmatch={}", n * 8 / (1 << 20), o, nw, o / nw, bm);
    }
}

//! A/B for the multi-head-attention reshape permute [B,S,H,D] -> [B,H,S,D]
//! (perm [0,2,1,3]) — a prefix+suffix-framed block-rotation now handled by the
//! cache-blocked transpose fast path in ft-autograd's permute_slice (batched
//! over B, transpose of [S,H] tiles whose element is a contiguous D-run). OLD =
//! the generic per-element N-D scatter that ran before. Both produce
//! bit-identical output. Self-contained (plain Vec<f64> logic).
//!   cargo run -q --release -p ft-kernel-cpu --example mha_permute_ab
use std::time::Instant;

fn generic_perm(src: &[f64], src_shape: &[usize], perm: &[usize]) -> Vec<f64> {
    let nd = src_shape.len();
    let mut sstr = vec![1usize; nd];
    for d in (0..nd - 1).rev() {
        sstr[d] = sstr[d + 1] * src_shape[d + 1];
    }
    let dshape: Vec<usize> = perm.iter().map(|&d| src_shape[d]).collect();
    let mut dstr = vec![1usize; nd];
    for d in (0..nd - 1).rev() {
        dstr[d] = dstr[d + 1] * dshape[d + 1];
    }
    let mut dst = vec![0.0f64; src.len()];
    let mut coord = vec![0usize; nd];
    for (fs, &v) in src.iter().enumerate() {
        let mut rem = fs;
        for d in 0..nd {
            coord[d] = rem / sstr[d];
            rem %= sstr[d];
        }
        let mut fd = 0;
        for d in 0..nd {
            fd += coord[perm[d]] * dstr[d];
        }
        dst[fd] = v;
    }
    dst
}

// Cache-blocked [0,2,1,3]: batched (b) transpose of [s, h] tiles of d-runs.
fn mha_blocked(src: &[f64], b: usize, s: usize, h: usize, d: usize) -> Vec<f64> {
    let plane = s * h * d;
    let mut dst = vec![0.0f64; b * plane];
    const TILE: usize = 16;
    for bi in 0..b {
        let base = bi * plane;
        let sgn = &src[base..base + plane];
        let dgn = &mut dst[base..base + plane];
        let mut ii = 0;
        while ii < s {
            let i_end = (ii + TILE).min(s);
            let mut jj = 0;
            while jj < h {
                let j_end = (jj + TILE).min(h);
                for i in ii..i_end {
                    for j in jj..j_end {
                        let so = (i * h + j) * d;
                        let do_ = (j * s + i) * d;
                        dgn[do_..do_ + d].clone_from_slice(&sgn[so..so + d]);
                    }
                }
                jj += TILE;
            }
            ii += TILE;
        }
    }
    dst
}

fn main() {
    // [B, S, H, D] multi-head attention reshape, a few representative shapes.
    for &(b, s, h, d) in &[(32usize, 512usize, 12usize, 64usize), (16, 1024, 16, 64)] {
        let shape = [b, s, h, d];
        let perm = [0usize, 2, 1, 3];
        let n = b * s * h * d;
        let src: Vec<f64> = (0..n).map(|i| (i % 9973) as f64 * 0.001).collect();
        let old = generic_perm(&src, &shape, &perm);
        let new = mha_blocked(&src, b, s, h, d);
        let bx = old.iter().zip(new.iter()).all(|(x, y)| x.to_bits() == y.to_bits());
        eprintln!("[MHA {b}x{s}x{h}x{d}] bit-exact: {bx}");
        assert!(bx, "mha blocked diverged from generic");
        let t = Instant::now();
        std::hint::black_box(generic_perm(&src, &shape, &perm));
        let o = t.elapsed().as_secs_f64() * 1e3;
        let t = Instant::now();
        std::hint::black_box(mha_blocked(&src, b, s, h, d));
        let ne = t.elapsed().as_secs_f64() * 1e3;
        eprintln!("  generic-4D {o:.2} ms / blocked {ne:.2} ms / speedup {:.2}x\n", o / ne);
    }
}

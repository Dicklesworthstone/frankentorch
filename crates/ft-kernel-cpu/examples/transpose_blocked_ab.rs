//! A/B for the cache-blocked 2-D transpose fast path in ft-autograd's
//! `permute_slice` (perm == [1,0]). The generic per-element scatter writes
//! dst[j*r+i] — striding by `r`, a cache miss per write for a large matrix.
//! Tiling confines each TILE×TILE block to cache. Pure permuted clone → both
//! produce bit-identical output. OLD = naive scatter (replicated); NEW = the
//! blocked algorithm now in production. Self-contained (plain Vec<f64> logic).
//!   cargo run -q --release -p ft-kernel-cpu --example transpose_blocked_ab
use std::time::Instant;

fn naive(src: &[f64], r: usize, c: usize) -> Vec<f64> {
    // Mirrors the generic permute_slice loop for perm=[1,0]: dst[j*r+i]=src[i*c+j].
    let mut dst = vec![0.0f64; r * c];
    for i in 0..r {
        for j in 0..c {
            dst[j * r + i] = src[i * c + j];
        }
    }
    dst
}

fn blocked(src: &[f64], r: usize, c: usize) -> Vec<f64> {
    let mut dst = vec![0.0f64; r * c];
    const TILE: usize = 16;
    let mut ii = 0;
    while ii < r {
        let i_end = (ii + TILE).min(r);
        let mut jj = 0;
        while jj < c {
            let j_end = (jj + TILE).min(c);
            for i in ii..i_end {
                let src_row = i * c;
                for j in jj..j_end {
                    dst[j * r + i] = src[src_row + j];
                }
            }
            jj += TILE;
        }
        ii += TILE;
    }
    dst
}

fn bench(name: &str, f: impl Fn(&[f64], usize, usize) -> Vec<f64>, src: &[f64], r: usize, c: usize, reps: usize) -> f64 {
    f(src, r, c);
    let mut best = f64::INFINITY;
    for _ in 0..reps {
        let t = Instant::now();
        std::hint::black_box(f(src, r, c));
        best = best.min(t.elapsed().as_secs_f64() * 1e3);
    }
    eprintln!("  {name:<8} {r}x{c}: {best:.3} ms");
    best
}

fn naive_batched(src: &[f64], batch: usize, r: usize, c: usize) -> Vec<f64> {
    let plane = r * c;
    let mut dst = vec![0.0f64; batch * plane];
    for b in 0..batch {
        let base = b * plane;
        for i in 0..r {
            for j in 0..c {
                dst[base + j * r + i] = src[base + i * c + j];
            }
        }
    }
    dst
}

fn blocked_batched(src: &[f64], batch: usize, r: usize, c: usize) -> Vec<f64> {
    let plane = r * c;
    let mut dst = vec![0.0f64; batch * plane];
    const TILE: usize = 16;
    for b in 0..batch {
        let base = b * plane;
        let s = &src[base..base + plane];
        let d = &mut dst[base..base + plane];
        let mut ii = 0;
        while ii < r {
            let i_end = (ii + TILE).min(r);
            let mut jj = 0;
            while jj < c {
                let j_end = (jj + TILE).min(c);
                for i in ii..i_end {
                    let sr = i * c;
                    for j in jj..j_end {
                        d[j * r + i] = s[sr + j];
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
    // Batched matrix-transpose (.mT) — attention QKᵀ shape [batch*heads, seq, dim].
    for &(batch, r, c) in &[(128usize, 512usize, 512usize), (256, 256, 256)] {
        let n = batch * r * c;
        let src: Vec<f64> = (0..n).map(|i| (i % 9973) as f64 * 0.001).collect();
        let a = naive_batched(&src, batch, r, c);
        let b = blocked_batched(&src, batch, r, c);
        let bx = a.iter().zip(b.iter()).all(|(x, y)| x.to_bits() == y.to_bits());
        eprintln!("[batched {batch}x{r}x{c}] bit-exact: {bx}");
        assert!(bx, "batched blocked diverged");
        let t = Instant::now();
        std::hint::black_box(naive_batched(&src, batch, r, c));
        let old = t.elapsed().as_secs_f64() * 1e3;
        let t = Instant::now();
        std::hint::black_box(blocked_batched(&src, batch, r, c));
        let new = t.elapsed().as_secs_f64() * 1e3;
        eprintln!("  naive {old:.2} ms / blocked {new:.2} ms / speedup {:.2}x\n", old / new);
    }
    for &(r, c) in &[(4096usize, 4096usize), (8192, 1024), (2048, 2048)] {
        let n = r * c;
        let src: Vec<f64> = (0..n).map(|i| (i % 9973) as f64 * 0.001).collect();
        let a = naive(&src, r, c);
        let b = blocked(&src, r, c);
        let bit_exact = a.iter().zip(b.iter()).all(|(x, y)| x.to_bits() == y.to_bits());
        eprintln!("[{r}x{c}] bit-exact (naive == blocked): {bit_exact}");
        assert!(bit_exact, "blocked transpose diverged from naive");
        let old = bench("naive", naive, &src, r, c, 30);
        let new = bench("blocked", blocked, &src, r, c, 30);
        eprintln!("  speedup: {:.2}x\n", old / new);
    }
}

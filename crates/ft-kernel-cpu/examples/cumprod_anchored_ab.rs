//! ANCHORED single-process A/B resolving the cumprod-vs-cumsum puzzle (66pe / h5gj).
//! The peer rejected cumprod parallelization on a SEPARATE-EXEC A/B (different workers,
//! ~1.4x variance — the same confound that faked the 1q8x "20x f32 gap"). Here we run
//! cumsum (a SHIPPED parallel win) as an ANCHOR and cumprod as the question, BOTH serial
//! and parallel, in ONE process on ONE worker under ONE thread pool. If the anchor shows
//! the expected parallel win and cumprod shows a comparable one, cumprod parallelization
//! is real and the rejection was worker-variance.
//!   cargo run -q --release -p ft-kernel-cpu --example cumprod_anchored_ab
use rayon::prelude::*;
use std::time::Instant;

#[inline]
fn cumsum_serial(data: &[f64], outer: usize, dim: usize, inner: usize) -> Vec<f64> {
    let lane = dim * inner;
    let mut out = vec![0.0; outer * lane];
    for o in 0..outer {
        for i in 0..inner {
            let mut acc = 0.0;
            for d in 0..dim {
                let idx = o * lane + d * inner + i;
                acc += data[idx];
                out[idx] = acc;
            }
        }
    }
    out
}
#[inline]
fn cumsum_parallel(data: &[f64], outer: usize, dim: usize, inner: usize) -> Vec<f64> {
    let lane = dim * inner;
    let mut out = vec![0.0; outer * lane];
    out.par_chunks_mut(lane).enumerate().for_each(|(o, chunk)| {
        let base = o * lane;
        for i in 0..inner {
            let mut acc = 0.0;
            for d in 0..dim {
                let idx = d * inner + i;
                acc += data[base + idx];
                chunk[idx] = acc;
            }
        }
    });
    out
}
#[inline]
fn cumprod_serial(data: &[f64], outer: usize, dim: usize, inner: usize) -> Vec<f64> {
    let lane = dim * inner;
    let mut out = vec![0.0; outer * lane];
    for o in 0..outer {
        for i in 0..inner {
            let mut acc = 1.0;
            for d in 0..dim {
                let idx = o * lane + d * inner + i;
                acc *= data[idx];
                out[idx] = acc;
            }
        }
    }
    out
}
#[inline]
fn cumprod_parallel(data: &[f64], outer: usize, dim: usize, inner: usize) -> Vec<f64> {
    let lane = dim * inner;
    let mut out = vec![0.0; outer * lane];
    out.par_chunks_mut(lane).enumerate().for_each(|(o, chunk)| {
        let base = o * lane;
        for i in 0..inner {
            let mut acc = 1.0;
            for d in 0..dim {
                let idx = d * inner + i;
                acc *= data[base + idx];
                chunk[idx] = acc;
            }
        }
    });
    out
}

fn best<F: Fn() -> Vec<f64>>(f: F) -> f64 {
    std::hint::black_box(f());
    let mut b = f64::INFINITY;
    for _ in 0..20 {
        let t = Instant::now();
        std::hint::black_box(f());
        b = b.min(t.elapsed().as_secs_f64() * 1e3);
    }
    b
}

fn main() {
    let (outer, dim, inner) = (8192usize, 1024usize, 1usize); // [8192,1024] dim=1
    let n = outer * dim * inner;
    let data: Vec<f64> = (0..n).map(|i| 1.0 + ((i % 7) as f64) * 1e-3).collect();
    let nthreads = rayon::current_num_threads();

    // bit-exactness: parallel == serial for BOTH ops
    let bit_exact = |a: &[f64], b: &[f64]| a.iter().zip(b).all(|(x, y)| x.to_bits() == y.to_bits());
    assert!(
        bit_exact(
            &cumsum_parallel(&data, outer, dim, inner),
            &cumsum_serial(&data, outer, dim, inner)
        ),
        "cumsum par != serial"
    );
    assert!(
        bit_exact(
            &cumprod_parallel(&data, outer, dim, inner),
            &cumprod_serial(&data, outer, dim, inner)
        ),
        "cumprod par != serial"
    );

    // Interleave serial/parallel timings on the SAME worker to neutralize drift.
    let cs_ser = best(|| cumsum_serial(&data, outer, dim, inner));
    let cs_par = best(|| cumsum_parallel(&data, outer, dim, inner));
    let cp_ser = best(|| cumprod_serial(&data, outer, dim, inner));
    let cp_par = best(|| cumprod_parallel(&data, outer, dim, inner));

    let (cs_x, cp_x) = (cs_ser / cs_par, cp_ser / cp_par);
    println!("[{outer},{dim}] dim=1  one process, {nthreads} threads, bit-exact OK");
    println!("ANCHOR  cumsum : serial {cs_ser:.2}ms  parallel {cs_par:.2}ms  => {cs_x:.2}x");
    println!("QUESTION cumprod: serial {cp_ser:.2}ms  parallel {cp_par:.2}ms  => {cp_x:.2}x");
}

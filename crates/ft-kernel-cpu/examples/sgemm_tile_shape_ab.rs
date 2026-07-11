//! Is `gemm::tile_shape` load-imbalanced at 32 threads, and does a balanced grid win?
//!
//! `tile_shape` picks `p = floor(sqrt(T))`, `q = ceil(T/p)`. At T=32 that is p=5, q=7 =>
//! **35 tiles on 32 threads** — one straggler wave of 3 tiles while 29 threads idle.
//! A balanced grid takes `p` = the largest DIVISOR of T that is <= sqrt(T)  (32 -> 4x8).
//!
//! Note the bug is thread-count-specific: T=64 gives 8x8 and T=16 gives 4x4, both already
//! balanced. T=32 is exactly franken_whisper's encoder thread cap, so it is the case that
//! matters. This example therefore FORCES a 32-thread pool.
//!
//! Arms (order rotated every rep; a single process, so no cross-worker variance):
//!   A  real `matmul_tensor_contiguous_f32_into`  (ships today; uses current tile_shape)
//!   B  local replication of the CURRENT grid     (fidelity guard — must equal A)
//!   C  local replication of the BALANCED grid    (the lever)
//!
//! B exists because a previous dig replicated ft's scheduler WRONG and drew a false
//! conclusion. If B does not match A bit-for-bit and closely in time, the replication is
//! broken and C's number means nothing. Do not delete arm B.
//!
//! All three must be BIT-EXACT: every output element's full k-accumulation happens inside
//! one serial micro-kernel call, and neither the row nor the column count changes that
//! order.
//!
//! Run remotely (local builds are disk-constrained):
//!   rch exec -- cargo run --release -p ft-kernel-cpu --example sgemm_tile_shape_ab

#![allow(unsafe_code)]

use ft_core::{DType, Device, TensorMeta};
use rayon::prelude::*;
use std::hint::black_box;
use std::time::Instant;

const MIN_BLOCK_ROWS: usize = 8;
const MIN_BLOCK_COLS: usize = 128;

#[derive(Clone, Copy)]
struct TilePtr(*mut f32);
unsafe impl Send for TilePtr {}
unsafe impl Sync for TilePtr {}

/// what `gemm::tile_shape` does today
fn grid_current(threads: usize) -> (usize, usize) {
    let p = (threads as f64).sqrt().floor().max(1.0) as usize;
    (p, threads.div_ceil(p))
}

/// largest divisor of `threads` that is <= sqrt(threads); q = threads / p  => p*q == threads
fn grid_balanced(threads: usize) -> (usize, usize) {
    let lim = (threads as f64).sqrt().floor().max(1.0) as usize;
    let mut p = 1;
    for cand in 1..=lim {
        if threads % cand == 0 {
            p = cand;
        }
    }
    (p, threads / p)
}

fn blocks(m: usize, n: usize, p: usize, q: usize) -> (usize, usize) {
    (
        m.div_ceil(p).max(MIN_BLOCK_ROWS),
        n.div_ceil(q).max(MIN_BLOCK_COLS),
    )
}

fn tile_count(m: usize, n: usize, mb: usize, nb: usize) -> usize {
    m.div_ceil(mb) * n.div_ceil(nb)
}

/// f32 mirror of `gemm::sgemm_2d_parallel` with an explicit (mb, nb)
fn tiled(a: &[f32], b: &[f32], c: &mut [f32], m: usize, k: usize, n: usize, mb: usize, nb: usize) {
    let cp = TilePtr(c.as_mut_ptr());
    let mut tiles: Vec<(usize, usize)> = Vec::new();
    let mut i0 = 0;
    while i0 < m {
        let mut j0 = 0;
        while j0 < n {
            tiles.push((i0, j0));
            j0 += nb;
        }
        i0 += mb;
    }
    tiles.into_par_iter().for_each(|(i0, j0)| {
        let cp = &cp;
        let bi = (i0 + mb).min(m) - i0;
        let bj = (j0 + nb).min(n) - j0;
        unsafe {
            matrixmultiply::sgemm(
                bi,
                k,
                bj,
                1.0,
                a.as_ptr().add(i0 * k),
                k as isize,
                1,
                b.as_ptr().add(j0),
                n as isize,
                1,
                0.0,
                cp.0.add(i0 * n + j0),
                n as isize,
                1,
            );
        }
    });
}

fn ft_mm(a: &[f32], b: &[f32], c: &mut Vec<f32>, m: usize, k: usize, n: usize) {
    let am = TensorMeta::from_shape(vec![m, k], DType::F32, Device::Cpu);
    let bm = TensorMeta::from_shape(vec![k, n], DType::F32, Device::Cpu);
    ft_kernel_cpu::matmul_tensor_contiguous_f32_into(c, a, b, &am, &bm).unwrap();
}

fn fill(seed: u64, n: usize) -> Vec<f32> {
    let mut s = seed | 1;
    (0..n)
        .map(|_| {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            ((s >> 40) as f32 / 16_777_216.0) - 0.5
        })
        .collect()
}

fn stat(v: &mut Vec<f64>) -> (f64, f64) {
    v.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let min = v[0];
    let mean = v.iter().sum::<f64>() / v.len() as f64;
    let sd = (v.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / v.len() as f64).sqrt();
    (min, 100.0 * sd / mean)
}

fn main() {
    let threads: usize = std::env::var("TILE_AB_THREADS")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(32);
    let avail = std::thread::available_parallelism().map_or(0, std::num::NonZeroUsize::get);
    rayon::ThreadPoolBuilder::new()
        .num_threads(threads)
        .build_global()
        .unwrap();
    let reps: usize = std::env::var("TILE_AB_REPS")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(9);

    let host = std::fs::read_to_string("/proc/sys/kernel/hostname")
        .map(|s| s.trim().to_string())
        .unwrap_or_else(|_| "?".into());
    // A ratio is admissible ONLY if both arms ran in this one process (they do) AND the
    // host can actually field the thread count under test. rch picks workers
    // non-deterministically and the ORIG/CAND ratio is NOT worker-invariant, so a ratio
    // compared across two rch invocations is meaningless -- never do that.
    let admissible = avail >= threads;

    let (pc, qc) = grid_current(threads);
    let (pb, qb) = grid_balanced(threads);
    println!("host={host}  threads={threads}  available_parallelism={avail}  reps={reps}");
    if !admissible {
        println!(
            "  !! WARNING: host has {avail} < {threads} hw threads -- pool is oversubscribed {}x;",
            threads.div_ceil(avail.max(1))
        );
        println!(
            "  !! rayon work-steals across the oversubscription and SMEARS the straggler being measured."
        );
    }
    println!("  current  grid p x q = {pc} x {qc}  (p*q = {})", pc * qc);
    println!("  balanced grid p x q = {pb} x {qb}  (p*q = {})", pb * qb);
    if pc * qc == threads {
        println!(
            "  !! at T={threads} the current grid is ALREADY balanced -- this thread count cannot show the lever"
        );
    }
    println!();

    let shapes: &[(&str, usize, usize, usize, usize)] = &[
        ("turbo qkv/out [1500,1280]x[1280,1280]", 1500, 1280, 1280, 4),
        ("turbo fc1     [1500,1280]x[1280,5120]", 1500, 1280, 5120, 1),
        ("turbo fc2     [1500,5120]x[5120,1280]", 1500, 5120, 1280, 1),
    ];

    println!(
        "{:<40} {:>8} {:>8} {:>8}   {:>7} {:>7}  {:>11}",
        "shape", "A ft ms", "B cur ms", "C bal ms", "B/A", "C/A", "tiles cur/bal"
    );
    let (mut la, mut lc) = (0.0f64, 0.0f64);
    let mut all_bit = true;

    for (label, m, k, n, cnt) in shapes {
        let (m, k, n, cnt) = (*m, *k, *n, *cnt);
        let a = fill(1, m * k);
        let b = fill(7, k * n);
        let numel = m * n;
        let (mbc, nbc) = blocks(m, n, pc, qc);
        let (mbb, nbb) = blocks(m, n, pb, qb);
        let (tc, tb) = (tile_count(m, n, mbc, nbc), tile_count(m, n, mbb, nbb));

        let mut ca: Vec<f32> = vec![0.0; numel];
        let mut cb: Vec<f32> = vec![0.0; numel];
        let mut cc: Vec<f32> = vec![0.0; numel];
        ft_mm(&a, &b, &mut ca, m, k, n);
        tiled(&a, &b, &mut cb, m, k, n, mbc, nbc);
        tiled(&a, &b, &mut cc, m, k, n, mbb, nbb);
        let ab = ca
            .iter()
            .zip(cb.iter())
            .all(|(x, y)| x.to_bits() == y.to_bits());
        let ac = ca
            .iter()
            .zip(cc.iter())
            .all(|(x, y)| x.to_bits() == y.to_bits());
        all_bit &= ab && ac;

        let ra = |c: &mut Vec<f32>| {
            let t = Instant::now();
            ft_mm(&a, &b, c, m, k, n);
            black_box(&c[0]);
            t.elapsed().as_secs_f64() * 1e3
        };
        let rb = |c: &mut Vec<f32>| {
            let t = Instant::now();
            tiled(&a, &b, c, m, k, n, mbc, nbc);
            black_box(&c[0]);
            t.elapsed().as_secs_f64() * 1e3
        };
        let rc = |c: &mut Vec<f32>| {
            let t = Instant::now();
            tiled(&a, &b, c, m, k, n, mbb, nbb);
            black_box(&c[0]);
            t.elapsed().as_secs_f64() * 1e3
        };
        black_box(ra(&mut ca));
        black_box(rb(&mut cb));
        black_box(rc(&mut cc));

        let (mut va, mut vb, mut vc) = (Vec::new(), Vec::new(), Vec::new());
        for r in 0..reps {
            match r % 3 {
                0 => {
                    va.push(ra(&mut ca));
                    vb.push(rb(&mut cb));
                    vc.push(rc(&mut cc));
                }
                1 => {
                    vc.push(rc(&mut cc));
                    va.push(ra(&mut ca));
                    vb.push(rb(&mut cb));
                }
                _ => {
                    vb.push(rb(&mut cb));
                    vc.push(rc(&mut cc));
                    va.push(ra(&mut ca));
                }
            }
        }
        let (fa, cva) = stat(&mut va);
        let (fb, cvb) = stat(&mut vb);
        let (fc, cvc) = stat(&mut vc);
        la += fa * cnt as f64;
        lc += fc * cnt as f64;
        println!(
            "{:<40} {:>8.2} {:>8.2} {:>8.2}   {:>6.3}x {:>6.3}x  {:>5}/{:<5}  cv {:.1}/{:.1}/{:.1}%  bitexact A==B {} A==C {}",
            label,
            fa,
            fb,
            fc,
            fa / fb,
            fa / fc,
            tc,
            tb,
            cva,
            cvb,
            cvc,
            ab,
            ac
        );
    }

    println!(
        "\nTURBO LINEAR-GEMM LAYER (4x qkv/out + fc1 + fc2):  A {:.2} ms -> C {:.2} ms = {:.3}x",
        la,
        lc,
        la / lc
    );
    println!("all arms bit-exact: {all_bit}   (this half IS certified: it is host-independent)");

    println!("\n=== VERDICT ===");
    if admissible {
        let enc = 1.0 / (1.0 - 0.728 + 0.728 / (la / lc));
        println!(
            "PERF ADMISSIBLE (host={host}, {avail} hw threads >= {threads} under test, single binary, single invocation)"
        );
        println!(
            "projected: encoder {:.3}x -> e2e {:.3}x  (linear GEMMs 72.8% of encoder_window, encoder 89.5% of e2e)",
            enc,
            1.0 / (0.105 + 0.895 / enc)
        );
    } else {
        println!(
            "PERF *** NOT ADMISSIBLE *** on host={host}: {avail} hw threads < {threads} under test."
        );
        println!(
            "DO NOT QUOTE ANY RATIO ABOVE. Re-run on a host with >= {threads} physical cores."
        );
    }
    println!(
        "\nFIDELITY GUARD: B/A must be ~1.00x and A==B bit-exact. If not, the replication is wrong and C is meaningless."
    );
    println!(
        "SUBSTRATE: all arms run in ONE binary + ONE invocation, order rotated per rep, so host identity and"
    );
    println!(
        "drift cancel WITHIN a run. Ratios are NOT worker-invariant: never compare a ratio from one rch"
    );
    println!("invocation against a ratio from another (franken_networkx br-r37-c1-839yx).");
}

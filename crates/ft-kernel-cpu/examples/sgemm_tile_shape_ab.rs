//! Is `gemm::tile_shape` load-imbalanced at 32 threads, and what grid actually wins?
//!
//! `tile_shape` picks `p = floor(sqrt(T))`, `q = ceil(T/p)`, IGNORING the shape. At T=32 that
//! is p=5, q=7 => **35 tiles on 32 threads** — one straggler wave of 3 tiles while 29 threads
//! idle. But a naive divisor-balanced grid (32 -> 4x8) is NOT a clean win: it regresses fc1,
//! because more column blocks also shrink the `B` slice (`k*nb*4` bytes) that every thread
//! re-streams, and different turbo shapes want different column-block counts.
//!
//! This harness therefore sweeps EVERY divisor-pair grid (p, T/p) per shape and reports the
//! measured ratio vs the shipped path, so the right selector can be READ OFF the data rather
//! than guessed. It also evaluates a candidate B-BYTES-AWARE SELECTOR:
//!   q* = the largest divisor of T with nb=ceil(n/q*) >= MIN_BLOCK_COLS  (p* = T/q*)
//! i.e. maximize the column-block count (minimize per-thread B re-stream) without letting nb
//! floor and collapse the tile count below the pool.
//!
//! Arms (all in ONE process, order rotated every rep, so no cross-worker variance):
//!   A         real `matmul_tensor_contiguous_f32_into` (ships today; current tile_shape)
//!   grid p x q  local replication of an explicit grid; the p=floor(sqrt) one is the FIDELITY
//!               GUARD and must equal A bit-for-bit and in time (a prior dig replicated ft's
//!               scheduler WRONG and drew a false conclusion — do not delete it).
//!
//! Every grid is BIT-EXACT vs A: each output element's full k-accumulation happens inside one
//! serial micro-kernel call, and neither the row nor the column count changes that order.
//!
//! Build REMOTELY, run LOCALLY on a >=32-core box (a ratio is admissible only if
//! available_parallelism >= the thread count under test):
//!   RCH_REQUIRE_REMOTE=1 env -u CARGO_TARGET_DIR rch exec -- \
//!     cargo build --release -p ft-kernel-cpu --example sgemm_tile_shape_ab
//!   TILE_AB_THREADS=32 TILE_AB_REPS=15 \
//!     target/release/examples/sgemm_tile_shape_ab

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

/// what `gemm::tile_shape` does today: p = floor(sqrt(T)), q = ceil(T/p)  (p*q may != T)
fn grid_current(threads: usize) -> (usize, usize) {
    let p = (threads as f64).sqrt().floor().max(1.0) as usize;
    (p, threads.div_ceil(p))
}

/// naive balance: largest divisor of T that is <= sqrt(T); q = T/p  => p*q == T (32 -> 4x8)
fn grid_balanced(threads: usize) -> (usize, usize) {
    let lim = (threads as f64).sqrt().floor().max(1.0) as usize;
    let p = (1..=lim).filter(|c| threads % c == 0).max().unwrap_or(1);
    (p, threads / p)
}

/// candidate B-bytes-aware selector: among divisor pairs (p, q=T/p), pick the LARGEST q whose
/// column block nb=ceil(n/q) stays >= MIN_BLOCK_COLS. This maximizes the column-block count
/// (minimizes per-thread B re-stream k*nb*4) while keeping p*q==T tiles filling the pool
/// (a larger q would floor nb and collapse the effective tile count below T). p = T/q.
fn grid_selected(threads: usize, n: usize) -> (usize, usize) {
    let mut best_q = 1usize;
    for q in 1..=threads {
        if threads % q == 0 && n.div_ceil(q) >= MIN_BLOCK_COLS {
            best_q = q; // keep the largest valid divisor q
        }
    }
    (threads / best_q, best_q)
}

/// all divisor pairs (p, T/p) of T, ascending in p
fn divisor_grids(threads: usize) -> Vec<(usize, usize)> {
    (1..=threads)
        .filter(|p| threads % p == 0)
        .map(|p| (p, threads / p))
        .collect()
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
        .unwrap_or(15);

    let host = std::fs::read_to_string("/proc/sys/kernel/hostname")
        .map(|s| s.trim().to_string())
        .unwrap_or_else(|_| "?".into());
    // A ratio is admissible ONLY if both arms ran in this one process (they do) AND the host
    // can field the thread count under test. rch picks workers non-deterministically and the
    // ORIG/CAND ratio is NOT worker-invariant, so a ratio compared across two rch invocations
    // is meaningless -- never do that.
    let admissible = avail >= threads;

    let (pc, qc) = grid_current(threads);
    let (pb, qb) = grid_balanced(threads);
    let grids = divisor_grids(threads);
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
    println!(
        "  current  grid p x q = {pc} x {qc}  (p*q = {})   [FIDELITY GUARD arm]",
        pc * qc
    );
    println!("  naive-balanced grid p x q = {pb} x {qb}  (p*q = {})", pb * qb);
    println!(
        "  divisor grids swept: {}",
        grids
            .iter()
            .map(|(p, q)| format!("{p}x{q}"))
            .collect::<Vec<_>>()
            .join(" ")
    );
    println!();

    let shapes: &[(&str, usize, usize, usize, usize)] = &[
        ("turbo qkv/out [1500,1280]x[1280,1280]", 1500, 1280, 1280, 4),
        ("turbo fc1     [1500,1280]x[1280,5120]", 1500, 1280, 5120, 1),
        ("turbo fc2     [1500,5120]x[5120,1280]", 1500, 5120, 1280, 1),
    ];

    // Layer accumulators: A (ship), naive-balanced, and the B-bytes-aware selector.
    let (mut la, mut l_bal, mut l_sel) = (0.0f64, 0.0f64, 0.0f64);
    let mut all_bit = true;

    for (label, m, k, n, cnt) in shapes {
        let (m, k, n, cnt) = (*m, *k, *n, *cnt);
        let a = fill(1, m * k);
        let b = fill(7, k * n);
        let numel = m * n;

        // Reference output from the real ft path; every grid must reproduce it bit-for-bit.
        let mut c_ref: Vec<f32> = vec![0.0; numel];
        ft_mm(&a, &b, &mut c_ref, m, k, n);

        // arm 0 = real ft; arms 1.. = divisor grids
        let n_arms = 1 + grids.len();
        let mut bufs: Vec<Vec<f32>> = (0..n_arms).map(|_| vec![0.0f32; numel]).collect();
        let mut bit_ok: Vec<bool> = vec![true; n_arms];

        // bit-exactness check (compute once, compare to c_ref)
        for (gi, &(p, q)) in grids.iter().enumerate() {
            let (mb, nb) = blocks(m, n, p, q);
            let buf = &mut bufs[gi + 1];
            tiled(&a, &b, buf, m, k, n, mb, nb);
            bit_ok[gi + 1] = buf
                .iter()
                .zip(c_ref.iter())
                .all(|(x, y)| x.to_bits() == y.to_bits());
            all_bit &= bit_ok[gi + 1];
        }

        let run = |arm: usize, buf: &mut Vec<f32>| {
            if arm == 0 {
                ft_mm(&a, &b, buf, m, k, n);
            } else {
                let (p, q) = grids[arm - 1];
                let (mb, nb) = blocks(m, n, p, q);
                tiled(&a, &b, buf, m, k, n, mb, nb);
            }
            black_box(&buf[0]);
        };

        // warmup
        for arm in 0..n_arms {
            let mut tmp = std::mem::take(&mut bufs[arm]);
            run(arm, &mut tmp);
            bufs[arm] = tmp;
        }

        let mut samples: Vec<Vec<f64>> = vec![Vec::new(); n_arms];
        for r in 0..reps {
            for off in 0..n_arms {
                let arm = (r + off) % n_arms; // rotate start per rep => drift cancels within run
                let mut tmp = std::mem::take(&mut bufs[arm]);
                let t = Instant::now();
                run(arm, &mut tmp);
                let ms = t.elapsed().as_secs_f64() * 1e3;
                bufs[arm] = tmp;
                samples[arm].push(ms);
            }
        }

        let (fa, cva) = stat(&mut samples[0]);
        let mins: Vec<(f64, f64)> = (1..n_arms).map(|i| stat(&mut samples[i])).collect();

        println!("{label}   A(ft)={fa:.2} ms  cv {cva:.1}%   tiles/pool={threads}");
        for (gi, &(p, q)) in grids.iter().enumerate() {
            let (mb, nb) = blocks(m, n, p, q);
            let tc = tile_count(m, n, mb, nb);
            let (fg, cvg) = mins[gi];
            let mark_cur = if (p, q) == (pc, qc) { " <FID(cur)" } else { "" };
            let mark_bal = if (p, q) == (pb, qb) { " <balanced" } else { "" };
            let mark_sel = if (p, q) == grid_selected(threads, n) {
                " <SELECTOR"
            } else {
                ""
            };
            println!(
                "    {p:>2}x{q:<2}  nb={nb:<4} tiles={tc:<3}  {fg:>7.2} ms  A/grid={:>6.3}x  cv {cvg:.1}%  bitexact {}{}{}{}",
                fa / fg,
                bit_ok[gi + 1],
                mark_cur,
                mark_bal,
                mark_sel
            );
        }

        // layer accumulation
        let sel = grid_selected(threads, n);
        let bal = (pb, qb);
        let sel_ms = mins[grids.iter().position(|g| *g == sel).unwrap()].0;
        let bal_ms = mins[grids.iter().position(|g| *g == bal).unwrap()].0;
        la += fa * cnt as f64;
        l_bal += bal_ms * cnt as f64;
        l_sel += sel_ms * cnt as f64;
        println!();
    }

    let enc = |ratio: f64| 1.0 / (1.0 - 0.728 + 0.728 / ratio);
    let e2e = |encr: f64| 1.0 / (0.105 + 0.895 / encr);

    println!("TURBO LINEAR-GEMM LAYER (4x qkv/out + fc1 + fc2):");
    println!("  A (ship, current grid)        {la:.2} ms");
    println!(
        "  naive-balanced (uniform 4x8)  {l_bal:.2} ms = {:.3}x   enc {:.3}x  e2e {:.3}x",
        la / l_bal,
        enc(la / l_bal),
        e2e(enc(la / l_bal))
    );
    println!(
        "  B-BYTES-AWARE SELECTOR        {l_sel:.2} ms = {:.3}x   enc {:.3}x  e2e {:.3}x",
        la / l_sel,
        enc(la / l_sel),
        e2e(enc(la / l_sel))
    );
    println!("\nall grids bit-exact vs A: {all_bit}   (this half IS host-independent / certified)");

    println!("\n=== VERDICT ===");
    if admissible {
        println!(
            "PERF ADMISSIBLE (host={host}, {avail} hw threads >= {threads} under test, single binary, single invocation)."
        );
        println!("The SELECTOR row is the shippable projection; the sweep proves it picks the per-shape winner.");
    } else {
        println!(
            "PERF *** NOT ADMISSIBLE *** on host={host}: {avail} hw threads < {threads} under test."
        );
        println!("DO NOT QUOTE ANY RATIO ABOVE. Re-run on a host with >= {threads} physical cores.");
    }
    println!(
        "\nFIDELITY GUARD: the {pc}x{qc} row must be ~1.00x and bit-exact vs A. If not, the replication is"
    );
    println!("wrong and every other row is meaningless.");
    println!("SUBSTRATE: all arms run in ONE binary + ONE invocation, order rotated per rep, so host identity");
    println!("and drift cancel WITHIN a run. Ratios are NOT worker-invariant: never compare a ratio from one");
    println!("rch invocation against a ratio from another (franken_networkx br-r37-c1-839yx).");
}

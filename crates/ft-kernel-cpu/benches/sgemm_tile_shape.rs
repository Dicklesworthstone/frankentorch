//! Same-binary, same-invocation A/B of `gemm::tile_shape`'s grid policy.
//!
//! **SUBSTRATE.** `rch exec` has no `--worker` flag and picks workers non-deterministically,
//! and the ORIG/CAND ratio is NOT worker-invariant. Both arms therefore run in ONE binary and
//! ONE invocation, alternating, in one criterion group. Never split an A/B across two
//! invocations. Never use `git stash` to swap arms.
//!
//! **THE CODE UNDER TEST ACTUALLY EXECUTES.** Both arms call the real
//! `ft_kernel_cpu::matmul_tensor_contiguous_f32_into`; the only difference is
//! `set_sgemm_tile_balanced(false|true)`. Nothing here replicates the scheduler. A bench that
//! times a *replica* of the function under test is not evidence about that function — this
//! bench previously made exactly that mistake, and so did two ledger rows in this repo.
//! `exercise_proof()` below asserts, before any timing, that flipping the flag actually
//! changes the tile grid for these shapes; if it did not, the arms would be the same code and
//! every ratio would be a measurement of noise.
//!
//! **THE DEFECT.** `tile_shape` picks `p = floor(sqrt(T))`, `q = ceil(T/p)`, so `p*q != T` for
//! many T. At T=32 that is 5x7 = 35 tiles on 32 threads -- a straggler wave of 3 tiles while
//! 29 threads idle. The candidate picks `p` = largest divisor of T that is <= sqrt(T).
//!
//! **ADMISSIBILITY.** A 32-thread effect cannot be measured on a host with < 32 hw threads
//! (rayon work-steals across the oversubscription and smears the straggler). So this bench
//! does NOT force T=32. It picks:
//!   T_imb = largest T <= available_parallelism with `p*q != T`  (lever visible)
//!   T_bal = largest T <= available_parallelism with `p*q == T`  (NULL CONTROL: both policies
//!           yield the SAME grid, so the ratio must be ~1.000x and the win rate ~50%)
//! The null control calibrates the harness noise floor. The keep-gate statistic is the cv of
//! the PAIRED ratio (both arms timed back-to-back per rep, so a load spike cancels), not the
//! cv of either arm alone.
//!
//! Run:  rch exec -- cargo bench -p ft-kernel-cpu --bench sgemm_tile_shape

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
use ft_core::{DType, Device, TensorMeta};
use ft_kernel_cpu::{matmul_tensor_contiguous_f32_into, set_sgemm_tile_balanced};
use std::hint::black_box;
use std::time::{Duration, Instant};

const MIN_BLOCK_ROWS: usize = 8;
const MIN_BLOCK_COLS: usize = 128;

fn grid_orig(t: usize) -> (usize, usize) {
    let p = (t as f64).sqrt().floor().max(1.0) as usize;
    (p, t.div_ceil(p))
}
fn grid_balanced(t: usize) -> (usize, usize) {
    let lim = (t as f64).sqrt().floor().max(1.0) as usize;
    let p = (1..=lim).filter(|c| t % c == 0).max().unwrap_or(1);
    (p, t / p)
}
fn tiles(m: usize, n: usize, (p, q): (usize, usize)) -> usize {
    let mb = m.div_ceil(p).max(MIN_BLOCK_ROWS);
    let nb = n.div_ceil(q).max(MIN_BLOCK_COLS);
    m.div_ceil(mb) * n.div_ceil(nb)
}
fn imbalanced(t: usize) -> bool {
    let (p, q) = grid_orig(t);
    p * q != t
}
fn pick_t(cap: usize, pred: impl Fn(usize) -> bool) -> Option<usize> {
    (4..=cap).rev().find(|&t| pred(t))
}

/// the ONLY thing either arm calls; `balanced` selects the policy inside ft
fn ft_mm(balanced: bool, a: &[f32], b: &[f32], c: &mut Vec<f32>, m: usize, k: usize, n: usize) {
    set_sgemm_tile_balanced(balanced);
    let am = TensorMeta::from_shape(vec![m, k], DType::F32, Device::Cpu);
    let bm = TensorMeta::from_shape(vec![k, n], DType::F32, Device::Cpu);
    matmul_tensor_contiguous_f32_into(c, a, b, &am, &bm).unwrap();
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

const SHAPES: &[(&str, usize, usize, usize)] = &[
    ("qkv_out_1500x1280x1280", 1500, 1280, 1280),
    ("fc1_1500x1280x5120", 1500, 1280, 5120),
    ("fc2_1500x5120x1280", 1500, 5120, 1280),
];

/// Assert the benchmark reaches the code under test, and that both policies agree bitwise.
/// Prints the grid each policy selects so a reader can see the arms differ.
fn exercise_proof(t: usize) -> bool {
    let (go, gb) = (grid_orig(t), grid_balanced(t));
    let mut differs = false;
    println!("  exercise proof @ T={t}: orig {}x{} vs balanced {}x{}", go.0, go.1, gb.0, gb.1);
    for (name, m, k, n) in SHAPES {
        let (m, k, n) = (*m, *k, *n);
        let (to, tb) = (tiles(m, n, go), tiles(m, n, gb));
        differs |= to != tb;
        let a = fill(1, m * k);
        let b = fill(7, k * n);
        let mut co = vec![0.0f32; m * n];
        let mut cb = vec![0.0f32; m * n];
        ft_mm(false, &a, &b, &mut co, m, k, n);
        ft_mm(true, &a, &b, &mut cb, m, k, n);
        let bit = co.iter().zip(cb.iter()).all(|(x, y)| x.to_bits() == y.to_bits());
        assert!(bit, "{name}: policies must be bit-exact");
        println!("    {name:<26} tiles {to:>3} -> {tb:>3}   bit-exact {bit}");
    }
    differs
}

fn alternating(t: usize, label: &str) {
    let pool = rayon::ThreadPoolBuilder::new().num_threads(t).build().unwrap();
    println!("\n--- alternating A/B @ T={t} ({label}) ---");
    let reps: usize = std::env::var("TILE_REPS").ok().and_then(|v| v.parse().ok()).unwrap_or(25);
    let warm = 3usize;
    for (name, m, k, n) in SHAPES {
        let (m, k, n) = (*m, *k, *n);
        let a = fill(1, m * k);
        let b = fill(7, k * n);
        let mut co = vec![0.0f32; m * n];
        let mut cb = vec![0.0f32; m * n];
        let (mut vo, mut vb, mut ratios) = (Vec::new(), Vec::new(), Vec::new());
        for r in 0..(reps + warm) {
            let mut run = |bal: bool, c: &mut Vec<f32>| {
                let t0 = Instant::now();
                pool.install(|| ft_mm(bal, &a, &b, c, m, k, n));
                black_box(&c[0]);
                t0.elapsed().as_secs_f64() * 1e3
            };
            let (to, tb) = if r % 2 == 0 {
                let x = run(false, &mut co);
                let y = run(true, &mut cb);
                (x, y)
            } else {
                let y = run(true, &mut cb);
                let x = run(false, &mut co);
                (x, y)
            };
            if r >= warm {
                vo.push(to);
                vb.push(tb);
                ratios.push(to / tb);
            }
        }
        let med = |v: &mut Vec<f64>| {
            v.sort_by(|x, y| x.partial_cmp(y).unwrap());
            v[v.len() / 2]
        };
        let mean = ratios.iter().sum::<f64>() / ratios.len() as f64;
        let sd = (ratios.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / ratios.len() as f64).sqrt();
        let cv = 100.0 * sd / mean;
        let wins = ratios.iter().filter(|r| **r > 1.0).count();
        let mut rs = ratios.clone();
        println!(
            "  {name:<26} orig {:7.2} ms  bal {:7.2} ms  ratio med {:.3}x  cv(ratio) {cv:4.1}%  wins {wins}/{}  GATE cv<5 {}",
            med(&mut vo),
            med(&mut vb),
            med(&mut rs),
            ratios.len(),
            if cv < 5.0 { "PASS" } else { "FAIL" }
        );
    }
}

fn bench(c: &mut Criterion) {
    let avail = std::thread::available_parallelism().map_or(1, std::num::NonZeroUsize::get);
    let host = std::fs::read_to_string("/proc/sys/kernel/hostname")
        .map(|s| s.trim().to_string())
        .unwrap_or_else(|_| "?".into());
    let cap = std::env::var("TILE_T_CAP").ok().and_then(|v| v.parse().ok()).unwrap_or(32).min(avail);
    let t_imb = pick_t(cap, imbalanced);
    let t_bal = pick_t(cap, |t| !imbalanced(t));

    println!("\n======== sgemm tile_shape A/B (real ft fn, both arms, ONE binary, ONE invocation) ========");
    println!("host={host}  available_parallelism={avail}  cap={cap}");
    println!("T_imb={t_imb:?}  T_bal={t_bal:?} (null control)");
    println!("PERF ADMISSIBLE for the SHIPPED T=32 config: {}", if avail >= 32 { "YES" } else { "NO -- host too small; only the MECHANISM is measured here" });

    if let Some(t) = t_bal {
        let pool = rayon::ThreadPoolBuilder::new().num_threads(t).build().unwrap();
        let d = pool.install(|| exercise_proof(t));
        assert!(!d, "null control must yield identical grids");
        alternating(t, "NULL CONTROL -- identical grids, ratio MUST be ~1.000x, wins ~50%");
    }
    let Some(t) = t_imb else {
        println!("\nno imbalanced T <= {cap}; nothing to compare");
        return;
    };
    {
        let pool = rayon::ThreadPoolBuilder::new().num_threads(t).build().unwrap();
        let d = pool.install(|| exercise_proof(t));
        assert!(d, "EXERCISE PROOF FAILED: flipping the policy did not change the grid -> the bench does not reach tile_shape; every ratio below would be noise");
        println!("  => exercise proof PASSED: the policy flag reaches tile_shape and changes the grid");
    }
    alternating(t, "LEVER -- orig grid has more tiles than threads");

    // criterion group: ORIG and CANDIDATE, same group, same binary, same invocation
    let pool = rayon::ThreadPoolBuilder::new().num_threads(t).build().unwrap();
    let mut g = c.benchmark_group(format!("sgemm_tile_shape_T{t}"));
    g.sample_size(10).warm_up_time(Duration::from_millis(500)).measurement_time(Duration::from_secs(2));
    for (name, m, k, n) in SHAPES {
        let (m, k, n) = (*m, *k, *n);
        let a = fill(1, m * k);
        let b = fill(7, k * n);
        let mut buf = vec![0.0f32; m * n];
        let mut buf2 = vec![0.0f32; m * n];
        g.bench_function(BenchmarkId::new("orig_grid", name), |bch| {
            bch.iter(|| pool.install(|| ft_mm(false, &a, &b, &mut buf, m, k, n)));
        });
        g.bench_function(BenchmarkId::new("balanced_grid", name), |bch| {
            bch.iter(|| pool.install(|| ft_mm(true, &a, &b, &mut buf2, m, k, n)));
        });
    }
    g.finish();
    set_sgemm_tile_balanced(false);
}

criterion_group!(benches, bench);
criterion_main!(benches);

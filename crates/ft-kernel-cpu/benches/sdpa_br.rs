//! `BR` (query-row tile) sweep for the REAL `sdpa_forward_f32`.
//!
//! **The lever.** The kernel tiles query rows at `BR` and, per block, calls
//! `sgemm_bt(m=BR, k=d_k, n=seq_k)` (B = `kh`) and `sgemm(m=BR, k=seq_k, n=d_v)` (B = `vh`).
//! `kh`/`vh` are invariant across a head's `ceil(seq_q/BR)` blocks, yet `matrixmultiply`
//! repacks them on every call — it has no prepack API. A larger `BR` amortizes both B-packs.
//! Counter-force: the per-block scratch is `BR * seq_k * 4` bytes and leaves L2.
//! `perf annotate` on the shipped engine puts `matrixmultiply::gemm::gemm_loop` (the pack
//! loop) at 4.05% of e2e self-time, ~23% of SDPA's sgemm cost.
//!
//! **The code under test really runs.** Both arms call the real `sdpa_forward_f32`, differing
//! only by `set_sdpa_br`. `exercise_proof` asserts, before timing, that the two `BR`s produce
//! different block counts (else the arms are the same code and every ratio is noise) and that
//! the outputs are **bit-identical** (BR can only change scheduling: matrixmultiply's
//! k-accumulation order is fixed by the micro-kernel and independent of the row count, and the
//! softmax is per-row).
//!
//! **Substrate.** Both arms live in ONE binary and ONE rch invocation, and are **interleaved
//! inside a single measured routine** (`paired`), forming a per-rep paired ratio so a load
//! spike cancels. Criterion group members run sequentially and would NOT cancel drift; this
//! bench does not rely on them. Every input is fed through `black_box` and the FULL output is
//! consumed through `black_box`. A NULL CONTROL (BR=64 vs BR=64) calibrates the noise floor —
//! its ratio must be ~1.000x with a ~50% win rate.
//!
//! **Generality.** The old parked sweep was blocked because the parallel-split guard read
//! `seq_q > BR`, so raising `BR` silently stripped the row-block split for consumers with
//! `64 < seq_q <= BR`. That guard is now the constant `SDPA_PAR_MIN_ROWS`, so a `BR` sweep
//! measures one scheduler, not two. This bench therefore also sweeps `seq_q` and `num_bh`.
//!
//! Run: RCH_REQUIRE_REMOTE=1 env -u CARGO_TARGET_DIR rch exec -- \
//!        cargo bench -p ft-kernel-cpu --bench sdpa_br

use criterion::{criterion_group, criterion_main, Criterion};
use ft_kernel_cpu::{sdpa_forward_f32, set_sdpa_br};
use std::hint::black_box;
use std::time::Instant;

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

#[inline]
fn consume(v: &[f32]) -> f32 {
    let mut a = 0.0f32;
    for c in v.chunks(97) {
        a += c[0];
    }
    black_box(a)
}

fn run(br: usize, q: &[f32], k: &[f32], v: &[f32], nbh: usize, sq: usize, sk: usize, d: usize, scale: f32) -> (f64, Vec<f32>) {
    set_sdpa_br(br);
    let t = Instant::now();
    let o = sdpa_forward_f32(
        black_box(q), black_box(k), black_box(v),
        black_box(nbh), black_box(sq), black_box(sk), black_box(d), black_box(d),
        black_box(scale), false,
    );
    let dt = t.elapsed().as_secs_f64() * 1e3;
    black_box(consume(&o));
    (dt, o)
}

fn blocks(sq: usize, br: usize) -> usize { sq.div_ceil(br) }

/// interleaved, paired, order-alternated. returns (med_a, med_b, med_ratio, cv_pct, wins, n)
#[allow(clippy::too_many_arguments)]
fn paired(a_br: usize, b_br: usize, q: &[f32], k: &[f32], v: &[f32], nbh: usize, sq: usize, sk: usize, d: usize, reps: usize) -> (f64, f64, f64, f64, usize, usize) {
    let scale = 1.0 / (d as f32).sqrt();
    let warm = 3usize;
    let (mut va, mut vb, mut rs) = (Vec::new(), Vec::new(), Vec::new());
    for r in 0..(reps + warm) {
        let (ta, tb) = if r % 2 == 0 {
            let (x, _) = run(a_br, q, k, v, nbh, sq, sk, d, scale);
            let (y, _) = run(b_br, q, k, v, nbh, sq, sk, d, scale);
            (x, y)
        } else {
            let (y, _) = run(b_br, q, k, v, nbh, sq, sk, d, scale);
            let (x, _) = run(a_br, q, k, v, nbh, sq, sk, d, scale);
            (x, y)
        };
        if r >= warm { va.push(ta); vb.push(tb); rs.push(ta / tb); }
    }
    let med = |x: &mut Vec<f64>| { x.sort_by(|p, q| p.partial_cmp(q).unwrap()); x[x.len() / 2] };
    let mean = rs.iter().sum::<f64>() / rs.len() as f64;
    let sd = (rs.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / rs.len() as f64).sqrt();
    let wins = rs.iter().filter(|x| **x > 1.0).count();
    let n = rs.len();
    let mut rc = rs.clone();
    (med(&mut va), med(&mut vb), med(&mut rc), 100.0 * sd / mean, wins, n)
}

fn bench(_c: &mut Criterion) {
    let avail = std::thread::available_parallelism().map_or(1, std::num::NonZeroUsize::get);
    let host = std::fs::read_to_string("/proc/sys/kernel/hostname").map(|s| s.trim().to_string()).unwrap_or_else(|_| "?".into());
    let reps: usize = std::env::var("BR_REPS").ok().and_then(|v| v.parse().ok()).unwrap_or(9);
    println!("\n===== sdpa_forward_f32 BR sweep — REAL fn, one binary, interleaved =====");
    println!("host={host} available_parallelism={avail} reps={reps} threads={}", rayon::current_num_threads());

    // large-v3-turbo encoder shape
    let (nbh, sq, sk, d) = (20usize, 1500usize, 1500usize, 64usize);
    let q = fill(1, nbh * sq * d);
    let k = fill(7, nbh * sk * d);
    let v = fill(13, nbh * sk * d);
    let scale = 1.0 / (d as f32).sqrt();

    // ---- exercise proof + bit-exactness, BEFORE any timing ----
    let (_, base) = run(64, &q, &k, &v, nbh, sq, sk, d, scale);
    println!("\nexercise proof (turbo shape nbh={nbh} seq={sq} d={d}):");
    for br in [32usize, 96, 128, 160] {
        let (_, o) = run(br, &q, &k, &v, nbh, sq, sk, d, scale);
        let bit = base.iter().zip(o.iter()).all(|(x, y)| x.to_bits() == y.to_bits());
        assert!(bit, "BR={br} changed results — BR must be bit-exact");
        assert_ne!(blocks(sq, br), blocks(sq, 64), "BR={br} yields the same block count as 64 — arms are identical code");
        println!("  BR {br:>3}: blocks {:>3} (vs 24 at BR=64)  scratch {:>4} KiB  bit-exact {bit}", blocks(sq, br), br * sk * 4 / 1024);
    }
    println!("  => flipping BR changes the schedule and never the result");

    println!("\n{:<34} {:>9} {:>9} {:>8} {:>7} {:>8} {:>6}", "arm (vs BR=64)", "BR64 ms", "cand ms", "ratio", "cv%", "wins", "gate");
    // NULL CONTROL first: identical code both arms
    let (a, b, r, cv, w, n) = paired(64, 64, &q, &k, &v, nbh, sq, sk, d, reps);
    println!("{:<34} {a:>9.1} {b:>9.1} {r:>7.4}x {cv:>6.1} {w:>4}/{n:<3} {}", "NULL CONTROL (BR=64 vs BR=64)", if cv < 5.0 { "PASS" } else { "FAIL" });

    for br in [96usize, 128, 160] {
        let (a, b, r, cv, w, n) = paired(64, br, &q, &k, &v, nbh, sq, sk, d, reps);
        println!("{:<34} {a:>9.1} {b:>9.1} {r:>7.4}x {cv:>6.1} {w:>4}/{n:<3} {}", format!("BR={br}"), if cv < 5.0 { "PASS" } else { "FAIL" });
    }

    // ---- generality: the guard is now a constant, so short-seq consumers keep their split ----
    println!("\ngenerality sweep (the reason this lever was parked): seq_q x num_bh, BR=64 vs 128");
    for (nbh2, sq2) in [(20usize, 96usize), (20, 128), (4, 1500), (64, 1500)] {
        let q2 = fill(1, nbh2 * sq2 * d);
        let k2 = fill(7, nbh2 * sq2 * d);
        let v2 = fill(13, nbh2 * sq2 * d);
        let (_, o64) = run(64, &q2, &k2, &v2, nbh2, sq2, sq2, d, scale);
        let (_, o128) = run(128, &q2, &k2, &v2, nbh2, sq2, sq2, d, scale);
        assert!(o64.iter().zip(o128.iter()).all(|(x, y)| x.to_bits() == y.to_bits()), "bit-exact must hold at every shape");
        let (a, b, r, cv, w, n) = paired(64, 128, &q2, &k2, &v2, nbh2, sq2, sq2, d, reps.min(7));
        println!("  nbh={nbh2:<3} seq={sq2:<5} BR64 {a:>7.2} ms  BR128 {b:>7.2} ms  ratio {r:.4}x  cv {cv:.1}%  wins {w}/{n}  bit-exact");
    }
    set_sdpa_br(64);
}

criterion_group!(benches, bench);
criterion_main!(benches);

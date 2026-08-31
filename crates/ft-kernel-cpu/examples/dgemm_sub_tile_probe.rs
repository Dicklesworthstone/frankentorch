//! `frankentorch-valnx` — ISOLATION proof for the `dgemm_sub_into` 2-D tile arm.
//!
//! # What is wrong today
//!
//! `dgemm_sub_into` is the trailing-update GEMM of the whole LU family — `lu_factor`, `slogdet`,
//! `inv`, `lu_solve` — and the blocked cholesky/QR/bidiag paths. Its ONLY parallelism is a split
//! of the N axis into `block_cols(n)`-wide strips, and `block_cols` floors the strip WIDTH at
//! `MIN_BLOCK_COLS = 128`. A floor on width is a CEILING ON COUNT:
//!
//!     n = 512, 16 threads  ->  512 / 128  =  4 blocks   (25% occupancy)
//!     n = 256              ->              =  2 blocks   (12.5%)
//!     n = 128              ->              =  1 block    (serial)
//!
//! and the LU trailing submatrix SHRINKS every panel, so a factorisation walks down that list
//! while its remaining work is still O(n^2) per step. The M axis is never split at all.
//!
//! # What this measures, and why isolation first
//!
//! GFLOP/s for the exact operation, at the exact shapes the LU trailing update produces, with the
//! tile arm OFF and ON. No session, no tape, no incumbent — if the kernel does not get faster in
//! isolation there is nothing to take to a lane, and `feedback_insitu_over_standalone` records the
//! converse trap (a standalone ladder that INVERTED in situ), so this proves the kernel claim only
//! and the lane row is taken separately.
//!
//! Shapes are taken from a real n=512 factorisation with panel width nb: after panel `pe` the
//! update is `m = n-pe-nb` rows by `n-pe-nb` cols with `k = nb`. That is a SKINNY-K GEMM, which is
//! why the column split's occupancy matters so much — there is little per-element work to hide it.
//!
//! BIT-EXACTNESS is asserted here too, not merely asserted in prose: both arms run the same
//! shapes and every output element is compared with `to_bits()`. K is never tiled, so each
//! element's reduction stays inside one `dgemm_mm` call; this checks that claim rather than
//! trusting it.
//!
//! Config is by ARGV, never env: `rch exec` does not forward the environment, so an env-configured
//! probe silently measures its default on a worker and exits 0.
//!
//!   cargo run --release -p frankentorch-kernel-cpu --example dgemm_sub_tile_probe -- [n] [nb] [reps]

use ft_kernel_cpu::set_dgemm_sub_tile_2d;

/// One trailing-update shape, timed. Returns (milliseconds, checksum-bits).
fn time_update(m: usize, k: usize, n: usize, reps: usize) -> (f64, Vec<f64>) {
    let a: Vec<f64> = (0..m * k).map(|i| (i % 97) as f64 * 0.01 - 0.5).collect();
    let b: Vec<f64> = (0..k * n).map(|i| (i % 89) as f64 * 0.013 - 0.4).collect();
    let base: Vec<f64> = (0..m * n).map(|i| (i % 71) as f64 * 0.02 - 0.7).collect();

    // Warm: first touch of a fresh buffer is serial page faults, which would land on whichever
    // arm ran first (ledger 289).
    for _ in 0..2 {
        let mut c = base.clone();
        ft_kernel_cpu::probe_dgemm_sub_into(m, k, n, &a, &b, &mut c, 0, n);
        std::hint::black_box(&c);
    }

    let mut best = f64::INFINITY;
    let mut out = Vec::new();
    for _ in 0..reps {
        let mut c = base.clone();
        let started = std::time::Instant::now();
        ft_kernel_cpu::probe_dgemm_sub_into(m, k, n, &a, &b, &mut c, 0, n);
        let ms = started.elapsed().as_secs_f64() * 1_000.0;
        best = best.min(ms);
        out = c;
    }
    (best, out)
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let n: usize = args.get(1).and_then(|v| v.parse().ok()).unwrap_or(512);
    let nb: usize = args.get(2).and_then(|v| v.parse().ok()).unwrap_or(64);
    let reps: usize = args.get(3).and_then(|v| v.parse().ok()).unwrap_or(9);

    let host = std::fs::read_to_string("/etc/hostname").unwrap_or_else(|_| "unknown\n".to_owned());
    let load = std::fs::read_to_string("/proc/loadavg").unwrap_or_else(|_| "unknown".to_owned());
    println!(
        "PROV host={} nproc={} rayon={} n={n} nb={nb} reps={reps} loadavg={}",
        host.trim(),
        std::thread::available_parallelism().map_or(0, std::num::NonZero::get),
        rayon::current_num_threads(),
        load.split_whitespace().take(3).collect::<Vec<_>>().join(","),
    );
    println!(
        "Shapes are the n={n} LU trailing updates: after panel pe the update is \
         (n-pe-nb) x nb x (n-pe-nb)."
    );
    println!(
        "\n{:>6} {:>6} {:>6} {:>10} {:>10} {:>9} {:>9} {:>8} {:>7}",
        "pe", "m", "k", "OFF ms", "ON ms", "OFF GF/s", "ON GF/s", "speedup", "bitwise"
    );

    let (mut sum_off, mut sum_on) = (0.0_f64, 0.0_f64);
    let mut all_exact = true;
    let mut pe = 0;
    while pe + nb < n {
        let m = n - pe - nb;
        let k = nb;
        let cols = m;
        // 2 flops per multiply-add.
        let flops = 2.0 * (m as f64) * (k as f64) * (cols as f64);

        set_dgemm_sub_tile_2d(false);
        let (off_ms, off_c) = time_update(m, k, cols, reps);
        set_dgemm_sub_tile_2d(true);
        let (on_ms, on_c) = time_update(m, k, cols, reps);
        set_dgemm_sub_tile_2d(false);

        let exact = off_c.len() == on_c.len()
            && off_c
                .iter()
                .zip(&on_c)
                .all(|(x, y)| x.to_bits() == y.to_bits());
        all_exact &= exact;
        sum_off += off_ms;
        sum_on += on_ms;
        println!(
            "{pe:>6} {m:>6} {k:>6} {off_ms:>10.4} {on_ms:>10.4} {:>9.1} {:>9.1} {:>8.3} {:>7}",
            flops / (off_ms / 1000.0) / 1e9,
            flops / (on_ms / 1000.0) / 1e9,
            off_ms / on_ms,
            if exact { "yes" } else { "NO" },
        );
        pe += nb;
    }

    println!(
        "\nTOTAL over all panels: OFF {sum_off:.3} ms  ON {sum_on:.3} ms  speedup {:.4}x",
        sum_off / sum_on
    );
    println!(
        "BITWISE across every shape: {}",
        if all_exact {
            "IDENTICAL"
        } else {
            "*** DIVERGED — the tile arm is not bit-exact and must not ship ***"
        }
    );
    println!(
        "READING: this is the KERNEL in isolation. A win here is necessary, not sufficient — \
         `feedback_insitu_over_standalone` records a 5.7x standalone ladder that became a 1.118x \
         REGRESSION in situ. The lane row is taken separately."
    );
}

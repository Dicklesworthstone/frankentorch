//! `frankentorch-valnx` — ISOLATION proof for the Cholesky panel's three formulations.
//!
//! # What is being decided
//!
//! Ledger 292/292a: the panel is 53-58% of an n=512 Cholesky, 96.5% of the panel is the
//! per-column sub-diagonal row dots, and it runs at **0.81 GFLOP/s** against the trailing update's
//! ~90. The rows are independent, so fanning them out is the obvious lever — and the obvious lever
//! is exactly the one that should be distrusted here.
//!
//! **The arithmetic predicts the hazard before the run.** A panel column carries ~2,730
//! multiply-accumulates and ~6.7 us of work, and item 255 priced a rayon fork on this host at
//! ~7 us. **The fork costs about what the column costs.** That is the same shape that made the
//! small in-place ops a loss (item 289d's 2 MiB row, 10.85x SLOWER), so this compares against the
//! SERIAL baseline AT PANEL SHAPES rather than at friendly ones: nb=128-wide columns whose row
//! count shrinks from nb-1 to 0 as the column advances, which is the only shape the panel ever
//! actually runs.
//!
//! # The arms
//!
//!   0  SHIPPED: serial per-column dots.
//!   1  PARALLEL ROWS: the same dots, independent rows fanned across the pool.
//!   2  LEVEL-2 RECAST: the same dots batched FOUR ROWS at a time over one pass of the diagonal
//!      row — one traversal of `drow` serving four rows, sixteen FMA chains in flight.
//!
//! All three are BIT-EXACT with each other (`cholesky_panel_modes_agree_bitwise`), because every
//! row keeps its own four-chain FMA sequence and its `(s0+s1)+(s2+s3)` tail. No accumulation moves,
//! so no tolerance argument is needed — and this probe re-checks it at every shape anyway rather
//! than relying on the unit test's shapes.
//!
//! # Registered prediction, before the run
//!
//! If dispatch dominates, arm 1 is at best flat and plausibly a LOSS at every nb, and it gets worse
//! as nb shrinks (fewer rows per fork). Arm 2 pays no dispatch at all, so if the panel is limited
//! by per-row overhead and diagonal-row re-reads it should win modestly and uniformly. If BOTH are
//! flat, the panel is limited by the dependent FMA chain itself and neither formulation is the
//! lever — which is a result, not a failure.
//!
//! Config by ARGV, never env.
//!
//!   cargo run --release -p frankentorch-kernel-cpu --example cholesky_panel_arms -- [n] [reps]

use ft_kernel_cpu::{probe_cholesky_panel_factor, set_cholesky_panel_mode};

/// Multiply-accumulates in one panel of width `nb`: column `c` does one diagonal dot of length `c`
/// and `nb-c-1` row dots of the same length.
fn panel_macs(nb: usize) -> f64 {
    (0..nb).map(|c| (c * (nb - c)) as f64).sum()
}

fn spd_panel(n: usize) -> Vec<f64> {
    let mut a = vec![0.0f64; n * n];
    for i in 0..n {
        for j in 0..i {
            let v = ((i * 17 + j * 31) % 23) as f64 * 0.01 - 0.1;
            a[i * n + j] = v;
            a[j * n + i] = v;
        }
        a[i * n + i] = n as f64;
    }
    a
}

fn time_mode(base: &[f64], n: usize, nb: usize, mode: u8, reps: usize) -> (f64, Vec<f64>) {
    set_cholesky_panel_mode(mode);
    // Warm: a fresh buffer's first touch is serial page faults and would land on whichever arm
    // ran first (ledger 289).
    for _ in 0..3 {
        let mut l = base.to_vec();
        probe_cholesky_panel_factor(&mut l, n, 0, nb).expect("panel");
        std::hint::black_box(&l);
    }
    let mut best = f64::INFINITY;
    let mut out = Vec::new();
    for _ in 0..reps {
        let mut l = base.to_vec();
        let started = std::time::Instant::now();
        probe_cholesky_panel_factor(&mut l, n, 0, nb).expect("panel");
        let ms = started.elapsed().as_secs_f64() * 1_000.0;
        if ms < best {
            best = ms;
        }
        out = l;
    }
    set_cholesky_panel_mode(0);
    (best, out)
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let n: usize = args.get(1).and_then(|v| v.parse().ok()).unwrap_or(512);
    let reps: usize = args.get(2).and_then(|v| v.parse().ok()).unwrap_or(31);

    let host = std::fs::read_to_string("/etc/hostname").unwrap_or_else(|_| "unknown\n".to_owned());
    let load = std::fs::read_to_string("/proc/loadavg").unwrap_or_else(|_| "unknown".to_owned());
    println!(
        "PROV host={} nproc={} rayon={} n={n} reps={reps} loadavg={}",
        host.trim(),
        std::thread::available_parallelism().map_or(0, std::num::NonZero::get),
        rayon::current_num_threads(),
        load.split_whitespace().take(3).collect::<Vec<_>>().join(","),
    );
    println!(
        "PREDICTION REGISTERED: a panel column is ~2,730 MACs / ~6.7 us and a rayon fork is ~7 us \
         (item 255), so arm 1 is expected to be flat-to-LOSS and to worsen as nb shrinks. Arm 2 \
         pays no dispatch. If BOTH are flat the panel is chain-limited and neither is the lever."
    );
    println!(
        "\n{:>5} {:>10} {:>10} {:>10} {:>9} {:>9} {:>9} {:>9} {:>8}",
        "nb", "serial ms", "par ms", "lvl2 ms", "ser GF/s", "par GF/s", "lvl2 GF/s", "par x",
        "lvl2 x"
    );

    let base = spd_panel(n);
    let mut all_exact = true;
    for nb in [16usize, 32, 64, 96, 128] {
        if nb > n {
            continue;
        }
        let macs = panel_macs(nb);
        let (s_ms, s_out) = time_mode(&base, n, nb, 0, reps);
        let (p_ms, p_out) = time_mode(&base, n, nb, 1, reps);
        let (l_ms, l_out) = time_mode(&base, n, nb, 2, reps);
        let exact = s_out
            .iter()
            .zip(&p_out)
            .all(|(x, y)| x.to_bits() == y.to_bits())
            && s_out
                .iter()
                .zip(&l_out)
                .all(|(x, y)| x.to_bits() == y.to_bits());
        all_exact &= exact;
        let gf = |ms: f64| 2.0 * macs / (ms / 1000.0) / 1e9;
        println!(
            "{nb:>5} {s_ms:>10.4} {p_ms:>10.4} {l_ms:>10.4} {:>9.2} {:>9.2} {:>9.2} {:>9.3} {:>8.3}{}",
            gf(s_ms),
            gf(p_ms),
            gf(l_ms),
            s_ms / p_ms,
            s_ms / l_ms,
            if exact { "" } else { "  *** NOT BIT-EXACT ***" }
        );
    }
    println!(
        "\nBITWISE across every shape: {}",
        if all_exact {
            "IDENTICAL (all three arms)"
        } else {
            "*** DIVERGED — an arm changed the arithmetic and must not ship ***"
        }
    );
    println!(
        "READING: isolation only. A win here is necessary and not sufficient — item 291 recorded a \
         bit-exact 1.28-1.40x isolation win that moved no lane at all. The paired cholesky row is \
         taken separately and decides."
    );
}

//! How much of the f32 GroupNorm engine gap is the gradient WIDENING? — `frankentorch-68pwz`.
//!
//! WHY. On the certified 8-thread board the GroupNorm family splits cleanly:
//!
//!   group_norm_f32_zeroed  (full session path)          0.185  = 5.41x SLOWER
//!   group_norm_f32_kernels (kernels direct, no tape)    1.259-1.439  = FASTER
//!   group_norm_f32_statskernels                         1.782-1.893  = FASTER
//!
//! The kernels already beat the incumbent; the ENGINE is the whole gap. Inside the f32
//! backward closure (`ft-api/src/lib.rs`, the GroupNormF32SumShortcut path) the gradient
//! is handed back to the tape as
//!
//!     Some(dx.iter().map(|&v| f64::from(v)).collect::<Vec<f64>>())
//!
//! `dx` is one element per input: for the scored lane [32,64,56,56] that is 6,422,528
//! values, so this is a SERIAL 51 MiB materialization sitting inside the timed backward.
//!
//! This probe sizes that one line before anything is changed, because a lever aimed at a
//! phase nobody has priced is how this ledger's refuted items got written. It is
//! arm-internal: no incumbent, no ratio, no drift gate.
//!
//! f32 -> f64 is EXACT (every f32 is representable), and an elementwise map has no
//! accumulation order, so any parallel form is bit-identical by construction. That is
//! asserted here rather than assumed.

use rayon::prelude::*;
use std::time::Instant;

/// The scored lane: [GN_N, GN_C, GN_H, GN_W] = [32, 64, 56, 56].
const NUMEL: usize = 32 * 64 * 56 * 56;

fn loadavg() -> String {
    std::fs::read_to_string("/proc/loadavg")
        .map(|raw| raw.split_whitespace().take(3).collect::<Vec<_>>().join(" / "))
        .unwrap_or_else(|_| "unavailable".to_owned())
}

fn cpu_mhz() -> String {
    let mut mhz: Vec<f64> = (0..)
        .map_while(|cpu| {
            std::fs::read_to_string(format!(
                "/sys/devices/system/cpu/cpu{cpu}/cpufreq/scaling_cur_freq"
            ))
            .ok()
        })
        .filter_map(|raw| raw.trim().parse::<f64>().ok().map(|khz| khz / 1000.0))
        .collect();
    if mhz.is_empty() {
        return "unavailable".to_owned();
    }
    mhz.sort_by(|a, b| a.partial_cmp(b).unwrap());
    format!(
        "min={:.0} mean={:.0} max={:.0} spread={:.2}x",
        mhz[0],
        mhz.iter().sum::<f64>() / mhz.len() as f64,
        mhz[mhz.len() - 1],
        mhz[mhz.len() - 1] / mhz[0]
    )
}

fn main() {
    // Non-trivial values so nothing folds, and a checksum so no route is elided.
    let source: Vec<f32> = (0..NUMEL)
        .map(|index| ((index % 9973) as f32) * 0.000_37 - 1.5)
        .collect();

    println!("gradient_widen_probe (frankentorch-68pwz)");
    println!("numel={NUMEL} ({:.1} MiB as f32 -> {:.1} MiB as f64)",
        (NUMEL * 4) as f64 / (1024.0 * 1024.0),
        (NUMEL * 8) as f64 / (1024.0 * 1024.0));
    println!("rayon_threads={}", rayon::current_num_threads());
    println!("pre  loadavg {}", loadavg());
    println!("pre  cpu_mhz {}", cpu_mhz());
    println!();

    let reps = 9;
    let mut serial_best = f64::INFINITY;
    let mut parallel_best = f64::INFINITY;
    let mut serial_last: Vec<f64> = Vec::new();
    let mut parallel_last: Vec<f64> = Vec::new();

    for _ in 0..reps {
        // (a) exactly what ships today
        let started = Instant::now();
        let widened: Vec<f64> = source.iter().map(|&v| f64::from(v)).collect();
        let elapsed = started.elapsed().as_secs_f64() * 1_000.0;
        assert_eq!(widened.len(), NUMEL);
        if elapsed < serial_best {
            serial_best = elapsed;
        }
        serial_last = widened;

        // (b) the same map, parallel. `collect` into a Vec via rayon still does one
        // pass and lets the workers do the page first-touch.
        let started = Instant::now();
        let widened: Vec<f64> = source.par_iter().map(|&v| f64::from(v)).collect();
        let elapsed = started.elapsed().as_secs_f64() * 1_000.0;
        assert_eq!(widened.len(), NUMEL);
        if elapsed < parallel_best {
            parallel_best = elapsed;
        }
        parallel_last = widened;
    }

    // Bit-exactness is structural here, but assert it rather than claim it.
    let mismatches = serial_last
        .iter()
        .zip(parallel_last.iter())
        .filter(|(a, b)| a.to_bits() != b.to_bits())
        .count();
    assert_eq!(mismatches, 0, "widening must be bit-identical");

    println!("{:>28}  {:>10}", "route", "min ms");
    println!("{:>28}  {serial_best:>10.3}", "serial  .iter().map()");
    println!("{:>28}  {parallel_best:>10.3}", "parallel .par_iter()");
    println!();
    println!(
        "speedup {:.2}x, absolute saving {:.3} ms per backward",
        serial_best / parallel_best,
        serial_best - parallel_best
    );
    println!("bit-identical: {} mismatches over {NUMEL} elements", mismatches);
    println!();
    println!(
        "CONTEXT: the group_norm_f32 lane's FT arm measured 30-52 ms forward+backward on \
         the board, so compare the saving against that."
    );
    println!();

    // Where should the parallel gate sit? Several of the 14 widening sites in ft-api are
    // per-channel (length 64), where rayon's fork/join would be pure loss. Measured
    // rather than guessed, so the constant that ships has a number behind it.
    println!("CROSSOVER SWEEP (min of 9; serial/parallel > 1 means parallel wins)");
    println!("{:>10}  {:>10}  {:>10}  {:>16}", "numel", "serial ms", "par ms", "serial/parallel");
    for &n in &[
        1_024usize, 4_096, 16_384, 65_536, 262_144, 1_048_576, 4_194_304,
    ] {
        let small = &source[..n];
        let mut s_best = f64::INFINITY;
        let mut p_best = f64::INFINITY;
        for _ in 0..9 {
            let started = Instant::now();
            let widened: Vec<f64> = small.iter().map(|&v| f64::from(v)).collect();
            let elapsed = started.elapsed().as_secs_f64() * 1_000.0;
            std::hint::black_box(&widened);
            s_best = s_best.min(elapsed);

            let started = Instant::now();
            let widened: Vec<f64> = small.par_iter().map(|&v| f64::from(v)).collect();
            let elapsed = started.elapsed().as_secs_f64() * 1_000.0;
            std::hint::black_box(&widened);
            p_best = p_best.min(elapsed);
        }
        println!(
            "{n:>10}  {s_best:>10.4}  {p_best:>10.4}  {:>15.2}x",
            s_best / p_best
        );
    }
    println!();
    println!("post loadavg {}", loadavg());
    println!("post cpu_mhz {}", cpu_mhz());
}

//! The OTHER numel-scaled pass in the f32 norm backward: the incoming `dy` NARROW —
//! `frankentorch-68pwz`.
//!
//! WHY THIS EXISTS. Item 69 priced and removed the serial f32 -> f64 WIDEN on the way OUT
//! of the f32 GroupNorm backward closure (24.9 ms of a 30-52 ms arm). It stopped there.
//! But every one of the four f32 norm backward closures in `ft-api/src/lib.rs` opens with
//! the MIRROR of that line, on the way IN:
//!
//!     let dy: Vec<f32> = grad_outputs[0].iter().map(|&v| v as f32).collect();
//!
//! `grad_outputs[0]` is the tape's f64 gradient — one element per input, the SAME
//! 6,422,528 values for the scored [32,64,56,56] lane. So the closure has always had TWO
//! serial numel-scaled materializations, and item 69 removed one of them.
//!
//! Item 69's follow-up sweep (item 69f) reported that every remaining conversion in the
//! file was "batch-, window-, index- or grid-sized". That sweep grepped `f64::from`. Both
//! directions of the norm-closure conversions are spelled `v as f64` / `v as f32`, so the
//! sweep could not see them and the finding was scoped to its own grep, not to the file.
//!
//! This probe sizes the narrow before anything is changed, and finds ITS crossover
//! separately rather than reusing the widen's: a narrow reads 8 bytes and writes 4 per
//! element, the widen reads 4 and writes 8, so they are not the same traffic and there is
//! no reason to assume they share a gate.
//!
//! Arm-internal: no incumbent, no ratio, no drift gate. It answers "how big is this line",
//! which is the question item 69 showed is worth asking first.
//!
//! f64 -> f32 ROUNDS (it is not exact like the widen), but it is still ELEMENTWISE: each
//! output depends on exactly one input and on nothing else, so splitting the range cannot
//! change a bit. That is the property the parallel form needs, and it is asserted here
//! over values that include subnormals, both infinities, NaN and f32-overflow rather than
//! claimed.

use rayon::prelude::*;
use std::time::Instant;

/// The scored lane: [GN_N, GN_C, GN_H, GN_W] = [32, 64, 56, 56].
const NUMEL: usize = 32 * 64 * 56 * 56;

fn loadavg() -> String {
    std::fs::read_to_string("/proc/loadavg")
        .map(|raw| {
            raw.split_whitespace()
                .take(3)
                .collect::<Vec<_>>()
                .join(" / ")
        })
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

/// Adversarial values for the bit-exactness assert: the narrow is the direction that
/// rounds, so the check has to straddle the cases where rounding is interesting.
fn edge_values() -> Vec<f64> {
    vec![
        0.0,
        -0.0,
        f64::INFINITY,
        f64::NEG_INFINITY,
        f64::NAN,
        f64::MAX,          // overflows f32 -> inf
        -f64::MAX,         // overflows f32 -> -inf
        f64::MIN_POSITIVE, // underflows f32 -> 0
        f64::from(f32::MAX),
        f64::from(f32::MIN_POSITIVE),       // smallest normal f32
        f64::from(f32::MIN_POSITIVE) / 2.0, // f32 subnormal
        f64::from(f32::EPSILON),
        // exact ties: round-to-nearest-even has to break these the same way in both routes
        f64::from(1.0f32) + f64::from(f32::EPSILON) / 2.0,
        f64::from(1.0f32) + f64::from(f32::EPSILON) * 1.5,
        3.402_823_6e38, // just under f32::MAX
        3.402_823_7e38, // just over -> inf
    ]
}

fn main() {
    // Non-trivial values so nothing folds, and a checksum so no route is elided. The tape
    // hands the closure genuine f64s, so the source here is f64 (unlike the widen probe,
    // whose source is the kernel's f32 output).
    let source: Vec<f64> = (0..NUMEL)
        .map(|index| ((index % 9973) as f64) * 0.000_37 - 1.5)
        .collect();

    println!("gradient_narrow_probe (frankentorch-68pwz)");
    println!(
        "numel={NUMEL} ({:.1} MiB as f64 -> {:.1} MiB as f32)",
        (NUMEL * 8) as f64 / (1024.0 * 1024.0),
        (NUMEL * 4) as f64 / (1024.0 * 1024.0)
    );
    println!("rayon_threads={}", rayon::current_num_threads());
    println!("pre  loadavg {}", loadavg());
    println!("pre  cpu_mhz {}", cpu_mhz());
    println!();

    let reps = 9;
    let mut serial_best = f64::INFINITY;
    let mut parallel_best = f64::INFINITY;
    let mut serial_last: Vec<f32> = Vec::new();
    let mut parallel_last: Vec<f32> = Vec::new();

    for _ in 0..reps {
        // (a) exactly what ships today, in all four f32 norm backward closures
        let started = Instant::now();
        #[allow(clippy::cast_possible_truncation)]
        let narrowed: Vec<f32> = source.iter().map(|&v| v as f32).collect();
        let elapsed = started.elapsed().as_secs_f64() * 1_000.0;
        assert_eq!(narrowed.len(), NUMEL);
        if elapsed < serial_best {
            serial_best = elapsed;
        }
        serial_last = narrowed;

        // (b) the same map, parallel — one pass, and the workers do the page first-touch.
        let started = Instant::now();
        #[allow(clippy::cast_possible_truncation)]
        let narrowed: Vec<f32> = source.par_iter().map(|&v| v as f32).collect();
        let elapsed = started.elapsed().as_secs_f64() * 1_000.0;
        assert_eq!(narrowed.len(), NUMEL);
        if elapsed < parallel_best {
            parallel_best = elapsed;
        }
        parallel_last = narrowed;
    }

    let mismatches = serial_last
        .iter()
        .zip(parallel_last.iter())
        .filter(|(a, b)| a.to_bits() != b.to_bits())
        .count();
    assert_eq!(mismatches, 0, "narrowing must be bit-identical");

    // The bulk values above are all ordinary; the rounding-interesting cases are here.
    let edges = edge_values();
    #[allow(clippy::cast_possible_truncation)]
    let edge_serial: Vec<f32> = edges.iter().map(|&v| v as f32).collect();
    #[allow(clippy::cast_possible_truncation)]
    let edge_parallel: Vec<f32> = edges.par_iter().map(|&v| v as f32).collect();
    let edge_mismatches = edge_serial
        .iter()
        .zip(edge_parallel.iter())
        .filter(|(a, b)| a.to_bits() != b.to_bits())
        .count();
    assert_eq!(
        edge_mismatches, 0,
        "narrowing must be bit-identical at edges"
    );

    println!("{:>28}  {:>10}", "route", "min ms");
    println!("{:>28}  {serial_best:>10.3}", "serial  .iter().map()");
    println!("{:>28}  {parallel_best:>10.3}", "parallel .par_iter()");
    println!();
    println!(
        "speedup {:.2}x, absolute saving {:.3} ms per backward",
        serial_best / parallel_best,
        serial_best - parallel_best
    );
    println!(
        "bit-identical: {mismatches} bulk mismatches over {NUMEL} elements, \
         {edge_mismatches} over {} edge values",
        edges.len()
    );
    println!();

    // The narrow's gate is measured separately from the widen's (1<<20). Half the write
    // traffic, so there is no reason to assume the same crossover.
    println!("CROSSOVER SWEEP (min of 9; serial/parallel > 1 means parallel wins)");
    println!(
        "{:>10}  {:>10}  {:>10}  {:>16}",
        "numel", "serial ms", "par ms", "serial/parallel"
    );
    for &n in &[
        1_024usize, 4_096, 16_384, 65_536, 262_144, 524_288, 1_048_576, 2_097_152, 4_194_304,
    ] {
        let small = &source[..n];
        let mut s_best = f64::INFINITY;
        let mut p_best = f64::INFINITY;
        for _ in 0..9 {
            let started = Instant::now();
            #[allow(clippy::cast_possible_truncation)]
            let narrowed: Vec<f32> = small.iter().map(|&v| v as f32).collect();
            let elapsed = started.elapsed().as_secs_f64() * 1_000.0;
            std::hint::black_box(&narrowed);
            s_best = s_best.min(elapsed);

            let started = Instant::now();
            #[allow(clippy::cast_possible_truncation)]
            let narrowed: Vec<f32> = small.par_iter().map(|&v| v as f32).collect();
            let elapsed = started.elapsed().as_secs_f64() * 1_000.0;
            std::hint::black_box(&narrowed);
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

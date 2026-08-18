//! Does conv2d's forward still stop scaling at `batch` threads? — `frankentorch-hi9r6`, item 173.
//!
//! **UNBUILT**: written under a build freeze, never compiled or run.
//!
//! THE QUESTION, AND WHOSE IT IS
//!
//! Item 158 replaced the forward's per-channel strided output gather with a per-batch blocked
//! transpose. Item 165c then recorded, against that same change, that it had cut the pass's
//! splittable units from `batch * out_ch` (256) to `batch` (8) — and item 164 established the
//! mechanism that makes that bite: `par_chunks_mut` cannot split BELOW a chunk boundary, so a
//! chunk count bounds rayon's task count from above, whatever the pool width. A peer's item 171
//! then repaired it, keeping item 158's traffic reduction and restoring a finer split, and closed
//! by asking for "a same-invocation 8-vs-64-thread pair on the conv2d forward before any number
//! from this is quoted anywhere". That pair is what this file is.
//!
//! Item 172's probe deliberately does NOT cover this: it times `conv2d_backward_f64`, which
//! contains the two GEMMs item 170 is about and does not contain the forward transpose at all.
//!
//! WHY A SCALING CURVE RATHER THAN A BEFORE/AFTER
//!
//! Item 171 was an edit, not a toggle, so there is no second arm to alternate against inside one
//! binary — and comparing against a pre-171 binary would be a cross-binary comparison, which item
//! 25 recorded cannot attribute a few percent to any one change, and which this campaign has got
//! wrong five times (items 123/135/139, 145, 169).
//!
//! A width sweep needs no second arm. The defect item 165c described has a SHAPE, not just a size:
//! a pass capped at `batch` tasks stops improving once the pool exceeds `batch`, so the curve goes
//! flat after 8 threads no matter how many cores are added. A pass that is genuinely split finer
//! keeps improving. The prediction is about the shape of the curve, so the curve is the
//! measurement, and every point of it is taken in one process on one ELF in one window.
//!
//! `batch = 8` is chosen to sit exactly on the elbow the defect would produce.
//!
//! ARM-INTERNAL: no incumbent, no ratio against PyTorch, no standing. Speedups here are
//! FrankenTorch-versus-FrankenTorch and are MAINTENANCE, not wins.

use std::time::Instant;

// conv2d_masked's shape — the lane behind this bead's certified 5.73x standing.
const BATCH: usize = 8;
const IN_CH: usize = 32;
const OUT_CH: usize = 32;
const H: usize = 32;
const W: usize = 32;
const K: usize = 3;

const PH: usize = H + 2;
const PW: usize = W + 2;
const OH: usize = PH - K + 1;
const OW: usize = PW - K + 1;

const REPS: usize = 9;
const WIDTHS: [usize; 7] = [1, 2, 4, 8, 16, 32, 64];

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
        "min={:.0} median={:.0} max={:.0} spread={:.2}x",
        mhz[0],
        mhz[mhz.len() / 2],
        mhz[mhz.len() - 1],
        mhz[mhz.len() - 1] / mhz[0]
    )
}

fn main() {
    let patch_width = IN_CH * K * K;
    let padded: Vec<f64> = (0..BATCH * IN_CH * PH * PW)
        .map(|i| ((i % 251) as f64) * 0.001 - 0.12)
        .collect();
    let weight: Vec<f64> = (0..OUT_CH * patch_width)
        .map(|i| ((i % 241) as f64) * 0.001 - 0.11)
        .collect();
    // Bias present and carrying a NEGATIVE ZERO: item 158 made the bias add a separate
    // unconditional pass precisely because `-0.0 + 0.0` is `+0.0`, so skipping a zero bias would
    // not be bit-exact. If that pass were ever made conditional, the checksum below moves.
    let bias: Vec<f64> = (0..OUT_CH)
        .map(|i| {
            if i % 3 == 0 {
                -0.0
            } else {
                0.25 - (i as f64) * 0.01
            }
        })
        .collect();

    println!("conv2d_forward_width_probe (frankentorch-hi9r6 item 173)");
    println!("shape [{BATCH},{IN_CH},{H},{W}] k={K} s=1 pad=1 out_ch={OUT_CH}");
    println!(
        "the pass under test: the output transpose, [batch*oh*ow, out_ch] -> [batch, out_ch, oh*ow]"
    );
    println!("batch={BATCH}, so a pass capped at `batch` tasks goes FLAT after 8 threads");
    println!("pre  loadavg {}", loadavg());
    println!("pre  cpu_mhz {}", cpu_mhz());
    println!();
    println!(
        "{:>7}  {:>10}  {:>10}  {:>12}",
        "threads", "min ms", "vs 1 thread", "vs previous"
    );

    let mut baseline = f64::NAN;
    let mut previous = f64::NAN;
    let mut checksums: Vec<(usize, u64)> = Vec::new();

    for width in WIDTHS {
        let pool = match rayon::ThreadPoolBuilder::new().num_threads(width).build() {
            Ok(p) => p,
            Err(e) => {
                println!("{width:>7}  pool build failed: {e}");
                continue;
            }
        };
        let mut best = f64::INFINITY;
        for _ in 0..REPS {
            let started = Instant::now();
            let out = pool.install(|| {
                ft_kernel_cpu::conv2d_forward_f64(
                    &padded,
                    &weight,
                    Some(&bias),
                    BATCH,
                    IN_CH,
                    PH,
                    PW,
                    K,
                    K,
                    OH,
                    OW,
                    1,
                    1,
                    OUT_CH,
                )
            });
            let elapsed = started.elapsed().as_secs_f64() * 1_000.0;
            best = best.min(elapsed);
            // Sum the raw BITS, not the values: a float sum would hide a reordering that a
            // scheduling change must not cause, and the whole premise here is that pool width
            // changes only the schedule.
            let bits = out
                .iter()
                .fold(0u64, |acc, v| acc.wrapping_add(v.to_bits()));
            checksums.push((width, bits));
        }
        if baseline.is_nan() {
            baseline = best;
        }
        let vs_prev = if previous.is_nan() {
            String::from("-")
        } else {
            format!("{:.3}x", previous / best)
        };
        println!(
            "{width:>7}  {best:>10.3}  {:>9.3}x  {vs_prev:>12}",
            baseline / best
        );
        previous = best;
    }

    println!();
    let all_same = checksums.windows(2).all(|w| w[0].1 == w[1].1);
    println!(
        "identical output across every pool width: {}",
        if all_same {
            "yes (bitwise; pool width is a schedule, not a value)"
        } else {
            "*** NO — the forward's output DEPENDS ON POOL WIDTH; that is a BUG and \
             every conv2d row ever taken is suspect ***"
        }
    );
    println!(
        "READING IT: item 165c's defect predicts the `vs previous` column collapses to ~1.00x \
         once threads exceed batch={BATCH}, because the transpose could not be split further. \
         Item 171 claims to have removed that cap. A curve that still flattens at 8 means the cap \
         survives its repair; a curve that keeps improving to 64 means it is gone. Neither \
         outcome is a win over PyTorch -- this is FT-versus-FT and therefore MAINTENANCE."
    );
    println!(
        "CAVEAT: the whole forward is timed, not the transpose alone, so a flat tail could also \
         mean some OTHER phase became the constraint. The GEMM and im2col phases are already \
         parallel over tiles, so they are the first place to look if it flattens."
    );
    println!("post loadavg {}", loadavg());
    println!("post cpu_mhz {}", cpu_mhz());
}

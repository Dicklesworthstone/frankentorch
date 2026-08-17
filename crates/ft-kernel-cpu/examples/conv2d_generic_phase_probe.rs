//! Where does the GENERIC conv2d backward's time go? — `frankentorch-hi9r6`.
//!
//! `conv2d_masked` measures the route a real objective reaches and stands at **7.4-8.8x
//! SLOWER** than PyTorch (item 128), which is the worst standing on the board — worse than
//! conv3d on BOTH routes, and ~4x worse than conv3d on the training one, despite conv3d
//! having received twelve ledger items this campaign while conv2d had no lane at all.
//!
//! WHY A PROBE FIRST, AND NOT A LEVER. This is the sibling of
//! `conv3d_generic_phase_probe`, written for the same reason its bead gives: on the conv3d
//! route, TWO of three attempts to fix the generic backward without a probe were regressions
//! (items 114 and 117, one costing 1.7x), and the one that worked came straight after the
//! probe (item 119). hi9r6 states the rule explicitly — attribute the phases BEFORE touching
//! the kernel — so this probe is the first step, not a preliminary to be skipped.
//!
//! Method is item 82's: subtract the phases you can time directly through PUBLIC kernel API,
//! and attribute the residual rather than guessing at it. Nothing is instrumented in the
//! shipping path, so the probe cannot drift from what ships.
//!
//!   TOTAL     `conv2d_backward_f64` with a NON-UNIFORM `dout` (the generic route; an
//!             all-ones `dout` takes the 3x3 fast path, which is a different kernel with its
//!             own ledger items — items 105 and 107 — and would measure the wrong thing)
//!   im2col    `conv2d_im2col_f64` — the panel the dweight GEMM consumes
//!   col2im    `conv2d_col2im_f64` — the scatter that turns dpanel into dpadded
//!   residual  TOTAL - im2col - col2im, which is the two GEMMs plus the dout_flat gather
//!
//! THE HYPOTHESIS THIS TESTS, stated so the numbers can refute it: hi9r6 suspects conv2d's
//! scaffolding-to-arithmetic ratio is far worse than conv3d's — an 18.9 MB panel against a
//! ~3 ms incumbent — and that this explains 7-8x rather than conv3d's 1.7x. If im2col+col2im
//! dominate, that is confirmed and the lever is the scaffolding. If the RESIDUAL dominates,
//! the hypothesis is wrong and the GEMMs are the target instead.
//!
//! Arm-internal: no incumbent, no ratio, no drift gate, so it stays honest on a busy host.
//! These numbers say which phase to attack; they are not a standing and must never be quoted
//! as one.

use std::time::Instant;

// hi9r6's measured shape, copied verbatim so the probe and the lane describe one workload.
const BATCH: usize = 8;
const IN_CH: usize = 32;
const OUT_CH: usize = 32;
const H: usize = 32;
const W: usize = 32;
const K: usize = 3;

// pad=1, stride=1
const PH: usize = H + 2;
const PW: usize = W + 2;
const OH: usize = PH - K + 1;
const OW: usize = PW - K + 1;

const REPS: usize = 9;

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
    let patch_count = OH * OW;
    let flat = BATCH * patch_count;

    let padded: Vec<f64> = (0..BATCH * IN_CH * PH * PW)
        .map(|i| ((i % 251) as f64) * 0.001 - 0.12)
        .collect();
    let weight: Vec<f64> = (0..OUT_CH * patch_width)
        .map(|i| ((i % 241) as f64) * 0.001 - 0.11)
        .collect();
    // NON-UNIFORM on purpose: `conv2d_backward_f64` branches on `dout` being exactly all
    // +1.0, and that branch is the height-1 / 3x3 fast path with its own ledger items. This
    // probe is about the route `conv2d_masked` exercises.
    let dout: Vec<f64> = (0..BATCH * OUT_CH * patch_count)
        .map(|i| ((i % 197) as f64) * 0.0007 + 0.25)
        .collect();
    assert!(
        dout.iter().any(|v| v.to_bits() != 1.0f64.to_bits()),
        "dout must be non-uniform or this probes the fast path"
    );
    let dpanel: Vec<f64> = (0..flat * patch_width)
        .map(|i| ((i % 173) as f64) * 0.0011 - 0.09)
        .collect();

    println!("conv2d_generic_phase_probe (frankentorch-hi9r6)");
    println!("shape [{BATCH},{IN_CH},{H},{W}] k={K} s=1 pad=1 out_ch={OUT_CH}");
    println!(
        "flat={flat} patch_width={patch_width} dpanel={:.1} MiB",
        (flat * patch_width * 8) as f64 / (1024.0 * 1024.0)
    );
    println!("rayon_threads={}", rayon::current_num_threads());
    println!("pre  loadavg {}", loadavg());
    println!("pre  cpu_mhz {}", cpu_mhz());
    println!();

    let mut total = f64::INFINITY;
    let mut im2col = f64::INFINITY;
    let mut col2im = f64::INFINITY;

    for _ in 0..REPS {
        let started = Instant::now();
        let (dpadded, dweight, _) = ft_kernel_cpu::conv2d_backward_f64(
            &dout, &padded, &weight, BATCH, IN_CH, PH, PW, K, K, OH, OW, 1, 1, OUT_CH, false,
        );
        total = total.min(started.elapsed().as_secs_f64() * 1_000.0);
        std::hint::black_box((&dpadded, &dweight));

        let started = Instant::now();
        let panel =
            ft_kernel_cpu::conv2d_im2col_f64(&padded, BATCH, IN_CH, PH, PW, K, K, OH, OW, 1, 1);
        im2col = im2col.min(started.elapsed().as_secs_f64() * 1_000.0);
        std::hint::black_box(&panel);

        let started = Instant::now();
        let scattered =
            ft_kernel_cpu::conv2d_col2im_f64(&dpanel, BATCH, IN_CH, PH, PW, K, K, OH, OW, 1, 1);
        col2im = col2im.min(started.elapsed().as_secs_f64() * 1_000.0);
        std::hint::black_box(&scattered);
    }

    let residual = total - im2col - col2im;
    println!("PHASES (min of {REPS}), ms:");
    println!("{:>44}  {:>9}  {:>7}", "phase", "min ms", "share");
    println!(
        "{:>44}  {total:>9.3}  {:>6.1}%",
        "TOTAL conv2d_backward_f64 (non-uniform)", 100.0
    );
    println!(
        "{:>44}  {im2col:>9.3}  {:>6.1}%",
        "im2col (dweight's panel)",
        100.0 * im2col / total
    );
    println!(
        "{:>44}  {col2im:>9.3}  {:>6.1}%",
        "col2im (dpanel -> dpadded scatter)",
        100.0 * col2im / total
    );
    println!(
        "{:>44}  {residual:>9.3}  {:>6.1}%   <- the two GEMMs + dout_flat",
        "RESIDUAL (total - im2col - col2im)",
        100.0 * residual / total
    );
    // SPLITTING THE RESIDUAL. `mod gemm` is private, so the two GEMMs cannot be called from
    // an example and the residual cannot be timed directly. The `dout_flat` gather CAN be
    // reproduced faithfully — it is a plain transpose-gather with no arithmetic — so timing a
    // copy of it bounds the third component and leaves the GEMMs by subtraction.
    //
    // This is a REIMPLEMENTATION, not the shipping call, and is labelled as such: it shares
    // the shipping loop's shape and parallelism but is not the same code, so read it as a
    // bound on the gather rather than a measurement of it.
    let mut gather = f64::INFINITY;
    for _ in 0..REPS {
        let started = Instant::now();
        let mut dout_flat = vec![0.0f64; flat * OUT_CH];
        {
            use rayon::prelude::*;
            dout_flat
                .par_chunks_mut(OUT_CH)
                .enumerate()
                .for_each(|(row, dr)| {
                    let n = row / patch_count;
                    let p = row % patch_count;
                    for (oc, d) in dr.iter_mut().enumerate() {
                        *d = dout[(n * OUT_CH + oc) * patch_count + p];
                    }
                });
        }
        gather = gather.min(started.elapsed().as_secs_f64() * 1_000.0);
        std::hint::black_box(&dout_flat);
    }
    let gemms = residual - gather;
    println!(
        "{:>44}  {gather:>9.3}  {:>6.1}%   (reimplemented, a BOUND not the shipping call)",
        "  of which dout_flat gather",
        100.0 * gather / total
    );
    println!(
        "{:>44}  {gemms:>9.3}  {:>6.1}%   <- the two GEMMs alone",
        "  of which the two GEMMs",
        100.0 * gemms / total
    );
    println!();
    println!(
        "GEMM SHAPES, which is what the number above is about:\n  \
         dweight  dgemm_tb(m={OUT_CH}, k={flat}, n={patch_width})   m is THIN\n  \
         dpanel   dgemm   (m={flat}, k={OUT_CH}, n={patch_width})   k is THIN"
    );
    println!(
        "  ~{:.0} MFLOP total across both; at {gemms:.3} ms that is ~{:.0} GFLOP/s",
        2.0 * 2.0 * (OUT_CH * flat * patch_width) as f64 / 1e6,
        2.0 * 2.0 * (OUT_CH * flat * patch_width) as f64 / (gemms * 1e6)
    );
    println!();
    println!(
        "SCAFFOLDING (im2col + col2im) = {:.3} ms, {:.1}% of the backward",
        im2col + col2im,
        100.0 * (im2col + col2im) / total
    );
    println!(
        "READING IT: hi9r6 predicts scaffolding DOMINATES here, worse than conv3d's 46%. If \
         the RESIDUAL dominates instead, that prediction is refuted and the GEMMs are the \
         target."
    );
    println!();
    println!("post loadavg {}", loadavg());
    println!("post cpu_mhz {}", cpu_mhz());
}

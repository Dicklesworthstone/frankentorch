//! Does the conv3d im2col panel's zero-fill actually cost anything? — `frankentorch-l2zki`.
//!
//! `conv3d_im2col_f64` allocates `vec![0.0f64; batch*patch_count*patch_width]` and then
//! overwrites EVERY element from a parallel `copy_from_slice` loop. For the scored lane
//! that is 4096 x 864 = 3,538,944 f64 = 28.3 MiB of zeroing in front of a parallel
//! writer, and it sits in BOTH conv3d backward paths (the ones-dout fast path the lane
//! takes, and the generic one).
//!
//! IT DOES NOT FOLLOW THAT REMOVING IT HELPS. At this size `vec![0.0; n]` goes through
//! `alloc_zeroed`, and a fresh mmap hands back lazily-mapped zero pages — no eager
//! memset at all, in which case an uninit allocation saves nothing and the parallel
//! writer pays the same page faults either way. It only costs real time when the
//! allocator RECYCLES a dirty block and has to memset it, which is what happens when the
//! function is called repeatedly, as the harness does.
//!
//! So this is measured, not assumed, and measured in the recycling regime (a loop) rather
//! than once from cold. Item 69's widen was a serial WRITE of every element and was worth
//! 20 ms; a zero-fill is a different animal and gets its own number.
//!
//! Arm-internal: no incumbent, no ratio, no drift gate.

use rayon::prelude::*;
use std::time::Instant;

const BATCH: usize = 2;
const IN_CH: usize = 32;
const SPATIAL_D: usize = 8;
const SPATIAL_H: usize = 16;
const SPATIAL_W: usize = 16;
const K: usize = 3;

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

/// The fill body, byte-identical between the two allocation strategies so the only
/// variable is how the buffer was obtained.
#[allow(clippy::too_many_arguments)]
fn fill(panel: &mut [f64], padded: &[f64], patch_width: usize, patch_count: usize,
        in_ch: usize, pd: usize, ph: usize, pw: usize, kd: usize, kh: usize, kw: usize,
        oh: usize, ow: usize, sd: usize, sh: usize, sw: usize) {
    panel
        .par_chunks_mut(patch_width)
        .enumerate()
        .for_each(|(row, prow)| {
            let b = row / patch_count;
            let pc = row % patch_count;
            let base_d = (pc / (oh * ow)) * sd;
            let rem = pc % (oh * ow);
            let base_h = (rem / ow) * sh;
            let base_w = (rem % ow) * sw;
            let batch_off = b * in_ch * pd * ph * pw;
            for c in 0..in_ch {
                let ch_off = batch_off + c * pd * ph * pw;
                let pch = c * kd * kh * kw;
                for kdd in 0..kd {
                    let d_off = ch_off + (base_d + kdd) * ph * pw;
                    let pkd = pch + kdd * kh * kw;
                    for kr in 0..kh {
                        let irow = d_off + (base_h + kr) * pw + base_w;
                        let prow_off = pkd + kr * kw;
                        prow[prow_off..(kw + prow_off)]
                            .copy_from_slice(&padded[irow..(kw + irow)]);
                    }
                }
            }
        });
}

fn main() {
    let (pd, ph, pw) = (SPATIAL_D + 2, SPATIAL_H + 2, SPATIAL_W + 2);
    let (od, oh, ow) = (pd - K + 1, ph - K + 1, pw - K + 1);
    let padded: Vec<f64> = (0..BATCH * IN_CH * pd * ph * pw)
        .map(|index| ((index % 251) as f64) * 0.001 - 0.12)
        .collect();
    let patch_width = IN_CH * K * K * K;
    let patch_count = od * oh * ow;
    let numel = BATCH * patch_count * patch_width;

    println!("im2col_zerofill_probe (frankentorch-l2zki)");
    println!("panel {} x {} = {numel} f64 = {:.1} MiB", BATCH * patch_count, patch_width,
        (numel * 8) as f64 / (1024.0 * 1024.0));
    println!("rayon_threads={}", rayon::current_num_threads());
    println!("pre  loadavg {}", loadavg());
    println!("pre  cpu_mhz {}", cpu_mhz());
    println!();

    let reps = 15;
    let mut zeroed_best = f64::INFINITY;
    let mut uninit_best = f64::INFINITY;
    let mut zeroed_last: Vec<f64> = Vec::new();
    let mut uninit_last: Vec<f64> = Vec::new();

    for _ in 0..reps {
        // (a) exactly what ships today
        let started = Instant::now();
        let mut panel = vec![0.0f64; numel];
        fill(&mut panel, &padded, patch_width, patch_count, IN_CH, pd, ph, pw, K, K, K,
             oh, ow, 1, 1, 1);
        let elapsed = started.elapsed().as_secs_f64() * 1_000.0;
        zeroed_best = zeroed_best.min(elapsed);
        zeroed_last = panel;

        // (b) same fill, uninitialized allocation
        let started = Instant::now();
        let panel = ft_kernel_cpu::build_uninit(numel, |panel: &mut [f64]| {
            fill(panel, &padded, patch_width, patch_count, IN_CH, pd, ph, pw, K, K, K,
                 oh, ow, 1, 1, 1);
        });
        let elapsed = started.elapsed().as_secs_f64() * 1_000.0;
        uninit_best = uninit_best.min(elapsed);
        uninit_last = panel;
    }

    // The fill writes every element, so the two must agree bit for bit. If they do not,
    // some element is unwritten and the uninit route is unsound for this fill.
    let mismatches = zeroed_last
        .iter()
        .zip(uninit_last.iter())
        .filter(|(a, b)| a.to_bits() != b.to_bits())
        .count();
    assert_eq!(mismatches, 0, "uninit panel diverged: some element is never written");

    println!("{:>30}  {:>10}", "allocation", "min ms");
    println!("{:>30}  {zeroed_best:>10.3}", "vec![0.0; n] + fill");
    println!("{:>30}  {uninit_best:>10.3}", "build_uninit + fill");
    println!();
    println!(
        "speedup {:.2}x, saving {:.3} ms per im2col   (bit-identical, {mismatches} mismatches)",
        zeroed_best / uninit_best,
        zeroed_best - uninit_best
    );
    println!(
        "CONTEXT: the lane's backward measured 10.290 ms total (item 70), and this panel \
         is built once per backward on BOTH conv3d backward paths."
    );
    println!("post loadavg {}", loadavg());
    println!("post cpu_mhz {}", cpu_mhz());
}

//! Where does the conv3d lane's time actually go? — `frankentorch-l2zki`.
//!
//! The scored h2h `conv3d` lane is f64 WITH GRAD and times forward + loss_sum + backward
//! together. Every measurement taken on it so far (item 68's route probe included) priced
//! only the FORWARD, so the split has never been read. This prices both phases at the
//! lane's exact shape before any lever is chosen — the same discipline that found the
//! GroupNorm widen in item 69, where the twin lanes said the kernels were already fine.
//!
//! The lane's loss is `out.sum()`, so its upstream gradient is all-ones and
//! `conv3d_backward_f64` takes its `conv3d_backward_ones_dout_f64` fast path. That is
//! reproduced here rather than assumed, by passing an all-ones `dout`, and the probe
//! prints both so a future reader can see which branch was priced.
//!
//! Arm-internal: no incumbent, no ratio, no drift gate.

use std::time::Instant;

const BATCH: usize = 2;
const IN_CH: usize = 32;
const OUT_CH: usize = 32;
const SPATIAL_D: usize = 8;
const SPATIAL_H: usize = 16;
const SPATIAL_W: usize = 16;
const K: usize = 3;

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

fn main() {
    let (pd, ph, pw) = (SPATIAL_D + 2, SPATIAL_H + 2, SPATIAL_W + 2);
    let (od, oh, ow) = (pd - K + 1, ph - K + 1, pw - K + 1);
    let padded: Vec<f64> = (0..BATCH * IN_CH * pd * ph * pw)
        .map(|index| ((index % 251) as f64) * 0.001 - 0.12)
        .collect();
    let weight: Vec<f64> = (0..OUT_CH * IN_CH * K * K * K)
        .map(|index| ((index % 241) as f64) * 0.001 - 0.11)
        .collect();
    let out_numel = BATCH * OUT_CH * od * oh * ow;
    // The lane's loss is out.sum(), so the upstream gradient is exactly all-ones.
    let dout_ones = vec![1.0f64; out_numel];
    // A non-ones control, to show which branch the fast path costs.
    let dout_mixed: Vec<f64> = (0..out_numel)
        .map(|index| 1.0 + ((index % 17) as f64) * 1e-9)
        .collect();

    println!("conv3d_phase_probe (frankentorch-l2zki)");
    println!(
        "shape batch={BATCH} in_ch={IN_CH} out_ch={OUT_CH} spatial={SPATIAL_D}x{SPATIAL_H}x{SPATIAL_W} k=3 s=1 pad=1"
    );
    println!("padded numel={}  out numel={out_numel}", padded.len());
    println!("rayon_threads={}", rayon::current_num_threads());
    println!("pre  loadavg {}", loadavg());
    println!("pre  cpu_mhz {}", cpu_mhz());
    println!();

    let reps = 7;
    let mut fwd = f64::INFINITY;
    let mut bwd_ones = f64::INFINITY;
    let mut bwd_mixed = f64::INFINITY;

    for _ in 0..reps {
        let started = Instant::now();
        let out = ft_kernel_cpu::conv3d_forward_f64(
            &padded, &weight, None, BATCH, IN_CH, pd, ph, pw, K, K, K, od, oh, ow, 1, 1, 1, OUT_CH,
        );
        let elapsed = started.elapsed().as_secs_f64() * 1_000.0;
        assert_eq!(out.len(), out_numel);
        std::hint::black_box(&out);
        fwd = fwd.min(elapsed);

        let started = Instant::now();
        let (dp, dw, _) = ft_kernel_cpu::conv3d_backward_f64(
            &dout_ones, &padded, &weight, BATCH, IN_CH, pd, ph, pw, K, K, K, od, oh, ow, 1, 1, 1,
            OUT_CH, false,
        );
        let elapsed = started.elapsed().as_secs_f64() * 1_000.0;
        std::hint::black_box((&dp, &dw));
        bwd_ones = bwd_ones.min(elapsed);

        let started = Instant::now();
        let (dp, dw, _) = ft_kernel_cpu::conv3d_backward_f64(
            &dout_mixed,
            &padded,
            &weight,
            BATCH,
            IN_CH,
            pd,
            ph,
            pw,
            K,
            K,
            K,
            od,
            oh,
            ow,
            1,
            1,
            1,
            OUT_CH,
            false,
        );
        let elapsed = started.elapsed().as_secs_f64() * 1_000.0;
        std::hint::black_box((&dp, &dw));
        bwd_mixed = bwd_mixed.min(elapsed);
    }

    let step = fwd + bwd_ones;
    println!("{:>34}  {:>10}  {:>8}", "phase", "min ms", "share");
    println!(
        "{:>34}  {fwd:>10.3}  {:>7.1}%",
        "forward (direct 3x3s1 route)",
        100.0 * fwd / step
    );
    println!(
        "{:>34}  {bwd_ones:>10.3}  {:>7.1}%",
        "backward (ones-dout fast path)",
        100.0 * bwd_ones / step
    );
    println!("{:>34}  {step:>10.3}", "forward + backward");
    println!();
    println!(
        "{:>34}  {bwd_mixed:>10.3}   (control: the generic backward, not taken by this lane)",
        "backward (non-ones dout)"
    );
    println!(
        "ones-dout fast path is worth {:.2}x against the generic backward",
        bwd_mixed / bwd_ones
    );
    println!();
    println!("post loadavg {}", loadavg());
    println!("post cpu_mhz {}", cpu_mhz());
}

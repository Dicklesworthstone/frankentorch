//! Where does the GENERIC conv3d backward's time go? — `frankentorch-l2zki`.
//!
//! `conv3d_masked` (NEGATIVE_EVIDENCE item 110) measures the route a real objective reaches
//! and stands at **2.6-2.9x SLOWER** than PyTorch, which is this bead's worst standing. Item
//! 117 reverted a lever aimed at it that was guessed rather than aimed, and cost 1.7x. This
//! probe exists so the next one is aimed.
//!
//! Item 82 established the method: subtract phases you can time directly, and attribute the
//! residual rather than guessing at it. Everything here is PUBLIC kernel API, so no
//! instrumentation is added to the shipping path and the probe cannot drift from it.
//!
//!   TOTAL     `conv3d_backward_f64` with a NON-UNIFORM `dout` (the generic route; an
//!             all-ones `dout` would take the fast path and measure the wrong thing)
//!   im2col    `conv3d_im2col_f64` — the panel the dweight GEMM consumes
//!   col2im    `conv3d_col2im_f64` — the scatter that turns dpanel into dpadded
//!   residual  TOTAL - im2col - col2im, which is the two GEMMs plus dout_flat
//!
//! Arm-internal: no incumbent, no ratio, no drift gate, so it is honest on a busy host. The
//! numbers say which phase to attack; they are not a standing and must never be quoted as one.

use std::time::Instant;

const BATCH: usize = 2;
const IN_CH: usize = 32;
const OUT_CH: usize = 32;
const SD: usize = 8;
const SH: usize = 16;
const SW: usize = 16;
const K: usize = 3;

const PD: usize = SD + 2;
const PH: usize = SH + 2;
const PW: usize = SW + 2;
const OD: usize = PD - K + 1;
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
    let patch_width = IN_CH * K * K * K;
    let patch_count = OD * OH * OW;
    let flat = BATCH * patch_count;

    let padded: Vec<f64> = (0..BATCH * IN_CH * PD * PH * PW)
        .map(|i| ((i % 251) as f64) * 0.001 - 0.12)
        .collect();
    let weight: Vec<f64> = (0..OUT_CH * patch_width)
        .map(|i| ((i % 241) as f64) * 0.001 - 0.11)
        .collect();
    // NON-UNIFORM on purpose: `conv3d_backward_f64` branches on `dout` being exactly all
    // +1.0, and the all-ones branch is a different kernel with its own ledger items. This
    // probe is about the route `conv3d_masked` exercises.
    let dout: Vec<f64> = (0..BATCH * OUT_CH * patch_count)
        .map(|i| ((i % 197) as f64) * 0.0007 + 0.25)
        .collect();
    assert!(
        dout.iter().any(|v| v.to_bits() != 1.0f64.to_bits()),
        "dout must be non-uniform or this probes the fast path"
    );
    // A dpanel-shaped buffer for the col2im phase, filled with plausible magnitudes. col2im's
    // cost is its scatter pattern, not its input values.
    let dpanel: Vec<f64> = (0..flat * patch_width)
        .map(|i| ((i % 173) as f64) * 0.0011 - 0.09)
        .collect();

    println!("conv3d_generic_phase_probe (frankentorch-l2zki)");
    println!("shape [{BATCH},{IN_CH},{SD},{SH},{SW}] k=3 s=1 pad=1 out_ch={OUT_CH}");
    println!(
        "flat={flat} patch_width={patch_width} dpanel={} MiB",
        (flat * patch_width * 8) >> 20
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
        let (dpadded, dweight, _) = ft_kernel_cpu::conv3d_backward_f64(
            &dout, &padded, &weight, BATCH, IN_CH, PD, PH, PW, K, K, K, OD, OH, OW, 1, 1, 1,
            OUT_CH, false,
        );
        total = total.min(started.elapsed().as_secs_f64() * 1_000.0);
        std::hint::black_box((&dpadded, &dweight));

        let started = Instant::now();
        let panel = ft_kernel_cpu::conv3d_im2col_f64(
            &padded, BATCH, IN_CH, PD, PH, PW, K, K, K, OD, OH, OW, 1, 1, 1,
        );
        im2col = im2col.min(started.elapsed().as_secs_f64() * 1_000.0);
        std::hint::black_box(&panel);

        let started = Instant::now();
        let scattered = ft_kernel_cpu::conv3d_col2im_f64(
            &dpanel, BATCH, IN_CH, PD, PH, PW, K, K, K, OD, OH, OW, 1, 1, 1,
        );
        col2im = col2im.min(started.elapsed().as_secs_f64() * 1_000.0);
        std::hint::black_box(&scattered);
    }

    let residual = total - im2col - col2im;
    println!("PHASES (min of {REPS}), ms:");
    println!("{:>44}  {:>9}  {:>7}", "phase", "min ms", "share");
    println!(
        "{:>44}  {total:>9.3}  {:>6.1}%",
        "TOTAL conv3d_backward_f64 (non-uniform)", 100.0
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
    println!();
    println!(
        "CONTEXT: conv3d_masked stands at 2.6-2.9x SLOWER (item 110/117), FT ~19 ms against \
         PyTorch ~7 ms. This probe times the KERNEL only; the lane also carries tape and \
         session cost, so these phases do not sum to the lane."
    );
    println!("post loadavg {}", loadavg());
    println!("post cpu_mhz {}", cpu_mhz());
}

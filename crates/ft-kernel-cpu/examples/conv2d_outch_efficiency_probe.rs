//! Is conv2d's GEMM inefficiency really the THIN `k`? Sweep `out_ch` and watch GFLOP/s —
//! `frankentorch-hi9r6`.
//!
//! WHERE THIS SITS. Items 130/131 measured conv2d's generic backward at 79-82% GEMM. Item 132
//! read the gate and blamed a threshold; item 134 MEASURED that and refuted it — relaxing the
//! gate bought 1.041x, less than its own control's noise, because three uneven column blocks
//! each re-pack the same A panel. What item 134 left standing was the shape itself, and one
//! candidate in particular that nobody has priced:
//!
//!     dpanel   dgemm(m=flat, k=out_ch, n=patch_width)
//!
//! `k` here IS `out_ch`. At the scored shape that is **32** — an eighth of one `DGEMM_KC`
//! (256) block — so the GEMM pays a full pack for an inner reduction too short to amortise it.
//!
//! HOW THIS TESTS IT WITHOUT GUESSING. `out_ch` is the only parameter that moves that `k`, and
//! it is reachable through the PUBLIC `conv2d_backward_f64`. Sweeping it and normalising by
//! FLOPs turns the question into one number per point: if the backward is SHAPE-bound, GFLOP/s
//! climbs steeply with `out_ch` and then flattens once `k` reaches a block; if it is bandwidth-
//! or scaffolding-bound, GFLOP/s stays roughly flat and the thin-`k` story is refuted like the
//! gate story was.
//!
//! `im2col` and `col2im` do NOT depend on `out_ch` — they are functions of the input geometry
//! and `patch_width` only — so they are subtracted at every point and double as a control: if
//! they drift across the sweep, the host moved and the sweep is not readable.
//!
//! `out_ch` also sets the dweight GEMM's `m`, so a rise cannot be attributed to `k` ALONE. This
//! probe answers "is it the shape" and deliberately not "which of the two shapes"; separating
//! them needs the two GEMMs timed apart, which `mod gemm` being private currently prevents.
//!
//! Arm-internal: no incumbent, no ratio, no gate. Not a standing.

use std::time::Instant;

const BATCH: usize = 8;
const IN_CH: usize = 32;
const H: usize = 32;
const W: usize = 32;
const K: usize = 3;

const PH: usize = H + 2;
const PW: usize = W + 2;
const OH: usize = PH - K + 1;
const OW: usize = PW - K + 1;

const REPS: usize = 7;
const DGEMM_KC: usize = 256; // matrixmultiply's f64 k-block, per item 97

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
    let dpanel: Vec<f64> = (0..flat * patch_width)
        .map(|i| ((i % 173) as f64) * 0.0011 - 0.09)
        .collect();

    println!("conv2d_outch_efficiency_probe (frankentorch-hi9r6)");
    println!(
        "fixed [{BATCH},{IN_CH},{H},{W}] k={K} s=1 pad=1  ->  flat={flat} patch_width={patch_width}"
    );
    println!("sweeping out_ch, which IS the dpanel GEMM's k (and the dweight GEMM's m)");
    println!("rayon_threads={}", rayon::current_num_threads());
    println!("pre  loadavg {}", loadavg());
    println!("pre  cpu_mhz {}", cpu_mhz());
    println!();
    println!(
        "{:>7}  {:>9}  {:>9}  {:>9}  {:>9}  {:>10}  {:>6}",
        "out_ch", "total ms", "im2col", "col2im", "gemm ms", "GFLOP/s", "k/KC"
    );

    // ASCENDING then DESCENDING in one process. The sweep order correlates with time, so a
    // host that drifts monotonically would manufacture a monotonic trend. Running it both ways
    // breaks that: a real shape effect appears in BOTH directions, a drift artefact reverses.
    let mut order: Vec<usize> = vec![8, 16, 32, 64, 128, 256];
    order.extend(vec![256usize, 128, 64, 32, 16, 8]);
    for &out_ch in &order {
        let weight: Vec<f64> = (0..out_ch * patch_width)
            .map(|i| ((i % 241) as f64) * 0.001 - 0.11)
            .collect();
        let dout: Vec<f64> = (0..BATCH * out_ch * patch_count)
            .map(|i| ((i % 197) as f64) * 0.0007 + 0.25)
            .collect();

        let mut total = f64::INFINITY;
        let mut im2col = f64::INFINITY;
        let mut col2im = f64::INFINITY;
        for _ in 0..REPS {
            let t = Instant::now();
            let (dpadded, dweight, _) = ft_kernel_cpu::conv2d_backward_f64(
                &dout, &padded, &weight, BATCH, IN_CH, PH, PW, K, K, OH, OW, 1, 1, out_ch, false,
            );
            total = total.min(t.elapsed().as_secs_f64() * 1e3);
            std::hint::black_box((&dpadded, &dweight));

            let t = Instant::now();
            let panel =
                ft_kernel_cpu::conv2d_im2col_f64(&padded, BATCH, IN_CH, PH, PW, K, K, OH, OW, 1, 1);
            im2col = im2col.min(t.elapsed().as_secs_f64() * 1e3);
            std::hint::black_box(&panel);

            let t = Instant::now();
            let scattered =
                ft_kernel_cpu::conv2d_col2im_f64(&dpanel, BATCH, IN_CH, PH, PW, K, K, OH, OW, 1, 1);
            col2im = col2im.min(t.elapsed().as_secs_f64() * 1e3);
            std::hint::black_box(&scattered);
        }

        let gemm = total - im2col - col2im;
        // Two GEMMs, each out_ch*flat*patch_width MACs, 2 flops per MAC.
        let gflops = 2.0 * 2.0 * (out_ch * flat * patch_width) as f64 / (gemm * 1e6);
        println!(
            "{out_ch:>7}  {total:>9.3}  {im2col:>9.3}  {col2im:>9.3}  {gemm:>9.3}  {gflops:>10.1}  {:>6.2}",
            out_ch as f64 / DGEMM_KC as f64
        );
    }

    println!();
    println!(
        "READING IT: im2col and col2im are INDEPENDENT of out_ch — if they drift across the \
         sweep the host moved and this table is not readable. If GFLOP/s climbs steeply and \
         then flattens near k/KC = 1, the backward is SHAPE-bound and the thin k is the cause. \
         If it stays flat, that story is refuted the way item 134 refuted the gate story."
    );
    println!("post loadavg {}", loadavg());
    println!("post cpu_mhz {}", cpu_mhz());
}

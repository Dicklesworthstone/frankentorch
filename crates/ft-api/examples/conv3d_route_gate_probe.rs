//! Is "go direct" actually the lever for conv3d's 4.06x standing? — `frankentorch-l2zki`.
//!
//! WHY THIS EXISTS. l2zki proposes a DIRECT f32 Conv3d kernel and is sized by the h2h
//! conv3d lane's 4.06x. NEGATIVE_EVIDENCE item 67b established that lane is f64 WITH
//! GRAD, so it cannot reach `conv3d_forward_f32` at all. What it DOES reach is
//! `conv3d_forward_f64`, which already carries a direct 3x3x3 stride-1 kernel gated on
//!
//!     kd==kh==kw==3 && sd==sh==sw==1 && in_ch>=8 && out_ch>=8 && out_ch % 4 == 0
//!
//! Item 67b recorded, as a LEAD and not a finding, that the lane's shape satisfies every
//! one of those. If it does, then the 4.06x is measured WITH a direct kernel already
//! running, and porting "direct" to f32 is not the lever for it — which would settle the
//! bead without needing the tolerance ratification at all.
//!
//! HOW THIS PROBES IT WITHOUT A SOURCE CHANGE. `out_ch % 4 == 0` is a DISCONTINUITY in
//! the dispatch that is reachable from the public API: out_ch 28/32/36 take the direct
//! kernel, 30/34 take the streamed im2col-GEMM. Everything else is held fixed. Cost is
//! reported per output channel so the smooth size trend divides out, and any systematic
//! step between the divisible-by-4 group and the rest is the ROUTE, not the shape.
//!
//! This is deliberately NOT a poison sentinel — a sentinel needs an edit to
//! `ft-kernel-cpu/src/lib.rs`, which currently carries another agent's uncommitted work.
//! The gate is observable from outside, so it is observed from outside.
//!
//! Run (local; no incumbent involved, this is arm-internal):
//! ```text
//! cargo run --release -p frankentorch-api --features fair-alloc --example conv3d_route_gate_probe
//! ```

use std::time::Instant;

/// Shape of the h2h `conv3d` lane: input [2,32,8,16,16], weight [32,32,3,3,3], pad 1.
const BATCH: usize = 2;
const IN_CH: usize = 32;
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

/// Per-core clocks, as the standing rule now requires on every banked row.
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
    let min = mhz[0];
    let max = mhz[mhz.len() - 1];
    let mean = mhz.iter().sum::<f64>() / mhz.len() as f64;
    format!(
        "min={min:.0} mean={mean:.0} max={max:.0} spread={:.2}x",
        max / min
    )
}

fn main() {
    let (pd, ph, pw) = (SPATIAL_D + 2, SPATIAL_H + 2, SPATIAL_W + 2);
    let (od, oh, ow) = (pd - K + 1, ph - K + 1, pw - K + 1);
    let padded: Vec<f64> = (0..BATCH * IN_CH * pd * ph * pw)
        .map(|index| ((index % 251) as f64) * 0.001 - 0.12)
        .collect();

    println!("conv3d_route_gate_probe (frankentorch-l2zki)");
    println!(
        "host={}",
        std::env::var("HOSTNAME").unwrap_or_else(|_| "thinkstation1".into())
    );
    println!("rayon_threads={}", rayon::current_num_threads());
    println!(
        "shape batch={BATCH} in_ch={IN_CH} spatial={SPATIAL_D}x{SPATIAL_H}x{SPATIAL_W} k=3 stride=1 pad=1"
    );
    println!("out={od}x{oh}x{ow}  patch_width={}", IN_CH * K * K * K);
    println!("pre  loadavg {}", loadavg());
    println!("pre  cpu_mhz {}", cpu_mhz());
    println!();
    println!("gate: out_ch % 4 == 0 -> DIRECT 3x3x3 kernel; otherwise STREAMED im2col+GEMM");
    println!();
    println!(
        "{:>7}  {:>7}  {:>10}  {:>14}  {:>10}",
        "out_ch", "route", "min ms", "ms/out_ch", "mhz spread"
    );

    let mut rows: Vec<(usize, bool, f64, f64)> = Vec::new();
    for &out_ch in &[28usize, 30, 32, 34, 36] {
        let weight: Vec<f64> = (0..out_ch * IN_CH * K * K * K)
            .map(|index| ((index % 241) as f64) * 0.001 - 0.11)
            .collect();
        let direct = out_ch % 4 == 0;
        // min-of-N: the estimator this ledger quotes, and the one least disturbed by a
        // neighbouring process stealing a core mid-sample.
        let mut best = f64::INFINITY;
        let mut checksum = 0.0f64;
        for _ in 0..7 {
            let started = Instant::now();
            let out = ft_kernel_cpu::conv3d_forward_f64(
                &padded, &weight, None, BATCH, IN_CH, pd, ph, pw, K, K, K, od, oh, ow, 1, 1, 1,
                out_ch,
            );
            let elapsed = started.elapsed().as_secs_f64() * 1_000.0;
            checksum = out.iter().sum::<f64>();
            if elapsed < best {
                best = elapsed;
            }
        }
        assert!(checksum.is_finite(), "conv3d output must be finite");
        let per_channel = best / out_ch as f64;
        println!(
            "{out_ch:>7}  {:>7}  {best:>10.3}  {per_channel:>14.4}  {:>10}",
            if direct { "DIRECT" } else { "stream" },
            cpu_mhz().split("spread=").nth(1).unwrap_or("?").to_owned()
        );
        rows.push((out_ch, direct, best, per_channel));
    }

    println!();
    let direct_cost: Vec<f64> = rows.iter().filter(|r| r.1).map(|r| r.3).collect();
    let stream_cost: Vec<f64> = rows.iter().filter(|r| !r.1).map(|r| r.3).collect();
    let mean = |values: &[f64]| values.iter().sum::<f64>() / values.len() as f64;
    let direct_mean = mean(&direct_cost);
    let stream_mean = mean(&stream_cost);
    println!("mean ms/out_ch  DIRECT {direct_mean:.4}   streamed {stream_mean:.4}");
    println!(
        "streamed / direct = {:.3}x  (>1 means the direct kernel is the faster route)",
        stream_mean / direct_mean
    );
    println!();
    in_channel_crossover(pd, ph, pw, od, oh, ow);
    println!();
    cross_route_bit_exactness(&padded, pd, ph, pw, od, oh, ow);
    println!();
    println!("post loadavg {}", loadavg());
    println!("post cpu_mhz {}", cpu_mhz());
}

/// WHERE SHOULD THE GATE SIT? The direct kernel presumably won when it landed, and its
/// proof test uses in_ch=8. The gate admits everything from in_ch>=8 upward with no
/// ceiling, so if the routes cross over somewhere above 8 the gate is simply mis-set.
/// Sweeps in_ch at fixed out_ch, comparing 32 (direct) against 30 (streamed) and
/// normalising per output channel.
fn in_channel_crossover(pd: usize, ph: usize, pw: usize, od: usize, oh: usize, ow: usize) {
    println!("IN_CH CROSSOVER (ms per out_ch; direct=out_ch 32, streamed=out_ch 30)");
    println!(
        "{:>7}  {:>7}  {:>10}  {:>10}  {:>16}",
        "in_ch", "k", "direct", "stream", "stream/direct"
    );
    for &in_ch in &[8usize, 12, 16, 24, 32, 48] {
        let padded: Vec<f64> = (0..BATCH * in_ch * pd * ph * pw)
            .map(|index| ((index % 251) as f64) * 0.001 - 0.12)
            .collect();
        let mut cost = [0.0f64; 2];
        for (slot, &out_ch) in [32usize, 30].iter().enumerate() {
            let weight: Vec<f64> = (0..out_ch * in_ch * K * K * K)
                .map(|index| ((index % 241) as f64) * 0.001 - 0.11)
                .collect();
            let mut best = f64::INFINITY;
            for _ in 0..5 {
                let started = Instant::now();
                let out = ft_kernel_cpu::conv3d_forward_f64(
                    &padded, &weight, None, BATCH, in_ch, pd, ph, pw, K, K, K, od, oh, ow, 1, 1, 1,
                    out_ch,
                );
                let elapsed = started.elapsed().as_secs_f64() * 1_000.0;
                assert!(out.iter().sum::<f64>().is_finite());
                if elapsed < best {
                    best = elapsed;
                }
            }
            cost[slot] = best / out_ch as f64;
        }
        println!(
            "{in_ch:>7}  {:>7}  {:>10.4}  {:>10.4}  {:>15.3}x",
            in_ch * K * K * K,
            cost[0],
            cost[1],
            cost[1] / cost[0]
        );
    }
    println!("  (stream/direct < 1 means the DIRECT kernel is winning at that in_ch)");
}

/// Would routing this shape AWAY from the direct kernel be bit-exact?
///
/// `conv3d_direct_3x3s1_matches_streamed_reference_bits` proves direct == streamed in
/// f64 at in_ch=8, i.e. k=216. The lane runs in_ch=32, k=864, and the f32 work in item
/// 67a showed a blocked GEMM need not agree with a sequential accumulation as k grows.
/// So the f64 agreement must be checked AT THE OPERATIVE k before the route swap can be
/// called free.
///
/// Both routes are reachable through the public API by exploiting the same gate: build
/// one weight of 30 output channels (streamed) and a 32-channel weight whose first 30
/// channels are identical (direct), then compare those 30 channels' outputs. Per-channel
/// arithmetic is identical by construction, so any difference is the route.
fn cross_route_bit_exactness(
    padded: &[f64],
    pd: usize,
    ph: usize,
    pw: usize,
    od: usize,
    oh: usize,
    ow: usize,
) {
    const STREAMED_CH: usize = 30; // 30 % 4 != 0 -> streamed
    const DIRECT_CH: usize = 32; //  32 % 4 == 0 -> direct
    let per_channel = IN_CH * K * K * K;
    let shared: Vec<f64> = (0..STREAMED_CH * per_channel)
        .map(|index| ((index % 241) as f64) * 0.001 - 0.11)
        .collect();
    let mut padded_weight = shared.clone();
    padded_weight.extend(
        (0..(DIRECT_CH - STREAMED_CH) * per_channel)
            .map(|index| ((index % 197) as f64) * 0.002 - 0.09),
    );

    let streamed = ft_kernel_cpu::conv3d_forward_f64(
        padded,
        &shared,
        None,
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
        STREAMED_CH,
    );
    let direct = ft_kernel_cpu::conv3d_forward_f64(
        padded,
        &padded_weight,
        None,
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
        DIRECT_CH,
    );

    let patch_count = od * oh * ow;
    let mut mismatches = 0usize;
    let mut worst_bits = 0u64;
    let mut worst_relative = 0.0f64;
    for n in 0..BATCH {
        for oc in 0..STREAMED_CH {
            for p in 0..patch_count {
                let s = streamed[(n * STREAMED_CH + oc) * patch_count + p];
                let d = direct[(n * DIRECT_CH + oc) * patch_count + p];
                if s.to_bits() != d.to_bits() {
                    mismatches += 1;
                    worst_bits = worst_bits.max(s.to_bits().abs_diff(d.to_bits()));
                    let denom = s.abs().max(f64::MIN_POSITIVE);
                    worst_relative = worst_relative.max((s - d).abs() / denom);
                }
            }
        }
    }
    let compared = BATCH * STREAMED_CH * patch_count;
    println!(
        "CROSS-ROUTE BIT-EXACTNESS at the operative k={} (in_ch={IN_CH})",
        per_channel
    );
    println!(
        "  compared {compared} outputs: {mismatches} differ; worst bit delta {worst_bits}, \
         worst relative {worst_relative:.3e}"
    );
    if mismatches == 0 {
        println!(
            "  => the two routes agree BITWISE at k={per_channel}, so re-gating this shape \
             away from the direct kernel is a BIT-EXACT change and needs no tolerance call."
        );
    } else {
        println!(
            "  => the routes DIFFER, so a re-gate is a tolerance change like item 67a and \
             must go the same ratification path."
        );
    }
}

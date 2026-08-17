//! Which conv3d route is CLOSER TO TORCH? — `frankentorch-conv3d-direct-gate-misset-w3pol`.
//!
//! WHY THIS EXISTS. Item 68d left a well-posed but unanswered question: re-gating conv3d
//! so `in_ch` above ~8 takes the streamed im2col+GEMM route is worth a measured 1.5-3.3x,
//! but it changes the produced bits by 4.770e-12 relative against the direct kernel that
//! ships today. That was recorded as needing a "tolerance ratification" — a policy call.
//!
//! IT IS NOT A POLICY CALL. It was only a policy call because item 68d compared the two
//! routes AGAINST EACH OTHER, which can say they disagree but cannot say which one is
//! right. Neither route is the reference. **Torch is the reference**, and this campaign's
//! parity rule is parity with torch, not self-consistency between two of our own kernels.
//!
//! So the real question is not "may we accept 4.770e-12 of drift" but:
//!
//!     Of the two routes, which one lands closer to what torch actually computes?
//!
//! If the streamed route is at least as close, the re-gate is not a parity regression to
//! be ratified — it is free, or an improvement, and the 1.5-3.3x is simply available.
//! Torch's own CPU conv3d is an im2col+GEMM, so there is a specific reason to expect the
//! streamed route to match it BETTER than a scalar four-accumulator loop does. That is a
//! prediction this probe can falsify.
//!
//! HOW IT REACHES BOTH ROUTES WITHOUT A SOURCE CHANGE. The dispatch gate requires
//! `out_ch % 4 == 0`, so out_ch=30 takes the streamed route while the SAME 30 output
//! channels sitting inside a 32-channel weight take the direct one. Item 68d established
//! this trick; it needs no edit to `ft-kernel-cpu/src/lib.rs`, which currently carries
//! another agent's uncommitted work.
//!
//! This probe writes the inputs and both routes' outputs as raw little-endian f64, and
//! `scripts/conv3d_route_torch_arbiter.py` computes the torch reference and reports each
//! route's error against it. Splitting it that way keeps the reference in torch's own
//! process rather than reimplementing a convolution here and calling it a reference.
//!
//! Arm-internal for the FT half; the comparison is against a real torch build, pinned.
//!
//! Run:
//! ```text
//! cargo run --release -p frankentorch-api --features fair-alloc \
//!     --example conv3d_route_torch_arbiter
//! /data/tmp/torchvenv-2121/bin/python scripts/conv3d_route_torch_arbiter.py
//! ```

use std::io::Write as _;

/// Shape of the h2h `conv3d` lane: input [2,32,8,16,16], weight [.,32,3,3,3], pad 1.
const BATCH: usize = 2;
const IN_CH: usize = 32;
const SPATIAL_D: usize = 8;
const SPATIAL_H: usize = 16;
const SPATIAL_W: usize = 16;
const K: usize = 3;

/// Channels compared. 30 is not a multiple of 4, so it takes the STREAMED route.
const OUT_CH_STREAM: usize = 30;
/// 32 is, so it takes the DIRECT route; its first 30 channels use identical weights.
const OUT_CH_DIRECT: usize = 32;

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
    let min = mhz[0];
    let max = mhz[mhz.len() - 1];
    let mean = mhz.iter().sum::<f64>() / mhz.len() as f64;
    format!(
        "min={min:.0} mean={mean:.0} max={max:.0} spread={:.2}x",
        max / min
    )
}

fn dump(dir: &str, name: &str, values: &[f64]) {
    let path = format!("{dir}/{name}");
    let mut bytes = Vec::with_capacity(values.len() * 8);
    for value in values {
        bytes.extend_from_slice(&value.to_le_bytes());
    }
    let mut file = std::fs::File::create(&path).expect("create dump");
    file.write_all(&bytes).expect("write dump");
    println!("wrote {path}  ({} f64)", values.len());
}

fn main() {
    let dir = std::env::var("CONV3D_ARBITER_DIR")
        .unwrap_or_else(|_| "/data/tmp/conv3d_arbiter_68pwz".to_owned());
    std::fs::create_dir_all(&dir).expect("create dump dir");

    let (pd, ph, pw) = (SPATIAL_D + 2, SPATIAL_H + 2, SPATIAL_W + 2);
    let (od, oh, ow) = (pd - K + 1, ph - K + 1, pw - K + 1);

    // The same generators the sibling conv3d probes use, so the numbers are comparable
    // with items 68 and 70 rather than being a fresh unrelated draw.
    let padded: Vec<f64> = (0..BATCH * IN_CH * pd * ph * pw)
        .map(|index| ((index % 251) as f64) * 0.001 - 0.12)
        .collect();
    let weight_direct: Vec<f64> = (0..OUT_CH_DIRECT * IN_CH * K * K * K)
        .map(|index| ((index % 241) as f64) * 0.001 - 0.11)
        .collect();
    // First 30 channels are bit-identical to the 32-channel weight's first 30, which is
    // what makes the two routes' outputs comparable element for element.
    let per_channel = IN_CH * K * K * K;
    let weight_stream: Vec<f64> = weight_direct[..OUT_CH_STREAM * per_channel].to_vec();

    println!("conv3d_route_torch_arbiter (frankentorch-conv3d-direct-gate-misset-w3pol)");
    println!(
        "host={}",
        std::env::var("HOSTNAME").unwrap_or_else(|_| "thinkstation1".into())
    );
    println!("rayon_threads={}", rayon::current_num_threads());
    println!(
        "input [{BATCH},{IN_CH},{pd},{ph},{pw}] (pre-padded)  weight [.,{IN_CH},3,3,3]  \
         out [{BATCH},.,{od},{oh},{ow}]  k={} ",
        per_channel
    );
    println!("pre  loadavg {}", loadavg());
    println!("pre  cpu_mhz {}", cpu_mhz());
    println!();

    let out_stream = ft_kernel_cpu::conv3d_forward_f64(
        &padded,
        &weight_stream,
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
        OUT_CH_STREAM,
    );
    let out_direct_full = ft_kernel_cpu::conv3d_forward_f64(
        &padded,
        &weight_direct,
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
        OUT_CH_DIRECT,
    );

    // Slice the direct run's first 30 channels out of its 32, per batch.
    let patch_count = od * oh * ow;
    let mut out_direct = Vec::with_capacity(BATCH * OUT_CH_STREAM * patch_count);
    for b in 0..BATCH {
        let base = b * OUT_CH_DIRECT * patch_count;
        out_direct.extend_from_slice(&out_direct_full[base..base + OUT_CH_STREAM * patch_count]);
    }
    assert_eq!(
        out_direct.len(),
        out_stream.len(),
        "the two routes must be compared over the same element count"
    );

    // Self-check FIRST, and print it, so a reader can see the routes really did diverge
    // in THIS run rather than trusting item 68d's number from a different binary.
    let mut differing = 0usize;
    let mut worst_pair = 0.0f64;
    for (s, d) in out_stream.iter().zip(out_direct.iter()) {
        if s.to_bits() != d.to_bits() {
            differing += 1;
            let scale = s.abs().max(d.abs()).max(f64::MIN_POSITIVE);
            worst_pair = worst_pair.max((s - d).abs() / scale);
        }
    }
    println!(
        "ROUTE-vs-ROUTE (reproduces item 68d): {differing} of {} outputs differ, \
         worst relative {worst_pair:.3e}",
        out_stream.len()
    );
    println!("  ^ this says only that they DISAGREE. It cannot say which one is right.");
    println!();

    dump(&dir, "padded.f64", &padded);
    dump(&dir, "weight.f64", &weight_stream);
    dump(&dir, "out_stream.f64", &out_stream);
    dump(&dir, "out_direct.f64", &out_direct);
    let shape = format!(
        "{BATCH} {IN_CH} {pd} {ph} {pw} {OUT_CH_STREAM} {K} {od} {oh} {ow}\n"
    );
    std::fs::write(format!("{dir}/shape.txt"), shape).expect("write shape");

    println!();
    println!("post loadavg {}", loadavg());
    println!("post cpu_mhz {}", cpu_mhz());
    println!();
    println!("NOW RUN THE ARBITER:");
    println!("  /data/tmp/torchvenv-2121/bin/python scripts/conv3d_route_torch_arbiter.py");
}

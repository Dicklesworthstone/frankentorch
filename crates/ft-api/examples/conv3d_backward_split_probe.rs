//! WHERE inside `tensor_backward` does the conv3d lane's 9 ms of non-kernel time go? —
//! `frankentorch-l2zki`.
//!
//! Item 76 cut the session arm into stages and found the backward stage is 78% of the
//! lane (14.468 ms) while item 73 measured the conv3d KERNEL backward at 5.539 ms. That
//! left ~9 ms attributed only to a list of suspects: tensor_pad's backward, tensor_sum's
//! backward, gradient accumulation, and the custom-op wrapper. A list of suspects is not
//! an attribution.
//!
//! A `perf record` of conv3d_engine_probe refuted the obvious guess: the whole of
//! `TensorTape::backward_with_options` (which is where tensor_pad's serial per-element
//! un-pad loop lives) is 0.34% of cycles. So this probe stops guessing and SUBTRACTS
//! LANES instead, using only the public session API — each lane below is the full lane
//! with exactly one node removed, so the difference is that node's backward cost
//! including its share of the wrapper.
//!
//!   A  leaf -> conv3d(pad=1) -> sum -> backward     the scored lane
//!   B  leaf(padded shape) -> conv3d(pad=0) -> sum -> backward    A minus the pad node
//!   C  leaf -> pad -> sum -> backward               A minus the conv node
//!   D  leaf(padded shape) -> sum -> backward        C minus the pad node
//!   K  ft_kernel_cpu::conv3d_backward_f64 alone     no tape at all
//!
//! Only the BACKWARD stage is timed in A-D (the timer starts after tensor_sum returns),
//! so forward and tape-build costs are outside every number. Arm-internal: no incumbent,
//! no ratio, no drift gate.

use ft_api::FrankenTorchSession;
use ft_core::ExecutionMode;
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
    let unpadded_numel = BATCH * IN_CH * SD * SH * SW;
    let padded_numel = BATCH * IN_CH * PD * PH * PW;
    let values: Vec<f64> = (0..unpadded_numel)
        .map(|index| ((index % 251) as f64) * 0.001 - 0.12)
        .collect();
    let padded_values: Vec<f64> = (0..padded_numel)
        .map(|index| ((index % 251) as f64) * 0.001 - 0.12)
        .collect();
    let weights: Vec<f64> = (0..OUT_CH * IN_CH * K * K * K)
        .map(|index| ((index % 241) as f64) * 0.001 - 0.11)
        .collect();

    println!("conv3d_backward_split_probe (frankentorch-l2zki)");
    println!("shape [{BATCH},{IN_CH},{SD},{SH},{SW}] k=3 s=1 pad=1 out_ch={OUT_CH}");
    println!("unpadded numel {unpadded_numel}   padded numel {padded_numel}");
    println!("rayon_threads={}", rayon::current_num_threads());
    println!("pre  loadavg {}", loadavg());
    println!("pre  cpu_mhz {}", cpu_mhz());
    println!();

    let mut lane_a = f64::INFINITY;
    let mut lane_b = f64::INFINITY;
    let mut lane_c = f64::INFINITY;
    let mut lane_d = f64::INFINITY;
    let mut lane_k = f64::INFINITY;

    // A padded input the kernel arm can reuse; built once, outside every timer.
    let mut padded_for_kernel = vec![0.0f64; padded_numel];
    for n in 0..BATCH {
        for c in 0..IN_CH {
            for d in 0..SD {
                for h in 0..SH {
                    let src = ((n * IN_CH + c) * SD + d) * SH * SW + h * SW;
                    let dst = ((n * IN_CH + c) * PD + d + 1) * PH * PW + (h + 1) * PW + 1;
                    padded_for_kernel[dst..dst + SW].copy_from_slice(&values[src..src + SW]);
                }
            }
        }
    }
    let dout_ones = vec![1.0f64; BATCH * OUT_CH * OD * OH * OW];

    for _ in 0..REPS {
        // ---- A: the scored lane. Timer starts AFTER the sum, so this is the backward
        // stage exactly as item 76 cut it. ----
        {
            let mut s = FrankenTorchSession::new(ExecutionMode::Strict);
            let x = s
                .tensor_variable(values.clone(), vec![BATCH, IN_CH, SD, SH, SW], true)
                .expect("leaf");
            let w = s
                .tensor_variable(weights.clone(), vec![OUT_CH, IN_CH, K, K, K], false)
                .expect("weight");
            let out = s
                .functional_conv3d(x, w, None, (1, 1, 1), (1, 1, 1))
                .expect("conv3d");
            let loss = s.tensor_sum(out).expect("sum");
            let started = Instant::now();
            let report = s.tensor_backward(loss).expect("backward");
            lane_a = lane_a.min(started.elapsed().as_secs_f64() * 1_000.0);
            assert!(report.gradient(x).is_some());
        }

        // ---- B: A with the PAD NODE REMOVED. Same conv kernel work (the conv always
        // runs on the padded buffer); the leaf is the padded tensor, so the tape is
        // leaf -> customfn -> sum instead of leaf -> pad -> customfn -> sum. ----
        {
            let mut s = FrankenTorchSession::new(ExecutionMode::Strict);
            let xp = s
                .tensor_variable(padded_values.clone(), vec![BATCH, IN_CH, PD, PH, PW], true)
                .expect("padded leaf");
            let w = s
                .tensor_variable(weights.clone(), vec![OUT_CH, IN_CH, K, K, K], false)
                .expect("weight");
            let out = s
                .functional_conv3d(xp, w, None, (1, 1, 1), (0, 0, 0))
                .expect("conv3d nopad");
            let loss = s.tensor_sum(out).expect("sum");
            let started = Instant::now();
            let report = s.tensor_backward(loss).expect("backward");
            lane_b = lane_b.min(started.elapsed().as_secs_f64() * 1_000.0);
            assert!(report.gradient(xp).is_some());
        }

        // ---- C: A with the CONV NODE REMOVED. leaf -> pad -> sum. The sum here reduces
        // the PADDED numel, so its backward broadcasts over padded_numel, matching the
        // gradient that reaches the pad node in lane A. ----
        {
            let mut s = FrankenTorchSession::new(ExecutionMode::Strict);
            let x = s
                .tensor_variable(values.clone(), vec![BATCH, IN_CH, SD, SH, SW], true)
                .expect("leaf");
            let padded = s.tensor_pad(x, &[1, 1, 1, 1, 1, 1], 0.0).expect("pad");
            let loss = s.tensor_sum(padded).expect("sum");
            let started = Instant::now();
            let report = s.tensor_backward(loss).expect("backward");
            lane_c = lane_c.min(started.elapsed().as_secs_f64() * 1_000.0);
            assert!(report.gradient(x).is_some());
        }

        // ---- D: C with the PAD NODE REMOVED: leaf(padded shape) -> sum. This is the
        // floor — one broadcast of the scalar seed over padded_numel plus report build. ----
        {
            let mut s = FrankenTorchSession::new(ExecutionMode::Strict);
            let xp = s
                .tensor_variable(padded_values.clone(), vec![BATCH, IN_CH, PD, PH, PW], true)
                .expect("padded leaf");
            let loss = s.tensor_sum(xp).expect("sum");
            let started = Instant::now();
            let report = s.tensor_backward(loss).expect("backward");
            lane_d = lane_d.min(started.elapsed().as_secs_f64() * 1_000.0);
            assert!(report.gradient(xp).is_some());
        }

        // ---- K: the conv3d backward kernel with no tape at all. ----
        {
            let started = Instant::now();
            let (dpadded, dweight, _) = ft_kernel_cpu::conv3d_backward_f64(
                &dout_ones,
                &padded_for_kernel,
                &weights,
                BATCH,
                IN_CH,
                PD,
                PH,
                PW,
                K,
                K,
                K,
                OD,
                OH,
                OW,
                1,
                1,
                1,
                OUT_CH,
                false,
            );
            lane_k = lane_k.min(started.elapsed().as_secs_f64() * 1_000.0);
            std::hint::black_box((&dpadded, &dweight));
        }
    }

    println!("BACKWARD STAGE ONLY (min of {REPS}), ms:");
    println!("{:>52}  {:>9}", "lane", "min ms");
    println!("{:>52}  {lane_a:>9.3}", "A  leaf -> conv3d(pad=1) -> sum   [the lane]");
    println!("{:>52}  {lane_b:>9.3}", "B  leaf(padded) -> conv3d(pad=0) -> sum");
    println!("{:>52}  {lane_c:>9.3}", "C  leaf -> pad -> sum");
    println!("{:>52}  {lane_d:>9.3}", "D  leaf(padded) -> sum");
    println!("{:>52}  {lane_k:>9.3}", "K  conv3d_backward_f64 kernel, no tape");
    println!();
    println!("SUBTRACTIONS:");
    println!(
        "{:>52}  {:>9.3}",
        "pad node backward            (C - D)",
        lane_c - lane_d
    );
    println!(
        "{:>52}  {:>9.3}",
        "pad node backward, in-lane   (A - B)",
        lane_a - lane_b
    );
    println!(
        "{:>52}  {:>9.3}",
        "conv custom-op wrapper       (B - D - K)",
        lane_b - lane_d - lane_k
    );
    println!(
        "{:>52}  {:>9.3}",
        "sum backward + report floor  (D)",
        lane_d
    );
    println!(
        "{:>52}  {:>9.3}",
        "kernel share of the lane     (K / A)",
        lane_k / lane_a
    );
    println!();
    println!("post loadavg {}", loadavg());
    println!("post cpu_mhz {}", cpu_mhz());
}

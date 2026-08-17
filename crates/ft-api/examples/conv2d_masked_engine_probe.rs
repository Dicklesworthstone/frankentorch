//! How much of the `conv2d_masked` lane is ENGINE rather than kernel? — `frankentorch-hi9r6`.
//!
//! Items 135c and 137d both named this probe as the ONLY sensible next step on this bead, and
//! item 137d added that nobody should touch the conv2d backward until it exists. Here it is.
//!
//! The case for it, from two directions:
//!   * SUMMED route — the all-ones backward kernel is 1.766 ms while PyTorch's whole lane is
//!     ~3.1 ms, yet our lane is 1.7-2.1x SLOWER (item 135). The backward is already ahead of
//!     the entire incumbent, so the loss is elsewhere.
//!   * MASKED route — the lane is 15.7-16.4 ms against a ~7.5 ms backward kernel (item 137),
//!     so about half of it is unaccounted for.
//!
//! Both of those subtract numbers taken in DIFFERENT processes at different pool widths, which
//! is the error item 123 caught in the other direction. This probe puts the session arm and the
//! kernels arm in ONE invocation so the difference is the engine term and nothing else.
//!
//!   session arm   FrankenTorchSession: conv2d -> mul(mask) -> sum -> backward
//!                 exactly the `conv2d_masked` lane's timed region
//!   kernels arm   the same work with NO session and NO tape: pad, conv2d_forward_f64, mask
//!                 multiply, sum, conv2d_backward_f64 fed `mask` as its dout
//!
//! RUN IT AT BOTH POOL WIDTHS. Item 123 measured the engine term moving 3.8x between 8 and 64
//! threads on conv3d while the kernels did not move at all, and the certified rows are taken at
//! `RAYON_NUM_THREADS=8`. A split measured at one width is not evidence about a standing taken
//! at the other.
//!
//! Arm-internal: no incumbent, no ratio, no drift gate, so it stays honest on a busy host.

use ft_api::FrankenTorchSession;
use ft_core::ExecutionMode;
use std::time::Instant;

// The conv2d_masked lane's shape, copied verbatim.
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
    let out_numel = BATCH * OUT_CH * OH * OW;
    let values: Vec<f64> = (0..BATCH * IN_CH * H * W)
        .map(|i| ((i % 251) as f64) * 0.001 - 0.12)
        .collect();
    let weights: Vec<f64> = (0..OUT_CH * IN_CH * K * K)
        .map(|i| ((i % 241) as f64) * 0.001 - 0.11)
        .collect();
    // Same generator the lane's `c2m` uses, so this probe's mask is the lane's mask.
    let mask: Vec<f64> = (0..out_numel)
        .map(|i| ((i % 251) as f64) * 0.001 - 0.12)
        .collect();

    println!("conv2d_masked_engine_probe (frankentorch-hi9r6)");
    println!("shape [{BATCH},{IN_CH},{H},{W}] k={K} s=1 pad=1 out_ch={OUT_CH}");
    println!("rayon_threads={}", rayon::current_num_threads());
    println!("pre  loadavg {}", loadavg());
    println!("pre  cpu_mhz {}", cpu_mhz());
    println!();

    let mut session_best = f64::INFINITY;
    let mut kernels_best = f64::INFINITY;
    let mut fwd_only = f64::INFINITY;

    for _ in 0..REPS {
        // ---- session arm: exactly the conv2d_masked lane's timed region ----
        let mut session = FrankenTorchSession::new(ExecutionMode::Strict);
        let x = session
            .tensor_variable(values.clone(), vec![BATCH, IN_CH, H, W], true)
            .expect("leaf");
        let w = session
            .tensor_variable(weights.clone(), vec![OUT_CH, IN_CH, K, K], false)
            .expect("weight");
        let m = session
            .tensor_variable(mask.clone(), vec![BATCH, OUT_CH, OH, OW], false)
            .expect("mask");
        let started = Instant::now();
        let out = session
            .functional_conv2d(x, w, None, (1, 1), (1, 1))
            .expect("conv2d");
        let scaled = session.tensor_mul(out, m).expect("mask multiply");
        let loss = session.tensor_sum(scaled).expect("sum");
        let report = session.tensor_backward(loss).expect("backward");
        let elapsed = started.elapsed().as_secs_f64() * 1_000.0;
        assert!(report.gradient(x).is_some());
        session_best = session_best.min(elapsed);

        // A second session cut at the forward, so the split says WHICH side the engine term
        // sits on rather than only that it exists.
        let mut s2 = FrankenTorchSession::new(ExecutionMode::Strict);
        let x2 = s2
            .tensor_variable(values.clone(), vec![BATCH, IN_CH, H, W], true)
            .expect("leaf");
        let w2 = s2
            .tensor_variable(weights.clone(), vec![OUT_CH, IN_CH, K, K], false)
            .expect("weight");
        let t0 = Instant::now();
        let o2 = s2
            .functional_conv2d(x2, w2, None, (1, 1), (1, 1))
            .expect("conv2d");
        fwd_only = fwd_only.min(t0.elapsed().as_secs_f64() * 1_000.0);
        std::hint::black_box(o2);

        // ---- kernels arm: same work, no session, no tape ----
        let started = Instant::now();
        let mut padded = vec![0.0f64; BATCH * IN_CH * PH * PW];
        for n in 0..BATCH {
            for c in 0..IN_CH {
                for h in 0..H {
                    let src = ((n * IN_CH + c) * H + h) * W;
                    let dst = ((n * IN_CH + c) * PH + h + 1) * PW + 1;
                    padded[dst..dst + W].copy_from_slice(&values[src..src + W]);
                }
            }
        }
        let out = ft_kernel_cpu::conv2d_forward_f64(
            &padded, &weights, None, BATCH, IN_CH, PH, PW, K, K, OH, OW, 1, 1, OUT_CH,
        );
        // The lane's loss is sum(out * mask), whose gradient wrt `out` is `mask` -- which is
        // what the tape hands the backward, so that is what the kernel arm feeds.
        let loss: f64 = out.iter().zip(mask.iter()).map(|(o, m)| o * m).sum();
        let (dpadded, dweight, _) = ft_kernel_cpu::conv2d_backward_f64(
            &mask, &padded, &weights, BATCH, IN_CH, PH, PW, K, K, OH, OW, 1, 1, OUT_CH, false,
        );
        let elapsed = started.elapsed().as_secs_f64() * 1_000.0;
        assert!(loss.is_finite());
        std::hint::black_box((&dpadded, &dweight));
        kernels_best = kernels_best.min(elapsed);
    }

    let engine = session_best - kernels_best;
    println!("{:>40}  {:>10}  {:>8}", "arm", "min ms", "share");
    println!(
        "{:>40}  {session_best:>10.3}  {:>7.1}%",
        "session (the lane's timed region)", 100.0
    );
    println!(
        "{:>40}  {kernels_best:>10.3}  {:>7.1}%",
        "kernels only (pad+fwd+loss+bwd)",
        100.0 * kernels_best / session_best
    );
    println!(
        "{:>40}  {engine:>10.3}  {:>7.1}%",
        "ENGINE (session - kernels)",
        100.0 * engine / session_best
    );
    println!(
        "{:>40}  {fwd_only:>10.3}",
        "session, through the forward only"
    );
    println!();
    println!(
        "CONTEXT: item 137 certified nothing but measured this lane at 15.7-16.4 ms against \
         PyTorch's 2.7-3.1 ms. If the ENGINE share is large at RAYON_NUM_THREADS=8 -- the width \
         the standings are taken at -- then the conv2d backward is NOT the lever, and items \
         133's kernel win buying nothing at the lane is explained."
    );
    println!("post loadavg {}", loadavg());
    println!("post cpu_mhz {}", cpu_mhz());
}

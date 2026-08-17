//! Where does the conv3d LANE's time go that the kernels do not explain? — `frankentorch-l2zki`.
//!
//! Item 74 certified conv3d at 3.04x SLOWER with the lane's FT arm reading 19.4-22.9 ms,
//! while item 73 measured forward+backward at 9.582 ms. **More than half the lane is
//! engine** — tape, session, padding — and it has never been priced.
//!
//! This is the same split the h2h board already carries for GroupNorm as twin lanes
//! (`group_norm_f32` against `group_norm_f32_kernels`), and reading those twins is what
//! found the 20 ms serial widen in item 69. conv3d has no twin lane, so this probe builds
//! one:
//!
//!   session arm   FrankenTorchSession: functional_conv3d -> tensor_sum -> tensor_backward
//!   kernels arm   the same work with NO session and NO tape: pad, conv3d_forward_f64,
//!                 sum, conv3d_backward_f64
//!
//! The difference is the engine term. Arm-internal: no incumbent, no ratio, no drift gate,
//! so it is honest on a busy host.

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
    let values: Vec<f64> = (0..BATCH * IN_CH * SD * SH * SW)
        .map(|index| ((index % 251) as f64) * 0.001 - 0.12)
        .collect();
    let weights: Vec<f64> = (0..OUT_CH * IN_CH * K * K * K)
        .map(|index| ((index % 241) as f64) * 0.001 - 0.11)
        .collect();

    println!("conv3d_engine_probe (frankentorch-l2zki)");
    println!("shape [{BATCH},{IN_CH},{SD},{SH},{SW}] k=3 s=1 pad=1 out_ch={OUT_CH}");
    println!("rayon_threads={}", rayon::current_num_threads());
    println!("pre  loadavg {}", loadavg());
    println!("pre  cpu_mhz {}", cpu_mhz());
    println!();

    let reps = 7;
    let mut session_best = f64::INFINITY;
    let mut kernels_best = f64::INFINITY;

    for _ in 0..reps {
        // ---- session arm: exactly the h2h conv3d lane's timed region ----
        let mut session = FrankenTorchSession::new(ExecutionMode::Strict);
        let x = session
            .tensor_variable(values.clone(), vec![BATCH, IN_CH, SD, SH, SW], true)
            .expect("conv3d leaf");
        let w = session
            .tensor_variable(weights.clone(), vec![OUT_CH, IN_CH, K, K, K], false)
            .expect("conv3d weight");
        let started = Instant::now();
        let out = session
            .functional_conv3d(x, w, None, (1, 1, 1), (1, 1, 1))
            .expect("conv3d");
        let loss = session.tensor_sum(out).expect("sum");
        let report = session.tensor_backward(loss).expect("backward");
        let elapsed = started.elapsed().as_secs_f64() * 1_000.0;
        let checksum: f64 = report
            .gradient(x)
            .expect("grad")
            .iter()
            .map(|g| g.abs())
            .sum();
        assert!(checksum.is_finite());
        session_best = session_best.min(elapsed);

        // ---- kernels arm: the same work, no session, no tape ----
        let (pd, ph, pw) = (SD + 2, SH + 2, SW + 2);
        let (od, oh, ow) = (pd - K + 1, ph - K + 1, pw - K + 1);
        let started = Instant::now();
        // The pad the session also has to do, kept INSIDE the timer so the difference
        // is tape/session overhead and not simply the padding.
        let mut padded = vec![0.0f64; BATCH * IN_CH * pd * ph * pw];
        for n in 0..BATCH {
            for c in 0..IN_CH {
                for d in 0..SD {
                    for h in 0..SH {
                        let src = ((n * IN_CH + c) * SD + d) * SH * SW + h * SW;
                        let dst = ((n * IN_CH + c) * pd + d + 1) * ph * pw + (h + 1) * pw + 1;
                        padded[dst..dst + SW].copy_from_slice(&values[src..src + SW]);
                    }
                }
            }
        }
        let out = ft_kernel_cpu::conv3d_forward_f64(
            &padded, &weights, None, BATCH, IN_CH, pd, ph, pw, K, K, K, od, oh, ow, 1, 1, 1,
            OUT_CH,
        );
        let loss: f64 = out.iter().sum();
        let dout = vec![1.0f64; out.len()];
        let (dpadded, dweight, _) = ft_kernel_cpu::conv3d_backward_f64(
            &dout, &padded, &weights, BATCH, IN_CH, pd, ph, pw, K, K, K, od, oh, ow, 1, 1, 1,
            OUT_CH, false,
        );
        let elapsed = started.elapsed().as_secs_f64() * 1_000.0;
        assert!(loss.is_finite());
        std::hint::black_box((&dpadded, &dweight));
        kernels_best = kernels_best.min(elapsed);
    }

    let engine = session_best - kernels_best;
    println!("{:>34}  {:>10}  {:>8}", "arm", "min ms", "share");
    println!(
        "{:>34}  {session_best:>10.3}  {:>7.1}%",
        "session (functional_conv3d + tape)", 100.0
    );
    println!(
        "{:>34}  {kernels_best:>10.3}  {:>7.1}%",
        "kernels only (pad+fwd+sum+bwd)",
        100.0 * kernels_best / session_best
    );
    println!(
        "{:>34}  {engine:>10.3}  {:>7.1}%",
        "ENGINE (session - kernels)",
        100.0 * engine / session_best
    );
    println!();
    println!(
        "CONTEXT: item 74 certified this lane at 0.329 (3.04x SLOWER) with FT 19.4-22.9 ms. \
         Items 72/73 cut the kernels 1.71x and moved the certified standing only 1.17x, \
         which is what this split explains."
    );
    println!("post loadavg {}", loadavg());
    println!("post cpu_mhz {}", cpu_mhz());
}

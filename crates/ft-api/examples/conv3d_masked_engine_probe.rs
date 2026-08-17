//! How much of the `conv3d_masked` lane is ENGINE rather than kernel? — `frankentorch-l2zki`.
//!
//! Item 119 certified that lane at **1.67x SLOWER** (FT 10.780 ms, PT 6.841 ms) and
//! `conv3d_generic_phase_probe` measured its backward KERNEL at 5.725 ms. Subtracting those
//! two across builds would be item 82's mistake in reverse — they were taken in different
//! processes at different loads, and the lane also contains a forward and a loss that the
//! incumbent pays for too. This probe puts both arms in ONE invocation so the difference is
//! the engine term and nothing else.
//!
//!   session arm   FrankenTorchSession: conv3d -> mul(mask) -> sum -> backward
//!                 exactly the `conv3d_masked` lane's timed region
//!   kernels arm   the same work with NO session and NO tape: pad, conv3d_forward_f64,
//!                 mask multiply, sum, conv3d_backward_f64 with the SAME non-uniform dout
//!
//! The mask is what keeps this on the generic route: an all-ones upstream would take
//! `conv3d_backward_f64`'s fast path, which is a different kernel with its own standing
//! (items 98, 110). The kernels arm therefore feeds the backward `mask` as its `dout`, which
//! is exactly what the tape delivers there.
//!
//! Arm-internal: no incumbent, no ratio, no drift gate, so it is honest on a busy host. It
//! says which half to attack; it is not a standing and must not be quoted as one.

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
    let out_numel = BATCH * OUT_CH * OD * OH * OW;
    let values: Vec<f64> = (0..BATCH * IN_CH * SD * SH * SW)
        .map(|i| ((i % 251) as f64) * 0.001 - 0.12)
        .collect();
    let weights: Vec<f64> = (0..OUT_CH * IN_CH * K * K * K)
        .map(|i| ((i % 241) as f64) * 0.001 - 0.11)
        .collect();
    // Same generator the h2h lane's `c3m` uses, so this probe's mask is the lane's mask.
    let mask: Vec<f64> = (0..out_numel)
        .map(|i| ((i % 251) as f64) * 0.001 - 0.12)
        .collect();

    println!("conv3d_masked_engine_probe (frankentorch-l2zki)");
    println!("shape [{BATCH},{IN_CH},{SD},{SH},{SW}] k=3 s=1 pad=1 out_ch={OUT_CH}");
    println!("rayon_threads={}", rayon::current_num_threads());
    println!("pre  loadavg {}", loadavg());
    println!("pre  cpu_mhz {}", cpu_mhz());
    println!();

    let mut session_best = f64::INFINITY;
    let mut kernels_best = f64::INFINITY;
    let mut fwd_only = f64::INFINITY;

    for _ in 0..REPS {
        // ---- session arm: exactly the conv3d_masked lane's timed region ----
        let mut session = FrankenTorchSession::new(ExecutionMode::Strict);
        let x = session
            .tensor_variable(values.clone(), vec![BATCH, IN_CH, SD, SH, SW], true)
            .expect("leaf");
        let w = session
            .tensor_variable(weights.clone(), vec![OUT_CH, IN_CH, K, K, K], false)
            .expect("weight");
        let m = session
            .tensor_variable(mask.clone(), vec![BATCH, OUT_CH, OD, OH, OW], false)
            .expect("mask");
        let started = Instant::now();
        let out = session
            .functional_conv3d(x, w, None, (1, 1, 1), (1, 1, 1))
            .expect("conv3d");
        let scaled = session.tensor_mul(out, m).expect("mask multiply");
        let loss = session.tensor_sum(scaled).expect("sum");
        let report = session.tensor_backward(loss).expect("backward");
        let elapsed = started.elapsed().as_secs_f64() * 1_000.0;
        assert!(report.gradient(x).is_some());
        session_best = session_best.min(elapsed);

        // A second session cut at the forward, so the split says whether the engine term is
        // in the forward or the backward rather than only that it exists.
        let mut s2 = FrankenTorchSession::new(ExecutionMode::Strict);
        let x2 = s2
            .tensor_variable(values.clone(), vec![BATCH, IN_CH, SD, SH, SW], true)
            .expect("leaf");
        let w2 = s2
            .tensor_variable(weights.clone(), vec![OUT_CH, IN_CH, K, K, K], false)
            .expect("weight");
        let t0 = Instant::now();
        let o2 = s2
            .functional_conv3d(x2, w2, None, (1, 1, 1), (1, 1, 1))
            .expect("conv3d");
        fwd_only = fwd_only.min(t0.elapsed().as_secs_f64() * 1_000.0);
        std::hint::black_box(o2);

        // ---- kernels arm: same work, no session, no tape ----
        let started = Instant::now();
        let mut padded = vec![0.0f64; BATCH * IN_CH * PD * PH * PW];
        for n in 0..BATCH {
            for c in 0..IN_CH {
                for d in 0..SD {
                    for h in 0..SH {
                        let src = ((n * IN_CH + c) * SD + d) * SH * SW + h * SW;
                        let dst = ((n * IN_CH + c) * PD + d + 1) * PH * PW + (h + 1) * PW + 1;
                        padded[dst..dst + SW].copy_from_slice(&values[src..src + SW]);
                    }
                }
            }
        }
        let out = ft_kernel_cpu::conv3d_forward_f64(
            &padded, &weights, None, BATCH, IN_CH, PD, PH, PW, K, K, K, OD, OH, OW, 1, 1, 1, OUT_CH,
        );
        // The loss the lane computes: sum(out * mask). Its gradient wrt `out` is `mask`,
        // which is what the tape hands the backward, so that is what the kernel arm feeds.
        let loss: f64 = out.iter().zip(mask.iter()).map(|(o, m)| o * m).sum();
        let (dpadded, dweight, _) = ft_kernel_cpu::conv3d_backward_f64(
            &mask, &padded, &weights, BATCH, IN_CH, PD, PH, PW, K, K, K, OD, OH, OW, 1, 1, 1,
            OUT_CH, false,
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
        "CONTEXT: item 119 certified this lane at 1.67x SLOWER, FT 10.780 ms against PyTorch \
         6.841 ms. If the ENGINE share above is large, the kernels are no longer what keeps \
         this lane behind."
    );
    println!("post loadavg {}", loadavg());
    println!("post cpu_mhz {}", cpu_mhz());
}

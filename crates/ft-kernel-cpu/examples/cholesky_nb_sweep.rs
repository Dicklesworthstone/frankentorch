//! `frankentorch-valnx` — the Cholesky NB sweep, with BOTH phases counted, at three sizes.
//!
//! # Converted to INTERLEAVED arms — `frankentorch-stale-tuning-constants-lzku6`, ledger 293
//!
//! This file used to run `for nb { for rep }`. That is the ordering ledger 293 indicted: the
//! incumbent is measured FIRST in every pass, so drift across the pass is confounded with the
//! candidate, and the geqrf sweep built on it won 12/12 kernel cells and then lost its paired
//! lane by 8% (292i). Cholesky's own 128 -> 64 result (292d) came off a sweep with the same
//! defect — **it is trustworthy because of its LANE CERTIFICATION, not because of this sweep**,
//! and 293 says its sweep number was probably inflated the same way and merely had the margin to
//! survive. So the instrument is rebuilt rather than the conclusion revisited.
//!
//! Now: every nb is sampled inside every rep, the arm order reverses on odd reps, each arm's rep
//! figure is the min of two samples, and the reported effect is the MEDIAN OF PER-REP PAIRED
//! RATIOS against the shipped arm. Both estimators, an exact sign test, the incumbent's within-run
//! spread and an A/A null arm are printed for every cell. See `interleaved/mod.rs` for the rules.
//!
//! The incumbent arm runs the SHIPPED path with the knob unset — not the shipped value re-fed
//! through the override, which is how `feedback_unset_knob_means_forced_off` records a straw-man
//! arm0 being built. Its width is read back from the kernel, never written here as a literal.
//!
//! # The model, stated before the run
//!
//! Panel MACs over a whole factorisation are `(n/nb)·nb³/6 = n·nb²/6`. At n=512, nb=128 that
//! predicts 1,398,101 against a measured census of 1,398,016 (ledger 292a), so the model is EXACT
//! rather than fitted. The trailing update always does ~`n³/3` MACs whatever nb is; only its `k`
//! changes. Measured rates: panel **0.85 GF/s**, trailing **~90 GF/s** — a 100x gap per FLOP.
//!
//! So shrinking nb trades a QUADRATIC panel term against a shape change in a term that is two
//! orders of magnitude faster. At n=512 the prediction is:
//!
//!     nb    panel MACs   panel ms   trailing ms (if it holds 90 GF/s)
//!     128    1,398,101     3.29            0.99
//!      64      349,525     0.82            0.99
//!      32       87,381     0.21            0.99
//!
//! The break-even is a trailing rate of 25.9 GF/s — the swap pays unless k=64 costs that GEMM a
//! 3.5x degradation from ~90.
//!
//! # Why three sizes
//!
//! The optimum is a CURVE in n, not a constant: the panel term scales as `n·nb²` and the trailing
//! term as `n³`, so their ratio moves with n and an nb tuned at one size is a gate fitted to one
//! dataset (item 279 — the size curve IS the result). n=256/512/1024.
//!
//! # What is reported
//!
//! Per (n, nb): every phase from the in-kernel counters — panel, TRSM, trailing, upper-zero — taken
//! from that arm's FASTEST sample, beside the predicted panel ms, so the accounting closes at each
//! cell and a model that is only checked where it was fitted is not left unchecked. Then the four
//! trust columns.
//!
//! NOT BIT-EXACT ACROSS nb, and it does not need to be: blocking reassociates the trailing sums,
//! which is why the blocked path is already tolerance-validated rather than bitwise. Correctness
//! across widths is pinned by `cholesky_nb_knob_is_neutral_at_default_and_correct_elsewhere`.
//!
//!   cargo run --release -p frankentorch-kernel-cpu --example cholesky_nb_sweep -- [reps] [dtype]

mod interleaved;

use ft_core::{DType, Device, TensorMeta};

fn spd(n: usize) -> Vec<f64> {
    let mut a = vec![0.0f64; n * n];
    for i in 0..n {
        for j in 0..i {
            let v = ((i * 17 + j * 31) % 23) as f64 * 0.01 - 0.1;
            a[i * n + j] = v;
            a[j * n + i] = v;
        }
        a[i * n + i] = n as f64;
    }
    a
}

/// Phases of one arm's fastest sample: `(wall_ms, panel_ns, trsm_ns, trailing_ns, zero_ns)`.
type Phases = (f64, u64, u64, u64, u64);

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let reps = interleaved::reps_for(args.get(1).and_then(|v| v.parse().ok()).unwrap_or(10));
    // arg 2 selects the dtype lane: "f32" sweeps `cholesky_contiguous_f32`, whose NB has NEVER
    // been swept. Its f64 twin moved 128 -> 64 this session at a certified 1.264x, which is
    // evidence for LOOKING and not evidence for changing it — an f32 element is half the bytes,
    // so the trailing GEMM's cache behaviour and the panel's rate both differ.
    let dtype = args
        .get(2)
        .map_or("f64".to_owned(), std::clone::Clone::clone);
    let f32_lane = dtype == "f32";

    interleaved::banner(&format!("cholesky NB, dtype={dtype}"), reps);
    println!(
        "MODEL (registered): panel MACs = n*nb^2/6 at ~0.85 GF/s; trailing = n^3/3 MACs at ~90 \
         GF/s regardless of nb. Predicted panel ms is printed beside the measured one."
    );
    let shipped_nb = if f32_lane {
        ft_kernel_cpu::cholesky_nb_f32()
    } else {
        ft_kernel_cpu::cholesky_nb()
    };
    println!("INCUMBENT: the shipped path with the knob UNSET; it resolves to nb={shipped_nb}.");

    for n in [256usize, 512, 1024] {
        let a = spd(n);
        let meta = TensorMeta::from_shape(vec![n, n], DType::F64, Device::Cpu);
        let a32: Vec<f32> = a.iter().map(|&v| v as f32).collect();
        let meta32 = TensorMeta::from_shape(vec![n, n], DType::F32, Device::Cpu);

        let nbs: Vec<usize> = [16usize, 32, 64, 128, 256]
            .into_iter()
            .filter(|&nb| nb <= n)
            .collect();
        // Arm 0 is the shipped path; arms 1..=nbs.len() are the candidates; the LAST arm is a
        // second copy of the shipped path — the A/A null. It sits last on purpose: the odd-rep
        // reversal then gives it the maximum position contrast against arm 0, so it is the
        // least flattering null this ordering can produce.
        let arms = nbs.len() + 2;
        let aa = arms - 1;
        let label = |i: usize| -> String {
            if i == 0 {
                format!("shipped({shipped_nb})")
            } else if i == aa {
                "A/A".to_owned()
            } else if nbs[i - 1] == shipped_nb {
                // CONTROL: the incumbent's own width, driven through the KNOB instead of left
                // unset. It should read 1.0x; how far it misses is this run's floor.
                format!("nb={}*", nbs[i - 1])
            } else {
                format!("nb={}", nbs[i - 1])
            }
        };

        let mut best: Vec<Phases> = vec![(f64::INFINITY, 0, 0, 0, 0); arms];
        let times = interleaved::run(arms, reps, 2, |i| {
            if i == 0 || i == aa {
                if f32_lane {
                    ft_kernel_cpu::set_cholesky_nb_f32(0);
                } else {
                    ft_kernel_cpu::set_cholesky_nb(0);
                }
            } else if f32_lane {
                ft_kernel_cpu::set_cholesky_nb_f32(nbs[i - 1]);
            } else {
                ft_kernel_cpu::set_cholesky_nb(nbs[i - 1]);
            }
            let _ = ft_kernel_cpu::cholesky_stage_take_ns();
            let _ = ft_kernel_cpu::cholesky_f32_stage_take_ns();
            let started = std::time::Instant::now();
            let wall = if f32_lane {
                let f =
                    ft_kernel_cpu::cholesky_contiguous_f32(&a32, &meta32, false).expect("chol f32");
                let w = started.elapsed().as_secs_f64() * 1_000.0;
                std::hint::black_box(&f);
                w
            } else {
                let f = ft_kernel_cpu::cholesky_contiguous_f64(&a, &meta, false).expect("chol");
                let w = started.elapsed().as_secs_f64() * 1_000.0;
                std::hint::black_box(&f);
                w
            };
            let (p, t, tr, z) = if f32_lane {
                let (p, t, tr) = ft_kernel_cpu::cholesky_f32_stage_take_ns();
                (p, t, tr, 0)
            } else {
                ft_kernel_cpu::cholesky_stage_take_ns()
            };
            if wall < best[i].0 {
                best[i] = (wall, p, t, tr, z);
            }
            wall
        });
        if f32_lane {
            ft_kernel_cpu::set_cholesky_nb_f32(0);
        } else {
            ft_kernel_cpu::set_cholesky_nb(0);
        }

        let (lo, hi, gate) = interleaved::spread(&times[0]);
        println!(
            "\nn={n}   trailing MACs ~ n^3/3 = {:.3e}  (nb-INDEPENDENT)",
            (n as f64).powi(3) / 3.0
        );
        println!(
            "  incumbent WITHIN-RUN spread {lo:.4}-{hi:.4} ms = {gate:.3}x  (IQR {:.3}x) \
             — THE GATE: an effect at or below {:.4} is UNRESOLVED, not a result",
            interleaved::iqr_ratio(&times[0]),
            gate - 1.0
        );
        println!(
            "  {:>13} {:>9} {:>9} {:>9} {:>9} {:>9} {:>9} {:>10} {}",
            "arm",
            "median",
            "panel",
            "pred",
            "TRSM",
            "trail",
            "zero",
            "trailGF/s",
            interleaved::Verdict::header(),
        );
        for i in 0..arms {
            let (_, p, t, tr, z) = best[i];
            let ms = |v: u64| v as f64 / 1e6;
            let nb = if i == 0 || i == aa {
                shipped_nb
            } else {
                nbs[i - 1]
            };
            let pred_panel = 2.0 * (n as f64) * (nb as f64) * (nb as f64) / 6.0 / 0.85e9 * 1000.0;
            // A single-block factorisation has no trailing update at all; printing `inf` for a
            // phase that did not run reads as a measurement, so say it did not run.
            let trail_gf = if tr == 0 {
                "-".to_owned()
            } else {
                format!(
                    "{:.1}",
                    2.0 * (n as f64).powi(3) / 3.0 / (ms(tr) / 1000.0) / 1e9
                )
            };
            let trust = if i == 0 {
                format!("{:>9} {:>9} {:>8} {:>8}  incumbent", "-", "-", "-", "-")
            } else {
                interleaved::verdict(&times[0], &times[i], gate).row()
            };
            println!(
                "  {:>13} {:>9.4} {:>9.4} {:>9.4} {:>9.4} {:>9.4} {:>9.4} {:>10} {trust}",
                label(i),
                interleaved::median(&times[i]),
                ms(p),
                pred_panel,
                ms(t),
                ms(tr),
                ms(z),
                trail_gf,
            );
        }
    }
    ft_kernel_cpu::set_cholesky_nb(0);
    ft_kernel_cpu::set_cholesky_nb_f32(0);
    println!(
        "\nREADING: the A/A row is the harness's own resolution floor — if it does not read ~1.0 \
         with a coin-flip sign test, nothing else in the table is a measurement. A TRUSTED WIN is \
         necessary and NOT sufficient: item 291 recorded a bit-exact 1.28-1.40x isolation win that \
         moved no lane, and 292i one that inverted. Any nb change is also NOT bit-exact (blocking \
         reassociates the trailing sums), so it needs reconstruction and the oracle, never \
         to_bits()."
    );
}

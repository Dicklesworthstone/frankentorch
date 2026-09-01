//! `frankentorch-stale-tuning-constants-lzku6` — interleaved full-`eigh` TRED2 threshold sweep.
//!
//! This sweeps `TRED2_PAR_MIN_L_DEFAULT=384` only through the full-eigenvector reducer.  It is
//! deliberately not an `eigvalsh` sweep: values-only uses a different reducer and never reads
//! this threshold.  `FT_TNB` must remain unset so n>=1024 retains the shipped blocked reducer.
//!
//! Ledger 293 owns the sampling contract.  The shared `interleaved` module owns the repetition
//! loop, alternates arm order, records paired and marginal estimators, runs the exact sign test,
//! and rejects effects no larger than the incumbent's within-run spread.
//!
//! The profile's wall clock counts the whole full-eigh internal route: input packing, TRED2,
//! form-Q backtransform, deferred TQL2, and profile/harness overhead.  The three phase counters
//! are printed only to attribute a result; no reduction-only row can justify changing 384.
//!
//! ```text
//! FT_OP=eigh RAYON_NUM_THREADS=16 cargo run --release -p frankentorch-kernel-cpu \
//!   --example tred2_par_min_eigh_sweep -- 12
//! ```

mod interleaved;

const SHIPPED_TPM: usize = 384;

/// One arm's fastest sample: whole-profile wall time plus reduce/form-Q/TQL2 stage times.
type Cell = (f64, u128, u128, u128);

/// The generic symmetric fixture used by the live `eigh` lane.
fn generic_symmetric(n: usize) -> Vec<f64> {
    let mut a = vec![0.0f64; n * n];
    for i in 0..n {
        for j in 0..n {
            let h = ((i as i64) * 73 + (j as i64) * 151 + ((i * j) as i64) % 257)
                .rem_euclid(2048);
            a[i * n + j] = h as f64 / 2048.0 - 1.0 + if i == j { 16.0 } else { 0.0 };
        }
    }
    let mut sym = vec![0.0f64; n * n];
    for i in 0..n {
        for j in 0..n {
            sym[i * n + j] = (a[i * n + j] + a[j * n + i]) * 0.5;
        }
    }
    sym
}

fn main() {
    let op = std::env::var("FT_OP").unwrap_or_else(|_| "eigh".to_owned());
    assert_eq!(
        op, "eigh",
        "this threshold is exercised only by the full-eigh route; FT_OP must be eigh"
    );
    assert!(
        std::env::var_os("FT_TNB").is_none(),
        "FT_TNB changes the blocked reducer and confounds this TRED2_PAR_MIN_L sweep"
    );

    let args: Vec<String> = std::env::args().collect();
    let reps = interleaved::reps_for(args.get(1).and_then(|v| v.parse().ok()).unwrap_or(12));
    let thresholds = [256usize, 320, SHIPPED_TPM, 448, 512];
    let arms = thresholds.len() + 2;
    let aa = arms - 1;

    interleaved::banner("full eigh TRED2_PAR_MIN_L", reps);
    println!(
        "SCOPE: FT_OP=eigh; FT_TNB is unset. Arm 0 leaves the override unset (shipped \
         threshold {SHIPPED_TPM}); threshold={SHIPPED_TPM}* is the knob-neutral control."
    );
    println!(
        "MODEL: n=512 keeps the historical unblocked control; n=1024 and n=1536 exercise the \
         current blocked reducer. The threshold gates GGS in both routes and the blocked route's \
         rank-2k flush as well, so report the entire full-eigh profile."
    );

    for n in [512usize, 1024, 1536] {
        let a = generic_symmetric(n);
        let threshold = |i: usize| {
            if i == 0 || i == aa {
                0
            } else {
                thresholds[i - 1]
            }
        };
        let resolved_threshold = |i: usize| {
            let value = threshold(i);
            if value == 0 { SHIPPED_TPM } else { value }
        };
        let label = |i: usize| {
            if i == 0 {
                format!("shipped({SHIPPED_TPM})")
            } else if i == aa {
                "A/A".to_owned()
            } else if threshold(i) == SHIPPED_TPM {
                format!("tpm={}*", threshold(i))
            } else {
                format!("tpm={}", threshold(i))
            }
        };

        let mut best: Vec<Cell> = vec![(f64::INFINITY, 0, 0, 0); arms];
        let times = interleaved::run(arms, reps, 2, |i| {
            // Keep arm 0 genuinely unset. The gated profiler needs the resolved threshold as an
            // argument because its non-gated twin deliberately hard-codes 384.
            let previous = ft_kernel_cpu::set_tred2_par_min_l(threshold(i));
            let started = std::time::Instant::now();
            let (reduce, backtransform, tql2) =
                ft_kernel_cpu::eigh_stage_profile_gated_f64(&a, n, resolved_threshold(i));
            let wall = started.elapsed().as_secs_f64() * 1_000.0;
            ft_kernel_cpu::set_tred2_par_min_l(previous);
            std::hint::black_box((reduce, backtransform, tql2));
            if wall < best[i].0 {
                best[i] = (wall, reduce, backtransform, tql2);
            }
            wall
        });
        ft_kernel_cpu::set_tred2_par_min_l(0);

        let (lo, hi, gate) = interleaved::spread(&times[0]);
        println!("\nn={n}   fixture=generic-symmetric");
        println!(
            "  incumbent WITHIN-RUN spread {lo:.4}-{hi:.4} ms = {gate:.3}x (IQR {:.3}x) \
             — THE GATE: an effect at or below {:.4} is UNRESOLVED, not a result",
            interleaved::iqr_ratio(&times[0]),
            gate - 1.0
        );
        println!(
            "  {:>13} {:>9} {:>10} {:>10} {:>10} {:>10} {}",
            "arm",
            "median",
            "reduce",
            "form-Q",
            "tql2",
            "other",
            interleaved::Verdict::header(),
        );
        for i in 0..arms {
            let (wall, reduce, backtransform, tql2) = best[i];
            let ms = |v: u128| v as f64 / 1e6;
            let accounted = ms(reduce) + ms(backtransform) + ms(tql2);
            let trust = if i == 0 {
                format!("{:>9} {:>9} {:>8} {:>8}  incumbent", "-", "-", "-", "-")
            } else {
                interleaved::verdict(&times[0], &times[i], gate).row()
            };
            println!(
                "  {:>13} {:>9.4} {:>10.4} {:>10.4} {:>10.4} {:>10.4} {trust}",
                label(i),
                interleaved::median(&times[i]),
                ms(reduce),
                ms(backtransform),
                ms(tql2),
                wall - accounted,
            );
        }
    }

    ft_kernel_cpu::set_tred2_par_min_l(0);
    println!(
        "\nREADING: a trusted isolation row is necessary, not sufficient. Default 384 remains until \
         every admitted size wins and a live-PyTorch paired lane certification passes. The phase \
         columns explain a row; they do not replace that whole-lane gate."
    );
}

//! `frankentorch-valnx` — the LU (getrf) NB sweep, both phases counted, at three sizes.
//!
//! # Converted to INTERLEAVED arms — `frankentorch-stale-tuning-constants-lzku6`, ledger 293
//!
//! This file used to run `for nb { for rep }`, which measures the incumbent FIRST in every pass
//! and confounds drift across the pass with the candidate. Built that way, the geqrf sweep won all
//! twelve of its kernel cells and then lost its paired lane by 8% (292i); re-run interleaved, its
//! winning width collapsed to a coin flip (293). getrf's own 128 -> 64 result (292e) came off a
//! sweep with the same defect and is trustworthy because of its LANE CERTIFICATION, not because of
//! this sweep. The instrument is rebuilt; the conclusion stands on the gate that actually held it.
//!
//! Every nb is now sampled inside every rep, the arm order reverses on odd reps, each arm's rep
//! figure is the min of two samples, and the effect is the MEDIAN OF PER-REP PAIRED RATIOS against
//! the incumbent. Both estimators, an exact sign test, the incumbent's within-run spread and an
//! A/A null arm print for every cell. See `interleaved/mod.rs`.
//!
//! The incumbent's width is READ BACK from the kernel (`lu_nb()`), never written here as a
//! literal — this file's own header called it 128 for a session after 292e moved it to 64, which
//! is the bead's subject matter happening to the bead's own instrument.
//!
//! # Why this is NOT the cholesky change repeated
//!
//! Cholesky's NB went 128 -> 64 for 1.264x (ledger 292d). Applying the same direction blindly would
//! have been a REGRESSION here: LU's NB had already been retuned 64 -> 128 by an earlier ladder.
//! **The two ops' optima move in OPPOSITE directions, and the panel-MAC model says why.**
//!
//!     cholesky panel is nb x nb   ->  MACs ~ n*nb^2/6   QUADRATIC in nb
//!     LU       panel is m  x nb   ->  MACs ~ n^2*nb/4   LINEAR    in nb
//!
//! Halving cholesky's nb cuts its panel work FOURFOLD, which is why small nb wins there. Halving
//! LU's only halves it, so the trailing GEMM's preference for a larger `k` competes on much more
//! even terms. LU's panel is also `lu_factor_panel_recursive_f64`, which turns its internal updates
//! into GEMM — it is not the pure scalar chain cholesky's panel is, so it runs at a far better rate
//! and tolerates being larger.
//!
//! **So the transferable thing is the METHOD, not the constant.** That is the whole point of
//! re-deriving the model per op instead of propagating a number.
//!
//! # What is actually open here
//!
//! The ladder that put the constant at 128 swept {16, 32, 48, 64, 96, 128} and **128 won at the TOP
//! of that grid** — a winner against nothing above it. This file's own ledger records the same
//! mistake with the sign reversed: the SVD panel width sat at 16 because the grid was {16,32,64}
//! and never held 8, and 8 then won 15 of 16 cells. So this sweeps 192, 256 and 384 as well, and
//! reports both phases per cell so the ANSWER comes with its mechanism.
//!
//! # Reported
//!
//! Per (n, nb): panel / solve / trailing from the in-kernel counters taken at that arm's FASTEST
//! sample, the residual, the median wall time, and the panel MAC count computed EXACTLY (not from
//! the asymptotic form) so the measured panel rate can be compared across nb — a model checked only
//! where it was fitted is not a model. Then the four trust columns.
//!
//! NOT BIT-EXACT across nb: blocking reassociates. LU is tolerance-validated for the same reason
//! cholesky is.
//!
//!   cargo run --release -p frankentorch-kernel-cpu --example lu_nb_sweep -- [reps]

mod interleaved;

use ft_core::{DType, Device, TensorMeta};

/// EXACT panel multiply-accumulates for a blocked getrf: panel `p` has `m = n - p*nb` rows, and
/// its unblocked factorisation does `(m-j-1)*(nb-j-1)` MACs for each column `j`.
fn lu_panel_macs(n: usize, nb: usize) -> f64 {
    let mut total = 0.0f64;
    let mut jb = 0usize;
    while jb < n {
        let width = nb.min(n - jb);
        let m = n - jb;
        for j in 0..width {
            let rows = m.saturating_sub(j + 1);
            let cols = width.saturating_sub(j + 1);
            total += (rows * cols) as f64;
        }
        jb += nb;
    }
    total
}

fn diag_dominant(n: usize) -> Vec<f64> {
    (0..n * n)
        .map(|idx| {
            let (i, j) = (idx / n, idx % n);
            let v = (((i * 7 + j * 13) % 101) as f64 - 50.0) / 25.0;
            if i == j { v + n as f64 } else { v }
        })
        .collect()
}

/// One arm's fastest sample: `(wall_ms, panel_ns, solve_ns, trailing_ns)`.
type Cell = (f64, u64, u64, u64);

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let reps = interleaved::reps_for(args.get(1).and_then(|v| v.parse().ok()).unwrap_or(10));

    interleaved::banner("getrf NB", reps);
    let shipped_nb = ft_kernel_cpu::lu_nb();
    println!(
        "MODEL: LU panel MACs ~ n^2*nb/4, LINEAR in nb (cholesky's is n*nb^2/6, QUADRATIC) — which \
         is why the two ops' optima move in OPPOSITE directions."
    );
    println!(
        "INCUMBENT: nb={shipped_nb}, read back from the kernel. The ladder that once set this \
         constant had its winner at the TOP of a {{16..128}} grid, so 192/256/384 remain the \
         untested side."
    );

    for n in [256usize, 512, 1024] {
        let a = diag_dominant(n);
        let meta = TensorMeta::from_shape(vec![n, n], DType::F64, Device::Cpu);

        let nbs: Vec<usize> = [32usize, 64, 96, 128, 192, 256, 384]
            .into_iter()
            .filter(|&nb| nb <= n)
            .collect();
        // Arm 0 is the incumbent; arms 1..=nbs.len() are the candidates; the LAST arm is a second
        // copy of the incumbent — the A/A null, placed last so the odd-rep order reversal gives it
        // the maximum position contrast and therefore the least flattering reading.
        let arms = nbs.len() + 2;
        let aa = arms - 1;
        let width = |i: usize| {
            if i == 0 || i == aa {
                shipped_nb
            } else {
                nbs[i - 1]
            }
        };
        let label = |i: usize| {
            if i == 0 {
                format!("shipped({shipped_nb})")
            } else if i == aa {
                "A/A".to_owned()
            } else if nbs[i - 1] == shipped_nb {
                // CONTROL: the incumbent's own width as an ordinary candidate arm. It should read
                // 1.0x; how far it misses is this run's position floor.
                format!("nb={}*", nbs[i - 1])
            } else {
                format!("nb={}", nbs[i - 1])
            }
        };

        let mut best: Vec<Cell> = vec![(f64::INFINITY, 0, 0, 0); arms];
        let times = interleaved::run(arms, reps, 2, |i| {
            let _ = ft_kernel_cpu::lu_stage_take_ns();
            let started = std::time::Instant::now();
            let f = ft_kernel_cpu::lu_factor_contiguous_nb_f64(&a, &meta, width(i)).expect("lu");
            let wall = started.elapsed().as_secs_f64() * 1_000.0;
            std::hint::black_box(&f);
            let (p, s, t) = ft_kernel_cpu::lu_stage_take_ns();
            if wall < best[i].0 {
                best[i] = (wall, p, s, t);
            }
            wall
        });

        let (lo, hi, gate) = interleaved::spread(&times[0]);
        println!(
            "\nn={n}   total getrf MACs ~ n^3/3 = {:.3e}",
            (n as f64).powi(3) / 3.0
        );
        println!(
            "  incumbent WITHIN-RUN spread {lo:.4}-{hi:.4} ms = {gate:.3}x  (IQR {:.3}x) \
             — THE GATE: an effect at or below {:.4} is UNRESOLVED, not a result",
            interleaved::iqr_ratio(&times[0]),
            gate - 1.0
        );
        println!(
            "  {:>13} {:>9} {:>9} {:>9} {:>9} {:>9} {:>11} {:>10} {}",
            "arm",
            "median",
            "panel",
            "solve",
            "trail",
            "resid",
            "panel MACs",
            "panelGF/s",
            interleaved::Verdict::header(),
        );
        for i in 0..arms {
            let (wall, p, s, t) = best[i];
            let ms = |v: u64| v as f64 / 1e6;
            let macs = lu_panel_macs(n, width(i));
            let trust = if i == 0 {
                format!("{:>9} {:>9} {:>8} {:>8}  incumbent", "-", "-", "-", "-")
            } else {
                interleaved::verdict(&times[0], &times[i], gate).row()
            };
            println!(
                "  {:>13} {:>9.4} {:>9.4} {:>9.4} {:>9.4} {:>9.4} {macs:>11.3e} {:>10.2} {trust}",
                label(i),
                interleaved::median(&times[i]),
                ms(p),
                ms(s),
                ms(t),
                wall - (ms(p) + ms(s) + ms(t)),
                2.0 * macs / (ms(p) / 1000.0) / 1e9,
            );
        }
    }
    println!(
        "\nREADING: the A/A row is the harness's own resolution floor — if it does not read ~1.0 \
         with a coin-flip sign test, nothing else in the table is a measurement. Ship only on an \
         ALL-CELLS TRUSTED WIN plus a paired lane certification, per the cholesky template (292d). \
         A winner in one cell of a noisy sweep is the friendliest reading, not the answer."
    );
}

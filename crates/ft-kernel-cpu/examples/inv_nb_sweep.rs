//! `frankentorch-stale-tuning-constants-lzku6` lane 2 — the getri (`inv`) NB sweep.
//!
//! # Converted to INTERLEAVED arms — ledger 293
//!
//! Written block-ordered (`for nb { for rep }`), this harness would measure the incumbent FIRST in
//! every pass and confound drift across the pass with the candidate. That is exactly how the geqrf
//! panel-width sweep won all twelve of its kernel cells and then lost its paired lane by 8% (292i);
//! re-run interleaved, its winning width collapsed to a coin flip (293). Every nb here is now
//! sampled inside every rep, the arm order reverses on odd reps, each arm's rep figure is the min
//! of two samples, and the effect is the MEDIAN OF PER-REP PAIRED RATIOS against the incumbent.
//! Both estimators, an exact sign test, the incumbent's within-run spread and an A/A null arm print
//! for every cell. See `interleaved/mod.rs` for the rules and why each one is there.
//!
//! The incumbent runs the SHIPPED path with the knob UNSET and reads its width back from the
//! kernel, so arm0 is not the shipped number re-fed through the override — that shape is
//! `feedback_unset_knob_means_forced_off`, which once made a geqrf lane read 13.630x against a
//! true 7.002x.
//!
//! # The census picked this constant, not the one the lane list started with
//!
//! Lane 2 was opened against `lu_solve_contiguous_f64`'s NB. A census killed that: neither slogdet
//! NOR inv executes `lu_solve` — its counters read 0.0000 ms for both — because `inv` goes through
//! `lu_inverse_from_factor_f64`, a different function with its own `NB`, and `lu_solve` serves
//! `linalg.solve` instead. Re-tuning the first constant would have measured nothing, which is
//! ledger 292's finding restated (cholesky never calls `dgemm_sub_into` either).
//!
//! **So the constant swept here is the one `inv` actually runs**, and `inv` is a board lane, which
//! is what makes it certifiable.
//!
//! # The model
//!
//! getri here is two blocked triangular solves against the identity: FORWARD `L y = I` restricted
//! to the columns that can be nonzero (the identity-structure restriction shipped under 37sxo), and
//! BACKWARD `U x = y`. Each block of width nb does a small triangular solve on the diagonal block
//! plus a GEMM update of the remaining columns — so the per-block scalar work grows with nb while
//! the GEMM's `k` grows with it too. That is the same competing pair as the factorisations, but the
//! solve's scalar half is a TRSM rather than a panel factorisation, so its exponent is different
//! again and the optimum has to be measured rather than inherited from 292d/292e.
//!
//! Both halves are counted in-call (`lu_inverse_half_take_ns`) so the sweep reports WHERE the time
//! moves, not just that it moved.
//!
//! # Rules
//!
//! Three sizes; sweep both sides of the incumbent (the getrf ladder's winner sat at the TOP of its
//! grid and was a winner against nothing above it); ship only on an ALL-CELLS TRUSTED win; then a
//! paired lane certification on `inv`. NOT bit-exact — blocking reassociates.
//!
//!   cargo run --release -p frankentorch-kernel-cpu --example inv_nb_sweep -- [reps]

mod interleaved;

use ft_core::{DType, Device, TensorMeta};

/// One arm's fastest sample: `(wall_ms, forward_ns, backward_ns)`.
type Cell = (f64, u64, u64);

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let reps = interleaved::reps_for(args.get(1).and_then(|v| v.parse().ok()).unwrap_or(10));

    interleaved::banner("getri (inv) NB", reps);
    println!(
        "CENSUS THAT PICKED THIS CONSTANT: slogdet and inv both read 0.0000 ms of lu_solve, so the \
         NB that `inv` executes is lu_inverse_from_factor_f64's, not lu_solve_contiguous_f64's."
    );
    let shipped_nb = ft_kernel_cpu::lu_inv_nb();
    println!("INCUMBENT: the shipped path with the knob UNSET; it resolves to nb={shipped_nb}.");

    for n in [256usize, 512, 1024] {
        let a: Vec<f64> = (0..n * n)
            .map(|idx| {
                let (i, j) = (idx / n, idx % n);
                let v = (((i * 7 + j * 13) % 101) as f64 - 50.0) / 25.0;
                if i == j { v + n as f64 } else { v }
            })
            .collect();
        let meta = TensorMeta::from_shape(vec![n, n], DType::F64, Device::Cpu);

        let nbs: Vec<usize> = [16usize, 32, 64, 96, 128, 192]
            .into_iter()
            .filter(|&nb| nb <= n)
            .collect();
        // Arm 0 is the shipped path; arms 1..=nbs.len() are the candidates; the LAST arm is a
        // second copy of the shipped path — the A/A null, placed last so the odd-rep order
        // reversal gives it the maximum position contrast and the least flattering reading.
        let arms = nbs.len() + 2;
        let aa = arms - 1;
        let label = |i: usize| {
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

        let mut best: Vec<Cell> = vec![(f64::INFINITY, 0, 0); arms];
        let times = interleaved::run(arms, reps, 2, |i| {
            ft_kernel_cpu::set_lu_inv_nb(if i == 0 || i == aa { 0 } else { nbs[i - 1] });
            let _ = ft_kernel_cpu::lu_inverse_half_take_ns();
            let started = std::time::Instant::now();
            let v = ft_kernel_cpu::inv_tensor_contiguous_f64(&a, &meta).expect("inv");
            let wall = started.elapsed().as_secs_f64() * 1_000.0;
            std::hint::black_box(&v);
            let (f, b) = ft_kernel_cpu::lu_inverse_half_take_ns();
            if wall < best[i].0 {
                best[i] = (wall, f, b);
            }
            wall
        });
        ft_kernel_cpu::set_lu_inv_nb(0);

        let (lo, hi, gate) = interleaved::spread(&times[0]);
        println!("\nn={n}");
        println!(
            "  incumbent WITHIN-RUN spread {lo:.4}-{hi:.4} ms = {gate:.3}x  (IQR {:.3}x) \
             — THE GATE: an effect at or below {:.4} is UNRESOLVED, not a result",
            interleaved::iqr_ratio(&times[0]),
            gate - 1.0
        );
        println!(
            "  {:>13} {:>10} {:>10} {:>10} {:>10} {}",
            "arm",
            "median",
            "forward",
            "backward",
            "resid",
            interleaved::Verdict::header(),
        );
        for i in 0..arms {
            let (wall, f, b) = best[i];
            let ms = |v: u64| v as f64 / 1e6;
            let trust = if i == 0 {
                format!("{:>9} {:>9} {:>8} {:>8}  incumbent", "-", "-", "-", "-")
            } else {
                interleaved::verdict(&times[0], &times[i], gate).row()
            };
            println!(
                "  {:>13} {:>10.4} {:>10.4} {:>10.4} {:>10.4} {trust}",
                label(i),
                interleaved::median(&times[i]),
                ms(f),
                ms(b),
                wall - (ms(f) + ms(b)),
            );
        }
    }
    ft_kernel_cpu::set_lu_inv_nb(0);
    println!(
        "\nREADING: the A/A row is the harness's own resolution floor — if it does not read ~1.0 \
         with a coin-flip sign test, nothing else in the table is a measurement. Ship only on an \
         ALL-CELLS TRUSTED WIN plus a paired lane certification on `inv`. A kernel win is not a \
         lane win — ledger 291."
    );
}

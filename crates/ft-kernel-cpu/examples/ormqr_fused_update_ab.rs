//! `frankentorch-g0wpj` — the fused ORMQR update against the materialised one, interleaved.
//!
//! Arm 0 is the SHIPPED path (toggle off): allocate `upd`, `dgemm` into it, then walk it
//! subtracting into `C`. Arm 1 is the candidate: one `dgemm_sub_into` with alpha=-1 accumulating
//! straight into `C`. Arm 2 is an A/A duplicate of the shipped path.
//!
//! # What is actually being removed
//!
//! Per panel, and there are `n/32` panels: an `m*cc` allocation (2 MB at n=512, 8 MB at n=1024),
//! the write of that buffer, and the read-back-and-subtract pass. The GEMM itself is unchanged —
//! it is the same product, written to a different destination. So this is not a scheduling change
//! and it is not a blocking change; it is the removal of a round trip through DRAM, on a family
//! `project_gemm_bandwidth_vein` records as bandwidth-bound.
//!
//! # The hit counter is asserted, not assumed
//!
//! Every rep drains `ormqr_fused_subtract_hits_take()`. An arm reporting zero hits measured the
//! shipped path twice and would report a clean, meaningless null —
//! `feedback_unset_knob_means_forced_off` records a geqrf lane reading 13.630x against a true
//! 7.002x for exactly that, and `feedback_conflated_counter_and_inflated_pair` a second case. The
//! count is printed per size so it can be checked rather than trusted.
//!
//! # Correctness is settled before timing
//!
//! `ormqr_fused_subtract_matches_materialised_bitwise` gates all four (left/right x transpose)
//! routes on `to_bits`, so nothing here needs a tolerance and a difference in output would be a
//! test failure, not a footnote.
//!
//! Interleaved per rep, order reversed on odd reps, per-rep min, median of per-rep paired ratios,
//! both estimators, exact sign test, incumbent within-run spread, A/A arm — ledger 293/293a.
//!
//!   cargo run --release -p frankentorch-kernel-cpu --example ormqr_fused_update_ab -- [reps]

mod interleaved;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let reps = interleaved::reps_for(args.get(1).and_then(|v| v.parse().ok()).unwrap_or(12));

    interleaved::banner("ORMQR fused C -= V*W vs materialised upd", reps);
    println!(
        "REMOVED PER PANEL: an m*cc allocation, its write, and the read-back subtract pass. The \
         GEMM is unchanged — same product, different destination. Not a scheduling change (that \
         was torch:4's NO_VERDICT arm) and not a blocking change."
    );

    for n in [256usize, 512, 1024] {
        let a: Vec<f64> = (0..n * n)
            .map(|idx| {
                let (i, j) = (idx / n, idx % n);
                (((i * 31 + j * 17) % 97) as f64 - 48.0) / 24.0
            })
            .collect();
        let (packed, tau) = ft_kernel_cpu::geqrf_blocked_f64(&a, n, n);
        let c0: Vec<f64> = (0..n * n).map(|i| ((i as f64) * 0.019).cos()).collect();

        let mut hits = [0u64; 3];
        let times = interleaved::run(3, reps, 2, |i| {
            ft_kernel_cpu::set_ormqr_fused_subtract(i == 1);
            let _ = ft_kernel_cpu::ormqr_fused_subtract_hits_take();
            let mut c = c0.clone();
            let started = std::time::Instant::now();
            ft_kernel_cpu::ormqr_blocked_f64(&packed, &tau, n, n, n, &mut c, n, n, true, false);
            let ms = started.elapsed().as_secs_f64() * 1_000.0;
            std::hint::black_box(&c);
            hits[i] += ft_kernel_cpu::ormqr_fused_subtract_hits_take();
            ms
        });
        ft_kernel_cpu::set_ormqr_fused_subtract(false);

        let (lo, hi, gate) = interleaved::spread(&times[0]);
        println!("\nn={n}   panels={}", n.div_ceil(32));
        println!(
            "  HITS  shipped {} / fused {} / A/A {}   <- the fused arm MUST be non-zero and the \
             others zero, or the arms are not what they are labelled",
            hits[0], hits[1], hits[2]
        );
        if hits[1] == 0 {
            println!("  FUSED ARM NEVER FIRED — this table measures the shipped path twice. STOP.");
            continue;
        }
        if hits[0] != 0 || hits[2] != 0 {
            println!("  A SHIPPED ARM TOOK THE FUSED PATH — the toggle leaked. STOP.");
            continue;
        }
        println!(
            "  incumbent (materialised) WITHIN-RUN spread {lo:.4}-{hi:.4} ms = {gate:.3}x \
             (IQR {:.3}x) — an effect at or below {:.4} is UNRESOLVED",
            interleaved::iqr_ratio(&times[0]),
            gate - 1.0
        );
        println!("  {:>14} {:>10} {}", "arm", "median", interleaved::Verdict::header());
        for i in 0..3 {
            let arm = match i {
                0 => "materialised",
                1 => "fused",
                _ => "A/A",
            };
            let trust = if i == 0 {
                format!("{:>9} {:>9} {:>8} {:>8}  incumbent", "-", "-", "-", "-")
            } else {
                interleaved::verdict(&times[0], &times[i], gate).row()
            };
            println!("  {arm:>14} {:>10.4} {trust}", interleaved::median(&times[i]));
        }
    }
    ft_kernel_cpu::set_ormqr_fused_subtract(false);
    println!(
        "\nREADING: a PAIRED ratio above 1.0 means the fused arm is faster. This is a KERNEL \
         result; ledger 291 has a bit-exact 1.28-1.40x isolation win that moved no lane and 292i \
         one that inverted, so an all-cells win here still owes a paired lane certification on the \
         ormqr h2h lane before any default moves."
    );
}

//! `frankentorch-stale-tuning-constants-lzku6` lane 4 — HOUSEHOLDER_PANEL_WIDTH.
//!
//! # THE MECHANISM MODEL, AND WHY THE LANE AS WRITTEN IS MALFORMED
//!
//! The bead opened this lane as "a width chosen for the old routing is a width chosen for deleted
//! code" — i.e. re-sweep the width. **That framing does not survive reading the constant.**
//!
//! `HOUSEHOLDER_PANEL_WIDTH` is not a panel width. It is one term of an EXACT-MATCH dispatch
//! predicate (`gemm::should_parallelize_householder_skinny`):
//!
//! ```text
//! admitted =  n >= 512
//!          && ( (m == WIDTH && k >= 512) || (k == WIDTH && m >= 512) )
//!          && threads > 1
//!          && m*k*n >= 2^23
//! ```
//!
//! and the panel width the Householder ops actually use is hard-coded **32 in three separate
//! places** that never consult this constant: `geqrf_blocked_f64`'s ladder comment, `let nb_block
//! = 32` in `orgqr_blocked_f64`, and `householder_panels_from_packed_f64(.., 32)` in
//! `ormqr_blocked_f64`.
//!
//! **So no value of this constant changes any panel.** Setting it to 48 does not make a 48-wide
//! panel; it makes `m == WIDTH` stop matching a panel that is still 32 wide, which silently
//! DISABLES the skinny route. There is no optimum to find along this axis — the constant has
//! exactly two states, "equals the ops' 32" and "does not". A width ladder here would have
//! measured the route being turned off, at five different values, and reported the shape of
//! nothing.
//!
//! **THE REAL QUESTION IS THEREFORE A ROUTE A/B, NOT A WIDTH SWEEP:** does the skinny column-split
//! earn its place at all? That is what this file measures.
//!
//! # Why the prior panel refutations do not apply
//!
//! 291/291a (eigh backtransform column-split 0.925x and row-split 0.947x) and 292b (both cholesky
//! panel formulations refuted) were attempts to PARALLELISE A PANEL COMPUTATION, and both failed
//! because the panel is chain-limited — the work per fork is the same order as the fork. This is
//! not that. Nothing here is a panel computation: it is a dispatch choice between two ways of
//! running an already-parallel GEMM, on shapes (`32 x m x n`, `m x 32 x n`) whose N axis is 512 or
//! wider. The failure mode of those refutations — too little work per task — is not available to a
//! GEMM with >= 2^23 FMAs. Their evidence neither supports nor refutes this route.
//!
//! # THE CENSUS COMES FIRST, AND IT IS NOT A TIMING
//!
//! Lane 2 of this bead died to a census: neither slogdet nor inv executed `lu_solve` at all, so
//! re-tuning its NB would have measured nothing. The same question has to be asked here, and the
//! geqrf sweep has already hinted at the answer — its `adm/rej` column read `0/165`, `0/206`,
//! `0/155` at every size, i.e. **geqrf consults this predicate constantly and is never admitted.**
//!
//! The census is a COUNT, not a duration, so it is valid under any host load and runs first,
//! unguarded. Only the A/B below needs a window.
//!
//! # The A/B
//!
//! Arms are the route LIVE (`width` unset, so the shipped 32, which matches the ops' 32) against
//! the route DEAD (`width` set to a prime larger than any dimension here, so `m == WIDTH` and
//! `k == WIDTH` are both unsatisfiable and every call falls through to the established dispatch).
//! That is an in-process, interleavable toggle; the `FT_HOUSEHOLDER_SKINNY_GEMM` env kill switch
//! is a `OnceLock` and could only have given a cross-process A/B, which
//! `feedback_tuning_grid_missing_the_winner` warns against.
//!
//! Interleaved per rep with the order reversed on odd reps, per-rep min, median of per-rep paired
//! ratios, both estimators, exact sign test, incumbent within-run spread, and an A/A arm — ledger
//! 293/293a, via `interleaved`.
//!
//!   cargo run --release -p frankentorch-kernel-cpu --example householder_skinny_route_ab -- [reps]

mod interleaved;

/// A width no GEMM dimension here can equal, so the predicate's exact-match terms are both
/// unsatisfiable and the skinny route is dead without touching any other dispatch decision.
const ROUTE_OFF_WIDTH: usize = 9973;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let reps = interleaved::reps_for(args.get(1).and_then(|v| v.parse().ok()).unwrap_or(12));

    interleaved::banner("householder skinny route (LIVE vs DEAD)", reps);
    println!(
        "MECHANISM: HOUSEHOLDER_PANEL_WIDTH is an EXACT-MATCH dispatch term, not a panel width — \
         geqrf/orgqr/ormqr each hard-code 32 independently. No value of it changes a panel; it \
         only decides whether the skinny column-split route is reachable. So this is a ROUTE A/B."
    );
    println!(
        "SHIPPED width reads back as {} (route LIVE); the dead arm uses {ROUTE_OFF_WIDTH}, which \
         no dimension here can equal.",
        ft_kernel_cpu::householder_panel_width()
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

        // ---- CENSUS (counts, not timings: valid under any load) --------------------------
        println!("\nn={n}  CENSUS — which ops the skinny predicate ADMITS (adm/rej per call)");
        ft_kernel_cpu::set_householder_panel_width(0);

        let _ = ft_kernel_cpu::householder_skinny_census_take();
        let _ = ft_kernel_cpu::geqrf_blocked_f64(&a, n, n);
        let (g_adm, g_rej) = ft_kernel_cpu::householder_skinny_census_take();

        let _ = ft_kernel_cpu::orgqr_blocked_f64(&packed, &tau, n, n, n);
        let (o_adm, o_rej) = ft_kernel_cpu::householder_skinny_census_take();

        let mut c = c0.clone();
        ft_kernel_cpu::ormqr_blocked_f64(&packed, &tau, n, n, n, &mut c, n, n, true, false);
        let (r_adm, r_rej) = ft_kernel_cpu::householder_skinny_census_take();

        println!("  geqrf  {g_adm:>6} admitted / {g_rej:>6} rejected");
        println!("  orgqr  {o_adm:>6} admitted / {o_rej:>6} rejected");
        println!("  ormqr  {r_adm:>6} admitted / {r_rej:>6} rejected");
        if g_adm + o_adm + r_adm == 0 {
            println!(
                "  NO OP ADMITS AT THIS SIZE — the constant governs nothing here, and no timing \
                 below can be attributed to it. This is lane 2's finding in a different family."
            );
            continue;
        }

        // ---- A/B on the ops that admit ---------------------------------------------------
        // arm 0 = route LIVE (shipped), arm 1 = route DEAD, arm 2 = A/A duplicate of LIVE.
        let label = |i: usize| match i {
            0 => "route LIVE".to_owned(),
            1 => "route DEAD".to_owned(),
            _ => "A/A".to_owned(),
        };
        for (op, admits) in [("orgqr", o_adm > 0), ("ormqr", r_adm > 0)] {
            if !admits {
                println!("\n  {op}: 0 admitted — skipped, nothing to A/B");
                continue;
            }
            let times = interleaved::run(3, reps, 2, |i| {
                ft_kernel_cpu::set_householder_panel_width(if i == 1 { ROUTE_OFF_WIDTH } else { 0 });
                let started = std::time::Instant::now();
                if op == "orgqr" {
                    let q = ft_kernel_cpu::orgqr_blocked_f64(&packed, &tau, n, n, n);
                    let ms = started.elapsed().as_secs_f64() * 1_000.0;
                    std::hint::black_box(&q);
                    ms
                } else {
                    let mut c = c0.clone();
                    ft_kernel_cpu::ormqr_blocked_f64(
                        &packed, &tau, n, n, n, &mut c, n, n, true, false,
                    );
                    let ms = started.elapsed().as_secs_f64() * 1_000.0;
                    std::hint::black_box(&c);
                    ms
                }
            });
            ft_kernel_cpu::set_householder_panel_width(0);

            let (lo, hi, gate) = interleaved::spread(&times[0]);
            println!("\n  {op}  n={n}");
            println!(
                "    incumbent (route LIVE) WITHIN-RUN spread {lo:.4}-{hi:.4} ms = {gate:.3}x \
                 (IQR {:.3}x) — an effect at or below {:.4} is UNRESOLVED",
                interleaved::iqr_ratio(&times[0]),
                gate - 1.0
            );
            println!("    {:>12} {:>10} {}", "arm", "median", interleaved::Verdict::header());
            for i in 0..3 {
                let trust = if i == 0 {
                    format!("{:>9} {:>9} {:>8} {:>8}  incumbent", "-", "-", "-", "-")
                } else {
                    interleaved::verdict(&times[0], &times[i], gate).row()
                };
                println!(
                    "    {:>12} {:>10.4} {trust}",
                    label(i),
                    interleaved::median(&times[i])
                );
            }
        }
    }
    ft_kernel_cpu::set_householder_panel_width(0);
    println!(
        "\nREADING. A PAIRED ratio ABOVE 1.0 means the route LIVE arm is FASTER, i.e. the skinny \
         split earns its place. Below 1.0 means the shipped dispatch is paying for a route that \
         costs it. Either way the constant is not a width and must not be swept as one; the only \
         defensible change here is keeping 32 (coupled to the ops' hard-coded 32) or removing the \
         route. And a kernel result is not a lane result — ledger 291 and 292i both."
    );
}

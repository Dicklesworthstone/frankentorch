//! Where does `slogdet` spend its time, on ONE estimator? — `frankentorch-x6wc3`.
//!
//! WHY THIS EXISTS, AND IT IS A CORRECTION TO MY OWN POSTED NUMBER. The h2h harness prints the LU
//! phase map from "one instrumented call" while the lane figure beside it is a MIN over 16 rounds.
//! I divided the first by the second and reported "slogdet is 64-78% LU". That mixes estimators
//! exactly the way NEGATIVE_EVIDENCE items 266 and 270 warn about, and the direction of the error
//! is not even obvious: a single call is usually SLOWER than a min-of-16, which would make the
//! true LU share LOWER than I quoted, but it depends on how cold that call is.
//!
//! Here every frame is min-of-N on the same host in the same process, so the split is a
//! decomposition rather than a quotient of two different statistics:
//!
//!   slogdet_contiguous_f64   the whole op
//!   lu_factor_contiguous_f64 the LU alone — `slogdet` is documented as returning sign and
//!                            logabsdet "from a single LU", so this is the shared frame
//!   difference               the slogdet-SPECIFIC residue: the sign/log-product pass
//!
//! plus `lu_stage_take_ns()` accumulated over the SAME calls, so the panel/solve/trailing split is
//! on the same population too.
//!
//! THE OVERLAP THIS DECIDES. `frankentorch-e1isq` owns the getrf panel. If slogdet is nearly all
//! LU, this bead has almost nothing of its own and should be handed over rather than worked in
//! parallel; if the residue is substantial, that residue is the only part that is mine.
//!
//! ARM-INTERNAL: no incumbent, no ratio, no drift gate, no PyTorch — runs on any rch worker.
//! Everything goes to STDERR so a remote runner returns it.

use ft_core::{DType, Device, TensorMeta};
use std::time::Instant;

fn main() {
    let reps: usize = std::env::var("FT_REPS")
        .ok()
        .and_then(|t| t.trim().parse().ok())
        .unwrap_or(9);
    // SWEEP n — `frankentorch-x6wc3`. Two phase maps of the same op disagree by ~10x on `solve`:
    // `frankentorch-e1isq` measured panel 79-82% / solve 2.4-2.8% / trailing 4.6-10.8% at
    // n=512 and n=1024, while this probe reads panel 36-38% / solve 22-25% / trailing 15% at
    // n=256. Two explanations, and they need different responses:
    //
    //   * SIZE — the panel is O(n^2) work per step against an O(n^3) trailing update, so its
    //     share genuinely falls as n falls, and e1isq's target is simply not the binding frame
    //     at n=256;
    //   * STALENESS — e1isq's numbers may predate `78cf5eea`, which its own last comment says
    //     shipped the recursive `lu_factor_panel_recursive_f64` it was asking for. If so the
    //     79-82% is a PRE-FIX profile and the panel share has already collapsed.
    //
    // A single sweep on one estimator distinguishes them: if the shares move smoothly with n and
    // reach 79-82% by n=1024, it is size; if the panel stays low at every n, e1isq's premise is
    // stale and the binding frame has moved for the whole LU family.
    let sizes: Vec<usize> = std::env::var("FT_SIZES")
        .unwrap_or_else(|_| "128,256,512,1024".to_string())
        .split(',')
        .filter_map(|t| t.trim().parse().ok())
        .collect();
    for n in sizes {
        if std::env::var("FT_AB").is_ok_and(|v| v.trim() == "1") {
            run_ab(n, reps);
        } else {
            run_one(n, reps);
        }
    }
}

fn run_one(n: usize, reps: usize) {

    // Diagonally dominant so the factorisation is well conditioned and pivoting does not dominate
    // by accident — the point is to time the common path, not a pathological one.
    let data: Vec<f64> = (0..n * n)
        .map(|idx| {
            let i = idx / n;
            let j = idx % n;
            let v = ((i * 31 + j * 17) % 23) as f64 * 0.05 - 0.5;
            if i == j { v + (n as f64) } else { v }
        })
        .collect();
    let meta = TensorMeta::from_shape(vec![n, n], DType::F64, Device::Cpu);

    eprintln!(
        "SLOGDET n={n} reps={reps} threads={} (arms adjacent inside each rep; min after \
         discarding the first)",
        rayon::current_num_threads()
    );

    let mut slog = f64::INFINITY;
    let mut lu = f64::INFINITY;
    let mut panel_ns = 0u64;
    let mut solve_ns = 0u64;
    let mut trail_ns = 0u64;
    let mut counted = 0u64;

    for rep in 0..reps {
        let start = Instant::now();
        let s = ft_kernel_cpu::slogdet_contiguous_f64(&data, &meta).expect("slogdet");
        let t_slog = start.elapsed().as_secs_f64() * 1e3;
        std::hint::black_box(&s);

        // Drain whatever the slogdet call left, then attribute only the LU call's own phases.
        let _ = ft_kernel_cpu::lu_stage_take_ns();
        let start = Instant::now();
        let l = ft_kernel_cpu::lu_factor_contiguous_f64(&data, &meta).expect("lu_factor");
        let t_lu = start.elapsed().as_secs_f64() * 1e3;
        std::hint::black_box(&l);
        let (p, s_ns, tr) = ft_kernel_cpu::lu_stage_take_ns();

        if rep > 0 {
            slog = slog.min(t_slog);
            lu = lu.min(t_lu);
            panel_ns += p;
            solve_ns += s_ns;
            trail_ns += tr;
            counted += 1;
        }
    }

    let ms = |v: u64| v as f64 / 1e6 / counted.max(1) as f64;
    let residue = slog - lu;
    eprintln!("SLOGDET slogdet_contiguous_f64   {slog:8.3} ms");
    eprintln!("SLOGDET lu_factor_contiguous_f64 {lu:8.3} ms   ({:.0}% of slogdet)", 100.0 * lu / slog);
    eprintln!(
        "SLOGDET slogdet-SPECIFIC residue {residue:8.3} ms   ({:.0}% of slogdet)",
        100.0 * residue / slog
    );
    eprintln!(
        "SLOGDET   LU phases, mean over the SAME calls: panel {:.3} solve {:.3} trailing {:.3} ms",
        ms(panel_ns),
        ms(solve_ns),
        ms(trail_ns)
    );
    eprintln!(
        "SLOGDET   panel is {:.0}% of the LU and {:.0}% of the whole op — that share is \
         frankentorch-e1isq's territory.",
        100.0 * ms(panel_ns) / lu,
        100.0 * ms(panel_ns) / slog
    );
}

/// Paired A/B for the row-accumulating triangular solve — `frankentorch-x6wc3`.
///
/// The lever: `gemm::ltrsm_unit_lower_panel_f64`'s parallel arm read-modify-writes `lu[i*n + j]`
/// once per `t`, so row `i` is loaded and stored `i - k0` times per column block while doing one
/// FMA per element. `LU_TRSM_ROW_ACCUM` holds that running total in a local buffer and stores it
/// once. Bit-exactness is pinned by `lu_trsm_row_accum_matches_the_shipped_solve_bitwise`, so this
/// is purely a question of whether halving the traffic on a 0.125 flop/byte loop moves the lane.
///
/// SHAPE OF THE MEASUREMENT, and it is deliberately paranoid because I have been burned:
///
///   * BOTH arms are in ONE process on ONE host, alternating OFF/ON/ON/OFF inside each rep. A
///     balanced square cancels any monotone drift across the rep to first order — with plain
///     A,B,A,B a host warming up or cooling down over the run leaks straight into the ratio.
///   * The FIRST rep is discarded outright. NEGATIVE_EVIDENCE has a standing note that a sweep's
///     first pass runs ~1.23x slower for reasons that are not contention, and including it
///     flatters whichever arm happens to go second.
///   * The A/A NULL is the two OFF positions against each other. They are the SAME code at
///     different points in the rep, so their ratio is a direct read of what this harness can
///     resolve; if the null is outside 0.97-1.03 the arm ratio beside it means nothing and the
///     row must be thrown away rather than argued with.
///   * `lu_stage_take_ns` is drained per call, so the solve column says whether the lever moved
///     the frame it TARGETS even when the lane total does not move. Those are different claims
///     and conflating them is how a null gets reported as a win.
fn run_ab(n: usize, reps: usize) {
    let data: Vec<f64> = (0..n * n)
        .map(|idx| {
            let i = idx / n;
            let j = idx % n;
            let v = ((i * 31 + j * 17) % 23) as f64 * 0.05 - 0.5;
            if i == j { v + (n as f64) } else { v }
        })
        .collect();
    let meta = TensorMeta::from_shape(vec![n, n], DType::F64, Device::Cpu);

    // One timed call at a given setting, returning the wall time and the solve frame it reported.
    let once = |on: bool| -> (f64, u64) {
        let previous = ft_kernel_cpu::set_lu_trsm_row_accum(on);
        let _ = ft_kernel_cpu::lu_stage_take_ns();
        let start = Instant::now();
        let l = ft_kernel_cpu::lu_factor_contiguous_f64(&data, &meta).expect("lu_factor");
        let ms = start.elapsed().as_secs_f64() * 1e3;
        std::hint::black_box(&l);
        let (_, solve_ns, _) = ft_kernel_cpu::lu_stage_take_ns();
        ft_kernel_cpu::set_lu_trsm_row_accum(previous);
        (ms, solve_ns)
    };

    // Four positions per rep: OFF ON ON OFF.
    let mut best = [f64::INFINITY; 4];
    let mut solve = [f64::INFINITY; 4];
    for rep in 0..reps {
        let r = [once(false), once(true), once(true), once(false)];
        if rep == 0 {
            continue;
        }
        for (slot, (ms, s_ns)) in r.iter().enumerate() {
            best[slot] = best[slot].min(*ms);
            solve[slot] = solve[slot].min(*s_ns as f64 / 1e6);
        }
    }

    let off = best[0].min(best[3]);
    let on = best[1].min(best[2]);
    let off_solve = solve[0].min(solve[3]);
    let on_solve = solve[1].min(solve[2]);
    let null = best[0] / best[3];

    eprintln!(
        "TRSM_AB n={n} reps={reps} threads={} (OFF ON ON OFF per rep, first rep discarded)",
        rayon::current_num_threads()
    );
    eprintln!("TRSM_AB   lu OFF (shipped)  {off:8.3} ms   solve frame {off_solve:7.3} ms");
    eprintln!("TRSM_AB   lu ON  (row-acc)  {on:8.3} ms   solve frame {on_solve:7.3} ms");
    eprintln!(
        "TRSM_AB   LANE  {:.4}x   SOLVE FRAME {:.4}x   A/A null {null:.4} {}",
        off / on,
        off_solve / on_solve.max(f64::MIN_POSITIVE),
        if (0.97..=1.03).contains(&null) {
            "PASS"
        } else {
            "FAIL — discard this row"
        }
    );
}

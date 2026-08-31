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
    // CONFIG COMES FROM ARGV, NOT THE ENVIRONMENT, and that is not a style preference — it is
    // forced. This probe runs on an rch worker via `cargo run`, and rch does NOT forward the
    // caller's environment: a run asking for `RAYON_NUM_THREADS=8 FT_AB=1 FT_SIZES=512,1024`
    // came back at rayon=16 having quietly executed the DEFAULT decomposition instead, exit 0,
    // no warning. An env knob that silently reverts to its default across the boundary is a
    // measurement hazard of exactly the kind that gets a wrong row banked, so every setting the
    // row depends on — including the pool width — is an argument the binary echoes back.
    //
    //   slogdet_frame_probe [mode] [sizes] [reps] [threads]
    //   slogdet_frame_probe phase 512 15 16
    let argv: Vec<String> = std::env::args().skip(1).collect();
    let arg = |i: usize| argv.get(i).map(String::as_str).unwrap_or("");
    let ab = arg(0) == "ab";
    let reps: usize = arg(2).parse().unwrap_or(9);
    let threads: usize = arg(3).parse().unwrap_or(0);
    if threads > 0 {
        rayon::ThreadPoolBuilder::new()
            .num_threads(threads)
            .build_global()
            .expect("rayon pool width");
    }
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
    // PROVENANCE, printed by the binary itself. This runs under `cargo run` on an rch worker, so
    // nothing outside the process knows which machine served the job — and a row without a machine
    // on it cannot be compared with any other row (the same cell has read 1.2693x and 0.0093x on
    // two workers with both nulls passing). The loadavg matters for the same reason: these workers
    // are shared, and a lane timed next to somebody's build is not a lane.
    eprintln!(
        "PROV host={} nproc={} rayon={} mode={} reps={reps} loadavg={}",
        std::fs::read_to_string("/etc/hostname").unwrap_or_default().trim(),
        std::thread::available_parallelism().map_or(0, std::num::NonZero::get),
        rayon::current_num_threads(),
        if ab { "ab" } else { "decompose" },
        std::fs::read_to_string("/proc/loadavg").unwrap_or_default().trim(),
    );

    let spec = if arg(1).is_empty() { "128,256,512,1024" } else { arg(1) };
    let sizes: Vec<usize> = spec.split(',').filter_map(|t| t.trim().parse().ok()).collect();
    for n in sizes {
        match arg(0) {
            "ab" => run_ab(n, reps),
            "inv" => run_inv(n, reps),
            "permab" => run_perm_ab(n, reps),
            "structab" => run_struct_ab(n, reps),
            "phase" => run_phase_split(n, reps),
            _ => run_one(n, reps),
        }
    }
}

/// Attribute one coherent `slogdet` call: all counters and the wall estimator come from the
/// same winning rep. This is deliberately distinct from `run_one`, which compares a separate
/// direct-LU arm to the full operation and therefore cannot truthfully use its phase counters
/// for the `slogdet` wall-time closure.
fn run_phase_split(n: usize, reps: usize) {
    let data: Vec<f64> = (0..n * n)
        .map(|idx| {
            let i = idx / n;
            let j = idx % n;
            let v = ((i * 31 + j * 17) % 23) as f64 * 0.05 - 0.5;
            if i == j { v + n as f64 } else { v }
        })
        .collect();
    let meta = TensorMeta::from_shape(vec![n, n], DType::F64, Device::Cpu);

    for _ in 0..2 {
        let value = ft_kernel_cpu::slogdet_contiguous_f64(&data, &meta).expect("warmup");
        std::hint::black_box(value);
    }

    let mut best = (u64::MAX, 0u64, 0u64, 0u64);
    for _ in 0..reps {
        let _ = ft_kernel_cpu::lu_stage_take_ns();
        let started = Instant::now();
        let value = ft_kernel_cpu::slogdet_contiguous_f64(&data, &meta).expect("slogdet");
        let wall = started.elapsed().as_nanos() as u64;
        std::hint::black_box(value);
        let (panel, solve, trailing) = ft_kernel_cpu::lu_stage_take_ns();
        if wall < best.0 {
            best = (wall, panel, solve, trailing);
        }
    }

    let (wall, panel, solve, trailing) = best;
    let ms = |v: u64| v as f64 / 1e6;
    let pct = |v: u64| 100.0 * v as f64 / wall.max(1) as f64;
    let accounted = panel.saturating_add(solve).saturating_add(trailing);
    let glue = wall.saturating_sub(accounted);

    eprintln!("SLOGDET PHASE SPLIT n={n}, min-wall rep of {reps}");
    eprintln!("  {:<28} {:>9} {:>8}", "phase", "ms", "% lane");
    eprintln!("  {:<28} {:>9.4} {:>7.1}%", "LU panel", ms(panel), pct(panel));
    eprintln!("  {:<28} {:>9.4} {:>7.1}%", "LU solve", ms(solve), pct(solve));
    eprintln!(
        "  {:<28} {:>9.4} {:>7.1}%",
        "LU trailing update",
        ms(trailing),
        pct(trailing)
    );
    eprintln!(
        "  {:<28} {:>9.4} {:>7.1}%   <- residual, not a measured phase",
        "glue (residual)",
        ms(glue),
        pct(glue)
    );
    eprintln!("  {:<28} {:>9.4} {:>7.1}%", "TOTAL (wall)", ms(wall), 100.0);
    eprintln!(
        "ACCOUNTING CLOSURE: LU stages {:.4} ms of {:.4} ms wall = {:.1}%. The residual is \
         shown as glue; the panel counter includes the terminal panel even when `tcols == 0`.",
        ms(accounted),
        ms(wall),
        100.0 * accounted as f64 / wall.max(1) as f64
    );
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

    // PER-REP PAIRING, and the first shape of this harness had to be thrown away to get here.
    //
    // v1 took a min over each POSITION across all reps, then compared position-minima. Its A/A
    // null — the two OFF positions against each other — came back 1.3332 at n=512 and 0.8134 at
    // n=1024, i.e. the same code at two points in the rep differed by 33% and 19%. That voids
    // any arm ratio printed beside it, and it voids it in BOTH directions: the null landed above
    // 1 at one size and below 1 at the other, so this is not a fixed position bias that could be
    // subtracted out. A min-over-positions estimator also SELECTS the luckiest sample per
    // position, which makes more reps look better rather than truer.
    //
    // So: pair inside the rep. Each rep contributes min-of-2 for OFF and min-of-2 for ON drawn
    // from the SAME rep, the ratio is formed per rep, and the reported figure is the MEDIAN of
    // those ratios. A rep that lands next to somebody else's build inflates both arms and mostly
    // cancels; a median then refuses to be moved by the one rep where it does not.
    let mut off_lane = Vec::new();
    let mut on_lane = Vec::new();
    let mut off_solve_v = Vec::new();
    let mut on_solve_v = Vec::new();
    let mut nulls = Vec::new();
    for rep in 0..reps {
        let r = [once(false), once(true), once(true), once(false)];
        if rep == 0 {
            continue;
        }
        off_lane.push(r[0].0.min(r[3].0));
        on_lane.push(r[1].0.min(r[2].0));
        off_solve_v.push((r[0].1.min(r[3].1)) as f64 / 1e6);
        on_solve_v.push((r[1].1.min(r[2].1)) as f64 / 1e6);
        // The null is the two OFF positions of THIS rep — same code, same rep, so it measures
        // exactly what this harness cannot resolve.
        nulls.push(r[0].0 / r[3].0);
    }

    let median = |v: &mut Vec<f64>| -> f64 {
        v.sort_by(f64::total_cmp);
        if v.is_empty() { f64::NAN } else { v[v.len() / 2] }
    };
    let ratios = |a: &[f64], b: &[f64]| -> f64 {
        let mut r: Vec<f64> = a.iter().zip(b).map(|(x, y)| x / y).collect();
        median(&mut r)
    };
    let lane_ratio = ratios(&off_lane, &on_lane);
    let solve_ratio = ratios(&off_solve_v, &on_solve_v);
    let null = median(&mut nulls.clone());
    let off = median(&mut off_lane.clone());
    let on = median(&mut on_lane.clone());
    let off_solve = median(&mut off_solve_v.clone());
    let on_solve = median(&mut on_solve_v.clone());

    eprintln!(
        "TRSM_AB n={n} reps={reps} threads={} (OFF ON ON OFF per rep, first rep discarded)",
        rayon::current_num_threads()
    );
    eprintln!("TRSM_AB   lu OFF (shipped)  {off:8.3} ms   solve frame {off_solve:7.3} ms");
    eprintln!("TRSM_AB   lu ON  (row-acc)  {on:8.3} ms   solve frame {on_solve:7.3} ms");
    eprintln!(
        "TRSM_AB   LANE  {lane_ratio:.4}x   SOLVE FRAME {solve_ratio:.4}x   A/A null {null:.4} {}",
        if (0.97..=1.03).contains(&null) {
            "PASS"
        } else {
            "FAIL — discard this row"
        }
    );
}

/// Where does `inv` spend its time? — `frankentorch-37sxo` (inv 3.99x slower than torch, f64 n=256).
///
/// The bead quotes a phase split of panel 46% / solve 37% / trailing 12%, but those are shares of
/// the LU, and `inv` is NOT mostly LU the way `slogdet` turned out to be. Reading the kernel,
/// `inv_tensor_contiguous_f64` is `lu_factor_contiguous_f64` followed by
/// `lu_solve_contiguous_f64` against a DENSE n x n identity — and that second call is the larger
/// piece by FLOP count: the factorisation is ~2n^3/3 while a forward plus a back substitution
/// with n right-hand sides is ~2n^3. If that holds, three quarters of this op sits OUTSIDE both
/// `frankentorch-e1isq`'s panel and the getrf trsm frame I just measured a ~1.3x ceiling on, and
/// the bead's own scoping note ("this bead scopes to inv-specific residue") points here.
///
/// I am not taking the FLOP count's word for it. The blocked multi-RHS path (`num_rhs >= 64 &&
/// n >= 256`, `frankentorch-kgs4.63`) makes the solve compute-bound and cache-blocked, so its
/// measured share need not track its arithmetic at all. Hence this decomposition, every frame on
/// ONE estimator in ONE process:
///
///   inv_tensor_contiguous_f64   the whole op
///   lu_factor_contiguous_f64    the shared factorisation (e1isq's panel lives in here)
///   lu_solve_contiguous_f64     the SAME identity solve `inv` performs, timed on its own
///
/// The three are timed adjacently inside each rep and reported as per-rep medians, so the split is
/// a decomposition rather than a quotient of statistics taken from different populations — which
/// is the error that produced my retracted "slogdet is 64-78% LU".
fn run_inv(n: usize, reps: usize) {
    let data: Vec<f64> = (0..n * n)
        .map(|idx| {
            let i = idx / n;
            let j = idx % n;
            let v = ((i * 31 + j * 17) % 23) as f64 * 0.05 - 0.5;
            if i == j { v + (n as f64) } else { v }
        })
        .collect();
    let meta = TensorMeta::from_shape(vec![n, n], DType::F64, Device::Cpu);

    let mut identity = vec![0.0f64; n * n];
    for i in 0..n {
        identity[i * n + i] = 1.0;
    }

    let mut inv_v = Vec::new();
    let mut fac_v = Vec::new();
    let mut sol_v = Vec::new();
    let mut fwd_v = Vec::new();
    let mut back_v = Vec::new();
    let mut perm_v = Vec::new();

    for rep in 0..reps {
        let start = Instant::now();
        let a = ft_kernel_cpu::inv_tensor_contiguous_f64(&data, &meta).expect("inv");
        let t_inv = start.elapsed().as_secs_f64() * 1e3;
        std::hint::black_box(&a);

        let start = Instant::now();
        let factor = ft_kernel_cpu::lu_factor_contiguous_f64(&data, &meta).expect("lu_factor");
        let t_fac = start.elapsed().as_secs_f64() * 1e3;

        // The identity solve exactly as `inv` issues it, so this frame is the real one and not a
        // proxy for it.
        // Drain BOTH counter families. Draining only the forward/back pair (as this first did)
        // leaves the permutation counter holding `inv`'s internal solve as well as the standalone
        // one, double-counting the frame — it printed 47% of the solve at n=1024 and drove the
        // "solve residue" to -22.6%, which is how the bug announced itself.
        let _ = ft_kernel_cpu::lu_solve_half_take_ns();
        let _ = ft_kernel_cpu::lu_solve_perm_take_ns();
        let start = Instant::now();
        let x = ft_kernel_cpu::lu_solve_contiguous_f64(&factor, &identity, &meta).expect("lu_solve");
        let t_sol = start.elapsed().as_secs_f64() * 1e3;
        let (fwd_ns, back_ns) = ft_kernel_cpu::lu_solve_half_take_ns();
        let perm_ns = ft_kernel_cpu::lu_solve_perm_take_ns();
        std::hint::black_box(&x);
        std::hint::black_box(&factor);

        if rep > 0 {
            inv_v.push(t_inv);
            fac_v.push(t_fac);
            sol_v.push(t_sol);
            fwd_v.push(fwd_ns as f64 / 1e6);
            back_v.push(back_ns as f64 / 1e6);
            perm_v.push(perm_ns as f64 / 1e6);
        }
    }

    let median = |v: &mut Vec<f64>| -> f64 {
        v.sort_by(f64::total_cmp);
        if v.is_empty() { f64::NAN } else { v[v.len() / 2] }
    };
    let inv = median(&mut inv_v);
    let fac = median(&mut fac_v);
    let sol = median(&mut sol_v);

    eprintln!("INV n={n} reps={reps} threads={}", rayon::current_num_threads());
    eprintln!("INV   inv_tensor_contiguous_f64  {inv:8.3} ms");
    eprintln!("INV   lu_factor_contiguous_f64   {fac:8.3} ms  ({:5.1}% of inv)  <- e1isq's panel lives here", 100.0 * fac / inv);
    eprintln!("INV   lu_solve (identity RHS)    {sol:8.3} ms  ({:5.1}% of inv)  <- inv-specific residue", 100.0 * sol / inv);
    let fwd = median(&mut fwd_v);
    let back = median(&mut back_v);
    eprintln!(
        "INV     forward L y = Pb        {fwd:8.3} ms  ({:5.1}% of the solve)  <- the ONLY half the identity-structure lever touches",
        100.0 * fwd / sol
    );
    eprintln!("INV     back    U x = y         {back:8.3} ms  ({:5.1}% of the solve)", 100.0 * back / sol);
    let perm = median(&mut perm_v);
    eprintln!(
        "INV     pivot permutation      {perm:8.3} ms  ({:5.1}% of the solve)  <- MEASURED, not inferred by subtraction",
        100.0 * perm / sol
    );
    eprintln!(
        "INV     solve residue          {:8.3} ms  ({:5.1}% of the solve)",
        sol - fwd - back - perm,
        100.0 * (sol - fwd - back - perm) / sol
    );
    eprintln!(
        "INV   accounted {:5.1}%   unattributed {:8.3} ms",
        100.0 * (fac + sol) / inv,
        inv - fac - sol
    );
}

/// Paired A/B for the row-wise RHS permutation — `frankentorch-37sxo`.
///
/// Same harness shape that caught the trsm lever's inverted sign: OFF/ON/ON/OFF inside each rep,
/// first rep discarded, per-rep min-of-2 per arm, MEDIAN of the per-rep ratios, and an A/A null
/// built from the two OFF positions of the SAME rep. A min-over-positions estimator printed a
/// confident 18% REGRESSION for a lever that shipped at 1.035-1.058x, so the shape is not
/// optional.
///
/// This times the WHOLE `inv` lane, not just `lu_solve`, because that is the figure the bead is
/// about — and it reports the permutation frame beside it so the lane movement can be checked
/// against the frame movement instead of taken on faith.
fn run_perm_ab(n: usize, reps: usize) {
    let data: Vec<f64> = (0..n * n)
        .map(|idx| {
            let i = idx / n;
            let j = idx % n;
            let v = ((i * 31 + j * 17) % 23) as f64 * 0.05 - 0.5;
            if i == j { v + (n as f64) } else { v }
        })
        .collect();
    let meta = TensorMeta::from_shape(vec![n, n], DType::F64, Device::Cpu);

    // Return the substitution halves alongside the permutation frame. THE DISPLACEMENT TEST: the
    // column gather first-touches the `x` buffer in a strided pattern, so if the row-wise form
    // merely DEFERS those cache misses into the forward substitution rather than removing the
    // work, `fwd + back` will grow by about what `perm` sheds and the lane will not move. That is
    // a different claim from "the lever does nothing", and only these counters separate them.
    let once = |on: bool| -> (f64, f64, f64) {
        let previous = ft_kernel_cpu::set_lu_solve_perm_rowwise(on);
        let _ = ft_kernel_cpu::lu_solve_perm_take_ns();
        let _ = ft_kernel_cpu::lu_solve_half_take_ns();
        let start = Instant::now();
        let a = ft_kernel_cpu::inv_tensor_contiguous_f64(&data, &meta).expect("inv");
        let ms = start.elapsed().as_secs_f64() * 1e3;
        std::hint::black_box(&a);
        let perm = ft_kernel_cpu::lu_solve_perm_take_ns() as f64 / 1e6;
        let (fwd, back) = ft_kernel_cpu::lu_solve_half_take_ns();
        ft_kernel_cpu::set_lu_solve_perm_rowwise(previous);
        (ms, perm, (fwd + back) as f64 / 1e6)
    };

    let mut off_v = Vec::new();
    let mut on_v = Vec::new();
    let mut off_p = Vec::new();
    let mut on_p = Vec::new();
    let mut off_h = Vec::new();
    let mut on_h = Vec::new();
    let mut nulls = Vec::new();
    for rep in 0..reps {
        // ALTERNATE THE SQUARE. A fixed OFF/ON/ON/OFF puts ON at the two MIDDLE positions every
        // rep, so any within-rep warming or cooling lands entirely on one arm — and the A/A null,
        // which compares the two OFF slots at positions 0 and 3, cannot see it because both of its
        // samples sit outside the middle. That is not hypothetical: the fixed square reported a
        // paired ratio of 1.0990x at n=1024 while the two marginal medians differed by 1.0127x.
        // Alternating gives each arm both the outer and the inner slots equally often.
        let r = if rep % 2 == 0 {
            let a = [once(false), once(true), once(true), once(false)];
            [a[0], a[1], a[2], a[3]]
        } else {
            let a = [once(true), once(false), once(false), once(true)];
            [a[1], a[0], a[3], a[2]]
        };
        if rep == 0 {
            continue;
        }
        // r[0], r[3] are the OFF samples of this rep and r[1], r[2] the ON ones, whichever slots
        // they actually occupied.
        off_v.push(r[0].0.min(r[3].0));
        on_v.push(r[1].0.min(r[2].0));
        off_p.push(r[0].1.min(r[3].1));
        on_p.push(r[1].1.min(r[2].1));
        off_h.push(r[0].2.min(r[3].2));
        on_h.push(r[1].2.min(r[2].2));
        nulls.push(r[0].0 / r[3].0);
    }

    let median = |v: &mut Vec<f64>| -> f64 {
        v.sort_by(f64::total_cmp);
        if v.is_empty() { f64::NAN } else { v[v.len() / 2] }
    };
    let ratios = |a: &[f64], b: &[f64]| -> f64 {
        let mut r: Vec<f64> = a.iter().zip(b).map(|(x, y)| x / y).collect();
        median(&mut r)
    };
    let lane = ratios(&off_v, &on_v);
    let perm_ratio = ratios(&off_p, &on_p);
    let null = median(&mut nulls.clone());

    eprintln!("PERM_AB n={n} reps={reps} threads={}", rayon::current_num_threads());
    eprintln!(
        "PERM_AB   inv OFF (column gather) {:8.3} ms   perm frame {:7.3} ms",
        median(&mut off_v.clone()),
        median(&mut off_p.clone())
    );
    eprintln!(
        "PERM_AB   inv ON  (row memcpy)    {:8.3} ms   perm frame {:7.3} ms",
        median(&mut on_v.clone()),
        median(&mut on_p.clone())
    );
    // A median of per-rep RATIOS and a ratio of the two MARGINAL medians answer different
    // questions, and at n=1024 they disagreed outright: 1.1230x against 101.192/101.190 = 1.0000.
    // The paired form is the one that respects the pairing, but a disagreement that large means
    // one of the two is being driven by the shape of the distribution rather than by the lever, so
    // print enough to tell which. The SIGN TEST is the robust arbiter — it asks only how often ON
    // beat OFF within a rep, and cares nothing for magnitudes or outliers.
    let wins = off_v.iter().zip(&on_v).filter(|(o, n)| n < o).count();
    let quart = |v: &[f64], q: f64| -> f64 {
        let mut w = v.to_vec();
        w.sort_by(f64::total_cmp);
        w[((w.len() - 1) as f64 * q) as usize]
    };
    eprintln!(
        "PERM_AB   marginal medians {:.3} / {:.3} = {:.4}x   paired-ratio median {lane:.4}x",
        median(&mut off_v.clone()),
        median(&mut on_v.clone()),
        median(&mut off_v.clone()) / median(&mut on_v.clone())
    );
    eprintln!(
        "PERM_AB   OFF p25/p50/p75 {:.3}/{:.3}/{:.3}   ON p25/p50/p75 {:.3}/{:.3}/{:.3}",
        quart(&off_v, 0.25), quart(&off_v, 0.5), quart(&off_v, 0.75),
        quart(&on_v, 0.25), quart(&on_v, 0.5), quart(&on_v, 0.75)
    );
    eprintln!(
        "PERM_AB   SIGN TEST: ON faster in {wins}/{} reps",
        off_v.len()
    );
    let off_half = median(&mut off_h.clone());
    let on_half = median(&mut on_h.clone());
    let off_perm = median(&mut off_p.clone());
    let on_perm = median(&mut on_p.clone());
    eprintln!(
        "PERM_AB   DISPLACEMENT: perm sheds {:.3} ms, fwd+back changes by {:+.3} ms  ({:.0}% of the shed cost reappears)",
        off_perm - on_perm,
        on_half - off_half,
        100.0 * (on_half - off_half) / (off_perm - on_perm).max(f64::MIN_POSITIVE)
    );
    eprintln!(
        "PERM_AB   fwd+back OFF {off_half:.3} ms   ON {on_half:.3} ms"
    );
    eprintln!(
        "PERM_AB   INV LANE {lane:.4}x   PERM FRAME {perm_ratio:.4}x   A/A null {null:.4} {}",
        if (0.97..=1.03).contains(&null) { "PASS" } else { "FAIL — discard this row" }
    );
}

/// Paired A/B for the identity-structure inverse — `frankentorch-37sxo`.
///
/// Same shape as the permutation A/B: alternating square, per-rep min-of-2, median of per-rep
/// ratios, marginal medians and a sign test printed alongside so a disagreement between the
/// estimators shows up as the harness defect it is rather than as a number to choose between.
fn run_struct_ab(n: usize, reps: usize) {
    let data: Vec<f64> = (0..n * n)
        .map(|idx| {
            let i = idx / n;
            let j = idx % n;
            let v = ((i * 31 + j * 17) % 23) as f64 * 0.05 - 0.5;
            if i == j { v + (n as f64) } else { v }
        })
        .collect();
    let meta = TensorMeta::from_shape(vec![n, n], DType::F64, Device::Cpu);

    let once = |on: bool| -> f64 {
        let previous = ft_kernel_cpu::set_lu_inv_identity_struct(on);
        let start = Instant::now();
        let a = ft_kernel_cpu::inv_tensor_contiguous_f64(&data, &meta).expect("inv");
        let ms = start.elapsed().as_secs_f64() * 1e3;
        std::hint::black_box(&a);
        ft_kernel_cpu::set_lu_inv_identity_struct(previous);
        ms
    };

    let mut off_v = Vec::new();
    let mut on_v = Vec::new();
    let mut nulls = Vec::new();
    for rep in 0..reps {
        let r = if rep % 2 == 0 {
            let a = [once(false), once(true), once(true), once(false)];
            [a[0], a[1], a[2], a[3]]
        } else {
            let a = [once(true), once(false), once(false), once(true)];
            [a[1], a[0], a[3], a[2]]
        };
        if rep == 0 {
            continue;
        }
        off_v.push(r[0].min(r[3]));
        on_v.push(r[1].min(r[2]));
        nulls.push(r[0] / r[3]);
    }

    let median = |v: &mut Vec<f64>| -> f64 {
        v.sort_by(f64::total_cmp);
        if v.is_empty() { f64::NAN } else { v[v.len() / 2] }
    };
    let mut r: Vec<f64> = off_v.iter().zip(&on_v).map(|(a, b)| a / b).collect();
    let lane = median(&mut r);
    let null = median(&mut nulls.clone());
    let wins = off_v.iter().zip(&on_v).filter(|(o, n)| n < o).count();

    eprintln!("STRUCT_AB n={n} reps={reps} threads={}", rayon::current_num_threads());
    eprintln!(
        "STRUCT_AB   inv OFF (dense identity) {:8.3} ms   ON (structured) {:8.3} ms",
        median(&mut off_v.clone()),
        median(&mut on_v.clone())
    );
    eprintln!(
        "STRUCT_AB   marginal {:.4}x   paired {lane:.4}x   SIGN TEST ON faster in {wins}/{} reps",
        median(&mut off_v.clone()) / median(&mut on_v.clone()),
        off_v.len()
    );
    eprintln!(
        "STRUCT_AB   INV LANE {lane:.4}x   A/A null {null:.4} {}",
        if (0.97..=1.03).contains(&null) { "PASS" } else { "FAIL — discard this row" }
    );
}

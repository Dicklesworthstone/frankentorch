//! Does the h2h board's `group_norm_f32` lane actually EXECUTE the narrow lever? —
//! `frankentorch-68pwz`.
//!
//! WHY THIS EXISTS. Item 103 landed `narrow_f64_to_f32` in four f32 norm backward closures
//! and priced it at 2.5 ms on the scored shape. Item 103b then measured the board's engine
//! term and declined to attribute its movement, because a peer's lever had landed in the
//! same term. This probe asks the question that comes BEFORE attribution and that neither
//! item asked: is the lever on the measured lane's path at all?
//!
//! There is a specific reason to doubt it. The f32 norm ops have TWO backward routes, and
//! the LOSS picks which one runs:
//!
//!   * a plain `tensor_sum` loss fires the GroupNorm sum-shortcut, whose backward takes a
//!     SCALAR upstream and never materializes a per-element `dy`;
//!   * any other loss takes the closure that does materialize one — the closure the narrow
//!     lever sits in.
//!
//! `timed_group_norm_f32` in `gauntlet_lane_sweep_h2h` ends in `tensor_sum`. So the board's
//! headline GroupNorm lane may never touch the lever, and reading the op's source cannot
//! settle it — this is exactly the case the campaign's sentinel rule was written for, after
//! source reading gave three confident wrong answers about which path executed.
//!
//! HOW IT ANSWERS. `ft_api::narrow_counts()` is incremented inside the helper itself, so a
//! non-zero count is execution, not inference. Both routes are run in one process against
//! the same op and shape, so the second is a positive control for the first: if the
//! sum-loss route reports zero while the non-sum route reports the full element count, the
//! zero is a real route split and not a broken counter.

use ft_api::FrankenTorchSession;
use ft_core::ExecutionMode;

/// The scored lane's shape, copied from `gauntlet_lane_sweep_h2h`.
const GN_N: usize = 32;
const GN_C: usize = 64;
const GN_H: usize = 56;
const GN_W: usize = 56;
const GN_GROUPS: usize = 32;

/// Build the scored leaf/weight/bias. A macro rather than a fn because `TensorNodeId` is
/// private to `ft_api`, so the tuple has no nameable type outside the crate.
macro_rules! build {
    ($session:expr) => {{
        let numel = GN_N * GN_C * GN_H * GN_W;
        let values: Vec<f32> = (0..numel)
            .map(|index| ((index % 9973) as f32) * 0.000_37 - 1.5)
            .collect();
        let weight: Vec<f32> = (0..GN_C).map(|j| 0.7 + (j % 5) as f32 * 0.1).collect();
        let bias: Vec<f32> = (0..GN_C).map(|j| (j % 3) as f32 * 0.05 - 0.05).collect();
        let x = $session
            .tensor_variable_f32(values, vec![GN_N, GN_C, GN_H, GN_W], true)
            .expect("leaf");
        let w = $session
            .tensor_variable_f32(weight, vec![GN_C], true)
            .expect("weight");
        let b = $session
            .tensor_variable_f32(bias, vec![GN_C], true)
            .expect("bias");
        (x, w, b)
    }};
}

fn main() {
    let numel = GN_N * GN_C * GN_H * GN_W;
    println!("narrow_route_sentinel (frankentorch-68pwz)");
    println!("shape [{GN_N},{GN_C},{GN_H},{GN_W}] groups={GN_GROUPS} numel={numel}");
    println!("rayon_threads={}", rayon::current_num_threads());
    println!();

    // ROUTE A — exactly what the board's `group_norm_f32` lane does: a plain sum loss.
    {
        let mut session = FrankenTorchSession::new(ExecutionMode::Strict);
        let (x, w, b) = build!(session);
        ft_api::reset_narrow_counts();
        let out = session
            .functional_group_norm(x, GN_GROUPS, Some(w), Some(b), 1e-5)
            .expect("group_norm");
        let loss = session.tensor_sum(out).expect("sum");
        let report = session.tensor_backward(loss).expect("backward");
        let grad_len = report.gradient(x).expect("grad").len();
        let (calls, elems) = ft_api::narrow_counts();
        println!("ROUTE A  loss = tensor_sum(out)          <- the h2h board's lane");
        println!("  narrow_f64_to_f32 calls={calls} elements={elems}  (grad len {grad_len})");
    }

    // ROUTE B — the positive control: any loss that is not a plain sum. Same op, same
    // shape, same process, so a difference here is the ROUTE and nothing else.
    {
        let mut session = FrankenTorchSession::new(ExecutionMode::Strict);
        let (x, w, b) = build!(session);
        ft_api::reset_narrow_counts();
        let out = session
            .functional_group_norm(x, GN_GROUPS, Some(w), Some(b), 1e-5)
            .expect("group_norm");
        let squared = session.tensor_mul(out, out).expect("mul");
        let loss = session.tensor_sum(squared).expect("sum");
        let report = session.tensor_backward(loss).expect("backward");
        let grad_len = report.gradient(x).expect("grad").len();
        let (calls, elems) = ft_api::narrow_counts();
        println!("ROUTE B  loss = tensor_sum(out*out)      <- positive control");
        println!("  narrow_f64_to_f32 calls={calls} elements={elems}  (grad len {grad_len})");
    }

    println!();
    println!(
        "READING IT: if A is 0 and B is {numel}, the lever is real but NOT on the lane the \
         board scores, and item 103's 2.5 ms cannot be claimed against that lane's engine \
         term. If both are non-zero, the lever is on both routes and the engine-term \
         attribution question stands."
    );

    // WHAT IT IS WORTH WHERE IT DOES FIRE. Paired ON/OFF over route B, alternating within
    // one process on one binary, a fresh session per rep (the tape never frees nodes, so a
    // reused session degrades linearly and would be measuring that instead). This is
    // FT-vs-FT — maintenance, not a win — and it is reported as the attribution number
    // item 103b owed, not as a ratio against the incumbent.
    println!();
    println!("PAIRED lever ON vs OFF on ROUTE B (FT-vs-FT; not a vs-incumbent ratio)");
    let reps = 9;
    let mut on_best = f64::INFINITY;
    let mut off_best = f64::INFINITY;
    let mut on_sum = 0.0;
    let mut off_sum = 0.0;
    for _ in 0..reps {
        for force_serial in [false, true] {
            let mut session = FrankenTorchSession::new(ExecutionMode::Strict);
            let (x, w, b) = build!(session);
            let previous = ft_api::set_gradient_narrow_serial(force_serial);
            let started = std::time::Instant::now();
            let out = session
                .functional_group_norm(x, GN_GROUPS, Some(w), Some(b), 1e-5)
                .expect("group_norm");
            let squared = session.tensor_mul(out, out).expect("mul");
            let loss = session.tensor_sum(squared).expect("sum");
            let report = session.tensor_backward(loss).expect("backward");
            let elapsed = started.elapsed().as_secs_f64() * 1_000.0;
            std::hint::black_box(report.gradient(x).expect("grad").len());
            ft_api::set_gradient_narrow_serial(previous);
            if force_serial {
                off_best = off_best.min(elapsed);
                off_sum += elapsed;
            } else {
                on_best = on_best.min(elapsed);
                on_sum += elapsed;
            }
        }
    }
    println!(
        "  lever ON  (parallel narrow)   min {on_best:.3} ms   mean {:.3} ms",
        on_sum / f64::from(reps)
    );
    println!(
        "  lever OFF (serial narrow)     min {off_best:.3} ms   mean {:.3} ms",
        off_sum / f64::from(reps)
    );
    println!(
        "  OFF/ON = {:.3}x on min, {:.3}x on mean; absolute {:.3} ms of a {off_best:.3} ms \
         backward",
        off_best / on_best,
        (off_sum / f64::from(reps)) / (on_sum / f64::from(reps)),
        off_best - on_best
    );
    println!("  loadavg {}", loadavg());
    println!("  cpu_mhz {}", cpu_mhz());

    // WHERE THE DENSE ROUTE'S TIME ACTUALLY GOES — NEGATIVE_EVIDENCE item 109.
    //
    // The board now certifies this route at 6.17-6.51x SLOWER than PyTorch, against the
    // sum-loss lane's 1.11-1.13x. Before anyone reaches for a lever, split the timed region
    // the lane measures (forward + square + sum + backward) into its four phases. This is
    // arm-internal attribution — no incumbent, no ratio, no gate — and it exists so the
    // next lever is aimed by measurement instead of by the fact that the narrow happens to
    // be the piece I already touched.
    println!();
    println!("PHASE SPLIT of the dense route's timed region (min of 9, arm-internal)");
    let mut fwd = f64::INFINITY;
    let mut sq = f64::INFINITY;
    let mut sum = f64::INFINITY;
    let mut bwd = f64::INFINITY;
    for _ in 0..9 {
        let mut session = FrankenTorchSession::new(ExecutionMode::Strict);
        let (x, w, b) = build!(session);
        let t0 = std::time::Instant::now();
        let out = session
            .functional_group_norm(x, GN_GROUPS, Some(w), Some(b), 1e-5)
            .expect("group_norm");
        let t1 = std::time::Instant::now();
        let squared = session.tensor_mul(out, out).expect("mul");
        let t2 = std::time::Instant::now();
        let loss = session.tensor_sum(squared).expect("sum");
        let t3 = std::time::Instant::now();
        let report = session.tensor_backward(loss).expect("backward");
        let t4 = std::time::Instant::now();
        std::hint::black_box(report.gradient(x).expect("grad").len());
        fwd = fwd.min((t1 - t0).as_secs_f64() * 1e3);
        sq = sq.min((t2 - t1).as_secs_f64() * 1e3);
        sum = sum.min((t3 - t2).as_secs_f64() * 1e3);
        bwd = bwd.min((t4 - t3).as_secs_f64() * 1e3);
    }
    let total = fwd + sq + sum + bwd;
    println!(
        "  forward  group_norm   {fwd:>9.3} ms  {:>5.1}%",
        100.0 * fwd / total
    );
    println!(
        "  square   tensor_mul   {sq:>9.3} ms  {:>5.1}%",
        100.0 * sq / total
    );
    println!(
        "  loss     tensor_sum   {sum:>9.3} ms  {:>5.1}%",
        100.0 * sum / total
    );
    println!(
        "  backward              {bwd:>9.3} ms  {:>5.1}%",
        100.0 * bwd / total
    );
    println!("  total                 {total:>9.3} ms");
    println!("  loadavg {}", loadavg());
    println!("  cpu_mhz {}", cpu_mhz());

    // SPLIT THE BACKWARD BY LANE SUBTRACTION, not by a profiler. The same graph with
    // exactly ONE node removed: `sum(x*x)` on the same f32 leaf, no group_norm. Whatever
    // that costs is the tape's own mul+sum backward over this many elements; the remainder
    // is what the GroupNorm backward closure adds. A cycle profile would divide a serial
    // region by the pool width and rank it near zero, which is why this campaign subtracts
    // lanes instead.
    println!();
    println!("BACKWARD SPLIT by lane subtraction (min of 9, arm-internal)");
    let mut nogn = f64::INFINITY;
    for _ in 0..9 {
        let mut session = FrankenTorchSession::new(ExecutionMode::Strict);
        let (x, _w, _b) = build!(session);
        let squared = session.tensor_mul(x, x).expect("mul");
        let loss = session.tensor_sum(squared).expect("sum");
        let t0 = std::time::Instant::now();
        let report = session.tensor_backward(loss).expect("backward");
        let elapsed = t0.elapsed().as_secs_f64() * 1e3;
        std::hint::black_box(report.gradient(x).expect("grad").len());
        nogn = nogn.min(elapsed);
    }
    println!("  backward WITH group_norm     {bwd:>9.3} ms");
    println!("  backward WITHOUT (sum(x*x))  {nogn:>9.3} ms   <- tape's own mul+sum backward");
    println!(
        "  difference                   {:>9.3} ms   <- what the GroupNorm backward adds",
        bwd - nogn
    );
    println!(
        "  tape mul+sum share of the whole timed region: {:.1}%",
        100.0 * nogn / total
    );

    // One more node removed: `sum(x)` alone. Its backward is a pure broadcast of 1.0 into a
    // numel-sized f64 gradient — no products at all. Separating it from the mul says whether
    // the lever is the ones-materialization or the elementwise product.
    let mut sumonly = f64::INFINITY;
    for _ in 0..9 {
        let mut session = FrankenTorchSession::new(ExecutionMode::Strict);
        let (x, _w, _b) = build!(session);
        let loss = session.tensor_sum(x).expect("sum");
        let t0 = std::time::Instant::now();
        let report = session.tensor_backward(loss).expect("backward");
        let elapsed = t0.elapsed().as_secs_f64() * 1e3;
        std::hint::black_box(report.gradient(x).expect("grad").len());
        sumonly = sumonly.min(elapsed);
    }
    println!("  backward of sum(x) alone     {sumonly:>9.3} ms   <- pure ones-broadcast");
    println!(
        "  so tensor_mul's backward adds {:>9.3} ms on top of it",
        nogn - sumonly
    );
    println!("  loadavg {}", loadavg());
    println!("  cpu_mhz {}", cpu_mhz());
}

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

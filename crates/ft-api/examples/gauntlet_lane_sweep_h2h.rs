//! Re-baseline the gauntlet lanes that never got a live-PyTorch comparison under
//! `--features fair-alloc` — `frankentorch-kgs4`.
//!
//! WHY THIS SHAPE OF MEASUREMENT. `frankentorch-ujw3g` established that the
//! gauntlet's headline ratios are dominated by each lane's per-iteration input
//! rebuild: for `avg_pool1d`, 57% of the step was the caller's 32 MiB `to_vec()`
//! and only ~12% was the pooling forward, and FrankenTorch's *op work* turned out
//! to be at ~1.18x while the whole-step ratio read 1.5-2.0x. A whole-step ratio
//! therefore mostly measures buffer-copy cost, which is allocator-shaped and
//! already settled by `frankentorch-1ji9l` option C.
//!
//! So this sweep times **op work only** — forward + backward with the input built
//! OUTSIDE the timed region on both sides. That is the number that says whether a
//! kernel is competitive, and it is the one a lever could actually move.
//!
//! WHY THE ARMS INTERLEAVE (`frankentorch-6atx2`). This harness used to run its
//! ENTIRE PyTorch arm to completion before the first FrankenTorch lane started,
//! so the two arms were sampled tens of seconds apart and any load shift in that
//! gap landed entirely, and undetectably, in the ratio. The contention preflight
//! could not cover it: it certifies only that nothing heavy sat on the placement
//! CPUs at the instant sampling began, not one second later. Repetition plus a
//! median averaged that effect down; nothing bounded it.
//!
//! The incumbent is therefore driven as a **co-process** — it sets up, warms up,
//! announces readiness, then returns exactly one timed sample per request — and
//! each round takes an incumbent sample immediately beside our samples for the
//! same lane. See `ft_api::harness_interleave` for the protocol and the schedule.
//!
//! THE ESTIMATOR IS PER-ROUND MEDIAN. The balanced square deliberately changes
//! the old min-of-seven estimator: every row now uses the median of four live
//! samples from each arm inside every round, then a median-ratio bootstrap over
//! those round medians. That is not comparable to the banked rows, but it is the
//! necessary cost of a paired comparison that remains valid on a shared host.
//!
//! Four lanes, shapes copied from `pytorch_gauntlet_bench` so the two describe the
//! same workloads: `max_pool1d`, `avg_pool2d`, `conv3d`, `max_pool3d`. Later rows
//! add `group_norm_f32` (kgs4.115) and `prelu` (k1hto) from the scorecard, each
//! beside a lever-off twin so one invocation carries both the standing and the
//! lever.
//!
//! Run (must be local; rch workers have no PyTorch):
//! ```text
//! PYTORCH_PYTHON=/path/to/python \
//!   cargo run --release -p frankentorch-api --features fair-alloc --example gauntlet_lane_sweep_h2h
//! ```

use std::io::{BufRead, BufReader, Write};
use std::process::{ChildStdin, ChildStdout, Command, Stdio};
use std::time::Instant;

use ft_api::FrankenTorchSession;
use ft_api::harness_interleave::{
    BALANCED_SQUARE, MAX_NULL_CI_WIDTH, QUIT_REQUEST, READY_MARKER, TIMED_STEPS,
    TIMED_STEPS_MARKER, adjudicate_null, parse_sample_line, parse_timed_steps, sample_request,
    timed_region_disagreement,
};
use ft_core::{DType, ExecutionMode, TensorMeta};

/// Every balanced-square round has four samples from each arm, so every arm
/// occupies two positions in each half even when the number of rounds is odd.
const DEFAULT_REPS: usize = 16;

/// Rounds for this run: `DEFAULT_REPS` unless `FT_H2H_REPS` overrides it.
///
/// The default is deliberately untouched, so an unadorned run is byte-comparable
/// with every banked one. The override exists because of what the 2026-08-14
/// session measured about this host: at load ~21, `max_pool3d`'s A/A gate came
/// back PASS in only 4 of 8 invocations and `avg_pool2d`'s in 1 of 8 — the other
/// rows were WIDE, i.e. undecidable, and an undecidable row cannot carry a
/// standing however many times it is repeated. More rounds is the only knob that
/// narrows a null CI without touching the estimator, so a measurement session on
/// a busy host can buy decidability with wall-clock instead of waiting for a
/// quiet machine that never arrives.
///
/// The floor gives the bootstrap enough independent round medians to resample.
fn reps() -> usize {
    let Ok(raw) = std::env::var("FT_H2H_REPS") else {
        return DEFAULT_REPS;
    };
    let Ok(parsed) = raw.trim().parse::<usize>() else {
        eprintln!("FT_H2H_REPS={raw:?} is not a number; using {DEFAULT_REPS}");
        return DEFAULT_REPS;
    };
    if parsed < 8 {
        eprintln!("FT_H2H_REPS={parsed} is below the floor of 8; using {DEFAULT_REPS}");
        return DEFAULT_REPS;
    }
    parsed
}

const BOOTSTRAP_REPS: usize = 2_000;
/// The balanced-square reference harness refuses a row unless each arm's A/A
/// point estimate stays within two percent of unity.  A CI that happens to
/// cover 1.0 is not enough: it can be broad, or centred away from unity, while
/// still passing the generic width gate.
const BALANCED_NULL_MAX_DEVIATION: f64 = 0.02;

// Shapes lifted verbatim from pytorch_gauntlet_bench.
const MP1_N: usize = 8;
const MP1_C: usize = 64;
const MP1_L: usize = 8192;

const AP2_N: usize = 8;
const AP2_C: usize = 64;
const AP2_H: usize = 64;
const AP2_W: usize = 64;

const C3_N: usize = 2;
const C3_CI: usize = 32;
const C3_CO: usize = 32;
const C3_D: usize = 8;
const C3_H: usize = 16;
const C3_W: usize = 16;
const C3_K: usize = 3;

const MP3_N: usize = 2;
const MP3_C: usize = 32;
const MP3_D: usize = 16;
const MP3_H: usize = 32;
const MP3_W: usize = 32;

// frankentorch-kgs4.115: GroupNorm f32 train step, copied from the scorecard row
// that records it at 19.04x slower — [8,64,28,28], 32 groups, affine grads, sum
// loss. That number predates the f32 affine-grad fused path (frankentorch-48w0b),
// so this lane exists to find out what the standing actually IS rather than to
// re-quote it.
const GN_N: usize = 8;
const GN_C: usize = 64;
const GN_H: usize = 28;
const GN_W: usize = 28;
const GN_GROUPS: usize = 32;

// frankentorch-k1hto: PReLU f64 scalar-loss train step, shape copied verbatim
// from the kgs4.157 scorecard row that records it at 2.58x slower. Both the input
// and the per-channel weight require grad, which is what makes the dense all-ones
// PReLU output gradient the thing the deforest lever removes.
const PR_N: usize = 32;
const PR_C: usize = 512;
const PR_W: usize = 256;

/// One FrankenTorch lane: runs a single timed forward+backward, returning
/// (milliseconds, gradient checksum).
type LaneRun<'a> = Box<dyn Fn() -> (f64, f64) + 'a>;

fn median(mut values: Vec<f64>) -> f64 {
    values.sort_by(f64::total_cmp);
    let middle = values.len() / 2;
    if values.len().is_multiple_of(2) {
        (values[middle - 1] + values[middle]) * 0.5
    } else {
        values[middle]
    }
}

/// `statistics.median` in the sanctioned Python balanced-square harness returns
/// the mean for an even pair. Keep that definition for each arm's two slots in
/// a half; selecting the upper value would manufacture a direction from slot
/// order before either A/A gate can detect it.
fn paired_slot_median(mut values: [f64; 2]) -> f64 {
    values.sort_by(f64::total_cmp);
    (values[0] + values[1]) * 0.5
}

/// Whether an arm's balanced-square A/A point estimate is centred at unity.
///
/// This is deliberately a second condition beside `adjudicate_null`: the
/// shared helper rejects broad or non-overlapping confidence intervals, while
/// the ported balanced-square protocol also requires the observed median to
/// land within its fixed +/-2% null band.
fn balanced_null_is_centered(point: f64) -> bool {
    point.is_finite() && (point - 1.0).abs() <= BALANCED_NULL_MAX_DEVIATION
}

#[cfg(test)]
mod tests {
    use super::{balanced_null_is_centered, median, paired_slot_median, timed_conv3d};

    #[test]
    fn balanced_square_overall_median_matches_python_statistics_median() {
        assert_eq!(median(vec![9.0, 1.0, 3.0, 7.0]), 5.0);
        assert_eq!(median(vec![9.0, 1.0, 3.0]), 3.0);
    }

    #[test]
    fn balanced_square_pair_median_matches_python_statistics_median() {
        assert_eq!(paired_slot_median([9.0, 1.0]), 5.0);
        assert_eq!(paired_slot_median([3.0, 3.0]), 3.0);
    }

    #[test]
    fn balanced_square_null_requires_a_centred_point_estimate() {
        assert!(balanced_null_is_centered(1.0));
        assert!(balanced_null_is_centered(1.019));
        assert!(!balanced_null_is_centered(1.021));
        assert!(!balanced_null_is_centered(f64::NAN));
    }

    #[test]
    fn timed_conv3d_builds_weight_before_the_timed_forward_backward_region() {
        // A 1x1x1 convolution gives a timing-independent contract:
        // sum(2*x).backward() has dx = 2. This exercises the helper that keeps
        // the weight leaf out of the FrankenTorch-only timed arm.
        let (milliseconds, checksum) =
            timed_conv3d(&[1.5], vec![1, 1, 1, 1, 1], &[2.0], vec![1, 1, 1, 1, 1]);
        assert!(milliseconds.is_finite() && milliseconds >= 0.0);
        assert_eq!(checksum.to_bits(), 2.0_f64.to_bits());
    }
}

fn next_random(state: &mut u64) -> u64 {
    *state ^= *state << 13;
    *state ^= *state >> 7;
    *state ^= *state << 17;
    *state
}

fn median_ratio_ci(numerator: &[f64], denominator: &[f64]) -> (f64, f64, f64) {
    assert_eq!(numerator.len(), denominator.len());
    let point = median(numerator.to_vec()) / median(denominator.to_vec());
    let mut samples = Vec::with_capacity(BOOTSTRAP_REPS);
    let mut state = 0x2f1e_9b47_c30d_a851_u64;
    for _ in 0..BOOTSTRAP_REPS {
        let mut left = Vec::with_capacity(numerator.len());
        let mut right = Vec::with_capacity(denominator.len());
        for _ in 0..numerator.len() {
            let index = (next_random(&mut state) as usize) % numerator.len();
            left.push(numerator[index]);
            right.push(denominator[index]);
        }
        samples.push(median(left) / median(right));
    }
    samples.sort_by(f64::total_cmp);
    (
        point,
        samples[BOOTSTRAP_REPS * 25 / 1_000],
        samples[BOOTSTRAP_REPS * 975 / 1_000],
    )
}

fn seq(n: usize) -> Vec<f64> {
    (0..n).map(|i| ((i % 251) as f64) * 0.001 - 0.12).collect()
}

/// Time forward+backward with the leaf already built. Returns (ms, grad checksum).
fn timed_op<F>(values: &[f64], shape: Vec<usize>, build: F) -> (f64, f64)
where
    F: Fn(&mut FrankenTorchSession, ft_autograd::TensorNodeId) -> ft_autograd::TensorNodeId,
{
    let mut session = FrankenTorchSession::new(ExecutionMode::Strict);
    let x = session
        .tensor_variable(values.to_vec(), shape, true)
        .expect("leaf");
    // Leaf construction is deliberately outside the timer — see the module note.
    let started = Instant::now();
    let out = build(&mut session, x);
    let loss = session.tensor_sum(out).expect("sum");
    let report = session.tensor_backward(loss).expect("backward");
    // frankentorch-574cu: STOP THE CLOCK HERE. The timed region is exactly
    // `harness_interleave::TIMED_STEPS` — forward, loss_sum, backward — which is
    // what the PyTorch arm times. The gradient checksum below is teardown: it
    // proves both sides computed the same thing, and on the avg_pool2d lane it
    // cost 1.599 ms (a serial dependent-add chain over 2M f64, 24% of that
    // lane's session) that the incumbent's timer never paid. Timing it on one arm
    // only biased every repetition in the same direction, which repetition cannot
    // average out.
    let elapsed = started.elapsed().as_secs_f64() * 1_000.0;
    let checksum = report
        .gradient(x)
        .expect("grad")
        .iter()
        .map(|g| g.abs())
        .sum::<f64>();
    (elapsed, checksum)
}

/// GroupNorm f32 train step: f32 input leaf plus f32 affine parameters, all three
/// requiring grad, with every leaf built before the declared timed region.
///
/// Deliberately a separate function rather than a generic over dtype: `timed_op`
/// builds an f64 leaf, and quietly handing this lane an f64 input would compare
/// FrankenTorch's f64 path against PyTorch's f32 one and read as a large loss for
/// a reason that has nothing to do with the kernel. The whole scorecard row is
/// about the f32 GRAD path, so the dtype is the measurement.
fn timed_group_norm_f32(values: &[f32], weight: &[f32], bias: &[f32]) -> (f64, f64) {
    let mut session = FrankenTorchSession::new(ExecutionMode::Strict);
    let x = session
        .tensor_variable_f32(values.to_vec(), vec![GN_N, GN_C, GN_H, GN_W], true)
        .expect("leaf");
    let w = session
        .tensor_variable_f32(weight.to_vec(), vec![GN_C], true)
        .expect("weight");
    let b = session
        .tensor_variable_f32(bias.to_vec(), vec![GN_C], true)
        .expect("bias");
    let started = Instant::now();
    let out = session
        .functional_group_norm(x, GN_GROUPS, Some(w), Some(b), 1e-5)
        .expect("group_norm");
    let loss = session.tensor_sum(out).expect("sum");
    let report = session.tensor_backward(loss).expect("backward");
    let elapsed = started.elapsed().as_secs_f64() * 1_000.0;
    let checksum = report
        .gradient(x)
        .expect("grad")
        .iter()
        .map(|g| g.abs())
        .sum::<f64>();
    (elapsed, checksum)
}

/// The same GroupNorm f32 work with NO session and NO tape: the kernels the
/// session lane actually calls, invoked directly, on f32 throughout.
///
/// This is the phase split, taken in the same invocation as the lane it splits,
/// so it costs one build instead of two round trips through a saturated fleet.
/// Reading the three numbers together answers the question the scorecard row
/// cannot:
///
///   PT vs `group_norm_f32`         the standing
///   `group_norm_f32` vs this       what the engine + dtype conversions cost
///   PT vs this                     whether the KERNEL is competitive at all
///
/// The engine term is the interesting one and it is structural, not incidental:
/// `apply_function_f32_output_with_create_graph_borrowed_inputs` takes incoming
/// gradients as `&[&[f64]]` and returns `Vec<Option<Vec<f64>>>`, so the tape's
/// gradient space is f64 even for an f32 op. Every f32 backward therefore
/// downcasts the incoming gradient and upcasts its results — two full-size
/// conversions plus their allocations, here over 401,408 elements. This lane
/// prices that, rather than assuming it matters.
///
/// ROUTE-MATCHED (frankentorch-npod3 ledger item 7f). The first version of this
/// lane called `group_norm_backward_f32` with a freshly built all-ones `dy`,
/// while the session lane takes the `sum`-loss shortcut
/// (`group_norm_f32_sum_shortcuts` -> `group_norm_backward_scalar_f32`) and never
/// materialises an upstream gradient at all. That split therefore measured
/// GENERAL route vs SHORTCUT route — a real number, but not the engine cost it
/// was read as, and it additionally charged this arm a 1.6 MiB `dy` allocation
/// plus the end-to-end scan the general backward performs just to discover the
/// gradient was all ones. Both arms now enter the same kernels, so the
/// difference between them is the tape.
/// `parallel_forward` selects the control arm, and it is the LEVER under test.
/// `false` reproduces the schedule the numel-only gate used to pick for this
/// shape (401,408 elements, 122,880 short of `NORM_FWD_PARALLEL_MIN`, so serial
/// on a 64-core host); `true` is what `group_norm_parallel_pays` now picks, since
/// the shape has 256 groups. Both produce bit-identical output, so the pair
/// isolates scheduling and nothing else — and it runs inside ONE invocation on
/// ONE host, which is the only form of this comparison that is admissible.
fn timed_group_norm_f32_kernels(
    values: &[f32],
    weight: &[f32],
    bias: &[f32],
    parallel_forward: bool,
) -> (f64, f64) {
    timed_group_norm_f32_kernels_inner(values, weight, bias, parallel_forward, false)
}

/// `reuse_stats` selects the `frankentorch-qkwsy` lever: the forward emits the
/// per-group statistics it computed anyway and the backward consumes them instead
/// of rebuilding them with two more full passes over the input. Both arms produce
/// bit-identical gradients (locked by
/// `group_norm_forward_f32_cpg2_stats_match_the_backward_recomputation`), so the
/// pair isolates the removed passes and nothing else, inside ONE invocation on
/// ONE host.
fn timed_group_norm_f32_kernels_inner(
    values: &[f32],
    weight: &[f32],
    bias: &[f32],
    parallel_forward: bool,
    reuse_stats: bool,
) -> (f64, f64) {
    if reuse_stats {
        let spatial = GN_H * GN_W;
        let out_meta = TensorMeta::from_shape(
            vec![GN_N, GN_C, GN_H, GN_W],
            DType::F32,
            ft_core::Device::Cpu,
        );
        let started = Instant::now();
        let (out, stats) = ft_kernel_cpu::group_norm_forward_f32_with_cpg2_stats(
            values,
            Some(weight),
            Some(bias),
            GN_N,
            GN_GROUPS,
            spatial,
            1e-5,
        );
        let loss = ft_kernel_cpu::sum_tensor_contiguous_f32(&out, &out_meta).expect("sum");
        let (dx, _, _) = ft_kernel_cpu::group_norm_backward_scalar_f32_with_cpg2_stats(
            1.0f32,
            values,
            Some(weight),
            &stats,
            GN_N,
            GN_GROUPS,
            spatial,
        );
        let elapsed = started.elapsed().as_secs_f64() * 1_000.0;
        assert!(loss.is_finite(), "group_norm f32 sum loss must be finite");
        let checksum = dx.iter().map(|g| f64::from(g.abs())).sum::<f64>();
        return (elapsed, checksum);
    }
    timed_group_norm_f32_kernels_recomputing(values, weight, bias, parallel_forward)
}

fn timed_group_norm_f32_kernels_recomputing(
    values: &[f32],
    weight: &[f32],
    bias: &[f32],
    parallel_forward: bool,
) -> (f64, f64) {
    let spatial = GN_H * GN_W;
    let channels_per_group = GN_C / GN_GROUPS;
    // Built outside the timer: the session lane's `tensor_sum` reads a meta that
    // already exists, so charging this arm for constructing one would price a
    // difference in harness bookkeeping as a difference in engine cost.
    let out_meta = TensorMeta::from_shape(
        vec![GN_N, GN_C, GN_H, GN_W],
        DType::F32,
        ft_core::Device::Cpu,
    );
    let started = Instant::now();
    let out = ft_kernel_cpu::group_norm_forward_f32_scheduled(
        values,
        Some(weight),
        Some(bias),
        GN_N,
        GN_GROUPS,
        channels_per_group,
        spatial,
        1e-5,
        parallel_forward,
    );
    // The `sum` loss itself, which the session lane also pays: its shortcut still
    // reduces the forward output to a scalar before backward runs.
    let loss = ft_kernel_cpu::sum_tensor_contiguous_f32(&out, &out_meta).expect("sum");
    // `sum` loss, so the upstream gradient is the scalar 1.0 — the shortcut the
    // session lane takes, entered here through the same kernel rather than
    // through the general backward's all-ones `dy` scan.
    let (dx, _, _) = ft_kernel_cpu::group_norm_backward_scalar_f32(
        1.0f32,
        values,
        Some(weight),
        GN_N,
        GN_GROUPS,
        channels_per_group,
        spatial,
        1e-5,
    );
    let elapsed = started.elapsed().as_secs_f64() * 1_000.0;
    // Keeps the loss observable so the compiler cannot delete the reduction that
    // was just timed.
    assert!(loss.is_finite(), "group_norm f32 sum loss must be finite");
    // Accumulate in f64. Summing 401,408 f32 values naively carries ~1e-4
    // relative error — a hundred times the parity tolerance — so an f32
    // accumulator reports MISMATCH against a torch arm that computed the SAME
    // gradients, purely from the accumulator's precision. The other lanes are
    // unaffected because the tape's gradients are already f64; this is the one
    // lane that touches raw f32 kernel output.
    let checksum = dx.iter().map(|g| f64::from(g.abs())).sum::<f64>();
    (elapsed, checksum)
}

/// PReLU f64 scalar-loss train step: input leaf plus per-channel weight leaf,
/// both requiring grad, both built before the declared timed region.
///
/// `defeat_shortcut` selects the control arm. `frankentorch-k1hto` made
/// `tensor_sum` on a PReLU output reconstruct both gradients from the saved input
/// and weight, so the dense all-ones output gradient is never materialised. The
/// shortcut declines whenever that output retains its gradient or carries a hook,
/// and the HOOK exit is the one to take here: an observation-only hook (returning
/// `Ok(None)`) costs one map insert at registration and one closure call in
/// backward, and copies nothing, so the control lane is the old materialising path
/// plus a constant far below the resolution of a lane this size. Registration sits
/// inside the timed region only because the node it names does not exist before
/// the forward. The retain_grad exit is NOT
/// usable for this — `backward` persists a retained gradient via
/// `contiguous_values_as_f64()`, charging the control arm a 33 MiB copy that has
/// nothing to do with the lever and would inflate the paired ratio.
fn timed_prelu(values: &[f64], weight: &[f64], defeat_shortcut: bool) -> (f64, f64) {
    let mut session = FrankenTorchSession::new(ExecutionMode::Strict);
    let x = session
        .tensor_variable(values.to_vec(), vec![PR_N, PR_C, PR_W], true)
        .expect("prelu leaf");
    let w = session
        .tensor_variable(weight.to_vec(), vec![PR_C], true)
        .expect("prelu weight");
    let started = Instant::now();
    let out = session.tensor_prelu(x, w).expect("prelu");
    if defeat_shortcut {
        session
            .tensor_register_hook(out, |_grad| Ok(None))
            .expect("observation-only hook");
    }
    let loss = session.tensor_sum(out).expect("sum");
    let report = session.tensor_backward(loss).expect("backward");
    let elapsed = started.elapsed().as_secs_f64() * 1_000.0;
    let checksum = report
        .gradient(x)
        .expect("grad")
        .iter()
        .map(|g| g.abs())
        .sum::<f64>();
    (elapsed, checksum)
}

/// Time Conv3d with both tensor leaves built before the declared timed region.
///
/// The PyTorch arm constructs `c3w` during setup and reuses it for every sample.
/// Creating the matching Frankentorch weight after `Instant::now()` would charge
/// tensor/tape construction to only one arm, so keep it beside the input leaf.
fn timed_conv3d(
    values: &[f64],
    input_shape: Vec<usize>,
    weights: &[f64],
    weight_shape: Vec<usize>,
) -> (f64, f64) {
    let mut session = FrankenTorchSession::new(ExecutionMode::Strict);
    let x = session
        .tensor_variable(values.to_vec(), input_shape, true)
        .expect("conv3d leaf");
    let weight = session
        .tensor_variable(weights.to_vec(), weight_shape, false)
        .expect("conv3d weight");
    // Both leaves now exist, matching the PyTorch arm. The timer remains exactly
    // forward + loss_sum + backward, as declared to the co-process.
    let started = Instant::now();
    let out = session
        .functional_conv3d(x, weight, None, (1, 1, 1), (1, 1, 1))
        .expect("conv3d");
    let loss = session.tensor_sum(out).expect("sum");
    let report = session.tensor_backward(loss).expect("backward");
    let elapsed = started.elapsed().as_secs_f64() * 1_000.0;
    let checksum = report
        .gradient(x)
        .expect("grad")
        .iter()
        .map(|g| g.abs())
        .sum::<f64>();
    (elapsed, checksum)
}

/// Ask the incumbent co-process for exactly one timed sample of `lane`.
///
/// Chatter the child may emit (warnings, notices) is skipped rather than parsed,
/// but a closed stdout is a hard failure: a silently-short arm would otherwise
/// leave the lane's remaining rounds measuring only our side.
fn incumbent_sample(
    stdin: &mut ChildStdin,
    reader: &mut BufReader<ChildStdout>,
    lane: &str,
) -> Result<(f64, f64), Box<dyn std::error::Error>> {
    writeln!(stdin, "{}", sample_request(lane))?;
    stdin.flush()?;
    let mut line = String::new();
    loop {
        line.clear();
        if reader.read_line(&mut line)? == 0 {
            return Err(format!(
                "the PyTorch co-process closed its stdout while lane `{lane}` was being sampled; \
                 a partially-measured arm cannot carry a vs-PyTorch claim"
            )
            .into());
        }
        if let Some(sample) = parse_sample_line(&line) {
            assert_eq!(
                sample.lane, lane,
                "co-process answered for the wrong lane; replies would be misfiled"
            );
            return Ok((sample.milliseconds, sample.gradient_checksum));
        }
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let reps = reps();

    let mp1 = seq(MP1_N * MP1_C * MP1_L);
    let ap2 = seq(AP2_N * AP2_C * AP2_H * AP2_W);
    let c3x = seq(C3_N * C3_CI * C3_D * C3_H * C3_W);
    let c3w = seq(C3_CO * C3_CI * C3_K * C3_K * C3_K);
    let mp3 = seq(MP3_N * MP3_C * MP3_D * MP3_H * MP3_W);
    // Built by the SAME formula the python arm uses, then cast — so the two arms
    // normalize identical numbers and the gradient checksum is a real parity check
    // rather than a coincidence of shapes.
    #[allow(clippy::cast_possible_truncation)]
    let gnx: Vec<f32> = seq(GN_N * GN_C * GN_H * GN_W)
        .into_iter()
        .map(|value| value as f32)
        .collect();
    #[allow(clippy::cast_possible_truncation)]
    let gnw: Vec<f32> = seq(GN_C)
        .into_iter()
        .map(|value| (value * 10.0 + 1.0) as f32)
        .collect();
    #[allow(clippy::cast_possible_truncation)]
    let gnb: Vec<f32> = seq(GN_C)
        .into_iter()
        .map(|value| (value * 3.0) as f32)
        .collect();

    // frankentorch-k1hto. `seq` spans -0.12..0.131, so both sides of the PReLU
    // kink carry real elements — a same-sign input would exercise one branch of
    // the backward and price the lever against work it never does.
    let prx = seq(PR_N * PR_C * PR_W);
    let prw: Vec<f64> = seq(PR_C)
        .into_iter()
        .map(|value| value * 0.1 + 0.05)
        .collect();

    let python = std::env::var("PYTORCH_PYTHON").unwrap_or_else(|_| "python3".to_string());
    // Setup only. The request/serve/quit loop is appended from the library so the
    // protocol markers cannot drift away from the parser that reads them.
    let py_setup = r#"
import time, torch
import torch.nn.functional as Fn
# frankentorch-wnku0: the arm self-reports its version, in this same invocation,
# BEFORE any timing — so a run that dies mid-measurement still leaves provenance.
print('PT_TORCH_VERSION %s' % torch.__version__, flush=True)
torch.set_num_threads(8)
def seq(n):
    return ((torch.arange(n,dtype=torch.int64)%251).double())*0.001-0.12
mp1=seq(8*64*8192).reshape(8,64,8192)
ap2=seq(8*64*64*64).reshape(8,64,64,64)
c3x=seq(2*32*8*16*16).reshape(2,32,8,16,16)
c3w=seq(32*32*3*3*3).reshape(32,32,3,3,3)
mp3=seq(2*32*16*32*32).reshape(2,32,16,32,32)
# frankentorch-kgs4.115 GroupNorm f32 train step, shape and groups copied verbatim
# from the scorecard row so the two describe the same workload. f32 on BOTH sides:
# `.float()` here, `tensor_variable_f32` there. The affine parameters require grad,
# which is the whole point of the row — the f32 no-grad path has long been fused,
# and it is the GRAD path the scorecard measured at 19.04x.
gnx=seq(8*64*28*28).reshape(8,64,28,28).float()
gnw=(seq(64)*10.0+1.0).float().requires_grad_(True)
gnb=(seq(64)*3.0).float().requires_grad_(True)
# frankentorch-k1hto PReLU f64 train step. The weight requires grad on both arms:
# the lever reconstructs BOTH gradients from the scalar upstream, so a no-grad
# weight would skip the half of the backward it changes most.
prx=seq(32*512*256).reshape(32,512,256)
prw=(seq(512)*0.1+0.05).requires_grad_(True)
# frankentorch-574cu: this arm declares the region it times, so a change to
# `run` below that is not mirrored here fails the run instead of silently
# biasing every ratio. Written as an independent literal rather than generated
# from the Rust constant, which would make the agreement check tautological.
print('PT_TIMED_STEPS forward,loss_sum,backward', flush=True)
def run(base, fn):
    # leaf built OUTSIDE the timed region, matching the FrankenTorch side
    x=base.detach().clone().requires_grad_(True)
    s=time.perf_counter()
    fn(x).sum().backward()          # <- forward, loss_sum, backward: the timed region
    elapsed=(time.perf_counter()-s)*1e3
    # teardown, deliberately AFTER the clock stops on BOTH arms
    return elapsed, x.grad.abs().sum().item()
LANES = {
    "max_pool1d": (mp1, lambda x: Fn.max_pool1d(x,2,2)),
    "avg_pool2d": (ap2, lambda x: Fn.avg_pool2d(x,(2,2),(2,2))),
    "conv3d":     (c3x, lambda x: Fn.conv3d(x,c3w,None,(1,1,1),(1,1,1))),
    "max_pool3d": (mp3, lambda x: Fn.max_pool3d(x,(2,2,2),(2,2,2))),
    # frankentorch-9pafs: the same torch op under a second name. The FrankenTorch
    # side runs this lane with its buffer pool DISABLED, so the pair isolates the
    # pool against one live incumbent inside one invocation. Because the incumbent
    # code is byte-identical for the two names, PT(max_pool3d_nopool)/PT(max_pool3d)
    # is a free control that must come out ~1.0; if it does not, the host moved
    # during the run and the FT comparison is not readable either.
    "max_pool3d_nopool": (mp3, lambda x: Fn.max_pool3d(x,(2,2,2),(2,2,2))),
    # frankentorch-7zqbc: same pairing for the two other lanes whose backward now
    # takes from the pool.
    "avg_pool2d_nopool": (ap2, lambda x: Fn.avg_pool2d(x,(2,2),(2,2))),
    "max_pool1d_nopool": (mp1, lambda x: Fn.max_pool1d(x,2,2)),
    "group_norm_f32": (gnx, lambda x: Fn.group_norm(x,32,gnw,gnb)),
    # The FrankenTorch side of this second name calls the two f32 kernels
    # DIRECTLY, with no session and no tape, to price the engine and the f64
    # grad-space conversions separately from the kernel. The incumbent is the
    # same op under both names, so PT(kernels)/PT(f32) is a free ~1.0 control.
    "group_norm_f32_kernels": (gnx, lambda x: Fn.group_norm(x,32,gnw,gnb)),
    # frankentorch-dmpho: the same torch op under a third name. The FrankenTorch
    # side runs this one with the group-norm forward forced onto the old serial
    # schedule, so the pair prices the parallel gate against one live incumbent
    # inside one invocation. PT(serialfwd)/PT(kernels) is a free ~1.0 control.
    "group_norm_f32_kernels_serialfwd": (gnx, lambda x: Fn.group_norm(x,32,gnw,gnb)),
    # frankentorch-qkwsy: the same torch op again for the forward-statistics-reuse
    # pair. PT(statskernels_recompute)/PT(statskernels) is a free ~1.0 control.
    "group_norm_f32_statskernels": (gnx, lambda x: Fn.group_norm(x,32,gnw,gnb)),
    "group_norm_f32_statskernels_recompute": (gnx, lambda x: Fn.group_norm(x,32,gnw,gnb)),
    "prelu": (prx, lambda x: Fn.prelu(x,prw)),
    # frankentorch-k1hto: the same torch op under a second name, exactly as the
    # `_nopool` lanes do. The FrankenTorch side runs this one with an
    # observation-only hook on the PReLU output, which makes the sum shortcut
    # decline and restores the materialising path. PT(noshortcut)/PT(prelu) is
    # therefore a free control that must land near 1.0.
    "prelu_noshortcut": (prx, lambda x: Fn.prelu(x,prw)),
}
"#;
    let py = format!("{py_setup}{}", ft_api::harness_interleave::SAMPLE_LOOP_PY);

    // `-c`, never `-`: the latter reads the program from stdin until EOF, which
    // deadlocks a co-process whose stdin must stay open for requests.
    let mut child = Command::new(&python)
        .args(ft_api::harness_interleave::interpreter_args(&py))
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .spawn()
        .map_err(|error| {
            format!(
                "could not start the PyTorch arm (`{python}`): {error}. Set PYTORCH_PYTHON to an \
                 interpreter with torch installed; a FrankenTorch-only run cannot carry a \
                 vs-PyTorch claim."
            )
        })?;
    let mut stdin = child
        .stdin
        .take()
        .ok_or_else(|| std::io::Error::other("no stdin"))?;
    let mut reader = BufReader::new(
        child
            .stdout
            .take()
            .ok_or_else(|| std::io::Error::other("no stdout"))?,
    );
    // Block until the arm has imported torch, built its tensors and warmed every
    // lane. Anything it prints before that (version line, warnings) is preamble.
    let mut preamble = String::new();
    loop {
        let mut line = String::new();
        if reader.read_line(&mut line)? == 0 {
            return Err(format!(
                "the PyTorch arm exited before announcing `{READY_MARKER}`; set PYTORCH_PYTHON to \
                 an interpreter with torch installed. A FrankenTorch-only run cannot carry a \
                 vs-PyTorch claim. Its output was:\n{preamble}"
            )
            .into());
        }
        if line.trim() == READY_MARKER {
            break;
        }
        preamble.push_str(&line);
    }

    // frankentorch-wnku0: hard-fails if the arm did not self-report, so this
    // harness cannot emit ratios without the version they were measured against.
    let torch_version = ft_api::harness_provenance::require_reported_version(&preamble)?;

    // frankentorch-574cu: both arms must have timed the SAME region, or the
    // ratios below are biased in one direction in every repetition and nothing
    // here is quotable. Fail the run rather than print a caveat nobody reads.
    let incumbent_steps = parse_timed_steps(&preamble).ok_or_else(|| {
        format!(
            "the PyTorch arm did not declare its timed region: no `{TIMED_STEPS_MARKER}` line. \
             Without it there is no evidence the two arms measured the same work."
        )
    })?;
    if let Some(message) = timed_region_disagreement(TIMED_STEPS, &incumbent_steps) {
        return Err(message.into());
    }

    println!(
        "executing_elf_sha256={}",
        ft_api::harness_provenance::executing_elf_sha256()
    );
    println!(
        "{}",
        ft_api::harness_provenance::incumbent_provenance_block(torch_version, 8)
    );
    // Names the machine this row was measured on. Both arms are sampled in this
    // one invocation on this one host, so the row is internally comparable; the
    // block exists so it can still be PLACED against other rows afterwards.
    println!(
        "{}",
        ft_api::harness_provenance::measurement_host_block(rayon::current_num_threads())
    );
    // frankentorch-2h8vi: the host's own movement, which no A/A null can see.
    let load_at_start = ft_api::harness_provenance::load_average_1m();
    println!(
        "allocator={}",
        if cfg!(feature = "fair-alloc") {
            "mimalloc (--features fair-alloc)"
        } else {
            "system (glibc malloc) — re-run with --features fair-alloc before quoting"
        }
    );
    println!(
        "measurement=OP WORK ONLY (forward+backward; leaf built outside the timer on BOTH sides)"
    );
    println!(
        "sampling=balanced-square {} (frankentorch-xdw0h); {} rounds, four live samples per arm \
         per round, torch threads=8",
        BALANCED_SQUARE
            .iter()
            .map(|incumbent| if *incumbent { 'A' } else { 'B' })
            .collect::<String>(),
        reps,
    );
    println!();
    println!(
        "lane          FT(ms)    PT(ms)   standing            PT/FT A/A gates and ratio CI                 parity"
    );

    let lanes: Vec<(&str, LaneRun<'_>)> = vec![
        (
            "max_pool1d",
            Box::new(|| {
                timed_op(&mp1, vec![MP1_N, MP1_C, MP1_L], |s, x| {
                    s.functional_max_pool1d(x, 2, 2).expect("max_pool1d")
                })
            }),
        ),
        (
            "avg_pool2d",
            Box::new(|| {
                timed_op(&ap2, vec![AP2_N, AP2_C, AP2_H, AP2_W], |s, x| {
                    s.functional_avg_pool2d_sum(x, (2, 2), (2, 2), (0, 0), false, true)
                        .expect("avg_pool2d")
                })
            }),
        ),
        (
            "max_pool3d",
            Box::new(|| {
                timed_op(&mp3, vec![MP3_N, MP3_C, MP3_D, MP3_H, MP3_W], |s, x| {
                    s.functional_max_pool3d(x, (2, 2, 2), (2, 2, 2))
                        .expect("max_pool3d")
                })
            }),
        ),
        (
            // frankentorch-9pafs: the SAME lane with `ft_core::buffer_pool` off.
            // Two things make this the readable form of the comparison. It is one
            // binary, so the two arms cannot differ by a compilation; and both are
            // sampled against one live incumbent inside one invocation, so the
            // ratio-of-ratios against `max_pool3d` is immune to the host drift
            // that makes cross-run ratios unquotable here.
            "max_pool3d_nopool",
            Box::new(|| {
                ft_core::buffer_pool::set_enabled(false);
                let sample = timed_op(&mp3, vec![MP3_N, MP3_C, MP3_D, MP3_H, MP3_W], |s, x| {
                    s.functional_max_pool3d(x, (2, 2, 2), (2, 2, 2))
                        .expect("max_pool3d")
                });
                ft_core::buffer_pool::set_enabled(true);
                sample
            }),
        ),
        (
            "avg_pool2d_nopool",
            Box::new(|| {
                ft_core::buffer_pool::set_enabled(false);
                let sample = timed_op(&ap2, vec![AP2_N, AP2_C, AP2_H, AP2_W], |s, x| {
                    s.functional_avg_pool2d_sum(x, (2, 2), (2, 2), (0, 0), false, true)
                        .expect("avg_pool2d")
                });
                ft_core::buffer_pool::set_enabled(true);
                sample
            }),
        ),
        (
            "max_pool1d_nopool",
            Box::new(|| {
                ft_core::buffer_pool::set_enabled(false);
                let sample = timed_op(&mp1, vec![MP1_N, MP1_C, MP1_L], |s, x| {
                    s.functional_max_pool1d(x, 2, 2).expect("max_pool1d")
                });
                ft_core::buffer_pool::set_enabled(true);
                sample
            }),
        ),
        (
            "group_norm_f32",
            Box::new(|| timed_group_norm_f32(&gnx, &gnw, &gnb)),
        ),
        (
            "group_norm_f32_kernels",
            Box::new(|| timed_group_norm_f32_kernels(&gnx, &gnw, &gnb, true)),
        ),
        (
            // frankentorch-dmpho: the lever-off twin. Same kernels, same shape,
            // same binary; only the forward's schedule differs.
            "group_norm_f32_kernels_serialfwd",
            Box::new(|| timed_group_norm_f32_kernels(&gnx, &gnw, &gnb, false)),
        ),
        (
            // frankentorch-qkwsy: lever ON — the forward emits its statistics and
            // the backward consumes them. This pair gets its own base rather than
            // reusing `group_norm_f32_kernels`, which is already the base for
            // `_serialfwd`; sharing it would leave a twin differing in two things
            // at once.
            "group_norm_f32_statskernels",
            Box::new(|| timed_group_norm_f32_kernels_inner(&gnx, &gnw, &gnb, true, true)),
        ),
        (
            // Lever OFF: identical work, statistics rebuilt in the backward.
            "group_norm_f32_statskernels_recompute",
            Box::new(|| timed_group_norm_f32_kernels_inner(&gnx, &gnw, &gnb, true, false)),
        ),
        ("prelu", Box::new(|| timed_prelu(&prx, &prw, false))),
        (
            // frankentorch-k1hto: the SAME lane with the PReLU+sum shortcut
            // declined. One binary, two arms against one live incumbent inside one
            // invocation, so the ratio-of-ratios against `prelu` is immune to the
            // host drift that makes cross-run ratios unquotable here.
            "prelu_noshortcut",
            Box::new(|| timed_prelu(&prx, &prw, true)),
        ),
        (
            "conv3d",
            Box::new(|| {
                timed_conv3d(
                    &c3x,
                    vec![C3_N, C3_CI, C3_D, C3_H, C3_W],
                    &c3w,
                    vec![C3_CO, C3_CI, C3_K, C3_K, C3_K],
                )
            }),
        ),
    ];

    // Our side warms here; the incumbent warmed itself before announcing ready.
    //
    // The COUNT must match the incumbent's, which warms `for _ in range(4)` per
    // lane in `harness_interleave::SAMPLE_LOOP_PY` (frankentorch-6atx2). It used
    // to be 3 here against torch's 4 (frankentorch-2kgum). That asymmetry was
    // conservative for every row this campaign has quoted -- under-warming OUR
    // arm makes our times slower, which UNDERSTATES a FrankenTorch-faster ratio,
    // and every certified row so far is FrankenTorch-faster. But the SIGN of that
    // bias is a property of which arm happens to be faster, not of the
    // instrument: the first time a lane is quoted where torch is ahead, the same
    // asymmetry inflates instead. An instrument must not have a bias whose
    // direction depends on the answer.
    const WARMUP_ITERS: usize = 4;
    for (_, run_lane) in &lanes {
        let mut warm = 0.0;
        for _ in 0..WARMUP_ITERS {
            warm += run_lane().0;
        }
        std::hint::black_box(warm);
    }

    let mut ft_times: Vec<Vec<f64>> = vec![Vec::with_capacity(reps); lanes.len()];
    // frankentorch-rled4: the same rounds reduced by MIN instead of median.
    let mut pt_round_min: Vec<Vec<f64>> = vec![Vec::with_capacity(reps); lanes.len()];
    let mut ft_round_min: Vec<Vec<f64>> = vec![Vec::with_capacity(reps); lanes.len()];
    let mut pt_first_half_min: Vec<Vec<f64>> = vec![Vec::with_capacity(reps); lanes.len()];
    let mut pt_second_half_min: Vec<Vec<f64>> = vec![Vec::with_capacity(reps); lanes.len()];
    let mut ft_first_half_min: Vec<Vec<f64>> = vec![Vec::with_capacity(reps); lanes.len()];
    let mut ft_second_half_min: Vec<Vec<f64>> = vec![Vec::with_capacity(reps); lanes.len()];
    let mut pt_times: Vec<Vec<f64>> = vec![Vec::with_capacity(reps); lanes.len()];
    let mut ft_first_half: Vec<Vec<f64>> = vec![Vec::with_capacity(reps); lanes.len()];
    let mut ft_second_half: Vec<Vec<f64>> = vec![Vec::with_capacity(reps); lanes.len()];
    let mut pt_first_half: Vec<Vec<f64>> = vec![Vec::with_capacity(reps); lanes.len()];
    let mut pt_second_half: Vec<Vec<f64>> = vec![Vec::with_capacity(reps); lanes.len()];
    let mut pt_grads: Vec<Option<f64>> = vec![None; lanes.len()];
    let mut checksums: Vec<f64> = vec![0.0; lanes.len()];

    // Ported from franken_networkx `balanced_square_ab.py` @72761094c. Its
    // `ABBAABBA` order gives each arm every half and slot position equally, so
    // host drift and peer work are common-mode instead of a precondition that
    // every CPU must be idle. Each arm's own half-vs-half A/A null remains a
    // mandatory gate; a failed row is refused rather than averaged into a ratio.

    for _ in 0..reps {
        for (index, (name, run_lane)) in lanes.iter().enumerate() {
            let mut incumbent_slots = Vec::with_capacity(4);
            let mut ft_slots = Vec::with_capacity(4);
            for incumbent_slot in BALANCED_SQUARE {
                if incumbent_slot {
                    let (ms, grad) = incumbent_sample(&mut stdin, &mut reader, name)?;
                    incumbent_slots.push(ms);
                    pt_grads[index] = Some(grad);
                } else {
                    let (ms, checksum) = run_lane();
                    ft_slots.push(ms);
                    checksums[index] = checksum;
                }
            }
            pt_first_half[index].push(paired_slot_median([incumbent_slots[0], incumbent_slots[1]]));
            pt_second_half[index]
                .push(paired_slot_median([incumbent_slots[2], incumbent_slots[3]]));
            ft_first_half[index].push(paired_slot_median([ft_slots[0], ft_slots[1]]));
            ft_second_half[index].push(paired_slot_median([ft_slots[2], ft_slots[3]]));
            // frankentorch-rled4: the same halves reduced by MIN. The A/A null is
            // adjudicated on the estimator, so a noisy estimator vetoes rows whose
            // arms are actually clean — and that is what has been happening: on
            // every lane where FrankenTorch reads FASTER than the incumbent, it is
            // the INCUMBENT's null that fails, not ours. Torch's own samples are
            // the noisy ones on this host.
            pt_first_half_min[index].push(incumbent_slots[0].min(incumbent_slots[1]));
            pt_second_half_min[index].push(incumbent_slots[2].min(incumbent_slots[3]));
            ft_first_half_min[index].push(ft_slots[0].min(ft_slots[1]));
            ft_second_half_min[index].push(ft_slots[2].min(ft_slots[3]));
            // frankentorch-rled4: keep the per-round FLOOR beside the per-round
            // median. Both are computed from the SAME four slots of the same
            // round, so the two estimators see identical work and differ only in
            // how they reduce it — which is the whole point, and the reason this
            // is not a replacement. Measured on this lane: the median reading of
            // a fixed workload moved 24% between invocations while the min moved
            // 9%, because on a host carrying a dozen peer agents the median
            // mostly measures the neighbours and the min measures the machine.
            pt_round_min[index].push(
                incumbent_slots
                    .iter()
                    .copied()
                    .fold(f64::INFINITY, f64::min),
            );
            ft_round_min[index].push(ft_slots.iter().copied().fold(f64::INFINITY, f64::min));
            pt_times[index].push(median(incumbent_slots));
            ft_times[index].push(median(ft_slots));
        }
    }

    writeln!(stdin, "{QUIT_REQUEST}")?;
    stdin.flush()?;
    drop(stdin);
    child.wait()?;

    // frankentorch-2h8vi. Sampling is done; read the host again. An A/A null
    // certifies that an arm was STABLE WITHIN this invocation and cannot see the
    // machine moving underneath BOTH arms — NEGATIVE_EVIDENCE item 18 recorded
    // two runs of one lane, every null passing, certifying OPPOSITE directions,
    // separated by nothing but the host filling up mid-run. The balanced square
    // protects against a step between arms, not against drift under both.
    let load_at_end = ft_api::harness_provenance::load_average_1m();
    let load_quotable =
        ft_api::harness_provenance::load_drift_is_quotable(load_at_start, load_at_end);
    let fmt_load = |value: Option<f64>| {
        value.map_or_else(|| "unknown".to_owned(), |load| format!("{load:.2}"))
    };
    println!(
        "load_1m start={} end={} drift_gate={} (frankentorch-2h8vi; the signal is DRIFT in either \
         direction, not level — a steady busy host is measurable, a host that moves under the \
         measurement is not)",
        fmt_load(load_at_start),
        fmt_load(load_at_end),
        if load_quotable {
            "PASS".to_owned()
        } else {
            format!(
                "LOAD-DRIFTED — no row from this invocation is quotable, whatever its nulls say \
                 (max {:.2}x)",
                ft_api::harness_provenance::MAX_LOAD_DRIFT
            )
        }
    );

    for (index, (name, _)) in lanes.iter().enumerate() {
        let (ratio, ratio_lo, ratio_hi) = median_ratio_ci(&pt_times[index], &ft_times[index]);
        let (pt_null_ratio, pt_null_lo, pt_null_hi) =
            median_ratio_ci(&pt_first_half[index], &pt_second_half[index]);
        let (ft_null_ratio, ft_null_lo, ft_null_hi) =
            median_ratio_ci(&ft_first_half[index], &ft_second_half[index]);
        let pt_null = adjudicate_null(pt_null_lo, pt_null_hi, MAX_NULL_CI_WIDTH);
        let ft_null = adjudicate_null(ft_null_lo, ft_null_hi, MAX_NULL_CI_WIDTH);
        let pt_null_quotable = pt_null.is_quotable() && balanced_null_is_centered(pt_null_ratio);
        let ft_null_quotable = ft_null.is_quotable() && balanced_null_is_centered(ft_null_ratio);
        let pt_null_label = if pt_null.is_quotable() && !pt_null_quotable {
            "OFFSET"
        } else {
            pt_null.label()
        };
        let ft_null_label = if ft_null.is_quotable() && !ft_null_quotable {
            "OFFSET"
        } else {
            ft_null.label()
        };
        let ft_ms = median(ft_times[index].clone());
        let (Some(pt_grad), false) = (pt_grads[index], pt_times[index].is_empty()) else {
            println!("  {name:<12} {ft_ms:8.3}       --   PyTorch row missing");
            continue;
        };
        let pt_ms = median(pt_times[index].clone());
        let standing = if ratio >= 1.0 {
            format!("FT {ratio:.2}x FASTER")
        } else {
            format!("FT {:.2}x SLOWER", 1.0 / ratio)
        };
        // Gradient-sum agreement: the two sides must be computing the same thing.
        let tolerance = 1e-6 * pt_grad.abs().max(1.0);
        let parity = if (checksums[index] - pt_grad).abs() <= tolerance {
            "match"
        } else {
            "MISMATCH"
        };
        println!(
            "  {name:<12} {ft_ms:8.3} {pt_ms:8.3}   {standing:<19} PT {} [{pt_null_lo:.3},{pt_null_hi:.3}] FT {} [{ft_null_lo:.3},{ft_null_hi:.3}] ratio {ratio:.3} [{ratio_lo:.3},{ratio_hi:.3}] {parity}",
            pt_null_label, ft_null_label,
        );
        if !(pt_null_quotable && ft_null_quotable) {
            println!(
                "    NULL-FAILED: incumbent {pt_null_ratio:.3}, FrankenTorch {ft_null_ratio:.3}; each must be within +/-{BALANCED_NULL_MAX_DEVIATION:.2} of 1.0 and carry a calm CI; do not quote this row"
            );
        }
        // frankentorch-rled4: the SAME row under the min estimator, nulls
        // included. A null is a verdict about the instrument, so it must be
        // adjudicated on the estimator the row is quoted under — quoting a
        // min-estimator ratio behind a median-estimator null would be the
        // mixed-estimator error of NEGATIVE_EVIDENCE item 12 wearing a gate's
        // clothes.
        let (min_ratio, min_ratio_lo, min_ratio_hi) =
            median_ratio_ci(&pt_round_min[index], &ft_round_min[index]);
        let (pt_min_null, pt_min_lo, pt_min_hi) =
            median_ratio_ci(&pt_first_half_min[index], &pt_second_half_min[index]);
        let (ft_min_null, ft_min_lo, ft_min_hi) =
            median_ratio_ci(&ft_first_half_min[index], &ft_second_half_min[index]);
        let pt_min_ok = adjudicate_null(pt_min_lo, pt_min_hi, MAX_NULL_CI_WIDTH).is_quotable()
            && balanced_null_is_centered(pt_min_null);
        let ft_min_ok = adjudicate_null(ft_min_lo, ft_min_hi, MAX_NULL_CI_WIDTH).is_quotable()
            && balanced_null_is_centered(ft_min_null);
        let min_standing = if min_ratio >= 1.0 {
            format!("FT {min_ratio:.2}x FASTER")
        } else {
            format!("FT {:.2}x SLOWER", 1.0 / min_ratio)
        };
        println!(
            "    MIN-ESTIMATOR {min_standing:<19} ratio {min_ratio:.3} [{min_ratio_lo:.3},{min_ratio_hi:.3}]  PT null {pt_min_null:.3} {}  FT null {ft_min_null:.3} {}  {}",
            if pt_min_ok { "PASS" } else { "FAIL" },
            if ft_min_ok { "PASS" } else { "FAIL" },
            if pt_min_ok && ft_min_ok && load_quotable {
                "QUOTABLE under the min estimator"
            } else if pt_min_ok && ft_min_ok {
                "not quotable — LOAD-DRIFTED (nulls passed; the host did not hold still)"
            } else {
                "not quotable"
            }
        );
    }
    println!(
        "\nQuote a lane only if both A/A gates say PASS, their point estimates are within +/-{BALANCED_NULL_MAX_DEVIATION:.2} of 1.0, and parity is `match`. WIDE means the null's\n\
         CI exceeded {MAX_NULL_CI_WIDTH:.2} — the sample was too noisy to support ANY verdict, so the row is\n\
         undecidable rather than a win or a loss (frankentorch-8ieqm). Op-work ratios are NOT\n\
         comparable to the gauntlet's whole-step ratios, which include each lane's input rebuild."
    );

    // frankentorch-v92uh: prove the pool was actually SERVING this run rather than
    // sitting inert. Without this line a pooled/unpooled A/B that came out flat
    // would be indistinguishable from one where every `take` missed, and the
    // paired verdict below would be reporting on a lever that never engaged.
    // Requests below `MIN_POOLED_LEN`, and every request made while the pool is
    // disabled, are not counted at all — so these are pooled-lane numbers.
    let pool = ft_core::buffer_pool::stats();
    println!(
        "\nbuffer_pool: hits={} misses={} parked={} buffers / {:.1} MiB",
        pool.hits,
        pool.misses,
        pool.parked_buffers,
        pool.parked_bytes as f64 / (1024.0 * 1024.0)
    );

    // ── frankentorch-v92uh: PAIRED analysis for `X` vs its lever-off twin ───
    //
    // The table above compares two lanes through their independent medians, and
    // on a shared host that is the weakest reading available: a load excursion
    // lands in one lane's median and not the other's, and the CI has to be wide
    // enough to cover that. The pair is sampled ADJACENTLY inside each round, so
    // the excursion is very nearly common to both — differencing per round
    // cancels it, and only then is a busy host survivable.
    //
    // Per round each arm is reduced by its balanced-square median. That keeps
    // every arm's estimator paired to the same host window and matches the row
    // estimator above.
    //
    // Reported as `off / on`, so > 1.0 means the lever is FASTER. The incumbent
    // rows carry their own control: PT is byte-identical code under both names,
    // so PT(off)/PT(on) must land near 1.0 or the run is not readable at all.
    //
    // Three suffixes name a lever-off twin. `_nopool` is `ft_core::buffer_pool`
    // switched off (frankentorch-v92uh, -9pafs, -7zqbc); `_noshortcut` is the
    // PReLU+sum deforest declined through its hook exit (frankentorch-k1hto);
    // `_serialfwd` is the group-norm forward forced onto the pre-`group_norm_parallel_pays`
    // serial schedule (frankentorch-dmpho).
    for (index, (name, _)) in lanes.iter().enumerate() {
        let Some((base, lever)) = name
            .strip_suffix("_nopool")
            .map(|base| (base, "buffer pool"))
            .or_else(|| {
                name.strip_suffix("_noshortcut")
                    .map(|base| (base, "sum shortcut"))
            })
            .or_else(|| {
                name.strip_suffix("_serialfwd")
                    .map(|base| (base, "parallel forward gate"))
            })
            .or_else(|| {
                name.strip_suffix("_recompute")
                    .map(|base| (base, "forward-statistics reuse"))
            })
        else {
            continue;
        };
        let Some(base_index) = lanes.iter().position(|(other, _)| *other == base) else {
            continue;
        };
        let rounds = ft_times[index].len().min(ft_times[base_index].len());
        let mut treated = Vec::with_capacity(rounds);
        let mut control = Vec::with_capacity(rounds);
        for round in 0..rounds {
            control.push(ft_times[index][round]);
            treated.push(ft_times[base_index][round]);
        }
        let paired: Vec<f64> = control
            .iter()
            .zip(treated.iter())
            .map(|(off, on)| off / on)
            .collect();
        let (point, lo, hi) = median_ratio_ci(&control, &treated);
        let wins = paired.iter().filter(|ratio| **ratio > 1.0).count();
        // The control is PAIRED BY SAMPLE INDEX, not min-over-lane / min-over-lane.
        // On an incumbent round every lane is sampled, so `pt_times[a][k]` and
        // `pt_times[b][k]` were taken seconds apart in the SAME round and their
        // ratio cancels the host state they share. Two independent minima do not:
        // each picks whichever round happened to be quietest for that lane, and
        // those can be different rounds. Measured cost of getting this wrong on
        // 2026-08-14: the min/min control moved more than 5% in 10 of 12
        // invocations and vetoed rows whose FrankenTorch arms were clean, purely
        // because torch's own samples are noisy on this host.
        let pt_control = {
            let samples = pt_times[index].len().min(pt_times[base_index].len());
            if samples == 0 {
                f64::NAN
            } else {
                median(
                    (0..samples)
                        .map(|k| pt_times[index][k] / pt_times[base_index][k])
                        .collect(),
                )
            }
        };
        println!(
            "\nPAIRED {base}: {lever} ON vs OFF, one binary, one invocation, per-round square medians\n  \
             ratio (off/on) = {point:.3}x  95% CI [{lo:.3},{hi:.3}]  {wins}/{rounds} rounds faster with the {lever}\n  \
             incumbent control PT(off)/PT(on) = {pt_control:.3} (paired by sample index; must be ~1.0, the arm is identical code)\n  \
             verdict: {}",
            if !pt_control.is_finite() || (pt_control - 1.0).abs() >= 0.05 {
                "UNREADABLE — the incumbent control moved, so the host shifted between the two lanes"
                    .to_string()
            } else if lo > 1.0 {
                format!("the {lever} is FASTER by the paired CI")
            } else if hi < 1.0 {
                format!("the {lever} is SLOWER by the paired CI")
            } else {
                "UNDECIDED — the paired CI brackets 1.0".to_string()
            }
        );

        // frankentorch-rled4: the SAME comparison, same rounds, same pairing,
        // reduced by the per-round FLOOR instead of the per-round median. Both
        // arms use the same estimator — mixing a min from one arm with a median
        // from the other is the error recorded as NEGATIVE_EVIDENCE item 12, and
        // at 1.33-1.51x the estimator difference on this lane is larger than any
        // lever measured so far.
        //
        // This does NOT replace the median row above, which stays comparable with
        // everything already banked. It answers a different question: whether the
        // lever is resolvable at all once the neighbours are taken out of the
        // reading.
        let (min_point, min_lo, min_hi) = median_ratio_ci(
            &ft_round_min[index][..rounds],
            &ft_round_min[base_index][..rounds],
        );
        let min_wins = (0..rounds)
            .filter(|&k| ft_round_min[index][k] > ft_round_min[base_index][k])
            .count();
        let pt_control_min = {
            let samples = pt_round_min[index]
                .len()
                .min(pt_round_min[base_index].len());
            if samples == 0 {
                f64::NAN
            } else {
                median(
                    (0..samples)
                        .map(|k| pt_round_min[index][k] / pt_round_min[base_index][k])
                        .collect(),
                )
            }
        };
        let min_verdict = if !pt_control_min.is_finite() || (pt_control_min - 1.0).abs() >= 0.05 {
            "UNREADABLE — the incumbent control moved".to_string()
        } else if min_lo > 1.0 {
            format!("the {lever} is FASTER by the paired CI")
        } else if min_hi < 1.0 {
            format!("the {lever} is SLOWER by the paired CI")
        } else {
            "UNDECIDED — the paired CI brackets 1.0".to_string()
        };
        println!(
            "  --- same rounds, per-round MIN estimator (frankentorch-rled4) ---\n  \
             ratio (off/on) = {min_point:.3}x  95% CI [{min_lo:.3},{min_hi:.3}]  {min_wins}/{rounds} rounds faster\n  \
             incumbent control = {pt_control_min:.3}\n  \
             verdict: {min_verdict}\n  \
             estimator agreement: {}",
            if (point - 1.0).signum() == (min_point - 1.0).signum() {
                "the two estimators agree on DIRECTION"
            } else {
                "WARNING — the two estimators DISAGREE on direction; this row is not usable under either"
            }
        );
    }

    group_norm_f32_kernel_breakdown(&gnx, &gnw, &gnb);
    max_pool3d_kernel_breakdown(&mp3);

    Ok(())
}

/// Which of the three f32 GroupNorm kernels the lane's time is actually in.
///
/// Runs AFTER every paired lane has finished, so it cannot perturb a single
/// number above, and with the allocator already warm — which is the state the
/// lanes themselves ran in, not the cold state a standalone ladder would measure
/// (`frankentorch` in-situ-over-standalone finding: a standalone ladder inverted
/// in situ purely from allocator warmth).
///
/// This is a BREAKDOWN, not a lane: there is no PyTorch arm to pair against, so
/// its numbers are FrankenTorch-internal attribution only and no ratio derived
/// from them is a win. It exists because `group_norm_f32_kernels` reads 16.72x
/// against a live arm while the full session lane reads 11.77x, i.e. the tape is
/// not the term — so the next question is strictly which kernel, and guessing
/// between three of them is how the last four levers on this lane were chosen.
/// Where the `max_pool3d` lane's time actually is (`frankentorch-87sz8`).
///
/// The bead is titled "9.39x slower — largest confirmed vs-upstream gap". On the
/// current instrument the lane measures ~2.85x (NEGATIVE_EVIDENCE item 16), and
/// GroupNorm f32 at 6-7x is the larger loss — so the title is now mis-ranking the
/// queue. Before spending another lever here, price the two kernels the lane
/// actually calls, exactly as the GroupNorm breakdown did. That attribution is
/// what redirected `frankentorch-dmpho` away from a phase that turned out not to
/// be the term.
///
/// FrankenTorch-internal, min of N, so no incumbent arm and no ratio: this says
/// which of OUR kernels to attack, not how we compare to torch.
fn max_pool3d_kernel_breakdown(values: &[f64]) {
    let (od, oh, ow) = (MP3_D / 2, MP3_H / 2, MP3_W / 2);
    let reps = 9;
    let mut fwd = Vec::with_capacity(reps);
    let mut bwd = Vec::with_capacity(reps);
    let mut bwd_generic = Vec::with_capacity(reps);

    // Built once, outside every timed region: the backward needs an upstream
    // gradient and the offsets the forward produced, and allocating either inside
    // the clock would price harness bookkeeping as kernel cost.
    let (_, offsets) = ft_kernel_cpu::max_pool3d_forward_with_indices_f64(
        values, MP3_N, MP3_C, MP3_D, MP3_H, MP3_W, 2, 2, 2, od, oh, ow, 2, 2, 2,
    );
    let dout = vec![1.0f64; MP3_N * MP3_C * od * oh * ow];

    for _ in 0..reps {
        let started = Instant::now();
        let (out, args) = ft_kernel_cpu::max_pool3d_forward_with_indices_f64(
            values, MP3_N, MP3_C, MP3_D, MP3_H, MP3_W, 2, 2, 2, od, oh, ow, 2, 2, 2,
        );
        fwd.push(started.elapsed().as_secs_f64() * 1_000.0);
        std::hint::black_box((&out, &args));

        // ROUTE-MATCHED. The lane is kernel 2x2x2 with stride 2x2x2, so
        // `kd == sd && kh == sh && kw == sw` and ft-api's backward closure takes
        // the NON-OVERLAPPING specialisation, not the generic scatter. Timing the
        // generic one here would repeat NEGATIVE_EVIDENCE item 7f: a split whose
        // two arms enter different kernels measures the routing, not the phase.
        let started = Instant::now();
        let din = ft_kernel_cpu::max_pool3d_backward_from_indices_nonoverlapping_f64(
            &dout, &offsets, MP3_N, MP3_C, MP3_D, MP3_H, MP3_W, od, oh, ow,
        );
        bwd.push(started.elapsed().as_secs_f64() * 1_000.0);
        std::hint::black_box(&din);

        // The generic scatter beside it, on the same data in the same rep, so the
        // specialisation's own payoff is visible rather than assumed.
        let started = Instant::now();
        let din_generic = ft_kernel_cpu::max_pool3d_backward_from_indices_f64(
            &dout, &offsets, MP3_N, MP3_C, MP3_D, MP3_H, MP3_W, od, oh, ow,
        );
        bwd_generic.push(started.elapsed().as_secs_f64() * 1_000.0);
        assert!(
            din_generic
                .iter()
                .zip(din.iter())
                .all(|(a, b)| a.to_bits() == b.to_bits()),
            "the non-overlapping specialisation must agree with the generic scatter bit for bit"
        );
        std::hint::black_box(&din_generic);
    }

    let floor = |mut v: Vec<f64>| {
        v.sort_by(f64::total_cmp);
        v[0]
    };
    let (fwd, bwd, bwd_generic) = (floor(fwd), floor(bwd), floor(bwd_generic));
    let in_numel = MP3_N * MP3_C * MP3_D * MP3_H * MP3_W;
    let out_numel = MP3_N * MP3_C * od * oh * ow;
    println!(
        "\nMAX_POOL3D KERNEL BREAKDOWN (FrankenTorch-internal attribution, min of {reps}; NOT a ratio)\n  \
         input numel={in_numel}  output numel={out_numel}  kernel 2x2x2 stride 2x2x2  rayon_threads={}\n  \
         forward_with_indices_f64  {fwd:8.3} ms   reads {in_numel} elems, 8 comparisons per output\n  \
         backward NONOVERLAPPING   {bwd:8.3} ms   the route ft-api actually takes (kernel == stride); scatters {out_numel} values into a dense {in_numel}-element buffer\n  \
         backward generic scatter  {bwd_generic:8.3} ms   same data, same rep, bit-identical output — what the specialisation buys\n  \
         total on the live route   {:8.3} ms",
        rayon::current_num_threads(),
        fwd + bwd,
    );
}

fn group_norm_f32_kernel_breakdown(values: &[f32], weight: &[f32], bias: &[f32]) {
    let spatial = GN_H * GN_W;
    let channels_per_group = GN_C / GN_GROUPS;
    let numel = GN_N * GN_C * GN_H * GN_W;
    let units = GN_N * GN_GROUPS;
    let out_meta = TensorMeta::from_shape(
        vec![GN_N, GN_C, GN_H, GN_W],
        DType::F32,
        ft_core::Device::Cpu,
    );
    // The forward's own output feeds the sum, so it is built once up front rather
    // than inside either timed region.
    let forward_out = ft_kernel_cpu::group_norm_forward_f32(
        values,
        Some(weight),
        Some(bias),
        GN_N,
        GN_GROUPS,
        channels_per_group,
        spatial,
        1e-5,
    );

    let reps = 9;
    let mut fwd = Vec::with_capacity(reps);
    let mut fwd_serial = Vec::with_capacity(reps);
    let mut sum = Vec::with_capacity(reps);
    let mut bwd = Vec::with_capacity(reps);
    let mut reduce_seq = Vec::with_capacity(reps);
    let mut reduce_wide = Vec::with_capacity(reps);
    for _ in 0..reps {
        let started = Instant::now();
        let out = ft_kernel_cpu::group_norm_forward_f32(
            values,
            Some(weight),
            Some(bias),
            GN_N,
            GN_GROUPS,
            channels_per_group,
            spatial,
            1e-5,
        );
        fwd.push(started.elapsed().as_secs_f64() * 1_000.0);
        std::hint::black_box(&out);

        // The SAME kernel forced onto the other schedule, interleaved rep by rep
        // with the one above. This is the sentinel for the `_serialfwd` twin
        // lane: if the paired lane reports no effect, these two numbers say
        // whether that is because the schedules cost the same or because the flag
        // never reached the kernel. A paired lane that reads ~1.0 for a broken
        // switch looks exactly like one that reads ~1.0 for an honest null.
        let started = Instant::now();
        let out_serial = ft_kernel_cpu::group_norm_forward_f32_scheduled(
            values,
            Some(weight),
            Some(bias),
            GN_N,
            GN_GROUPS,
            channels_per_group,
            spatial,
            1e-5,
            false,
        );
        fwd_serial.push(started.elapsed().as_secs_f64() * 1_000.0);
        // Cheap on top of a 401,408-element kernel, and it turns the bit-identity
        // claim into something this run actually checks rather than cites.
        assert!(
            out_serial
                .iter()
                .zip(out.iter())
                .all(|(a, b)| a.to_bits() == b.to_bits()),
            "the two group_norm forward schedules must agree bit for bit"
        );
        std::hint::black_box(&out_serial);

        let started = Instant::now();
        let loss = ft_kernel_cpu::sum_tensor_contiguous_f32(&forward_out, &out_meta).expect("sum");
        sum.push(started.elapsed().as_secs_f64() * 1_000.0);
        std::hint::black_box(loss);

        let started = Instant::now();
        let (dx, _, _) = ft_kernel_cpu::group_norm_backward_scalar_f32(
            1.0f32,
            values,
            Some(weight),
            GN_N,
            GN_GROUPS,
            channels_per_group,
            spatial,
            1e-5,
        );
        bwd.push(started.elapsed().as_secs_f64() * 1_000.0);
        std::hint::black_box(&dx);

        // REDUCTION ARM (frankentorch-zv1y1). Every per-group statistic is a
        // scalar accumulation over `group_numel` elements, and float addition is
        // not associative, so LLVM may neither vectorise it nor break the
        // dependency chain. This prices what multiple independent accumulators
        // would buy BEFORE anyone is asked to accept the tolerance change that
        // reassociation implies — if it buys little, the question never needs
        // asking.
        //
        // Both arms read the SAME slice in the same cache state, interleaved rep
        // by rep, and both are reduced by min below: comparing a min against a
        // median is the error that cost this lane a whole bead
        // (NEGATIVE_EVIDENCE item 12), and at 1.33-1.51x the estimator difference
        // alone would swamp the effect being measured.
        let probe_group = &values[..channels_per_group * spatial];
        let started = Instant::now();
        let mut sequential = 0.0f32;
        for &v in probe_group {
            sequential += v;
        }
        reduce_seq.push(started.elapsed().as_secs_f64() * 1_000_000.0);
        std::hint::black_box(sequential);

        let started = Instant::now();
        let mut acc = [0.0f32; 8];
        let mut chunks = probe_group.chunks_exact(8);
        for chunk in &mut chunks {
            for (slot, &v) in acc.iter_mut().zip(chunk) {
                *slot += v;
            }
        }
        let mut widened = acc.iter().sum::<f32>();
        for &v in chunks.remainder() {
            widened += v;
        }
        reduce_wide.push(started.elapsed().as_secs_f64() * 1_000_000.0);
        std::hint::black_box(widened);
    }

    // ESTIMATOR ARM (frankentorch-59kjf). The three per-kernel figures above sum
    // to far less than the lane that runs the same three kernels — ~0.85 ms
    // against ~2.0 ms, i.e. most of that lane is outside its own kernels. Two
    // explanations have opposite consequences: per-rep allocation churn (a real
    // lever) or the ESTIMATOR (no lever at all — the lane takes a median of four
    // samples per round, this block takes a min, and on a host carrying a dozen
    // peer agents a median mostly measures the neighbours).
    //
    // So run the lane's exact work here and report it BOTH ways. Estimator is
    // then the only difference between the two numbers, inside one invocation on
    // one host. Adding a separate probe instead is what produced the harness
    // disagreement recorded as NEGATIVE_EVIDENCE item 11.
    let step_groups = 10;
    let mut step_samples = Vec::with_capacity(step_groups * 4);
    for _ in 0..step_groups * 4 {
        let started = Instant::now();
        // Exactly what `timed_group_norm_f32_kernels` times, including summing
        // the freshly produced forward output rather than a buffer built once.
        let out = ft_kernel_cpu::group_norm_forward_f32(
            values,
            Some(weight),
            Some(bias),
            GN_N,
            GN_GROUPS,
            channels_per_group,
            spatial,
            1e-5,
        );
        let loss = ft_kernel_cpu::sum_tensor_contiguous_f32(&out, &out_meta).expect("sum");
        let (dx, _, _) = ft_kernel_cpu::group_norm_backward_scalar_f32(
            1.0f32,
            values,
            Some(weight),
            GN_N,
            GN_GROUPS,
            channels_per_group,
            spatial,
            1e-5,
        );
        step_samples.push(started.elapsed().as_secs_f64() * 1_000.0);
        std::hint::black_box((&out, loss, &dx));
    }
    let step_min = step_samples
        .iter()
        .copied()
        .fold(f64::INFINITY, |acc, sample| acc.min(sample));
    // The lane's estimator: median of four samples per round, then the median of
    // those round medians. `median` here is the mean of the middle pair for an
    // even count, matching the harness's `paired_slot_median`.
    let mid = |mut v: Vec<f64>| {
        v.sort_by(f64::total_cmp);
        let n = v.len();
        if n.is_multiple_of(2) {
            (v[n / 2 - 1] + v[n / 2]) * 0.5
        } else {
            v[n / 2]
        }
    };
    let step_lane_estimator = mid(step_samples
        .chunks_exact(4)
        .map(|round| mid(round.to_vec()))
        .collect::<Vec<_>>());

    // Min, not median: this is attribution of a floor, and on a host carrying a
    // dozen other agents the median mostly measures the neighbours.
    let floor = |mut v: Vec<f64>| {
        v.sort_by(f64::total_cmp);
        v[0]
    };
    let (fwd, fwd_serial, sum, bwd) = (floor(fwd), floor(fwd_serial), floor(sum), floor(bwd));
    let (reduce_seq, reduce_wide) = (floor(reduce_seq), floor(reduce_wide));
    println!(
        "\nGROUP_NORM f32 KERNEL BREAKDOWN (FrankenTorch-internal attribution, min of {reps}; NOT a ratio)\n  \
         numel={numel}  groups={GN_GROUPS}  cpg={channels_per_group}  spatial={spatial}  rayon_threads={}\n  \
         forward_f32              {fwd:8.3} ms   schedule {}: {} groups; the numel-only gate ({} vs NORM_FWD_PARALLEL_MIN {}) would have said {}\n  \
         forward_f32 SERIAL       {fwd_serial:8.3} ms   same kernel, schedule forced off, interleaved rep-by-rep (sentinel for the _serialfwd lane)\n  \
         sum_tensor_contiguous_f32 {sum:7.3} ms\n  \
         backward_scalar_f32      {bwd:8.3} ms   (cpg==2 path, unconditionally parallel)\n  \
         total                    {:8.3} ms\n  \
         ---- same three kernels, one timed step, two estimators (frankentorch-59kjf) ----\n  \
         step min of {}            {step_min:8.3} ms   comparable to the per-kernel figures above\n  \
         step lane estimator      {step_lane_estimator:8.3} ms   median of 4 per round, then median over {step_groups} rounds\n  \
         estimator ratio          {:8.3}x  (lane estimator / min; if this alone reaches ~2.4x the lane's missing 57% is the ESTIMATOR, not allocation)\n  \
         ---- one group's reduction, sequential vs 8 accumulators (frankentorch-zv1y1) ----\n  \
         reduce sequential        {reduce_seq:8.3} us   {} elems, one dependency chain — what every statistic pass costs today\n  \
         reduce 8 accumulators    {reduce_wide:8.3} us   same data, same rep, chain broken 8 ways\n  \
         reduction headroom       {:8.3}x  (NOT bit-exact: reassociation moves mean/rstd/output by an ULP; this prices the tolerance question, it does not answer it)",
        rayon::current_num_threads(),
        if units >= 64 && numel >= (1 << 17) {
            "PARALLEL"
        } else {
            "SERIAL"
        },
        units,
        numel,
        1u32 << 19,
        if numel >= (1 << 19) {
            "PARALLEL"
        } else {
            "SERIAL"
        },
        fwd + sum + bwd,
        step_samples.len(),
        step_lane_estimator / step_min,
        channels_per_group * spatial,
        reduce_seq / reduce_wide,
    );
}

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
//! same workloads: `max_pool1d`, `avg_pool2d`, `conv3d`, `max_pool3d`.
//!
//! Run (must be local; rch workers have no PyTorch):
//! ```text
//! PYTORCH_PYTHON=/path/to/python \
//!   cargo run --release -p ft-api --features fair-alloc --example gauntlet_lane_sweep_h2h
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
use ft_core::ExecutionMode;

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
    let checksum = report.gradient(x).expect("grad").iter().sum::<f64>();
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
    let checksum = report.gradient(x).expect("grad").iter().sum::<f64>();
    (elapsed, checksum)
}

/// The same GroupNorm f32 work with NO session and NO tape: the two kernels
/// called directly, on f32 throughout.
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
fn timed_group_norm_f32_kernels(values: &[f32], weight: &[f32], bias: &[f32]) -> (f64, f64) {
    let spatial = GN_H * GN_W;
    let channels_per_group = GN_C / GN_GROUPS;
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
    // `sum` loss, so the upstream gradient is all ones — the same thing the
    // session lane's `tensor_sum` produces, kept in f32 here.
    let dy = vec![1.0f32; out.len()];
    let (dx, _, _) = ft_kernel_cpu::group_norm_backward_f32(
        &dy,
        values,
        Some(weight),
        GN_N,
        GN_GROUPS,
        channels_per_group,
        spatial,
        1e-5,
    );
    let elapsed = started.elapsed().as_secs_f64() * 1_000.0;
    let checksum = f64::from(dx.iter().sum::<f32>());
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
    let checksum = report.gradient(x).expect("grad").iter().sum::<f64>();
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
    return elapsed, x.grad.sum().item()
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
            Box::new(|| timed_group_norm_f32_kernels(&gnx, &gnw, &gnb)),
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
    for (_, run_lane) in &lanes {
        let mut warm = 0.0;
        for _ in 0..3 {
            warm += run_lane().0;
        }
        std::hint::black_box(warm);
    }

    let mut ft_times: Vec<Vec<f64>> = vec![Vec::with_capacity(reps); lanes.len()];
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
            pt_times[index].push(median(incumbent_slots));
            ft_times[index].push(median(ft_slots));
        }
    }

    writeln!(stdin, "{QUIT_REQUEST}")?;
    stdin.flush()?;
    drop(stdin);
    child.wait()?;

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

    // ── frankentorch-v92uh: PAIRED analysis for `X` vs `X_nopool` ───────────
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
    // Reported as `nopool / pooled`, so > 1.0 means the pool is FASTER. The
    // incumbent rows carry their own control: PT is byte-identical code under
    // both names, so PT(nopool)/PT(pooled) must land near 1.0 or the run is not
    // readable at all.
    for (index, (name, _)) in lanes.iter().enumerate() {
        let Some(base) = name.strip_suffix("_nopool") else {
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
            "\nPAIRED {base}: pool ON vs OFF, one binary, one invocation, per-round square medians\n  \
             ratio (off/on) = {point:.3}x  95% CI [{lo:.3},{hi:.3}]  {wins}/{rounds} rounds faster with the pool\n  \
             incumbent control PT(off)/PT(on) = {pt_control:.3} (paired by sample index; must be ~1.0, the arm is identical code)\n  \
             verdict: {}",
            if !pt_control.is_finite() || (pt_control - 1.0).abs() >= 0.05 {
                "UNREADABLE — the incumbent control moved, so the host shifted between the two lanes"
            } else if lo > 1.0 {
                "the pool is FASTER by the paired CI"
            } else if hi < 1.0 {
                "the pool is SLOWER by the paired CI"
            } else {
                "UNDECIDED — the paired CI brackets 1.0"
            }
        );
    }
    Ok(())
}

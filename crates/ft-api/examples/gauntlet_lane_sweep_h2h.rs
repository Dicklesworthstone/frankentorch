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

/// Total CPU seconds this process and its REAPED children have burned, from `/proc/self/stat`.
///
/// `frankentorch-vyaia` acceptance item 2 asks for the harness's own load to be re-derived at the
/// current lane count. The obvious instrument — watch `loadavg` rise while we run — cannot do it
/// on this box: the first full sweep under it read `+50.18` while three peer `rustc` processes
/// were live, and no amount of waiting makes a shared machine quiet on demand.
///
/// This one does not care. Mean parallelism is our own CPU time divided by wall time, and a peer's
/// compile contributes exactly zero to our `utime`. It is the same quantity loadavg approximates
/// — the average number of tasks we kept runnable — measured on us alone.
///
/// `cutime`/`cstime` cover the incumbent arm, which is a child process and is reaped by
/// `child.wait()` before this is sampled, so the returned figure is BOTH arms.
///
/// The comm field can contain spaces and parentheses, so the split is after the LAST `)`, never on
/// whitespace from the start.
fn self_cpu_seconds() -> Option<f64> {
    let stat = std::fs::read_to_string("/proc/self/stat").ok()?;
    let after_comm = &stat[stat.rfind(')')? + 1..];
    let fields: Vec<&str> = after_comm.split_whitespace().collect();
    // `after_comm` starts at field 3 (state), so field N lives at index N - 3:
    // utime 14, stime 15, cutime 16, cstime 17.
    let ticks: u64 = [11usize, 12, 13, 14]
        .iter()
        .map(|&i| fields.get(i).and_then(|v| v.parse::<u64>().ok()))
        .sum::<Option<u64>>()?;
    // USER_HZ, which the kernel fixes at 100 for /proc regardless of CONFIG_HZ.
    Some(ticks as f64 / 100.0)
}

/// Threads the incumbent arm runs with, reported in the provenance block and used as half of the
/// self-load ceiling in `frankentorch-vyaia`'s EXTERNAL LOAD check. It was a bare `8` in the
/// provenance call; the ceiling needs the same number, and two literals that must agree is how
/// a constant stops being one.
const INCUMBENT_THREADS: usize = 8;

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

// frankentorch-58zjz. dgemm_tb's column gate is `n > 4*m` with m = out_features and
// n = in_features, so LIN_OUT_WIDE clears it (1024 > 512) and LIN_OUT_NARROW does not
// (1024 > 2048 is false). Both are far past PAR_MIN_FLOPS_COLS, so the gate is the only
// difference between the two lanes.
// frankentorch-58zjz item 126d. 3x3 stride-1 pad-1, so the output extents equal the input's and
// the 2026-07-05 all-ones adjoint is eligible. Sized so the im2col panel (8192 x 288 f64,
// 18.9 MB) is the same order as conv3d's, keeping the two lanes comparable.
// frankentorch-58zjz: THE ATTENTION LANE THE BEAD NAMED AND NOBODY BUILT.
//
// The bead's list of `dgemm_tb` callers has two attention entries (ft-kernel-cpu `lib.rs:5214` and
// `:5298`, the dV gradient), and item 119 gave that GEMM a column-parallel path on the strength of
// a conv3d row. Linear lanes landed under item 126 and conv2d lanes under items 144/209; attention
// was the last of the three the bead opened with and had no live incumbent arm at all.
//
// [4, 8, 256, 64] is a small transformer block's self-attention: the scores matrix is
// 4*8*256*256 = 2.1M elements, so the lane spends real time in softmax and in BOTH matmuls rather
// than in leaf construction. Deliberately ONE lane, not a straddling pair — the bead asks for
// shapes either side of the `n > 4*m` gate, and which side a given attention shape lands on depends
// on the argument mapping into `dgemm_tb`, which this lane is the instrument for discovering rather
// than something to assert in a constant.
const ATTN_B: usize = 4;
const ATTN_H: usize = 8;
const ATTN_S: usize = 256;
const ATTN_D: usize = 64;

const C2_N: usize = 8;
/// The `conv2d_big*` twins' batch — `frankentorch-hi9r6`, item 144.
///
/// Item 137 could not certify EITHER conv2d lane: our A/A null passed 5/5 and PyTorch's failed
/// 5/5, and item 137c blamed the incumbent arm's duration (2.3-3.1 ms) rather than the host,
/// pointing at `frankentorch-uilzh`'s group_norm precedent where a lane only nulled once the
/// incumbent sat in the 4-7 ms band.
///
/// That is a HYPOTHESIS about why the null fails, and doubling the batch is what tests it. If
/// PyTorch's null passes here at ~5-6 ms, the duration explanation holds and these twins become
/// conv2d's certifiable lanes; if it fails here too, the explanation is refuted and the cause is
/// something other than arm length. Either way the answer is a ledger item.
///
/// The twins are ADDED rather than swapped in. Item 137c proposed resizing in place and called
/// the loss of comparability with items 128 and 137 "the price"; that price only buys something
/// if the resize works, and it is not yet known that it does. Keeping both sizes costs two lanes
/// of sweep time and keeps the old rows meaningful, and the small pair can be retired later on
/// evidence rather than on the expectation of it.
///
/// BATCH is the dimension doubled, deliberately. It scales `flat` (8192 -> 16384) and so both
/// GEMMs proportionally, while leaving `out_ch` and `patch_width` fixed -- item 136 measured
/// GEMM efficiency rising with `out_ch`, so growing THAT would move the lane along the very
/// curve the bead is trying to measure and confound the resize with a shape change.
const C2B_N: usize = 16;
/// The f32 conv2d lanes' batch — `frankentorch-hi9r6`, item 191.
///
/// NOT `C2_N`, and the difference is the whole reason the lane can be certified. f32 conv2d runs
/// about twice as fast as f64, so at the f64 lane's batch of 8 the INCUMBENT arm measures
/// **0.656 ms** — four times shorter than the ~3 ms arm item 137c already blamed for conv2d's
/// nulls failing 5 of 5, and twice as short as the 0.199-0.310 ms GroupNorm arm that `uilzh`
/// resized 16x after it nulled 0 of 11.
///
/// Sized by MEASURING the incumbent rather than by picking a round multiple, at 8 torch threads:
///
///     batch      8     16     32     48     64     96
///     summed  0.756  0.943  1.754  3.671  3.236  5.790 ms
///     masked  0.703  0.981  1.763  3.667  3.402  5.789 ms
///
/// **The "4-7 ms band" this constant originally targeted was wrong, and item 203 measured it.** In
/// one settled invocation the incumbent A/A null tracked arm duration monotonically:
///
///     conv2d               PT  3.217 ms   FAIL
///     conv2d_masked        PT  3.390 ms   OFFSET
///     conv2d_masked_train  PT  5.081 ms   OFFSET
///     conv2d_big           PT 11.019 ms   PASS
///     conv2d_big_masked    PT 11.590 ms   PASS
///
/// 5.08 ms is OFFSET, not PASS, so a lane sized to ~5.8 ms would have been born unquotable for the
/// second time — the very failure this constant exists to avoid. 160 puts both routes near 11 ms,
/// which is where the incumbent actually nulls. BATCH is the axis grown, per item 144: it scales `flat` and both
/// GEMMs proportionally while leaving `out_ch` and `patch_width` fixed, so the resize does not
/// slide the lane along the `out_ch` GEMM-efficiency curve item 136 measured — which is the very
/// thing this bead is trying to see.
///
/// This lane is therefore NOT comparable with the f64 conv2d rows, and is not meant to be. It
/// carries its own control (`conv2d_f32_masked`) at the same size.
const C2F32_N: usize = 160;
/// The SUMMED conv2d lane's batch — `frankentorch-hi9r6`, item 209.
///
/// WHY A THIRD SIZE. `conv2d_big` is the only lane that reaches the all-ones adjoints items
/// 174/176/177 rewrote, and across four invocations it certified ONCE: PT PASS/FT PASS, then
/// PASS/OFFSET, WIDE/OFFSET, OFFSET/PASS, with the ratio spanning 1.135-1.421. Item 208 showed
/// doubling the rounds does not fix it — more rounds narrows an interval, it does not move a
/// biased point estimate.
///
/// What DOES separate the lanes on this board is arm duration, and the evidence is now four-deep:
///
///     lane                FT arm    PT arm    certified
///     conv2d_big           8.8 ms   10.5 ms   1 of 4
///     conv2d_big_masked   27.9 ms   11.0 ms   4 of 4
///
/// So our arm at ~9 ms is the short one, and ~28 ms demonstrably nulls. The incumbent side was
/// measured directly here at 8 torch threads, and our side extrapolated linearly from the 8.8 ms
/// at `C2B_N` — sound because every phase of the summed route (the dweight column-sum, the dpadded
/// scatter, im2col and the forward GEMM) is linear in batch:
///
///     batch   PT ms    our arm, extrapolated
///        16    9.406    8.8
///        32   17.917   17.6
///        48   26.412   26.4
///        64   32.908   35.2
///
/// 64 puts BOTH arms past every duration that has certified on this board. 48 would put ours at
/// 26.4 ms, just under the 27.9 that works, and item 203 is one turn old and exactly about sizing
/// a lane to the edge of a threshold. The margin is deliberate.
///
/// Frozen weight and no mask, matching `conv2d_big` exactly, so the two differ only in size.
const C2XL_N: usize = 64;
const C2_CI: usize = 32;
const C2_CO: usize = 32;
const C2_H: usize = 32;
const C2_W: usize = 32;
const C2_K: usize = 3;

const LIN_B: usize = 512;
const LIN_IN: usize = 1024;
const LIN_OUT_WIDE: usize = 128;
const LIN_OUT_NARROW: usize = 512;

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
// frankentorch-uilzh: RESIZED 16x from the scorecard's [8,64,28,28]. At that size
// the incumbent arm ran 0.199-0.310 ms — an order of magnitude shorter than any
// other lane here — and its A/A null failed one-sided in 11 of 11 runs of one
// binary, so no GroupNorm ratio from this harness was ever gate-quotable
// (NEGATIVE_EVIDENCE item 24b, and MossyOtter's item 27 showing the arms perturb
// each other monotonically in lane duration). 16x puts the incumbent near 4-5 ms,
// in the band where max_pool3d_nopool, conv3d and prelu all null cleanly.
//
// This BREAKS COMPARABILITY with every group_norm row banked before it. That is
// intended: those rows are withdrawn precisely because the lane could not be
// gated at the old size.
const GN_N: usize = 32;
const GN_C: usize = 64;
const GN_H: usize = 56;
const GN_W: usize = 56;
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
    use super::{
        balanced_null_is_centered, group_norm_dense_dy, median, paired_slot_median, timed_conv3d,
    };

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

    #[test]
    fn group_norm_dense_loss_derivative_cannot_take_the_scalar_backward() {
        // `sum(out*out)` supplies `2*out`, not the all-ones upstream that the scalar
        // GroupNorm backward specializes. This is the route contract for mdsmm's direct-kernel
        // twins; keeping it as a small pure test avoids conflating that contract with timing.
        let dy = group_norm_dense_dy(&[-0.75, 0.0, 0.25, 1.5]);
        assert_eq!(dy, vec![-1.5, 0.0, 0.5, 3.0]);
        assert!(dy.iter().any(|value| value.to_bits() != 1.0_f32.to_bits()));
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

/// frankentorch-yc7ud: forward + SQUARED-sum loss + backward. The squared loss is the
/// whole point: `timed_op` uses a plain `tensor_sum`, which fires the pooling ops'
/// scalar sum-shortcut and routes the backward through
/// `avg_pool1d_backward_scalar_f64`. With `sum(out * out)` the upstream gradient is
/// `2*out` rather than a uniform scalar, the shortcut does not fire, and the DENSE
/// `avg_pool1d_backward_f64` runs — which is the route yc7ud's lever is in and which no
/// existing lane exercised.
fn timed_op_sq<F>(values: &[f64], shape: Vec<usize>, build: F) -> (f64, f64)
where
    F: Fn(&mut FrankenTorchSession, ft_autograd::TensorNodeId) -> ft_autograd::TensorNodeId,
{
    let mut session = FrankenTorchSession::new(ExecutionMode::Strict);
    let x = session
        .tensor_variable(values.to_vec(), shape, true)
        .expect("leaf");
    let started = Instant::now();
    let out = build(&mut session, x);
    let sq = session.tensor_mul(out, out).expect("square");
    let loss = session.tensor_sum(sq).expect("sum");
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

/// GroupNorm f32 on the DENSE route — `frankentorch-68pwz`, NEGATIVE_EVIDENCE item 103c.
///
/// WHY THIS LANE HAD TO EXIST. `timed_group_norm_f32` ends in a plain `tensor_sum`, which
/// fires the GroupNorm sum-shortcut: its backward takes a SCALAR upstream and never
/// materializes a per-element `dy`. An execution sentinel put a number on that — the scored
/// lane calls `narrow_f64_to_f32` **zero** times, while the same op under a non-sum loss
/// calls it once over all 6,422,528 elements. So every f32-norm engine lever that lives in
/// the dy/dx conversion boundary was invisible to the board: there was no lane on the route
/// that executes it, and an FT-only paired toggle is maintenance, not a win.
///
/// This lane is that route with a live incumbent beside it. The loss is `sum(out*out)`, and
/// the PyTorch twin returns `Fn.group_norm(...)**2` so the sample loop's `fn(x).sum()`
/// gives the incumbent the SAME loss — the convention `avg_pool1d_dense` established for
/// exactly this reason (`frankentorch-yc7ud`). Without the square the two arms would time
/// different work and parity would mismatch.
fn timed_group_norm_f32_dense(values: &[f32], weight: &[f32], bias: &[f32]) -> (f64, f64) {
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
    let squared = session.tensor_mul(out, out).expect("square");
    let loss = session.tensor_sum(squared).expect("sum");
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

/// BatchNorm2d f32 — the half of `frankentorch-68pwz` that has never had a lane at all.
///
/// The bead has carried "BatchNorm2d is about 5.7x slower" as an UNVERIFIED figure since it
/// was filed, explicitly because there was no h2h lane and therefore no way to check it with
/// the trusted instrument. Every number this bead has produced so far is GroupNorm's. These
/// two lanes are the first live incumbent arm BatchNorm2d has had.
///
/// SHAPE AND PARAMETERS ARE THE GroupNorm FIXTURES, deliberately: same `[32,64,56,56]` input
/// and the same 64-channel affine pair, so a BatchNorm row can be read directly against the
/// GroupNorm rows beside it without a second shape to control for.
///
/// `running_mean`/`running_var` are None on BOTH arms. Torch's `F.batch_norm` UPDATES running
/// statistics in place when they are passed under `training=True`, so a lane that supplied
/// them would mutate its own fixture between samples and neither arm would be timing the same
/// work twice. None keeps both arms pure and still exercises the training path, which is the
/// one with the backward.
///
/// `dense` selects the loss: `false` is `sum(out)`, which fires whatever sum-shortcut exists;
/// `true` is `sum(out*out)`, the route ordinary training takes. Item 109 showed those are
/// different backwards for GroupNorm, and the point of carrying both here is to find out
/// whether BatchNorm splits the same way.
/// BatchNorm2d on the DENSE route in f64 — the lane that can actually CERTIFY.
///
/// WHY AN f64 LANE EXISTS BESIDE THE f32 ONE. The f32 BatchNorm lanes cannot clear this
/// harness's parity gate at ANY shape, and that is arithmetic rather than a defect in either
/// implementation. The gate is `1e-6` relative on a gradient checksum; two different f32
/// batch-norm implementations differ by ~`2e-5` relative on these gradients, because the
/// gradient VALUES are f32 and the two libraries round their intermediates differently.
/// Measured, with an f64 arbiter placing both arms on the same axis
/// (`examples/bn_parity_arbiter.rs`): after fixing our accumulators, ours sits `1.966e-5`
/// from the f64 truth and PyTorch's sits `2.501e-5`, and the two differ from EACH OTHER by
/// 0.616 absolute against a tolerance of 0.0138. The gate is 20x below the noise floor of
/// the quantity it is gating.
///
/// f64 removes that floor: both arms carry the same values to ~1e-16, so the checksum
/// comparison means what it claims to. This is the convention the board's `max_pool`,
/// `avg_pool` and `conv3d` lanes already use, and it is why THEY certify.
///
/// It measures the f64 BatchNorm path, not the f32 one — stated plainly rather than papered
/// over. The f32 dense lane is kept alongside precisely so the f32 engine's timing stays
/// visible; it simply reports MISMATCH and is not quotable, which is the honest label for it.
fn timed_batch_norm2d_f64_dense(values: &[f64], weight: &[f64], bias: &[f64]) -> (f64, f64) {
    let mut session = FrankenTorchSession::new(ExecutionMode::Strict);
    let x = session
        .tensor_variable(values.to_vec(), vec![GN_N, GN_C, GN_H, GN_W], true)
        .expect("leaf");
    let w = session
        .tensor_variable(weight.to_vec(), vec![GN_C], true)
        .expect("weight");
    let b = session
        .tensor_variable(bias.to_vec(), vec![GN_C], true)
        .expect("bias");
    let started = Instant::now();
    let (out, _, _) = session
        .functional_batch_norm2d(x, None, None, Some(w), Some(b), true, 0.1, 1e-5)
        .expect("batch_norm2d");
    let squared = session.tensor_mul(out, out).expect("square");
    let loss = session.tensor_sum(squared).expect("sum");
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

fn timed_batch_norm2d_f32(values: &[f32], weight: &[f32], bias: &[f32], dense: bool) -> (f64, f64) {
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
    let (out, _, _) = session
        .functional_batch_norm2d(x, None, None, Some(w), Some(b), true, 0.1, 1e-5)
        .expect("batch_norm2d");
    let scored = if dense {
        session.tensor_mul(out, out).expect("square")
    } else {
        out
    };
    let loss = session.tensor_sum(scored).expect("sum");
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

/// Derivative of `sum(out*out)` with respect to `out`.
///
/// This deliberately stays separate from the scalar-loss kernel helpers below. The kernel
/// families they invoke take a scalar upstream by construction; the dense H2H twins must pass
/// this full `dy` through `group_norm_backward_f32` so the all-ones branch cannot hide the route
/// the session's dense loss uses.
fn group_norm_dense_dy(out: &[f32]) -> Vec<f32> {
    out.iter().map(|value| 2.0 * value).collect()
}

/// Direct-kernel counterpart of the GroupNorm dense session lane.
///
/// `reuse_stats` preserves the forward configuration of the existing scalar kernel lanes. There
/// is intentionally no claim that the scalar-only stats-reuse backward is exercised here: the
/// dense backward has no such entry point, so it recomputes its statistics through the real
/// generic kernel. These rows make that absence visible instead of mislabeling a scalar route as
/// a dense kernels-vs-engine split.
fn timed_group_norm_f32_kernels_dense_inner(
    values: &[f32],
    weight: &[f32],
    bias: &[f32],
    parallel_forward: bool,
    reuse_stats: bool,
) -> (f64, f64) {
    let spatial = GN_H * GN_W;
    let channels_per_group = GN_C / GN_GROUPS;
    let out_meta = TensorMeta::from_shape(
        vec![GN_N, GN_C, GN_H, GN_W],
        DType::F32,
        ft_core::Device::Cpu,
    );
    let started = Instant::now();
    let out = if reuse_stats {
        let (out, _stats) = ft_kernel_cpu::group_norm_forward_f32_with_cpg2_stats(
            values,
            Some(weight),
            Some(bias),
            GN_N,
            GN_GROUPS,
            spatial,
            1e-5,
        );
        out
    } else {
        ft_kernel_cpu::group_norm_forward_f32_scheduled(
            values,
            Some(weight),
            Some(bias),
            GN_N,
            GN_GROUPS,
            channels_per_group,
            spatial,
            1e-5,
            parallel_forward,
        )
    };
    let squared: Vec<f32> = out.iter().map(|value| value * value).collect();
    let loss = ft_kernel_cpu::sum_tensor_contiguous_f32(&squared, &out_meta).expect("sum");
    let dy = group_norm_dense_dy(&out);
    assert!(
        dy.iter().any(|value| value.to_bits() != 1.0_f32.to_bits()),
        "dense loss must not enter GroupNorm's all-ones scalar backward"
    );
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
    assert!(loss.is_finite(), "group_norm f32 dense loss must be finite");
    let checksum = dx
        .iter()
        .map(|gradient| f64::from(gradient.abs()))
        .sum::<f64>();
    (elapsed, checksum)
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
///
/// `square_loss` selects the DENSE-route arm (frankentorch-mdsmm). `defeat_shortcut` disables
/// the FUSION while leaving the upstream gradient uniform; `square_loss` instead makes the
/// upstream gradient `2*out`, so `try_prelu_sum_shortcut`'s own predicate has to decline. Those
/// are different questions and item 111 left the second one unmeasured for prelu.
///
/// Squaring is valid here because `tensor_prelu` returns the ELEMENTWISE output — the shortcut
/// is applied later, inside `tensor_sum`. The `functional_*_sum` entries return an
/// already-reduced scalar and must never be squared; confusing the two cost a retracted P0
/// (NEGATIVE_EVIDENCE item 112).
fn timed_prelu(
    values: &[f64],
    weight: &[f64],
    defeat_shortcut: bool,
    square_loss: bool,
) -> (f64, f64) {
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
    let reduced = if square_loss {
        session.tensor_mul(out, out).expect("square")
    } else {
        out
    };
    let loss = session.tensor_sum(reduced).expect("sum");
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

/// Conv2d, both loss routes — `frankentorch-58zjz`, item 126d.
///
/// conv2d shares conv3d's im2col decomposition and has the SAME all-ones fast path
/// (`conv2d_backward_3x3_stride1_ones_dout_f64`, 2026-07-05), so it has both a summed route and
/// a generic one and, until now, no lane for either. Item 124 found the remaining conv3d gap is
/// ALGORITHMIC — im2col + col2im are 46% of a backward PyTorch does not pay — and if that
/// transfers, conv2d is where it shows up next.
///
/// `mask` multiplies the output by a non-uniform tensor before the sum, exactly as
/// `timed_conv3d_masked` does, so the upstream gradient is `mask` rather than all-ones and the
/// call reaches the generic backward. It is a leaf, so its construction sits outside the timer
/// on both arms.
///
/// `batch` is a parameter rather than `C2_N` so the item 144 `conv2d_big*` twins share this exact
/// body: a resize whose two sizes ran through different code would test the code as much as the
/// size.
/// `weight_grad` decides whether the WEIGHT requires grad — `frankentorch-hi9r6`, item 182.
///
/// Every conv2d lane on this board has passed `false`, which means PyTorch and FrankenTorch have
/// both been measured on a step that computes NO weight gradient. `timed_linear` took the opposite
/// decision deliberately, and says why in its own comment: a no-grad weight "would skip dweight
/// and measure the wrong half". Conv2d never got that treatment.
///
/// This is not a cosmetic difference. Item 178 found that our first-order backward computed
/// `dweight` regardless and threw it away — about 3.0 ms of a 16.6 ms backward — and fixed it by
/// honouring `needs_input_grad`. That fix is worth ~18% on a lane whose weight is frozen and
/// exactly nothing on a lane whose weight is not, so which lane the board carries decides whether
/// item 178 reads as a real training win or as a faster route around work a training step needs.
///
/// Answering that by ARGUMENT is what this campaign keeps getting wrong, so both are now measured.
fn timed_conv2d(
    values: &[f64],
    weights: &[f64],
    mask: Option<&[f64]>,
    batch: usize,
    weight_grad: bool,
) -> (f64, f64) {
    let mut session = FrankenTorchSession::new(ExecutionMode::Strict);
    let x = session
        .tensor_variable(values.to_vec(), vec![batch, C2_CI, C2_H, C2_W], true)
        .expect("conv2d leaf");
    let w = session
        .tensor_variable(
            weights.to_vec(),
            vec![C2_CO, C2_CI, C2_K, C2_K],
            weight_grad,
        )
        .expect("conv2d weight");
    let m = mask.map(|values| {
        session
            .tensor_variable(values.to_vec(), vec![batch, C2_CO, C2_H, C2_W], false)
            .expect("conv2d mask")
    });
    let started = Instant::now();
    let out = session
        .functional_conv2d(x, w, None, (1, 1), (1, 1))
        .expect("conv2d");
    let scored = match m {
        Some(mask_leaf) => session.tensor_mul(out, mask_leaf).expect("mask multiply"),
        None => out,
    };
    let loss = session.tensor_sum(scored).expect("sum");
    // THE FORWARD/BACKWARD BOUNDARY — `frankentorch-hi9r6`.
    //
    // With the streamed `dweight` shipped, the frames on this lane read pad 0.870 | forward
    // 1.072 | backward 3.680 ms of KERNEL work against a 7.161 ms lane, which leaves ~2.0 ms
    // (28%) that is session and tape. That residue is now the largest single frame and nothing
    // has ever split it. This boundary does: everything before it is the forward session (the
    // internal pad, the fused mask multiply, the sum, and the tape nodes for all three),
    // everything after is `tensor_backward` (the tape walk, the dsum broadcast, the gradient
    // allocation). Subtracting the kernels-only lane's own forward and backward then attributes
    // each half separately instead of leaving one 2 ms lump.
    //
    // Read OUTSIDE the timed arithmetic in the sense that matters — it is one `Instant::now()`
    // between two operations that were already sequential, so it adds no work to either side.
    let forward_ms = started.elapsed().as_secs_f64() * 1_000.0;
    let report = session.tensor_backward(loss).expect("backward");
    let elapsed = started.elapsed().as_secs_f64() * 1_000.0;
    CONV2D_SESSION_SPLIT_MS.with(|cell| cell.borrow_mut().push((forward_ms, elapsed - forward_ms)));
    let checksum = report
        .gradient(x)
        .expect("grad")
        .iter()
        .map(|g| g.abs())
        .sum::<f64>();
    (elapsed, checksum)
}

/// conv2d f32 train step — `frankentorch-hi9r6`, item 191. The dtype twin of [`timed_conv2d`],
/// and the only lane that can price items 179, 181, 185 and 187.
///
/// WHY THIS LANE DID NOT EXIST UNTIL NOW. Item 185e declined to write it because the harness's
/// parity gate is a flat `1e-6` relative on a gradient checksum, and this file already records two
/// f32 lanes' trouble with it: `group_norm_f32` clears it only by accumulating the checksum in
/// f64, and the f32 BatchNorm sum lane was DELETED rather than left red because its `dx` is
/// analytically zero. Guessing which of those an f32 conv2d lane resembles would have put a red or
/// uninformative row on a board a dozen agents read.
///
/// IT WAS MEASURED INSTEAD, with torch alone and no FrankenTorch involved. Torch has two
/// independent f32 implementations of this convolution — the fused kernel and an unfold+matmul
/// composition — so their disagreement estimates what any second implementation would show:
///
///     route     f32-native vs f32-unfold      f32-native vs f64
///     summed              7.012e-08                 8.729e-08
///     masked              9.640e-09                 9.271e-09
///
/// Both clear `1e-6` with 14-100x of margin, so the lane is gate-able and the deferral is lifted.
///
/// READ ITS PARITY COLUMN WEAKLY, THOUGH. The same experiment puts the WORST PER-ELEMENT relative
/// disagreement at **4.0e-03** on the masked route — five orders of magnitude above what the
/// checksum shows, because `sum(|dx|)` over 262,144 elements lets independent errors cancel. On an
/// f64 lane that headroom is irrelevant (per-element error ~1e-16); on this one it means `match`
/// certifies "computed the same thing", NOT "computed it to the same bits". The bit-level claims
/// for these routes live in the kernel crate's differential tests, which is where they belong.
///
/// The lane runs at `C2F32_N`, not `C2_N`, because at the f64 batch the incumbent arm is 0.656 ms
/// and could not have nulled; see that constant for the measured sizing table.
///
/// `weight_grad` mirrors `timed_conv2d`'s parameter for the same reason item 182 added it: a
/// frozen weight and a training weight exercise different halves, and item 187 gave f32 the
/// `needs_input_grad` skip whose whole effect is on the frozen one.
/// conv2d f32 with NO session and NO tape — the kernels-only twin of `conv2d_f32`.
///
/// `frankentorch-qif1n`. Two certified rows say our f32 conv2d is **1.28x SLOWER per sample than
/// our own f64** (`conv2d_f32` 3.07x slower than torch; `conv2d_xl`, f64, 1.72x FASTER). A probe on
/// the raw kernels says the opposite: `conv2d_forward_f32` and `conv2d_backward_f32` each beat
/// their f64 twins by 1.53-1.55x, stable across a 2.5x batch change. Both cannot be true of the
/// same code path, so roughly 2x of f32-specific cost sits OUTSIDE those two kernels — and this
/// lane is the subtraction that prices it.
///
/// Follows the `group_norm_f32_kernels` precedent exactly: same shape, same dtype, same incumbent
/// op under a second name, but our arm calls `ft_kernel_cpu` directly. `conv2d_f32` minus this lane
/// IS the session and tape cost, measured inside one invocation against one live incumbent, with
/// PT(kernels)/PT(f32) as a free ~1.0 control.
///
/// WHAT IS DELIBERATELY INSIDE THE TIMER: the pad, and the crop of `dpadded` back to the input
/// extent. The session pays both inside its own timed region, so charging them here keeps the
/// subtraction honest — leaving them out would move cost into the very residue being measured.
///
/// The all-ones `dout` mirrors the summed loss the paired lane uses. `conv2d_backward_f32` picks
/// its route from `conv2d_ones_dout_route`, so passing ones takes the same route the session's
/// backward takes rather than a different one.
thread_local! {
    /// Milliseconds the kernels-only conv2d f32 lane spent on its hand-rolled pad, which is inside
    /// its timed region but is NOT kernel work. See `timed_conv2d_f32_kernels`.
    static CONV2D_F32_KERNELS_PAD_MS: std::cell::Cell<f64> = const { std::cell::Cell::new(0.0) };
}

fn timed_conv2d_f32_kernels(values: &[f32], weights: &[f32], batch: usize) -> (f64, f64) {
    let ph = C2_H + 2;
    let pw = C2_W + 2;
    let started = Instant::now();
    // THE PAD IS INSIDE THE TIMER, AND IT IS NOT FREE — read the diagnostic below before using
    // `conv2d_f32 - conv2d_f32_kernels` as "the session cost".
    //
    // That subtraction has been reading NEGATIVE (64 rounds: session 26.260 ms, kernels 29.396 ms
    // — the kernels-only arm SLOWER than the arm that also carries a session and a tape), which is
    // not a thing session overhead can do. The cause is here: this arm hand-rolls its padding as a
    // `vec![0.0f32; ..]` zero-init of 5.92M elements (~23.7 MB at this lane's batch 160) plus a
    // SERIAL scalar row copy over 5.24M elements, while the session arm reaches the same padded
    // buffer through `tensor_pad`, a real tape op. Both arms legitimately pay for padding — the
    // session arm's `let started` sits before `functional_conv2d`, which pads internally — but
    // they do not pay the SAME pad, and the difference is charged to "session" with the wrong
    // sign.
    //
    // The timed region is deliberately LEFT ALONE so this lane stays byte-comparable with every
    // banked row. Instead the pad is measured separately and reported, so the correction is
    // available to anyone doing the subtraction: the session cost is
    // `conv2d_f32 - (conv2d_f32_kernels - pad_ms)`, not `conv2d_f32 - conv2d_f32_kernels`.
    let pad_started = Instant::now();
    // Pad [batch, CI, H, W] -> [batch, CI, H+2, W+2] with a one-pixel zero border.
    let mut padded = vec![0.0f32; batch * C2_CI * ph * pw];
    for bc in 0..batch * C2_CI {
        let src = bc * C2_H * C2_W;
        let dst = bc * ph * pw;
        for row in 0..C2_H {
            let from = src + row * C2_W;
            let to = dst + (row + 1) * pw + 1;
            padded[to..to + C2_W].copy_from_slice(&values[from..from + C2_W]);
        }
    }
    let pad_ms = pad_started.elapsed().as_secs_f64() * 1_000.0;
    CONV2D_F32_KERNELS_PAD_MS.with(|cell| cell.set(pad_ms));
    let _out = ft_kernel_cpu::conv2d_forward_f32(
        &padded, weights, None, batch, C2_CI, ph, pw, C2_K, C2_K, C2_H, C2_W, 1, 1, C2_CO,
    );
    let dout = vec![1.0f32; batch * C2_CO * C2_H * C2_W];
    let (dpadded, _dweight, _dbias) = ft_kernel_cpu::conv2d_backward_f32(
        &dout, &padded, weights, batch, C2_CI, ph, pw, C2_K, C2_K, C2_H, C2_W, 1, 1, C2_CO, false,
    );
    // Crop back to the input extent, which is what the session hands back as the leaf gradient.
    // f64 accumulator per the `timed_group_norm_f32` precedent: summing this many f32 magnitudes in
    // f32 carries enough error to swamp the agreement being checked.
    let mut checksum = 0.0f64;
    for bc in 0..batch * C2_CI {
        let dst = bc * ph * pw;
        for row in 0..C2_H {
            let to = dst + (row + 1) * pw + 1;
            for value in &dpadded[to..to + C2_W] {
                checksum += f64::from(value.abs());
            }
        }
    }
    let elapsed = started.elapsed().as_secs_f64() * 1_000.0;
    (elapsed, checksum)
}

/// The FT-side conv2d frame diagnostics, printed BEFORE the PyTorch gate.
/// `frankentorch-hi9r6`.
///
/// Every number here is our own arm's timing, so none of it needs an incumbent. They used to sit
/// BELOW the `PyTorch row missing` early-continue, which meant the one way to run this
/// attribution on a quiet rch worker — `FT_H2H_NO_INCUMBENT=1`, since no worker has torch — was
/// also the one way to guarantee it printed nothing. That is how a 50-minute wait for a
/// contended local window became the only route to numbers that never needed the incumbent.
fn conv2d_frame_diagnostics(name: &str, ft_ms: f64) {
    if name == "conv2d_masked_train_dwpanel" {
        let calls = CONV2D_STREAMED_CALLS.with(std::cell::Cell::get);
        let legacy = CONV2D_LEGACY_CALLS.with(std::cell::Cell::get);
        println!(
            "    sentinel: streamed dweight ran {calls} time(s) in the incumbent \
             conv2d_masked_train and {legacy} time(s) in this forced-legacy arm. The pair \
             prices the lever only if the first is NONZERO and the second is ZERO — a lever \
             that never executed and a lever with no effect read identically."
        );
    }
    if name == "conv2d_masked_train" {
        let (fwd, bwd) = CONV2D_SESSION_SPLIT_MS.with(|cell| {
            let samples = cell.borrow();
            (
                median(samples.iter().map(|s| s.0).collect()),
                median(samples.iter().map(|s| s.1).collect()),
            )
        });
        if fwd > 0.0 || bwd > 0.0 {
            println!(
                "    session split: forward (pad + conv + fused mask + sum) {fwd:.3} ms | \
                 backward (tape walk + dsum + grad alloc) {bwd:.3} ms of this lane's \
                 {ft_ms:.3} ms. Subtract conv2d_masked_train_kernels' own forward/backward \
                 frames to get the session and tape cost of each half separately."
            );
        }
    }
    if name == "conv2d_masked_train_kernels" {
        let (pad, fwd, bwd) = CONV2D_KERNELS_SPLIT_MS.with(|cell| {
            let samples = cell.borrow();
            (
                median(samples.iter().map(|s| s.0).collect()),
                median(samples.iter().map(|s| s.1).collect()),
                median(samples.iter().map(|s| s.2).collect()),
            )
        });
        if fwd > 0.0 || bwd > 0.0 {
            println!(
                "    frames: pad {pad:.3} ms | forward {fwd:.3} ms | backward {bwd:.3} ms \
                 of this lane's {ft_ms:.3} ms. Session cost = conv2d_masked_train - (this - \
                 pad); the uncorrected subtraction charges a pad the session pays in another \
                 form."
            );
        }
    }
}

thread_local! {
    /// `(forward_ms, backward_ms)` from the last `timed_conv2d` call: the SESSION-level split,
    /// forward being everything up to and including `tensor_sum` and backward being
    /// `tensor_backward` alone. Paired with the kernels-only lane's frames this separates the
    /// ~2 ms of session/tape that is now `conv2d_masked_train`'s largest unattributed frame.
    /// EVERY sample, not the last one. The lane's own figure is a MEDIAN over rounds, so a
    /// frame reported from a single sample is on a different estimator and cannot be compared
    /// with it — measured: on a worker whose load tripled mid-run the last-sample frames summed
    /// to 156% of the lane's median. `feedback_estimator_and_provenance` is the standing rule;
    /// this is it recurring inside a diagnostic rather than inside a claim.
    static CONV2D_SESSION_SPLIT_MS: std::cell::RefCell<Vec<(f64, f64)>> =
        const { std::cell::RefCell::new(Vec::new()) };
}

thread_local! {
    /// How many times the streamed `dweight` route ran in the SHIPPED `conv2d_masked_train`
    /// lane. Zero means the incumbent arm is not on the path this pair claims to price.
    static CONV2D_STREAMED_CALLS: std::cell::Cell<u64> = const { std::cell::Cell::new(0) };
    /// The same count for the forced-legacy `conv2d_masked_train_dwpanel` arm. It must be 0.
    static CONV2D_LEGACY_CALLS: std::cell::Cell<u64> = const { std::cell::Cell::new(0) };
}

thread_local! {
    /// `(pad_ms, forward_ms, backward_ms)` from the last `timed_conv2d_masked_train_kernels`
    /// call. See that function.
    static CONV2D_KERNELS_SPLIT_MS: std::cell::RefCell<Vec<(f64, f64, f64)>> =
        const { std::cell::RefCell::new(Vec::new()) };
}

/// f64 conv2d TRAINING step with NO session and NO tape — the kernels-only twin of
/// `conv2d_masked_train`. `frankentorch-hi9r6`.
///
/// WHY IT EXISTS. Item 223 priced conv2d's frames by DIFFERENCING two certified rows
/// (`conv2d_big_masked_train` minus `conv2d_big_masked`) and got `dweight` = 1.40x against
/// 2.38x for everything else. That instrument can separate exactly one component, and it has
/// been used on the only pair the board had. The frames inside the 2.38x — the FORWARD, the
/// backward, and the session/tape surround — have never been separated on the f64 lane at all,
/// so every lever aimed at them has been aimed by argument.
///
/// This lane separates them in ONE invocation against ONE live incumbent:
///
/// ```text
///   conv2d_masked_train - (this - pad_ms)   = the session and tape cost
///   this row's forward_ms / backward_ms     = the split inside the kernels
/// ```
///
/// It follows `timed_conv2d_f32_kernels` exactly, including its correction: the hand-rolled pad
/// is INSIDE the timed region (so the row stays comparable with the session lane, which pads
/// inside `functional_conv2d`) but is also measured and printed, because the two arms do not pay
/// the SAME pad and charging the difference to "session" gets the sign wrong.
///
/// THE `dout` IS THE MASK, not all-ones, and that is the whole point: `conv2d_backward_f64`
/// routes on `conv2d_ones_dout_route` plus an all-ones scan, so an all-ones `dout` would take
/// the 3x3 adjoint and measure a different kernel than the lane it is the twin of.
///
/// The checksum is `sum |crop(dpadded)|`, which is what the session lane's `report.gradient(x)`
/// sums, so the parity column compares the same quantity on both.
fn timed_conv2d_masked_train_kernels(
    values: &[f64],
    weights: &[f64],
    mask: &[f64],
    batch: usize,
) -> (f64, f64) {
    let ph = C2_H + 2;
    let pw = C2_W + 2;
    // Drain the streamed-dweight counter on entry. This lane's panel is above the gate, so it
    // DOES stream, and leaving its increments undrained is what mis-attributed 160 executions to
    // the forced-legacy arm of the paired lane.
    let _ = ft_kernel_cpu::take_conv2d_dweight_streamed_calls();
    let started = Instant::now();
    let pad_started = Instant::now();
    let mut padded = vec![0.0f64; batch * C2_CI * ph * pw];
    for bc in 0..batch * C2_CI {
        let src = bc * C2_H * C2_W;
        let dst = bc * ph * pw;
        for row in 0..C2_H {
            let from = src + row * C2_W;
            let to = dst + (row + 1) * pw + 1;
            padded[to..to + C2_W].copy_from_slice(&values[from..from + C2_W]);
        }
    }
    let pad_ms = pad_started.elapsed().as_secs_f64() * 1_000.0;

    let fwd_started = Instant::now();
    let _out = ft_kernel_cpu::conv2d_forward_f64(
        &padded, weights, None, batch, C2_CI, ph, pw, C2_K, C2_K, C2_H, C2_W, 1, 1, C2_CO,
    );
    let fwd_ms = fwd_started.elapsed().as_secs_f64() * 1_000.0;

    let bwd_started = Instant::now();
    let (dpadded, _dweight, _dbias) = ft_kernel_cpu::conv2d_backward_f64(
        mask, &padded, weights, batch, C2_CI, ph, pw, C2_K, C2_K, C2_H, C2_W, 1, 1, C2_CO, false,
    );
    let bwd_ms = bwd_started.elapsed().as_secs_f64() * 1_000.0;
    CONV2D_KERNELS_SPLIT_MS.with(|cell| cell.borrow_mut().push((pad_ms, fwd_ms, bwd_ms)));

    let mut checksum = 0.0f64;
    for bc in 0..batch * C2_CI {
        let dst = bc * ph * pw;
        for row in 0..C2_H {
            let to = dst + (row + 1) * pw + 1;
            for value in &dpadded[to..to + C2_W] {
                checksum += value.abs();
            }
        }
    }
    let elapsed = started.elapsed().as_secs_f64() * 1_000.0;
    (elapsed, checksum)
}

fn timed_conv2d_f32(
    values: &[f32],
    weights: &[f32],
    mask: Option<&[f32]>,
    batch: usize,
    weight_grad: bool,
) -> (f64, f64) {
    let mut session = FrankenTorchSession::new(ExecutionMode::Strict);
    let x = session
        .tensor_variable_f32(values.to_vec(), vec![batch, C2_CI, C2_H, C2_W], true)
        .expect("conv2d f32 leaf");
    let w = session
        .tensor_variable_f32(
            weights.to_vec(),
            vec![C2_CO, C2_CI, C2_K, C2_K],
            weight_grad,
        )
        .expect("conv2d f32 weight");
    let m = mask.map(|values| {
        session
            .tensor_variable_f32(values.to_vec(), vec![batch, C2_CO, C2_H, C2_W], false)
            .expect("conv2d f32 mask")
    });
    let started = Instant::now();
    let out = session
        .functional_conv2d(x, w, None, (1, 1), (1, 1))
        .expect("conv2d f32");
    let scored = match m {
        Some(mask_leaf) => session.tensor_mul(out, mask_leaf).expect("mask multiply"),
        None => out,
    };
    let loss = session.tensor_sum(scored).expect("sum");
    let report = session.tensor_backward(loss).expect("backward");
    let elapsed = started.elapsed().as_secs_f64() * 1_000.0;
    // f64 accumulator, per the `timed_group_norm_f32` precedent: summing 262,144 f32 magnitudes
    // in f32 carries enough error on its own to swamp the gradients' actual agreement.
    let checksum = report
        .gradient(x)
        .expect("grad")
        .iter()
        .map(|g| g.abs())
        .sum::<f64>();
    (elapsed, checksum)
}

/// Linear train step: `y = x @ W^T`, with the WEIGHT requiring grad — the only reason this lane
/// exists (`frankentorch-58zjz`).
///
/// `linear_backward_f64`'s `dweight` is `gemm::dgemm_tb(out_features, batch, in_features, ..)`,
/// and item 119 gave `dgemm_tb` its first parallel path. That entry is also conv2d's and
/// attention's weight-gradient GEMM, but item 120 recorded that none of them had a lane — so a
/// library-wide scheduling change shipped on the strength of a conv3d row. This lane prices it
/// on the op that dominates its use.
///
/// A no-grad weight would skip `dweight` entirely and measure the wrong half, which is why
/// `requires_grad` is true on both arms — the same reason `prelu` and `group_norm` carry
/// grad-requiring parameters.
fn timed_linear(
    values: &[f64],
    weight: &[f64],
    batch: usize,
    in_f: usize,
    out_f: usize,
) -> (f64, f64) {
    let mut session = FrankenTorchSession::new(ExecutionMode::Strict);
    let x = session
        .tensor_variable(values.to_vec(), vec![batch, in_f], true)
        .expect("linear leaf");
    let w = session
        .tensor_variable(weight.to_vec(), vec![out_f, in_f], true)
        .expect("linear weight");
    let started = Instant::now();
    let out = session.functional_linear(x, w, None).expect("linear");
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

thread_local! {
    /// `(forward_ms, loss_sum_ms, backward_ms)` from the last `timed_attention` call.
    /// The three frames are timed inside the same operation interval as the live H2H lane.
    static ATTENTION_SPLIT_MS: std::cell::Cell<(f64, f64, f64)> =
        const { std::cell::Cell::new((0.0, 0.0, 0.0)) };
}

/// Scaled dot-product attention, `[B, H, S, D]`, with Q, K and V all requiring grad.
///
/// `frankentorch-58zjz`. The bead's point is that `dgemm_tb` — given a column-parallel path by item
/// 119 — is the dV gradient for every attention backward in the library and had never faced a live
/// incumbent. All three inputs require grad so the backward reaches dV, dK and the softmax's dQ
/// path rather than only one of them.
///
/// The timed region matches every other lane on this board: forward, loss sum, backward, with the
/// leaves built OUTSIDE the timer. The checksum is the query gradient, which is the leaf the
/// incumbent arm also differentiates, so the two sides are compared on the same tensor.
fn timed_attention(query: &[f64], key: &[f64], value: &[f64]) -> (f64, f64) {
    let mut session = FrankenTorchSession::new(ExecutionMode::Strict);
    let shape = vec![ATTN_B, ATTN_H, ATTN_S, ATTN_D];
    let q = session
        .tensor_variable(query.to_vec(), shape.clone(), true)
        .expect("attention query");
    let k = session
        .tensor_variable(key.to_vec(), shape.clone(), true)
        .expect("attention key");
    let v = session
        .tensor_variable(value.to_vec(), shape, true)
        .expect("attention value");
    let started = Instant::now();
    let forward_started = Instant::now();
    let out = session
        .functional_scaled_dot_product_attention(q, k, v, None, false, None)
        .expect("scaled_dot_product_attention");
    let forward_ms = forward_started.elapsed().as_secs_f64() * 1_000.0;
    let loss_started = Instant::now();
    let loss = session.tensor_sum(out).expect("sum");
    let loss_sum_ms = loss_started.elapsed().as_secs_f64() * 1_000.0;
    let backward_started = Instant::now();
    let report = session.tensor_backward(loss).expect("backward");
    let backward_ms = backward_started.elapsed().as_secs_f64() * 1_000.0;
    ATTENTION_SPLIT_MS.with(|cell| cell.set((forward_ms, loss_sum_ms, backward_ms)));
    let elapsed = started.elapsed().as_secs_f64() * 1_000.0;
    let checksum = report
        .gradient(q)
        .expect("grad")
        .iter()
        .map(|g| g.abs())
        .sum::<f64>();
    (elapsed, checksum)
}

/// Conv3d under a NON-UNIFORM loss — the only lane on this board that reaches conv3d's
/// GENERIC backward.
///
/// Every other conv3d lane ends in `tensor_sum(out)`, so the output gradient is exactly all
/// `+1.0` and `conv3d_backward_f64` takes its all-ones fast path. That is a legitimate case
/// (`loss.backward()` on a summed output) but it is NOT what training does, and it means the
/// generic route — the one a real objective reaches — has never had a live incumbent arm.
/// NEGATIVE_EVIDENCE item 104 removed a 28.3 MB im2col panel from precisely that route and
/// could not be measured at all for this reason; item 108d named this lane as the fix.
///
/// The loss is `(out * mask).sum()`, which makes the incoming gradient `mask` rather than
/// ones. The mask is built from the same `seq` generator both arms use, so the two sides
/// multiply by bit-identical values, and it is deliberately non-uniform: a constant mask
/// would still be a uniform `dout` and would land back on the fast path.
///
/// The extra elementwise multiply is real work and is charged to BOTH arms — 131,072
/// elements against a lane in the millisecond range — so it shifts both sides together and
/// does not favour either. It is inside the timed region on both arms because it is part of
/// the loss, which is what `loss_sum` in the declared region covers. frankentorch-l2zki.
fn timed_conv3d_masked(
    values: &[f64],
    input_shape: Vec<usize>,
    weights: &[f64],
    weight_shape: Vec<usize>,
    mask: &[f64],
    out_shape: Vec<usize>,
) -> (f64, f64) {
    let mut session = FrankenTorchSession::new(ExecutionMode::Strict);
    let x = session
        .tensor_variable(values.to_vec(), input_shape, true)
        .expect("conv3d leaf");
    let weight = session
        .tensor_variable(weights.to_vec(), weight_shape, false)
        .expect("conv3d weight");
    // The mask is a leaf too, and it is built here for the same reason the weight is: the
    // PyTorch arm makes it once during setup, so constructing it inside the timer would
    // charge tensor construction to one arm only.
    let mask_leaf = session
        .tensor_variable(mask.to_vec(), out_shape, false)
        .expect("conv3d mask");
    let started = Instant::now();
    let out = session
        .functional_conv3d(x, weight, None, (1, 1, 1), (1, 1, 1))
        .expect("conv3d");
    let scaled = session.tensor_mul(out, mask_leaf).expect("mask multiply");
    let loss = session.tensor_sum(scaled).expect("sum");
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
    // Establish FrankenTorch's hardware-aware global pool before this harness asks rayon for
    // provenance. `rayon::current_num_threads()` lazily creates the default 64-thread pool, so
    // doing the query first would hide the library's no-environment policy from every lane.
    // `RAYON_NUM_THREADS`, when explicitly set by a caller, still wins in `configure_global_pool`.
    let _ = ft_kernel_cpu::pool::configure_global_pool();
    // frankentorch-yu1zm: THE A/A CONTROL FOR THE UNINIT PAIR. With
    // `FT_POOL_ZEROED_OUTPUT=1` the global default becomes the zeroed path, and the
    // `max_pool1d_zeroed` lane — which sets the toggle and then RESTORES the previous
    // value — leaves both arms on the zeroed path. The paired ratio must then collapse
    // to ~1.0.
    //
    // This control is not optional. Adding the twin lane moved the BASE lane from
    // 17.3 ms to 8.6 ms, so lane composition perturbs this lane by 2x on its own, and
    // a paired ratio between two positions that are not interchangeable would report a
    // lever effect that is really an ordering effect. The A/A is what tells the two
    // apart, and until it has run the A/B number means nothing.
    ft_kernel_cpu::init_pool_output_toggle_from_env();
    let reps = reps();

    let mp1 = seq(MP1_N * MP1_C * MP1_L);
    let ap2 = seq(AP2_N * AP2_C * AP2_H * AP2_W);
    let c3x = seq(C3_N * C3_CI * C3_D * C3_H * C3_W);
    let c3w = seq(C3_CO * C3_CI * C3_K * C3_K * C3_K);
    // frankentorch-l2zki: the conv3d_masked lane's loss weights, over the OUTPUT shape.
    // stride 1 / pad 1 / k 3 makes the output extents equal the input's, so this is the same
    // element count as `c3x` and the same `seq` call the python arm makes for `c3m` — the two
    // arms multiply by bit-identical values, which is what keeps the gradient checksum a
    // parity check rather than a coincidence.
    let c3m = seq(C3_N * C3_CO * C3_D * C3_H * C3_W);
    // frankentorch-58zjz linear fixtures, same `seq` generator as the python arm.
    let c2x = seq(C2_N * C2_CI * C2_H * C2_W);
    let c2w = seq(C2_CO * C2_CI * C2_K * C2_K);
    let c2m = seq(C2_N * C2_CO * C2_H * C2_W);
    // item 144's doubled-batch twins. Same `seq` generator, same weights -- only the batch
    // differs, so any change in the ratio is the resize and not a different workload.
    // f32 twins of the conv2d fixtures — item 191. Same generator then `as f32`, which is exactly
    // what `.float()` does to the same f64 values on the incumbent arm, so both arms carry
    // identical bits and the parity column means what it claims. Sized at `C2F32_N`, not `C2_N`,
    // for the reason that constant documents.
    #[allow(clippy::cast_possible_truncation)]
    let c2x32: Vec<f32> = seq(C2F32_N * C2_CI * C2_H * C2_W)
        .into_iter()
        .map(|value| value as f32)
        .collect();
    #[allow(clippy::cast_possible_truncation)]
    let c2w32: Vec<f32> = c2w.iter().map(|&v| v as f32).collect();
    #[allow(clippy::cast_possible_truncation)]
    let c2m32: Vec<f32> = seq(C2F32_N * C2_CO * C2_H * C2_W)
        .into_iter()
        .map(|value| value as f32)
        .collect();
    // item 209: the summed route at a size where BOTH arms are long enough to null.
    let c2xlx = seq(C2XL_N * C2_CI * C2_H * C2_W);
    let c2bx = seq(C2B_N * C2_CI * C2_H * C2_W);
    let c2bm = seq(C2B_N * C2_CO * C2_H * C2_W);
    let attnq = seq(ATTN_B * ATTN_H * ATTN_S * ATTN_D);
    let attnk = seq(ATTN_B * ATTN_H * ATTN_S * ATTN_D);
    let attnv = seq(ATTN_B * ATTN_H * ATTN_S * ATTN_D);
    let linx = seq(LIN_B * LIN_IN);
    let linw_wide = seq(LIN_OUT_WIDE * LIN_IN);
    let linw_narrow = seq(LIN_OUT_NARROW * LIN_IN);
    let mp3 = seq(MP3_N * MP3_C * MP3_D * MP3_H * MP3_W);
    // Built by the SAME formula the python arm uses, then cast — so the two arms
    // normalize identical numbers and the gradient checksum is a real parity check
    // rather than a coincidence of shapes.
    // frankentorch-68pwz: the f64 BatchNorm lane's fixtures. Same generator and same shape
    // as the f32 ones, so the two BatchNorm rows describe the same workload in two dtypes.
    let bnx: Vec<f64> = seq(GN_N * GN_C * GN_H * GN_W);
    let bnw: Vec<f64> = seq(GN_C)
        .into_iter()
        .map(|value| value * 10.0 + 1.0)
        .collect();
    let bnb: Vec<f64> = seq(GN_C).into_iter().map(|value| value * 3.0).collect();
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
# frankentorch-l2zki: the conv3d_masked lane's loss weights. Same `seq` generator as every
# other tensor here and the same one the Rust arm uses, so both sides multiply by
# bit-identical values. Non-uniform on purpose -- a constant mask is still a uniform `dout`
# and would land back on the all-ones fast path this lane exists to avoid. Output shape.
c3m=seq(2*32*8*16*16).reshape(2,32,8,16,16)
# frankentorch-58zjz linear fixtures. requires_grad on the WEIGHTS so this arm computes dweight
# too, matching the FrankenTorch side; the checksum stays x.grad on both.
c2x=seq(8*32*32*32).reshape(8,32,32,32)
c2w=seq(32*32*3*3).reshape(32,32,3,3)
c2m=seq(8*32*32*32).reshape(8,32,32,32)
# item 182: the SAME weight values, but requiring grad, so the incumbent computes
# dweight too. A lane where only OUR arm skips the weight gradient would not be a
# comparison, it would be a handicap.
c2w_train=seq(32*32*3*3).reshape(32,32,3,3).requires_grad_(True)
# item 191: f32 twins. `.float()` here, `as f32` there, from the SAME f64 generator, so both
# arms carry identical bits. Measured justification for the lane existing at all is in
# `timed_conv2d_f32`'s comment: two independent f32 conv implementations agree to 7.0e-08 on
# the summed route and 9.6e-09 on the masked one, well inside the 1e-6 parity gate.
# Batch 160, NOT 8: f32 conv2d is ~2x faster than f64, and at batch 8 this arm measures
# 0.656 ms. Item 203 measured where the incumbent actually nulls: OFFSET at 5.08 ms, PASS at
# 11.0 and 11.6 ms. 160 puts both routes near 11 ms. Measured sizing table is in
# C2F32_N's comment on the Rust side. Keep this in lockstep with C2F32_N.
c2x32=seq(160*32*32*32).reshape(160,32,32,32).float()
c2w32=c2w.float()
c2m32=seq(160*32*32*32).reshape(160,32,32,32).float()
# item 144: the doubled-batch twins, same weights, same generator.
# item 209: summed route at batch 64, where both arms clear every duration that has
# certified on this board. Keep in lockstep with C2XL_N.
c2xlx=seq(64*32*32*32).reshape(64,32,32,32)
c2bx=seq(16*32*32*32).reshape(16,32,32,32)
c2bm=seq(16*32*32*32).reshape(16,32,32,32)
attnq=seq(4*8*256*64).reshape(4,8,256,64)
attnk=seq(4*8*256*64).reshape(4,8,256,64).requires_grad_(True)
attnv=seq(4*8*256*64).reshape(4,8,256,64).requires_grad_(True)
linx=seq(512*1024).reshape(512,1024)
linw_wide=seq(128*1024).reshape(128,1024).requires_grad_(True)
linw_narrow=seq(512*1024).reshape(512,1024).requires_grad_(True)
mp3=seq(2*32*16*32*32).reshape(2,32,16,32,32)
# frankentorch-kgs4.115 GroupNorm f32 train step, shape and groups copied verbatim
# from the scorecard row so the two describe the same workload. f32 on BOTH sides:
# `.float()` here, `tensor_variable_f32` there. The affine parameters require grad,
# which is the whole point of the row — the f32 no-grad path has long been fused,
# and it is the GRAD path the scorecard measured at 19.04x.
gnx=seq(32*64*56*56).reshape(32,64,56,56).float()   # frankentorch-uilzh: keep in lockstep with GN_N/GN_C/GN_H/GN_W
gnw=(seq(64)*10.0+1.0).float().requires_grad_(True)
gnb=(seq(64)*3.0).float().requires_grad_(True)
# frankentorch-68pwz: f64 BatchNorm fixtures — same generator and shape as the f32 ones, no
# `.float()`, so the f64 lane's two arms carry identical values to ~1e-16 and its parity
# checksum means what it claims. The f32 BatchNorm lane cannot say that at any shape.
bnx=seq(32*64*56*56).reshape(32,64,56,56)
bnw=(seq(64)*10.0+1.0).requires_grad_(True)
bnb=(seq(64)*3.0).requires_grad_(True)
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
    # frankentorch-58zjz: linear with a GRAD-REQUIRING weight, which is the point -- a no-grad
    # weight skips dweight and measures the wrong half. Two shapes straddling dgemm_tb's
    # `in_features > 4*out_features` column gate.
    # frankentorch-58zjz item 126d: conv2d's two loss routes. The *c2m makes the upstream
    # gradient non-uniform, so the second lane reaches the generic backward as conv3d_masked does.
    "conv2d":        (c2x, lambda x: Fn.conv2d(x,c2w,None,(1,1),(1,1))),
    "conv2d_masked": (c2x, lambda x: Fn.conv2d(x,c2w,None,(1,1),(1,1))*c2m),
    # item 144: the same two routes at double the batch, to test whether a ~5-6 ms incumbent
    # arm nulls where a ~3 ms one would not.
    "conv2d_big":        (c2bx, lambda x: Fn.conv2d(x,c2w,None,(1,1),(1,1))),
    # item 209: the summed route, long enough on BOTH arms to null.
    "conv2d_xl":         (c2xlx, lambda x: Fn.conv2d(x,c2w,None,(1,1),(1,1))),
    # item 212: same incumbent code under a second name -- PT(legacy)/PT(xl) is a free ~1.0
    # control. The FrankenTorch side of this name runs the pre-item-174 scatter.
    "conv2d_xl_legacy":  (c2xlx, lambda x: Fn.conv2d(x,c2w,None,(1,1),(1,1))),
    "conv2d_big_masked": (c2bx, lambda x: Fn.conv2d(x,c2w,None,(1,1),(1,1))*c2bm),
    # item 220: identical twin; only OUR arm's tile floor differs between the two lanes.
    "conv2d_big_masked_tile": (c2bx, lambda x: Fn.conv2d(x,c2w,None,(1,1),(1,1))*c2bm),
    # item 216: the train twin of the line above -- c2w_train, so BOTH arms compute dweight.
    # Sized at batch 16 because that is the one masked conv2d lane measured certifying 4 of 4.
    "conv2d_big_masked_train": (c2bx, lambda x: Fn.conv2d(x,c2w_train,None,(1,1),(1,1))*c2bm),
    # hi9r6 dinput-blocking: same incumbent code under two more names, so PT(panel)/PT(base) is a
    # free ~1.0 control on each pair. Only OUR arm differs -- the `_panel` names run the
    # pre-blocking dpanel + col2im dinput route.
    "conv2d_big_masked_panel": (c2bx, lambda x: Fn.conv2d(x,c2w,None,(1,1),(1,1))*c2bm),
    "conv2d_big_masked_train_panel": (c2bx, lambda x: Fn.conv2d(x,c2w_train,None,(1,1),(1,1))*c2bm),
    # item 190: byte-identical twin of conv2d_masked, so the warm/cold A/B is one invocation.
    "conv2d_masked_warm": (c2x, lambda x: Fn.conv2d(x,c2w,None,(1,1),(1,1))*c2m),
    # item 182: masked route with a GRAD-REQUIRING weight on BOTH arms.
    "conv2d_masked_train": (c2x, lambda x: Fn.conv2d(x,c2w_train,None,(1,1),(1,1))*c2m),
    # frankentorch-hi9r6: the kernels-only twin. Our arm calls ft_kernel_cpu directly with no
    # session and no tape, so conv2d_masked_train minus (this - pad_ms) is the session cost, and
    # PT(this)/PT(conv2d_masked_train) is a free ~1.0 control on the window.
    "conv2d_masked_train_kernels": (c2x, lambda x: Fn.conv2d(x,c2w_train,None,(1,1),(1,1))*c2m),
    # frankentorch-hi9r6: the legacy panel-dweight arm of the same lane. Same torch code under
    # a second name, so PT(dwpanel)/PT(train) is a free ~1.0 control on the window.
    "conv2d_masked_train_dwpanel": (c2x, lambda x: Fn.conv2d(x,c2w_train,None,(1,1),(1,1))*c2m),
    # frankentorch-hi9r6: same incumbent code under a second name, so PT(panel)/PT(base) is a free
    # ~1.0 control. Only OUR arm differs -- `_panel` runs the pre-164e159d dpanel + col2im dinput,
    # which is what this batch-8 lane took before the channel-group route opened the gate.
    "conv2d_masked_train_panel": (c2x, lambda x: Fn.conv2d(x,c2w_train,None,(1,1),(1,1))*c2m),
    # item 191: f32 conv2d, both loss routes. The summed one is the only lane that reaches the
    # f32 all-ones adjoints; the masked one is its control on the generic route.
    "conv2d_f32":        (c2x32, lambda x: Fn.conv2d(x,c2w32,None,(1,1),(1,1))),
    # frankentorch-qif1n: the same torch op again for the kernels-only pair. Our arm calls
    # ft_kernel_cpu directly with no session or tape, so conv2d_f32 minus conv2d_f32_kernels is the
    # session cost. PT(kernels)/PT(f32) is a free control that must land near 1.0.
    "conv2d_f32_kernels": (c2x32, lambda x: Fn.conv2d(x,c2w32,None,(1,1),(1,1))),
    "conv2d_f32_masked": (c2x32, lambda x: Fn.conv2d(x,c2w32,None,(1,1),(1,1))*c2m32),
    # frankentorch-hi9r6: same incumbent code under a second name, so PT(panel)/PT(base) is a free
    # ~1.0 control. Only OUR arm differs -- `_panel` runs the pre-88d36e2f dpanel + col2im dinput.
    "conv2d_f32_masked_panel": (c2x32, lambda x: Fn.conv2d(x,c2w32,None,(1,1),(1,1))*c2m32),
    "linear_wide":   (linx, lambda x: Fn.linear(x, linw_wide)),
    "linear_narrow": (linx, lambda x: Fn.linear(x, linw_narrow)),
    # frankentorch-58zjz: the query is the timed leaf; K and V require grad too, so the backward
    # reaches BOTH dV (the dgemm_tb entry this bead is about) and the softmax/dQ path.
    "attention": (attnq, lambda x: Fn.scaled_dot_product_attention(x, attnk, attnv)),
    "conv3d":     (c3x, lambda x: Fn.conv3d(x,c3w,None,(1,1,1),(1,1,1))),
    # frankentorch-l2zki: NON-UNIFORM loss, the only lane here that reaches conv3d's GENERIC
    # backward. The `*c3m` sits inside the lane's own fn, so `run`'s `fn(x).sum()` becomes
    # `(conv3d(x)*c3m).sum()` and the output gradient is `c3m` instead of all-ones -- no
    # change to the shared serve loop, which unpacks a 2-tuple for every lane. The multiply
    # is charged to both arms alike and the mask is built in setup, outside the timer.
    "conv3d_masked": (c3x, lambda x: Fn.conv3d(x,c3w,None,(1,1,1),(1,1,1))*c3m),
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
    # frankentorch-yu1zm: same torch code under a third name; PT(zeroed)/PT(base)
    # must come out ~1.0 or the host moved between the two lanes.
    "max_pool1d_zeroed": (mp1, lambda x: Fn.max_pool1d(x,2,2)),
    # frankentorch-372h8: avg_pool1d on the same tensor, exact 2/2 tiling.
    "avg_pool1d": (mp1, lambda x: Fn.avg_pool1d(x,2,2)),
    "avg_pool1d_zeroed": (mp1, lambda x: Fn.avg_pool1d(x,2,2)),
    # frankentorch-yc7ud: the incumbent twins for the dense-route lanes. The FT
    # side squares before summing; torch is byte-identical code under both names,
    # which is what makes PT(off)/PT(on) a free control.
    # SQUARED on this side too: the loop does fn(x).sum().backward(), so returning the
    # square makes the incumbent's loss sum(out*out) and matches what the FT arm times.
    # Without it the two arms would measure different work and parity would mismatch.
    "avg_pool1d_dense": (mp1, lambda x: Fn.avg_pool1d(x,2,2)**2),
    "avg_pool1d_dense_zeroed": (mp1, lambda x: Fn.avg_pool1d(x,2,2)**2),
    # frankentorch-lu3ht: incumbent twin for the avg_pool2d uninit A/B.
    "avg_pool2d_zeroed": (ap2, lambda x: Fn.avg_pool2d(x,(2,2),(2,2))),
    # frankentorch-mdsmm: incumbent twins for the three DENSE-route lanes. SQUARED here too,
    # for the same reason avg_pool1d_dense is: the loop does fn(x).sum().backward(), so
    # returning the square makes this arm's loss sum(out*out) and matches the work the FT arm
    # times. Without it the arms measure different things and parity mismatches. torch is
    # byte-identical code under the plain and dense names, so PT(dense)/PT(plain) is a free
    # control that prices the squaring itself.
    "avg_pool2d_dense": (ap2, lambda x: Fn.avg_pool2d(x,(2,2),(2,2))**2),
    # frankentorch-mdsmm: the buffer-pool and zeroed-output controls must be
    # priced on this same dense route, rather than only behind the sum shortcut.
    "avg_pool2d_nopool_dense": (ap2, lambda x: Fn.avg_pool2d(x,(2,2),(2,2))**2),
    "avg_pool2d_dense_zeroed": (ap2, lambda x: Fn.avg_pool2d(x,(2,2),(2,2))**2),
    "max_pool3d_dense": (mp3, lambda x: Fn.max_pool3d(x,(2,2,2),(2,2,2))**2),
    # frankentorch-mdsmm: this is the buffer-pool-off control on the SAME non-uniform
    # loss route as max_pool3d_dense. Keep the square in the Python callable because
    # the shared loop only adds the final sum().backward().
    "max_pool3d_nopool_dense": (mp3, lambda x: Fn.max_pool3d(x,(2,2,2),(2,2,2))**2),
    "max_pool1d_dense": (mp1, lambda x: Fn.max_pool1d(x,2,2)**2),
    "max_pool1d_nopool_dense": (mp1, lambda x: Fn.max_pool1d(x,2,2)**2),
    "group_norm_f32": (gnx, lambda x: Fn.group_norm(x,32,gnw,gnb)),
    # frankentorch-68pwz item 103c: the DENSE-route twin. The board's other group_norm
    # lanes all end in a plain sum, which fires the sum-shortcut whose backward never
    # materializes a per-element dy -- a sentinel measured the narrow lever executing
    # ZERO times on those lanes. This name is the same op under a non-sum loss, which is
    # the route the f32 engine's dy/dx conversions actually run on.
    # SQUARED on this side too, the avg_pool1d_dense convention: the loop does
    # fn(x).sum().backward(), so returning the square makes the incumbent's loss
    # sum(out*out) and matches what the FT arm times. Without it the two arms would
    # measure different work and parity would mismatch.
    "group_norm_f32_dense": (gnx, lambda x: Fn.group_norm(x,32,gnw,gnb)**2),
    # frankentorch-68pwz: BatchNorm2d, the half of the bead that never had a lane. Same
    # fixtures as the group_norm rows so the two are directly comparable. running_mean and
    # running_var are None on BOTH arms — torch UPDATES them in place under training=True,
    # so passing them would mutate the fixture between samples and neither arm would time
    # the same work twice. Both loss shapes are carried, to find out whether BatchNorm
    # splits into shortcut and dense routes the way group_norm does (item 109).
    # The f32 SUM-loss lane was REMOVED after one invocation, not merely left failing: under a
    # bare sum loss batch-norm's dx is ANALYTICALLY ZERO (f64 arbiter: ours 4.49e-9, torch
    # 3.84e-9), so its gradient checksum compared two computations of nothing and could never
    # mean anything. Deleting a lane that cannot carry information beats leaving it red on a
    # board a dozen agents run.
    "batch_norm2d_f32_dense": (gnx, lambda x: Fn.batch_norm(x,None,None,gnw,gnb,True,0.1,1e-5)**2),
    # The f64 twin, which is the one that can CERTIFY — see timed_batch_norm2d_f64_dense for
    # why an f32 BatchNorm lane cannot clear a 1e-6 parity gate at any shape.
    "batch_norm2d_f64_dense": (bnx, lambda x: Fn.batch_norm(x,None,None,bnw,bnb,True,0.1,1e-5)**2),
    # frankentorch-jlcmi: incumbent twin for the group_norm uninit A/B.
    "group_norm_f32_zeroed": (gnx, lambda x: Fn.group_norm(x,32,gnw,gnb)),
    "group_norm_f32_dense_zeroed": (gnx, lambda x: Fn.group_norm(x,32,gnw,gnb)**2),
    # The FrankenTorch side of this second name calls the two f32 kernels
    # DIRECTLY, with no session and no tape, to price the engine and the f64
    # grad-space conversions separately from the kernel. The incumbent is the
    # same op under both names, so PT(kernels)/PT(f32) is a free ~1.0 control.
    "group_norm_f32_kernels": (gnx, lambda x: Fn.group_norm(x,32,gnw,gnb)),
    # frankentorch-mdsmm: direct-kernel dense twins. The four scalar kernel lanes below call
    # group_norm_backward_scalar_f32 directly, so their historical kernels-vs-engine split never
    # built a dense dy. Squaring makes run's shared sum produce sum(out*out) on BOTH arms.
    "group_norm_f32_kernels_dense": (gnx, lambda x: Fn.group_norm(x,32,gnw,gnb)**2),
    # frankentorch-dmpho: the same torch op under a third name. The FrankenTorch
    # side runs this one with the group-norm forward forced onto the old serial
    # schedule, so the pair prices the parallel gate against one live incumbent
    # inside one invocation. PT(serialfwd)/PT(kernels) is a free ~1.0 control.
    "group_norm_f32_kernels_serialfwd": (gnx, lambda x: Fn.group_norm(x,32,gnw,gnb)),
    "group_norm_f32_kernels_serialfwd_dense": (gnx, lambda x: Fn.group_norm(x,32,gnw,gnb)**2),
    # frankentorch-qkwsy: the same torch op again for the forward-statistics-reuse
    # pair. PT(statskernels_recompute)/PT(statskernels) is a free ~1.0 control.
    "group_norm_f32_statskernels": (gnx, lambda x: Fn.group_norm(x,32,gnw,gnb)),
    "group_norm_f32_statskernels_recompute": (gnx, lambda x: Fn.group_norm(x,32,gnw,gnb)),
    "group_norm_f32_statskernels_dense": (gnx, lambda x: Fn.group_norm(x,32,gnw,gnb)**2),
    "group_norm_f32_statskernels_recompute_dense": (gnx, lambda x: Fn.group_norm(x,32,gnw,gnb)**2),
    "prelu": (prx, lambda x: Fn.prelu(x,prw)),
    # frankentorch-k1hto: the same torch op under a second name, exactly as the
    # `_nopool` lanes do. The FrankenTorch side runs this one with an
    # observation-only hook on the PReLU output, which makes the sum shortcut
    # decline and restores the materialising path. PT(noshortcut)/PT(prelu) is
    # therefore a free control that must land near 1.0.
    "prelu_noshortcut": (prx, lambda x: Fn.prelu(x,prw)),
    # frankentorch-mdsmm: prelu's DENSE-route incumbent twin. SQUARED here so this arm's loss is
    # sum(out*out) and matches what the FT arm times, exactly as avg_pool1d_dense does it.
    "prelu_dense": (prx, lambda x: Fn.prelu(x,prw)**2),
}
"#;
    // ISOLATION MODE PICKS A DIFFERENT CO-PROCESS — `frankentorch-rayon-pool-width-qq8as`.
    //
    // With `FT_H2H_NO_INCUMBENT` the driver never sends a SAMPLE request, so the incumbent's work
    // was already unused; what it was NOT was absent. The full program still imported torch, built
    // every lane, and ran 32 warm-ups PER LANE before `PT_READY`, so an "isolated" run was preceded
    // by minutes of torch work on the same box — warming the same caches and pulling the same
    // clocks the run then measures.
    //
    // It also kept the mode off any machine without torch, which is precisely where it is needed:
    // qq8as asks whether the width finding generalises past THIS host, and the second machines
    // available (rch workers) have no torch. An arm-internal measurement that cannot run without
    // torch cannot visit a second machine.
    //
    // Read here rather than at the flag's other use site further down, because the decision has to
    // be made BEFORE the child is spawned.
    let isolate_arm_early = std::env::var("FT_H2H_NO_INCUMBENT").is_ok();
    let py = if isolate_arm_early {
        ft_api::harness_interleave::ISOLATION_STUB_PY.to_owned()
    } else {
        format!("{py_setup}{}", ft_api::harness_interleave::SAMPLE_LOOP_PY)
    };

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
    if isolate_arm_early {
        // Never print `incumbent=PyTorch <version>` for a run that had no incumbent. The stub
        // reports a stub identifier precisely so this branch can be honest rather than plausible.
        println!(
            "incumbent=NONE (FT_H2H_NO_INCUMBENT: the co-process is a stub that imports no torch \
             and does no work, so this run carries NO vs-PyTorch claim and its PT columns are \
             placeholders). Arm-internal comparisons only: FT time against FT time, within this \
             one invocation on this one machine."
        );
    } else {
        println!(
            "{}",
            ft_api::harness_provenance::incumbent_provenance_block(
                torch_version,
                INCUMBENT_THREADS
            )
        );
    }
    // Names the machine this row was measured on. Both arms are sampled in this
    // one invocation on this one host, so the row is internally comparable; the
    // block exists so it can still be PLACED against other rows afterwards.
    println!(
        "{}",
        ft_api::harness_provenance::measurement_host_block(rayon::current_num_threads())
    );
    // frankentorch-hi9r6 item 195: announce this run BEFORE the /proc scan below, so a peer that
    // starts while we sample sees us. `_slot` must be bound, not discarded: the announcement lives
    // exactly as long as this binding, and `let _ = ..` would drop it here and announce nothing.
    let (_slot, slot_line) = ft_api::harness_provenance::announce_measurement(
        &std::env::var("FT_H2H_LANES").unwrap_or_else(|_| "all".to_owned()),
    );
    println!("{slot_line}");
    // frankentorch-hi9r6 item 193: WHO ELSE was measuring. Printed next to the host block because
    // it answers the question loadavg cannot — a run at loadavg 85 of compilation and a run at
    // loadavg 85 of two other harnesses sampling are not the same measurement, and only the second
    // one inverts ratios.
    println!(
        "{}",
        ft_api::harness_provenance::concurrent_measurement_block()
    );
    // frankentorch-68pwz: the clock domain each arm ran in. Printed unconditionally
    // because a row that does not say this cannot be read: at a 2.8x cross-core spread,
    // a ratio can be a frequency artefact and every gate in this harness is blind to it.
    println!(
        "{}",
        ft_api::harness_provenance::cpu_clock_block(
            std::env::var("FT_H2H_PIN_CORES").ok().as_deref()
        )
    );
    // frankentorch-2h8vi: the host's own movement, which no A/A null can see.
    let load_at_start = ft_api::harness_provenance::load_average_1m();
    // frankentorch-vyaia: how long the load series spans. `loadavg` is a 1-minute EWMA, so the
    // DURATION of the window decides how much of any load step it could have shown at all — a
    // figure the load lines have never carried and cannot be read without.
    let load_window_started = Instant::now();
    let cpu_at_start = self_cpu_seconds();
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
    // Item 167: discarded samples per lane per ROUND, symmetric across both arms. Default 0, so
    // every existing row is unaffected. Declared here rather than beside the round loop so it can
    // be REPORTED with the rest of the provenance — a row taken under it is not comparable to one
    // taken without, so the output has to say which it is.
    let round_warmup: usize = std::env::var("FT_H2H_ROUND_WARMUP")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(0);
    // PER-LANE warm-up — `frankentorch-hi9r6`, item 190, fixing a defect in item 167.
    //
    // Item 167 made the warm-up symmetric across the two ARMS and then made the toggle itself
    // impossible to alternate within a process: it is read once into a `let`, so its own A/B could
    // only ever be two invocations. Item 189 then ran exactly that A/B and had to discard the
    // magnitude, because loadavg was 16.2 in one run and 27.5 in the other — the cross-run pairing
    // this campaign has got wrong more than any other single thing.
    //
    // `set_pool_output_zeroed` and `set_gemm_tile_col_floor_adaptive` were both given in-process
    // switchability deliberately, for the reason item 25 gives: a cross-binary — or here
    // cross-invocation — comparison cannot attribute a few percent to any one change. The warm-up
    // needed the same property and did not have it.
    //
    // Naming the lanes rather than flipping a global is what makes it work inside ONE sweep: a
    // lane registered twice under two names, with only one name listed here, is measured warm and
    // cold in the same window, on the same ELF, interleaved round by round with the same incumbent.
    //
    // Empty (the default) preserves item 167's behaviour exactly: the global applies to every lane,
    // so no existing invocation changes meaning.
    let round_warmup_lanes: Vec<String> = std::env::var("FT_H2H_ROUND_WARMUP_LANES")
        .map(|spec| {
            spec.split(',')
                .map(|s| s.trim().to_owned())
                .filter(|s| !s.is_empty())
                .collect()
        })
        .unwrap_or_default();
    // PER-LANE GEMM TILE FLOOR — `frankentorch-hi9r6`, item 220.
    //
    // Item 217 measured `set_gemm_tile_col_floor_adaptive` arm-internally at **1.223x at 8
    // threads** on conv2d's two backward GEMMs, and a REGRESSION (0.950x) at 64. That is a large
    // effect at the width every certified row uses, and it is MAINTENANCE until a PyTorch arm sees
    // it — a FrankenTorch-versus-FrankenTorch number is not a win, however big.
    //
    // The toggle is an `AtomicBool` precisely so both arms can run in one process (item 25), but
    // nothing could USE that: it is global, so a sweep could only ever be all-on or all-off, and
    // two invocations is the cross-run comparison a peer's item 189 showed is worthless here — the
    // incumbent arm moved 1.94x between two runs of the SAME ELF.
    //
    // Naming lanes fixes it the way item 190 fixed the warm-up: register a lane twice and list one
    // name, and the same work is measured with and without the toggle **in one window, one ELF,
    // interleaved round by round against the same incumbent**. Empty (the default) touches nothing.
    let tile_adaptive_lanes: Vec<String> = std::env::var("FT_H2H_TILE_ADAPTIVE_LANES")
        .map(|spec| {
            spec.split(',')
                .map(|s| s.trim().to_owned())
                .filter(|s| !s.is_empty())
                .collect()
        })
        .unwrap_or_default();
    if !tile_adaptive_lanes.is_empty() {
        println!(
            "tile_adaptive_lanes={} (FT_H2H_TILE_ADAPTIVE_LANES; item 217 measured this at 1.223x \
             at 8 threads and 0.950x at 64, so a row taken under it is width-specific)",
            tile_adaptive_lanes.join(",")
        );
    }
    if round_warmup > 0 {
        println!(
            "round_warmup_lanes={} (FT_H2H_ROUND_WARMUP_LANES; empty = every lane)",
            if round_warmup_lanes.is_empty() {
                "<all>".to_owned()
            } else {
                round_warmup_lanes.join(",")
            }
        );
    }
    println!(
        "sampling=balanced-square {} (frankentorch-xdw0h); {} rounds, four live samples per arm \
         per round, torch threads=8",
        BALANCED_SQUARE
            .iter()
            .map(|incumbent| if *incumbent { 'A' } else { 'B' })
            .collect::<String>(),
        reps,
    );
    println!(
        "round_warmup={round_warmup} (FT_H2H_ROUND_WARMUP; discarded samples per lane per round, \
         BOTH arms). {}",
        if round_warmup == 0 {
            "0 = the board's default: each lane's slot 0 each round arrives cold, which \
             NEGATIVE_EVIDENCE item 147 measured at +20-30% on the summed conv2d lanes"
        } else {
            "NON-ZERO: this row measures STEADY-STATE cost and is NOT comparable to any \
             certified standing on the board, all of which were taken at 0"
        }
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
            // frankentorch-yu1zm: identical work to `max_pool1d`, with the uninit
            // first-touch lever switched OFF, so the pair measures the lever
            // against its own predecessor inside ONE binary and ONE invocation.
            // NEGATIVE_EVIDENCE item 25 could only compare across two binaries
            // built 90 minutes apart with peer commits between them, which cannot
            // attribute a 2-3.5% difference to any single change.
            "max_pool1d_zeroed",
            Box::new(|| {
                let previous = ft_kernel_cpu::set_pool_output_zeroed(true);
                let sample = timed_op(&mp1, vec![MP1_N, MP1_C, MP1_L], |s, x| {
                    s.functional_max_pool1d(x, 2, 2).expect("max_pool1d")
                });
                ft_kernel_cpu::set_pool_output_zeroed(previous);
                sample
            }),
        ),
        (
            // frankentorch-372h8: avg_pool1d on the max_pool1d shape [8,64,8192], which
            // tiles exactly at kernel=stride=2 and so takes the total-coverage backward.
            // timed_op SUMS, so this lane exercises `avg_pool1d_backward_scalar_f64` —
            // the scalar shortcut — and NOT the dense `avg_pool1d_backward_f64`. That
            // distinction is the whole reason this lane exists on this route.
            "avg_pool1d",
            Box::new(|| {
                timed_op(&mp1, vec![MP1_N, MP1_C, MP1_L], |s, x| {
                    s.functional_avg_pool1d(x, 2, 2).expect("avg_pool1d")
                })
            }),
        ),
        (
            // frankentorch-yc7ud: squared loss, so the sum-shortcut does NOT fire and
            // the DENSE backward runs. Same tensor and tiling as the avg_pool1d lane.
            "avg_pool1d_dense",
            Box::new(|| {
                timed_op_sq(&mp1, vec![MP1_N, MP1_C, MP1_L], |s, x| {
                    s.functional_avg_pool1d(x, 2, 2).expect("avg_pool1d")
                })
            }),
        ),
        (
            "avg_pool1d_dense_zeroed",
            Box::new(|| {
                let previous = ft_kernel_cpu::set_pool_output_zeroed(true);
                let sample = timed_op_sq(&mp1, vec![MP1_N, MP1_C, MP1_L], |s, x| {
                    s.functional_avg_pool1d(x, 2, 2).expect("avg_pool1d")
                });
                ft_kernel_cpu::set_pool_output_zeroed(previous);
                sample
            }),
        ),
        (
            "avg_pool1d_zeroed",
            Box::new(|| {
                let previous = ft_kernel_cpu::set_pool_output_zeroed(true);
                let sample = timed_op(&mp1, vec![MP1_N, MP1_C, MP1_L], |s, x| {
                    s.functional_avg_pool1d(x, 2, 2).expect("avg_pool1d")
                });
                ft_kernel_cpu::set_pool_output_zeroed(previous);
                sample
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
        // frankentorch-mdsmm, NEGATIVE_EVIDENCE item 110/111: DENSE-route twins for the
        // three lanes that still had none. Each op below owns a scalar sum-shortcut
        // (`try_avg_pool2d_sum_shortcut`, `try_max_pool3d_sum_shortcut`,
        // `try_max_pool1d_sum_shortcut`), so the plain-sum lane above it scores the
        // shortcut and never runs the dense backward a real objective reaches. `timed_op_sq`
        // already existed for exactly this — yc7ud built it for avg_pool1d and item 103c
        // used the same idea for group_norm — so these are registrations, not new machinery.
        (
            // NOT `functional_avg_pool2d_sum` — that one is documented as the "fused scalar
            // loss for sum(avg_pool2d(input))" and RETURNS THE SCALAR, so squaring it would
            // time `(sum(pool(x)))^2` against torch's `sum(pool(x)^2)`: two different losses,
            // disagreeing gradients, and a lane that never leaves the fused path it is
            // supposed to avoid. That is exactly the mistake this lane shipped with for one
            // commit; see NEGATIVE_EVIDENCE item 112. The plain lane above deliberately keeps
            // the fused entry, because measuring it IS that lane's purpose.
            "avg_pool2d_dense",
            Box::new(|| {
                timed_op_sq(&ap2, vec![AP2_N, AP2_C, AP2_H, AP2_W], |s, x| {
                    s.functional_avg_pool2d(x, (2, 2), (2, 2), (0, 0), false, true)
                        .expect("avg_pool2d")
                })
            }),
        ),
        (
            // The buffer-pool control must be visible behind the non-uniform loss too:
            // `functional_avg_pool2d_sum` would keep the sum shortcut live, whereas this
            // returns a tensor whose `sum(out*out)` loss materializes the dense backward.
            "avg_pool2d_nopool_dense",
            Box::new(|| {
                ft_core::buffer_pool::set_enabled(false);
                let sample = timed_op_sq(&ap2, vec![AP2_N, AP2_C, AP2_H, AP2_W], |s, x| {
                    s.functional_avg_pool2d(x, (2, 2), (2, 2), (0, 0), false, true)
                        .expect("avg_pool2d")
                });
                ft_core::buffer_pool::set_enabled(true);
                sample
            }),
        ),
        (
            // Same dense route as `avg_pool2d_dense`; only its allocation policy differs.
            "avg_pool2d_dense_zeroed",
            Box::new(|| {
                let previous = ft_kernel_cpu::set_pool_output_zeroed(true);
                let sample = timed_op_sq(&ap2, vec![AP2_N, AP2_C, AP2_H, AP2_W], |s, x| {
                    s.functional_avg_pool2d(x, (2, 2), (2, 2), (0, 0), false, true)
                        .expect("avg_pool2d")
                });
                ft_kernel_cpu::set_pool_output_zeroed(previous);
                sample
            }),
        ),
        (
            "max_pool3d_dense",
            Box::new(|| {
                timed_op_sq(&mp3, vec![MP3_N, MP3_C, MP3_D, MP3_H, MP3_W], |s, x| {
                    s.functional_max_pool3d(x, (2, 2, 2), (2, 2, 2))
                        .expect("max_pool3d")
                })
            }),
        ),
        (
            // `max_pool3d_nopool` above only prices the buffer pool under the scalar
            // sum-shortcut. This twin uses `sum(out*out)`, so `try_max_pool3d_sum_shortcut`
            // declines and the buffer-pool control is visible on the backward route training
            // actually reaches. The Python registration squares too, keeping both arms on the
            // same loss and making PT(nopool_dense)/PT(dense) the incumbent control.
            "max_pool3d_nopool_dense",
            Box::new(|| {
                ft_core::buffer_pool::set_enabled(false);
                let sample = timed_op_sq(&mp3, vec![MP3_N, MP3_C, MP3_D, MP3_H, MP3_W], |s, x| {
                    s.functional_max_pool3d(x, (2, 2, 2), (2, 2, 2))
                        .expect("max_pool3d")
                });
                ft_core::buffer_pool::set_enabled(true);
                sample
            }),
        ),
        (
            "max_pool1d_dense",
            Box::new(|| {
                timed_op_sq(&mp1, vec![MP1_N, MP1_C, MP1_L], |s, x| {
                    s.functional_max_pool1d(x, 2, 2).expect("max_pool1d")
                })
            }),
        ),
        (
            // As for max_pool3d, the buffer-pool control needs the real dense route rather
            // than the all-ones scalar shortcut timed by `max_pool1d_nopool`.
            "max_pool1d_nopool_dense",
            Box::new(|| {
                ft_core::buffer_pool::set_enabled(false);
                let sample = timed_op_sq(&mp1, vec![MP1_N, MP1_C, MP1_L], |s, x| {
                    s.functional_max_pool1d(x, 2, 2).expect("max_pool1d")
                });
                ft_core::buffer_pool::set_enabled(true);
                sample
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
            // frankentorch-jlcmi: the uninit twin for the group_norm forward, chosen
            // by the item 28d predictor BEFORE measuring — 25.7 MB of zeroing for a
            // few SIMD ops per element is the max_pool1d profile (which pays), not
            // the avg_pool2d one (which does not). Prediction registered in the
            // bead; this lane is what tests it.
            "group_norm_f32_zeroed",
            Box::new(|| {
                let previous = ft_kernel_cpu::set_pool_output_zeroed(true);
                let sample = timed_group_norm_f32(&gnx, &gnw, &gnb);
                ft_kernel_cpu::set_pool_output_zeroed(previous);
                sample
            }),
        ),
        (
            // Pair the uninitialized-output control with the route that actually materializes
            // the f32 per-element gradient, not the scalar sum-shortcut route above.
            "group_norm_f32_dense_zeroed",
            Box::new(|| {
                let previous = ft_kernel_cpu::set_pool_output_zeroed(true);
                let sample = timed_group_norm_f32_dense(&gnx, &gnw, &gnb);
                ft_kernel_cpu::set_pool_output_zeroed(previous);
                sample
            }),
        ),
        (
            // frankentorch-68pwz item 103c: the dense route, where the f32 engine's
            // per-element dy/dx conversions actually execute. Not a twin of
            // `group_norm_f32` — it is a DIFFERENT backward, so it carries its own
            // incumbent rather than being read as a lever-off pair.
            "group_norm_f32_dense",
            Box::new(|| timed_group_norm_f32_dense(&gnx, &gnw, &gnb)),
        ),
        (
            // frankentorch-68pwz: BatchNorm2d's first live incumbent arm. Both loss shapes,
            // so the sum-shortcut/dense split item 109 found on group_norm is measured here
            // rather than assumed to repeat.
            // f32: NOT quotable and expected to report MISMATCH — kept so the f32 engine's
            // timing stays visible, with the honest label attached. The sum-loss sibling was
            // removed: its dx is analytically zero, so its checksum compared two
            // computations of nothing.
            "batch_norm2d_f32_dense",
            Box::new(|| timed_batch_norm2d_f32(&gnx, &gnw, &gnb, true)),
        ),
        (
            // f64: the lane that can certify, on the board's dominant convention.
            "batch_norm2d_f64_dense",
            Box::new(|| timed_batch_norm2d_f64_dense(&bnx, &bnw, &bnb)),
        ),
        (
            "group_norm_f32_kernels",
            Box::new(|| timed_group_norm_f32_kernels(&gnx, &gnw, &gnb, true)),
        ),
        (
            // mdsmm: the route-matched kernels-vs-engine split. Unlike the scalar sibling,
            // this builds dy = 2*out and enters group_norm_backward_f32's generic path.
            "group_norm_f32_kernels_dense",
            Box::new(|| timed_group_norm_f32_kernels_dense_inner(&gnx, &gnw, &gnb, true, false)),
        ),
        (
            // frankentorch-dmpho: the lever-off twin. Same kernels, same shape,
            // same binary; only the forward's schedule differs.
            "group_norm_f32_kernels_serialfwd",
            Box::new(|| timed_group_norm_f32_kernels(&gnx, &gnw, &gnb, false)),
        ),
        (
            "group_norm_f32_kernels_serialfwd_dense",
            Box::new(|| timed_group_norm_f32_kernels_dense_inner(&gnx, &gnw, &gnb, false, false)),
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
        (
            // The scalar-only stats-reuse backward has no dense analogue. These two rows retain
            // the corresponding forward forms but use generic backward, making that boundary
            // explicit rather than presenting the scalar A/B as a dense-route result.
            "group_norm_f32_statskernels_dense",
            Box::new(|| timed_group_norm_f32_kernels_dense_inner(&gnx, &gnw, &gnb, true, true)),
        ),
        (
            "group_norm_f32_statskernels_recompute_dense",
            Box::new(|| timed_group_norm_f32_kernels_dense_inner(&gnx, &gnw, &gnb, true, false)),
        ),
        ("prelu", Box::new(|| timed_prelu(&prx, &prw, false, false))),
        (
            // frankentorch-mdsmm: prelu's DENSE route. `prelu_noshortcut` defeats the fusion
            // with a hook but keeps the upstream gradient uniform; this makes it `2*out`, so
            // the shortcut's predicate must decline on its own. One of the three shortcuts
            // item 111 left with no dense measurement.
            "prelu_dense",
            Box::new(|| timed_prelu(&prx, &prw, false, true)),
        ),
        (
            // frankentorch-k1hto: the SAME lane with the PReLU+sum shortcut
            // declined. One binary, two arms against one live incumbent inside one
            // invocation, so the ratio-of-ratios against `prelu` is immune to the
            // host drift that makes cross-run ratios unquotable here.
            "prelu_noshortcut",
            Box::new(|| timed_prelu(&prx, &prw, true, false)),
        ),
        (
            // frankentorch-lu3ht: the avg_pool2d forward is the ONE member of the
            // sibling uninit set that sits on a live lane, so it is the only one
            // that can be A/B'd against a real incumbent rather than assumed to
            // inherit max_pool1d's 1.36x. Item 26c is explicit that it must not be
            // assumed: the lever's worth depends on whether the allocation is
            // served by fresh mmap zero pages or recycled dirty arena memory.
            "avg_pool2d_zeroed",
            Box::new(|| {
                let previous = ft_kernel_cpu::set_pool_output_zeroed(true);
                let sample = timed_op(&ap2, vec![AP2_N, AP2_C, AP2_H, AP2_W], |s, x| {
                    s.functional_avg_pool2d_sum(x, (2, 2), (2, 2), (0, 0), false, true)
                        .expect("avg_pool2d")
                });
                ft_kernel_cpu::set_pool_output_zeroed(previous);
                sample
            }),
        ),
        (
            // frankentorch-58zjz item 126d: conv2d's SUMMED route, which takes the all-ones
            // fast path its 2026-07-05 adjoint provides.
            "conv2d",
            Box::new(|| timed_conv2d(&c2x, &c2w, None, C2_N, false)),
        ),
        (
            // The same op under a NON-UNIFORM loss, reaching the generic backward -- the route
            // real training takes, and the one item 124 found algorithmically behind for conv3d.
            "conv2d_masked",
            Box::new(|| timed_conv2d(&c2x, &c2w, Some(&c2m), C2_N, false)),
        ),
        (
            // item 144: both routes again at double the batch. Item 137 certified NEITHER
            // conv2d lane because PyTorch's A/A null failed 5/5 while ours passed 5/5, and
            // item 137c blamed the incumbent arm's ~3 ms duration. These twins test that.
            "conv2d_big",
            Box::new(|| timed_conv2d(&c2bx, &c2w, None, C2B_N, false)),
        ),
        (
            "conv2d_big_masked",
            Box::new(|| timed_conv2d(&c2bx, &c2w, Some(&c2bm), C2B_N, false)),
        ),
        (
            // item 220: byte-identical twin of conv2d_big_masked, existing only so
            // FT_H2H_TILE_ADAPTIVE_LANES can name ONE of the pair. Item 219 measured
            // conv2d_big_masked certifying 4 of 4 at this size, so the pair inherits a lane whose
            // nullability is established rather than hoped for -- item 209's rule.
            //
            // The closure is identical on purpose: if the two rows differ by anything but the tile
            // floor, the difference is the host, and both rows are in the same window to say so.
            "conv2d_big_masked_tile",
            Box::new(|| timed_conv2d(&c2bx, &c2w, Some(&c2bm), C2B_N, false)),
        ),
        (
            // item 216: the TRAIN twin of the lane above, and the whole point of the pair.
            //
            // Item 182 added `conv2d_masked_train` to ask whether item 178's `needs_input_grad`
            // skip is a saving a real training step can take. That lane is at `C2_N` (batch 8),
            // whose incumbent arm runs ~3.8 ms -- the SHORTEST on this board, and item 203's
            // sizing table measured the incumbent as OFFSET at 5.08 ms and PASS only at 11.0 and
            // 11.6 ms. Three invocations confirmed it: the pair reads consistently but its A/A
            // null failed every time, so the question has never been answerable from it.
            //
            // `conv2d_big_masked` is the SAME masked route at batch 16, and item 209's table
            // recorded it certifying 4 of 4 -- the only masked conv2d lane on this board that
            // ever has. So this twin is sized by a MEASURED certification, not by a guess about
            // how long is long enough, and it needs no new fixture: `c2bx`/`c2bm` already exist
            // on both arms and the incumbent already has `c2w_train`.
            //
            // The two lanes differ in EXACTLY the weight's `requires_grad`, on both arms, so the
            // pair separates the two readings item 182 could not: if the ratio is unchanged
            // between them, item 178's skip is worth nothing to a training step and the board's
            // frozen-weight conv2d lanes flatter us; if the train ratio is worse, our dweight
            // GEMM is a disproportionate offender and items 170/172 should aim there.
            "conv2d_big_masked_train",
            Box::new(|| timed_conv2d(&c2bx, &c2w, Some(&c2bm), C2B_N, true)),
        ),
        (
            // `frankentorch-hi9r6`: the two lanes above with the dinput BLOCKING toggled OFF, so
            // each pair differs in exactly one thing and both halves sample the same host minute.
            //
            // The lever: the generic backward's `dpadded` was `dgemm(flat, out_ch, patch_width)`
            // into a `flat x patch_width` panel — 37.7 MB here — that `conv2d_col2im_f64` read
            // once and dropped. It is now blocked on `m` into an L2-resident buffer and scattered
            // per block, inside ONE parallel region over the batch image. NEGATIVE_EVIDENCE item
            // 117 reverted the conv3d version of this at 1.7x SLOWER because it fragmented the
            // fork/join; 117c's retry predicate — "a design that keeps ONE wide parallel region" —
            // is what the image-parallel form and its `batch >= current_num_threads()` gate are
            // for, and this pair is what says whether it worked.
            //
            // Item 25's rule is why this is a toggle and not a second binary. The two paths are
            // BIT-IDENTICAL (`conv2d_dinput_panel_legacy_toggle_selects_a_bit_identical_path`),
            // so the pair can move time and cannot move a number — and PyTorch runs the SAME code
            // under both names, making PT(panel)/PT(base) a free control that must come out ~1.0.
            "conv2d_big_masked_panel",
            Box::new(|| {
                let previous = ft_kernel_cpu::set_conv2d_dinput_panel_legacy(true);
                let sample = timed_conv2d(&c2bx, &c2w, Some(&c2bm), C2B_N, false);
                ft_kernel_cpu::set_conv2d_dinput_panel_legacy(previous);
                sample
            }),
        ),
        (
            "conv2d_big_masked_train_panel",
            Box::new(|| {
                let previous = ft_kernel_cpu::set_conv2d_dinput_panel_legacy(true);
                let sample = timed_conv2d(&c2bx, &c2w, Some(&c2bm), C2B_N, true);
                ft_kernel_cpu::set_conv2d_dinput_panel_legacy(previous);
                sample
            }),
        ),
        (
            // item 209: the SUMMED route, sized so our arm lands near 35 ms. `conv2d_big` reaches
            // the same code at 8.8 ms and certified 1 of 4; this is the same lane with the one
            // property that separates certifying lanes from non-certifying ones on this board.
            "conv2d_xl",
            Box::new(|| timed_conv2d(&c2xlx, &c2w, None, C2XL_N, false)),
        ),
        (
            // item 212: the SAME lane with items 174/177's scatter collapse toggled OFF, so the
            // pair differs in exactly one thing and both halves sample the same host minute.
            // Item 25's rule is why this is a toggle and not a second binary: a cross-invocation
            // comparison cannot attribute a few percent to any one change.
            //
            // The two paths are BIT-IDENTICAL (`conv2d_ones_scatter_toggle_selects_a_bit_identical
            // _path`), so this pair can move time and cannot move a number — and PyTorch runs the
            // SAME code under both names, making PT(legacy)/PT(xl) a free control that must come
            // out ~1.0. If it does not, the host moved and neither half is readable.
            "conv2d_xl_legacy",
            Box::new(|| {
                let previous = ft_kernel_cpu::set_conv2d_ones_scatter_legacy(true);
                let sample = timed_conv2d(&c2xlx, &c2w, None, C2XL_N, false);
                ft_kernel_cpu::set_conv2d_ones_scatter_legacy(previous);
                sample
            }),
        ),
        (
            // item 190: the SAME lane as conv2d_masked, registered under a second name so one
            // sweep can measure it warmed and cold in the same window. The closure is identical
            // on purpose -- if these two rows differ by anything but the warm-up, the difference
            // is the host, and that is exactly what item 189 could not rule out across two runs.
            "conv2d_masked_warm",
            Box::new(|| timed_conv2d(&c2x, &c2w, Some(&c2m), C2_N, false)),
        ),
        (
            // item 182: the SAME masked route with a GRAD-REQUIRING weight, so both arms compute
            // dweight. Every other conv2d lane freezes the weight, which is not what a training
            // step does and is the half `timed_linear` deliberately refuses to skip.
            //
            // ADDED, not swapped in — item 144e's rule. The frozen-weight lanes carry the
            // certified 5.73x and every row quoted from it; moving them would invalidate that
            // history to answer a question a new lane answers for free.
            //
            // This is also the control for item 178: honouring `needs_input_grad` is worth ~18%
            // where the weight is frozen and NOTHING here, so the pair separates "we stopped
            // computing a discarded gradient" from "conv2d got faster".
            "conv2d_masked_train",
            Box::new(|| {
                // DRAIN FIRST, and this is not belt-and-braces — it is the fix for a real
                // mis-attribution. `conv2d_masked_train_kernels` ALSO runs the streamed dweight
                // (its panel is 18 MB, above the gate) and did not drain, so its increments were
                // attributed to whichever arm drained next. That is how the forced-LEGACY arm
                // came to report 160 executions when it must report 0. Draining on entry makes
                // each arm's count depend only on its own work, whatever any other lane does.
                let _ = ft_kernel_cpu::take_conv2d_dweight_streamed_calls();
                let sample = timed_conv2d(&c2x, &c2w, Some(&c2m), C2_N, true);
                // SENTINEL on the INCUMBENT side. Counting only the legacy arm would prove the
                // toggle turned something OFF and say nothing about whether the shipped path
                // turns it ON. Both counts are printed on the `_dwpanel` row. Read outside the
                // timed region, so it cannot move the sample.
                CONV2D_STREAMED_CALLS.with(|cell| {
                    cell.set(cell.get() + ft_kernel_cpu::take_conv2d_dweight_streamed_calls());
                });
                sample
            }),
        ),
        (
            // `frankentorch-hi9r6`: the kernels-only twin of the lane above, so the training
            // step's FORWARD, BACKWARD and SESSION frames separate inside one invocation
            // against one live incumbent. See `timed_conv2d_masked_train_kernels` — and read
            // its `pad_ms` diagnostic before differencing, because the two arms do not pay the
            // same pad.
            "conv2d_masked_train_kernels",
            Box::new(|| timed_conv2d_masked_train_kernels(&c2x, &c2w, &c2m, C2_N)),
        ),
        (
            // `frankentorch-hi9r6`: `conv2d_masked_train` with the streamed dweight FORCED OFF,
            // so the pair differs in exactly one thing and both halves sample the same host
            // minute. The arm is the LEGACY one because streaming is now the shipped default —
            // `feedback_unset_knob_means_forced_off` is the rule this follows: an incumbent arm
            // must be what production runs, so the toggled arm is the one that departs from it.
            // While the lever was default-OFF this lane forced it ON and was named `_streamed`.
            //
            // THE LEVER. The generic backward's `dweight` reads an 18.9 MB im2col panel that is
            // built solely to feed one `dgemm_tb` — 37.7 MB of DRAM traffic for 75.5 MMAC, over
            // a `padded` input that is only 2.37 MB and cache-resident. The streamed form feeds
            // that same GEMM `DGEMM_KC`-aligned k-tiles from a per-task scratch and never
            // materialises the panel. See `ft_kernel_cpu::conv2d_dweight_streamed`.
            //
            // Item 25's rule is why this is a toggle and not a second binary: the two paths are
            // BIT-IDENTICAL (`conv2d_dweight_streamed_matches_the_panel_gemm_bitwise`), so the
            // pair can move time and cannot move a number, and PyTorch runs the SAME code under
            // both names as a free ~1.0 control on the window.
            "conv2d_masked_train_dwpanel",
            Box::new(|| {
                let previous = ft_kernel_cpu::set_conv2d_dweight_streamed(false);
                // Drain first — see the incumbent arm above for why.
                let _ = ft_kernel_cpu::take_conv2d_dweight_streamed_calls();
                let sample = timed_conv2d(&c2x, &c2w, Some(&c2m), C2_N, true);
                // SENTINEL. "no effect" and "never executed" are indistinguishable in a paired
                // lane, and this lever's FIRST paired row was a 1.017x taken on a branch that
                // never ran: the fused masked backward keeps its own copy of the panel GEMM and
                // the toggle had only been wired into the generic entry. The count is read on
                // this thread, which is the thread that calls the gate.
                CONV2D_LEGACY_CALLS.with(|cell| {
                    cell.set(cell.get() + ft_kernel_cpu::take_conv2d_dweight_streamed_calls());
                });
                ft_kernel_cpu::set_conv2d_dweight_streamed(previous);
                sample
            }),
        ),
        (
            // `frankentorch-hi9r6`: the lane above with the blocked dinput toggled OFF, so the
            // pair differs in exactly one thing and both halves sample the same host minute.
            //
            // WHY IT EXISTS. This lane is batch 8. Until 164e159d the blocked dinput was gated on
            // `batch >= current_num_threads()` and switched OFF here, leaving it on the old
            // `dpanel` + `col2im` round trip at 2.24x SLOWER while its batch-16 twin ran 1.10x
            // FASTER. The channel-group route opened that gate. Measuring what that is worth
            // ACROSS invocations does not work on this host: the last attempt moved our arm
            // 10.666 -> 9.259 ms but every PyTorch null failed, and a cross-run delta is not
            // separable from the incumbent's own movement. Item 25's rule is the fix.
            //
            // The two paths are BIT-IDENTICAL — `conv2d_dinput_grouped_matches_panel_col2im
            // _bitwise` asserts that at seven shapes by every divisor of `in_ch` by four block
            // widths — so this pair can move time and cannot move a number, and PyTorch runs the
            // SAME code under both names as a free ~1.0 control.
            "conv2d_masked_train_panel",
            Box::new(|| {
                let previous = ft_kernel_cpu::set_conv2d_dinput_panel_legacy(true);
                let sample = timed_conv2d(&c2x, &c2w, Some(&c2m), C2_N, true);
                ft_kernel_cpu::set_conv2d_dinput_panel_legacy(previous);
                sample
            }),
        ),
        (
            // item 191: the f32 SUMMED route. This is the only lane that reaches the all-ones
            // adjoints items 179/181 added for f32, and the `ft-api` narrow-skip of item 185 --
            // none of which any board row has ever priced, because f32 conv2d had no lane.
            "conv2d_f32",
            Box::new(|| timed_conv2d_f32(&c2x32, &c2w32, None, C2F32_N, false)),
        ),
        (
            "conv2d_f32_kernels",
            Box::new(|| timed_conv2d_f32_kernels(&c2x32, &c2w32, C2F32_N)),
        ),
        (
            // The f32 GENERIC route, and the control for the row above: it shares every code path
            // except the all-ones adjoints, so the pair separates "the f32 adjoints help" from
            // "f32 conv2d moved". It is also where item 187's `needs_input_grad` skip acts, the
            // weight being frozen here exactly as it is on the f64 masked lane.
            "conv2d_f32_masked",
            Box::new(|| timed_conv2d_f32(&c2x32, &c2w32, Some(&c2m32), C2F32_N, false)),
        ),
        (
            // `frankentorch-hi9r6`: the lane above with 88d36e2f's blocked image-parallel dinput
            // toggled OFF, so the pair differs in exactly one thing and both halves sample the
            // same host minute.
            //
            // WHY THIS LANE HAD TO EXIST. 88d36e2f took our arm on `conv2d_f32_masked` from
            // 114.667 ms to a stable 69.7-73.0 ms by removing a 189 MB `dpanel` round trip. That
            // is a SELF-SPEEDUP measured across invocations, which is maintenance, not a win --
            // and it could not be converted into one, because PyTorch's arm on this lane swung
            // 18.2 to 27.0 ms (48%) across four runs, so the vs-incumbent delta was not separable
            // from the incumbent's own movement. Item 25's rule is the fix: both arms in ONE
            // invocation against ONE live incumbent, with PT(panel)/PT(base) as a free ~1.0
            // control that says whether the window held still.
            //
            // The two paths are BIT-IDENTICAL — `conv2d_dinput_direct_f32_matches_panel_col2im
            // _bitwise` asserts that at eight shapes by five block widths — so this pair can move
            // time and cannot move a number.
            "conv2d_f32_masked_panel",
            Box::new(|| {
                let previous = ft_kernel_cpu::set_conv2d_dinput_panel_legacy(true);
                let sample = timed_conv2d_f32(&c2x32, &c2w32, Some(&c2m32), C2F32_N, false);
                ft_kernel_cpu::set_conv2d_dinput_panel_legacy(previous);
                sample
            }),
        ),
        (
            // frankentorch-58zjz: in_features > 4*out_features, so dgemm_tb's column gate
            // ENGAGES here (m=128, n=1024: n > 4m, and m*k*n = 67M > 16.8M).
            "linear_wide",
            Box::new(|| timed_linear(&linx, &linw_wide, LIN_B, LIN_IN, LIN_OUT_WIDE)),
        ),
        (
            // The same op on the OTHER side of that gate (m=512, n=1024: n > 4m is false), so
            // item 119's path does NOT engage. Carried so the pair says where the change acts
            // rather than implying it acts everywhere.
            "linear_narrow",
            Box::new(|| timed_linear(&linx, &linw_narrow, LIN_B, LIN_IN, LIN_OUT_NARROW)),
        ),
        (
            "attention",
            Box::new(|| timed_attention(&attnq, &attnk, &attnv)),
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
        (
            // frankentorch-l2zki, NEGATIVE_EVIDENCE item 108d. The board's conv3d lane ends
            // in `sum()`, so its output gradient is all-ones and it takes conv3d's fast
            // path. This twin ends in `(out * mask).sum()`, which is the only way from here
            // to the GENERIC backward — the route a real objective reaches, and the one item
            // 104's 28.3 MB panel removal lives on. Same shape, same weight, same incumbent
            // op; the ONLY difference between the two lanes is the loss, so PT(conv3d_masked)
            // / PT(conv3d) also prices what the mask multiply itself costs.
            "conv3d_masked",
            Box::new(|| {
                timed_conv3d_masked(
                    &c3x,
                    vec![C3_N, C3_CI, C3_D, C3_H, C3_W],
                    &c3w,
                    vec![C3_CO, C3_CI, C3_K, C3_K, C3_K],
                    &c3m,
                    vec![C3_N, C3_CO, C3_D, C3_H, C3_W],
                )
            }),
        ),
    ];

    // frankentorch-68pwz, NEGATIVE_EVIDENCE item 58: restrict the sweep to named
    // lanes, so ROUND COUNT and WALL CLOCK stop being the same dial.
    //
    // The two gates had been deadlocked. A null needs a calm CI, which needs
    // rounds: sweep_d's group_norm min nulls were 1.014 and 1.012 — INSIDE the
    // +/-0.02 band — and still failed because the CI was [0.158,0.511], i.e.
    // undecidable. The drift gate needs a short window: every 16-round sweep has
    // drifted past 1.25x. Those pulled in opposite directions only because a sweep
    // runs all 21 lanes, so more rounds always meant a longer run.
    //
    // Filtering to one lane makes a 16- or 24-round sweep SHORTER in wall clock
    // than an 8-round sweep over 21 lanes, which buys tight CIs and low drift
    // exposure at the same time. It changes no arm, no estimator and no gate — the
    // remaining lanes are simply not measured.
    //
    // `FT_H2H_LANES` is a comma-separated list of substrings; a lane runs if any
    // matches. Unset runs everything, so an unadorned invocation stays
    // byte-comparable with every banked row.
    let lanes = match std::env::var("FT_H2H_LANES") {
        Err(_) => lanes,
        Ok(spec) => {
            let wanted: Vec<String> = spec
                .split(',')
                .map(|s| s.trim().to_owned())
                .filter(|s| !s.is_empty())
                .collect();
            // EXACT MATCHING, OPT-IN — `frankentorch-hi9r6`, item 226.
            //
            // Substring matching is why item 225's sweep ran SIX lanes when it wanted two:
            // `conv2d_masked` also selects `conv2d_masked_warm` and `conv2d_masked_train`, and
            // `conv2d_big_masked` selects its `_tile` and `_train` twins. Every twin added since
            // item 182 widens the blast radius of an existing filter string.
            //
            // That is not a tidiness problem. Item 225's run was voided by LOAD DRIFT after the
            // quietest window of the session collapsed mid-sweep, and a sweep three times longer
            // than intended is three times as much window to have to stay quiet. **Sweep length is
            // a measurement risk, not just a wait.**
            //
            // Default is unchanged, deliberately: substring matching is what every banked row and
            // every peer's invocation was taken under, and silently narrowing it would change what
            // an existing command means. `FT_H2H_LANES_EXACT=1` opts in.
            let exact = std::env::var("FT_H2H_LANES_EXACT").is_ok_and(|v| v == "1");
            let kept: Vec<_> = lanes
                .into_iter()
                .filter(|(name, _)| {
                    wanted.iter().any(|w| {
                        if exact {
                            *name == w.as_str()
                        } else {
                            name.contains(w.as_str())
                        }
                    })
                })
                .collect();
            assert!(
                !kept.is_empty(),
                "FT_H2H_LANES={spec:?} matched no lane; refusing to run an empty sweep"
            );
            println!(
                "lane_filter={spec:?} match={} kept={} of the full set (frankentorch-68pwz \
                 item 58; item 226: substring is the default, FT_H2H_LANES_EXACT=1 narrows it)",
                if exact { "exact" } else { "substring" },
                kept.len()
            );
            kept
        }
    };

    // LANE COUNT IN THE PROVENANCE — `frankentorch-vyaia`, acceptance item 3.
    //
    // The bead's correction to its own author: frankentorch-5q3io bounded this harness's self-load
    // at ~+0.62 absolute and used that to argue self-load was under 10% of the signal in every
    // failing run. The bound rotted silently, because the sweep GAINED LANES and a self-load
    // figure is a property of the harness AT A LANE COUNT, not a constant. Nothing in the row
    // format recorded the lane count, so the rot was invisible in the rows themselves and could
    // only be found by re-measuring.
    //
    // Printed unconditionally, including on an UNFILTERED full sweep — which is exactly the case
    // the `lane_filter=` line above does not cover, and exactly how the stale bound was taken.
    println!(
        "lane_count={} rounds={reps} (frankentorch-vyaia: the harness's own load scales with this \
         product; a self-load bound quoted without it cannot be checked later)",
        lanes.len()
    );

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
    // DEFAULT RAISED 4 -> 32 (frankentorch-68pwz, NEGATIVE_EVIDENCE item 54), and
    // the "counts must match" rule above is now WRONG. It is kept in place because
    // the reasoning that produced it is still correct about BIAS; it was simply
    // answering the wrong question.
    //
    // WHY 32, AND A SIGN ERROR IN THE ORIGINAL JUSTIFICATION — READ THIS BEFORE
    // TRUSTING THE REASONING BELOW IT.
    //
    // Measured across 21 lanes on a drift-clean run, the MIN-estimator A/A nulls
    // were:
    //
    //   FT  0.794 0.845 0.878 0.902 0.905 0.908 0.910 0.914 0.922 0.927 0.975 ...
    //
    // Eleven of twenty-one below 0.98 and only two in band. I originally wrote that
    // a null below 1.0 means the SECOND half is FASTER — an arm still warming up.
    // THAT IS BACKWARDS. The null is
    // `median_ratio_ci(first_half, second_half) = median(first)/median(second)`
    // over TIMES, so a value below 1.0 means the first half was CHEAPER and the
    // SECOND HALF IS SLOWER. The arm DEGRADES across a round; it does not warm up.
    // A cold arm would put the null ABOVE 1.0.
    //
    // The empirical effect of raising the count is real and unchanged: the same
    // binary and rounds at 32 moved the in-band count 2 -> 5 and the median FT null
    // 0.927 -> 0.983. So 32 stays. But it was NOT fixing insufficient warmup, and
    // the mechanism is still open — pre-touching more pages before the timed region
    // plausibly reduces later allocator/page-fault work, which would attenuate a
    // degradation rather than complete a warm-up.
    //
    // The same inversion is corrected in NEGATIVE_EVIDENCE item 54.
    //
    // The incumbent's nulls skew the OTHER way (12 of 21 above 1.02), which under
    // the corrected reading means torch's second half is FASTER — torch warms
    // during the run where we degrade. Its count is left alone because more warmup
    // is the right remedy for that direction and it already reads this variable.
    //
    // This does not reintroduce the frankentorch-2kgum bias. That bias came from
    // under-warming one arm relative to what IT needs; both arms are now warmed to
    // their own sufficiency, which is what makes the comparison fair.
    let warmup_iters: usize = std::env::var("FT_H2H_WARMUP")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(32);
    for (_, run_lane) in &lanes {
        let mut warm = 0.0;
        for _ in 0..warmup_iters {
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
    // Per-round `slot0 / median(slot1..3)` for OUR arm — `frankentorch-hi9r6`, item 169.
    //
    // NEGATIVE_EVIDENCE item 147 found the summed conv2d lanes' failing FT null is entirely a
    // cold slot 0, and item 149 then tried to test whether that generalises. Item 149's answer
    // was wrong, and the reason is the whole justification for this field: it paired slot
    // profiles taken in ONE run against null outcomes quoted from OTHER runs. Re-paired inside a
    // single run, its own strongest counterexample reverses — `linear_narrow` had a FAILING null
    // (1.102) alongside a slot-0 step of 1.283, which supports the mechanism it was cited against.
    //
    // The defect was structural, not careless: the null is printed by the harness while the slot
    // profile needed `FT_H2H_DUMP_SLOTS` plus an external script, so pairing them was always a
    // manual cross-referencing step, and that step is where runs got mixed. Computing the ratio
    // here and printing it next to the null makes the pairing automatic and same-run by
    // construction.
    //
    // Kept as a per-round ratio and reduced by MEDIAN, deliberately: a ratio of medians across
    // rounds would let a between-round load ramp leak into it, which is exactly the failure the
    // drift-robust estimator in `scripts/h2h_slot_profile.py` exists to avoid.
    let mut ft_slot0_ratio: Vec<Vec<f64>> = vec![Vec::with_capacity(reps); lanes.len()];
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

    // frankentorch-68pwz / NEGATIVE_EVIDENCE item 49: sample the load EVERY ROUND,
    // not just at the two endpoints. The endpoint pair cannot see a mid-run
    // excursion — a real run read 17.78 at start and 17.72 at end and was certified
    // steady, and those two numbers are equally consistent with a host that climbed
    // to 60 in between. The balanced square interleaves arms throughout, so an
    // excursion lands on some samples and not others, which is exactly the confound
    // the gate exists to catch.
    let mut load_series: Vec<f64> = Vec::with_capacity(reps + 2);
    if let Some(l) = load_at_start {
        load_series.push(l);
    }

    // frankentorch-pbkvs: see the ISOLATION PROBE note in the slot loop below.
    // frankentorch-68pwz: PER-ARM CLOCKS. Cores on this box run at different speeds at
    // the same instant -- measured 1429 MHz against 4018 MHz, a 2.812x spread, bimodal
    // with about a quarter parked at the floor. A ratio whose arms sat at different
    // clocks is partly a frequency ratio, and loadavg cannot see it.
    //
    // Sampled immediately after each arm's four slots, so each accumulator holds the
    // clock state that arm was actually measured under rather than a run-level average.
    let mut ft_mhz: Vec<f64> = Vec::with_capacity(reps * lanes.len());
    let mut pt_mhz: Vec<f64> = Vec::with_capacity(reps * lanes.len());
    // frankentorch-rayon-pool-width-qq8as: let a paired run flip the shipped chunk-width
    // lever. 0 restores the pre-lever one-chunk-per-plane split, which is the A arm.
    if let Some(width) = std::env::var("FT_PARALLEL_TARGET_WORKERS")
        .ok()
        .and_then(|v| v.parse::<usize>().ok())
    {
        let previous = ft_kernel_cpu::set_parallel_target_workers(width);
        println!(
            "parallel_target_workers={width} (was {previous}); 0 means one chunk per \
             plane, the pre-qq8as split"
        );
    }
    // frankentorch-rayon-pool-width-qq8as: flip the per-op NARROW POOL for a paired run.
    // Its width is fixed at first use via FT_NARROW_POOL_WIDTH (default 8, where item 51's
    // curve turns); this only selects whether kernels use it.
    if let Ok(raw) = std::env::var("FT_NARROW_POOL") {
        let on = raw == "1";
        // Renamed to `set_max_pool3d_pool_enabled` by 9075bcdb when the narrow pool became
        // MaxPool3d's own; this call site was the one the rename missed, and it left the whole
        // gauntlet board unbuildable. Same semantics, same bool, same previous-value return.
        let previous = ft_kernel_cpu::set_max_pool3d_pool_enabled(on);
        println!("narrow_pool={on} (was {previous})");
    }
    // Same flag, read once above so the co-process choice could be made before the spawn.
    let isolate_arm = isolate_arm_early;
    // frankentorch-68pwz item 48: the MIRROR of the probe above. With
    // `FT_H2H_NO_FT_ARM` the incumbent keeps its four slots in their square positions
    // and NO FrankenTorch work runs between them, so its uncontended per-slot times can
    // be read from THIS harness rather than a separate script.
    //
    // That matters because item 48's contention figure leaned on a standalone Python
    // probe which clones its leaf per sample — a caveat that made the number a bound
    // rather than a measurement. Same harness, same protocol, same reduction, only our
    // arm removed: the difference IS the contention we impose.
    let isolate_incumbent = std::env::var("FT_H2H_NO_FT_ARM").is_ok();
    if isolate_incumbent {
        println!(
            "INCUMBENT-ONLY MODE (FT_H2H_NO_FT_ARM): no FrankenTorch work runs between \
             the incumbent's samples. Every FT column and every ratio below is a \
             PLACEHOLDER and carries no claim. Read the incumbent's per-slot times only."
        );
    }
    if isolate_arm {
        println!(
            "ISOLATION MODE (FT_H2H_NO_INCUMBENT): no incumbent work runs between our \
             samples. Every PT column and every vs-PyTorch ratio below is a PLACEHOLDER \
             and carries no claim. Read the FT A/A null and nothing else."
        );
    }
    for _ in 0..reps {
        if let Some(l) = ft_api::harness_provenance::load_average_1m() {
            load_series.push(l);
        }
        for (index, (name, run_lane)) in lanes.iter().enumerate() {
            // PER-ROUND WARM-UP, OPT-IN AND SYMMETRIC — `frankentorch-hi9r6`, item 167.
            //
            // NEGATIVE_EVIDENCE item 147 located the summed conv2d lanes' one-sided FT null
            // exactly: SLOT 0. The round's first FT sample costs 20-30% more than the other
            // three, which agree with each other, because `FT_H2H_WARMUP` runs ONCE per lane
            // before any round begins (line ~1766) and the other lanes run in between, so a
            // lane's first sample each round arrives cold. The null is first-half/second-half
            // and slot 0 sits in the first half, so it fails while the 4-sample-median ratio
            // barely moves.
            //
            // Item 147d named the fix — a discarded sample per lane per round — and declined to
            // make it, because changing the default would move EVERY lane's numbers on a board a
            // dozen agents quote from. That objection is about the DEFAULT, not the code, so the
            // flag defaults to 0 and nothing moves unless it is asked for.
            //
            // SYMMETRIC, AND THAT IS NOT A DETAIL. Warming only our arm would delete a real cost
            // from our side and leave it on the incumbent's — a better ratio produced by the
            // harness rather than by the code, which is precisely what this campaign's rules
            // exist to forbid. Both arms get the same number of discarded samples, in the same
            // place, so whatever the warm-up removes it removes from both.
            //
            // WHAT IT CHANGES, STATED SO NO ROW IS MIXED UP: with this set the lanes measure
            // STEADY-STATE cost; without it they measure cost including a per-round cold start.
            // Both are legitimate questions and they are DIFFERENT questions. A row taken with
            // this flag is comparable only to another row taken with the same flag, and every
            // certified standing on the board today was taken without it.
            //
            // Item 147's diagnosis predicts the summed lanes' FT null moves to ~1.0 here while
            // the masked lanes (which show no slot-0 step) barely move. If instead the summed
            // null stays high, item 147 is wrong about the mechanism and slot 0 is a symptom of
            // something else.
            // item 190: warm only the lanes named, so one sweep can carry a lane twice — once
            // warmed, once not — and the comparison stays inside a single window.
            // item 220: this lane's tile-floor setting, chosen by name exactly as
            // `lane_warmup` is. Applied around OUR arm only — the incumbent is
            // untouched, which is the point: the same PyTorch arm, our knob moved.
            let lane_tile = tile_adaptive_lanes.iter().any(|l| l.as_str() == *name);
            let lane_warmup =
                // `name` is `&&str` here (the lane vec is `Vec<(&str, LaneRun)>`), so this compares
                // via `as_str()` and a deref rather than `&String == &&str`, which has no impl.
                if round_warmup_lanes.is_empty()
                    || round_warmup_lanes.iter().any(|l| l.as_str() == *name)
                {
                    round_warmup
                } else {
                    0
                };
            for _ in 0..lane_warmup {
                if !isolate_incumbent {
                    let _ = run_lane();
                }
                if !isolate_arm {
                    let _ = incumbent_sample(&mut stdin, &mut reader, name)?;
                }
            }
            let mut incumbent_slots = Vec::with_capacity(4);
            let mut ft_slots = Vec::with_capacity(4);
            for incumbent_slot in BALANCED_SQUARE {
                if incumbent_slot {
                    // frankentorch-pbkvs: ISOLATION PROBE. With `FT_H2H_NO_INCUMBENT`
                    // set, our four slots keep their positions in the balanced square
                    // but NO incumbent work runs between them. Everything else --
                    // warmup, round count, reduction, the null -- is identical, so the
                    // co-process is the only variable.
                    //
                    // It exists because three hypotheses for this lane's one-sided FT
                    // null (accumulation, host load, sampling order) are all refuted,
                    // and every survivor lives or dies on whether the co-process is
                    // present. If the null goes clean here, the interleaved null is
                    // partly measuring the interleaving -- which would bear on every
                    // A/A null read on a contended lane, INCLUDING ones that passed.
                    //
                    // The incumbent columns are placeholders in this mode and every
                    // vs-PyTorch number in the output is meaningless; the banner says
                    // so, and only the FT null should be read.
                    if isolate_arm {
                        incumbent_slots.push(1.0);
                        continue;
                    }
                    let (ms, grad) = incumbent_sample(&mut stdin, &mut reader, name)?;
                    incumbent_slots.push(ms);
                    pt_grads[index] = Some(grad);
                } else {
                    if isolate_incumbent {
                        ft_slots.push(1.0);
                        continue;
                    }
                    let prev_tile = ft_kernel_cpu::gemm_tile_col_floor_adaptive();
                    ft_kernel_cpu::set_gemm_tile_col_floor_adaptive(lane_tile);
                    let (ms, checksum) = run_lane();
                    ft_kernel_cpu::set_gemm_tile_col_floor_adaptive(prev_tile);
                    ft_slots.push(ms);
                    checksums[index] = checksum;
                }
            }
            // frankentorch-68pwz: instrument the CERTIFICATION GATE itself. Every
            // hypothesis so far (allocator, warmup, host drift) was aimed at why
            // the null fails; none asked what the null is actually comparing.
            //
            // It is NOT first-rounds against last-rounds. Within EACH round the
            // balanced square gives each arm four slots, and the null compares
            // slots 0,1 against slots 2,3 — a WITHIN-ROUND position effect,
            // accumulated across rounds. A null below 1.0 therefore says the arm
            // is FASTER in the later slots of the same round, every round.
            //
            // I guessed that was cache eviction between rounds — slots 0,1 paying
            // a refill that 2,3 skip. THE DUMP REFUTED THAT. The spikes are not
            // positional at all; they land in any slot, on either arm:
            //
            //   ft=[35.0 42.1 36.7 34.6]   clean round
            //   ft=[83.4 38.7 50.8 38.1]   slot 0 spiked 2.2x
            //   ft=[35.7 38.1 39.4 75.7]   slot 3 spiked 2.1x
            //   pt=[ 6.2  8.0  7.2 37.8]   the INCUMBENT spiked 5x
            //
            // So the gate is rejecting SPORADIC CONTENDED SAMPLES, not a bias.
            // That matters for the estimator: `paired_slot_median` over TWO values
            // is a mean, so one spike drags the half it lands in, and the median
            // null fails. The MIN reduction of the same two slots discards the
            // spike, and on the very run that produced the dump above the min
            // nulls read 1.002 and 0.994 — both PASS — while the median null
            // failed at 0.855. The estimator, not the host, was vetoing the row.
            //
            // Set FT_H2H_DUMP_SLOTS to a lane name (or `*`) to print the raw four
            // slots per arm per round. Nothing else in the harness shows them, and
            // three hypotheses died before anyone looked.
            if let Ok(target) = std::env::var("FT_H2H_DUMP_SLOTS")
                && (target == "*" || target.as_str() == *name)
            {
                println!(
                    "SLOTS lane={name} ft=[{:.4} {:.4} {:.4} {:.4}] pt=[{:.4} {:.4} {:.4} {:.4}]",
                    ft_slots[0],
                    ft_slots[1],
                    ft_slots[2],
                    ft_slots[3],
                    incumbent_slots[0],
                    incumbent_slots[1],
                    incumbent_slots[2],
                    incumbent_slots[3]
                );
            }
            if let Some((min_mhz, median_mhz, _, spread)) =
                ft_api::harness_provenance::cpu_mhz_stats()
            {
                // TWO different numbers, and the distinction decides whether a ratio is
                // readable. `median_mhz` is how fast a typical core is running -- its
                // variation over time is thermal/boost behaviour. `spread` is how far
                // apart two cores are AT THE SAME INSTANT, which is what makes one arm's
                // threads faster than the other's. An idle snapshot shows 2.9x; only the
                // in-measurement value says whether that mattered while we sampled.
                ft_mhz.push(median_mhz);
                pt_mhz.push(spread);
                let _ = min_mhz;
            }
            pt_first_half[index].push(paired_slot_median([incumbent_slots[0], incumbent_slots[1]]));
            pt_second_half[index]
                .push(paired_slot_median([incumbent_slots[2], incumbent_slots[3]]));
            // item 169: slot 0 against the median of the other three, WITHIN this round.
            let tail_median = median(vec![ft_slots[1], ft_slots[2], ft_slots[3]]);
            if tail_median > 0.0 {
                ft_slot0_ratio[index].push(ft_slots[0] / tail_median);
            }
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

    if !ft_mhz.is_empty() {
        let summarise = |mut v: Vec<f64>| {
            v.sort_by(f64::total_cmp);
            (v[0], v[v.len() / 2], v[v.len() - 1])
        };
        let (flo, fmid, fhi) = summarise(ft_mhz.clone());
        let (slo, smid, shi) = summarise(pt_mhz.clone());
        println!(
            "\nCLOCKS DURING SAMPLING (sampled once per lane per round, both arms inside \
             the same round)\n  \
             typical core   min {flo:.0} median {fmid:.0} max {fhi:.0} MHz   (boost/thermal \
             behaviour over time)\n  \
             CROSS-CORE SPREAD at one instant   min {slo:.3}x median {smid:.3}x max \
             {shi:.3}x   (how far apart two cores were WHILE we sampled)\n  \
             The spread is the number that decides comparability: it is what makes one \
             arm's threads faster than the other's. Compare it against the idle snapshot in \
             the cpu_mhz header line -- if the idle spread is large and this one is small, \
             the cores boosted once work arrived and the clock confound did not bite."
        );
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
    if let Some(l) = load_at_end {
        load_series.push(l);
    }
    // The SERIES gate is authoritative; the endpoint gate is still computed and
    // printed so the two can be compared on real runs. It is strictly stricter —
    // identical when the extremes are the endpoints, refusing more otherwise — so
    // this can only reject runs the old gate accepted, never the reverse.
    let endpoint_quotable =
        ft_api::harness_provenance::load_drift_is_quotable(load_at_start, load_at_end);
    let load_quotable = ft_api::harness_provenance::load_series_is_quotable(&load_series);
    let series_drift = ft_api::harness_provenance::load_series_drift(&load_series);
    println!(
        "load_series n={} worst_drift={} endpoint_gate={} series_gate={} \
         (frankentorch-68pwz item 49: the endpoint pair cannot see a mid-run excursion; \
         a divergence between these two gates IS the finding)",
        load_series.len(),
        series_drift.map_or_else(|| "unknown".to_owned(), |d| format!("{d:.3}x")),
        if endpoint_quotable { "PASS" } else { "DRIFTED" },
        if load_quotable { "PASS" } else { "DRIFTED" }
    );
    // OVERSUBSCRIPTION GATE — `frankentorch-hi9r6`, item 194.
    //
    // The drift gate below tests whether load MOVED, and says so in its own message: "a steady
    // busy host is measurable". That is true up to a point and false past it. This invocation
    // proved where the line is — it ran at load 79.87 -> 88.19 on a 64-core box, the drift gate
    // said PASS because the load was steady, and PyTorch's arm read 78-113 ms on lanes where it
    // normally reads a few. A steady queue of 88 runnable tasks on 64 cores is not a busy host,
    // it is a host where every arm is waiting for a core, and the wait lands on whichever arm
    // happens to be scheduled worse.
    //
    // `loadavg > online_cpus` is the principled form of that: above it the run queue exceeds the
    // machine, so timings include queueing delay rather than work. Below it a busy host really is
    // measurable and the drift gate is the right instrument.
    //
    // Reported, not enforced. The harness refuses nothing on its own — it prints what it saw and
    // the reader decides — but a row taken above this line should be treated exactly like a
    // LOAD-DRIFTED one, whatever its nulls say. The nulls do NOT protect against this: in the run
    // that motivated the gate, one lane's FT null read 1.013 and its PT null 1.032 while the
    // incumbent arm was inflated twentyfold.
    let peak_load = load_series.iter().copied().fold(f64::NAN, f64::max);
    let cores = std::thread::available_parallelism().map_or(0, std::num::NonZeroUsize::get);
    if cores > 0 && peak_load.is_finite() && peak_load > cores as f64 {
        println!(
            "OVERSUBSCRIBED: peak loadavg {peak_load:.2} exceeds online_cpus {cores} — the run \
             queue was longer than the machine, so every arm spent time WAITING rather than \
             working. Treat every row below as unquotable, exactly as if the drift gate had \
             failed; passing A/A nulls do not rescue it (item 194)."
        );
    }
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

    // COLD-START DETECTOR — `frankentorch-vyaia`, acceptance item 1.
    //
    // The bead's measurement: five consecutive runs of ONE binary, and the two that passed the
    // drift gate were the two that STARTED at the level the harness itself sustains. From an idle
    // box the sweep supplies its own +20 and the RELATIVE gate reads 9 -> 28 as a 3x excursion.
    // Waiting for a quiet host therefore makes this gate HARDER to pass, which is the reverse of
    // how every runbook in this repo reads.
    //
    // The rise is MEASURED here rather than assumed, because the constant is not a constant:
    // frankentorch-5q3io banked the self-load at ~+0.62 absolute and that figure is now off by
    // more than an order of magnitude, purely because the sweep gained lanes underneath it.
    // Printing the rise every run means the number is re-derived by the instrument instead of
    // being quoted from a bead that cannot know today's lane count.
    //
    // Advisory, never a refusal. A run that trips this is not WRONG, it is the WARM-UP: discard it
    // unread and measure from the next one. The discard is fixed in advance and unconditional, so
    // it is a scheduling rule and not a filter on results.
    //
    // The rise is a LOWER bound on our own contribution and an UPPER bound on nothing: a peer that
    // started while we sampled lands in it too. `concurrent_measurements` above is what separates
    // those two, and it is printed first for that reason.
    //
    // THE WINDOW LENGTH IS PART OF THE READING, and the first run under this code is why. It
    // printed `rise=+0.00` over a seven-second sweep, which reads as "this harness self-loads
    // nothing" and means no such thing: `loadavg` is a 1-minute EWMA, so a step of S sustained
    // for T seconds moves it by S * (1 - exp(-T/60)) and a seven-second window can show at most
    // 11% of our own load however large it is. A short sweep CANNOT refute self-load. The response
    // factor is therefore printed next to the rise, so the two can never be read apart.
    let load_window = load_window_started.elapsed().as_secs_f64();
    let ewma_response = 1.0 - (-load_window / 60.0).exp();
    // THE DERIVATION THAT DOES NOT NEED A QUIET HOST — `frankentorch-vyaia` acceptance item 2.
    //
    // Everything above measures the HOST moving and cannot say how much of the movement is ours.
    // This measures us directly: CPU seconds burned by this process and its reaped incumbent
    // child, over the wall time of the window. That ratio is mean parallelism — the average number
    // of tasks we kept runnable — which is the quantity loadavg approximates, computed on our own
    // accounting where a peer's compile contributes exactly zero.
    //
    // It reads slightly LOW against loadavg by construction: loadavg also counts tasks in
    // uninterruptible sleep, and CPU time does not. For a compute-bound sweep that gap is small,
    // and it errs in the safe direction for a bound that exists to be compared against ~+0.62.
    if let (Some(before), Some(after)) = (cpu_at_start, self_cpu_seconds()) {
        let cpu = after - before;
        if load_window > 0.0 {
            println!(
                "self_load DIRECT={:.2} (both arms burned {cpu:.1} CPU-seconds over \
                 window={load_window:.0}s at lane_count={} rounds={reps}) — measured on our own \
                 /proc accounting, so a peer's compile contributes nothing to it. Compare against \
                 frankentorch-5q3io's ~+0.62, which was inferred from loadavg deltas at an \
                 unrecorded lane count (frankentorch-vyaia acceptance item 2).",
                cpu / load_window,
                lanes.len()
            );
        }
    }
    if let Some(start) = load_at_start.filter(|_| peak_load.is_finite()) {
        let rise = peak_load - start;
        println!(
            "self_load rise={rise:+.2} (start {start:.2} -> peak {peak_load:.2}) over \
             window={load_window:.0}s at lane_count={} rounds={reps} — loadavg is a 1-minute EWMA, \
             so this window could show at most {:.0}% of any sustained step \
             (frankentorch-vyaia: measured, not assumed)",
            lanes.len(),
            ewma_response * 100.0
        );
        if load_window < 60.0 {
            println!(
                "self_load NOT INTERPRETABLE: a {load_window:.0}s window is shorter than the \
                 loadavg time constant, so this rise is a floor and a +0.00 is not evidence of \
                 anything. If the whole rise were ours and sustained it would imply {:.2}. \
                 Re-derive the bound on a FULL sweep — the configuration every banked row and \
                 frankentorch-5q3io's stale ~+0.62 were taken under (frankentorch-vyaia).",
                if ewma_response > 0.0 {
                    rise / ewma_response
                } else {
                    f64::NAN
                }
            );
        }
        // WHOSE LOAD IS IT — a correction to the line above, forced by its own first full run.
        //
        // That run printed `rise=+50.18` and fired COLD-START, which as first worded claimed
        // "this invocation supplied MORE load than the host was already carrying". It did not.
        // The sweep ran at `rayon_threads=8` against an 8-thread incumbent, and a process cannot
        // make more tasks runnable than it has threads: our structural ceiling was 16, so at least
        // 34 of that 50 came from the peers compiling on the box. The rise is OURS PLUS THEIRS and
        // attributing it to us is exactly the over-claim this bead exists to correct upstream.
        //
        // The ceiling is a hard bound rather than an estimate, which is why it is worth printing:
        // anything above it is external BY CONSTRUCTION, with no assumption about the peers at all.
        let self_ceiling = (rayon::current_num_threads() + INCUMBENT_THREADS) as f64;
        let ours = rise.min(self_ceiling);
        if rise > self_ceiling {
            println!(
                "EXTERNAL LOAD: the rise ({rise:.2}) exceeds our structural ceiling \
                 ({self_ceiling:.0} = {} rayon + {INCUMBENT_THREADS} incumbent threads), so at \
                 least {:.2} of it came from OTHER processes. This window mixes our ramp with the \
                 host's and its rise cannot be read as self-load (frankentorch-vyaia).",
                rayon::current_num_threads(),
                rise - self_ceiling
            );
        }
        if ours > start {
            println!(
                "COLD-START: the part of the rise that CAN be ours ({ours:.2}, capped at the \
                 {self_ceiling:.0}-thread ceiling) already exceeds what the host was carrying when \
                 we started ({start:.2}), so the drift gate above is largely reading OUR OWN ramp \
                 rather than the host's. Treat this run as the discarded warm-up and measure from \
                 the NEXT invocation on this host (frankentorch-vyaia)."
            );
        }
    }

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
        conv2d_frame_diagnostics(name, ft_ms);
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
            // ROUNDS ARE A LEVER ON THE NULL — `frankentorch-hi9r6`, item 206.
            //
            // Item 193c measured this rather than assuming it: on `conv2d_big_masked`, at loadavg
            // 22-35 throughout, the same ELF walked
            //
            //     16 rounds   PT 1.050 FAIL   FT 1.003
            //     32 rounds   PT 0.985 PASS   FT 0.977   (short by 0.003)
            //     64 rounds   PT PASS         FT PASS
            //
            // The A/A null is a ratio of per-round medians, so its variance falls with the round
            // count: the gate can be bought with TIME instead of with a quiet host. Every earlier
            // attempt that session waited for a calm window that never came.
            //
            // Suggested only when BOTH nulls are already close, because rounds cannot rescue a row
            // that is failing for a reason other than sampling noise — a genuinely drifting or
            // oversubscribed host (item 194) moves the arms themselves, and more of it just
            // averages a different quantity more precisely.
            let worst = (pt_null_ratio - 1.0).abs().max((ft_null_ratio - 1.0).abs());
            if worst < 4.0 * BALANCED_NULL_MAX_DEVIATION {
                println!(
                    "      ROUNDS MAY CLEAR IT: worst null is {worst:.3} off 1.0 against a \
                     +/-{BALANCED_NULL_MAX_DEVIATION:.2} band, which is sampling noise rather than \
                     a moving host. The null is a ratio of per-round medians, so try \
                     FT_H2H_REPS={} (item 193c walked 16 -> 32 -> 64 to PASS at loadavg 22-35).",
                    reps.saturating_mul(2).max(32)
                );
            }
        }
        // item 169: print the slot-0 ratio on EVERY row, passing or failing, next to the null it
        // is meant to explain. Printed unconditionally on purpose — item 149 went wrong by
        // pairing a slot profile from one run against a null from another, and a diagnostic that
        // only appears on failing rows would still force that cross-referencing for the passing
        // ones it has to be compared against.
        // The kernels-only f32 conv2d lane carries a hand-rolled pad inside its timed region that
        // the session lane does not pay in the same form. Print it beside the row so nobody
        // computes "session cost" from a subtraction that is contaminated in one direction.
        if *name == "conv2d_f32_kernels" {
            let pad = CONV2D_F32_KERNELS_PAD_MS.with(std::cell::Cell::get);
            if pad > 0.0 {
                println!(
                    "    pad_ms = {pad:.3} of this lane's {ft_ms:.3} ms is a hand-rolled \
                     zero-init + SERIAL scalar copy, not kernel work. The session cost is \
                     conv2d_f32 - (this - pad_ms), NOT conv2d_f32 - this; the uncorrected \
                     subtraction has been reading NEGATIVE."
                );
            }
        }
        if *name == "attention" {
            let (forward, loss_sum, backward) = ATTENTION_SPLIT_MS.with(std::cell::Cell::get);
            if forward > 0.0 || backward > 0.0 {
                println!(
                    "    frames: forward {forward:.3} ms | loss sum {loss_sum:.3} ms | backward \
                     {backward:.3} ms of this lane's {ft_ms:.3} ms. Forward is tiled QK-to-softmax-to-V; \
                     backward recomputes per-head score scratch."
                );
            }
        }
        if !ft_slot0_ratio[index].is_empty() {
            let slot0 = median(ft_slot0_ratio[index].clone());
            println!(
                "    slot0/median(slot1..3) = {slot0:.3} (our arm, per-round median over {} rounds){}",
                ft_slot0_ratio[index].len(),
                if slot0 > 1.0 + BALANCED_NULL_MAX_DEVIATION {
                    " <- the round's FIRST sample is COLD; NEGATIVE_EVIDENCE item 147. \
                     Re-run with FT_H2H_ROUND_WARMUP=1 to see whether the null follows it"
                } else {
                    ""
                }
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
    // Four suffixes name a lever-off twin. `_nopool` and `_nopool_dense` are
    // `ft_core::buffer_pool` switched off (the latter pairs to the dense base);
    // `_noshortcut` is the
    // PReLU+sum deforest declined through its hook exit (frankentorch-k1hto);
    // `_serialfwd` is the group-norm forward forced onto the pre-`group_norm_parallel_pays`
    // serial schedule (frankentorch-dmpho).
    for (index, (name, _)) in lanes.iter().enumerate() {
        let Some((base, lever)) = name
            .strip_suffix("_nopool_dense")
            .map(|base| (format!("{base}_dense"), "buffer pool"))
            .or_else(|| {
                name.strip_suffix("_nopool")
                    .map(|base| (base.to_owned(), "buffer pool"))
            })
            .or_else(|| {
                name.strip_suffix("_noshortcut")
                    .map(|base| (base.to_owned(), "sum shortcut"))
            })
            .or_else(|| {
                name.strip_suffix("_serialfwd")
                    .map(|base| (base.to_owned(), "parallel forward gate"))
            })
            .or_else(|| {
                name.strip_suffix("_recompute")
                    .map(|base| (base.to_owned(), "forward-statistics reuse"))
            })
            .or_else(|| {
                // frankentorch-yu1zm: the pooling forwards' UNINIT output vs the
                // pre-lever zeroed allocation, flipped per arm inside one process.
                name.strip_suffix("_zeroed")
                    .map(|base| (base.to_owned(), "uninit output"))
            })
        else {
            continue;
        };
        let Some(base_index) = lanes.iter().position(|(other, _)| *other == base.as_str()) else {
            continue;
        };
        let rounds = ft_times[index].len().min(ft_times[base_index].len());
        let mut treated = Vec::with_capacity(rounds);
        let mut control = Vec::with_capacity(rounds);
        for (control_time, treated_time) in ft_times[index]
            .iter()
            .zip(ft_times[base_index].iter())
            .take(rounds)
        {
            control.push(*control_time);
            treated.push(*treated_time);
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
        // NOT rewritten to `as_chunks::<8>()` as clippy asks: this loop is INSIDE the
        // timed region it exists to measure (the 8-accumulator reduction against the
        // sequential one), and swapping the iterator changes the code being timed. The
        // lint is a style preference; silently editing a measured region to satisfy it
        // would invalidate the comparison this probe banks. frankentorch-l2zki.
        #[allow(clippy::chunks_exact_to_as_chunks)]
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
        .as_chunks::<4>()
        .0
        .iter()
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

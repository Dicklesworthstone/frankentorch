//! frankentorch-5237b. A size-gated kernel must compute the same thing on both sides of its gate.
//!
//! FrankenTorch selects a different implementation by tensor SIZE — `PARALLEL_THRESHOLD` = 8192,
//! `SOFTMAX_PARALLEL_NUMEL_THRESHOLD` = 65536, `SCALAR_UNARY`/`SIMD_UNARY_PARALLEL_THRESHOLD` =
//! 524288, plus ~57 further size-gated sites in ft-api alone. Above the gate, a second
//! implementation of the same math runs. Nothing systematically checked that the two agree.
//!
//! This is not a hypothetical class. It has produced two real defects:
//!
//! * `frankentorch-fmmns` (open): the `max_pool1d` 2x2 pair SPECIALIZATION disagrees with the
//!   generic route, and the divergent window turns out to be all-NaN.
//! * `frankentorch-x4cx3`: prelu's zero-convention bug had to be driven past the 65_536-element
//!   gate specifically, because the parallel branches carry their own compare. Fixing the serial
//!   path alone would have left them wrong.
//!
//! # Why this needs no oracle
//!
//! For an ELEMENTWISE op, chunking is semantically the identity: `f([a, b, c])` must equal
//! `[f([a]), f([b]), f([c])]` concatenated. So the same values are pushed through the op twice —
//! once as one large tensor (above every gate, parallel path) and once as small chunks (below
//! every gate, serial path) — and compared. FrankenTorch is its own oracle here, and any
//! difference is a defect regardless of what torch does. That is the same certificate shape the
//! RNG sub-stream work used, where serial == parallel bit-identical *was* the proof.
//!
//! # Bit patterns, not values
//!
//! Every comparison is on `to_bits()`. `-0.0 == +0.0` is true under `==`, and the sign of zero is
//! precisely what distinguishes "took the multiply branch" from "took the identity branch" — the
//! discrimination that caught the rrelu forward bug in `frankentorch-3eq5b`. A value comparison
//! would be blind to it.
//!
//! # The inputs straddle every branch boundary
//!
//! `x4cx3` survived for years because no PReLU fixture contained an exact `0.0`. So the payload
//! here leads with the boundary values these ops actually branch on — signed zeros, the relu6 `6`,
//! the hardtanh `±1`, the hardswish/hardsigmoid `±3`, the shrink `±0.5`, the softplus `20`, plus
//! NaN, infinities and denormals — before padding with benign values to cross the size gate.

use ft_api::FrankenTorchSession;
use ft_autograd::TensorNodeId;
use ft_core::ExecutionMode;

/// Comfortably above `SCALAR_UNARY_PARALLEL_THRESHOLD` (524_288), so the whole-tensor arm is on
/// the parallel side of every gate in the crate.
const BIG: usize = 600_000;

/// Comfortably below `PARALLEL_THRESHOLD` (8192), so each chunk is on the serial side of every
/// gate. Not a divisor of `BIG`, deliberately: the final chunk is short, which is where an
/// off-by-one in a chunked kernel would show up.
const CHUNK: usize = 4_096;

/// Values these activations actually branch on, in the first positions so they land in the head
/// of the tensor, and repeated later (see `payload`) so they also land mid-tensor where a
/// parallel implementation splits work.
fn boundary_values() -> Vec<f64> {
    vec![
        0.0,
        -0.0,
        1.0,
        -1.0,
        6.0,
        -6.0,
        3.0,
        -3.0,
        0.5,
        -0.5,
        20.0,
        -20.0,
        f64::NAN,
        f64::INFINITY,
        f64::NEG_INFINITY,
        f64::MIN_POSITIVE,
        -f64::MIN_POSITIVE,
        f64::MIN_POSITIVE / 2.0, // subnormal
        2.5,
        -2.5,
    ]
}

/// `BIG` values: the boundary set at the front, the same set again straddling a chunk seam and a
/// likely rayon split point, and benign varied values elsewhere.
fn payload() -> Vec<f64> {
    let bounds = boundary_values();
    // Benign, varied, and not monotonic — a bug that only shows on some sign or magnitude still
    // gets hit. Drawn from a table rather than computed from the index, so no integer-to-float
    // cast is needed (this crate lints at pedantic).
    const FILLER: [f64; 11] = [
        -6.5, -3.25, -1.75, -0.75, -0.125, 0.0625, 0.875, 1.5, 2.75, 4.5, 7.25,
    ];
    let mut v: Vec<f64> = (0..BIG).map(|i| FILLER[i % FILLER.len()]).collect();
    for (k, b) in bounds.iter().enumerate() {
        v[k] = *b;
        // Straddle a chunk seam.
        v[CHUNK - 3 + k] = *b;
        // And a point deep inside the tensor.
        v[BIG / 2 + k] = *b;
        // And the tail, where a short final chunk lives.
        v[BIG - bounds.len() + k] = *b;
    }
    v
}

/// An op under test: applied to a whole tensor and to chunks of the same values.
struct Op {
    name: &'static str,
    apply: fn(&mut FrankenTorchSession, TensorNodeId) -> TensorNodeId,
}

fn ops() -> Vec<Op> {
    vec![
        Op {
            name: "relu",
            apply: |s, x| s.tensor_relu(x).unwrap(),
        },
        Op {
            name: "relu6",
            apply: |s, x| s.tensor_relu6(x).unwrap(),
        },
        Op {
            name: "leaky_relu",
            apply: |s, x| s.tensor_leaky_relu(x).unwrap(),
        },
        Op {
            name: "elu",
            apply: |s, x| s.tensor_elu(x).unwrap(),
        },
        Op {
            name: "celu",
            apply: |s, x| s.tensor_celu(x, 2.0).unwrap(),
        },
        Op {
            name: "selu",
            apply: |s, x| s.tensor_selu(x).unwrap(),
        },
        Op {
            name: "hardtanh",
            apply: |s, x| s.tensor_hardtanh(x).unwrap(),
        },
        Op {
            name: "hardswish",
            apply: |s, x| s.tensor_hardswish(x).unwrap(),
        },
        Op {
            name: "hardsigmoid",
            apply: |s, x| s.tensor_hardsigmoid(x).unwrap(),
        },
        Op {
            name: "softplus",
            apply: |s, x| s.tensor_softplus(x).unwrap(),
        },
        Op {
            name: "rrelu",
            apply: |s, x| s.tensor_rrelu(x, 0.125, 1.0 / 3.0).unwrap(),
        },
        Op {
            name: "hardshrink",
            apply: |s, x| s.tensor_hardshrink(x, 0.5).unwrap(),
        },
        Op {
            name: "softshrink",
            apply: |s, x| s.tensor_softshrink(x, 0.5).unwrap(),
        },
        Op {
            name: "sigmoid",
            apply: |s, x| s.tensor_sigmoid(x).unwrap(),
        },
        Op {
            name: "tanh",
            apply: |s, x| s.tensor_tanh(x).unwrap(),
        },
        Op {
            name: "abs",
            apply: |s, x| s.tensor_abs(x).unwrap(),
        },
        Op {
            name: "exp",
            apply: |s, x| s.tensor_exp(x).unwrap(),
        },
        Op {
            name: "threshold",
            apply: |s, x| s.tensor_threshold(x, 1.0, 9.0).unwrap(),
        },
        // prelu carries an EXPLICIT `xv.len() >= 65_536` gate selecting a parallel backward, and
        // it is the op whose parallel-branch compare was actually wrong in frankentorch-x4cx3 —
        // the serial fix alone would not have covered it. The weight is built inside the closure
        // and left no-grad, so only grad_x is compared, which is elementwise.
        Op {
            name: "prelu",
            apply: |s, x| {
                let w = s.tensor_variable(vec![0.25], vec![1], false).unwrap();
                s.tensor_prelu(x, w).unwrap()
            },
        },
        // Negative weight: multiplying is not the identity on zero, so this is the variant where
        // the SIGN OF ZERO distinguishes "took the multiply branch" from "passed the input
        // through" — the discrimination that caught the rrelu forward bug in frankentorch-3eq5b.
        Op {
            name: "prelu_negw",
            apply: |s, x| {
                let w = s.tensor_variable(vec![-0.25], vec![1], false).unwrap();
                s.tensor_prelu(x, w).unwrap()
            },
        },
    ]
}

/// Forward values for `vals` in one shot (parallel side of every gate).
fn forward_whole(op: &Op, vals: &[f64]) -> Vec<f64> {
    let mut s = FrankenTorchSession::new(ExecutionMode::Strict);
    let x = s
        .tensor_variable(vals.to_vec(), vec![vals.len()], false)
        .expect("leaf");
    let out = (op.apply)(&mut s, x);
    s.tensor_values(out).expect("values")
}

/// Forward values for `vals` in sub-threshold chunks (serial side of every gate).
fn forward_chunked(op: &Op, vals: &[f64]) -> Vec<f64> {
    let mut acc = Vec::with_capacity(vals.len());
    for chunk in vals.chunks(CHUNK) {
        let mut s = FrankenTorchSession::new(ExecutionMode::Strict);
        let x = s
            .tensor_variable(chunk.to_vec(), vec![chunk.len()], false)
            .expect("leaf");
        let out = (op.apply)(&mut s, x);
        acc.extend(s.tensor_values(out).expect("values"));
    }
    acc
}

/// Input gradient of `sum(op(x) * 3)`, in one shot. The scale makes the upstream gradient 3.0
/// rather than 1.0, so a pass-through is distinguishable from a hardcoded 1.0.
fn grad_whole(op: &Op, vals: &[f64]) -> Vec<f64> {
    let mut s = FrankenTorchSession::new(ExecutionMode::Strict);
    let x = s
        .tensor_variable(vals.to_vec(), vec![vals.len()], true)
        .expect("leaf");
    let out = (op.apply)(&mut s, x);
    let scaled = s.tensor_mul_scalar(out, 3.0).expect("scale");
    let loss = s.tensor_sum(scaled).expect("sum");
    let report = s.tensor_backward(loss).expect("backward");
    s.tensor_gradient(&report, x).expect("grad").to_vec()
}

/// Same gradient, computed in sub-threshold chunks.
fn grad_chunked(op: &Op, vals: &[f64]) -> Vec<f64> {
    let mut acc = Vec::with_capacity(vals.len());
    for chunk in vals.chunks(CHUNK) {
        let mut s = FrankenTorchSession::new(ExecutionMode::Strict);
        let x = s
            .tensor_variable(chunk.to_vec(), vec![chunk.len()], true)
            .expect("leaf");
        let out = (op.apply)(&mut s, x);
        let scaled = s.tensor_mul_scalar(out, 3.0).expect("scale");
        let loss = s.tensor_sum(scaled).expect("sum");
        let report = s.tensor_backward(loss).expect("backward");
        acc.extend(s.tensor_gradient(&report, x).expect("grad"));
    }
    acc
}

/// Compare two runs bit-for-bit, reporting the first few divergences with their input value.
fn diff_report(
    name: &str,
    what: &str,
    vals: &[f64],
    whole: &[f64],
    chunked: &[f64],
) -> Vec<String> {
    let mut out = Vec::new();
    if whole.len() != chunked.len() {
        out.push(format!(
            "  {name:12} {what}: length {} vs {}",
            whole.len(),
            chunked.len()
        ));
        return out;
    }
    let mut shown = 0;
    let mut total = 0;
    for (i, (w, c)) in whole.iter().zip(chunked.iter()).enumerate() {
        if w.to_bits() != c.to_bits() {
            total += 1;
            if shown < 4 {
                out.push(format!(
                    "  {name:12} {what}[{i}] x={:?} whole={:?} (0x{:016x}) chunked={:?} (0x{:016x})",
                    vals[i],
                    w,
                    w.to_bits(),
                    c,
                    c.to_bits()
                ));
                shown += 1;
            }
        }
    }
    if total > shown {
        out.push(format!(
            "  {name:12} {what}: ...and {} more divergent elements",
            total - shown
        ));
    }
    out
}

/// The forward result must not depend on which side of the size gate the tensor falls.
#[test]
fn forward_is_identical_across_the_parallel_threshold() {
    let vals = payload();
    let mut failures = Vec::new();
    for op in ops() {
        let whole = forward_whole(&op, &vals);
        let chunked = forward_chunked(&op, &vals);
        failures.extend(diff_report(op.name, "fwd", &vals, &whole, &chunked));
    }
    assert!(
        failures.is_empty(),
        "size-gated kernels disagree with their own sub-threshold path (frankentorch-5237b).\n\
         Chunking is the identity for an elementwise op, so these are real defects:\n{}",
        failures.join("\n")
    );
}

/// And neither must the gradient — the branch that `x4cx3` found wrong lived in the backward.
#[test]
fn backward_is_identical_across_the_parallel_threshold() {
    let vals = payload();
    let mut failures = Vec::new();
    for op in ops() {
        let whole = grad_whole(&op, &vals);
        let chunked = grad_chunked(&op, &vals);
        failures.extend(diff_report(op.name, "bwd", &vals, &whole, &chunked));
    }
    assert!(
        failures.is_empty(),
        "size-gated BACKWARD kernels disagree with their own sub-threshold path \
         (frankentorch-5237b):\n{}",
        failures.join("\n")
    );
}

/// The harness itself must be able to fail. If chunking were silently not exercising a different
/// path — or if `diff_report` were blind — the two tests above would pass vacuously.
///
/// Feeds a deliberately mismatched pair through the same comparison and asserts it is caught,
/// including the `-0.0` vs `+0.0` case that a value comparison would miss.
#[test]
fn the_comparison_actually_detects_a_difference() {
    let vals = vec![1.0, 0.0, 2.0];
    let a = vec![1.0, 0.0, 2.0];
    let b = vec![1.0, -0.0, 2.0];
    let report = diff_report("selftest", "fwd", &vals, &a, &b);
    assert!(
        !report.is_empty(),
        "diff_report must catch -0.0 vs +0.0; it compares bits precisely so this cannot be missed"
    );
    assert!(
        diff_report("selftest", "fwd", &vals, &a, &a).is_empty(),
        "identical inputs must produce no findings"
    );
}

// ---------------------------------------------------------------------------
// Row-chunking, for the ops that element-chunking cannot reach (frankentorch-h37tn)
// ---------------------------------------------------------------------------

/// Rows in the whole-tensor arm. `ROWS * COLS` = 131_072, above the 65_536
/// `SOFTMAX_PARALLEL_NUMEL_THRESHOLD`.
const ROWS: usize = 256;
/// Columns. Each single-row slice is `COLS` = 512 elements, far below every gate.
///
/// Deliberately modest: the difference between the two arms must be the NUMEL gate and
/// nothing else. A very wide row could plausibly get its own per-row reduction
/// parallelised, and then a legitimate re-association would be indistinguishable from a
/// defect.
const COLS: usize = 512;

/// A reduction/normalisation applied along `dim = 1`, where every output element depends
/// only on its own row.
struct RowOp {
    name: &'static str,
    apply: fn(&mut FrankenTorchSession, TensorNodeId) -> TensorNodeId,
}

/// Only ops that are genuinely ROW-INDEPENDENT along dim=1. This list is a claim about
/// each op, not a convenience: a reduction over dim=0 would NOT belong here, and putting
/// one in would make the test assert something false.
fn row_ops() -> Vec<RowOp> {
    vec![
        RowOp {
            name: "softmax",
            apply: |s, x| s.tensor_softmax(x, 1).unwrap(),
        },
        RowOp {
            name: "log_softmax",
            apply: |s, x| s.tensor_log_softmax(x, 1).unwrap(),
        },
        RowOp {
            name: "sum_dim",
            apply: |s, x| s.tensor_sum_dim(x, 1).unwrap(),
        },
        RowOp {
            name: "mean_dim",
            apply: |s, x| s.tensor_mean_dim(x, 1).unwrap(),
        },
        RowOp {
            name: "cumsum",
            apply: |s, x| s.tensor_cumsum(x, 1).unwrap(),
        },
    ]
}

/// `ROWS * COLS` values. Every row carries the boundary set in its first positions, so the
/// boundaries are present in each independently-computed slice rather than only in row 0.
fn row_payload() -> Vec<f64> {
    let bounds = boundary_values();
    let mut v = Vec::with_capacity(ROWS * COLS);
    for r in 0..ROWS {
        for c in 0..COLS {
            if c < bounds.len() {
                v.push(bounds[c]);
            } else {
                // Varied per row so rows are not identical — an implementation that
                // computed one row and broadcast it would still be caught.
                let f = [0.25_f64, -0.75, 1.5, -2.25, 3.5, -0.125, 0.875];
                v.push(f[(r + c) % f.len()]);
            }
        }
    }
    v
}

fn row_forward_whole(op: &RowOp, vals: &[f64]) -> Vec<f64> {
    let mut s = FrankenTorchSession::new(ExecutionMode::Strict);
    let x = s
        .tensor_variable(vals.to_vec(), vec![ROWS, COLS], false)
        .expect("leaf");
    let out = (op.apply)(&mut s, x);
    s.tensor_values(out).expect("values")
}

/// The same rows, each computed on its own as a `[1, COLS]` tensor — below every gate.
fn row_forward_chunked(op: &RowOp, vals: &[f64]) -> Vec<f64> {
    let mut acc = Vec::new();
    for row in vals.chunks(COLS) {
        let mut s = FrankenTorchSession::new(ExecutionMode::Strict);
        let x = s
            .tensor_variable(row.to_vec(), vec![1, COLS], false)
            .expect("leaf");
        let out = (op.apply)(&mut s, x);
        acc.extend(s.tensor_values(out).expect("values"));
    }
    acc
}

fn row_grad_whole(op: &RowOp, vals: &[f64]) -> Vec<f64> {
    let mut s = FrankenTorchSession::new(ExecutionMode::Strict);
    let x = s
        .tensor_variable(vals.to_vec(), vec![ROWS, COLS], true)
        .expect("leaf");
    let out = (op.apply)(&mut s, x);
    let scaled = s.tensor_mul_scalar(out, 3.0).expect("scale");
    let loss = s.tensor_sum(scaled).expect("sum");
    let report = s.tensor_backward(loss).expect("backward");
    s.tensor_gradient(&report, x).expect("grad").to_vec()
}

fn row_grad_chunked(op: &RowOp, vals: &[f64]) -> Vec<f64> {
    let mut acc = Vec::new();
    for row in vals.chunks(COLS) {
        let mut s = FrankenTorchSession::new(ExecutionMode::Strict);
        let x = s
            .tensor_variable(row.to_vec(), vec![1, COLS], true)
            .expect("leaf");
        let out = (op.apply)(&mut s, x);
        let scaled = s.tensor_mul_scalar(out, 3.0).expect("scale");
        let loss = s.tensor_sum(scaled).expect("sum");
        let report = s.tensor_backward(loss).expect("backward");
        acc.extend(s.tensor_gradient(&report, x).expect("grad"));
    }
    acc
}

/// Row-independent ops must not depend on how many rows share the tensor.
///
/// Element-chunking (the tests above) cannot reach these: a softmax row is computed from
/// the whole row, so splitting by element would change the answer. Splitting by ROW is the
/// identity instead, one axis up, and it still straddles the 65_536 numel gate because the
/// whole arm is 131_072 elements and each slice is 512.
///
/// A FULL reduction is deliberately absent. `tensor_sum` over everything associates its
/// additions differently in a parallel tree than in a serial loop, so the two are EXPECTED
/// to differ in the last bits; asserting bit-equality there would be wrong, not strict.
#[test]
fn row_independent_forward_is_identical_across_the_numel_gate() {
    let vals = row_payload();
    let mut failures = Vec::new();
    for op in row_ops() {
        let whole = row_forward_whole(&op, &vals);
        let chunked = row_forward_chunked(&op, &vals);
        failures.extend(diff_report(op.name, "row-fwd", &vals, &whole, &chunked));
    }
    assert!(
        failures.is_empty(),
        "row-independent kernels disagree with their own per-row path (frankentorch-h37tn).\n\
         Splitting by row is the identity for these ops, so these are real defects:\n{}",
        failures.join("\n")
    );
}

/// And the same for their gradients.
#[test]
fn row_independent_backward_is_identical_across_the_numel_gate() {
    let vals = row_payload();
    let mut failures = Vec::new();
    for op in row_ops() {
        let whole = row_grad_whole(&op, &vals);
        let chunked = row_grad_chunked(&op, &vals);
        failures.extend(diff_report(op.name, "row-bwd", &vals, &whole, &chunked));
    }
    assert!(
        failures.is_empty(),
        "row-independent BACKWARD kernels disagree with their own per-row path \
         (frankentorch-h37tn):\n{}",
        failures.join("\n")
    );
}

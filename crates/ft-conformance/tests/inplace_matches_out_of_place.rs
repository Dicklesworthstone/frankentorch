//! frankentorch-q06x9. `x.foo_()` must leave exactly what `y = x.foo()` would have produced.
//!
//! FrankenTorch ships 222 in-place `tensor_*_` ops. Each is a SECOND IMPLEMENTATION of an op
//! that also exists out-of-place, and the contract between them is exact, not approximate.
//! Nothing swept it.
//!
//! # This has already gone wrong here, and the source says so
//!
//! From `tensor_hardswish_` in `ft-api`:
//!
//! ```text
//! // Match the kernel (and torch) hardswish: x*(x+3)/6 on the active window,
//! // NOT x*(x/6+0.5).clamp which differs by 1 ULP from the out-of-place
//! // tensor_hardswish. frankentorch in-place parity vein.
//! ```
//!
//! So an in-place op has already drifted 1 ULP from its out-of-place sibling and had to be
//! repaired — found by hand, on one op, out of 222. The in-place surface was also
//! bulk-converted (~73 ops collapsed from clone → serial map → writeback into a single
//! `par_iter_mut` pass, claiming bit-exactness). A bulk mechanical change verified per-op by
//! hand is exactly where one op ends up with a subtly different expression.
//!
//! # No oracle needed
//!
//! FrankenTorch is its own reference: run both forms on the same values and compare. Any
//! difference is a defect whatever torch does.
//!
//! # Bit patterns, not values
//!
//! The hardswish_ precedent was **1 ULP**, which a tolerance comparison hides, and `-0.0 ==
//! +0.0` under `==`, which hides a sign-of-zero branch difference. Everything here compares
//! `to_bits()`.
//!
//! # Inputs straddle the branch boundaries, at a parallel size
//!
//! `frankentorch-x4cx3`'s PReLU zero bug survived years because no fixture contained an exact
//! `0.0`. The payload leads with the values these ops branch on and is sized above the
//! parallel gates, because the in-place ops are the ones that were parallelised.

use ft_api::FrankenTorchSession;
use ft_autograd::TensorNodeId;
use ft_core::ExecutionMode;

/// Above `SCALAR_UNARY_PARALLEL_THRESHOLD` (524_288) so both forms run their parallel paths.
const N: usize = 600_000;

/// Values these ops branch on, plus the awkward ones.
fn boundaries() -> Vec<f64> {
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
        f64::MIN_POSITIVE / 2.0,
        2.5,
        -2.5,
    ]
}

/// `N` values with the boundary set at the front, mid-tensor, and in the tail.
fn payload() -> Vec<f64> {
    const FILLER: [f64; 11] = [
        -6.5, -3.25, -1.75, -0.75, -0.125, 0.0625, 0.875, 1.5, 2.75, 4.5, 7.25,
    ];
    let b = boundaries();
    let mut v: Vec<f64> = (0..N).map(|i| FILLER[i % FILLER.len()]).collect();
    for (k, val) in b.iter().enumerate() {
        v[k] = *val;
        v[N / 2 + k] = *val;
        v[N - b.len() + k] = *val;
    }
    v
}

/// An op that exists in both forms.
struct Pair {
    name: &'static str,
    out_of_place: fn(&mut FrankenTorchSession, TensorNodeId) -> TensorNodeId,
    in_place: fn(&mut FrankenTorchSession, TensorNodeId),
}

fn pairs() -> Vec<Pair> {
    vec![
        Pair {
            name: "relu",
            out_of_place: |s, x| s.tensor_relu(x).unwrap(),
            in_place: |s, t| s.tensor_relu_(t).unwrap(),
        },
        Pair {
            name: "relu6",
            out_of_place: |s, x| s.tensor_relu6(x).unwrap(),
            in_place: |s, t| s.tensor_relu6_(t).unwrap(),
        },
        Pair {
            name: "abs",
            out_of_place: |s, x| s.tensor_abs(x).unwrap(),
            in_place: |s, t| s.tensor_abs_(t).unwrap(),
        },
        Pair {
            name: "ceil",
            out_of_place: |s, x| s.tensor_ceil(x).unwrap(),
            in_place: |s, t| s.tensor_ceil_(t).unwrap(),
        },
        Pair {
            name: "floor",
            out_of_place: |s, x| s.tensor_floor(x).unwrap(),
            in_place: |s, t| s.tensor_floor_(t).unwrap(),
        },
        Pair {
            name: "sqrt",
            out_of_place: |s, x| s.tensor_sqrt(x).unwrap(),
            in_place: |s, t| s.tensor_sqrt_(t).unwrap(),
        },
        Pair {
            name: "exp",
            out_of_place: |s, x| s.tensor_exp(x).unwrap(),
            in_place: |s, t| s.tensor_exp_(t).unwrap(),
        },
        Pair {
            name: "sigmoid",
            out_of_place: |s, x| s.tensor_sigmoid(x).unwrap(),
            in_place: |s, t| s.tensor_sigmoid_(t).unwrap(),
        },
        Pair {
            name: "tanh",
            out_of_place: |s, x| s.tensor_tanh(x).unwrap(),
            in_place: |s, t| s.tensor_tanh_(t).unwrap(),
        },
        // The op that is already known to have drifted 1 ULP. If the sweep ever finds
        // anything, expect it to look like this one did.
        Pair {
            name: "hardswish",
            out_of_place: |s, x| s.tensor_hardswish(x).unwrap(),
            in_place: |s, t| s.tensor_hardswish_(t).unwrap(),
        },
        Pair {
            name: "hardsigmoid",
            out_of_place: |s, x| s.tensor_hardsigmoid(x).unwrap(),
            in_place: |s, t| s.tensor_hardsigmoid_(t).unwrap(),
        },
        Pair {
            name: "hardtanh",
            out_of_place: |s, x| s.tensor_hardtanh(x).unwrap(),
            // The in-place form is parameterised where the out-of-place one hardcodes
            // torch's defaults, so the bounds are passed explicitly to compare the same op.
            in_place: |s, t| s.tensor_hardtanh_(t, -1.0, 1.0).unwrap(),
        },
        Pair {
            name: "selu",
            out_of_place: |s, x| s.tensor_selu(x).unwrap(),
            in_place: |s, t| s.tensor_selu_(t).unwrap(),
        },
        Pair {
            name: "softplus",
            out_of_place: |s, x| s.tensor_softplus(x).unwrap(),
            in_place: |s, t| s.tensor_softplus_(t).unwrap(),
        },
        Pair {
            name: "celu",
            out_of_place: |s, x| s.tensor_celu(x, 2.0).unwrap(),
            in_place: |s, t| s.tensor_celu_(t, 2.0).unwrap(),
        },
        Pair {
            name: "rrelu",
            out_of_place: |s, x| s.tensor_rrelu(x, 0.125, 1.0 / 3.0).unwrap(),
            in_place: |s, t| s.tensor_rrelu_(t, 0.125, 1.0 / 3.0).unwrap(),
        },
    ]
}

/// Report bit differences, with the input that produced them.
fn diff(name: &str, vals: &[f64], oop: &[f64], ip: &[f64]) -> Vec<String> {
    let mut out = Vec::new();
    if oop.len() != ip.len() {
        out.push(format!(
            "  {name:12} length {} (out-of-place) vs {} (in-place)",
            oop.len(),
            ip.len()
        ));
        return out;
    }
    let (mut shown, mut total) = (0, 0);
    for (i, (a, b)) in oop.iter().zip(ip.iter()).enumerate() {
        if a.to_bits() != b.to_bits() {
            total += 1;
            if shown < 4 {
                out.push(format!(
                    "  {name:12} [{i}] x={:?}  out-of-place={:?} (0x{:016x})  in-place={:?} (0x{:016x})",
                    vals[i],
                    a,
                    a.to_bits(),
                    b,
                    b.to_bits()
                ));
                shown += 1;
            }
        }
    }
    if total > shown {
        out.push(format!("  {name:12} ...and {} more", total - shown));
    }
    out
}

/// Every in-place op must agree bit-for-bit with its out-of-place sibling.
#[test]
fn in_place_ops_match_their_out_of_place_siblings() {
    let vals = payload();
    let mut failures = Vec::new();
    for p in pairs() {
        // Out-of-place: read the fresh output.
        let oop = {
            let mut s = FrankenTorchSession::new(ExecutionMode::Strict);
            let x = s
                .tensor_variable(vals.clone(), vec![N], false)
                .expect("leaf");
            let out = (p.out_of_place)(&mut s, x);
            s.tensor_values(out).expect("values")
        };
        // In-place: mutate a fresh leaf holding the same values, then read the leaf itself.
        let ip = {
            let mut s = FrankenTorchSession::new(ExecutionMode::Strict);
            let t = s
                .tensor_variable(vals.clone(), vec![N], false)
                .expect("leaf");
            (p.in_place)(&mut s, t);
            s.tensor_values(t).expect("values")
        };
        failures.extend(diff(p.name, &vals, &oop, &ip));
    }
    assert!(
        failures.is_empty(),
        "in-place ops disagree with their out-of-place siblings (frankentorch-q06x9).\n\
         The two are the same op and must be bit-identical; hardswish_ already drifted 1 ULP \
         once:\n{}",
        failures.join("\n")
    );
}

/// The comparator must be able to fail, or the test above proves nothing.
///
/// Includes the two cases that motivated comparing bits: a 1-ULP difference (the hardswish_
/// precedent) and `-0.0` vs `+0.0`, both of which a value comparison would pass.
#[test]
fn the_comparator_catches_one_ulp_and_signed_zero() {
    let vals = [1.0, 0.0];
    // The value torch's scalar mish kernel produces, and its immediate f64 neighbour.
    // Built with from_bits(+1) rather than two near-identical decimal literals: it says
    // "exactly one ULP apart" in the code instead of asking the reader to count digits,
    // and it cannot silently collapse to the same f64 if someone edits a digit.
    let base = 0.561_148_377_643_851_8_f64;
    let a = [base, 0.0];
    let one_ulp_up = [f64::from_bits(base.to_bits() + 1), 0.0];
    assert_ne!(
        base.to_bits(),
        one_ulp_up[0].to_bits(),
        "the two probe values must actually differ, or this self-test is vacuous"
    );
    assert!(
        !diff("selftest", &vals, &a, &one_ulp_up).is_empty(),
        "a 1-ULP difference must be caught — that is exactly what hardswish_ had"
    );
    assert!(
        !diff("selftest", &vals, &a, &[a[0], -0.0]).is_empty(),
        "-0.0 vs +0.0 must be caught; it is equal under == and hides a branch difference"
    );
    assert!(
        diff("selftest", &vals, &a, &a).is_empty(),
        "identical inputs must produce no findings"
    );
}

// ---------------------------------------------------------------------------
// Round 2: the scalar-arg and binary forms (frankentorch-jvst1)
// ---------------------------------------------------------------------------

/// A second operand that straddles boundaries in its own right.
///
/// A binary op can branch on **either** side — division by zero, `atan2`'s quadrant
/// selection, the sign rules of `remainder` — so holding the right-hand side at a constant
/// would leave half of each op's branch structure untested.
fn rhs_payload() -> Vec<f64> {
    const RHS: [f64; 9] = [2.0, -2.0, 0.5, -0.5, 1.0, -1.0, 3.0, -3.0, 0.25];
    let b = boundaries();
    let mut v: Vec<f64> = (0..N).map(|i| RHS[i % RHS.len()]).collect();
    // Boundary values on the right-hand side too, offset from the left's so the pair
    // (lhs, rhs) hits combinations like (0.0, NaN) and (inf, 0.0) rather than only the
    // diagonal (0.0, 0.0), (NaN, NaN).
    for (k, val) in b.iter().enumerate() {
        v[k] = b[(k + 3) % b.len()];
        v[N / 2 + k] = *val;
        v[N - b.len() + k] = b[(b.len() - 1) - k];
    }
    v
}

/// An op taking one extra scalar argument, applied identically to both forms.
struct ScalarPair {
    name: &'static str,
    out_of_place: fn(&mut FrankenTorchSession, TensorNodeId) -> TensorNodeId,
    in_place: fn(&mut FrankenTorchSession, TensorNodeId),
}

/// **The arguments are matched deliberately, and that is the whole difficulty.**
///
/// `q06x9` found `tensor_hardtanh_(t, min, max)` against `tensor_hardtanh(x)`, which
/// hardcodes torch's defaults. The same trap appears twice more here: `tensor_leaky_relu_`
/// takes an explicit slope where `tensor_leaky_relu` hardcodes `0.01`, and `tensor_elu_`
/// takes alpha where `tensor_elu` hardcodes `1.0`. Passing anything else would compare two
/// *different* ops and either fail spuriously or — worse — pass while measuring nothing.
fn scalar_pairs() -> Vec<ScalarPair> {
    vec![
        // EXCLUDED, deliberately: `tensor_add_scalar_` exists in-place but there is no
        // out-of-place `tensor_add_scalar` to compare it against — unlike `mul_scalar`,
        // which has both. Nothing to assert here, so it is named rather than silently
        // dropped; whether the missing out-of-place form is a surface gap is a separate
        // question from whether the two forms agree.
        ScalarPair {
            name: "mul_scalar",
            out_of_place: |s, x| s.tensor_mul_scalar(x, -1.25).unwrap(),
            in_place: |s, t| s.tensor_mul_scalar_(t, -1.25).unwrap(),
        },
        ScalarPair {
            name: "pow",
            out_of_place: |s, x| s.tensor_pow(x, 2.0).unwrap(),
            in_place: |s, t| s.tensor_pow_(t, 2.0).unwrap(),
        },
        ScalarPair {
            name: "clamp",
            out_of_place: |s, x| s.tensor_clamp(x, -1.0, 1.0).unwrap(),
            in_place: |s, t| s.tensor_clamp_(t, -1.0, 1.0).unwrap(),
        },
        // 0.01 is what the out-of-place form hardcodes; anything else compares two ops.
        ScalarPair {
            name: "leaky_relu",
            out_of_place: |s, x| s.tensor_leaky_relu(x).unwrap(),
            in_place: |s, t| s.tensor_leaky_relu_(t, 0.01).unwrap(),
        },
        // Likewise alpha = 1.0 for elu.
        ScalarPair {
            name: "elu",
            out_of_place: |s, x| s.tensor_elu(x).unwrap(),
            in_place: |s, t| s.tensor_elu_(t, 1.0).unwrap(),
        },
        ScalarPair {
            name: "softshrink",
            out_of_place: |s, x| s.tensor_softshrink(x, 0.5).unwrap(),
            in_place: |s, t| s.tensor_softshrink_(t, 0.5).unwrap(),
        },
        ScalarPair {
            name: "hardshrink",
            out_of_place: |s, x| s.tensor_hardshrink(x, 0.5).unwrap(),
            in_place: |s, t| s.tensor_hardshrink_(t, 0.5).unwrap(),
        },
        ScalarPair {
            name: "threshold",
            out_of_place: |s, x| s.tensor_threshold(x, 1.0, 9.0).unwrap(),
            in_place: |s, t| s.tensor_threshold_(t, 1.0, 9.0).unwrap(),
        },
    ]
}

/// An op taking a second tensor.
struct BinaryPair {
    name: &'static str,
    out_of_place: fn(&mut FrankenTorchSession, TensorNodeId, TensorNodeId) -> TensorNodeId,
    in_place: fn(&mut FrankenTorchSession, TensorNodeId, TensorNodeId),
}

fn binary_pairs() -> Vec<BinaryPair> {
    vec![
        BinaryPair {
            name: "add",
            out_of_place: |s, a, b| s.tensor_add(a, b).unwrap(),
            in_place: |s, t, b| s.tensor_add_(t, b).unwrap(),
        },
        BinaryPair {
            name: "mul",
            out_of_place: |s, a, b| s.tensor_mul(a, b).unwrap(),
            in_place: |s, t, b| s.tensor_mul_(t, b).unwrap(),
        },
        BinaryPair {
            name: "div",
            out_of_place: |s, a, b| s.tensor_div(a, b).unwrap(),
            in_place: |s, t, b| s.tensor_div_(t, b).unwrap(),
        },
        BinaryPair {
            name: "atan2",
            out_of_place: |s, a, b| s.tensor_atan2(a, b).unwrap(),
            in_place: |s, t, b| s.tensor_atan2_(t, b).unwrap(),
        },
    ]
}

/// Scalar-argument in-place ops must match their out-of-place siblings.
#[test]
fn scalar_arg_in_place_ops_match_their_out_of_place_siblings() {
    let vals = payload();
    let mut failures = Vec::new();
    for p in scalar_pairs() {
        let oop = {
            let mut s = FrankenTorchSession::new(ExecutionMode::Strict);
            let x = s
                .tensor_variable(vals.clone(), vec![N], false)
                .expect("leaf");
            let out = (p.out_of_place)(&mut s, x);
            s.tensor_values(out).expect("values")
        };
        let ip = {
            let mut s = FrankenTorchSession::new(ExecutionMode::Strict);
            let t = s
                .tensor_variable(vals.clone(), vec![N], false)
                .expect("leaf");
            (p.in_place)(&mut s, t);
            s.tensor_values(t).expect("values")
        };
        failures.extend(diff(p.name, &vals, &oop, &ip));
    }
    assert!(
        failures.is_empty(),
        "scalar-argument in-place ops disagree with their out-of-place siblings \
         (frankentorch-jvst1):\n{}",
        failures.join("\n")
    );
}

/// Binary in-place ops must match too, with a right-hand side that also straddles
/// boundaries so combinations like `(0.0, NaN)` and `(inf, 0.0)` are exercised.
#[test]
fn binary_in_place_ops_match_their_out_of_place_siblings() {
    let lhs = payload();
    let rhs = rhs_payload();
    let mut failures = Vec::new();
    for p in binary_pairs() {
        let oop = {
            let mut s = FrankenTorchSession::new(ExecutionMode::Strict);
            let a = s.tensor_variable(lhs.clone(), vec![N], false).expect("lhs");
            let b = s.tensor_variable(rhs.clone(), vec![N], false).expect("rhs");
            let out = (p.out_of_place)(&mut s, a, b);
            s.tensor_values(out).expect("values")
        };
        let ip = {
            let mut s = FrankenTorchSession::new(ExecutionMode::Strict);
            let t = s.tensor_variable(lhs.clone(), vec![N], false).expect("lhs");
            let b = s.tensor_variable(rhs.clone(), vec![N], false).expect("rhs");
            (p.in_place)(&mut s, t, b);
            s.tensor_values(t).expect("values")
        };
        failures.extend(diff(p.name, &lhs, &oop, &ip));
    }
    assert!(
        failures.is_empty(),
        "binary in-place ops disagree with their out-of-place siblings \
         (frankentorch-jvst1):\n{}",
        failures.join("\n")
    );
}

// ---------------------------------------------------------------------------
// Round 3: the matmul family (frankentorch-5fppy)
// ---------------------------------------------------------------------------

/// These ops take several tensor operands with shape relationships, so they cannot share
/// the flat payload the earlier rounds use. Each case builds its own operands and runs the
/// op either in place or out of place, selected by `in_place`, returning the resulting
/// values so the two can be compared bit-for-bit.
///
/// **A different risk profile from rounds 1-2.** These route through GEMM-like kernels with
/// their own blocking and parallel gates, so a divergence here would most likely be a
/// re-association showing up in the last bits across many elements, rather than the clean
/// NaN-vs-0.0 split `hardshrink_` produced. That is still a defect — the two forms are the
/// same op — and the comparison stays exact rather than being softened to a tolerance.
struct MatCase {
    name: &'static str,
    run: fn(&mut FrankenTorchSession, bool) -> Vec<f64>,
}

/// Deterministic, varied, and small in magnitude so a GEMM accumulation stays well inside
/// f64 range — the point is bit-equality of two implementations, not stressing the numerics.
fn seq(n: usize, offset: usize) -> Vec<f64> {
    const T: [f64; 13] = [
        0.5, -0.25, 1.5, -0.75, 2.25, -1.25, 0.125, -0.0625, 3.5, -2.5, 0.875, -1.75, 1.0,
    ];
    (0..n).map(|i| T[(i + offset) % T.len()]).collect()
}

fn mat_cases() -> Vec<MatCase> {
    vec![
        MatCase {
            name: "addmm",
            run: |s, ip| {
                // input [64,32], mat1 [64,48], mat2 [48,32]
                let inp = s
                    .tensor_variable(seq(64 * 32, 0), vec![64, 32], false)
                    .unwrap();
                let m1 = s
                    .tensor_variable(seq(64 * 48, 3), vec![64, 48], false)
                    .unwrap();
                let m2 = s
                    .tensor_variable(seq(48 * 32, 7), vec![48, 32], false)
                    .unwrap();
                if ip {
                    s.tensor_addmm_(inp, m1, m2, 0.75, 1.25).unwrap();
                    s.tensor_values(inp).unwrap()
                } else {
                    let o = s.tensor_addmm(inp, m1, m2, 0.75, 1.25).unwrap();
                    s.tensor_values(o).unwrap()
                }
            },
        },
        MatCase {
            name: "baddbmm",
            run: |s, ip| {
                // input [4,16,12], batch1 [4,16,20], batch2 [4,20,12]
                let inp = s
                    .tensor_variable(seq(4 * 16 * 12, 1), vec![4, 16, 12], false)
                    .unwrap();
                let b1 = s
                    .tensor_variable(seq(4 * 16 * 20, 5), vec![4, 16, 20], false)
                    .unwrap();
                let b2 = s
                    .tensor_variable(seq(4 * 20 * 12, 9), vec![4, 20, 12], false)
                    .unwrap();
                if ip {
                    s.tensor_baddbmm_(inp, b1, b2, 0.5, 1.5).unwrap();
                    s.tensor_values(inp).unwrap()
                } else {
                    let o = s.tensor_baddbmm(inp, b1, b2, 0.5, 1.5).unwrap();
                    s.tensor_values(o).unwrap()
                }
            },
        },
        MatCase {
            name: "addbmm",
            run: |s, ip| {
                // input [16,12], batch1 [4,16,20], batch2 [4,20,12] — reduces over the batch,
                // so this is the case where accumulation ORDER is most likely to differ.
                let inp = s
                    .tensor_variable(seq(16 * 12, 2), vec![16, 12], false)
                    .unwrap();
                let b1 = s
                    .tensor_variable(seq(4 * 16 * 20, 6), vec![4, 16, 20], false)
                    .unwrap();
                let b2 = s
                    .tensor_variable(seq(4 * 20 * 12, 11), vec![4, 20, 12], false)
                    .unwrap();
                if ip {
                    s.tensor_addbmm_(inp, b1, b2, 0.25, 1.75).unwrap();
                    s.tensor_values(inp).unwrap()
                } else {
                    let o = s.tensor_addbmm(inp, b1, b2, 0.25, 1.75).unwrap();
                    s.tensor_values(o).unwrap()
                }
            },
        },
        MatCase {
            name: "addmv",
            run: |s, ip| {
                // input [64], mat [64,48], vec [48]
                let inp = s.tensor_variable(seq(64, 4), vec![64], false).unwrap();
                let mat = s
                    .tensor_variable(seq(64 * 48, 8), vec![64, 48], false)
                    .unwrap();
                let v = s.tensor_variable(seq(48, 12), vec![48], false).unwrap();
                if ip {
                    s.tensor_addmv_(inp, mat, v, 0.5, 2.0).unwrap();
                    s.tensor_values(inp).unwrap()
                } else {
                    let o = s.tensor_addmv(inp, mat, v, 0.5, 2.0).unwrap();
                    s.tensor_values(o).unwrap()
                }
            },
        },
        MatCase {
            name: "addr",
            run: |s, ip| {
                // input [64,32], vec1 [64], vec2 [32]
                let inp = s
                    .tensor_variable(seq(64 * 32, 5), vec![64, 32], false)
                    .unwrap();
                let v1 = s.tensor_variable(seq(64, 2), vec![64], false).unwrap();
                let v2 = s.tensor_variable(seq(32, 10), vec![32], false).unwrap();
                if ip {
                    s.tensor_addr_(inp, v1, v2, 1.25, 0.75).unwrap();
                    s.tensor_values(inp).unwrap()
                } else {
                    let o = s.tensor_addr(inp, v1, v2, 1.25, 0.75).unwrap();
                    s.tensor_values(o).unwrap()
                }
            },
        },
        MatCase {
            name: "addcmul",
            run: |s, ip| {
                // Elementwise, so this one crosses the parallel gate like rounds 1-2 and
                // carries boundary values.
                let mut a = seq(70_000, 0);
                let b = boundaries();
                for (k, v) in b.iter().enumerate() {
                    a[k] = *v;
                    a[35_000 + k] = *v;
                }
                let inp = s.tensor_variable(a, vec![70_000], false).unwrap();
                let t1 = s
                    .tensor_variable(seq(70_000, 4), vec![70_000], false)
                    .unwrap();
                let t2 = s
                    .tensor_variable(seq(70_000, 9), vec![70_000], false)
                    .unwrap();
                if ip {
                    s.tensor_addcmul_(inp, t1, t2, 1.5).unwrap();
                    s.tensor_values(inp).unwrap()
                } else {
                    let o = s.tensor_addcmul(inp, t1, t2, 1.5).unwrap();
                    s.tensor_values(o).unwrap()
                }
            },
        },
        MatCase {
            name: "addcdiv",
            run: |s, ip| {
                let mut a = seq(70_000, 1);
                let b = boundaries();
                for (k, v) in b.iter().enumerate() {
                    a[k] = *v;
                    a[35_000 + k] = *v;
                }
                let inp = s.tensor_variable(a, vec![70_000], false).unwrap();
                let t1 = s
                    .tensor_variable(seq(70_000, 6), vec![70_000], false)
                    .unwrap();
                // Divisor: no zeros, so the lane tests the op rather than division by zero,
                // which addcmul's boundary set already covers on the accumulator side.
                let t2 = s
                    .tensor_variable(seq(70_000, 2), vec![70_000], false)
                    .unwrap();
                if ip {
                    s.tensor_addcdiv_(inp, t1, t2, 0.75).unwrap();
                    s.tensor_values(inp).unwrap()
                } else {
                    let o = s.tensor_addcdiv(inp, t1, t2, 0.75).unwrap();
                    s.tensor_values(o).unwrap()
                }
            },
        },
    ]
}

/// The matmul-family in-place ops must match their out-of-place siblings bit-for-bit.
#[test]
fn matmul_family_in_place_ops_match_their_out_of_place_siblings() {
    let mut failures = Vec::new();
    for c in mat_cases() {
        let oop = {
            let mut s = FrankenTorchSession::new(ExecutionMode::Strict);
            (c.run)(&mut s, false)
        };
        let ip = {
            let mut s = FrankenTorchSession::new(ExecutionMode::Strict);
            (c.run)(&mut s, true)
        };
        // No per-element input to quote here (operands differ per op), so the values
        // themselves carry the diagnosis.
        let vals = vec![f64::NAN; oop.len().max(ip.len())];
        failures.extend(diff(c.name, &vals, &oop, &ip));
    }
    assert!(
        failures.is_empty(),
        "matmul-family in-place ops disagree with their out-of-place siblings \
         (frankentorch-5fppy). These are the same op; a re-association difference between \
         the two forms is a defect, not a tolerance question:\n{}",
        failures.join("\n")
    );
}

/// `addcdiv` scales BEFORE dividing, like torch — pinned against torch's values, not against
/// FrankenTorch's own other form.
///
/// This is the test that actually decides the question. The in-place/out-of-place sweep above
/// found that the two forms disagreed, but a disagreement says only that one of them is wrong,
/// not which. torch is the arbiter:
///
/// ```text
/// torch.addcdiv(tensor([0.5]), tensor([2.25]), tensor([-1.75]), value=0.75)
///   -> -0.4642857142857143   (0xbfddb6db6db6db6e)
/// input + (value*t1)/t2      -> 0xbfddb6db6db6db6e   MATCHES
/// input + value*(t1/t2)      -> 0xbfddb6db6db6db70   differs by 2 ULP
/// ```
///
/// Measured over 400 random f64 cases on torch 2.12.1+cpu, `(value*t1)/t2` matched **400/400**
/// and `value*(t1/t2)` only **329/400**. FrankenTorch's out-of-place form used the latter in
/// all three of its paths — f64 fast, f32 fast, and the composed fallback — and they were
/// verified bit-exact *against each other*, which is why it survived: a fused==compose lock
/// test passes while both arms are wrong versus real torch. frankentorch-5fppy.
#[test]
fn addcdiv_scales_before_dividing_like_torch() {
    let mut s = FrankenTorchSession::new(ExecutionMode::Strict);
    let inp = s.tensor_variable(vec![0.5], vec![1], false).unwrap();
    let t1 = s.tensor_variable(vec![2.25], vec![1], false).unwrap();
    let t2 = s.tensor_variable(vec![-1.75], vec![1], false).unwrap();
    let out = s.tensor_addcdiv(inp, t1, t2, 0.75).unwrap();
    let got = s.tensor_values(out).unwrap()[0];

    // torch 2.12.1+cpu, bit pattern captured from the oracle.
    let want = f64::from_bits(0xbfdd_b6db_6db6_db6e);
    assert_eq!(
        got.to_bits(),
        want.to_bits(),
        "addcdiv must equal torch's input + (value*t1)/t2 = {want:?} (0x{:016x}), got {got:?} \
         (0x{:016x}). The other association, input + value*(t1/t2), gives \
         0xbfddb6db6db6db70 and is what this op used to compute.",
        want.to_bits(),
        got.to_bits()
    );

    // And the wrong association must be genuinely distinguishable here, or the test is vacuous.
    let wrong = 0.5_f64 + 0.75 * (2.25 / -1.75);
    assert_ne!(
        wrong.to_bits(),
        want.to_bits(),
        "this fixture must separate the two associations"
    );
}

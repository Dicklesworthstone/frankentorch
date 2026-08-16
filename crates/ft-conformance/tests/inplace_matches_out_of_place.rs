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

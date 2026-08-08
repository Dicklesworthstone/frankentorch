//! frankentorch-1jzt6. Pins the SIGN OF ZERO in gradients against the real torch oracle.
//!
//! Why this file exists, and why a unit test was not enough. frankentorch-dtyiz fixed
//! FrankenTorch canonicalizing `-0.0` gradients to `+0.0`. Before that, commits b26420e0 and
//! fbe2c3f5 had added four mutation-verified unit tests locking the OPPOSITE convention. Those
//! tests were green and wrong: they compared FrankenTorch's four gradient accumulators against
//! EACH OTHER and never against torch. Self-consistency passed while parity failed. A unit test
//! cannot distinguish "matches torch" from "matches what I believed about torch" — only the
//! oracle can, which is what this file is for.
//!
//! Everything here compares BIT PATTERNS. `-0.0 == +0.0` is true under `==`, so a value
//! comparison passes under either convention and would be exactly as blind as the unit tests it
//! is meant to backstop.

use ft_api::FrankenTorchSession;
use ft_conformance::{HarnessConfig, run_legacy_oracle_script};
use ft_core::ExecutionMode;
use serde_json::{Value, json};

/// How the gradient reaches `x`, which decides WHICH accumulator builds the slot.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Shape {
    /// `sum(x * k)` — one contribution, so the slot is EMPTY when it arrives. This is the
    /// empty-slot construction path, where the canonicalization lived.
    Single,
    /// `sum(x * k1 + x * k2)` — `x` is used twice, so the second contribution lands on an
    /// already-populated slot. Exercises the accumulate-into-existing path, which dtyiz did not
    /// change and which must therefore stay plain IEEE addition.
    Accumulated,
}

#[derive(Debug, Clone)]
struct GradCase {
    name: &'static str,
    /// Leaf values. Irrelevant to the result (d/dx of x*k is k) but they must not be zero, so a
    /// bug that returned the INPUT's sign instead of the multiplier's would be visible.
    input: Vec<f64>,
    /// First multiplier, no-grad. Its sign of zero is what the gradient should carry.
    k1: Vec<f64>,
    /// Second multiplier, used only by `Shape::Accumulated`.
    k2: Vec<f64>,
    shape: Vec<usize>,
    form: Shape,
}

fn grad_cases() -> Vec<GradCase> {
    vec![
        GradCase {
            name: "negative_zero_multiplier",
            input: vec![1.0, 2.0],
            k1: vec![-0.0, -0.0],
            k2: vec![],
            shape: vec![2],
            form: Shape::Single,
        },
        GradCase {
            name: "positive_zero_multiplier",
            input: vec![1.0, 2.0],
            k1: vec![0.0, 0.0],
            k2: vec![],
            shape: vec![2],
            form: Shape::Single,
        },
        // Per-element mixed signs in ONE tensor. A fix that canonicalized the whole buffer one
        // way or the other passes both single-sign cases above and fails this one.
        GradCase {
            name: "mixed_zero_signs_within_one_tensor",
            input: vec![1.0, 2.0, 3.0, 4.0],
            k1: vec![-0.0, 0.0, -0.0, 0.0],
            k2: vec![],
            shape: vec![4],
            form: Shape::Single,
        },
        GradCase {
            name: "rank2_mixed_zero_signs",
            input: vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            k1: vec![-0.0, 0.0, -0.0, 0.0, -0.0, 0.0],
            k2: vec![],
            shape: vec![2, 3],
            form: Shape::Single,
        },
        // -0.0 + -0.0 == -0.0 in IEEE, so the sign survives accumulation.
        GradCase {
            name: "two_negative_zero_contributions_accumulate",
            input: vec![1.0, 2.0],
            k1: vec![-0.0, -0.0],
            k2: vec![-0.0, -0.0],
            shape: vec![2],
            form: Shape::Accumulated,
        },
        // The discriminating case: -0.0 + +0.0 == +0.0 in IEEE. torch is NOT preserving
        // negative zero as a policy — it is simply not canonicalizing, and ordinary addition
        // does the rest. An implementation that "preserved -0.0" by special-casing it would
        // get this one wrong in the other direction.
        GradCase {
            name: "negative_plus_positive_zero_is_positive",
            input: vec![1.0, 2.0],
            k1: vec![-0.0, -0.0],
            k2: vec![0.0, 0.0],
            shape: vec![2],
            form: Shape::Accumulated,
        },
    ]
}

/// Zeros must cross the JSON boundary as tags: `serde_json` and Python's `json` both render
/// `-0.0` in ways that can round-trip to `+0.0`, which would silently erase the very property
/// under test.
fn encode_scalar(value: f64) -> Value {
    if value.to_bits() == (-0.0f64).to_bits() {
        Value::String("-0.0".to_string())
    } else {
        json!(value)
    }
}

fn decode_scalar(value: &Value) -> f64 {
    match value {
        Value::String(tag) if tag == "-0.0" => -0.0,
        Value::Number(number) => number.as_f64().unwrap_or(f64::NAN),
        _ => f64::NAN,
    }
}

fn torch_available() -> bool {
    let script = r#"
import json
import torch
print(json.dumps({"ok": True}, sort_keys=True))
"#;
    run_legacy_oracle_script(
        &HarnessConfig::default(),
        script,
        &json!({"probe": "torch"}),
    )
    .is_ok()
}

fn query_torch_gradients(cases: &[GradCase]) -> Option<Value> {
    if !torch_available() {
        eprintln!("pytorch_signed_zero_gradient_conformance: torch unavailable, skipping");
        return None;
    }

    let payload = json!({
        "cases": cases
            .iter()
            .map(|case| {
                json!({
                    "name": case.name,
                    "input": case.input.iter().copied().map(encode_scalar).collect::<Vec<_>>(),
                    "k1": case.k1.iter().copied().map(encode_scalar).collect::<Vec<_>>(),
                    "k2": case.k2.iter().copied().map(encode_scalar).collect::<Vec<_>>(),
                    "shape": case.shape,
                    "accumulated": case.form == Shape::Accumulated,
                })
            })
            .collect::<Vec<_>>(),
    });

    let script = r#"
import json
import math
import sys
import torch

def decode_scalar(value):
    if isinstance(value, str) and value == "-0.0":
        return -0.0
    return float(value)

def encode_scalar(value):
    if value == 0.0 and math.copysign(1.0, value) < 0:
        return "-0.0"
    return value

req = json.loads(sys.stdin.read())
out = []
for case in req["cases"]:
    shape = case["shape"]
    x = torch.tensor(
        [decode_scalar(v) for v in case["input"]], dtype=torch.float64
    ).reshape(shape).requires_grad_(True)
    k1 = torch.tensor(
        [decode_scalar(v) for v in case["k1"]], dtype=torch.float64
    ).reshape(shape)
    product = x * k1
    if case["accumulated"]:
        k2 = torch.tensor(
            [decode_scalar(v) for v in case["k2"]], dtype=torch.float64
        ).reshape(shape)
        product = product + x * k2
    product.sum().backward()
    out.append({
        "name": case["name"],
        "grad": [encode_scalar(float(v)) for v in x.grad.flatten().tolist()],
    })

print(json.dumps({"cases": out, "torch_version": torch.__version__}, sort_keys=True))
"#;

    Some(
        run_legacy_oracle_script(&HarnessConfig::default(), script, &payload)
            .expect("torch signed-zero gradient oracle must run after availability check"),
    )
}

fn run_frankentorch(case: &GradCase) -> Vec<f64> {
    let mut session = FrankenTorchSession::new(ExecutionMode::Strict);
    let x = session
        .tensor_variable(case.input.clone(), case.shape.clone(), true)
        .expect("leaf must be constructible");
    let k1 = session
        .tensor_variable(case.k1.clone(), case.shape.clone(), false)
        .expect("first multiplier must be constructible");
    let mut product = session.tensor_mul(x, k1).expect("x * k1");
    if case.form == Shape::Accumulated {
        let k2 = session
            .tensor_variable(case.k2.clone(), case.shape.clone(), false)
            .expect("second multiplier must be constructible");
        let second = session.tensor_mul(x, k2).expect("x * k2");
        product = session.tensor_add(product, second).expect("x*k1 + x*k2");
    }
    let root = session.tensor_sum(product).expect("sum");
    let report = session.tensor_backward(root).expect("backward");
    session
        .tensor_gradient(&report, x)
        .expect("leaf must have a gradient")
        .to_vec()
}

/// Bit-for-bit. Using `==` here would pass under either convention and defeat the whole file.
fn assert_bits_eq(case_name: &str, actual: &[f64], expected: &[f64]) {
    assert_eq!(
        actual.len(),
        expected.len(),
        "{case_name}: gradient length differs from torch"
    );
    for (index, (got, want)) in actual.iter().zip(expected.iter()).enumerate() {
        assert_eq!(
            got.to_bits(),
            want.to_bits(),
            "{case_name}: gradient[{index}] bits differ — FrankenTorch {got:?} \
             (bits {:#018x}), torch {want:?} (bits {:#018x}). \
             -0.0 and +0.0 compare EQUAL, so this is a signed-zero divergence \
             (frankentorch-dtyiz / frankentorch-1jzt6).",
            got.to_bits(),
            want.to_bits()
        );
    }
}

#[test]
fn pytorch_signed_zero_gradient_subprocess_conformance() {
    let cases = grad_cases();
    let Some(response) = query_torch_gradients(&cases) else {
        return;
    };

    // The torch version IS part of the measurement (see the repo ledger), so surface it rather
    // than letting a silent version drift rewrite the golden underneath this test.
    let torch_version = response
        .get("torch_version")
        .and_then(Value::as_str)
        .unwrap_or("unknown");
    eprintln!("pytorch_signed_zero_gradient_conformance: oracle torch {torch_version}");

    let oracle_cases = response
        .get("cases")
        .and_then(Value::as_array)
        .expect("torch response must include cases");
    assert_eq!(
        oracle_cases.len(),
        cases.len(),
        "torch must answer every case"
    );

    for (case, oracle) in cases.iter().zip(oracle_cases) {
        assert_eq!(
            oracle.get("name").and_then(Value::as_str),
            Some(case.name),
            "oracle cases must stay aligned with local cases"
        );
        let torch_grad = oracle
            .get("grad")
            .and_then(Value::as_array)
            .expect("grad")
            .iter()
            .map(decode_scalar)
            .collect::<Vec<_>>();
        let ft_grad = run_frankentorch(case);
        assert_bits_eq(case.name, &ft_grad, &torch_grad);
    }

    // Guard the guard. If the JSON boundary ever flattened -0.0 to +0.0, every assertion above
    // would compare +0.0 against +0.0 and pass while testing nothing — the same "green but
    // measuring nothing" failure that frankentorch-imtpq is about.
    let saw_negative_zero = oracle_cases.iter().any(|oracle| {
        oracle
            .get("grad")
            .and_then(Value::as_array)
            .is_some_and(|values| {
                values
                    .iter()
                    .any(|value| decode_scalar(value).to_bits() == (-0.0f64).to_bits())
            })
    });
    assert!(
        saw_negative_zero,
        "no -0.0 survived the oracle round-trip, so this test proved nothing about signed zeros"
    );
}

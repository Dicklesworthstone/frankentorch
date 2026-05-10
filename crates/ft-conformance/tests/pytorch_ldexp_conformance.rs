use ft_api::FrankenTorchSession;
use ft_conformance::{HarnessConfig, run_legacy_oracle_script};
use ft_core::ExecutionMode;
use serde_json::{Value, json};

#[derive(Debug, Clone)]
struct LdexpCase {
    name: &'static str,
    input: Vec<f64>,
    other: Vec<i64>,
    shape: Vec<usize>,
}

fn ldexp_cases() -> Vec<LdexpCase> {
    vec![
        LdexpCase {
            name: "moderate_integer_exponents",
            input: vec![-3.5, 0.0, 1.25, 7.0, -2.0],
            other: vec![0, 1, -2, 3, -4],
            shape: vec![5],
        },
        LdexpCase {
            name: "signed_zero_preservation",
            input: vec![-0.0, 0.0, -0.0, 0.0],
            other: vec![5, 5, -5, -5],
            shape: vec![4],
        },
        LdexpCase {
            name: "ieee_specials",
            input: vec![f64::NAN, f64::INFINITY, f64::NEG_INFINITY, 1.0, -1.0],
            other: vec![2, -3, 4, 1024, 1024],
            shape: vec![5],
        },
        LdexpCase {
            name: "rank2_boundary_values",
            input: vec![1.0, -1.0, 0.5, -0.5, 1024.0, -1024.0],
            other: vec![10, 10, -10, -10, 8, -8],
            shape: vec![2, 3],
        },
    ]
}

fn encode_scalar(value: f64) -> Value {
    if value.is_nan() {
        Value::String("nan".to_string())
    } else if value == f64::INFINITY {
        Value::String("inf".to_string())
    } else if value == f64::NEG_INFINITY {
        Value::String("-inf".to_string())
    } else if value.to_bits() == (-0.0f64).to_bits() {
        Value::String("-0.0".to_string())
    } else {
        json!(value)
    }
}

fn decode_scalar(value: &Value) -> f64 {
    match value {
        Value::String(tag) if tag == "nan" => f64::NAN,
        Value::String(tag) if tag == "inf" => f64::INFINITY,
        Value::String(tag) if tag == "-inf" => f64::NEG_INFINITY,
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

fn query_torch_ldexp(cases: &[LdexpCase]) -> Option<Value> {
    if !torch_available() {
        eprintln!("pytorch_ldexp_subprocess_conformance: torch unavailable, skipping");
        return None;
    }

    let payload = json!({
        "cases": cases
            .iter()
            .map(|case| {
                json!({
                    "name": case.name,
                    "input": case.input.iter().copied().map(encode_scalar).collect::<Vec<_>>(),
                    "other": case.other,
                    "shape": case.shape,
                })
            })
            .collect::<Vec<_>>(),
        "mismatch": {
            "input": [1.0, 2.0],
            "other": [1, 2, 3],
        }
    });

    let script = r#"
import json
import math
import sys
import torch

def decode_scalar(value):
    if isinstance(value, str):
        if value == "nan":
            return float("nan")
        if value == "inf":
            return float("inf")
        if value == "-inf":
            return float("-inf")
        if value == "-0.0":
            return -0.0
    return float(value)

def encode_scalar(value):
    if math.isnan(value):
        return "nan"
    if math.isinf(value):
        return "inf" if value > 0 else "-inf"
    if value == 0.0 and math.copysign(1.0, value) < 0:
        return "-0.0"
    return value

req = json.loads(sys.stdin.read())
out = []
for case in req["cases"]:
    input_tensor = torch.tensor(
        [decode_scalar(value) for value in case["input"]],
        dtype=torch.float64,
    ).reshape(case["shape"])
    other_tensor = torch.tensor(case["other"], dtype=torch.int64).reshape(case["shape"])
    result = torch.ldexp(input_tensor, other_tensor)
    out.append({
        "name": case["name"],
        "shape": list(result.shape),
        "values": [encode_scalar(float(value)) for value in result.flatten().tolist()],
    })

mismatch_error = False
try:
    torch.ldexp(
        torch.tensor(req["mismatch"]["input"], dtype=torch.float64),
        torch.tensor(req["mismatch"]["other"], dtype=torch.int64),
    )
except RuntimeError:
    mismatch_error = True

print(json.dumps({"cases": out, "mismatch_error": mismatch_error}, sort_keys=True))
"#;

    Some(
        run_legacy_oracle_script(&HarnessConfig::default(), script, &payload)
            .expect("torch ldexp oracle must run after availability check"),
    )
}

fn run_frankentorch(case: &LdexpCase) -> (Vec<usize>, Vec<f64>) {
    let mut session = FrankenTorchSession::new(ExecutionMode::Strict);
    let input = session
        .tensor_variable(case.input.clone(), case.shape.clone(), false)
        .expect("input tensor must be constructible");
    let other_values = case
        .other
        .iter()
        .map(|value| *value as f64)
        .collect::<Vec<_>>();
    let other = session
        .tensor_variable(other_values, case.shape.clone(), false)
        .expect("other tensor must be constructible");
    let output = session
        .tensor_ldexp(input, other)
        .expect("same-shape ldexp must run");
    let shape = session.tensor_shape(output).expect("output shape");
    let values = session.tensor_values(output).expect("output values");
    (shape, values)
}

fn assert_close(case_name: &str, actual: &[f64], expected: &[f64]) {
    assert_eq!(
        actual.len(),
        expected.len(),
        "{case_name}: output value length differs"
    );
    for (index, (got, want)) in actual.iter().zip(expected.iter()).enumerate() {
        if got.is_nan() || want.is_nan() {
            assert!(
                got.is_nan() && want.is_nan(),
                "{case_name}: value[{index}] got {got:?}, want {want:?}"
            );
            continue;
        }
        let got_is_zero = got.to_bits() == 0 || got.to_bits() == (-0.0f64).to_bits();
        let want_is_zero = want.to_bits() == 0 || want.to_bits() == (-0.0f64).to_bits();
        if got_is_zero || want_is_zero {
            assert_eq!(
                got.to_bits(),
                want.to_bits(),
                "{case_name}: value[{index}] signed zero mismatch"
            );
            continue;
        }
        if got.is_infinite() || want.is_infinite() {
            assert_eq!(
                got, want,
                "{case_name}: value[{index}] got {got:?}, want {want:?}"
            );
            continue;
        }
        let diff = (got - want).abs();
        let scale = got.abs().max(want.abs()).max(1.0);
        assert!(
            diff <= 1e-12 * scale,
            "{case_name}: value[{index}] got {got:?}, want {want:?}, diff {diff:e}"
        );
    }
}

#[test]
fn pytorch_ldexp_subprocess_conformance() {
    let cases = ldexp_cases();
    let Some(response) = query_torch_ldexp(&cases) else {
        return;
    };

    assert_eq!(
        response.get("mismatch_error").and_then(Value::as_bool),
        Some(true),
        "torch.ldexp must reject non-broadcast mismatched shapes"
    );
    let mut session = FrankenTorchSession::new(ExecutionMode::Strict);
    let input = session
        .tensor_variable(vec![1.0, 2.0], vec![2], false)
        .expect("mismatch input");
    let other = session
        .tensor_variable(vec![1.0, 2.0, 3.0], vec![3], false)
        .expect("mismatch other");
    assert!(
        session.tensor_ldexp(input, other).is_err(),
        "FrankenTorch tensor_ldexp must reject the same mismatch"
    );

    let oracle_cases = response
        .get("cases")
        .and_then(Value::as_array)
        .expect("torch response must include cases");
    assert_eq!(oracle_cases.len(), cases.len());

    for (case, oracle) in cases.iter().zip(oracle_cases) {
        let (ft_shape, ft_values) = run_frankentorch(case);
        let torch_name = oracle.get("name").and_then(Value::as_str).unwrap_or("");
        assert_eq!(torch_name, case.name);
        let torch_shape = oracle
            .get("shape")
            .and_then(Value::as_array)
            .expect("shape")
            .iter()
            .map(|value| value.as_u64().unwrap_or(u64::MAX) as usize)
            .collect::<Vec<_>>();
        let torch_values = oracle
            .get("values")
            .and_then(Value::as_array)
            .expect("values")
            .iter()
            .map(decode_scalar)
            .collect::<Vec<_>>();
        assert_eq!(
            ft_shape, torch_shape,
            "{}: shape mismatch against torch",
            case.name
        );
        assert_close(case.name, &ft_values, &torch_values);
    }
}

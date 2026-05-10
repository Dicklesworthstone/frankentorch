use ft_api::FrankenTorchSession;
use ft_conformance::{HarnessConfig, run_legacy_oracle_script};
use ft_core::ExecutionMode;
use serde_json::{Value, json};

#[derive(Debug, Clone)]
struct AvgPool2dCase {
    name: &'static str,
    values: Vec<f64>,
    shape: Vec<usize>,
    kernel: (usize, usize),
    stride: (usize, usize),
    padding: (usize, usize),
    ceil_mode: bool,
    count_include_pad: bool,
}

fn conformance_cases() -> Vec<AvgPool2dCase> {
    vec![
        AvgPool2dCase {
            name: "baseline_no_padding",
            values: vec![1.0, 2.0, 3.0, 4.0],
            shape: vec![1, 1, 2, 2],
            kernel: (2, 2),
            stride: (2, 2),
            padding: (0, 0),
            ceil_mode: false,
            count_include_pad: true,
        },
        AvgPool2dCase {
            name: "padding_excludes_pad_from_divisor",
            values: vec![1.0, 2.0, 3.0, 4.0],
            shape: vec![1, 1, 2, 2],
            kernel: (3, 3),
            stride: (1, 1),
            padding: (1, 1),
            ceil_mode: false,
            count_include_pad: false,
        },
        AvgPool2dCase {
            name: "padding_includes_pad_in_divisor",
            values: vec![1.0, 2.0, 3.0, 4.0],
            shape: vec![1, 1, 2, 2],
            kernel: (3, 3),
            stride: (1, 1),
            padding: (1, 1),
            ceil_mode: false,
            count_include_pad: true,
        },
        AvgPool2dCase {
            name: "ceil_mode_partial_windows_exclude_pad",
            values: (1..=9).map(f64::from).collect(),
            shape: vec![1, 1, 3, 3],
            kernel: (2, 2),
            stride: (2, 2),
            padding: (0, 0),
            ceil_mode: true,
            count_include_pad: false,
        },
        AvgPool2dCase {
            name: "batched_channels_stride_padding",
            values: (1..=48).map(|value| f64::from(value) / 10.0).collect(),
            shape: vec![2, 2, 3, 4],
            kernel: (2, 3),
            stride: (2, 2),
            padding: (1, 1),
            ceil_mode: true,
            count_include_pad: false,
        },
    ]
}

fn run_frankentorch(case: &AvgPool2dCase) -> (Vec<usize>, Vec<f64>) {
    let mut session = FrankenTorchSession::new(ExecutionMode::Strict);
    let input = session
        .tensor_variable(case.values.clone(), case.shape.clone(), false)
        .expect("input tensor must be constructible");
    let output = session
        .functional_avg_pool2d(
            input,
            case.kernel,
            case.stride,
            case.padding,
            case.ceil_mode,
            case.count_include_pad,
        )
        .expect("avg_pool2d must run");
    let shape = session.tensor_shape(output).expect("output shape");
    let values = session.tensor_values(output).expect("output values");
    (shape, values)
}

fn assert_close(case: &str, got: &[f64], want: &[f64]) {
    assert_eq!(
        got.len(),
        want.len(),
        "{case}: output value length differs: got {} want {}",
        got.len(),
        want.len()
    );
    for (index, (actual, expected)) in got.iter().zip(want).enumerate() {
        let diff = (actual - expected).abs();
        assert!(
            diff <= 1e-12,
            "{case}: value[{index}] got {actual:?}, want {expected:?}, diff {diff:e}"
        );
    }
}

#[test]
fn avg_pool2d_padding_divisor_matches_pytorch_known_fixtures() {
    let cases = conformance_cases();

    let exclude_pad = cases
        .iter()
        .find(|case| case.name == "padding_excludes_pad_from_divisor")
        .expect("exclude-pad fixture");
    let (shape, values) = run_frankentorch(exclude_pad);
    assert_eq!(shape, vec![1, 1, 2, 2]);
    assert_close(exclude_pad.name, &values, &[2.5, 2.5, 2.5, 2.5]);

    let include_pad = cases
        .iter()
        .find(|case| case.name == "padding_includes_pad_in_divisor")
        .expect("include-pad fixture");
    let (shape, values) = run_frankentorch(include_pad);
    assert_eq!(shape, vec![1, 1, 2, 2]);
    assert_close(
        include_pad.name,
        &values,
        &[10.0 / 9.0, 10.0 / 9.0, 10.0 / 9.0, 10.0 / 9.0],
    );

    let ceil_mode = cases
        .iter()
        .find(|case| case.name == "ceil_mode_partial_windows_exclude_pad")
        .expect("ceil-mode fixture");
    let (shape, values) = run_frankentorch(ceil_mode);
    assert_eq!(shape, vec![1, 1, 2, 2]);
    assert_close(ceil_mode.name, &values, &[3.0, 4.5, 7.5, 9.0]);
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

fn query_torch_avg_pool2d(cases: &[AvgPool2dCase]) -> Option<Value> {
    if !torch_available() {
        eprintln!("pytorch_avg_pool2d_subprocess_conformance: torch unavailable, skipping");
        return None;
    }

    let payload_cases = cases
        .iter()
        .map(|case| {
            json!({
                "name": case.name,
                "values": case.values.clone(),
                "shape": case.shape.clone(),
                "kernel": [case.kernel.0, case.kernel.1],
                "stride": [case.stride.0, case.stride.1],
                "padding": [case.padding.0, case.padding.1],
                "ceil_mode": case.ceil_mode,
                "count_include_pad": case.count_include_pad,
            })
        })
        .collect::<Vec<_>>();
    let payload = json!({ "cases": payload_cases });

    let script = r#"
import json
import sys
import torch
import torch.nn.functional as F

req = json.loads(sys.stdin.read())
out = []
for case in req["cases"]:
    tensor = torch.tensor(case["values"], dtype=torch.float64).reshape(case["shape"])
    result = F.avg_pool2d(
        tensor,
        kernel_size=tuple(case["kernel"]),
        stride=tuple(case["stride"]),
        padding=tuple(case["padding"]),
        ceil_mode=case["ceil_mode"],
        count_include_pad=case["count_include_pad"],
    )
    out.append({
        "name": case["name"],
        "shape": list(result.shape),
        "values": [float(v) for v in result.flatten().tolist()],
    })
print(json.dumps({"cases": out}))
"#;

    Some(
        run_legacy_oracle_script(&HarnessConfig::default(), script, &payload)
            .expect("torch avg_pool2d oracle must run after availability check"),
    )
}

#[test]
fn pytorch_avg_pool2d_subprocess_conformance() {
    let cases = conformance_cases();
    let Some(response) = query_torch_avg_pool2d(&cases) else {
        return;
    };
    let oracle_cases = response
        .get("cases")
        .and_then(Value::as_array)
        .expect("torch response must include cases");
    assert_eq!(oracle_cases.len(), cases.len());

    for (case, oracle) in cases.iter().zip(oracle_cases) {
        let (ft_shape, ft_values) = run_frankentorch(case);
        let torch_name = oracle.get("name").and_then(Value::as_str).expect("name");
        assert_eq!(torch_name, case.name);
        let torch_shape = oracle
            .get("shape")
            .and_then(Value::as_array)
            .expect("shape")
            .iter()
            .map(|value| {
                usize::try_from(value.as_u64().expect("shape dimension must be unsigned"))
                    .expect("shape dimension must fit usize")
            })
            .collect::<Vec<_>>();
        let torch_values = oracle
            .get("values")
            .and_then(Value::as_array)
            .expect("values")
            .iter()
            .map(|value| value.as_f64().expect("value must be f64"))
            .collect::<Vec<_>>();
        assert_eq!(
            ft_shape, torch_shape,
            "{}: shape mismatch against torch",
            case.name
        );
        assert_close(case.name, &ft_values, &torch_values);
    }
}

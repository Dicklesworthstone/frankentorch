use std::path::Path;

use ft_api::FrankenTorchSession;
use ft_conformance::{
    HarnessConfig, run_autograd_scheduler_conformance, run_dispatch_conformance,
    run_optimizer_conformance, run_scalar_conformance, run_serialization_conformance, run_smoke,
    run_tensor_meta_conformance,
};
use ft_core::{DType, DenseTensor, Device, ExecutionMode, TensorMeta};
use ft_runtime::{EvidenceKind, RuntimeContext};
use ft_serialize::{DecodeMode, decode_checkpoint};

#[test]
fn smoke_report_is_stable() {
    let cfg = HarnessConfig::default_paths();
    let report = run_smoke(&cfg);
    assert_eq!(report.suite, "smoke");
    assert!(report.fixture_count >= 1);
    assert_eq!(report.oracle_present, cfg.oracle_root.exists());

    let fixture_path = cfg.fixture_root.join("smoke_case.json");
    assert!(Path::new(&fixture_path).exists());
}

#[test]
fn scalar_fixture_executes_in_strict_mode() {
    let cfg = HarnessConfig::default_paths();
    let (report, cases) =
        run_scalar_conformance(&cfg, ExecutionMode::Strict).expect("scalar conformance should run");

    assert_eq!(report.cases_total, cases.len());
    assert_eq!(report.cases_total, report.cases_passed);
}

#[test]
fn dispatch_fixture_executes_in_both_modes() {
    let cfg = HarnessConfig::default_paths();
    let (strict_report, _) =
        run_dispatch_conformance(&cfg, ExecutionMode::Strict).expect("strict dispatch should run");
    let (hardened_report, _) = run_dispatch_conformance(&cfg, ExecutionMode::Hardened)
        .expect("hardened dispatch should run");

    assert_eq!(strict_report.cases_total, strict_report.cases_passed);
    assert_eq!(hardened_report.cases_total, hardened_report.cases_passed);
}

#[test]
fn tensor_meta_fixture_executes_in_both_modes() {
    let cfg = HarnessConfig::default_paths();
    let (strict_report, _) = run_tensor_meta_conformance(&cfg, ExecutionMode::Strict)
        .expect("strict tensor-meta should run");
    let (hardened_report, _) = run_tensor_meta_conformance(&cfg, ExecutionMode::Hardened)
        .expect("hardened tensor-meta should run");

    assert_eq!(strict_report.cases_total, strict_report.cases_passed);
    assert_eq!(hardened_report.cases_total, hardened_report.cases_passed);
}

#[test]
fn scheduler_fixture_executes_in_both_modes() {
    let cfg = HarnessConfig::default_paths();
    let (strict_report, _) = run_autograd_scheduler_conformance(&cfg, ExecutionMode::Strict)
        .expect("strict scheduler should run");
    let (hardened_report, _) = run_autograd_scheduler_conformance(&cfg, ExecutionMode::Hardened)
        .expect("hardened scheduler should run");

    assert_eq!(strict_report.cases_total, strict_report.cases_passed);
    assert_eq!(hardened_report.cases_total, hardened_report.cases_passed);
}

#[test]
fn serialization_fixture_executes_in_both_modes() {
    let cfg = HarnessConfig::default_paths();
    let (strict_report, _) = run_serialization_conformance(&cfg, ExecutionMode::Strict)
        .expect("strict serialization should run");
    let (hardened_report, _) = run_serialization_conformance(&cfg, ExecutionMode::Hardened)
        .expect("hardened serialization should run");

    assert_eq!(strict_report.cases_total, strict_report.cases_passed);
    assert_eq!(hardened_report.cases_total, hardened_report.cases_passed);
}

#[test]
fn optimizer_fixture_executes_in_both_modes() {
    let cfg = HarnessConfig::default_paths();
    let (strict_report, _) = run_optimizer_conformance(&cfg, ExecutionMode::Strict)
        .expect("strict optimizer_state should run");
    let (hardened_report, _) = run_optimizer_conformance(&cfg, ExecutionMode::Hardened)
        .expect("hardened optimizer_state should run");

    assert_eq!(strict_report.cases_total, strict_report.cases_passed);
    assert_eq!(hardened_report.cases_total, hardened_report.cases_passed);
}

#[test]
fn tensor_session_path_executes_in_strict_mode() {
    let mut session = FrankenTorchSession::new(ExecutionMode::Strict);
    let x = session
        .tensor_variable(vec![1.0, 2.0, 3.0], vec![3], true)
        .expect("lhs tensor variable should succeed");
    let y = session
        .tensor_variable(vec![4.0, 5.0, 6.0], vec![3], true)
        .expect("rhs tensor variable should succeed");
    let z = session.tensor_add(x, y).expect("tensor add should succeed");
    assert_eq!(
        session
            .tensor_values(z)
            .expect("tensor values should resolve"),
        vec![5.0, 7.0, 9.0]
    );

    let report = session
        .tensor_backward(z)
        .expect("tensor backward should succeed");
    assert_eq!(
        session
            .tensor_gradient(&report, x)
            .expect("x gradient should exist"),
        &[1.0, 1.0, 1.0]
    );
    assert_eq!(
        session
            .tensor_gradient(&report, y)
            .expect("y gradient should exist"),
        &[1.0, 1.0, 1.0]
    );
}

#[test]
fn tensor_session_mul_path_executes_in_strict_mode() {
    let mut session = FrankenTorchSession::new(ExecutionMode::Strict);
    let x = session
        .tensor_variable(vec![2.0, 4.0], vec![2], true)
        .expect("lhs tensor variable should succeed");
    let y = session
        .tensor_variable(vec![3.0, 2.0], vec![2], true)
        .expect("rhs tensor variable should succeed");
    let z = session.tensor_mul(x, y).expect("tensor mul should succeed");
    assert_eq!(
        session
            .tensor_values(z)
            .expect("tensor values should resolve"),
        vec![6.0, 8.0]
    );

    let report = session
        .tensor_backward(z)
        .expect("tensor backward should succeed");
    assert_eq!(
        session
            .tensor_gradient(&report, x)
            .expect("x gradient should exist"),
        &[3.0, 2.0]
    );
    assert_eq!(
        session
            .tensor_gradient(&report, y)
            .expect("y gradient should exist"),
        &[2.0, 4.0]
    );
}

#[test]
fn tensor_session_div_path_executes_in_strict_mode() {
    let mut session = FrankenTorchSession::new(ExecutionMode::Strict);
    let x = session
        .tensor_variable(vec![2.0, 4.0], vec![2], true)
        .expect("lhs tensor variable should succeed");
    let y = session
        .tensor_variable(vec![3.0, 2.0], vec![2], true)
        .expect("rhs tensor variable should succeed");
    let z = session.tensor_div(x, y).expect("tensor div should succeed");
    let values = session
        .tensor_values(z)
        .expect("tensor values should resolve");
    assert!((values[0] - (2.0 / 3.0)).abs() <= 1e-12);
    assert!((values[1] - 2.0).abs() <= 1e-12);

    let report = session
        .tensor_backward(z)
        .expect("tensor backward should succeed");
    let x_grad = session
        .tensor_gradient(&report, x)
        .expect("x gradient should exist");
    let y_grad = session
        .tensor_gradient(&report, y)
        .expect("y gradient should exist");
    let expected_x_grad = [1.0 / 3.0, 0.5];
    let expected_y_grad = [-2.0 / 9.0, -1.0];
    for (actual, expected) in x_grad.iter().zip(expected_x_grad) {
        assert!((actual - expected).abs() <= 1e-12);
    }
    for (actual, expected) in y_grad.iter().zip(expected_y_grad) {
        assert!((actual - expected).abs() <= 1e-12);
    }
}

#[test]
fn tensor_session_fails_closed_on_non_contiguous_input() {
    let mut session = FrankenTorchSession::new(ExecutionMode::Strict);
    let lhs_meta =
        TensorMeta::from_shape_and_strides(vec![2, 2], vec![4, 1], 0, DType::F64, Device::Cpu)
            .expect("non-contiguous meta should validate");
    let lhs = DenseTensor::from_storage(lhs_meta, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
        .expect("lhs tensor should build");
    let rhs = session
        .tensor_variable(vec![5.0, 6.0, 7.0, 8.0], vec![2, 2], true)
        .expect("rhs tensor variable should build");
    let lhs = session.tensor_variable_from_storage(lhs, true);

    let err = session
        .tensor_add(lhs, rhs)
        .expect_err("non-contiguous tensor input must fail closed");
    assert!(
        err.to_string()
            .contains("unsupported non-contiguous layout on lhs")
    );
}

#[test]
fn tensor_session_fails_closed_on_device_mismatch() {
    let mut session = FrankenTorchSession::new(ExecutionMode::Strict);
    let lhs_meta =
        TensorMeta::from_shape_and_strides(vec![2], vec![1], 0, DType::F64, Device::Cuda)
            .expect("cuda meta should validate");
    let lhs = DenseTensor::from_storage(lhs_meta, vec![1.0, 2.0]).expect("lhs tensor should build");
    let rhs = session
        .tensor_variable(vec![3.0, 4.0], vec![2], true)
        .expect("rhs tensor variable should build");
    let lhs = session.tensor_variable_from_storage(lhs, true);

    let err = session
        .tensor_add(lhs, rhs)
        .expect_err("device-mismatched tensor input must fail closed");
    let message = err.to_string();
    assert!(
        message.contains("device mismatch"),
        "unexpected error: {message}"
    );
    assert!(
        message.contains("lhs=Cuda, rhs=Cpu"),
        "missing device mismatch payload: {message}"
    );
}

#[test]
fn runtime_records_durability_evidence_for_decode_failure() {
    let mut ctx = RuntimeContext::new(ExecutionMode::Strict);
    let payload = r#"{
        "schema_version": 1,
        "mode": "strict",
        "entries": [],
        "source_hash": "det64:placeholder",
        "extra": 1
    }"#;

    let err = decode_checkpoint(payload, DecodeMode::Strict)
        .expect_err("unknown field payload must fail strict decode");
    ctx.record_checkpoint_decode_failure("strict", &err);

    let durability_entry = ctx
        .ledger()
        .entries()
        .iter()
        .rev()
        .find(|entry| entry.kind == EvidenceKind::Durability)
        .expect("durability evidence entry should be present");
    assert!(
        durability_entry
            .summary
            .contains("checkpoint decode failure"),
        "unexpected durability summary: {}",
        durability_entry.summary
    );
    assert!(
        durability_entry.summary.contains("unknown field"),
        "durability summary should include decode diagnostic: {}",
        durability_entry.summary
    );
}

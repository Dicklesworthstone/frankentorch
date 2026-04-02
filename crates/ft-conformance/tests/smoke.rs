use std::path::Path;

use ft_api::FrankenTorchSession;
use ft_conformance::{
    HarnessConfig, run_autograd_scheduler_conformance, run_dispatch_conformance,
    run_optimizer_conformance, run_scalar_conformance, run_serialization_conformance, run_smoke,
    run_tensor_advanced_conformance, run_tensor_comparison_conformance,
    run_tensor_einsum_conformance, run_tensor_elementwise_cmp_conformance,
    run_tensor_factory_conformance, run_tensor_indexing_conformance, run_tensor_init_conformance,
    run_tensor_inplace_conformance, run_tensor_join_conformance, run_tensor_linalg_conformance,
    run_tensor_loss_conformance, run_tensor_meta_conformance, run_tensor_normalize_conformance,
    run_tensor_reduction_conformance, run_tensor_scan_conformance,
    run_tensor_searchsorted_conformance, run_tensor_shape_conformance, run_tensor_sort_conformance,
    run_tensor_unary_conformance,
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
fn tensor_unary_fixture_executes_in_both_modes() {
    let cfg = HarnessConfig::default_paths();
    let (strict_report, strict_cases) = run_tensor_unary_conformance(&cfg, ExecutionMode::Strict)
        .expect("strict tensor-unary should run");
    let (hardened_report, _) = run_tensor_unary_conformance(&cfg, ExecutionMode::Hardened)
        .expect("hardened tensor-unary should run");

    assert!(
        strict_report.cases_total >= 29,
        "expected at least 29 unary cases"
    );
    assert_eq!(strict_report.cases_total, strict_cases.len());
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

#[test]
fn tensor_comparison_fixture_executes_in_both_modes() {
    let cfg = HarnessConfig::default_paths();
    let (strict_report, strict_cases) =
        run_tensor_comparison_conformance(&cfg, ExecutionMode::Strict)
            .expect("strict tensor-comparison should run");
    let (hardened_report, _) = run_tensor_comparison_conformance(&cfg, ExecutionMode::Hardened)
        .expect("hardened tensor-comparison should run");

    assert!(
        strict_report.cases_total >= 6,
        "expected at least 6 comparison cases"
    );
    assert_eq!(strict_report.cases_total, strict_cases.len());
    assert_eq!(strict_report.cases_total, strict_report.cases_passed);
    assert_eq!(hardened_report.cases_total, hardened_report.cases_passed);
}

#[test]
fn tensor_factory_fixture_executes_in_both_modes() {
    let cfg = HarnessConfig::default_paths();
    let (strict_report, strict_cases) = run_tensor_factory_conformance(&cfg, ExecutionMode::Strict)
        .expect("strict tensor-factory should run");
    let (hardened_report, _) = run_tensor_factory_conformance(&cfg, ExecutionMode::Hardened)
        .expect("hardened tensor-factory should run");

    assert!(
        strict_report.cases_total >= 12,
        "expected at least 12 factory cases"
    );
    assert_eq!(strict_report.cases_total, strict_cases.len());
    assert_eq!(strict_report.cases_total, strict_report.cases_passed);
    assert_eq!(hardened_report.cases_total, hardened_report.cases_passed);
}

#[test]
fn tensor_init_fixture_executes_in_both_modes() {
    let cfg = HarnessConfig::default_paths();
    let (strict_report, strict_cases) = run_tensor_init_conformance(&cfg, ExecutionMode::Strict)
        .expect("strict tensor-init should run");
    let (hardened_report, _) = run_tensor_init_conformance(&cfg, ExecutionMode::Hardened)
        .expect("hardened tensor-init should run");

    assert!(
        strict_report.cases_total >= 12,
        "expected at least 12 init cases"
    );
    assert_eq!(strict_report.cases_total, strict_cases.len());
    assert_eq!(strict_report.cases_total, strict_report.cases_passed);
    assert_eq!(hardened_report.cases_total, hardened_report.cases_passed);
}

#[test]
fn tensor_einsum_fixture_executes_in_both_modes() {
    let cfg = HarnessConfig::default_paths();
    let (strict_report, strict_cases) = run_tensor_einsum_conformance(&cfg, ExecutionMode::Strict)
        .expect("strict tensor-einsum should run");
    let (hardened_report, _) = run_tensor_einsum_conformance(&cfg, ExecutionMode::Hardened)
        .expect("hardened tensor-einsum should run");

    assert!(
        strict_report.cases_total >= 8,
        "expected at least 8 einsum cases"
    );
    assert_eq!(strict_report.cases_total, strict_cases.len());
    assert_eq!(strict_report.cases_total, strict_report.cases_passed);
    assert_eq!(hardened_report.cases_total, hardened_report.cases_passed);
}

#[test]
fn tensor_searchsorted_fixture_executes_in_both_modes() {
    let cfg = HarnessConfig::default_paths();
    let (strict_report, strict_cases) =
        run_tensor_searchsorted_conformance(&cfg, ExecutionMode::Strict)
            .expect("strict tensor-searchsorted should run");
    let (hardened_report, _) = run_tensor_searchsorted_conformance(&cfg, ExecutionMode::Hardened)
        .expect("hardened tensor-searchsorted should run");

    assert!(
        strict_report.cases_total >= 5,
        "expected at least 5 searchsorted cases"
    );
    assert_eq!(strict_report.cases_total, strict_cases.len());
    assert_eq!(strict_report.cases_total, strict_report.cases_passed);
    assert_eq!(hardened_report.cases_total, hardened_report.cases_passed);
}

#[test]
fn tensor_reduction_fixture_executes_in_both_modes() {
    let cfg = HarnessConfig::default_paths();
    let (strict_report, strict_cases) =
        run_tensor_reduction_conformance(&cfg, ExecutionMode::Strict)
            .expect("strict tensor-reduction should run");
    let (hardened_report, _) = run_tensor_reduction_conformance(&cfg, ExecutionMode::Hardened)
        .expect("hardened tensor-reduction should run");

    assert!(
        strict_report.cases_total >= 10,
        "expected at least 10 reduction cases"
    );
    assert_eq!(strict_report.cases_total, strict_cases.len());
    assert_eq!(strict_report.cases_total, strict_report.cases_passed);
    assert_eq!(hardened_report.cases_total, hardened_report.cases_passed);
}

#[test]
fn tensor_loss_fixture_executes_in_both_modes() {
    let cfg = HarnessConfig::default_paths();
    let (strict_report, strict_cases) = run_tensor_loss_conformance(&cfg, ExecutionMode::Strict)
        .expect("strict tensor-loss should run");
    let (hardened_report, _) = run_tensor_loss_conformance(&cfg, ExecutionMode::Hardened)
        .expect("hardened tensor-loss should run");

    assert!(
        strict_report.cases_total >= 12,
        "expected at least 12 loss cases"
    );
    assert_eq!(strict_report.cases_total, strict_cases.len());
    assert_eq!(strict_report.cases_total, strict_report.cases_passed);
    assert_eq!(hardened_report.cases_total, hardened_report.cases_passed);
}

#[test]
fn tensor_linalg_fixture_executes_in_both_modes() {
    let cfg = HarnessConfig::default_paths();
    let (strict_report, strict_cases) = run_tensor_linalg_conformance(&cfg, ExecutionMode::Strict)
        .expect("strict tensor-linalg should run");
    let (hardened_report, _) = run_tensor_linalg_conformance(&cfg, ExecutionMode::Hardened)
        .expect("hardened tensor-linalg should run");

    assert!(
        strict_report.cases_total >= 9,
        "expected at least 9 linalg cases"
    );
    assert_eq!(strict_report.cases_total, strict_cases.len());
    assert_eq!(strict_report.cases_total, strict_report.cases_passed);
    assert_eq!(hardened_report.cases_total, hardened_report.cases_passed);
}

#[test]
fn tensor_normalize_fixture_executes_in_both_modes() {
    let cfg = HarnessConfig::default_paths();
    let (strict_report, strict_cases) =
        run_tensor_normalize_conformance(&cfg, ExecutionMode::Strict)
            .expect("strict tensor-normalize should run");
    let (hardened_report, _) = run_tensor_normalize_conformance(&cfg, ExecutionMode::Hardened)
        .expect("hardened tensor-normalize should run");

    assert!(
        strict_report.cases_total >= 7,
        "expected at least 7 normalize cases"
    );
    assert_eq!(strict_report.cases_total, strict_cases.len());
    assert_eq!(strict_report.cases_total, strict_report.cases_passed);
    assert_eq!(hardened_report.cases_total, hardened_report.cases_passed);
}

#[test]
fn tensor_elementwise_cmp_fixture_executes_in_both_modes() {
    let cfg = HarnessConfig::default_paths();
    let (strict_report, strict_cases) =
        run_tensor_elementwise_cmp_conformance(&cfg, ExecutionMode::Strict)
            .expect("strict tensor-elementwise-cmp should run");
    let (hardened_report, _) =
        run_tensor_elementwise_cmp_conformance(&cfg, ExecutionMode::Hardened)
            .expect("hardened tensor-elementwise-cmp should run");

    assert!(
        strict_report.cases_total >= 9,
        "expected at least 9 elementwise cmp cases"
    );
    assert_eq!(strict_report.cases_total, strict_cases.len());
    assert_eq!(strict_report.cases_total, strict_report.cases_passed);
    assert_eq!(hardened_report.cases_total, hardened_report.cases_passed);
}

#[test]
fn tensor_shape_fixture_executes_in_both_modes() {
    let cfg = HarnessConfig::default_paths();
    let (strict_report, strict_cases) = run_tensor_shape_conformance(&cfg, ExecutionMode::Strict)
        .expect("strict tensor-shape should run");
    let (hardened_report, _) = run_tensor_shape_conformance(&cfg, ExecutionMode::Hardened)
        .expect("hardened tensor-shape should run");

    assert!(
        strict_report.cases_total >= 12,
        "expected at least 12 shape cases"
    );
    assert_eq!(strict_report.cases_total, strict_cases.len());
    assert_eq!(strict_report.cases_total, strict_report.cases_passed);
    assert_eq!(hardened_report.cases_total, hardened_report.cases_passed);
}

#[test]
fn tensor_scan_fixture_executes_in_both_modes() {
    let cfg = HarnessConfig::default_paths();
    let (strict_report, strict_cases) = run_tensor_scan_conformance(&cfg, ExecutionMode::Strict)
        .expect("strict tensor-scan should run");
    let (hardened_report, _) = run_tensor_scan_conformance(&cfg, ExecutionMode::Hardened)
        .expect("hardened tensor-scan should run");

    assert!(
        strict_report.cases_total >= 8,
        "expected at least 8 scan cases"
    );
    assert_eq!(strict_report.cases_total, strict_cases.len());
    assert_eq!(strict_report.cases_total, strict_report.cases_passed);
    assert_eq!(hardened_report.cases_total, hardened_report.cases_passed);
}

#[test]
fn tensor_join_fixture_executes_in_both_modes() {
    let cfg = HarnessConfig::default_paths();
    let (strict_report, strict_cases) = run_tensor_join_conformance(&cfg, ExecutionMode::Strict)
        .expect("strict tensor-join should run");
    let (hardened_report, _) = run_tensor_join_conformance(&cfg, ExecutionMode::Hardened)
        .expect("hardened tensor-join should run");

    assert!(
        strict_report.cases_total >= 9,
        "expected at least 9 join cases"
    );
    assert_eq!(strict_report.cases_total, strict_cases.len());
    assert_eq!(strict_report.cases_total, strict_report.cases_passed);
    assert_eq!(hardened_report.cases_total, hardened_report.cases_passed);
}

#[test]
fn tensor_sort_fixture_executes_in_both_modes() {
    let cfg = HarnessConfig::default_paths();
    let (strict_report, strict_cases) = run_tensor_sort_conformance(&cfg, ExecutionMode::Strict)
        .expect("strict tensor-sort should run");
    let (hardened_report, _) = run_tensor_sort_conformance(&cfg, ExecutionMode::Hardened)
        .expect("hardened tensor-sort should run");

    assert!(
        strict_report.cases_total >= 9,
        "expected at least 9 sort cases"
    );
    assert_eq!(strict_report.cases_total, strict_cases.len());
    assert_eq!(strict_report.cases_total, strict_report.cases_passed);
    assert_eq!(hardened_report.cases_total, hardened_report.cases_passed);
}

#[test]
fn tensor_indexing_fixture_executes_in_both_modes() {
    let cfg = HarnessConfig::default_paths();
    let (strict_report, strict_cases) =
        run_tensor_indexing_conformance(&cfg, ExecutionMode::Strict)
            .expect("strict tensor-indexing should run");
    let (hardened_report, _) = run_tensor_indexing_conformance(&cfg, ExecutionMode::Hardened)
        .expect("hardened tensor-indexing should run");

    assert!(
        strict_report.cases_total >= 10,
        "expected at least 10 indexing cases"
    );
    assert_eq!(strict_report.cases_total, strict_cases.len());
    assert_eq!(strict_report.cases_total, strict_report.cases_passed);
    assert_eq!(hardened_report.cases_total, hardened_report.cases_passed);
}

#[test]
fn tensor_inplace_fixture_executes_in_both_modes() {
    let cfg = HarnessConfig::default_paths();
    let (strict_report, strict_cases) = run_tensor_inplace_conformance(&cfg, ExecutionMode::Strict)
        .expect("strict tensor-inplace should run");
    let (hardened_report, _) = run_tensor_inplace_conformance(&cfg, ExecutionMode::Hardened)
        .expect("hardened tensor-inplace should run");

    assert!(
        strict_report.cases_total >= 9,
        "expected at least 9 inplace cases"
    );
    assert_eq!(strict_report.cases_total, strict_cases.len());
    assert_eq!(strict_report.cases_total, strict_report.cases_passed);
    assert_eq!(hardened_report.cases_total, hardened_report.cases_passed);
}

#[test]
fn tensor_advanced_fixture_executes_in_both_modes() {
    let cfg = HarnessConfig::default_paths();
    let (strict_report, strict_cases) =
        run_tensor_advanced_conformance(&cfg, ExecutionMode::Strict)
            .expect("strict tensor-advanced should run");
    let (hardened_report, _) = run_tensor_advanced_conformance(&cfg, ExecutionMode::Hardened)
        .expect("hardened tensor-advanced should run");

    assert!(
        strict_report.cases_total >= 10,
        "expected at least 10 advanced cases"
    );
    assert_eq!(strict_report.cases_total, strict_cases.len());
    assert_eq!(strict_report.cases_total, strict_report.cases_passed);
    assert_eq!(hardened_report.cases_total, hardened_report.cases_passed);
}

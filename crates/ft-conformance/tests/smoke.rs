use std::path::Path;

use ft_conformance::{
    HarnessConfig, run_autograd_scheduler_conformance, run_dispatch_conformance,
    run_scalar_conformance, run_serialization_conformance, run_smoke, run_tensor_meta_conformance,
};
use ft_core::ExecutionMode;

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

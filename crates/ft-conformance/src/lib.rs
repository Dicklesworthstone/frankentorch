#![forbid(unsafe_code)]

mod logging;

use std::collections::{BTreeMap, BTreeSet};
use std::fs;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::time::Instant;

use ft_api::FrankenTorchSession;
use ft_autograd::{AutogradError, BackwardOptions, ReentrantPolicy, Tape};
use ft_core::{DType, Device, ExecutionMode, ScalarTensor, TensorMeta, contiguous_strides};
use ft_dispatch::{
    BinaryOp, DispatchKey, DispatchKeySet, dispatch_scalar_binary,
    dispatch_scalar_binary_with_keyset,
};
use ft_serialize::{
    CheckpointMode, DecodeMode, SnapshotEntry as SerializedSnapshotEntry, decode_checkpoint,
    encode_checkpoint, generate_raptorq_sidecar,
};
use logging::{StructuredCaseLog, mode_label};
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};

#[derive(Debug, Clone)]
pub struct HarnessConfig {
    pub oracle_root: PathBuf,
    pub fixture_root: PathBuf,
    pub allowlist_path: PathBuf,
    pub strict_mode: bool,
    pub legacy_oracle_python: Option<PathBuf>,
}

impl HarnessConfig {
    #[must_use]
    pub fn default_paths() -> Self {
        let repo_root = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..");
        Self {
            oracle_root: repo_root.join("legacy_pytorch_code/pytorch"),
            fixture_root: PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("fixtures"),
            allowlist_path: repo_root
                .join("artifacts/phase2c/HARDENED_DEVIATION_ALLOWLIST_V1.json"),
            strict_mode: true,
            legacy_oracle_python: default_oracle_python(&repo_root),
        }
    }
}

impl Default for HarnessConfig {
    fn default() -> Self {
        Self::default_paths()
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct CaseReport {
    pub name: String,
    pub mode: ExecutionMode,
    pub output_ok: bool,
    pub lhs_grad_ok: bool,
    pub rhs_grad_ok: bool,
    pub forensic_log: StructuredCaseLog,
}

#[derive(Debug, Clone, PartialEq)]
pub struct DispatchCaseReport {
    pub name: String,
    pub mode: ExecutionMode,
    pub output_ok: bool,
    pub selected_key_ok: bool,
    pub backend_key_ok: bool,
    pub kernel_ok: bool,
    pub fallback_ok: bool,
    pub error_ok: bool,
    pub forensic_log: StructuredCaseLog,
}

impl DispatchCaseReport {
    #[must_use]
    pub fn passed(&self) -> bool {
        self.output_ok
            && self.selected_key_ok
            && self.backend_key_ok
            && self.kernel_ok
            && self.fallback_ok
            && self.error_ok
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct SchedulerCaseReport {
    pub name: String,
    pub mode: ExecutionMode,
    pub grad_ok: bool,
    pub order_ok: bool,
    pub reentrant_policy_ok: bool,
    pub forensic_log: StructuredCaseLog,
}

impl SchedulerCaseReport {
    #[must_use]
    pub fn passed(&self) -> bool {
        self.grad_ok && self.order_ok && self.reentrant_policy_ok
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct SerializationCaseReport {
    pub name: String,
    pub mode: ExecutionMode,
    pub decode_ok: bool,
    pub sidecar_ok: bool,
    pub proof_deterministic_ok: bool,
    pub forensic_log: StructuredCaseLog,
}

impl SerializationCaseReport {
    #[must_use]
    pub fn passed(&self) -> bool {
        self.decode_ok && self.sidecar_ok && self.proof_deterministic_ok
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct TensorMetaCaseReport {
    pub name: String,
    pub mode: ExecutionMode,
    pub meta_ok: bool,
    pub index_ok: bool,
    pub alias_ok: bool,
    pub forensic_log: StructuredCaseLog,
}

impl TensorMetaCaseReport {
    #[must_use]
    pub fn passed(&self) -> bool {
        self.meta_ok && self.index_ok && self.alias_ok
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct HarnessReport {
    pub suite: &'static str,
    pub oracle_present: bool,
    pub fixture_count: usize,
    pub strict_mode: bool,
    pub cases_total: usize,
    pub cases_passed: usize,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BenchReport {
    pub iterations: usize,
    pub p50_ns: u128,
    pub p95_ns: u128,
    pub p99_ns: u128,
    pub mean_ns: u128,
}

#[derive(Debug, Clone, Deserialize)]
struct ScalarFixtureFile {
    cases: Vec<ScalarCase>,
}

#[derive(Debug, Clone, Deserialize)]
struct ScalarCase {
    name: String,
    op: String,
    lhs: f64,
    rhs: f64,
    expected_output: f64,
    expected_lhs_grad: f64,
    expected_rhs_grad: f64,
    tolerance: Option<f64>,
}

#[derive(Debug, Clone, Deserialize)]
struct TensorMetaFixtureFile {
    cases: Vec<TensorMetaCase>,
}

#[derive(Debug, Clone, Deserialize)]
struct TensorMetaCase {
    name: String,
    shape: Vec<usize>,
    strides: Option<Vec<usize>>,
    storage_offset: Option<usize>,
    index: Vec<usize>,
    expected_linear_index: Option<usize>,
    expected_numel: Option<usize>,
    expected_contiguous: Option<bool>,
    expect_valid: Option<bool>,
    alias_offset: Option<usize>,
}

#[derive(Debug, Clone, Deserialize)]
struct DispatchFixtureFile {
    cases: Vec<DispatchCase>,
}

#[derive(Debug, Clone, Deserialize)]
struct DispatchCase {
    name: String,
    op: String,
    lhs: f64,
    rhs: f64,
    requires_grad: bool,
    keyset: Option<Vec<String>>,
    strict: DispatchModeExpectation,
    hardened: DispatchModeExpectation,
    tolerance: Option<f64>,
}

#[derive(Debug, Clone, Deserialize)]
struct DispatchModeExpectation {
    expected_output: Option<f64>,
    expected_selected_key: Option<String>,
    expected_backend_key: Option<String>,
    expected_kernel: Option<String>,
    expected_fallback: Option<bool>,
    expect_error: Option<bool>,
}

#[derive(Debug, Clone, Deserialize)]
struct SchedulerFixtureFile {
    cases: Vec<SchedulerCase>,
}

#[derive(Debug, Clone, Deserialize)]
struct SchedulerCase {
    name: String,
    x: f64,
    y: f64,
    expected_x_grad: f64,
    expected_y_grad: f64,
    expected_execution_order: Vec<usize>,
    tolerance: Option<f64>,
}

#[derive(Debug, Clone, Deserialize)]
struct SerializationFixtureFile {
    cases: Vec<SerializationCase>,
}

#[derive(Debug, Clone, Deserialize)]
struct SerializationCase {
    name: String,
    entries: Vec<SerializationCaseEntry>,
    repair_symbols: Option<usize>,
}

#[derive(Debug, Clone, Deserialize)]
struct SerializationCaseEntry {
    node_id: usize,
    value: f64,
    grad: Option<f64>,
}

#[derive(Debug, Clone, PartialEq)]
struct ScalarObservation {
    output: f64,
    lhs_grad: f64,
    rhs_grad: f64,
}

#[derive(Debug, Clone, PartialEq)]
struct LegacyUnaryGradObservation {
    output: f64,
    x_grad: f64,
    y_grad: f64,
    reentrant_guard_triggered: bool,
}

#[derive(Debug, Clone, PartialEq)]
struct TensorMetaObservation {
    valid: bool,
    numel: Option<usize>,
    contiguous: Option<bool>,
    linear_index: Option<usize>,
    alias_ok: bool,
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct AllowlistIndex {
    by_packet: BTreeMap<String, BTreeSet<String>>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct E2EForensicsSummary {
    pub output_path: PathBuf,
    pub log_entries: usize,
    pub failed_entries: usize,
    pub modes: Vec<ExecutionMode>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct DifferentialHarnessReport {
    pub schema_version: &'static str,
    pub oracle: LegacyOracleStatus,
    pub modes: Vec<&'static str>,
    pub total_checks: usize,
    pub failed_checks: usize,
    pub allowlisted_drifts: usize,
    pub blocking_drifts: usize,
    pub checks: Vec<DifferentialCheck>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct LegacyOracleStatus {
    pub configured_python: Option<String>,
    pub active_python: Option<String>,
    pub available: bool,
    pub message: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct DifferentialCheck {
    pub suite: &'static str,
    pub packet_id: &'static str,
    pub scenario_id: String,
    pub case_name: String,
    pub mode: &'static str,
    pub comparator: &'static str,
    pub status: &'static str,
    pub allowlisted: bool,
    pub drift_id: Option<String>,
    pub reason_code: String,
    pub observed: String,
    pub expected: String,
    pub evidence_refs: Vec<String>,
}

#[must_use]
pub fn run_smoke(config: &HarnessConfig) -> HarnessReport {
    let fixture_count = fs::read_dir(&config.fixture_root)
        .ok()
        .into_iter()
        .flat_map(|it| it.filter_map(Result::ok))
        .count();

    let mode = if config.strict_mode {
        ExecutionMode::Strict
    } else {
        ExecutionMode::Hardened
    };

    let (scalar_total, scalar_passed) =
        run_scalar_conformance(config, mode).map_or((0, 0), |(_, cases)| {
            summarize_passes(
                cases
                    .iter()
                    .map(|case| case.output_ok && case.lhs_grad_ok && case.rhs_grad_ok),
            )
        });
    let (tensor_meta_total, tensor_meta_passed) = run_tensor_meta_conformance(config, mode)
        .map_or((0, 0), |(_, cases)| {
            summarize_passes(cases.iter().map(TensorMetaCaseReport::passed))
        });
    let (dispatch_total, dispatch_passed) = run_dispatch_conformance(config, mode)
        .map_or((0, 0), |(_, cases)| {
            summarize_passes(cases.iter().map(DispatchCaseReport::passed))
        });
    let (scheduler_total, scheduler_passed) = run_autograd_scheduler_conformance(config, mode)
        .map_or((0, 0), |(_, cases)| {
            summarize_passes(cases.iter().map(SchedulerCaseReport::passed))
        });
    let (serialization_total, serialization_passed) = run_serialization_conformance(config, mode)
        .map_or((0, 0), |(_, cases)| {
            summarize_passes(cases.iter().map(SerializationCaseReport::passed))
        });

    HarnessReport {
        suite: "smoke",
        oracle_present: config.oracle_root.exists(),
        fixture_count,
        strict_mode: config.strict_mode,
        cases_total: scalar_total
            + tensor_meta_total
            + dispatch_total
            + scheduler_total
            + serialization_total,
        cases_passed: scalar_passed
            + tensor_meta_passed
            + dispatch_passed
            + scheduler_passed
            + serialization_passed,
    }
}

pub fn run_scalar_conformance(
    config: &HarnessConfig,
    mode: ExecutionMode,
) -> Result<(HarnessReport, Vec<CaseReport>), String> {
    let fixture_path = config.fixture_root.join("scalar_autograd_cases.json");
    let fixture: ScalarFixtureFile = load_fixture(&fixture_path)?;

    let mut case_reports = Vec::with_capacity(fixture.cases.len());
    for case in fixture.cases {
        case_reports.push(run_scalar_case(&case, mode)?);
    }

    let (cases_total, cases_passed) = summarize_passes(
        case_reports
            .iter()
            .map(|case| case.output_ok && case.lhs_grad_ok && case.rhs_grad_ok),
    );

    let report = HarnessReport {
        suite: "scalar_dac",
        oracle_present: config.oracle_root.exists(),
        fixture_count: 1,
        strict_mode: mode == ExecutionMode::Strict,
        cases_total,
        cases_passed,
    };

    Ok((report, case_reports))
}

pub fn run_tensor_meta_conformance(
    config: &HarnessConfig,
    mode: ExecutionMode,
) -> Result<(HarnessReport, Vec<TensorMetaCaseReport>), String> {
    let fixture_path = config.fixture_root.join("tensor_meta_cases.json");
    let fixture: TensorMetaFixtureFile = load_fixture(&fixture_path)?;

    let mut case_reports = Vec::with_capacity(fixture.cases.len());
    for case in fixture.cases {
        case_reports.push(run_tensor_meta_case(&case, mode)?);
    }

    let (cases_total, cases_passed) =
        summarize_passes(case_reports.iter().map(TensorMetaCaseReport::passed));

    let report = HarnessReport {
        suite: "tensor_meta",
        oracle_present: config.oracle_root.exists(),
        fixture_count: 1,
        strict_mode: mode == ExecutionMode::Strict,
        cases_total,
        cases_passed,
    };

    Ok((report, case_reports))
}

pub fn run_dispatch_conformance(
    config: &HarnessConfig,
    mode: ExecutionMode,
) -> Result<(HarnessReport, Vec<DispatchCaseReport>), String> {
    let fixture_path = config.fixture_root.join("dispatch_key_cases.json");
    let fixture: DispatchFixtureFile = load_fixture(&fixture_path)?;

    let mut case_reports = Vec::with_capacity(fixture.cases.len());
    for case in fixture.cases {
        case_reports.push(run_dispatch_case(&case, mode)?);
    }

    let (cases_total, cases_passed) =
        summarize_passes(case_reports.iter().map(DispatchCaseReport::passed));

    let report = HarnessReport {
        suite: "dispatch_key",
        oracle_present: config.oracle_root.exists(),
        fixture_count: 1,
        strict_mode: mode == ExecutionMode::Strict,
        cases_total,
        cases_passed,
    };

    Ok((report, case_reports))
}

pub fn run_autograd_scheduler_conformance(
    config: &HarnessConfig,
    mode: ExecutionMode,
) -> Result<(HarnessReport, Vec<SchedulerCaseReport>), String> {
    let fixture_path = config.fixture_root.join("autograd_scheduler_cases.json");
    let fixture: SchedulerFixtureFile = load_fixture(&fixture_path)?;

    let mut case_reports = Vec::with_capacity(fixture.cases.len());
    for case in fixture.cases {
        case_reports.push(run_scheduler_case(&case, mode)?);
    }

    let (cases_total, cases_passed) =
        summarize_passes(case_reports.iter().map(SchedulerCaseReport::passed));

    let report = HarnessReport {
        suite: "autograd_scheduler",
        oracle_present: config.oracle_root.exists(),
        fixture_count: 1,
        strict_mode: mode == ExecutionMode::Strict,
        cases_total,
        cases_passed,
    };

    Ok((report, case_reports))
}

pub fn run_serialization_conformance(
    config: &HarnessConfig,
    mode: ExecutionMode,
) -> Result<(HarnessReport, Vec<SerializationCaseReport>), String> {
    let fixture_path = config.fixture_root.join("serialization_cases.json");
    let fixture: SerializationFixtureFile = load_fixture(&fixture_path)?;

    let mut case_reports = Vec::with_capacity(fixture.cases.len());
    for case in fixture.cases {
        case_reports.push(run_serialization_case(&case, mode)?);
    }

    let (cases_total, cases_passed) =
        summarize_passes(case_reports.iter().map(SerializationCaseReport::passed));

    let report = HarnessReport {
        suite: "serialization",
        oracle_present: config.oracle_root.exists(),
        fixture_count: 1,
        strict_mode: mode == ExecutionMode::Strict,
        cases_total,
        cases_passed,
    };

    Ok((report, case_reports))
}

pub fn emit_e2e_forensics_matrix(
    config: &HarnessConfig,
    output_path: &Path,
    modes: &[ExecutionMode],
) -> Result<E2EForensicsSummary, String> {
    emit_e2e_forensics_matrix_filtered(config, output_path, modes, None)
}

pub fn emit_e2e_forensics_matrix_filtered(
    config: &HarnessConfig,
    output_path: &Path,
    modes: &[ExecutionMode],
    packet_filter: Option<&str>,
) -> Result<E2EForensicsSummary, String> {
    let selected_modes = if modes.is_empty() {
        vec![ExecutionMode::Strict, ExecutionMode::Hardened]
    } else {
        modes.to_vec()
    };

    let mut logs = Vec::new();
    for mode in selected_modes.iter().copied() {
        let (_, scalar_cases) = run_scalar_conformance(config, mode)?;
        logs.extend(scalar_cases.into_iter().map(|case| case.forensic_log));

        let (_, tensor_meta_cases) = run_tensor_meta_conformance(config, mode)?;
        logs.extend(tensor_meta_cases.into_iter().map(|case| case.forensic_log));

        let (_, dispatch_cases) = run_dispatch_conformance(config, mode)?;
        logs.extend(dispatch_cases.into_iter().map(|case| case.forensic_log));

        let (_, scheduler_cases) = run_autograd_scheduler_conformance(config, mode)?;
        logs.extend(scheduler_cases.into_iter().map(|case| case.forensic_log));

        let (_, serialization_cases) = run_serialization_conformance(config, mode)?;
        logs.extend(
            serialization_cases
                .into_iter()
                .map(|case| case.forensic_log),
        );
    }

    if let Some(packet_id) = packet_filter {
        logs.retain(|entry| entry.packet_id == packet_id);
    }

    let mut lines = String::new();
    for entry in &logs {
        let line = serde_json::to_string(entry)
            .map_err(|error| format!("failed to serialize structured log entry: {error}"))?;
        lines.push_str(&line);
        lines.push('\n');
    }

    if let Some(parent) = output_path.parent() {
        fs::create_dir_all(parent).map_err(|error| {
            format!(
                "failed to create e2e forensics output dir {}: {error}",
                parent.display()
            )
        })?;
    }

    fs::write(output_path, lines).map_err(|error| {
        format!(
            "failed to write e2e forensics log {}: {error}",
            output_path.display()
        )
    })?;

    let failed_entries = logs.iter().filter(|entry| entry.outcome != "pass").count();

    Ok(E2EForensicsSummary {
        output_path: output_path.to_path_buf(),
        log_entries: logs.len(),
        failed_entries,
        modes: selected_modes,
    })
}

pub fn run_differential_conformance(
    config: &HarnessConfig,
    modes: &[ExecutionMode],
) -> Result<DifferentialHarnessReport, String> {
    let selected_modes = if modes.is_empty() {
        vec![ExecutionMode::Strict, ExecutionMode::Hardened]
    } else {
        modes.to_vec()
    };
    let allowlist = load_allowlist(config.allowlist_path.as_path())?;
    let oracle_status = probe_legacy_oracle(config);

    let mut checks = Vec::new();
    for mode in selected_modes.iter().copied() {
        let mode_str = mode_label(mode);

        let scalar_fixture: ScalarFixtureFile =
            load_fixture(&config.fixture_root.join("scalar_autograd_cases.json"))?;
        for case in scalar_fixture.cases {
            let local = evaluate_scalar_with_session(&case, mode)?;
            match query_legacy_scalar_oracle(config, &case) {
                Ok(oracle) => {
                    checks.push(compare_abs_tol(
                        &allowlist,
                        "scalar_dac",
                        "FT-P2C-001",
                        mode,
                        case.name.as_str(),
                        "abs_tol",
                        "scalar.output_mismatch",
                        local.output,
                        oracle.output,
                        case.tolerance.unwrap_or(1e-12),
                        vec![
                            "crates/ft-conformance/fixtures/scalar_autograd_cases.json".to_string(),
                            "artifacts/phase2c/FT-P2C-001/parity_report.json".to_string(),
                        ],
                    ));
                    checks.push(compare_abs_tol(
                        &allowlist,
                        "scalar_dac",
                        "FT-P2C-001",
                        mode,
                        case.name.as_str(),
                        "abs_tol",
                        "scalar.lhs_grad_mismatch",
                        local.lhs_grad,
                        oracle.lhs_grad,
                        case.tolerance.unwrap_or(1e-12),
                        vec![
                            "crates/ft-conformance/fixtures/scalar_autograd_cases.json".to_string(),
                            "artifacts/phase2c/FT-P2C-001/parity_report.json".to_string(),
                        ],
                    ));
                    checks.push(compare_abs_tol(
                        &allowlist,
                        "scalar_dac",
                        "FT-P2C-001",
                        mode,
                        case.name.as_str(),
                        "abs_tol",
                        "scalar.rhs_grad_mismatch",
                        local.rhs_grad,
                        oracle.rhs_grad,
                        case.tolerance.unwrap_or(1e-12),
                        vec![
                            "crates/ft-conformance/fixtures/scalar_autograd_cases.json".to_string(),
                            "artifacts/phase2c/FT-P2C-001/parity_report.json".to_string(),
                        ],
                    ));
                }
                Err(reason) => checks.push(DifferentialCheck {
                    suite: "scalar_dac",
                    packet_id: "FT-P2C-001",
                    scenario_id: scenario_id("scalar_dac", mode, case.name.as_str()),
                    case_name: case.name.clone(),
                    mode: mode_str,
                    comparator: "oracle.scalar",
                    status: "oracle_unavailable",
                    allowlisted: false,
                    drift_id: None,
                    reason_code: "legacy_oracle_unavailable".to_string(),
                    observed: reason,
                    expected: "legacy_oracle_response".to_string(),
                    evidence_refs: vec![
                        "crates/ft-conformance/fixtures/scalar_autograd_cases.json".to_string(),
                    ],
                }),
            }
        }

        let tensor_meta_fixture: TensorMetaFixtureFile =
            load_fixture(&config.fixture_root.join("tensor_meta_cases.json"))?;
        for case in tensor_meta_fixture.cases {
            let local = evaluate_tensor_meta_observation(&case)?;
            let expect_valid = case.expect_valid.unwrap_or(true);

            if !expect_valid {
                checks.push(compare_bool(
                    &allowlist,
                    "tensor_meta",
                    "FT-P2C-001",
                    mode,
                    case.name.as_str(),
                    "fail_closed",
                    "tensor_meta.invalid_fail_closed_mismatch",
                    !local.valid,
                    true,
                    vec![
                        "crates/ft-conformance/fixtures/tensor_meta_cases.json".to_string(),
                        "artifacts/phase2c/FT-P2C-001/parity_report.json".to_string(),
                    ],
                ));

                if !can_execute_tensor_meta_oracle_case(&case) {
                    checks.push(DifferentialCheck {
                        suite: "tensor_meta",
                        packet_id: "FT-P2C-001",
                        scenario_id: scenario_id("tensor_meta", mode, case.name.as_str()),
                        case_name: case.name.clone(),
                        mode: mode_str,
                        comparator: "fail_closed_oracle_guard",
                        status: "pass",
                        allowlisted: false,
                        drift_id: None,
                        reason_code: "oracle_guard_skip".to_string(),
                        observed: "guarded_skip".to_string(),
                        expected: "guarded_skip".to_string(),
                        evidence_refs: vec![
                            "crates/ft-conformance/fixtures/tensor_meta_cases.json".to_string(),
                            "artifacts/phase2c/FT-P2C-001/parity_report.json".to_string(),
                        ],
                    });
                } else if oracle_status.available {
                    checks.push(compare_bool(
                        &allowlist,
                        "tensor_meta",
                        "FT-P2C-001",
                        mode,
                        case.name.as_str(),
                        "fail_closed_oracle",
                        "tensor_meta.invalid_oracle_policy_mismatch",
                        query_legacy_tensor_meta_oracle(config, &case).is_err(),
                        true,
                        vec![
                            "crates/ft-conformance/fixtures/tensor_meta_cases.json".to_string(),
                            "artifacts/phase2c/FT-P2C-001/parity_report.json".to_string(),
                        ],
                    ));
                } else {
                    checks.push(DifferentialCheck {
                        suite: "tensor_meta",
                        packet_id: "FT-P2C-001",
                        scenario_id: scenario_id("tensor_meta", mode, case.name.as_str()),
                        case_name: case.name.clone(),
                        mode: mode_str,
                        comparator: "oracle.tensor_meta",
                        status: "oracle_unavailable",
                        allowlisted: false,
                        drift_id: None,
                        reason_code: "legacy_oracle_unavailable".to_string(),
                        observed: oracle_status.message.clone(),
                        expected: "legacy_oracle_response".to_string(),
                        evidence_refs: vec![
                            "crates/ft-conformance/fixtures/tensor_meta_cases.json".to_string(),
                        ],
                    });
                }
                continue;
            }

            let oracle_guard_ok = can_execute_tensor_meta_oracle_case(&case);
            checks.push(compare_bool(
                &allowlist,
                "tensor_meta",
                "FT-P2C-001",
                mode,
                case.name.as_str(),
                "oracle_guard",
                "tensor_meta.oracle_guard_triggered",
                oracle_guard_ok,
                true,
                vec![
                    "crates/ft-conformance/fixtures/tensor_meta_cases.json".to_string(),
                    "artifacts/phase2c/FT-P2C-001/parity_report.json".to_string(),
                ],
            ));
            if !oracle_guard_ok {
                continue;
            }

            match query_legacy_tensor_meta_oracle(config, &case) {
                Ok(oracle) => {
                    checks.push(compare_bool(
                        &allowlist,
                        "tensor_meta",
                        "FT-P2C-001",
                        mode,
                        case.name.as_str(),
                        "numel",
                        "tensor_meta.numel_mismatch",
                        local
                            .numel
                            .zip(oracle.numel)
                            .is_some_and(|(lhs, rhs)| lhs == rhs),
                        true,
                        vec![
                            "crates/ft-conformance/fixtures/tensor_meta_cases.json".to_string(),
                            "artifacts/phase2c/FT-P2C-001/parity_report.json".to_string(),
                        ],
                    ));
                    checks.push(compare_bool(
                        &allowlist,
                        "tensor_meta",
                        "FT-P2C-001",
                        mode,
                        case.name.as_str(),
                        "contiguous",
                        "tensor_meta.contiguous_mismatch",
                        local
                            .contiguous
                            .zip(oracle.contiguous)
                            .is_some_and(|(lhs, rhs)| lhs == rhs),
                        true,
                        vec![
                            "crates/ft-conformance/fixtures/tensor_meta_cases.json".to_string(),
                            "artifacts/phase2c/FT-P2C-001/parity_report.json".to_string(),
                        ],
                    ));
                    checks.push(compare_bool(
                        &allowlist,
                        "tensor_meta",
                        "FT-P2C-001",
                        mode,
                        case.name.as_str(),
                        "linear_index",
                        "tensor_meta.linear_index_mismatch",
                        local
                            .linear_index
                            .zip(oracle.linear_index)
                            .is_some_and(|(lhs, rhs)| lhs == rhs),
                        true,
                        vec![
                            "crates/ft-conformance/fixtures/tensor_meta_cases.json".to_string(),
                            "artifacts/phase2c/FT-P2C-001/parity_report.json".to_string(),
                        ],
                    ));

                    if let Some(shifted_case) = offset_shift_tensor_meta_case(&case, 1) {
                        let shifted_local = evaluate_tensor_meta_observation(&shifted_case)?;
                        checks.push(compare_bool(
                            &allowlist,
                            "tensor_meta",
                            "FT-P2C-001",
                            mode,
                            case.name.as_str(),
                            "metamorphic_offset_shift_numel_local",
                            "tensor_meta.metamorphic_offset_shift_numel_local_mismatch",
                            shifted_local
                                .numel
                                .zip(local.numel)
                                .is_some_and(|(shifted, base)| shifted == base),
                            true,
                            vec![
                                "crates/ft-conformance/fixtures/tensor_meta_cases.json".to_string(),
                                "artifacts/phase2c/FT-P2C-001/parity_report.json".to_string(),
                            ],
                        ));
                        checks.push(compare_bool(
                            &allowlist,
                            "tensor_meta",
                            "FT-P2C-001",
                            mode,
                            case.name.as_str(),
                            "metamorphic_offset_shift_contiguous_local",
                            "tensor_meta.metamorphic_offset_shift_contiguous_local_mismatch",
                            shifted_local
                                .contiguous
                                .zip(local.contiguous)
                                .is_some_and(|(shifted, base)| shifted == base),
                            true,
                            vec![
                                "crates/ft-conformance/fixtures/tensor_meta_cases.json".to_string(),
                                "artifacts/phase2c/FT-P2C-001/parity_report.json".to_string(),
                            ],
                        ));
                        checks.push(compare_bool(
                            &allowlist,
                            "tensor_meta",
                            "FT-P2C-001",
                            mode,
                            case.name.as_str(),
                            "metamorphic_offset_shift_linear_local",
                            "tensor_meta.metamorphic_offset_shift_linear_local_mismatch",
                            linear_index_shift_is_delta(
                                local.linear_index,
                                shifted_local.linear_index,
                                1,
                            ),
                            true,
                            vec![
                                "crates/ft-conformance/fixtures/tensor_meta_cases.json".to_string(),
                                "artifacts/phase2c/FT-P2C-001/parity_report.json".to_string(),
                            ],
                        ));

                        if can_execute_tensor_meta_oracle_case(&shifted_case) {
                            match query_legacy_tensor_meta_oracle(config, &shifted_case) {
                                Ok(shifted_oracle) => {
                                    checks.push(compare_bool(
                                        &allowlist,
                                        "tensor_meta",
                                        "FT-P2C-001",
                                        mode,
                                        case.name.as_str(),
                                        "metamorphic_offset_shift_numel_oracle",
                                        "tensor_meta.metamorphic_offset_shift_numel_oracle_mismatch",
                                        shifted_oracle
                                            .numel
                                            .zip(oracle.numel)
                                            .is_some_and(|(shifted, base)| shifted == base),
                                        true,
                                        vec![
                                            "crates/ft-conformance/fixtures/tensor_meta_cases.json"
                                                .to_string(),
                                            "artifacts/phase2c/FT-P2C-001/parity_report.json"
                                                .to_string(),
                                        ],
                                    ));
                                    checks.push(compare_bool(
                                        &allowlist,
                                        "tensor_meta",
                                        "FT-P2C-001",
                                        mode,
                                        case.name.as_str(),
                                        "metamorphic_offset_shift_contiguous_oracle",
                                        "tensor_meta.metamorphic_offset_shift_contiguous_oracle_mismatch",
                                        shifted_oracle
                                            .contiguous
                                            .zip(oracle.contiguous)
                                            .is_some_and(|(shifted, base)| shifted == base),
                                        true,
                                        vec![
                                            "crates/ft-conformance/fixtures/tensor_meta_cases.json"
                                                .to_string(),
                                            "artifacts/phase2c/FT-P2C-001/parity_report.json"
                                                .to_string(),
                                        ],
                                    ));
                                    checks.push(compare_bool(
                                        &allowlist,
                                        "tensor_meta",
                                        "FT-P2C-001",
                                        mode,
                                        case.name.as_str(),
                                        "metamorphic_offset_shift_linear_oracle",
                                        "tensor_meta.metamorphic_offset_shift_linear_oracle_mismatch",
                                        linear_index_shift_is_delta(
                                            oracle.linear_index,
                                            shifted_oracle.linear_index,
                                            1,
                                        ),
                                        true,
                                        vec![
                                            "crates/ft-conformance/fixtures/tensor_meta_cases.json"
                                                .to_string(),
                                            "artifacts/phase2c/FT-P2C-001/parity_report.json"
                                                .to_string(),
                                        ],
                                    ));
                                }
                                Err(reason) => checks.push(DifferentialCheck {
                                    suite: "tensor_meta",
                                    packet_id: "FT-P2C-001",
                                    scenario_id: scenario_id(
                                        "tensor_meta",
                                        mode,
                                        shifted_case.name.as_str(),
                                    ),
                                    case_name: shifted_case.name.clone(),
                                    mode: mode_str,
                                    comparator: "oracle.tensor_meta",
                                    status: "oracle_unavailable",
                                    allowlisted: false,
                                    drift_id: None,
                                    reason_code: "legacy_oracle_unavailable".to_string(),
                                    observed: reason,
                                    expected: "legacy_oracle_response".to_string(),
                                    evidence_refs: vec![
                                        "crates/ft-conformance/fixtures/tensor_meta_cases.json"
                                            .to_string(),
                                    ],
                                }),
                            }
                        } else {
                            checks.push(compare_bool(
                                &allowlist,
                                "tensor_meta",
                                "FT-P2C-001",
                                mode,
                                case.name.as_str(),
                                "metamorphic_offset_shift_oracle_guard",
                                "tensor_meta.metamorphic_offset_shift_oracle_guard_triggered",
                                false,
                                true,
                                vec![
                                    "crates/ft-conformance/fixtures/tensor_meta_cases.json"
                                        .to_string(),
                                    "artifacts/phase2c/FT-P2C-001/parity_report.json".to_string(),
                                ],
                            ));
                        }
                    }
                }
                Err(reason) => checks.push(DifferentialCheck {
                    suite: "tensor_meta",
                    packet_id: "FT-P2C-001",
                    scenario_id: scenario_id("tensor_meta", mode, case.name.as_str()),
                    case_name: case.name.clone(),
                    mode: mode_str,
                    comparator: "oracle.tensor_meta",
                    status: "oracle_unavailable",
                    allowlisted: false,
                    drift_id: None,
                    reason_code: "legacy_oracle_unavailable".to_string(),
                    observed: reason,
                    expected: "legacy_oracle_response".to_string(),
                    evidence_refs: vec![
                        "crates/ft-conformance/fixtures/tensor_meta_cases.json".to_string(),
                    ],
                }),
            }

            checks.push(compare_bool(
                &allowlist,
                "tensor_meta",
                "FT-P2C-001",
                mode,
                case.name.as_str(),
                "alias_policy",
                "tensor_meta.alias_identity_mismatch",
                local.alias_ok,
                true,
                vec![
                    "crates/ft-conformance/fixtures/tensor_meta_cases.json".to_string(),
                    "artifacts/phase2c/FT-P2C-001/parity_report.json".to_string(),
                ],
            ));
        }

        let dispatch_fixture: DispatchFixtureFile =
            load_fixture(&config.fixture_root.join("dispatch_key_cases.json"))?;
        for case in dispatch_fixture.cases {
            if case.keyset.is_some() {
                if mode == ExecutionMode::Hardened {
                    checks.push(compare_bool(
                        &allowlist,
                        "dispatch_key",
                        "FT-P2C-002",
                        mode,
                        case.name.as_str(),
                        "mode_split_policy",
                        "dispatch.composite_backend_fallback",
                        true,
                        false,
                        vec![
                            "crates/ft-conformance/fixtures/dispatch_key_cases.json".to_string(),
                            "artifacts/phase2c/HARDENED_DEVIATION_ALLOWLIST_V1.json".to_string(),
                        ],
                    ));
                } else {
                    checks.push(DifferentialCheck {
                        suite: "dispatch_key",
                        packet_id: "FT-P2C-002",
                        scenario_id: scenario_id("dispatch_key", mode, case.name.as_str()),
                        case_name: case.name.clone(),
                        mode: mode_str,
                        comparator: "mode_split_policy",
                        status: "pass",
                        allowlisted: false,
                        drift_id: None,
                        reason_code: "strict_fail_closed_mode_split".to_string(),
                        observed: "strict_fail_closed".to_string(),
                        expected: "strict_fail_closed".to_string(),
                        evidence_refs: vec![
                            "crates/ft-conformance/fixtures/dispatch_key_cases.json".to_string(),
                        ],
                    });
                }
                continue;
            }

            let local = evaluate_dispatch_output(&case, mode);
            let expected_error = match mode {
                ExecutionMode::Strict => case.strict.expect_error.unwrap_or(false),
                ExecutionMode::Hardened => case.hardened.expect_error.unwrap_or(false),
            };

            if expected_error {
                checks.push(compare_bool(
                    &allowlist,
                    "dispatch_key",
                    "FT-P2C-002",
                    mode,
                    case.name.as_str(),
                    "error_contract",
                    "dispatch.expected_error_mismatch",
                    local.is_err(),
                    true,
                    vec![
                        "crates/ft-conformance/fixtures/dispatch_key_cases.json".to_string(),
                        "artifacts/phase2c/FT-P2C-002/parity_report.json".to_string(),
                    ],
                ));
                continue;
            }

            let local_output = local?;
            match query_legacy_scalar_oracle(
                config,
                &ScalarCase {
                    name: case.name.clone(),
                    op: case.op.clone(),
                    lhs: case.lhs,
                    rhs: case.rhs,
                    expected_output: 0.0,
                    expected_lhs_grad: 0.0,
                    expected_rhs_grad: 0.0,
                    tolerance: case.tolerance,
                },
            ) {
                Ok(oracle) => checks.push(compare_abs_tol(
                    &allowlist,
                    "dispatch_key",
                    "FT-P2C-002",
                    mode,
                    case.name.as_str(),
                    "abs_tol",
                    "dispatch.output_mismatch",
                    local_output,
                    oracle.output,
                    case.tolerance.unwrap_or(1e-12),
                    vec![
                        "crates/ft-conformance/fixtures/dispatch_key_cases.json".to_string(),
                        "artifacts/phase2c/FT-P2C-002/parity_report.json".to_string(),
                    ],
                )),
                Err(reason) => checks.push(DifferentialCheck {
                    suite: "dispatch_key",
                    packet_id: "FT-P2C-002",
                    scenario_id: scenario_id("dispatch_key", mode, case.name.as_str()),
                    case_name: case.name.clone(),
                    mode: mode_str,
                    comparator: "oracle.dispatch",
                    status: "oracle_unavailable",
                    allowlisted: false,
                    drift_id: None,
                    reason_code: "legacy_oracle_unavailable".to_string(),
                    observed: reason,
                    expected: "legacy_oracle_response".to_string(),
                    evidence_refs: vec![
                        "crates/ft-conformance/fixtures/dispatch_key_cases.json".to_string(),
                    ],
                }),
            }
        }

        let scheduler_fixture: SchedulerFixtureFile =
            load_fixture(&config.fixture_root.join("autograd_scheduler_cases.json"))?;
        for case in scheduler_fixture.cases {
            let local = evaluate_scheduler_with_tape(&case, mode)?;
            match query_legacy_scheduler_oracle(config, &case) {
                Ok(oracle) => {
                    checks.push(compare_abs_tol(
                        &allowlist,
                        "autograd_scheduler",
                        "FT-P2C-004",
                        mode,
                        case.name.as_str(),
                        "abs_tol",
                        "autograd.output_mismatch",
                        local.output,
                        oracle.output,
                        case.tolerance.unwrap_or(1e-12),
                        vec![
                            "crates/ft-conformance/fixtures/autograd_scheduler_cases.json"
                                .to_string(),
                            "artifacts/phase2c/FT-P2C-004/parity_report.json".to_string(),
                        ],
                    ));
                    checks.push(compare_abs_tol(
                        &allowlist,
                        "autograd_scheduler",
                        "FT-P2C-004",
                        mode,
                        case.name.as_str(),
                        "abs_tol",
                        "autograd.x_grad_mismatch",
                        local.x_grad,
                        oracle.x_grad,
                        case.tolerance.unwrap_or(1e-12),
                        vec![
                            "crates/ft-conformance/fixtures/autograd_scheduler_cases.json"
                                .to_string(),
                            "artifacts/phase2c/FT-P2C-004/parity_report.json".to_string(),
                        ],
                    ));
                    checks.push(compare_abs_tol(
                        &allowlist,
                        "autograd_scheduler",
                        "FT-P2C-004",
                        mode,
                        case.name.as_str(),
                        "abs_tol",
                        "autograd.y_grad_mismatch",
                        local.y_grad,
                        oracle.y_grad,
                        case.tolerance.unwrap_or(1e-12),
                        vec![
                            "crates/ft-conformance/fixtures/autograd_scheduler_cases.json"
                                .to_string(),
                            "artifacts/phase2c/FT-P2C-004/parity_report.json".to_string(),
                        ],
                    ));
                }
                Err(reason) => checks.push(DifferentialCheck {
                    suite: "autograd_scheduler",
                    packet_id: "FT-P2C-004",
                    scenario_id: scenario_id("autograd_scheduler", mode, case.name.as_str()),
                    case_name: case.name.clone(),
                    mode: mode_str,
                    comparator: "oracle.autograd",
                    status: "oracle_unavailable",
                    allowlisted: false,
                    drift_id: None,
                    reason_code: "legacy_oracle_unavailable".to_string(),
                    observed: reason,
                    expected: "legacy_oracle_response".to_string(),
                    evidence_refs: vec![
                        "crates/ft-conformance/fixtures/autograd_scheduler_cases.json".to_string(),
                    ],
                }),
            }

            if mode == ExecutionMode::Hardened && local.reentrant_guard_triggered {
                checks.push(compare_bool(
                    &allowlist,
                    "autograd_scheduler",
                    "FT-P2C-004",
                    mode,
                    case.name.as_str(),
                    "policy",
                    "autograd.reentrant_depth_bounded_fallback",
                    true,
                    false,
                    vec![
                        "crates/ft-conformance/fixtures/autograd_scheduler_cases.json".to_string(),
                        "artifacts/phase2c/HARDENED_DEVIATION_ALLOWLIST_V1.json".to_string(),
                    ],
                ));
            } else {
                checks.push(DifferentialCheck {
                    suite: "autograd_scheduler",
                    packet_id: "FT-P2C-004",
                    scenario_id: scenario_id("autograd_scheduler", mode, case.name.as_str()),
                    case_name: case.name.clone(),
                    mode: mode_str,
                    comparator: "policy",
                    status: "pass",
                    allowlisted: false,
                    drift_id: None,
                    reason_code: "reentrant_policy_match".to_string(),
                    observed: if local.reentrant_guard_triggered {
                        "guard_triggered"
                    } else {
                        "guard_not_triggered"
                    }
                    .to_string(),
                    expected: if mode == ExecutionMode::Strict {
                        "strict_fail_closed"
                    } else {
                        "hardened_guard_optional"
                    }
                    .to_string(),
                    evidence_refs: vec![
                        "crates/ft-conformance/fixtures/autograd_scheduler_cases.json".to_string(),
                    ],
                });
            }
        }
    }

    checks.sort_by(|left, right| {
        (
            left.packet_id,
            left.suite,
            left.mode,
            left.case_name.as_str(),
            left.comparator,
            left.drift_id.as_deref().unwrap_or(""),
        )
            .cmp(&(
                right.packet_id,
                right.suite,
                right.mode,
                right.case_name.as_str(),
                right.comparator,
                right.drift_id.as_deref().unwrap_or(""),
            ))
    });

    let total_checks = checks.len();
    let allowlisted_drifts = checks
        .iter()
        .filter(|check| check.status == "allowlisted_drift")
        .count();
    let blocking_drifts = checks
        .iter()
        .filter(|check| check.status == "blocking_drift")
        .count();
    let failed_checks = checks
        .iter()
        .filter(|check| check.status == "blocking_drift" || check.status == "oracle_unavailable")
        .count();

    Ok(DifferentialHarnessReport {
        schema_version: "ft-conformance-differential-v1",
        oracle: oracle_status,
        modes: selected_modes
            .iter()
            .map(|mode| mode_label(*mode))
            .collect(),
        total_checks,
        failed_checks,
        allowlisted_drifts,
        blocking_drifts,
        checks,
    })
}

pub fn emit_differential_report(
    config: &HarnessConfig,
    output_path: &Path,
    modes: &[ExecutionMode],
) -> Result<DifferentialHarnessReport, String> {
    let report = run_differential_conformance(config, modes)?;
    if let Some(parent) = output_path.parent() {
        fs::create_dir_all(parent).map_err(|error| {
            format!(
                "failed to create differential report dir {}: {error}",
                parent.display()
            )
        })?;
    }

    fs::write(
        output_path,
        serde_json::to_string_pretty(&report)
            .map_err(|error| format!("failed to serialize differential report: {error}"))?,
    )
    .map_err(|error| {
        format!(
            "failed to write differential report {}: {error}",
            output_path.display()
        )
    })?;

    Ok(report)
}

#[must_use]
pub fn run_scalar_microbench(iterations: usize, mode: ExecutionMode) -> BenchReport {
    let mut samples = Vec::with_capacity(iterations.max(1));

    for _ in 0..iterations.max(1) {
        let started = Instant::now();
        let mut session = FrankenTorchSession::new(mode);
        let x = session.variable(2.0, true);
        let y = session.variable(3.0, true);
        let z = session.add(x, y).expect("microbench add should succeed");
        let out = session.mul(z, x).expect("microbench mul should succeed");
        let _ = session
            .backward(out)
            .expect("microbench backward should succeed");
        samples.push(started.elapsed().as_nanos());
    }

    samples.sort_unstable();
    let sum = samples.iter().copied().sum::<u128>();
    let mean = sum / (samples.len() as u128);

    BenchReport {
        iterations: samples.len(),
        p50_ns: percentile(&samples, 50),
        p95_ns: percentile(&samples, 95),
        p99_ns: percentile(&samples, 99),
        mean_ns: mean,
    }
}

fn run_scalar_case(case: &ScalarCase, mode: ExecutionMode) -> Result<CaseReport, String> {
    let mut session = FrankenTorchSession::new(mode);
    let lhs = session.variable(case.lhs, true);
    let rhs = session.variable(case.rhs, true);

    let out = match case.op.as_str() {
        "add" => session.add(lhs, rhs),
        "mul" => session.mul(lhs, rhs),
        _ => return Err(format!("unsupported operation '{}'", case.op)),
    }
    .map_err(|error| format!("operation '{}' failed: {error}", case.name))?;

    let actual_output = session
        .value(out)
        .map_err(|error| format!("value read failed for '{}': {error}", case.name))?;

    let backward = session
        .backward(out)
        .map_err(|error| format!("backward failed for '{}': {error}", case.name))?;

    let actual_lhs_grad = session
        .gradient(&backward, lhs)
        .ok_or_else(|| format!("missing lhs grad for '{}'", case.name))?;
    let actual_rhs_grad = session
        .gradient(&backward, rhs)
        .ok_or_else(|| format!("missing rhs grad for '{}'", case.name))?;

    let tolerance = case.tolerance.unwrap_or(1e-12);
    let output_ok = within(actual_output, case.expected_output, tolerance);
    let lhs_grad_ok = within(actual_lhs_grad, case.expected_lhs_grad, tolerance);
    let rhs_grad_ok = within(actual_rhs_grad, case.expected_rhs_grad, tolerance);
    let outcome = if output_ok && lhs_grad_ok && rhs_grad_ok {
        "pass"
    } else {
        "fail"
    };

    Ok(CaseReport {
        name: case.name.clone(),
        mode,
        output_ok,
        lhs_grad_ok,
        rhs_grad_ok,
        forensic_log: StructuredCaseLog::new(
            "scalar_dac",
            "scalar_autograd_cases.json",
            "FT-P2C-001",
            case.name.as_str(),
            mode,
            vec![
                "crates/ft-conformance/fixtures/scalar_autograd_cases.json".to_string(),
                "artifacts/phase2c/FT-P2C-001/parity_report.json".to_string(),
            ],
            format!(
                "cargo test -p ft-conformance strict_scalar_conformance_is_green -- --nocapture # mode={}",
                mode_label(mode)
            ),
            outcome,
            if outcome == "pass" {
                "parity_ok"
            } else {
                "scalar_or_grad_mismatch"
            },
        ),
    })
}

fn run_tensor_meta_case(
    case: &TensorMetaCase,
    mode: ExecutionMode,
) -> Result<TensorMetaCaseReport, String> {
    let local = evaluate_tensor_meta_observation(case)?;
    let expect_valid = case.expect_valid.unwrap_or(true);

    let (meta_ok, index_ok, alias_ok) = if !expect_valid {
        (!local.valid, true, true)
    } else {
        let numel_ok = case
            .expected_numel
            .zip(local.numel)
            .is_none_or(|(expected, actual)| expected == actual);
        let contiguous_ok = case
            .expected_contiguous
            .zip(local.contiguous)
            .is_none_or(|(expected, actual)| expected == actual);
        let index_ok = case
            .expected_linear_index
            .zip(local.linear_index)
            .is_none_or(|(expected, actual)| expected == actual);
        let meta_ok = local.valid && numel_ok && contiguous_ok;
        (meta_ok, index_ok, local.alias_ok)
    };

    let passed = meta_ok && index_ok && alias_ok;

    Ok(TensorMetaCaseReport {
        name: case.name.clone(),
        mode,
        meta_ok,
        index_ok,
        alias_ok,
        forensic_log: StructuredCaseLog::new(
            "tensor_meta",
            "tensor_meta_cases.json",
            "FT-P2C-001",
            case.name.as_str(),
            mode,
            vec![
                "crates/ft-conformance/fixtures/tensor_meta_cases.json".to_string(),
                "artifacts/phase2c/FT-P2C-001/parity_report.json".to_string(),
            ],
            format!(
                "cargo test -p ft-conformance strict_tensor_meta_conformance_is_green -- --nocapture # mode={}",
                mode_label(mode)
            ),
            if passed { "pass" } else { "fail" },
            if passed {
                "tensor_meta_parity_ok"
            } else {
                "tensor_meta_expectation_mismatch"
            },
        ),
    })
}

fn evaluate_tensor_meta_observation(
    case: &TensorMetaCase,
) -> Result<TensorMetaObservation, String> {
    let storage_offset = case.storage_offset.unwrap_or(0);
    let meta_result = if let Some(strides) = &case.strides {
        TensorMeta::from_shape_and_strides(
            case.shape.clone(),
            strides.clone(),
            storage_offset,
            DType::F64,
            Device::Cpu,
        )
    } else {
        let meta = TensorMeta::from_shape(case.shape.clone(), DType::F64, Device::Cpu)
            .with_storage_offset(storage_offset);
        meta.validate().map(|_| meta)
    };

    match meta_result {
        Ok(meta) => {
            let linear_index = meta.storage_index_for(case.index.as_slice()).ok();
            let alias_ok = if let Some(alias_offset) = case.alias_offset {
                let tensor = ScalarTensor::new(1.0, DType::F64, Device::Cpu);
                tensor.alias_view(alias_offset).is_ok_and(|alias| {
                    alias.storage_id() == tensor.storage_id() && alias.id() != tensor.id()
                })
            } else {
                true
            };
            Ok(TensorMetaObservation {
                valid: true,
                numel: Some(meta.numel()),
                contiguous: Some(meta.is_contiguous()),
                linear_index,
                alias_ok,
            })
        }
        Err(_error) => Ok(TensorMetaObservation {
            valid: false,
            numel: None,
            contiguous: None,
            linear_index: None,
            alias_ok: true,
        }),
    }
}

const LEGACY_TENSOR_META_ORACLE_MAX_ELEMENTS: usize = 1_000_000;

fn can_execute_tensor_meta_oracle_case(case: &TensorMetaCase) -> bool {
    let inferred_strides;
    let strides = if let Some(strides) = case.strides.as_deref() {
        strides
    } else {
        inferred_strides = contiguous_strides(case.shape.as_slice());
        inferred_strides.as_slice()
    };

    let mut max_linear = case.storage_offset.unwrap_or(0);
    for (size, stride) in case.shape.iter().copied().zip(strides.iter().copied()) {
        if size == 0 {
            continue;
        }
        let Some(span) = stride.checked_mul(size.saturating_sub(1)) else {
            return false;
        };
        let Some(next_max) = max_linear.checked_add(span) else {
            return false;
        };
        max_linear = next_max;
        if max_linear > LEGACY_TENSOR_META_ORACLE_MAX_ELEMENTS.saturating_sub(1) {
            return false;
        }
    }

    true
}

fn offset_shift_tensor_meta_case(case: &TensorMetaCase, delta: usize) -> Option<TensorMetaCase> {
    let mut shifted = case.clone();
    let next_offset = shifted.storage_offset.unwrap_or(0).checked_add(delta)?;
    shifted.storage_offset = Some(next_offset);
    shifted.expect_valid = Some(true);
    shifted.expected_numel = None;
    shifted.expected_contiguous = None;
    shifted.expected_linear_index = None;
    shifted.alias_offset = None;
    shifted.name = format!("{}__offset_plus_{delta}", shifted.name);
    Some(shifted)
}

fn linear_index_shift_is_delta(base: Option<usize>, shifted: Option<usize>, delta: usize) -> bool {
    base.zip(shifted)
        .is_some_and(|(base_index, shifted_index)| {
            base_index
                .checked_add(delta)
                .is_some_and(|expected| shifted_index == expected)
        })
}

fn run_dispatch_case(
    case: &DispatchCase,
    mode: ExecutionMode,
) -> Result<DispatchCaseReport, String> {
    let expectation = match mode {
        ExecutionMode::Strict => &case.strict,
        ExecutionMode::Hardened => &case.hardened,
    };

    let op = parse_binary_op(&case.op)?;
    let lhs = ScalarTensor::new(case.lhs, DType::F64, Device::Cpu);
    let rhs = ScalarTensor::new(case.rhs, DType::F64, Device::Cpu);

    let result = if let Some(keys) = &case.keyset {
        let keyset = parse_keyset(keys)?;
        dispatch_scalar_binary_with_keyset(op, mode, &lhs, &rhs, keyset)
    } else {
        dispatch_scalar_binary(op, mode, &lhs, &rhs, case.requires_grad)
    };

    let expected_error = expectation.expect_error.unwrap_or(false);
    let tolerance = case.tolerance.unwrap_or(1e-12);

    if expected_error {
        let error_ok = result.is_err();
        return Ok(DispatchCaseReport {
            name: case.name.clone(),
            mode,
            output_ok: true,
            selected_key_ok: true,
            backend_key_ok: true,
            kernel_ok: true,
            fallback_ok: true,
            error_ok,
            forensic_log: StructuredCaseLog::new(
                "dispatch_key",
                "dispatch_key_cases.json",
                "FT-P2C-002",
                case.name.as_str(),
                mode,
                vec![
                    "crates/ft-conformance/fixtures/dispatch_key_cases.json".to_string(),
                    "artifacts/phase2c/FT-P2C-002/parity_report.json".to_string(),
                ],
                format!(
                    "cargo test -p ft-conformance strict_dispatch_conformance_is_green -- --nocapture # mode={}",
                    mode_label(mode)
                ),
                if error_ok { "pass" } else { "fail" },
                if error_ok {
                    "expected_error_observed"
                } else {
                    "expected_error_missing"
                },
            ),
        });
    }

    let outcome =
        result.map_err(|error| format!("dispatch case '{}' failed: {error}", case.name))?;

    let output_ok = expectation
        .expected_output
        .is_none_or(|expected| within(outcome.tensor.value(), expected, tolerance));

    let selected_key_ok = expectation
        .expected_selected_key
        .as_deref()
        .and_then(parse_dispatch_key)
        .is_none_or(|expected| expected == outcome.decision.selected_key);

    let backend_key_ok = expectation
        .expected_backend_key
        .as_deref()
        .and_then(parse_dispatch_key)
        .is_none_or(|expected| expected == outcome.decision.backend_key);

    let kernel_ok = expectation
        .expected_kernel
        .as_deref()
        .is_none_or(|expected| expected == outcome.decision.kernel);

    let fallback_ok = expectation
        .expected_fallback
        .is_none_or(|expected| expected == outcome.decision.fallback_used);
    let passed = output_ok && selected_key_ok && backend_key_ok && kernel_ok && fallback_ok;

    Ok(DispatchCaseReport {
        name: case.name.clone(),
        mode,
        output_ok,
        selected_key_ok,
        backend_key_ok,
        kernel_ok,
        fallback_ok,
        error_ok: true,
        forensic_log: StructuredCaseLog::new(
            "dispatch_key",
            "dispatch_key_cases.json",
            "FT-P2C-002",
            case.name.as_str(),
            mode,
            vec![
                "crates/ft-conformance/fixtures/dispatch_key_cases.json".to_string(),
                "artifacts/phase2c/FT-P2C-002/parity_report.json".to_string(),
            ],
            format!(
                "cargo test -p ft-conformance strict_dispatch_conformance_is_green -- --nocapture # mode={}",
                mode_label(mode)
            ),
            if passed { "pass" } else { "fail" },
            if passed {
                "dispatch_parity_ok"
            } else {
                "dispatch_expectation_mismatch"
            },
        ),
    })
}

fn run_scheduler_case(
    case: &SchedulerCase,
    mode: ExecutionMode,
) -> Result<SchedulerCaseReport, String> {
    let mut tape = Tape::new();
    let x = tape.leaf(case.x, true);
    let y = tape.leaf(case.y, true);
    let (sum, _) = tape
        .add(x, y, mode)
        .map_err(|error| format!("scheduler case '{}' add failed: {error}", case.name))?;
    let (out, _) = tape
        .mul(sum, x, mode)
        .map_err(|error| format!("scheduler case '{}' mul failed: {error}", case.name))?;

    let report = tape
        .backward_with_options(out, BackwardOptions::for_mode(mode))
        .map_err(|error| format!("scheduler case '{}' backward failed: {error}", case.name))?;

    let tolerance = case.tolerance.unwrap_or(1e-12);
    let grad_ok = report
        .gradient(x)
        .is_some_and(|value| within(value, case.expected_x_grad, tolerance))
        && report
            .gradient(y)
            .is_some_and(|value| within(value, case.expected_y_grad, tolerance));

    let actual_order: Vec<usize> = report
        .telemetry
        .execution_order
        .iter()
        .map(|node| node.0)
        .collect();
    let order_ok = actual_order == case.expected_execution_order;

    let reentrant_policy_ok = match mode {
        ExecutionMode::Strict => matches!(
            tape.backward_with_options(
                out,
                BackwardOptions {
                    max_reentrant_depth: 1,
                    current_reentrant_depth: 2,
                    policy: ReentrantPolicy::StrictFail,
                }
            ),
            Err(AutogradError::ReentrantDepthExceeded { .. })
        ),
        ExecutionMode::Hardened => tape
            .backward_with_options(
                out,
                BackwardOptions {
                    max_reentrant_depth: 1,
                    current_reentrant_depth: 2,
                    policy: ReentrantPolicy::HardenedBoundedFallback,
                },
            )
            .map(|overflow_report| overflow_report.telemetry.reentrant_guard_triggered)
            .unwrap_or(false),
    };
    let passed = grad_ok && order_ok && reentrant_policy_ok;

    Ok(SchedulerCaseReport {
        name: case.name.clone(),
        mode,
        grad_ok,
        order_ok,
        reentrant_policy_ok,
        forensic_log: StructuredCaseLog::new(
            "autograd_scheduler",
            "autograd_scheduler_cases.json",
            "FT-P2C-004",
            case.name.as_str(),
            mode,
            vec![
                "crates/ft-conformance/fixtures/autograd_scheduler_cases.json".to_string(),
                "artifacts/phase2c/FT-P2C-004/parity_report.json".to_string(),
            ],
            format!(
                "cargo test -p ft-conformance strict_scheduler_conformance_is_green -- --nocapture # mode={}",
                mode_label(mode)
            ),
            if passed { "pass" } else { "fail" },
            if passed {
                "scheduler_parity_ok"
            } else {
                "scheduler_expectation_mismatch"
            },
        ),
    })
}

fn run_serialization_case(
    case: &SerializationCase,
    mode: ExecutionMode,
) -> Result<SerializationCaseReport, String> {
    let checkpoint_mode = match mode {
        ExecutionMode::Strict => CheckpointMode::Strict,
        ExecutionMode::Hardened => CheckpointMode::Hardened,
    };
    let decode_mode = match mode {
        ExecutionMode::Strict => DecodeMode::Strict,
        ExecutionMode::Hardened => DecodeMode::Hardened,
    };

    let entries: Vec<SerializedSnapshotEntry> = case
        .entries
        .iter()
        .map(|entry| SerializedSnapshotEntry {
            node_id: entry.node_id,
            value: entry.value,
            grad: entry.grad,
        })
        .collect();

    let payload = encode_checkpoint(entries.as_slice(), checkpoint_mode);
    let decoded = decode_checkpoint(payload.as_str(), decode_mode)
        .map_err(|error| format!("serialization case '{}' decode failed: {error}", case.name))?;

    let mut expected_entries = entries.clone();
    expected_entries.sort_by_key(|entry| entry.node_id);
    let decode_ok = decoded.entries == expected_entries;

    let repair_symbols = case.repair_symbols.unwrap_or(4);
    let (sidecar_a, proof_a) = generate_raptorq_sidecar(payload.as_str(), repair_symbols)
        .map_err(|error| format!("serialization case '{}' sidecar failed: {error}", case.name))?;
    let (_sidecar_b, proof_b) = generate_raptorq_sidecar(payload.as_str(), repair_symbols)
        .map_err(|error| {
            format!(
                "serialization case '{}' sidecar repeat failed: {error}",
                case.name
            )
        })?;

    let sidecar_ok = sidecar_a.repair_symbol_count >= 1 && sidecar_a.constraints_symbol_count >= 1;
    let proof_deterministic_ok = proof_a.proof_hash == proof_b.proof_hash;
    let passed = decode_ok && sidecar_ok && proof_deterministic_ok;

    Ok(SerializationCaseReport {
        name: case.name.clone(),
        mode,
        decode_ok,
        sidecar_ok,
        proof_deterministic_ok,
        forensic_log: StructuredCaseLog::new(
            "serialization",
            "serialization_cases.json",
            "FT-P2C-006",
            case.name.as_str(),
            mode,
            vec![
                "crates/ft-conformance/fixtures/serialization_cases.json".to_string(),
                "artifacts/phase2c/FT-P2C-006/parity_report.json".to_string(),
                "artifacts/phase2c/FT-P2C-006/parity_report.raptorq.json".to_string(),
            ],
            format!(
                "cargo test -p ft-conformance strict_serialization_conformance_is_green -- --nocapture # mode={}",
                mode_label(mode)
            ),
            if passed { "pass" } else { "fail" },
            if passed {
                "serialization_parity_ok"
            } else {
                "serialization_expectation_mismatch"
            },
        ),
    })
}

#[allow(clippy::too_many_arguments)]
fn compare_abs_tol(
    allowlist: &AllowlistIndex,
    suite: &'static str,
    packet_id: &'static str,
    mode: ExecutionMode,
    case_name: &str,
    comparator: &'static str,
    drift_id: &'static str,
    observed: f64,
    expected: f64,
    tolerance: f64,
    evidence_refs: Vec<String>,
) -> DifferentialCheck {
    let mismatch = !within(observed, expected, tolerance);
    let (status, allowlisted, drift) =
        classify_drift_status(allowlist, packet_id, mode, mismatch, drift_id);

    DifferentialCheck {
        suite,
        packet_id,
        scenario_id: scenario_id(suite, mode, case_name),
        case_name: case_name.to_string(),
        mode: mode_label(mode),
        comparator,
        status,
        allowlisted,
        drift_id: drift,
        reason_code: if mismatch {
            drift_id.to_string()
        } else {
            "parity_ok".to_string()
        },
        observed: format!("{observed:.15}"),
        expected: format!("{expected:.15}"),
        evidence_refs,
    }
}

#[allow(clippy::too_many_arguments)]
fn compare_bool(
    allowlist: &AllowlistIndex,
    suite: &'static str,
    packet_id: &'static str,
    mode: ExecutionMode,
    case_name: &str,
    comparator: &'static str,
    drift_id: &'static str,
    observed: bool,
    expected: bool,
    evidence_refs: Vec<String>,
) -> DifferentialCheck {
    let mismatch = observed != expected;
    let (status, allowlisted, drift) =
        classify_drift_status(allowlist, packet_id, mode, mismatch, drift_id);

    DifferentialCheck {
        suite,
        packet_id,
        scenario_id: scenario_id(suite, mode, case_name),
        case_name: case_name.to_string(),
        mode: mode_label(mode),
        comparator,
        status,
        allowlisted,
        drift_id: drift,
        reason_code: if mismatch {
            drift_id.to_string()
        } else {
            "parity_ok".to_string()
        },
        observed: observed.to_string(),
        expected: expected.to_string(),
        evidence_refs,
    }
}

fn classify_drift_status(
    allowlist: &AllowlistIndex,
    packet_id: &str,
    mode: ExecutionMode,
    mismatch: bool,
    drift_id: &str,
) -> (&'static str, bool, Option<String>) {
    if !mismatch {
        return ("pass", false, None);
    }

    if mode == ExecutionMode::Hardened && allowlist.contains(packet_id, drift_id) {
        return ("allowlisted_drift", true, Some(drift_id.to_string()));
    }

    ("blocking_drift", false, Some(drift_id.to_string()))
}

fn evaluate_scalar_with_session(
    case: &ScalarCase,
    mode: ExecutionMode,
) -> Result<ScalarObservation, String> {
    let mut session = FrankenTorchSession::new(mode);
    let lhs = session.variable(case.lhs, true);
    let rhs = session.variable(case.rhs, true);
    let out = match case.op.as_str() {
        "add" => session.add(lhs, rhs),
        "mul" => session.mul(lhs, rhs),
        _ => return Err(format!("unsupported operation '{}'", case.op)),
    }
    .map_err(|error| format!("operation '{}' failed: {error}", case.name))?;

    let actual_output = session
        .value(out)
        .map_err(|error| format!("value read failed for '{}': {error}", case.name))?;

    let backward = session
        .backward(out)
        .map_err(|error| format!("backward failed for '{}': {error}", case.name))?;

    let actual_lhs_grad = session
        .gradient(&backward, lhs)
        .ok_or_else(|| format!("missing lhs grad for '{}'", case.name))?;
    let actual_rhs_grad = session
        .gradient(&backward, rhs)
        .ok_or_else(|| format!("missing rhs grad for '{}'", case.name))?;

    Ok(ScalarObservation {
        output: actual_output,
        lhs_grad: actual_lhs_grad,
        rhs_grad: actual_rhs_grad,
    })
}

fn evaluate_dispatch_output(case: &DispatchCase, mode: ExecutionMode) -> Result<f64, String> {
    let op = parse_binary_op(&case.op)?;
    let lhs = ScalarTensor::new(case.lhs, DType::F64, Device::Cpu);
    let rhs = ScalarTensor::new(case.rhs, DType::F64, Device::Cpu);

    let outcome = if let Some(keys) = &case.keyset {
        let keyset = parse_keyset(keys)?;
        dispatch_scalar_binary_with_keyset(op, mode, &lhs, &rhs, keyset)
            .map_err(|error| format!("dispatch case '{}' failed: {error}", case.name))?
    } else {
        dispatch_scalar_binary(op, mode, &lhs, &rhs, case.requires_grad)
            .map_err(|error| format!("dispatch case '{}' failed: {error}", case.name))?
    };

    Ok(outcome.tensor.value())
}

fn evaluate_scheduler_with_tape(
    case: &SchedulerCase,
    mode: ExecutionMode,
) -> Result<LegacyUnaryGradObservation, String> {
    let mut tape = Tape::new();
    let x = tape.leaf(case.x, true);
    let y = tape.leaf(case.y, true);
    let (sum, _) = tape
        .add(x, y, mode)
        .map_err(|error| format!("scheduler case '{}' add failed: {error}", case.name))?;
    let (out, _) = tape
        .mul(sum, x, mode)
        .map_err(|error| format!("scheduler case '{}' mul failed: {error}", case.name))?;

    let report = tape
        .backward_with_options(out, BackwardOptions::for_mode(mode))
        .map_err(|error| format!("scheduler case '{}' backward failed: {error}", case.name))?;

    let x_grad = report
        .gradient(x)
        .ok_or_else(|| format!("missing x grad for '{}'", case.name))?;
    let y_grad = report
        .gradient(y)
        .ok_or_else(|| format!("missing y grad for '{}'", case.name))?;

    let reentrant_guard_triggered = match mode {
        ExecutionMode::Strict => false,
        ExecutionMode::Hardened => tape
            .backward_with_options(
                out,
                BackwardOptions {
                    max_reentrant_depth: 1,
                    current_reentrant_depth: 2,
                    policy: ReentrantPolicy::HardenedBoundedFallback,
                },
            )
            .map(|overflow_report| overflow_report.telemetry.reentrant_guard_triggered)
            .unwrap_or(false),
    };

    Ok(LegacyUnaryGradObservation {
        output: case.x.mul_add(case.x, case.x * case.y),
        x_grad,
        y_grad,
        reentrant_guard_triggered,
    })
}

fn query_legacy_scalar_oracle(
    config: &HarnessConfig,
    case: &ScalarCase,
) -> Result<ScalarObservation, String> {
    let payload = json!({
        "op": case.op,
        "lhs": case.lhs,
        "rhs": case.rhs
    });
    let value = run_legacy_oracle_script(config, LEGACY_SCALAR_ORACLE_SCRIPT, &payload)?;
    let output = value
        .get("output")
        .and_then(Value::as_f64)
        .ok_or_else(|| "legacy scalar oracle response missing output".to_string())?;
    let lhs_grad = value
        .get("lhs_grad")
        .and_then(Value::as_f64)
        .ok_or_else(|| "legacy scalar oracle response missing lhs_grad".to_string())?;
    let rhs_grad = value
        .get("rhs_grad")
        .and_then(Value::as_f64)
        .ok_or_else(|| "legacy scalar oracle response missing rhs_grad".to_string())?;

    Ok(ScalarObservation {
        output,
        lhs_grad,
        rhs_grad,
    })
}

fn query_legacy_tensor_meta_oracle(
    config: &HarnessConfig,
    case: &TensorMetaCase,
) -> Result<TensorMetaObservation, String> {
    if !can_execute_tensor_meta_oracle_case(case) {
        return Err(format!(
            "legacy tensor-meta oracle guarded: required elements exceeded {}",
            LEGACY_TENSOR_META_ORACLE_MAX_ELEMENTS
        ));
    }

    let payload = json!({
        "shape": case.shape,
        "strides": case.strides,
        "storage_offset": case.storage_offset.unwrap_or(0),
        "index": case.index,
    });
    let value = run_legacy_oracle_script(config, LEGACY_TENSOR_META_ORACLE_SCRIPT, &payload)?;
    let numel_raw = value
        .get("numel")
        .and_then(Value::as_u64)
        .ok_or_else(|| "legacy tensor-meta oracle response missing numel".to_string())?;
    let numel = usize::try_from(numel_raw)
        .map_err(|_| format!("legacy tensor-meta numel out of range: {numel_raw}"))?;

    let contiguous = value
        .get("contiguous")
        .and_then(Value::as_bool)
        .ok_or_else(|| "legacy tensor-meta oracle response missing contiguous".to_string())?;

    let linear_raw = value
        .get("linear_index")
        .and_then(Value::as_u64)
        .ok_or_else(|| "legacy tensor-meta oracle response missing linear_index".to_string())?;
    let linear_index = usize::try_from(linear_raw)
        .map_err(|_| format!("legacy tensor-meta linear_index out of range: {linear_raw}"))?;

    Ok(TensorMetaObservation {
        valid: true,
        numel: Some(numel),
        contiguous: Some(contiguous),
        linear_index: Some(linear_index),
        alias_ok: true,
    })
}

fn query_legacy_scheduler_oracle(
    config: &HarnessConfig,
    case: &SchedulerCase,
) -> Result<LegacyUnaryGradObservation, String> {
    let payload = json!({
        "x": case.x,
        "y": case.y
    });
    let value = run_legacy_oracle_script(config, LEGACY_SCHEDULER_ORACLE_SCRIPT, &payload)?;
    let output = value
        .get("output")
        .and_then(Value::as_f64)
        .ok_or_else(|| "legacy scheduler oracle response missing output".to_string())?;
    let x_grad = value
        .get("x_grad")
        .and_then(Value::as_f64)
        .ok_or_else(|| "legacy scheduler oracle response missing x_grad".to_string())?;
    let y_grad = value
        .get("y_grad")
        .and_then(Value::as_f64)
        .ok_or_else(|| "legacy scheduler oracle response missing y_grad".to_string())?;

    Ok(LegacyUnaryGradObservation {
        output,
        x_grad,
        y_grad,
        reentrant_guard_triggered: false,
    })
}

fn run_legacy_oracle_script(
    config: &HarnessConfig,
    script: &str,
    payload: &Value,
) -> Result<Value, String> {
    let python = config
        .legacy_oracle_python
        .clone()
        .unwrap_or_else(|| PathBuf::from("python3"));

    let mut child = Command::new(&python)
        .arg("-c")
        .arg(script)
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .map_err(|error| {
            format!(
                "failed to spawn legacy oracle via {}: {error}",
                python.display()
            )
        })?;

    if let Some(mut stdin) = child.stdin.take() {
        let body = serde_json::to_vec(payload)
            .map_err(|error| format!("failed to serialize oracle payload: {error}"))?;
        stdin
            .write_all(body.as_slice())
            .map_err(|error| format!("failed writing oracle stdin payload: {error}"))?;
    }

    let output = child
        .wait_with_output()
        .map_err(|error| format!("legacy oracle process wait failed: {error}"))?;
    if !output.status.success() {
        return Err(format!(
            "legacy oracle exited with status {}: {}",
            output.status,
            String::from_utf8_lossy(&output.stderr).trim()
        ));
    }

    let stdout = String::from_utf8(output.stdout)
        .map_err(|error| format!("legacy oracle stdout was not utf8: {error}"))?;
    let line = stdout
        .lines()
        .rev()
        .find(|candidate| !candidate.trim().is_empty())
        .ok_or_else(|| "legacy oracle produced empty stdout".to_string())?;

    serde_json::from_str::<Value>(line)
        .map_err(|error| format!("legacy oracle output parse failure: {error}; raw={line}"))
}

fn probe_legacy_oracle(config: &HarnessConfig) -> LegacyOracleStatus {
    let configured_python = config
        .legacy_oracle_python
        .as_ref()
        .map(|path| path.display().to_string());

    match run_legacy_oracle_script(config, LEGACY_ORACLE_PROBE_SCRIPT, &json!({"probe": true})) {
        Ok(value) => {
            let version = value
                .get("torch_version")
                .and_then(Value::as_str)
                .unwrap_or("unknown");
            let python = config
                .legacy_oracle_python
                .as_ref()
                .map(|path| path.display().to_string());
            LegacyOracleStatus {
                configured_python: configured_python.clone(),
                active_python: python,
                available: true,
                message: format!("torch_available:{version}"),
            }
        }
        Err(reason) => LegacyOracleStatus {
            configured_python,
            active_python: None,
            available: false,
            message: reason,
        },
    }
}

fn load_allowlist(path: &Path) -> Result<AllowlistIndex, String> {
    let value: Value = load_fixture(path)?;
    let packets = value
        .get("packets")
        .and_then(Value::as_object)
        .ok_or_else(|| format!("allowlist {} missing packets object", path.display()))?;

    let mut by_packet: BTreeMap<String, BTreeSet<String>> = BTreeMap::new();
    for (packet_id, packet_value) in packets {
        let mut ids = BTreeSet::new();
        if let Some(deviations) = packet_value
            .get("allowed_deviations")
            .and_then(Value::as_array)
        {
            for deviation in deviations {
                if let Some(id) = deviation.get("id").and_then(Value::as_str) {
                    ids.insert(id.to_string());
                }
            }
        }
        by_packet.insert(packet_id.clone(), ids);
    }

    Ok(AllowlistIndex { by_packet })
}

impl AllowlistIndex {
    fn contains(&self, packet_id: &str, drift_id: &str) -> bool {
        self.by_packet
            .get(packet_id)
            .is_some_and(|ids| ids.contains(drift_id))
    }
}

fn scenario_id(suite: &str, mode: ExecutionMode, case_name: &str) -> String {
    format!(
        "{suite}/{}:{}",
        mode_label(mode),
        canonical_case_name(case_name)
    )
}

fn canonical_case_name(case_name: &str) -> String {
    let mut out = String::with_capacity(case_name.len());
    for ch in case_name.chars() {
        if ch.is_ascii_alphanumeric() || ch == '-' || ch == '_' {
            out.push(ch.to_ascii_lowercase());
        } else {
            out.push('_');
        }
    }
    out
}

fn default_oracle_python(repo_root: &Path) -> Option<PathBuf> {
    if let Ok(path) = std::env::var("FT_LEGACY_ORACLE_PYTHON") {
        return Some(PathBuf::from(path));
    }
    let py314 = repo_root.join(".venv-py314/bin/python");
    if py314.exists() {
        return Some(py314);
    }
    Some(PathBuf::from("python3"))
}

const LEGACY_ORACLE_PROBE_SCRIPT: &str = r#"
import json
import torch
print(json.dumps({"torch_version": torch.__version__}, sort_keys=True))
"#;

const LEGACY_SCALAR_ORACLE_SCRIPT: &str = r#"
import json
import sys
import torch
payload = json.loads(sys.stdin.read())
lhs = torch.tensor(payload["lhs"], dtype=torch.float64, requires_grad=True)
rhs = torch.tensor(payload["rhs"], dtype=torch.float64, requires_grad=True)
op = payload["op"]
if op == "add":
    out = lhs + rhs
elif op == "mul":
    out = lhs * rhs
else:
    raise RuntimeError(f"unsupported op {op}")
out.backward()
print(json.dumps({
    "output": float(out.item()),
    "lhs_grad": float(lhs.grad.item()),
    "rhs_grad": float(rhs.grad.item())
}, sort_keys=True))
"#;

const LEGACY_TENSOR_META_ORACLE_SCRIPT: &str = r#"
import json
import sys
import torch

payload = json.loads(sys.stdin.read())
shape = [int(v) for v in payload["shape"]]
strides_payload = payload.get("strides")
storage_offset = int(payload.get("storage_offset", 0))
index = tuple(int(v) for v in payload["index"])

if strides_payload is None:
    strides = []
    running = 1
    for size in reversed(shape):
        strides.insert(0, running)
        running *= int(size)
else:
    strides = [int(v) for v in strides_payload]

max_linear = storage_offset
for size, stride in zip(shape, strides):
    if size > 0:
        max_linear += (size - 1) * stride
required = max_linear + 1

base = torch.arange(required, dtype=torch.float64)
view = torch.as_strided(base, size=tuple(shape), stride=tuple(strides), storage_offset=storage_offset)
linear_index = int(view[index].item())

print(json.dumps({
    "numel": int(view.numel()),
    "contiguous": bool(view.is_contiguous()),
    "linear_index": linear_index
}, sort_keys=True))
"#;

const LEGACY_SCHEDULER_ORACLE_SCRIPT: &str = r#"
import json
import sys
import torch
payload = json.loads(sys.stdin.read())
x = torch.tensor(payload["x"], dtype=torch.float64, requires_grad=True)
y = torch.tensor(payload["y"], dtype=torch.float64, requires_grad=True)
out = (x + y) * x
out.backward()
print(json.dumps({
    "output": float(out.item()),
    "x_grad": float(x.grad.item()),
    "y_grad": float(y.grad.item())
}, sort_keys=True))
"#;

fn parse_binary_op(op: &str) -> Result<BinaryOp, String> {
    match op {
        "add" => Ok(BinaryOp::Add),
        "mul" => Ok(BinaryOp::Mul),
        _ => Err(format!("unsupported binary op '{op}'")),
    }
}

fn parse_dispatch_key(raw: &str) -> Option<DispatchKey> {
    match raw {
        "Undefined" => Some(DispatchKey::Undefined),
        "BackendSelect" => Some(DispatchKey::BackendSelect),
        "CompositeImplicitAutograd" => Some(DispatchKey::CompositeImplicitAutograd),
        "CompositeExplicitAutograd" => Some(DispatchKey::CompositeExplicitAutograd),
        "CPU" => Some(DispatchKey::CPU),
        "AutogradCPU" => Some(DispatchKey::AutogradCPU),
        _ => None,
    }
}

fn parse_keyset(keys: &[String]) -> Result<DispatchKeySet, String> {
    let mut parsed = Vec::with_capacity(keys.len());
    for key in keys {
        let parsed_key =
            parse_dispatch_key(key).ok_or_else(|| format!("unknown dispatch key '{key}'"))?;
        parsed.push(parsed_key);
    }
    Ok(DispatchKeySet::from_keys(parsed.as_slice()))
}

fn load_fixture<T>(path: &Path) -> Result<T, String>
where
    T: for<'de> Deserialize<'de>,
{
    let raw = fs::read_to_string(path)
        .map_err(|error| format!("failed reading fixture {}: {error}", path.display()))?;
    serde_json::from_str::<T>(&raw)
        .map_err(|error| format!("failed parsing fixture {}: {error}", path.display()))
}

fn summarize_passes<I>(iter: I) -> (usize, usize)
where
    I: Iterator<Item = bool>,
{
    let mut total = 0usize;
    let mut passed = 0usize;
    for is_passed in iter {
        total += 1;
        if is_passed {
            passed += 1;
        }
    }
    (total, passed)
}

fn within(actual: f64, expected: f64, tolerance: f64) -> bool {
    (actual - expected).abs() <= tolerance
}

fn percentile(samples: &[u128], p: usize) -> u128 {
    if samples.is_empty() {
        return 0;
    }
    let clamped = p.min(100);
    let idx = ((samples.len() - 1) * clamped) / 100;
    samples[idx]
}

#[cfg(test)]
mod tests {
    use std::fs;
    use std::time::{SystemTime, UNIX_EPOCH};

    use super::{
        ExecutionMode, HarnessConfig, emit_differential_report, emit_e2e_forensics_matrix,
        emit_e2e_forensics_matrix_filtered, load_allowlist, run_autograd_scheduler_conformance,
        run_differential_conformance, run_dispatch_conformance, run_scalar_conformance,
        run_scalar_microbench, run_serialization_conformance, run_smoke,
        run_tensor_meta_conformance,
    };

    #[test]
    fn smoke_harness_finds_oracle_and_fixtures() {
        let cfg = HarnessConfig::default_paths();
        let report = run_smoke(&cfg);
        assert!(report.oracle_present, "oracle repo should be present");
        assert!(report.fixture_count >= 1, "expected at least one fixture");
        assert!(report.strict_mode);
        assert!(
            report.cases_total >= 4,
            "expected at least one case from each fixture family"
        );
    }

    #[test]
    fn strict_scalar_conformance_is_green() {
        let cfg = HarnessConfig::default_paths();
        let (report, case_reports) = run_scalar_conformance(&cfg, ExecutionMode::Strict)
            .expect("strict conformance should run");

        assert_eq!(report.cases_total, case_reports.len());
        assert_eq!(report.cases_total, report.cases_passed);
    }

    #[test]
    fn strict_dispatch_conformance_is_green() {
        let cfg = HarnessConfig::default_paths();
        let (report, cases) =
            run_dispatch_conformance(&cfg, ExecutionMode::Strict).expect("dispatch should run");

        assert_eq!(report.cases_total, cases.len());
        assert_eq!(report.cases_total, report.cases_passed);
    }

    #[test]
    fn strict_tensor_meta_conformance_is_green() {
        let cfg = HarnessConfig::default_paths();
        let (report, cases) = run_tensor_meta_conformance(&cfg, ExecutionMode::Strict)
            .expect("tensor-meta should run");

        assert_eq!(report.cases_total, cases.len());
        assert_eq!(report.cases_total, report.cases_passed);
    }

    #[test]
    fn hardened_tensor_meta_conformance_is_green() {
        let cfg = HarnessConfig::default_paths();
        let (report, _) = run_tensor_meta_conformance(&cfg, ExecutionMode::Hardened)
            .expect("tensor-meta should run");

        assert_eq!(report.cases_total, report.cases_passed);
    }

    #[test]
    fn hardened_dispatch_conformance_is_green() {
        let cfg = HarnessConfig::default_paths();
        let (report, _) =
            run_dispatch_conformance(&cfg, ExecutionMode::Hardened).expect("dispatch should run");

        assert_eq!(report.cases_total, report.cases_passed);
    }

    #[test]
    fn strict_scheduler_conformance_is_green() {
        let cfg = HarnessConfig::default_paths();
        let (report, _) = run_autograd_scheduler_conformance(&cfg, ExecutionMode::Strict)
            .expect("scheduler conformance should run");

        assert_eq!(report.cases_total, report.cases_passed);
    }

    #[test]
    fn hardened_scheduler_conformance_is_green() {
        let cfg = HarnessConfig::default_paths();
        let (report, _) = run_autograd_scheduler_conformance(&cfg, ExecutionMode::Hardened)
            .expect("scheduler conformance should run");

        assert_eq!(report.cases_total, report.cases_passed);
    }

    #[test]
    fn strict_serialization_conformance_is_green() {
        let cfg = HarnessConfig::default_paths();
        let (report, _) = run_serialization_conformance(&cfg, ExecutionMode::Strict)
            .expect("serialization conformance should run");

        assert_eq!(report.cases_total, report.cases_passed);
    }

    #[test]
    fn hardened_serialization_conformance_is_green() {
        let cfg = HarnessConfig::default_paths();
        let (report, _) = run_serialization_conformance(&cfg, ExecutionMode::Hardened)
            .expect("serialization conformance should run");

        assert_eq!(report.cases_total, report.cases_passed);
    }

    #[test]
    fn structured_logs_include_replay_contract_fields() {
        let cfg = HarnessConfig::default_paths();
        let (_, cases) = run_scalar_conformance(&cfg, ExecutionMode::Strict)
            .expect("strict conformance should run");

        assert!(!cases.is_empty(), "expected at least one scalar case");
        let log = &cases[0].forensic_log;
        assert_eq!(log.schema_version, "ft-conformance-log-v1");
        assert!(!log.scenario_id.is_empty());
        assert!(log.seed > 0);
        assert_eq!(log.mode, "strict");
        assert!(log.env_fingerprint.starts_with("det64:"));
        assert!(!log.replay_command.is_empty());
        assert!(!log.artifact_refs.is_empty());
    }

    #[test]
    fn e2e_matrix_writer_emits_jsonl() {
        let cfg = HarnessConfig::default_paths();
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map_or(0, |duration| duration.as_millis());
        let output_path = std::env::temp_dir().join(format!(
            "ft_conformance_e2e_{}_{}.jsonl",
            std::process::id(),
            now
        ));

        let summary = emit_e2e_forensics_matrix(
            &cfg,
            output_path.as_path(),
            &[ExecutionMode::Strict, ExecutionMode::Hardened],
        )
        .expect("e2e matrix should emit logs");

        assert!(summary.log_entries >= 8);
        let raw = fs::read_to_string(&output_path).expect("jsonl output should be readable");
        let mut lines = raw.lines();
        let first = lines
            .next()
            .expect("jsonl should contain at least one line");
        let value: serde_json::Value =
            serde_json::from_str(first).expect("jsonl line should be valid json");

        for required in [
            "scenario_id",
            "seed",
            "mode",
            "env_fingerprint",
            "artifact_refs",
            "replay_command",
            "outcome",
        ] {
            assert!(
                value.get(required).is_some(),
                "missing required key {required}"
            );
        }

        let _ = fs::remove_file(output_path);
    }

    #[test]
    fn e2e_matrix_packet_filter_limits_output() {
        let cfg = HarnessConfig::default_paths();
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map_or(0, |duration| duration.as_millis());
        let output_path = std::env::temp_dir().join(format!(
            "ft_conformance_e2e_packet_filter_{}_{}.jsonl",
            std::process::id(),
            now
        ));

        let summary = emit_e2e_forensics_matrix_filtered(
            &cfg,
            output_path.as_path(),
            &[ExecutionMode::Strict, ExecutionMode::Hardened],
            Some("FT-P2C-004"),
        )
        .expect("packet-filtered e2e matrix should emit logs");

        assert!(summary.log_entries > 0);
        let raw = fs::read_to_string(&output_path).expect("jsonl output should be readable");
        for line in raw.lines() {
            let value: serde_json::Value =
                serde_json::from_str(line).expect("jsonl line should be valid json");
            assert_eq!(
                value.get("packet_id").and_then(serde_json::Value::as_str),
                Some("FT-P2C-004")
            );
        }

        let _ = fs::remove_file(output_path);
    }

    #[test]
    fn microbench_produces_percentiles() {
        let report = run_scalar_microbench(10, ExecutionMode::Strict);
        eprintln!(
            "microbench_ns p50={} p95={} p99={} mean={}",
            report.p50_ns, report.p95_ns, report.p99_ns, report.mean_ns
        );
        assert_eq!(report.iterations, 10);
        assert!(report.p50_ns > 0);
        assert!(report.p95_ns >= report.p50_ns);
        assert!(report.p99_ns >= report.p95_ns);
    }

    #[test]
    fn differential_harness_emits_sorted_checks() {
        let cfg = HarnessConfig::default_paths();
        let report =
            run_differential_conformance(&cfg, &[ExecutionMode::Strict, ExecutionMode::Hardened])
                .expect("differential report should run");

        assert!(report.total_checks > 0);
        assert_eq!(report.total_checks, report.checks.len());
        for window in report.checks.windows(2) {
            let left = &window[0];
            let right = &window[1];
            assert!(
                (
                    left.packet_id,
                    left.suite,
                    left.mode,
                    left.case_name.as_str(),
                    left.comparator
                ) <= (
                    right.packet_id,
                    right.suite,
                    right.mode,
                    right.case_name.as_str(),
                    right.comparator
                ),
                "differential checks should be sorted for deterministic diffs"
            );
        }
    }

    #[test]
    fn differential_tensor_meta_adds_metamorphic_and_adversarial_checks() {
        let cfg = HarnessConfig::default_paths();
        let report = run_differential_conformance(&cfg, &[ExecutionMode::Strict])
            .expect("differential report should run");

        assert!(report.checks.iter().any(|check| {
            check.suite == "tensor_meta"
                && check.case_name == "contiguous_basic_index"
                && check.comparator == "metamorphic_offset_shift_linear_local"
        }));
        assert!(report.checks.iter().any(|check| {
            check.suite == "tensor_meta"
                && check.case_name == "invalid_rank_stride_mismatch"
                && check.comparator == "fail_closed"
        }));

        if report.oracle.available {
            assert!(report.checks.iter().any(|check| {
                check.suite == "tensor_meta"
                    && check.case_name == "invalid_rank_stride_mismatch"
                    && check.comparator == "fail_closed_oracle"
            }));
        }
    }

    #[test]
    fn differential_report_writer_emits_json() {
        let cfg = HarnessConfig::default_paths();
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map_or(0, |duration| duration.as_millis());
        let output_path = std::env::temp_dir().join(format!(
            "ft_conformance_differential_{}_{}.json",
            std::process::id(),
            now
        ));

        let report =
            emit_differential_report(&cfg, output_path.as_path(), &[ExecutionMode::Strict])
                .expect("differential report should emit");
        assert!(report.total_checks > 0);

        let raw = fs::read_to_string(&output_path).expect("json output should be readable");
        let value: serde_json::Value =
            serde_json::from_str(&raw).expect("output should be valid json");
        for key in [
            "schema_version",
            "oracle",
            "modes",
            "total_checks",
            "failed_checks",
            "checks",
        ] {
            assert!(value.get(key).is_some(), "missing required key {key}");
        }

        let _ = fs::remove_file(output_path);
    }

    #[test]
    fn allowlist_contains_known_hardened_deviations() {
        let cfg = HarnessConfig::default_paths();
        let allowlist =
            load_allowlist(cfg.allowlist_path.as_path()).expect("allowlist fixture should parse");
        assert!(allowlist.contains("FT-P2C-002", "dispatch.composite_backend_fallback"));
        assert!(allowlist.contains("FT-P2C-004", "autograd.reentrant_depth_bounded_fallback"));
    }
}

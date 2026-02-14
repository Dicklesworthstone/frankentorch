#![forbid(unsafe_code)]

mod logging;

use std::fs;
use std::path::{Path, PathBuf};
use std::time::Instant;

use ft_api::FrankenTorchSession;
use ft_autograd::{AutogradError, BackwardOptions, ReentrantPolicy, Tape};
use ft_core::{DType, Device, ExecutionMode, ScalarTensor};
use ft_dispatch::{
    BinaryOp, DispatchKey, DispatchKeySet, dispatch_scalar_binary,
    dispatch_scalar_binary_with_keyset,
};
use ft_serialize::{
    CheckpointMode, DecodeMode, SnapshotEntry as SerializedSnapshotEntry, decode_checkpoint,
    encode_checkpoint, generate_raptorq_sidecar,
};
use logging::{StructuredCaseLog, mode_label};
use serde::Deserialize;

#[derive(Debug, Clone)]
pub struct HarnessConfig {
    pub oracle_root: PathBuf,
    pub fixture_root: PathBuf,
    pub strict_mode: bool,
}

impl HarnessConfig {
    #[must_use]
    pub fn default_paths() -> Self {
        let repo_root = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..");
        Self {
            oracle_root: repo_root.join("legacy_pytorch_code/pytorch"),
            fixture_root: PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("fixtures"),
            strict_mode: true,
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

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct E2EForensicsSummary {
    pub output_path: PathBuf,
    pub log_entries: usize,
    pub failed_entries: usize,
    pub modes: Vec<ExecutionMode>,
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
        cases_total: scalar_total + dispatch_total + scheduler_total + serialization_total,
        cases_passed: scalar_passed + dispatch_passed + scheduler_passed + serialization_passed,
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
        ExecutionMode, HarnessConfig, emit_e2e_forensics_matrix,
        emit_e2e_forensics_matrix_filtered, run_autograd_scheduler_conformance,
        run_dispatch_conformance, run_scalar_conformance, run_scalar_microbench,
        run_serialization_conformance, run_smoke,
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
}

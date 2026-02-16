#![forbid(unsafe_code)]

mod logging;

use std::collections::{BTreeMap, BTreeSet};
use std::fs;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::sync::{Mutex, OnceLock};
use std::time::Instant;

use ft_api::FrankenTorchSession;
use ft_autograd::{AutogradError, BackwardOptions, ReentrantPolicy, SchedulerTelemetry, Tape};
use ft_core::{DType, Device, ExecutionMode, ScalarTensor, TensorMeta, contiguous_strides};
use ft_dispatch::{
    BinaryOp, DispatchKey, DispatchKeySet, ParsedSchemaInput, dispatch_scalar_binary,
    dispatch_scalar_binary_with_keyset, parse_schema_or_name, schema_dispatch_keyset_from_tags,
};
use ft_serialize::{
    CheckpointMode, DecodeMode, DecodeProofArtifact, RaptorQSidecar,
    SnapshotEntry as SerializedSnapshotEntry, decode_checkpoint, encode_checkpoint,
    generate_raptorq_sidecar,
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
pub struct OpSchemaCaseReport {
    pub name: String,
    pub mode: ExecutionMode,
    pub parse_ok: bool,
    pub schema_variant_ok: bool,
    pub out_variant_ok: bool,
    pub dispatch_ok: bool,
    pub normalization_ok: bool,
    pub forensic_log: StructuredCaseLog,
}

impl OpSchemaCaseReport {
    #[must_use]
    pub fn passed(&self) -> bool {
        self.parse_ok
            && self.schema_variant_ok
            && self.out_variant_ok
            && self.dispatch_ok
            && self.normalization_ok
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct NnStateCaseReport {
    pub name: String,
    pub mode: ExecutionMode,
    pub contract_ok: bool,
    pub expectation_ok: bool,
    pub detail_ok: bool,
    pub forensic_log: StructuredCaseLog,
}

impl NnStateCaseReport {
    #[must_use]
    pub fn passed(&self) -> bool {
        self.expectation_ok && self.detail_ok
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
    #[serde(default)]
    contract_ids: Vec<String>,
    #[serde(default)]
    e2e_scenarios: Vec<String>,
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
    #[serde(default)]
    contract_ids: Vec<String>,
    #[serde(default)]
    e2e_scenarios: Vec<String>,
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
    lhs_dtype: Option<String>,
    rhs_dtype: Option<String>,
    lhs_device: Option<String>,
    rhs_device: Option<String>,
    requires_grad: bool,
    keyset: Option<Vec<String>>,
    strict: DispatchModeExpectation,
    hardened: DispatchModeExpectation,
    tolerance: Option<f64>,
    #[serde(default)]
    contract_ids: Vec<String>,
    #[serde(default)]
    e2e_scenarios: Vec<String>,
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

#[derive(Debug, Clone, Deserialize)]
struct OpSchemaFixtureFile {
    cases: Vec<OpSchemaCase>,
}

#[derive(Debug, Clone, Deserialize)]
struct OpSchemaCase {
    name: String,
    schema_input: String,
    name_input: Option<String>,
    dispatch_tags: Option<Vec<String>>,
    expect_parse_ok: bool,
    expect_schema_variant: Option<bool>,
    expect_out_variant: Option<bool>,
    expect_dispatch_ok: Option<bool>,
    expect_name_normalization: Option<bool>,
}

#[derive(Debug, Clone, Deserialize)]
struct NnStateFixtureFile {
    cases: Vec<NnStateCase>,
}

#[derive(Debug, Clone, Deserialize)]
struct NnStateCase {
    name: String,
    operation: String,
    module_path: String,
    state_key: Option<String>,
    state_key_kind: Option<String>,
    #[serde(default)]
    parameter_keys: Vec<String>,
    #[serde(default)]
    persistent_buffer_keys: Vec<String>,
    #[serde(default)]
    non_persistent_buffer_keys: Vec<String>,
    #[serde(default)]
    expected_state_keys: Vec<String>,
    #[serde(default)]
    training_transitions: Vec<bool>,
    initial_training: Option<bool>,
    expected_training_flag: Option<bool>,
    #[serde(default)]
    missing_keys: Vec<String>,
    #[serde(default)]
    unexpected_keys: Vec<String>,
    #[serde(default)]
    incompatible_shapes: Vec<String>,
    #[serde(default)]
    prefix_keys: Vec<String>,
    #[serde(default)]
    expected_canonical_keys: Vec<String>,
    allow_prefix_normalization: Option<bool>,
    #[serde(default)]
    hook_trace: Vec<String>,
    #[serde(default)]
    expected_hook_trace: Vec<String>,
    assign_flag: Option<bool>,
    strict: NnStateModeExpectation,
    hardened: NnStateModeExpectation,
    #[serde(default)]
    contract_ids: Vec<String>,
    #[serde(default)]
    e2e_scenarios: Vec<String>,
}

#[derive(Debug, Clone, Deserialize)]
struct NnStateModeExpectation {
    expect_pass: bool,
    expected_reason_code: Option<String>,
    expect_prefix_normalization_applied: Option<bool>,
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

type SidecarCache = BTreeMap<(String, usize), (RaptorQSidecar, DecodeProofArtifact)>;

static SERIALIZATION_SIDECAR_CACHE: OnceLock<Mutex<SidecarCache>> = OnceLock::new();

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
    let (op_schema_total, op_schema_passed) = run_op_schema_conformance(config, mode)
        .map_or((0, 0), |(_, cases)| {
            summarize_passes(cases.iter().map(OpSchemaCaseReport::passed))
        });
    let (scheduler_total, scheduler_passed) = run_autograd_scheduler_conformance(config, mode)
        .map_or((0, 0), |(_, cases)| {
            summarize_passes(cases.iter().map(SchedulerCaseReport::passed))
        });
    let (serialization_total, serialization_passed) = run_serialization_conformance(config, mode)
        .map_or((0, 0), |(_, cases)| {
            summarize_passes(cases.iter().map(SerializationCaseReport::passed))
        });
    let (nn_state_total, nn_state_passed) = run_nn_state_conformance(config, mode)
        .map_or((0, 0), |(_, cases)| {
            summarize_passes(cases.iter().map(NnStateCaseReport::passed))
        });

    HarnessReport {
        suite: "smoke",
        oracle_present: config.oracle_root.exists(),
        fixture_count,
        strict_mode: config.strict_mode,
        cases_total: scalar_total
            + tensor_meta_total
            + dispatch_total
            + op_schema_total
            + scheduler_total
            + serialization_total
            + nn_state_total,
        cases_passed: scalar_passed
            + tensor_meta_passed
            + dispatch_passed
            + op_schema_passed
            + scheduler_passed
            + serialization_passed
            + nn_state_passed,
    }
}

pub fn run_scalar_conformance(
    config: &HarnessConfig,
    mode: ExecutionMode,
) -> Result<(HarnessReport, Vec<CaseReport>), String> {
    let fixture_path = config.fixture_root.join("scalar_autograd_cases.json");
    let fixture: ScalarFixtureFile = load_fixture(&fixture_path)?;
    run_scalar_conformance_with_fixture(config, mode, &fixture)
}

fn run_scalar_conformance_with_fixture(
    config: &HarnessConfig,
    mode: ExecutionMode,
    fixture: &ScalarFixtureFile,
) -> Result<(HarnessReport, Vec<CaseReport>), String> {
    let mut case_reports = Vec::with_capacity(fixture.cases.len());
    for case in &fixture.cases {
        case_reports.push(run_scalar_case(case, mode)?);
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
    run_tensor_meta_conformance_with_fixture(config, mode, &fixture)
}

fn run_tensor_meta_conformance_with_fixture(
    config: &HarnessConfig,
    mode: ExecutionMode,
    fixture: &TensorMetaFixtureFile,
) -> Result<(HarnessReport, Vec<TensorMetaCaseReport>), String> {
    let mut case_reports = Vec::with_capacity(fixture.cases.len());
    for case in &fixture.cases {
        case_reports.push(run_tensor_meta_case(case, mode)?);
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
    run_dispatch_conformance_with_fixture(config, mode, &fixture)
}

fn run_dispatch_conformance_with_fixture(
    config: &HarnessConfig,
    mode: ExecutionMode,
    fixture: &DispatchFixtureFile,
) -> Result<(HarnessReport, Vec<DispatchCaseReport>), String> {
    let mut case_reports = Vec::with_capacity(fixture.cases.len());
    for case in &fixture.cases {
        case_reports.push(run_dispatch_case(case, mode)?);
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

pub fn run_op_schema_conformance(
    config: &HarnessConfig,
    mode: ExecutionMode,
) -> Result<(HarnessReport, Vec<OpSchemaCaseReport>), String> {
    let fixture_path = config.fixture_root.join("op_schema_cases.json");
    let fixture: OpSchemaFixtureFile = load_fixture(&fixture_path)?;

    let mut case_reports = Vec::with_capacity(fixture.cases.len());
    for case in fixture.cases {
        case_reports.push(run_op_schema_case(&case, mode)?);
    }

    let (cases_total, cases_passed) =
        summarize_passes(case_reports.iter().map(OpSchemaCaseReport::passed));

    let report = HarnessReport {
        suite: "op_schema",
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

pub fn run_nn_state_conformance(
    config: &HarnessConfig,
    mode: ExecutionMode,
) -> Result<(HarnessReport, Vec<NnStateCaseReport>), String> {
    let fixture_path = config.fixture_root.join("nn_state_cases.json");
    let fixture: NnStateFixtureFile = load_fixture(&fixture_path)?;
    run_nn_state_conformance_with_fixture(config, mode, &fixture)
}

fn run_nn_state_conformance_with_fixture(
    config: &HarnessConfig,
    mode: ExecutionMode,
    fixture: &NnStateFixtureFile,
) -> Result<(HarnessReport, Vec<NnStateCaseReport>), String> {
    let mut case_reports = Vec::with_capacity(fixture.cases.len());
    for case in &fixture.cases {
        case_reports.push(run_nn_state_case(case, mode)?);
    }

    let (cases_total, cases_passed) =
        summarize_passes(case_reports.iter().map(NnStateCaseReport::passed));

    let report = HarnessReport {
        suite: "nn_state",
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
    let include_ft_p2c_001 = packet_in_scope(packet_filter, "FT-P2C-001");
    let include_ft_p2c_002 = packet_in_scope(packet_filter, "FT-P2C-002");
    let include_ft_p2c_003 = packet_in_scope(packet_filter, "FT-P2C-003");
    let include_ft_p2c_004 = packet_in_scope(packet_filter, "FT-P2C-004");
    let include_ft_p2c_005 = packet_in_scope(packet_filter, "FT-P2C-005");
    let include_ft_p2c_006 = packet_in_scope(packet_filter, "FT-P2C-006");
    let include_ft_p2c_007 = packet_in_scope(packet_filter, "FT-P2C-007");
    let include_ft_p2c_008 = packet_in_scope(packet_filter, "FT-P2C-008");

    let scalar_fixture = if include_ft_p2c_001 || include_ft_p2c_005 {
        let fixture_path = config.fixture_root.join("scalar_autograd_cases.json");
        Some(load_fixture::<ScalarFixtureFile>(&fixture_path)?)
    } else {
        None
    };
    let tensor_meta_fixture = if include_ft_p2c_001 || include_ft_p2c_005 {
        let fixture_path = config.fixture_root.join("tensor_meta_cases.json");
        Some(load_fixture::<TensorMetaFixtureFile>(&fixture_path)?)
    } else {
        None
    };
    let dispatch_fixture = if include_ft_p2c_002 || include_ft_p2c_005 || include_ft_p2c_007 {
        let fixture_path = config.fixture_root.join("dispatch_key_cases.json");
        Some(load_fixture::<DispatchFixtureFile>(&fixture_path)?)
    } else {
        None
    };
    let nn_state_fixture = if include_ft_p2c_008 {
        let fixture_path = config.fixture_root.join("nn_state_cases.json");
        Some(load_fixture::<NnStateFixtureFile>(&fixture_path)?)
    } else {
        None
    };

    let mut logs = Vec::new();
    for mode in selected_modes.iter().copied() {
        if let Some(fixture) = scalar_fixture.as_ref() {
            let (_, scalar_cases) = run_scalar_conformance_with_fixture(config, mode, fixture)?;
            let scalar_logs = scalar_cases
                .into_iter()
                .map(|case| case.forensic_log)
                .collect::<Vec<_>>();
            extend_ft_p2c_005_projection_logs(
                &mut logs,
                scalar_logs,
                include_ft_p2c_001,
                include_ft_p2c_005,
            );
        }

        if let Some(fixture) = tensor_meta_fixture.as_ref() {
            let (_, tensor_meta_cases) =
                run_tensor_meta_conformance_with_fixture(config, mode, fixture)?;
            let tensor_meta_logs = tensor_meta_cases
                .into_iter()
                .map(|case| case.forensic_log)
                .collect::<Vec<_>>();
            extend_ft_p2c_005_projection_logs(
                &mut logs,
                tensor_meta_logs,
                include_ft_p2c_001,
                include_ft_p2c_005,
            );
        }

        if let Some(fixture) = dispatch_fixture.as_ref() {
            let (_, dispatch_cases) = run_dispatch_conformance_with_fixture(config, mode, fixture)?;
            let dispatch_logs = dispatch_cases
                .into_iter()
                .map(|case| case.forensic_log)
                .collect::<Vec<_>>();
            extend_dispatch_projection_logs(
                &mut logs,
                dispatch_logs,
                include_ft_p2c_002,
                include_ft_p2c_005,
                include_ft_p2c_007,
            );
        }

        if include_ft_p2c_003 {
            let (_, op_schema_cases) = run_op_schema_conformance(config, mode)?;
            logs.extend(op_schema_cases.into_iter().map(|case| case.forensic_log));
        }

        if include_ft_p2c_004 {
            let (_, scheduler_cases) = run_autograd_scheduler_conformance(config, mode)?;
            logs.extend(scheduler_cases.into_iter().map(|case| case.forensic_log));
        }

        if include_ft_p2c_006 {
            let (_, serialization_cases) = run_serialization_conformance(config, mode)?;
            logs.extend(
                serialization_cases
                    .into_iter()
                    .map(|case| case.forensic_log),
            );
        }

        if let Some(fixture) = nn_state_fixture.as_ref() {
            let (_, nn_state_cases) = run_nn_state_conformance_with_fixture(config, mode, fixture)?;
            logs.extend(nn_state_cases.into_iter().map(|case| case.forensic_log));
        }
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

#[cfg(test)]
fn emit_e2e_forensics_matrix_filtered_legacy(
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
        let include_ft_p2c_001 = packet_in_scope(packet_filter, "FT-P2C-001");
        let include_ft_p2c_002 = packet_in_scope(packet_filter, "FT-P2C-002");
        let include_ft_p2c_003 = packet_in_scope(packet_filter, "FT-P2C-003");
        let include_ft_p2c_004 = packet_in_scope(packet_filter, "FT-P2C-004");
        let include_ft_p2c_005 = packet_in_scope(packet_filter, "FT-P2C-005");
        let include_ft_p2c_006 = packet_in_scope(packet_filter, "FT-P2C-006");
        let include_ft_p2c_007 = packet_in_scope(packet_filter, "FT-P2C-007");
        let include_ft_p2c_008 = packet_in_scope(packet_filter, "FT-P2C-008");

        if include_ft_p2c_001 {
            let (_, scalar_cases) = run_scalar_conformance(config, mode)?;
            logs.extend(scalar_cases.into_iter().map(|case| case.forensic_log));

            let (_, tensor_meta_cases) = run_tensor_meta_conformance(config, mode)?;
            logs.extend(tensor_meta_cases.into_iter().map(|case| case.forensic_log));
        }

        if include_ft_p2c_002 || include_ft_p2c_005 || include_ft_p2c_007 {
            let (_, dispatch_cases) = run_dispatch_conformance(config, mode)?;
            let dispatch_logs = dispatch_cases
                .into_iter()
                .map(|case| case.forensic_log)
                .collect::<Vec<_>>();
            extend_dispatch_projection_logs(
                &mut logs,
                dispatch_logs,
                include_ft_p2c_002,
                include_ft_p2c_005,
                include_ft_p2c_007,
            );
        }

        if include_ft_p2c_003 {
            let (_, op_schema_cases) = run_op_schema_conformance(config, mode)?;
            logs.extend(op_schema_cases.into_iter().map(|case| case.forensic_log));
        }

        if include_ft_p2c_004 {
            let (_, scheduler_cases) = run_autograd_scheduler_conformance(config, mode)?;
            logs.extend(scheduler_cases.into_iter().map(|case| case.forensic_log));
        }

        if include_ft_p2c_006 {
            let (_, serialization_cases) = run_serialization_conformance(config, mode)?;
            logs.extend(
                serialization_cases
                    .into_iter()
                    .map(|case| case.forensic_log),
            );
        }

        if include_ft_p2c_008 {
            let (_, nn_state_cases) = run_nn_state_conformance(config, mode)?;
            logs.extend(nn_state_cases.into_iter().map(|case| case.forensic_log));
        }

        if include_ft_p2c_005 {
            let (_, scalar_cases) = run_scalar_conformance(config, mode)?;
            logs.extend(
                scalar_cases
                    .into_iter()
                    .map(|case| project_log_to_ft_p2c_005(case.forensic_log)),
            );

            let (_, tensor_meta_cases) = run_tensor_meta_conformance(config, mode)?;
            logs.extend(
                tensor_meta_cases
                    .into_iter()
                    .map(|case| project_log_to_ft_p2c_005(case.forensic_log)),
            );
        }
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

fn extend_ft_p2c_005_projection_logs(
    logs: &mut Vec<StructuredCaseLog>,
    base_logs: Vec<StructuredCaseLog>,
    include_base_packet: bool,
    include_ft_p2c_005: bool,
) {
    match (include_base_packet, include_ft_p2c_005) {
        (true, true) => {
            logs.extend(base_logs.iter().cloned());
            logs.extend(base_logs.into_iter().map(project_log_to_ft_p2c_005));
        }
        (true, false) => {
            logs.extend(base_logs);
        }
        (false, true) => {
            logs.extend(base_logs.into_iter().map(project_log_to_ft_p2c_005));
        }
        (false, false) => {}
    }
}

fn extend_dispatch_projection_logs(
    logs: &mut Vec<StructuredCaseLog>,
    base_logs: Vec<StructuredCaseLog>,
    include_base_packet: bool,
    include_ft_p2c_005: bool,
    include_ft_p2c_007: bool,
) {
    match (include_base_packet, include_ft_p2c_005, include_ft_p2c_007) {
        (false, false, false) => {}
        _ => {
            for log in base_logs {
                if include_base_packet {
                    logs.push(log.clone());
                }
                if include_ft_p2c_005 {
                    logs.push(project_log_to_ft_p2c_005(log.clone()));
                }
                if include_ft_p2c_007 {
                    logs.push(project_log_to_ft_p2c_007(log));
                }
            }
        }
    }
}

fn packet_in_scope(packet_filter: Option<&str>, packet_id: &str) -> bool {
    packet_filter.is_none_or(|filter| filter == packet_id)
}

fn project_log_to_ft_p2c_005(mut log: StructuredCaseLog) -> StructuredCaseLog {
    let original_packet = log.packet_id;
    log.packet_id = "FT-P2C-005";
    log.scenario_id = format!("ft_p2c_005/{}", log.scenario_id);
    sanitize_projection_extra_fields(&mut log);
    log.replay_command = format!(
        "cargo run -p ft-conformance --bin run_e2e_matrix -- --mode {} --packet FT-P2C-005 --output artifacts/phase2c/e2e_forensics/ft-p2c-005.jsonl",
        log.mode
    );
    log.artifact_refs
        .push("artifacts/phase2c/FT-P2C-005/contract_table.md".to_string());
    log.artifact_refs
        .push("artifacts/phase2c/FT-P2C-005/unit_property_quality_report_v1.json".to_string());
    log.extra_fields.insert(
        "packet_projection".to_string(),
        json!(format!("{original_packet}->FT-P2C-005")),
    );
    log
}

fn project_log_to_ft_p2c_007(mut log: StructuredCaseLog) -> StructuredCaseLog {
    let original_packet = log.packet_id;
    log.packet_id = "FT-P2C-007";
    log.scenario_id = format!("ft_p2c_007/{}", log.scenario_id);
    sanitize_projection_extra_fields(&mut log);
    log.replay_command = format!(
        "cargo run -p ft-conformance --bin run_e2e_matrix -- --mode {} --packet FT-P2C-007 --output artifacts/phase2c/e2e_forensics/ft-p2c-007.jsonl",
        log.mode
    );
    log.artifact_refs
        .push("artifacts/phase2c/FT-P2C-007/contract_table.md".to_string());
    log.artifact_refs
        .push("artifacts/phase2c/FT-P2C-007/unit_property_quality_report_v1.json".to_string());
    log.extra_fields.insert(
        "packet_projection".to_string(),
        json!(format!("{original_packet}->FT-P2C-007")),
    );
    log
}

fn sanitize_projection_extra_fields(log: &mut StructuredCaseLog) {
    // Flattened structured logs must not shadow top-level envelope fields.
    for shadowed_key in [
        "schema_version",
        "ts_unix_ms",
        "suite_id",
        "scenario_id",
        "fixture_id",
        "packet_id",
        "mode",
        "seed",
        "env_fingerprint",
        "artifact_refs",
        "replay_command",
        "outcome",
        "reason_code",
    ] {
        log.extra_fields.remove(shadowed_key);
    }
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
                    contract_ids: case.contract_ids.clone(),
                    e2e_scenarios: case.e2e_scenarios.clone(),
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

            let swapped_case = DispatchCase {
                name: format!("{}__swapped_args", case.name),
                op: case.op.clone(),
                lhs: case.rhs,
                rhs: case.lhs,
                lhs_dtype: case.rhs_dtype.clone(),
                rhs_dtype: case.lhs_dtype.clone(),
                lhs_device: case.rhs_device.clone(),
                rhs_device: case.lhs_device.clone(),
                requires_grad: case.requires_grad,
                keyset: None,
                strict: case.strict.clone(),
                hardened: case.hardened.clone(),
                tolerance: case.tolerance,
                contract_ids: case.contract_ids.clone(),
                e2e_scenarios: case.e2e_scenarios.clone(),
            };
            let swapped_local_output = evaluate_dispatch_output(&swapped_case, mode)?;
            checks.push(compare_abs_tol(
                &allowlist,
                "dispatch_key",
                "FT-P2C-002",
                mode,
                case.name.as_str(),
                "metamorphic_commutative_local",
                "dispatch.metamorphic_commutative_local_mismatch",
                swapped_local_output,
                local_output,
                case.tolerance.unwrap_or(1e-12),
                vec![
                    "crates/ft-conformance/fixtures/dispatch_key_cases.json".to_string(),
                    "artifacts/phase2c/FT-P2C-002/parity_report.json".to_string(),
                ],
            ));
        }

        let malformed_dispatch_key_rejected =
            parse_keyset(["UnknownDispatchKey".to_string()].as_slice()).is_err();
        checks.push(compare_bool(
            &allowlist,
            "dispatch_key",
            "FT-P2C-002",
            mode,
            "adversarial_unknown_key",
            "adversarial_unknown_key_rejected",
            "dispatch.adversarial_unknown_key_accepted",
            malformed_dispatch_key_rejected,
            true,
            vec![
                "crates/ft-conformance/fixtures/dispatch_key_cases.json".to_string(),
                "artifacts/phase2c/FT-P2C-002/parity_report.json".to_string(),
            ],
        ));

        let adversarial_lhs = ScalarTensor::new(1.0, DType::F64, Device::Cpu);
        let adversarial_rhs = ScalarTensor::new(2.0, DType::F64, Device::Cpu);
        let autograd_without_cpu_rejected = dispatch_scalar_binary_with_keyset(
            BinaryOp::Add,
            mode,
            &adversarial_lhs,
            &adversarial_rhs,
            DispatchKeySet::from_keys(&[DispatchKey::AutogradCPU]),
        )
        .is_err();
        checks.push(compare_bool(
            &allowlist,
            "dispatch_key",
            "FT-P2C-002",
            mode,
            "adversarial_autograd_without_cpu",
            "adversarial_autograd_without_cpu_rejected",
            "dispatch.adversarial_autograd_without_cpu_accepted",
            autograd_without_cpu_rejected,
            true,
            vec![
                "crates/ft-conformance/fixtures/dispatch_key_cases.json".to_string(),
                "artifacts/phase2c/FT-P2C-002/parity_report.json".to_string(),
            ],
        ));

        let op_schema_fixture: OpSchemaFixtureFile =
            load_fixture(&config.fixture_root.join("op_schema_cases.json"))?;
        for case in op_schema_fixture.cases {
            let parsed_schema = parse_schema_or_name(case.schema_input.as_str());
            let parse_ok = parsed_schema.is_ok();
            let (parse_comparator, parse_drift_id) = if case.expect_parse_ok {
                ("schema_parse", "op_schema.parse_rejected")
            } else {
                (
                    "adversarial_malformed_schema_rejected",
                    "op_schema.adversarial_malformed_schema_accepted",
                )
            };
            let evidence_refs = op_schema_evidence_refs();
            checks.push(compare_bool(
                &allowlist,
                "op_schema",
                "FT-P2C-003",
                mode,
                case.name.as_str(),
                parse_comparator,
                parse_drift_id,
                parse_ok,
                case.expect_parse_ok,
                evidence_refs.clone(),
            ));

            if let Some(expected_schema_variant) = case.expect_schema_variant {
                let observed_schema_variant =
                    matches!(parsed_schema.as_ref(), Ok(ParsedSchemaInput::Schema(_)));
                checks.push(compare_bool(
                    &allowlist,
                    "op_schema",
                    "FT-P2C-003",
                    mode,
                    case.name.as_str(),
                    "schema_variant_classification",
                    "op_schema.schema_variant_classification_mismatch",
                    observed_schema_variant,
                    expected_schema_variant,
                    evidence_refs.clone(),
                ));
            }

            if let Some(expected_out_variant) = case.expect_out_variant {
                let observed_out_variant = matches!(
                    parsed_schema.as_ref(),
                    Ok(ParsedSchemaInput::Schema(schema)) if schema.is_out_variant
                );
                checks.push(compare_bool(
                    &allowlist,
                    "op_schema",
                    "FT-P2C-003",
                    mode,
                    case.name.as_str(),
                    "out_variant_alias_contract",
                    "op_schema.out_variant_alias_mismatch",
                    observed_out_variant,
                    expected_out_variant,
                    evidence_refs.clone(),
                ));
            }

            if let Some(expected_dispatch_ok) = case.expect_dispatch_ok {
                let observed_dispatch_ok = case.dispatch_tags.as_ref().is_some_and(|tags| {
                    let tag_refs: Vec<&str> = tags.iter().map(String::as_str).collect();
                    schema_dispatch_keyset_from_tags(tag_refs.as_slice()).is_ok()
                });
                let (comparator, drift_id) = if expected_dispatch_ok {
                    (
                        "dispatch_keyset_contract",
                        "op_schema.dispatch_keyset_contract_mismatch",
                    )
                } else {
                    (
                        "adversarial_dispatch_metadata_rejected",
                        "op_schema.adversarial_dispatch_metadata_accepted",
                    )
                };
                checks.push(compare_bool(
                    &allowlist,
                    "op_schema",
                    "FT-P2C-003",
                    mode,
                    case.name.as_str(),
                    comparator,
                    drift_id,
                    observed_dispatch_ok,
                    expected_dispatch_ok,
                    evidence_refs.clone(),
                ));
            }

            if let Some(expected_name_normalization) = case.expect_name_normalization {
                let observed_name_normalization = case.name_input.as_deref().is_some_and(|input| {
                    let lhs_name = match parsed_schema.as_ref() {
                        Ok(ParsedSchemaInput::Schema(schema)) => Some(schema.op.unambiguous_name()),
                        Ok(ParsedSchemaInput::Name(name)) => Some(name.unambiguous_name()),
                        Err(_) => None,
                    };
                    let rhs_name = match parse_schema_or_name(input) {
                        Ok(ParsedSchemaInput::Name(name)) => Some(name.unambiguous_name()),
                        _ => None,
                    };
                    lhs_name
                        .zip(rhs_name)
                        .is_some_and(|(left, right)| left == right)
                });
                checks.push(compare_bool(
                    &allowlist,
                    "op_schema",
                    "FT-P2C-003",
                    mode,
                    case.name.as_str(),
                    "metamorphic_name_normalization",
                    "op_schema.metamorphic_name_normalization_mismatch",
                    observed_name_normalization,
                    expected_name_normalization,
                    evidence_refs,
                ));
            }
        }

        let scheduler_fixture: SchedulerFixtureFile =
            load_fixture(&config.fixture_root.join("autograd_scheduler_cases.json"))?;
        for case in scheduler_fixture.cases {
            let tolerance = case.tolerance.unwrap_or(1e-12);
            let scheduler_evidence_refs = autograd_scheduler_evidence_refs();
            let local = evaluate_scheduler_with_tape(&case, mode)?;
            let oracle = query_legacy_scheduler_oracle(config, &case);
            match oracle.as_ref() {
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
                        tolerance,
                        scheduler_evidence_refs.clone(),
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
                        tolerance,
                        scheduler_evidence_refs.clone(),
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
                        tolerance,
                        scheduler_evidence_refs.clone(),
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
                    observed: reason.clone(),
                    expected: "legacy_oracle_response".to_string(),
                    evidence_refs: vec![
                        "crates/ft-conformance/fixtures/autograd_scheduler_cases.json".to_string(),
                    ],
                }),
            }

            let scaled_case = scaled_scheduler_case(&case, 2.0);
            let scaled_local = evaluate_scheduler_with_tape(&scaled_case, mode)?;
            checks.push(compare_bool(
                &allowlist,
                "autograd_scheduler",
                "FT-P2C-004",
                mode,
                case.name.as_str(),
                "metamorphic_scale_relation_local",
                "autograd.metamorphic_scale_relation_local_mismatch",
                scheduler_scale_relation_holds(&local, &scaled_local, 2.0, tolerance),
                true,
                scheduler_evidence_refs.clone(),
            ));

            if let Ok(base_oracle) = oracle.as_ref() {
                match query_legacy_scheduler_oracle(config, &scaled_case) {
                    Ok(scaled_oracle) => checks.push(compare_bool(
                        &allowlist,
                        "autograd_scheduler",
                        "FT-P2C-004",
                        mode,
                        case.name.as_str(),
                        "metamorphic_scale_relation_oracle",
                        "autograd.metamorphic_scale_relation_oracle_mismatch",
                        scheduler_scale_relation_holds(base_oracle, &scaled_oracle, 2.0, tolerance),
                        true,
                        scheduler_evidence_refs.clone(),
                    )),
                    Err(reason) => checks.push(DifferentialCheck {
                        suite: "autograd_scheduler",
                        packet_id: "FT-P2C-004",
                        scenario_id: scenario_id(
                            "autograd_scheduler",
                            mode,
                            scaled_case.name.as_str(),
                        ),
                        case_name: scaled_case.name.clone(),
                        mode: mode_str,
                        comparator: "metamorphic_oracle.autograd",
                        status: "oracle_unavailable",
                        allowlisted: false,
                        drift_id: None,
                        reason_code: "legacy_oracle_unavailable".to_string(),
                        observed: reason,
                        expected: "legacy_oracle_response".to_string(),
                        evidence_refs: scheduler_evidence_refs.clone(),
                    }),
                }
            }

            if mode == ExecutionMode::Strict {
                checks.push(compare_bool(
                    &allowlist,
                    "autograd_scheduler",
                    "FT-P2C-004",
                    mode,
                    case.name.as_str(),
                    "adversarial_strict_reentrant_overflow_rejected",
                    "autograd.adversarial_strict_reentrant_overflow_accepted",
                    strict_overflow_rejected(&case)?,
                    true,
                    scheduler_evidence_refs.clone(),
                ));
            } else {
                checks.push(compare_bool(
                    &allowlist,
                    "autograd_scheduler",
                    "FT-P2C-004",
                    mode,
                    case.name.as_str(),
                    "adversarial_hardened_reentrant_overflow_guarded",
                    "autograd.adversarial_hardened_reentrant_overflow_unflagged",
                    hardened_overflow_guarded(&case)?,
                    true,
                    scheduler_evidence_refs.clone(),
                ));
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
                    {
                        let mut refs = autograd_scheduler_evidence_refs();
                        refs.push(
                            "artifacts/phase2c/HARDENED_DEVIATION_ALLOWLIST_V1.json".to_string(),
                        );
                        refs
                    },
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
                    evidence_refs: scheduler_evidence_refs,
                });
            }
        }

        let serialization_fixture: SerializationFixtureFile =
            load_fixture(&config.fixture_root.join("serialization_cases.json"))?;
        for case in serialization_fixture.cases {
            let serialization_evidence_refs = serialization_evidence_refs();
            let checkpoint_mode = if mode == ExecutionMode::Strict {
                CheckpointMode::Strict
            } else {
                CheckpointMode::Hardened
            };
            let decode_mode = if mode == ExecutionMode::Strict {
                DecodeMode::Strict
            } else {
                DecodeMode::Hardened
            };

            let entries = serialization_entries(&case);
            let payload = encode_checkpoint(entries.as_slice(), checkpoint_mode);
            let mut expected_entries = entries.clone();
            expected_entries.sort_by_key(|entry| entry.node_id);

            let decode_roundtrip_ok = decode_checkpoint(payload.as_str(), decode_mode)
                .map(|decoded| decoded.entries == expected_entries)
                .unwrap_or(false);
            checks.push(compare_bool(
                &allowlist,
                "serialization",
                "FT-P2C-006",
                mode,
                case.name.as_str(),
                "decode_roundtrip_contract",
                "serialization.decode_roundtrip_mismatch",
                decode_roundtrip_ok,
                true,
                serialization_evidence_refs.clone(),
            ));

            let repair_symbols = case.repair_symbols.unwrap_or(4);
            let sidecar_a =
                serialization_generate_sidecar_with_retry(payload.as_str(), repair_symbols);
            let sidecar_b = serialization_generate_sidecar_with_retry_uncached(
                payload.as_str(),
                repair_symbols,
            );

            let sidecar_integrity_ok = sidecar_a.as_ref().is_ok_and(|(sidecar, _proof)| {
                sidecar.repair_symbol_count >= 1 && sidecar.constraints_symbol_count >= 1
            });
            let proof_deterministic_ok = sidecar_a
                .as_ref()
                .ok()
                .zip(sidecar_b.as_ref().ok())
                .is_some_and(|((_, proof_a), (_, proof_b))| {
                    proof_a.proof_hash == proof_b.proof_hash
                        && proof_a.source_hash == proof_b.source_hash
                });
            checks.push(compare_bool(
                &allowlist,
                "serialization",
                "FT-P2C-006",
                mode,
                case.name.as_str(),
                "proof_determinism_contract",
                "serialization.decode_proof_nondeterministic",
                sidecar_integrity_ok && proof_deterministic_ok,
                true,
                serialization_evidence_refs.clone(),
            ));

            let mut reversed_entries = entries.clone();
            reversed_entries.reverse();
            let reversed_payload = encode_checkpoint(reversed_entries.as_slice(), checkpoint_mode);
            checks.push(compare_bool(
                &allowlist,
                "serialization",
                "FT-P2C-006",
                mode,
                case.name.as_str(),
                "metamorphic_entry_order_hash_invariant",
                "serialization.metamorphic_entry_order_hash_mismatch",
                payload == reversed_payload,
                true,
                serialization_evidence_refs.clone(),
            ));

            let unknown_field_rejected = serde_json::from_str::<Value>(payload.as_str())
                .ok()
                .map(|mut raw| {
                    raw["unknown_field_probe"] = json!(17);
                    decode_checkpoint(raw.to_string().as_str(), decode_mode).is_err()
                })
                .unwrap_or(false);
            checks.push(compare_bool(
                &allowlist,
                "serialization",
                "FT-P2C-006",
                mode,
                case.name.as_str(),
                "adversarial_unknown_field_rejected",
                "serialization.adversarial_unknown_field_accepted",
                unknown_field_rejected,
                true,
                serialization_evidence_refs.clone(),
            ));

            let version_mismatch_rejected = serde_json::from_str::<Value>(payload.as_str())
                .ok()
                .map(|mut raw| {
                    raw["schema_version"] =
                        json!(u64::from(ft_serialize::CHECKPOINT_SCHEMA_VERSION) + 1);
                    decode_checkpoint(raw.to_string().as_str(), decode_mode).is_err()
                })
                .unwrap_or(false);
            checks.push(compare_bool(
                &allowlist,
                "serialization",
                "FT-P2C-006",
                mode,
                case.name.as_str(),
                "adversarial_version_mismatch_rejected",
                "serialization.adversarial_version_mismatch_accepted",
                version_mismatch_rejected,
                true,
                serialization_evidence_refs.clone(),
            ));

            let checksum_tamper_rejected = serde_json::from_str::<Value>(payload.as_str())
                .ok()
                .map(|mut raw| {
                    raw["source_hash"] = json!("det64:0000000000000000");
                    decode_checkpoint(raw.to_string().as_str(), decode_mode).is_err()
                })
                .unwrap_or(false);
            checks.push(compare_bool(
                &allowlist,
                "serialization",
                "FT-P2C-006",
                mode,
                case.name.as_str(),
                "adversarial_checksum_tamper_rejected",
                "serialization.adversarial_checksum_tamper_accepted",
                checksum_tamper_rejected,
                true,
                serialization_evidence_refs.clone(),
            ));

            let malformed_payload = format!("{{ malformed {}", case.name);
            let malformed_error = decode_checkpoint(malformed_payload.as_str(), decode_mode).err();
            checks.push(compare_bool(
                &allowlist,
                "serialization",
                "FT-P2C-006",
                mode,
                case.name.as_str(),
                "adversarial_malformed_json_rejected",
                "serialization.adversarial_malformed_json_accepted",
                malformed_error.is_some(),
                true,
                serialization_evidence_refs.clone(),
            ));

            if mode == ExecutionMode::Hardened {
                let bounded_diagnostic_ok = malformed_error.as_ref().is_some_and(|error| {
                    let message = error.to_string();
                    message.contains("payload_prefix=") && message.len() < 320
                });
                checks.push(compare_bool(
                    &allowlist,
                    "serialization",
                    "FT-P2C-006",
                    mode,
                    case.name.as_str(),
                    "policy",
                    "serialization.bounded_malformed_diagnostic",
                    bounded_diagnostic_ok,
                    false,
                    {
                        let mut refs = serialization_evidence_refs.clone();
                        refs.push(
                            "artifacts/phase2c/HARDENED_DEVIATION_ALLOWLIST_V1.json".to_string(),
                        );
                        refs
                    },
                ));
            } else {
                checks.push(DifferentialCheck {
                    suite: "serialization",
                    packet_id: "FT-P2C-006",
                    scenario_id: scenario_id("serialization", mode, case.name.as_str()),
                    case_name: case.name.clone(),
                    mode: mode_str,
                    comparator: "policy",
                    status: "pass",
                    allowlisted: false,
                    drift_id: None,
                    reason_code: "strict_fail_closed_mode_split".to_string(),
                    observed: "strict_fail_closed".to_string(),
                    expected: "strict_fail_closed".to_string(),
                    evidence_refs: serialization_evidence_refs.clone(),
                });
            }

            let corruption_detected = sidecar_a.as_ref().is_ok_and(|(_sidecar, proof)| {
                let mut corrupted_payload = payload.clone();
                corrupted_payload.push(' ');
                match serialization_generate_sidecar_with_retry(
                    corrupted_payload.as_str(),
                    repair_symbols,
                ) {
                    Ok((_tampered_sidecar, tampered_proof)) => {
                        tampered_proof.source_hash != proof.source_hash
                            || tampered_proof.proof_hash != proof.proof_hash
                    }
                    Err(_) => true,
                }
            });
            checks.push(compare_bool(
                &allowlist,
                "serialization",
                "FT-P2C-006",
                mode,
                case.name.as_str(),
                "adversarial_raptorq_corruption_probe",
                "serialization.adversarial_raptorq_corruption_undetected",
                corruption_detected,
                true,
                serialization_evidence_refs,
            ));
        }

        let nn_state_fixture: NnStateFixtureFile =
            load_fixture(&config.fixture_root.join("nn_state_cases.json"))?;
        for case in nn_state_fixture.cases {
            let nn_state_evidence_refs = nn_state_differential_evidence_refs();
            let local = run_nn_state_case(&case, mode)?;

            checks.push(compare_bool(
                &allowlist,
                "nn_state",
                "FT-P2C-008",
                mode,
                case.name.as_str(),
                "contract_expectation",
                "nn_state.contract_expectation_mismatch",
                local.expectation_ok,
                true,
                nn_state_evidence_refs.clone(),
            ));
            checks.push(compare_bool(
                &allowlist,
                "nn_state",
                "FT-P2C-008",
                mode,
                case.name.as_str(),
                "structured_log_detail",
                "nn_state.structured_log_detail_mismatch",
                local.detail_ok,
                true,
                nn_state_evidence_refs.clone(),
            ));

            if case.operation == "register_parameter" || case.operation == "register_buffer" {
                let registration_rejected =
                    !nn_state_is_valid_key(case.state_key.as_deref().unwrap_or_default());
                checks.push(compare_bool(
                    &allowlist,
                    "nn_state",
                    "FT-P2C-008",
                    mode,
                    case.name.as_str(),
                    "adversarial_invalid_registration_name_rejected",
                    "nn_state.adversarial_invalid_registration_name_accepted",
                    registration_rejected,
                    true,
                    nn_state_evidence_refs.clone(),
                ));
            }

            if case.operation == "state_export" {
                let mut permuted_case = case.clone();
                permuted_case.name = format!("{}__metamorphic_permuted", case.name);
                permuted_case.parameter_keys.reverse();
                permuted_case.persistent_buffer_keys.reverse();
                let permuted = run_nn_state_case(&permuted_case, mode)?;

                let base_keys = log_string_vec_field(&local.forensic_log, "actual_state_keys");
                let permuted_keys =
                    log_string_vec_field(&permuted.forensic_log, "actual_state_keys");
                checks.push(compare_bool(
                    &allowlist,
                    "nn_state",
                    "FT-P2C-008",
                    mode,
                    case.name.as_str(),
                    "metamorphic_state_export_order_invariant",
                    "nn_state.metamorphic_state_export_order_mismatch",
                    base_keys == permuted_keys,
                    true,
                    nn_state_evidence_refs.clone(),
                ));

                let non_persistent_excluded = case
                    .non_persistent_buffer_keys
                    .iter()
                    .all(|key| !base_keys.contains(key));
                checks.push(compare_bool(
                    &allowlist,
                    "nn_state",
                    "FT-P2C-008",
                    mode,
                    case.name.as_str(),
                    "adversarial_non_persistent_buffer_excluded",
                    "nn_state.adversarial_non_persistent_buffer_leak",
                    non_persistent_excluded,
                    true,
                    nn_state_evidence_refs.clone(),
                ));
            }

            if case.operation == "mode_transition" {
                let rerun = run_nn_state_case(&case, mode)?;
                let left_trace =
                    log_bool_vec_field(&local.forensic_log, "training_flag_transition");
                let right_trace =
                    log_bool_vec_field(&rerun.forensic_log, "training_flag_transition");
                checks.push(compare_bool(
                    &allowlist,
                    "nn_state",
                    "FT-P2C-008",
                    mode,
                    case.name.as_str(),
                    "metamorphic_mode_transition_idempotent",
                    "nn_state.metamorphic_mode_transition_idempotence_mismatch",
                    left_trace == right_trace,
                    true,
                    nn_state_evidence_refs.clone(),
                ));
            }

            if case.operation == "load_state" {
                let has_missing_or_unexpected =
                    !case.missing_keys.is_empty() || !case.unexpected_keys.is_empty();
                let has_incompatible = !case.incompatible_shapes.is_empty();
                if has_missing_or_unexpected && !has_incompatible {
                    if mode == ExecutionMode::Hardened {
                        let mut evidence_refs = nn_state_evidence_refs.clone();
                        evidence_refs.push(
                            "artifacts/phase2c/HARDENED_DEVIATION_ALLOWLIST_V1.json".to_string(),
                        );
                        checks.push(compare_bool(
                            &allowlist,
                            "nn_state",
                            "FT-P2C-008",
                            mode,
                            case.name.as_str(),
                            "policy",
                            "nn_state.non_strict_missing_unexpected",
                            local.contract_ok,
                            false,
                            evidence_refs,
                        ));
                    } else {
                        checks.push(DifferentialCheck {
                            suite: "nn_state",
                            packet_id: "FT-P2C-008",
                            scenario_id: scenario_id("nn_state", mode, case.name.as_str()),
                            case_name: case.name.clone(),
                            mode: mode_str,
                            comparator: "policy",
                            status: "pass",
                            allowlisted: false,
                            drift_id: None,
                            reason_code: "strict_fail_closed_mode_split".to_string(),
                            observed: "strict_fail_closed".to_string(),
                            expected: "strict_fail_closed".to_string(),
                            evidence_refs: nn_state_evidence_refs.clone(),
                        });
                    }
                }
                if has_incompatible {
                    checks.push(compare_bool(
                        &allowlist,
                        "nn_state",
                        "FT-P2C-008",
                        mode,
                        case.name.as_str(),
                        "adversarial_incompatible_shape_rejected",
                        "nn_state.adversarial_incompatible_shape_accepted",
                        !local.contract_ok,
                        true,
                        nn_state_evidence_refs.clone(),
                    ));
                }
                if case.assign_flag.unwrap_or(false) {
                    checks.push(compare_bool(
                        &allowlist,
                        "nn_state",
                        "FT-P2C-008",
                        mode,
                        case.name.as_str(),
                        "adversarial_assign_shape_rejected",
                        "nn_state.adversarial_assign_shape_accepted",
                        !local.contract_ok,
                        true,
                        nn_state_evidence_refs.clone(),
                    ));
                }
            }

            if case.operation == "prefix_normalization" {
                let (normalized_once, _) = nn_state_normalize_prefix_keys(
                    case.prefix_keys.as_slice(),
                    case.allow_prefix_normalization.unwrap_or(false),
                );
                let (normalized_twice, applied_twice) =
                    nn_state_normalize_prefix_keys(normalized_once.as_slice(), true);
                checks.push(compare_bool(
                    &allowlist,
                    "nn_state",
                    "FT-P2C-008",
                    mode,
                    case.name.as_str(),
                    "metamorphic_prefix_normalization_idempotent",
                    "nn_state.metamorphic_prefix_normalization_idempotence_mismatch",
                    normalized_once == normalized_twice && !applied_twice,
                    true,
                    nn_state_evidence_refs.clone(),
                ));
            }

            if case.operation == "hook_trace" {
                let hook_trace = log_string_vec_field(&local.forensic_log, "hook_trace");
                checks.push(compare_bool(
                    &allowlist,
                    "nn_state",
                    "FT-P2C-008",
                    mode,
                    case.name.as_str(),
                    "metamorphic_hook_trace_stable",
                    "nn_state.metamorphic_hook_trace_mismatch",
                    hook_trace == case.expected_hook_trace,
                    true,
                    nn_state_evidence_refs,
                ));
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
    emit_differential_report_filtered(config, output_path, modes, None)
}

pub fn emit_differential_report_filtered(
    config: &HarnessConfig,
    output_path: &Path,
    modes: &[ExecutionMode],
    packet_filter: Option<&str>,
) -> Result<DifferentialHarnessReport, String> {
    let report = run_differential_conformance(config, modes)?;
    let report = if let Some(packet_id) = packet_filter {
        project_differential_report_to_packet(report, packet_id)
    } else {
        report
    };
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

fn project_differential_report_to_packet(
    mut report: DifferentialHarnessReport,
    packet_id: &str,
) -> DifferentialHarnessReport {
    report.checks = report
        .checks
        .into_iter()
        .filter_map(|check| project_differential_check_to_packet(check, packet_id))
        .collect();
    report.total_checks = report.checks.len();
    report.allowlisted_drifts = report
        .checks
        .iter()
        .filter(|check| check.status == "allowlisted_drift")
        .count();
    report.blocking_drifts = report
        .checks
        .iter()
        .filter(|check| check.status == "blocking_drift")
        .count();
    report.failed_checks = report
        .checks
        .iter()
        .filter(|check| check.status == "blocking_drift" || check.status == "oracle_unavailable")
        .count();
    report
}

fn project_differential_check_to_packet(
    check: DifferentialCheck,
    packet_id: &str,
) -> Option<DifferentialCheck> {
    match packet_id {
        "FT-P2C-005" => project_differential_check_to_ft_p2c_005(check),
        _ => (check.packet_id == packet_id).then_some(check),
    }
}

fn project_differential_check_to_ft_p2c_005(
    mut check: DifferentialCheck,
) -> Option<DifferentialCheck> {
    if !(matches!(check.packet_id, "FT-P2C-001" | "FT-P2C-002")
        && matches!(check.suite, "scalar_dac" | "tensor_meta" | "dispatch_key"))
    {
        return None;
    }

    let source_packet = check.packet_id;
    check.packet_id = "FT-P2C-005";
    check.scenario_id = format!("ft_p2c_005/{source_packet}/{}", check.scenario_id);
    check
        .evidence_refs
        .extend(ft_p2c_005_differential_evidence_refs());
    check.evidence_refs.sort();
    check.evidence_refs.dedup();
    Some(check)
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

pub fn run_packet_e2e_microbench(
    config: &HarnessConfig,
    iterations: usize,
    packet_id: &str,
) -> Result<BenchReport, String> {
    let mut samples = Vec::with_capacity(iterations.max(1));
    let output_path = std::env::temp_dir().join(format!(
        "ft_conformance_packet_e2e_microbench_{}_{}.jsonl",
        packet_id.to_ascii_lowercase(),
        std::process::id()
    ));

    for _ in 0..iterations.max(1) {
        let started = Instant::now();
        let summary = emit_e2e_forensics_matrix_filtered(
            config,
            output_path.as_path(),
            &[ExecutionMode::Strict, ExecutionMode::Hardened],
            Some(packet_id),
        )?;
        if summary.failed_entries > 0 {
            return Err(format!(
                "packet e2e microbench observed {} failed entries for packet {}",
                summary.failed_entries, packet_id
            ));
        }
        samples.push(started.elapsed().as_nanos());
    }

    let _ = fs::remove_file(output_path);
    samples.sort_unstable();
    let sum = samples.iter().copied().sum::<u128>();
    let mean = sum / (samples.len() as u128);

    Ok(BenchReport {
        iterations: samples.len(),
        p50_ns: percentile(&samples, 50),
        p95_ns: percentile(&samples, 95),
        p99_ns: percentile(&samples, 99),
        mean_ns: mean,
    })
}

#[cfg(test)]
fn run_packet_e2e_microbench_legacy(
    config: &HarnessConfig,
    iterations: usize,
    packet_id: &str,
) -> Result<BenchReport, String> {
    let mut samples = Vec::with_capacity(iterations.max(1));
    let output_path = std::env::temp_dir().join(format!(
        "ft_conformance_packet_e2e_microbench_legacy_{}_{}.jsonl",
        packet_id.to_ascii_lowercase(),
        std::process::id()
    ));

    for _ in 0..iterations.max(1) {
        let started = Instant::now();
        let summary = emit_e2e_forensics_matrix_filtered_legacy(
            config,
            output_path.as_path(),
            &[ExecutionMode::Strict, ExecutionMode::Hardened],
            Some(packet_id),
        )?;
        if summary.failed_entries > 0 {
            return Err(format!(
                "packet e2e legacy microbench observed {} failed entries for packet {}",
                summary.failed_entries, packet_id
            ));
        }
        samples.push(started.elapsed().as_nanos());
    }

    let _ = fs::remove_file(output_path);
    samples.sort_unstable();
    let sum = samples.iter().copied().sum::<u128>();
    let mean = sum / (samples.len() as u128);

    Ok(BenchReport {
        iterations: samples.len(),
        p50_ns: percentile(&samples, 50),
        p95_ns: percentile(&samples, 95),
        p99_ns: percentile(&samples, 99),
        mean_ns: mean,
    })
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
    let reason_code = if outcome == "pass" {
        "parity_ok"
    } else {
        "scalar_or_grad_mismatch"
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
            reason_code,
        )
        .with_extra_fields(scalar_forensic_fields(
            case,
            mode,
            actual_output,
            actual_lhs_grad,
            actual_rhs_grad,
            outcome == "pass",
            reason_code,
        )),
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
        )
        .with_extra_fields(tensor_meta_forensic_fields(case, &local, passed)),
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
    let lhs_dtype = case
        .lhs_dtype
        .as_deref()
        .map_or(Ok(DType::F64), parse_dtype)?;
    let rhs_dtype = case
        .rhs_dtype
        .as_deref()
        .map_or(Ok(DType::F64), parse_dtype)?;
    let lhs_device = case
        .lhs_device
        .as_deref()
        .map_or(Ok(Device::Cpu), parse_device)?;
    let rhs_device = case
        .rhs_device
        .as_deref()
        .map_or(Ok(Device::Cpu), parse_device)?;
    let lhs = ScalarTensor::new(case.lhs, lhs_dtype, lhs_device);
    let rhs = ScalarTensor::new(case.rhs, rhs_dtype, rhs_device);

    let expected_error = expectation.expect_error.unwrap_or(false);
    let result = if let Some(keys) = &case.keyset {
        match parse_keyset(keys) {
            Ok(keyset) => dispatch_scalar_binary_with_keyset(op, mode, &lhs, &rhs, keyset)
                .map_err(|error| error.to_string()),
            Err(error) => Err(error),
        }
    } else {
        dispatch_scalar_binary(op, mode, &lhs, &rhs, case.requires_grad)
            .map_err(|error| error.to_string())
    };

    let tolerance = case.tolerance.unwrap_or(1e-12);

    if expected_error {
        let error_ok = result.is_err();
        let error_message = result.as_ref().err().map(ToString::to_string);
        let reason_code = if error_ok {
            "expected_error_observed"
        } else {
            "expected_error_missing"
        };
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
                reason_code,
            )
            .with_extra_fields(dispatch_error_forensic_fields(
                case,
                mode,
                lhs_dtype,
                rhs_dtype,
                lhs_device,
                rhs_device,
                reason_code,
                error_message,
            )),
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
    let reason_code = if passed {
        "dispatch_parity_ok"
    } else {
        "dispatch_expectation_mismatch"
    };

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
            reason_code,
        )
        .with_extra_fields(dispatch_forensic_fields(
            case,
            mode,
            outcome.tensor.value(),
            outcome.decision.selected_key,
            outcome.decision.backend_key,
            outcome.decision.kernel,
            outcome.decision.keyset_bits,
            outcome.decision.fallback_used,
            lhs_dtype,
            rhs_dtype,
            lhs_device,
            rhs_device,
            reason_code,
        )),
    })
}

fn run_op_schema_case(
    case: &OpSchemaCase,
    mode: ExecutionMode,
) -> Result<OpSchemaCaseReport, String> {
    let parsed_schema = parse_schema_or_name(case.schema_input.as_str());
    let parse_ok = parsed_schema.is_ok() == case.expect_parse_ok;

    let schema_variant_ok = case.expect_schema_variant.is_none_or(|expected| {
        matches!(parsed_schema.as_ref(), Ok(ParsedSchemaInput::Schema(_))) == expected
    });

    let out_variant_ok = case.expect_out_variant.is_none_or(|expected| {
        matches!(
            parsed_schema.as_ref(),
            Ok(ParsedSchemaInput::Schema(schema)) if schema.is_out_variant
        ) == expected
    });

    let dispatch_observed = case.dispatch_tags.as_ref().is_some_and(|tags| {
        let tag_refs: Vec<&str> = tags.iter().map(String::as_str).collect();
        schema_dispatch_keyset_from_tags(tag_refs.as_slice()).is_ok()
    });
    let dispatch_ok = case
        .expect_dispatch_ok
        .is_none_or(|expected| dispatch_observed == expected);

    let normalization_observed = case.name_input.as_deref().is_some_and(|input| {
        let lhs_name = match parsed_schema.as_ref() {
            Ok(ParsedSchemaInput::Schema(schema)) => Some(schema.op.unambiguous_name()),
            Ok(ParsedSchemaInput::Name(name)) => Some(name.unambiguous_name()),
            Err(_) => None,
        };
        let rhs_name = match parse_schema_or_name(input) {
            Ok(ParsedSchemaInput::Name(name)) => Some(name.unambiguous_name()),
            _ => None,
        };
        lhs_name
            .zip(rhs_name)
            .is_some_and(|(left, right)| left == right)
    });
    let normalization_ok = case
        .expect_name_normalization
        .is_none_or(|expected| normalization_observed == expected);

    let passed = parse_ok && schema_variant_ok && out_variant_ok && dispatch_ok && normalization_ok;
    let reason_code = if passed {
        if case.expect_parse_ok {
            "op_schema_parity_ok"
        } else {
            "op_schema_adversarial_fail_closed_ok"
        }
    } else if !parse_ok {
        "op_schema_parse_expectation_mismatch"
    } else if !schema_variant_ok {
        "op_schema_schema_variant_mismatch"
    } else if !out_variant_ok {
        "op_schema_out_variant_mismatch"
    } else if !dispatch_ok {
        "op_schema_dispatch_expectation_mismatch"
    } else {
        "op_schema_name_normalization_mismatch"
    };

    Ok(OpSchemaCaseReport {
        name: case.name.clone(),
        mode,
        parse_ok,
        schema_variant_ok,
        out_variant_ok,
        dispatch_ok,
        normalization_ok,
        forensic_log: StructuredCaseLog::new(
            "op_schema",
            "op_schema_cases.json",
            "FT-P2C-003",
            case.name.as_str(),
            mode,
            vec![
                "crates/ft-conformance/fixtures/op_schema_cases.json".to_string(),
                "artifacts/phase2c/FT-P2C-003/unit_property_quality_report_v1.json".to_string(),
                "artifacts/phase2c/FT-P2C-003/differential_packet_report_v1.json".to_string(),
            ],
            format!(
                "cargo run -p ft-conformance --bin run_e2e_matrix -- --mode {} --packet FT-P2C-003 --output artifacts/phase2c/e2e_forensics/ft-p2c-003.jsonl",
                mode_label(mode),
            ),
            if passed { "pass" } else { "fail" },
            reason_code,
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
    let reason_code = if passed {
        "scheduler_parity_ok"
    } else {
        "scheduler_expectation_mismatch"
    };

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
            autograd_scheduler_evidence_refs(),
            format!(
                "cargo test -p ft-conformance strict_scheduler_conformance_is_green -- --nocapture # mode={}",
                mode_label(mode)
            ),
            if passed { "pass" } else { "fail" },
            reason_code,
        )
        .with_extra_fields(scheduler_forensic_fields(&report.telemetry)),
    })
}

fn scheduler_forensic_fields(telemetry: &SchedulerTelemetry) -> BTreeMap<String, Value> {
    let mut fields = BTreeMap::new();
    fields.insert(
        "execution_order".to_string(),
        json!(
            telemetry
                .execution_order
                .iter()
                .map(|node| node.0)
                .collect::<Vec<_>>()
        ),
    );
    fields.insert("queue_pushes".to_string(), json!(telemetry.queue_pushes));
    fields.insert("queue_pops".to_string(), json!(telemetry.queue_pops));
    fields.insert("max_queue_len".to_string(), json!(telemetry.max_queue_len));
    fields.insert(
        "dependency_snapshot".to_string(),
        json!(telemetry.dependency_snapshot),
    );
    fields.insert(
        "reentrant_depth".to_string(),
        json!(telemetry.reentrant_depth),
    );
    fields.insert(
        "reentrant_guard_triggered".to_string(),
        json!(telemetry.reentrant_guard_triggered),
    );
    fields.insert(
        "hardened_fallback_used".to_string(),
        json!(telemetry.hardened_fallback_used),
    );
    fields
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

    let entries = serialization_entries(case);

    let payload = encode_checkpoint(entries.as_slice(), checkpoint_mode);
    let decoded = decode_checkpoint(payload.as_str(), decode_mode)
        .map_err(|error| format!("serialization case '{}' decode failed: {error}", case.name))?;

    let mut expected_entries = entries.clone();
    expected_entries.sort_by_key(|entry| entry.node_id);
    let decode_ok = decoded.entries == expected_entries;

    let repair_symbols = case.repair_symbols.unwrap_or(4);
    let (sidecar_a, proof_a) =
        serialization_generate_sidecar_with_retry(payload.as_str(), repair_symbols).map_err(
            |error| format!("serialization case '{}' sidecar failed: {error}", case.name),
        )?;
    let (_sidecar_b, proof_b) =
        serialization_generate_sidecar_with_retry_uncached(payload.as_str(), repair_symbols)
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
            serialization_evidence_refs(),
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

fn run_nn_state_case(case: &NnStateCase, mode: ExecutionMode) -> Result<NnStateCaseReport, String> {
    let expectation = match mode {
        ExecutionMode::Strict => &case.strict,
        ExecutionMode::Hardened => &case.hardened,
    };
    let strict_flag = mode == ExecutionMode::Strict;

    let (
        operation_ok,
        policy_ok,
        observed_reason_code,
        actual_state_keys,
        prefix_normalization_applied,
        training_flag_transition,
    ) = match case.operation.as_str() {
        "register_parameter" | "register_buffer" => {
            let key = case.state_key.as_deref().unwrap_or_default();
            let operation_ok = nn_state_is_valid_key(key);
            let observed_reason_code = if operation_ok {
                "nn_state_register_ok".to_string()
            } else {
                "nn_state_register_rejected".to_string()
            };
            (
                operation_ok,
                true,
                observed_reason_code,
                Vec::new(),
                false,
                Vec::new(),
            )
        }
        "state_export" => {
            let actual_state_keys = nn_state_export_keys(case);
            let expected_state_keys = dedupe_sorted(case.expected_state_keys.clone());
            let operation_ok = actual_state_keys == expected_state_keys;
            let observed_reason_code = if operation_ok {
                "nn_state_state_export_ok".to_string()
            } else {
                "nn_state_state_export_mismatch".to_string()
            };
            (
                operation_ok,
                true,
                observed_reason_code,
                actual_state_keys,
                false,
                Vec::new(),
            )
        }
        "mode_transition" => {
            let mut training = case.initial_training.unwrap_or(true);
            let mut training_flag_transition = vec![training];
            for next in case.training_transitions.iter().copied() {
                training = next;
                training_flag_transition.push(training);
            }
            let expected_training = case.expected_training_flag.unwrap_or(training);
            let operation_ok = training == expected_training;
            let observed_reason_code = if operation_ok {
                "nn_state_mode_transition_ok".to_string()
            } else {
                "nn_state_mode_transition_mismatch".to_string()
            };
            (
                operation_ok,
                true,
                observed_reason_code,
                Vec::new(),
                false,
                training_flag_transition,
            )
        }
        "load_state" => {
            let has_missing = !case.missing_keys.is_empty();
            let has_unexpected = !case.unexpected_keys.is_empty();
            let has_incompatible = !case.incompatible_shapes.is_empty();

            let strict_error = has_missing || has_unexpected || has_incompatible;
            let hardened_error = has_incompatible;
            let observed_error = if strict_flag {
                strict_error
            } else {
                hardened_error
            };

            let operation_ok = !observed_error;
            let policy_ok = if strict_flag {
                observed_error == strict_error
            } else {
                observed_error == hardened_error
            };

            let observed_reason_code = if observed_error {
                if has_incompatible {
                    "nn_state_load_incompatible_shapes_rejected".to_string()
                } else {
                    "nn_state_load_missing_or_unexpected_rejected".to_string()
                }
            } else if has_missing || has_unexpected {
                "nn_state_load_hardened_allowlisted".to_string()
            } else {
                "nn_state_load_ok".to_string()
            };
            (
                operation_ok,
                policy_ok,
                observed_reason_code,
                Vec::new(),
                false,
                Vec::new(),
            )
        }
        "prefix_normalization" => {
            let allow = case.allow_prefix_normalization.unwrap_or(false);
            let (normalized_keys, applied) =
                nn_state_normalize_prefix_keys(case.prefix_keys.as_slice(), allow);
            let expected_canonical = if case.expected_canonical_keys.is_empty() {
                normalized_keys.clone()
            } else {
                dedupe_sorted(case.expected_canonical_keys.clone())
            };
            let operation_ok = normalized_keys == expected_canonical;
            let observed_reason_code = if operation_ok {
                "nn_state_prefix_normalization_ok".to_string()
            } else {
                "nn_state_prefix_normalization_mismatch".to_string()
            };
            (
                operation_ok,
                true,
                observed_reason_code,
                normalized_keys,
                applied,
                Vec::new(),
            )
        }
        "hook_trace" => {
            let operation_ok = case.hook_trace == case.expected_hook_trace;
            let observed_reason_code = if operation_ok {
                "nn_state_hook_trace_ok".to_string()
            } else {
                "nn_state_hook_trace_mismatch".to_string()
            };
            (
                operation_ok,
                true,
                observed_reason_code,
                Vec::new(),
                false,
                Vec::new(),
            )
        }
        other => {
            return Err(format!(
                "unsupported nn_state operation '{other}' in case '{}'",
                case.name
            ));
        }
    };

    let contract_ok = operation_ok && policy_ok;
    let expectation_ok = contract_ok == expectation.expect_pass;
    let mut detail_ok = true;
    let mut reason_code = observed_reason_code.clone();

    if !expectation_ok {
        reason_code = "nn_state_mode_expectation_mismatch".to_string();
    }
    if let Some(expected_prefix_applied) = expectation.expect_prefix_normalization_applied
        && prefix_normalization_applied != expected_prefix_applied
    {
        detail_ok = false;
        reason_code = "nn_state_prefix_policy_mismatch".to_string();
    }
    if let Some(expected_reason_code) = expectation.expected_reason_code.as_deref()
        && observed_reason_code != expected_reason_code
    {
        detail_ok = false;
        reason_code = "nn_state_reason_code_mismatch".to_string();
    }

    let passed = expectation_ok && detail_ok;
    if passed {
        reason_code = observed_reason_code.clone();
    }

    Ok(NnStateCaseReport {
        name: case.name.clone(),
        mode,
        contract_ok,
        expectation_ok,
        detail_ok,
        forensic_log: StructuredCaseLog::new(
            "nn_state",
            "nn_state_cases.json",
            "FT-P2C-008",
            case.name.as_str(),
            mode,
            nn_state_evidence_refs(),
            format!(
                "cargo test -p ft-conformance strict_nn_state_conformance_is_green -- --nocapture # mode={}",
                mode_label(mode)
            ),
            if passed { "pass" } else { "fail" },
            reason_code.clone(),
        )
        .with_extra_fields(nn_state_forensic_fields(
            case,
            mode,
            contract_ok,
            expectation_ok,
            detail_ok,
            actual_state_keys.as_slice(),
            prefix_normalization_applied,
            training_flag_transition.as_slice(),
            observed_reason_code.as_str(),
        )),
    })
}

fn nn_state_export_keys(case: &NnStateCase) -> Vec<String> {
    let mut keys = case.parameter_keys.clone();
    keys.extend(case.persistent_buffer_keys.iter().cloned());
    dedupe_sorted(keys)
}

fn nn_state_normalize_prefix_keys(keys: &[String], allow: bool) -> (Vec<String>, bool) {
    let mut normalized = Vec::with_capacity(keys.len());
    let mut applied = false;
    for key in keys {
        if allow && let Some(stripped) = key.strip_prefix("module.") {
            normalized.push(stripped.to_string());
            applied = true;
            continue;
        }
        normalized.push(key.clone());
    }
    (dedupe_sorted(normalized), applied)
}

fn nn_state_is_valid_key(key: &str) -> bool {
    !key.is_empty()
        && !key.starts_with('.')
        && !key.ends_with('.')
        && !key.contains("..")
        && key.split('.').all(|segment| {
            !segment.is_empty()
                && segment
                    .chars()
                    .all(|ch| ch.is_ascii_alphanumeric() || ch == '_')
        })
}

fn log_string_vec_field(log: &StructuredCaseLog, key: &str) -> Vec<String> {
    log.extra_fields
        .get(key)
        .and_then(Value::as_array)
        .map(|values| {
            values
                .iter()
                .filter_map(Value::as_str)
                .map(ToString::to_string)
                .collect()
        })
        .unwrap_or_default()
}

fn log_bool_vec_field(log: &StructuredCaseLog, key: &str) -> Vec<bool> {
    log.extra_fields
        .get(key)
        .and_then(Value::as_array)
        .map(|values| values.iter().filter_map(Value::as_bool).collect())
        .unwrap_or_default()
}

fn dedupe_sorted(mut keys: Vec<String>) -> Vec<String> {
    keys.sort();
    keys.dedup();
    keys
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

fn scaled_scheduler_case(base: &SchedulerCase, scale: f64) -> SchedulerCase {
    SchedulerCase {
        name: format!("{}__scaled_x{scale}", base.name),
        x: base.x * scale,
        y: base.y * scale,
        expected_x_grad: base.expected_x_grad * scale,
        expected_y_grad: base.expected_y_grad * scale,
        expected_execution_order: base.expected_execution_order.clone(),
        tolerance: base.tolerance,
    }
}

fn scheduler_scale_relation_holds(
    base: &LegacyUnaryGradObservation,
    scaled: &LegacyUnaryGradObservation,
    scale: f64,
    tolerance: f64,
) -> bool {
    within(scaled.output, base.output * scale * scale, tolerance)
        && within(scaled.x_grad, base.x_grad * scale, tolerance)
        && within(scaled.y_grad, base.y_grad * scale, tolerance)
}

fn strict_overflow_rejected(case: &SchedulerCase) -> Result<bool, String> {
    let mut tape = Tape::new();
    let x = tape.leaf(case.x, true);
    let y = tape.leaf(case.y, true);
    let (sum, _) = tape
        .add(x, y, ExecutionMode::Strict)
        .map_err(|error| format!("strict overflow probe '{}' add failed: {error}", case.name))?;
    let (out, _) = tape
        .mul(sum, x, ExecutionMode::Strict)
        .map_err(|error| format!("strict overflow probe '{}' mul failed: {error}", case.name))?;
    let overflow = tape.backward_with_options(
        out,
        BackwardOptions {
            max_reentrant_depth: 1,
            current_reentrant_depth: 2,
            policy: ReentrantPolicy::StrictFail,
        },
    );
    Ok(matches!(
        overflow,
        Err(AutogradError::ReentrantDepthExceeded { .. })
    ))
}

fn hardened_overflow_guarded(case: &SchedulerCase) -> Result<bool, String> {
    let mut tape = Tape::new();
    let x = tape.leaf(case.x, true);
    let y = tape.leaf(case.y, true);
    let (sum, _) = tape.add(x, y, ExecutionMode::Hardened).map_err(|error| {
        format!(
            "hardened overflow probe '{}' add failed: {error}",
            case.name
        )
    })?;
    let (out, _) = tape.mul(sum, x, ExecutionMode::Hardened).map_err(|error| {
        format!(
            "hardened overflow probe '{}' mul failed: {error}",
            case.name
        )
    })?;
    let report = tape
        .backward_with_options(
            out,
            BackwardOptions {
                max_reentrant_depth: 1,
                current_reentrant_depth: 2,
                policy: ReentrantPolicy::HardenedBoundedFallback,
            },
        )
        .map_err(|error| {
            format!(
                "hardened overflow probe '{}' backward failed unexpectedly: {error}",
                case.name
            )
        })?;
    Ok(report.telemetry.reentrant_guard_triggered && report.telemetry.hardened_fallback_used)
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

fn scalar_forensic_fields(
    case: &ScalarCase,
    mode: ExecutionMode,
    actual_output: f64,
    actual_lhs_grad: f64,
    actual_rhs_grad: f64,
    passed: bool,
    reason_code: &str,
) -> BTreeMap<String, Value> {
    let mut fields = BTreeMap::new();
    let selected_kernel = match case.op.as_str() {
        "add" => "autograd_cpu::add_scalar",
        "mul" => "autograd_cpu::mul_scalar",
        _ => "autograd_cpu::unknown",
    };
    fields.insert("contract_ids".to_string(), json!(case.contract_ids));
    fields.insert(
        "downstream_e2e_scenarios".to_string(),
        json!(case.e2e_scenarios),
    );
    fields.insert("op".to_string(), json!(case.op));
    fields.insert("lhs".to_string(), json!(case.lhs));
    fields.insert("rhs".to_string(), json!(case.rhs));
    fields.insert("expected_output".to_string(), json!(case.expected_output));
    fields.insert("actual_output".to_string(), json!(actual_output));
    fields.insert(
        "expected_lhs_grad".to_string(),
        json!(case.expected_lhs_grad),
    );
    fields.insert("actual_lhs_grad".to_string(), json!(actual_lhs_grad));
    fields.insert(
        "expected_rhs_grad".to_string(),
        json!(case.expected_rhs_grad),
    );
    fields.insert("actual_rhs_grad".to_string(), json!(actual_rhs_grad));
    fields.insert("dispatch_key".to_string(), json!("AutogradCPU"));
    fields.insert("selected_kernel".to_string(), json!(selected_kernel));
    fields.insert("backend_key".to_string(), json!("CPU"));
    fields.insert("input_shape".to_string(), json!([[]]));
    fields.insert("output_shape".to_string(), json!([]));
    fields.insert("dtype_pair".to_string(), json!("F64/F64"));
    fields.insert("broadcast_applied".to_string(), json!(false));
    fields.insert("fallback_path".to_string(), json!(false));
    fields.insert("pass".to_string(), json!(passed));
    fields.insert("mode".to_string(), json!(mode_label(mode)));
    fields.insert("reason_code".to_string(), json!(reason_code));
    fields
}

fn tensor_meta_forensic_fields(
    case: &TensorMetaCase,
    observed: &TensorMetaObservation,
    passed: bool,
) -> BTreeMap<String, Value> {
    let mut fields = BTreeMap::new();
    fields.insert("contract_ids".to_string(), json!(case.contract_ids));
    fields.insert(
        "downstream_e2e_scenarios".to_string(),
        json!(case.e2e_scenarios),
    );
    fields.insert("shape".to_string(), json!(case.shape));
    fields.insert("strides".to_string(), json!(case.strides));
    fields.insert(
        "storage_offset".to_string(),
        json!(case.storage_offset.unwrap_or(0)),
    );
    fields.insert("index".to_string(), json!(case.index));
    fields.insert(
        "expected_linear_index".to_string(),
        json!(case.expected_linear_index),
    );
    fields.insert(
        "actual_linear_index".to_string(),
        json!(observed.linear_index),
    );
    fields.insert("expected_numel".to_string(), json!(case.expected_numel));
    fields.insert("actual_numel".to_string(), json!(observed.numel));
    fields.insert(
        "expected_contiguous".to_string(),
        json!(case.expected_contiguous),
    );
    fields.insert("actual_contiguous".to_string(), json!(observed.contiguous));
    fields.insert(
        "expect_valid".to_string(),
        json!(case.expect_valid.unwrap_or(true)),
    );
    fields.insert("actual_valid".to_string(), json!(observed.valid));
    fields.insert("pass".to_string(), json!(passed));
    fields.insert(
        "selected_kernel".to_string(),
        json!("tensor_meta::validate"),
    );
    fields.insert("dispatch_key".to_string(), json!("CPU"));
    fields.insert("backend_key".to_string(), json!("CPU"));
    fields.insert("broadcast_applied".to_string(), json!(false));
    fields.insert("fallback_path".to_string(), json!(false));
    fields
}

#[allow(clippy::too_many_arguments)]
fn dispatch_forensic_fields(
    case: &DispatchCase,
    mode: ExecutionMode,
    output_value: f64,
    selected_key: DispatchKey,
    backend_key: DispatchKey,
    kernel: &str,
    keyset_bits: u64,
    fallback_used: bool,
    lhs_dtype: DType,
    rhs_dtype: DType,
    lhs_device: Device,
    rhs_device: Device,
    reason_code: &str,
) -> BTreeMap<String, Value> {
    let mut fields = BTreeMap::new();
    fields.insert("contract_ids".to_string(), json!(case.contract_ids));
    fields.insert(
        "downstream_e2e_scenarios".to_string(),
        json!(case.e2e_scenarios),
    );
    fields.insert("op".to_string(), json!(case.op));
    fields.insert("lhs".to_string(), json!(case.lhs));
    fields.insert("rhs".to_string(), json!(case.rhs));
    fields.insert("mode".to_string(), json!(mode_label(mode)));
    fields.insert(
        "dispatch_key".to_string(),
        json!(format!("{selected_key:?}")),
    );
    fields.insert("selected_kernel".to_string(), json!(kernel));
    fields.insert("backend_key".to_string(), json!(format!("{backend_key:?}")));
    fields.insert("keyset_bits".to_string(), json!(keyset_bits));
    fields.insert("fallback_path".to_string(), json!(fallback_used));
    fields.insert(
        "dtype_pair".to_string(),
        json!(format!("{lhs_dtype:?}/{rhs_dtype:?}")),
    );
    fields.insert(
        "device_pair".to_string(),
        json!(format!("{lhs_device:?}/{rhs_device:?}")),
    );
    fields.insert("input_shape".to_string(), json!([[], []]));
    fields.insert("output_shape".to_string(), json!([]));
    fields.insert("broadcast_applied".to_string(), json!(false));
    fields.insert("actual_output".to_string(), json!(output_value));
    fields.insert("reason_code".to_string(), json!(reason_code));
    fields
}

#[allow(clippy::too_many_arguments)]
fn dispatch_error_forensic_fields(
    case: &DispatchCase,
    mode: ExecutionMode,
    lhs_dtype: DType,
    rhs_dtype: DType,
    lhs_device: Device,
    rhs_device: Device,
    reason_code: &str,
    error_message: Option<String>,
) -> BTreeMap<String, Value> {
    let mut fields = BTreeMap::new();
    fields.insert("contract_ids".to_string(), json!(case.contract_ids));
    fields.insert(
        "downstream_e2e_scenarios".to_string(),
        json!(case.e2e_scenarios),
    );
    fields.insert("op".to_string(), json!(case.op));
    fields.insert("lhs".to_string(), json!(case.lhs));
    fields.insert("rhs".to_string(), json!(case.rhs));
    fields.insert("mode".to_string(), json!(mode_label(mode)));
    fields.insert(
        "dtype_pair".to_string(),
        json!(format!("{lhs_dtype:?}/{rhs_dtype:?}")),
    );
    fields.insert(
        "device_pair".to_string(),
        json!(format!("{lhs_device:?}/{rhs_device:?}")),
    );
    fields.insert("input_shape".to_string(), json!([[], []]));
    fields.insert("output_shape".to_string(), Value::Null);
    fields.insert("dispatch_key".to_string(), Value::Null);
    fields.insert("selected_kernel".to_string(), Value::Null);
    fields.insert("backend_key".to_string(), Value::Null);
    fields.insert("broadcast_applied".to_string(), json!(false));
    fields.insert("fallback_path".to_string(), json!(false));
    fields.insert("reason_code".to_string(), json!(reason_code));
    fields.insert("error_message".to_string(), json!(error_message));
    fields
}

#[allow(clippy::too_many_arguments)]
fn nn_state_forensic_fields(
    case: &NnStateCase,
    mode: ExecutionMode,
    contract_ok: bool,
    expectation_ok: bool,
    detail_ok: bool,
    actual_state_keys: &[String],
    prefix_normalization_applied: bool,
    training_flag_transition: &[bool],
    observed_reason_code: &str,
) -> BTreeMap<String, Value> {
    let mut fields = BTreeMap::new();
    fields.insert("contract_ids".to_string(), json!(case.contract_ids));
    fields.insert(
        "downstream_e2e_scenarios".to_string(),
        json!(case.e2e_scenarios),
    );
    fields.insert("operation".to_string(), json!(case.operation));
    fields.insert("module_path".to_string(), json!(case.module_path));
    fields.insert("state_key".to_string(), json!(case.state_key));
    fields.insert("state_key_kind".to_string(), json!(case.state_key_kind));
    fields.insert(
        "strict_flag".to_string(),
        json!(mode == ExecutionMode::Strict),
    );
    fields.insert(
        "assign_flag".to_string(),
        json!(case.assign_flag.unwrap_or(false)),
    );
    fields.insert("missing_keys".to_string(), json!(case.missing_keys));
    fields.insert("unexpected_keys".to_string(), json!(case.unexpected_keys));
    fields.insert(
        "incompatible_shapes".to_string(),
        json!(case.incompatible_shapes),
    );
    fields.insert("hook_trace".to_string(), json!(case.hook_trace));
    fields.insert(
        "prefix_normalization_applied".to_string(),
        json!(prefix_normalization_applied),
    );
    fields.insert(
        "training_flag_transition".to_string(),
        json!(training_flag_transition),
    );
    fields.insert("parameter_keys".to_string(), json!(case.parameter_keys));
    fields.insert(
        "persistent_buffer_keys".to_string(),
        json!(case.persistent_buffer_keys),
    );
    fields.insert(
        "non_persistent_buffer_keys".to_string(),
        json!(case.non_persistent_buffer_keys),
    );
    fields.insert(
        "expected_state_keys".to_string(),
        json!(case.expected_state_keys),
    );
    fields.insert("actual_state_keys".to_string(), json!(actual_state_keys));
    fields.insert("prefix_keys".to_string(), json!(case.prefix_keys));
    fields.insert(
        "expected_canonical_keys".to_string(),
        json!(case.expected_canonical_keys),
    );
    fields.insert("contract_ok".to_string(), json!(contract_ok));
    fields.insert("expectation_ok".to_string(), json!(expectation_ok));
    fields.insert("detail_ok".to_string(), json!(detail_ok));
    fields.insert(
        "observed_reason_code".to_string(),
        json!(observed_reason_code),
    );
    fields
}

fn parse_binary_op(op: &str) -> Result<BinaryOp, String> {
    match op {
        "add" => Ok(BinaryOp::Add),
        "mul" => Ok(BinaryOp::Mul),
        _ => Err(format!("unsupported binary op '{op}'")),
    }
}

fn parse_dtype(raw: &str) -> Result<DType, String> {
    match raw {
        "F64" => Ok(DType::F64),
        "F32" => Ok(DType::F32),
        _ => Err(format!("unsupported dtype '{raw}'")),
    }
}

fn parse_device(raw: &str) -> Result<Device, String> {
    match raw {
        "Cpu" | "CPU" => Ok(Device::Cpu),
        "Cuda" | "CUDA" => Ok(Device::Cuda),
        _ => Err(format!("unsupported device '{raw}'")),
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

fn op_schema_evidence_refs() -> Vec<String> {
    vec![
        "crates/ft-conformance/fixtures/op_schema_cases.json".to_string(),
        "artifacts/phase2c/FT-P2C-003/contract_table.md".to_string(),
        "artifacts/phase2c/FT-P2C-003/behavior_extraction_ledger.md".to_string(),
        "artifacts/phase2c/FT-P2C-003/unit_property_quality_report_v1.json".to_string(),
    ]
}

fn autograd_scheduler_evidence_refs() -> Vec<String> {
    vec![
        "crates/ft-conformance/fixtures/autograd_scheduler_cases.json".to_string(),
        "artifacts/phase2c/FT-P2C-004/parity_report.json".to_string(),
        "artifacts/phase2c/FT-P2C-004/unit_property_quality_report_v1.json".to_string(),
        "artifacts/phase2c/FT-P2C-004/differential_packet_report_v1.json".to_string(),
        "artifacts/phase2c/FT-P2C-004/differential_reconciliation_v1.md".to_string(),
    ]
}

fn serialization_evidence_refs() -> Vec<String> {
    vec![
        "crates/ft-conformance/fixtures/serialization_cases.json".to_string(),
        "artifacts/phase2c/FT-P2C-006/parity_report.json".to_string(),
        "artifacts/phase2c/FT-P2C-006/parity_report.raptorq.json".to_string(),
        "artifacts/phase2c/FT-P2C-006/unit_property_quality_report_v1.json".to_string(),
        "artifacts/phase2c/FT-P2C-006/differential_packet_report_v1.json".to_string(),
        "artifacts/phase2c/FT-P2C-006/differential_reconciliation_v1.md".to_string(),
    ]
}

fn nn_state_evidence_refs() -> Vec<String> {
    vec![
        "crates/ft-conformance/fixtures/nn_state_cases.json".to_string(),
        "artifacts/phase2c/FT-P2C-008/contract_table.md".to_string(),
        "artifacts/phase2c/FT-P2C-008/threat_model.md".to_string(),
        "artifacts/phase2c/FT-P2C-008/unit_property_quality_report_v1.json".to_string(),
    ]
}

fn nn_state_differential_evidence_refs() -> Vec<String> {
    vec![
        "crates/ft-conformance/fixtures/nn_state_cases.json".to_string(),
        "artifacts/phase2c/FT-P2C-008/contract_table.md".to_string(),
        "artifacts/phase2c/FT-P2C-008/threat_model.md".to_string(),
        "artifacts/phase2c/FT-P2C-008/unit_property_quality_report_v1.json".to_string(),
        "artifacts/phase2c/FT-P2C-008/differential_packet_report_v1.json".to_string(),
        "artifacts/phase2c/FT-P2C-008/differential_reconciliation_v1.md".to_string(),
    ]
}

fn ft_p2c_005_differential_evidence_refs() -> Vec<String> {
    vec![
        "artifacts/phase2c/FT-P2C-005/contract_table.md".to_string(),
        "artifacts/phase2c/FT-P2C-005/threat_model.md".to_string(),
        "artifacts/phase2c/FT-P2C-005/behavior_extraction_ledger.md".to_string(),
        "artifacts/phase2c/FT-P2C-005/differential_packet_report_v1.json".to_string(),
        "artifacts/phase2c/FT-P2C-005/differential_reconciliation_v1.md".to_string(),
        "artifacts/phase2c/e2e_forensics/ft-p2c-005.jsonl".to_string(),
    ]
}

fn serialization_entries(case: &SerializationCase) -> Vec<SerializedSnapshotEntry> {
    case.entries
        .iter()
        .map(|entry| SerializedSnapshotEntry {
            node_id: entry.node_id,
            value: entry.value,
            grad: entry.grad,
        })
        .collect()
}

fn serialization_generate_sidecar_with_retry(
    payload: &str,
    repair_symbols: usize,
) -> Result<(RaptorQSidecar, DecodeProofArtifact), String> {
    serialization_generate_sidecar_with_retry_inner(payload, repair_symbols, true)
}

fn serialization_generate_sidecar_with_retry_uncached(
    payload: &str,
    repair_symbols: usize,
) -> Result<(RaptorQSidecar, DecodeProofArtifact), String> {
    serialization_generate_sidecar_with_retry_inner(payload, repair_symbols, false)
}

fn serialization_generate_sidecar_with_retry_inner(
    payload: &str,
    repair_symbols: usize,
    use_cache: bool,
) -> Result<(RaptorQSidecar, DecodeProofArtifact), String> {
    let cache_key = (payload.to_string(), repair_symbols);
    if use_cache
        && let Ok(cache_guard) = serialization_sidecar_cache().lock()
        && let Some(cached) = cache_guard.get(&cache_key)
    {
        return Ok(cached.clone());
    }

    let mut budgets = Vec::with_capacity(7);
    budgets.push(repair_symbols.max(1));
    for fallback in [4usize, 8, 16, 32, 64, 128] {
        if !budgets.contains(&fallback) {
            budgets.push(fallback);
        }
    }

    let mut last_error = None;
    for budget in budgets {
        match generate_raptorq_sidecar(payload, budget) {
            Ok(result) => {
                if let Ok(mut cache_guard) = serialization_sidecar_cache().lock() {
                    cache_guard.insert(cache_key, result.clone());
                }
                return Ok(result);
            }
            Err(error) => last_error = Some(error),
        }
    }

    Err(match last_error {
        Some(error) => format!("sidecar generation retries exhausted: {error}"),
        None => "sidecar generation retries exhausted without attempts".to_string(),
    })
}

fn serialization_sidecar_cache() -> &'static Mutex<SidecarCache> {
    SERIALIZATION_SIDECAR_CACHE.get_or_init(|| Mutex::new(BTreeMap::new()))
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
    use std::collections::{BTreeMap, BTreeSet};
    use std::fs;
    use std::time::{SystemTime, UNIX_EPOCH};

    use serde_json::json;

    use super::{
        ExecutionMode, HarnessConfig, NnStateCase, NnStateCaseReport, NnStateModeExpectation,
        StructuredCaseLog, emit_differential_report, emit_differential_report_filtered,
        emit_e2e_forensics_matrix, emit_e2e_forensics_matrix_filtered, load_allowlist,
        nn_state_export_keys, nn_state_is_valid_key, project_log_to_ft_p2c_005,
        project_log_to_ft_p2c_007, run_autograd_scheduler_conformance,
        run_differential_conformance, run_dispatch_conformance, run_nn_state_conformance,
        run_op_schema_conformance, run_packet_e2e_microbench, run_packet_e2e_microbench_legacy,
        run_scalar_conformance, run_scalar_microbench, run_serialization_conformance, run_smoke,
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
        let failed_cases: Vec<&str> = cases
            .iter()
            .filter(|case| !(case.meta_ok && case.index_ok && case.alias_ok))
            .map(|case| case.name.as_str())
            .collect();

        assert_eq!(report.cases_total, cases.len());
        assert_eq!(
            report.cases_total, report.cases_passed,
            "strict tensor-meta failing cases: {failed_cases:?}"
        );
    }

    #[test]
    fn hardened_tensor_meta_conformance_is_green() {
        let cfg = HarnessConfig::default_paths();
        let (report, cases) = run_tensor_meta_conformance(&cfg, ExecutionMode::Hardened)
            .expect("tensor-meta should run");
        let failed_cases: Vec<&str> = cases
            .iter()
            .filter(|case| !(case.meta_ok && case.index_ok && case.alias_ok))
            .map(|case| case.name.as_str())
            .collect();

        assert_eq!(
            report.cases_total, report.cases_passed,
            "hardened tensor-meta failing cases: {failed_cases:?}"
        );
    }

    #[test]
    fn hardened_dispatch_conformance_is_green() {
        let cfg = HarnessConfig::default_paths();
        let (report, _) =
            run_dispatch_conformance(&cfg, ExecutionMode::Hardened).expect("dispatch should run");

        assert_eq!(report.cases_total, report.cases_passed);
    }

    #[test]
    fn strict_op_schema_conformance_is_green() {
        let cfg = HarnessConfig::default_paths();
        let (report, cases) =
            run_op_schema_conformance(&cfg, ExecutionMode::Strict).expect("op schema should run");

        assert_eq!(report.suite, "op_schema");
        assert!(!cases.is_empty());
        assert!(cases.iter().all(|case| case.passed()));
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
    fn strict_nn_state_conformance_is_green() {
        let cfg = HarnessConfig::default_paths();
        let (report, cases) =
            run_nn_state_conformance(&cfg, ExecutionMode::Strict).expect("nn_state should run");

        assert_eq!(report.suite, "nn_state");
        assert_eq!(report.cases_total, cases.len());
        assert_eq!(report.cases_total, report.cases_passed);
        assert!(cases.iter().all(NnStateCaseReport::passed));
    }

    #[test]
    fn hardened_nn_state_conformance_is_green() {
        let cfg = HarnessConfig::default_paths();
        let (report, cases) =
            run_nn_state_conformance(&cfg, ExecutionMode::Hardened).expect("nn_state should run");

        assert_eq!(report.suite, "nn_state");
        assert_eq!(report.cases_total, cases.len());
        assert_eq!(report.cases_total, report.cases_passed);
        assert!(cases.iter().all(NnStateCaseReport::passed));
    }

    #[test]
    fn nn_state_logs_include_packet_008_contract_fields() {
        let cfg = HarnessConfig::default_paths();
        let (_, cases) =
            run_nn_state_conformance(&cfg, ExecutionMode::Strict).expect("nn_state should run");

        assert!(!cases.is_empty(), "expected at least one nn_state case");
        let value = serde_json::to_value(&cases[0].forensic_log)
            .expect("nn_state log should serialize to json");
        for key in [
            "module_path",
            "state_key",
            "state_key_kind",
            "strict_flag",
            "assign_flag",
            "missing_keys",
            "unexpected_keys",
            "incompatible_shapes",
            "hook_trace",
            "prefix_normalization_applied",
            "training_flag_transition",
        ] {
            assert!(value.get(key).is_some(), "missing nn_state field {key}");
        }
    }

    #[test]
    fn nn_state_property_key_validation_stays_fail_closed() {
        let valid_keys = [
            "weight",
            "layer1.weight",
            "encoder_block_0.buffer_1",
            "module_2.submodule_3.param4",
        ];
        for key in valid_keys {
            assert!(nn_state_is_valid_key(key), "expected valid key: {key}");
        }

        let invalid_keys = [
            "",
            ".weight",
            "weight.",
            "layer..weight",
            "layer-1.weight",
            "layer weight",
        ];
        for key in invalid_keys {
            assert!(!nn_state_is_valid_key(key), "expected invalid key: {key}");
        }
    }

    #[test]
    fn nn_state_property_export_excludes_non_persistent_buffers() {
        let expectation = NnStateModeExpectation {
            expect_pass: true,
            expected_reason_code: None,
            expect_prefix_normalization_applied: None,
        };

        let parameter_sets = [
            Vec::<String>::new(),
            vec!["encoder.weight".to_string()],
            vec!["encoder.weight".to_string(), "decoder.bias".to_string()],
        ];
        let persistent_sets = [
            Vec::<String>::new(),
            vec!["running_mean".to_string()],
            vec!["running_mean".to_string(), "running_var".to_string()],
        ];
        let non_persistent_sets = [
            Vec::<String>::new(),
            vec!["tmp_stats".to_string()],
            vec!["tmp_stats".to_string(), "scratch".to_string()],
        ];

        for parameter_keys in parameter_sets {
            for persistent_buffer_keys in persistent_sets.clone() {
                for non_persistent_buffer_keys in non_persistent_sets.clone() {
                    let case = NnStateCase {
                        name: "property_export".to_string(),
                        operation: "state_export".to_string(),
                        module_path: "root".to_string(),
                        state_key: None,
                        state_key_kind: None,
                        parameter_keys: parameter_keys.clone(),
                        persistent_buffer_keys: persistent_buffer_keys.clone(),
                        non_persistent_buffer_keys: non_persistent_buffer_keys.clone(),
                        expected_state_keys: Vec::new(),
                        training_transitions: Vec::new(),
                        initial_training: None,
                        expected_training_flag: None,
                        missing_keys: Vec::new(),
                        unexpected_keys: Vec::new(),
                        incompatible_shapes: Vec::new(),
                        prefix_keys: Vec::new(),
                        expected_canonical_keys: Vec::new(),
                        allow_prefix_normalization: None,
                        hook_trace: Vec::new(),
                        expected_hook_trace: Vec::new(),
                        assign_flag: None,
                        strict: expectation.clone(),
                        hardened: expectation.clone(),
                        contract_ids: Vec::new(),
                        e2e_scenarios: Vec::new(),
                    };
                    let exported = nn_state_export_keys(&case);
                    for key in non_persistent_buffer_keys {
                        assert!(
                            !exported.contains(&key),
                            "non-persistent key leaked into state export: {key}"
                        );
                    }
                }
            }
        }
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
    fn scheduler_logs_include_packet_004_telemetry_fields() {
        let cfg = HarnessConfig::default_paths();
        let (_, cases) = run_autograd_scheduler_conformance(&cfg, ExecutionMode::Strict)
            .expect("scheduler conformance should run");

        assert!(!cases.is_empty(), "expected at least one scheduler case");
        let value =
            serde_json::to_value(&cases[0].forensic_log).expect("scheduler log should serialize");
        for key in [
            "execution_order",
            "queue_pushes",
            "queue_pops",
            "max_queue_len",
            "dependency_snapshot",
            "reentrant_depth",
            "reentrant_guard_triggered",
            "hardened_fallback_used",
        ] {
            assert!(
                value.get(key).is_some(),
                "missing scheduler telemetry key {key}"
            );
        }
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
    fn e2e_matrix_unfiltered_keeps_ft_p2c_005_projection_entries() {
        let cfg = HarnessConfig::default_paths();
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map_or(0, |duration| duration.as_millis());
        let output_path = std::env::temp_dir().join(format!(
            "ft_conformance_e2e_unfiltered_projection_{}_{}.jsonl",
            std::process::id(),
            now
        ));

        let summary = emit_e2e_forensics_matrix_filtered(
            &cfg,
            output_path.as_path(),
            &[ExecutionMode::Strict, ExecutionMode::Hardened],
            None,
        )
        .expect("unfiltered e2e matrix should emit logs");

        assert!(summary.log_entries >= 20);
        let raw = fs::read_to_string(&output_path).expect("jsonl output should be readable");
        let mut saw_ft_p2c_001 = false;
        let mut saw_ft_p2c_002 = false;
        let mut saw_ft_p2c_007 = false;
        let mut saw_ft_p2c_008 = false;
        let mut ft_p2c_005_suites = BTreeSet::new();
        for line in raw.lines() {
            let value: serde_json::Value =
                serde_json::from_str(line).expect("jsonl line should be valid json");
            let packet_id = value
                .get("packet_id")
                .and_then(serde_json::Value::as_str)
                .expect("packet_id must be present");
            let suite_id = value
                .get("suite_id")
                .and_then(serde_json::Value::as_str)
                .expect("suite_id must be present");
            match packet_id {
                "FT-P2C-001" if matches!(suite_id, "scalar_dac" | "tensor_meta") => {
                    saw_ft_p2c_001 = true;
                }
                "FT-P2C-002" if suite_id == "dispatch_key" => {
                    saw_ft_p2c_002 = true;
                }
                "FT-P2C-005" => {
                    let scenario_id = value
                        .get("scenario_id")
                        .and_then(serde_json::Value::as_str)
                        .expect("scenario_id must be present");
                    assert!(
                        scenario_id.starts_with("ft_p2c_005/"),
                        "FT-P2C-005 projection must retain namespaced scenario IDs"
                    );
                    ft_p2c_005_suites.insert(suite_id.to_string());
                }
                "FT-P2C-007" if suite_id == "dispatch_key" => {
                    saw_ft_p2c_007 = true;
                }
                "FT-P2C-008" if suite_id == "nn_state" => {
                    saw_ft_p2c_008 = true;
                }
                _ => {}
            }
        }

        assert!(saw_ft_p2c_001, "expected FT-P2C-001 source entries");
        assert!(saw_ft_p2c_002, "expected FT-P2C-002 source entries");
        assert!(
            saw_ft_p2c_007,
            "expected FT-P2C-007 projected dispatch entries"
        );
        assert!(saw_ft_p2c_008, "expected FT-P2C-008 nn_state entries");
        assert!(
            ft_p2c_005_suites.contains("scalar_dac")
                && ft_p2c_005_suites.contains("tensor_meta")
                && ft_p2c_005_suites.contains("dispatch_key"),
            "expected FT-P2C-005 projected suites from scalar/tensor_meta/dispatch sources"
        );

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
    fn e2e_matrix_packet_filter_includes_op_schema_packet_entries() {
        let cfg = HarnessConfig::default_paths();
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map_or(0, |duration| duration.as_millis());
        let output_path = std::env::temp_dir().join(format!(
            "ft_conformance_e2e_packet_filter_op_schema_{}_{}.jsonl",
            std::process::id(),
            now
        ));

        let summary = emit_e2e_forensics_matrix_filtered(
            &cfg,
            output_path.as_path(),
            &[ExecutionMode::Strict, ExecutionMode::Hardened],
            Some("FT-P2C-003"),
        )
        .expect("packet-filtered e2e matrix should emit logs");

        assert!(summary.log_entries > 0);
        let raw = fs::read_to_string(&output_path).expect("jsonl output should be readable");
        for line in raw.lines() {
            let value: serde_json::Value =
                serde_json::from_str(line).expect("jsonl line should be valid json");
            assert_eq!(
                value.get("packet_id").and_then(serde_json::Value::as_str),
                Some("FT-P2C-003")
            );
            assert_eq!(
                value.get("suite_id").and_then(serde_json::Value::as_str),
                Some("op_schema")
            );
        }

        let _ = fs::remove_file(output_path);
    }

    #[test]
    fn e2e_matrix_packet_filter_includes_autograd_scheduler_packet_entries() {
        let cfg = HarnessConfig::default_paths();
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map_or(0, |duration| duration.as_millis());
        let output_path = std::env::temp_dir().join(format!(
            "ft_conformance_e2e_packet_filter_autograd_scheduler_{}_{}.jsonl",
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

        assert_eq!(summary.log_entries, 6);
        let raw = fs::read_to_string(&output_path).expect("jsonl output should be readable");
        for line in raw.lines() {
            let value: serde_json::Value =
                serde_json::from_str(line).expect("jsonl line should be valid json");
            assert_eq!(
                value.get("packet_id").and_then(serde_json::Value::as_str),
                Some("FT-P2C-004")
            );
            assert_eq!(
                value.get("suite_id").and_then(serde_json::Value::as_str),
                Some("autograd_scheduler")
            );
        }

        let _ = fs::remove_file(output_path);
    }

    #[test]
    fn e2e_matrix_packet_filter_includes_cpu_kernel_packet_entries() {
        let cfg = HarnessConfig::default_paths();
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map_or(0, |duration| duration.as_millis());
        let output_path = std::env::temp_dir().join(format!(
            "ft_conformance_e2e_packet_filter_cpu_kernel_{}_{}.jsonl",
            std::process::id(),
            now
        ));

        let summary = emit_e2e_forensics_matrix_filtered(
            &cfg,
            output_path.as_path(),
            &[ExecutionMode::Strict, ExecutionMode::Hardened],
            Some("FT-P2C-005"),
        )
        .expect("packet-filtered e2e matrix should emit logs");

        assert!(summary.log_entries >= 24);
        let raw = fs::read_to_string(&output_path).expect("jsonl output should be readable");
        let mut suites = BTreeSet::new();
        for line in raw.lines() {
            let value: serde_json::Value =
                serde_json::from_str(line).expect("jsonl line should be valid json");
            assert_eq!(
                value.get("packet_id").and_then(serde_json::Value::as_str),
                Some("FT-P2C-005")
            );
            assert!(
                value.get("contract_ids").is_some(),
                "missing contract_ids field for FT-P2C-005 log"
            );
            let suite = value
                .get("suite_id")
                .and_then(serde_json::Value::as_str)
                .expect("suite_id must be present");
            suites.insert(suite.to_string());
        }

        assert!(suites.contains("scalar_dac"));
        assert!(suites.contains("dispatch_key"));
        assert!(suites.contains("tensor_meta"));
        let _ = fs::remove_file(output_path);
    }

    #[test]
    fn ft_p2c_005_projection_strips_shadowed_flatten_keys() {
        let mut extra_fields = BTreeMap::new();
        extra_fields.insert("mode".to_string(), json!("shadowed_mode"));
        extra_fields.insert("reason_code".to_string(), json!("shadowed_reason"));
        extra_fields.insert("contract_ids".to_string(), json!(["CPU-KERNEL-001"]));

        let base_log = StructuredCaseLog::new(
            "scalar_dac",
            "scalar_autograd_cases.json",
            "FT-P2C-001",
            "add_basic",
            ExecutionMode::Strict,
            vec!["artifacts/phase2c/FT-P2C-001/parity_report.json".to_string()],
            "cargo test -p ft-conformance strict_scalar_conformance_is_green -- --nocapture"
                .to_string(),
            "pass",
            "parity_ok",
        )
        .with_extra_fields(extra_fields);

        let projected = project_log_to_ft_p2c_005(base_log);
        assert_eq!(projected.packet_id, "FT-P2C-005");
        assert!(
            projected.scenario_id.starts_with("ft_p2c_005/"),
            "scenario must be namespaced under FT-P2C-005 projection"
        );
        assert!(
            !projected.extra_fields.contains_key("mode"),
            "mode must stay only at top-level envelope"
        );
        assert!(
            !projected.extra_fields.contains_key("reason_code"),
            "reason_code must stay only at top-level envelope"
        );
        assert!(
            projected.extra_fields.contains_key("contract_ids"),
            "non-shadowed forensic fields must be preserved"
        );
    }

    #[test]
    fn ft_p2c_007_projection_strips_shadowed_flatten_keys() {
        let mut extra_fields = BTreeMap::new();
        extra_fields.insert("mode".to_string(), json!("shadowed_mode"));
        extra_fields.insert("reason_code".to_string(), json!("shadowed_reason"));
        extra_fields.insert("contract_ids".to_string(), json!(["DEVICE-GUARD-001"]));

        let base_log = StructuredCaseLog::new(
            "dispatch_key",
            "dispatch_key_cases.json",
            "FT-P2C-002",
            "strict_cpu_route",
            ExecutionMode::Strict,
            vec!["artifacts/phase2c/FT-P2C-002/parity_report.json".to_string()],
            "cargo test -p ft-conformance strict_dispatch_conformance_is_green -- --nocapture"
                .to_string(),
            "pass",
            "dispatch_parity_ok",
        )
        .with_extra_fields(extra_fields);

        let projected = project_log_to_ft_p2c_007(base_log);
        assert_eq!(projected.packet_id, "FT-P2C-007");
        assert!(
            projected.scenario_id.starts_with("ft_p2c_007/"),
            "scenario must be namespaced under FT-P2C-007 projection"
        );
        assert!(
            projected.replay_command.contains("--packet FT-P2C-007"),
            "replay command must target packet FT-P2C-007"
        );
        assert!(
            projected
                .artifact_refs
                .contains(&"artifacts/phase2c/FT-P2C-007/contract_table.md".to_string()),
            "projected artifact refs must include packet-007 contract table"
        );
        assert!(
            projected.artifact_refs.contains(
                &"artifacts/phase2c/FT-P2C-007/unit_property_quality_report_v1.json".to_string()
            ),
            "projected artifact refs must include packet-007 unit/property evidence"
        );
        assert!(
            !projected.extra_fields.contains_key("mode"),
            "mode must stay only at top-level envelope"
        );
        assert!(
            !projected.extra_fields.contains_key("reason_code"),
            "reason_code must stay only at top-level envelope"
        );
        assert!(
            projected.extra_fields.contains_key("contract_ids"),
            "non-shadowed forensic fields must be preserved"
        );
    }

    #[test]
    fn e2e_matrix_packet_filter_includes_serialization_packet_entries() {
        let cfg = HarnessConfig::default_paths();
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map_or(0, |duration| duration.as_millis());
        let output_path = std::env::temp_dir().join(format!(
            "ft_conformance_e2e_packet_filter_serialization_{}_{}.jsonl",
            std::process::id(),
            now
        ));

        let summary = emit_e2e_forensics_matrix_filtered(
            &cfg,
            output_path.as_path(),
            &[ExecutionMode::Strict, ExecutionMode::Hardened],
            Some("FT-P2C-006"),
        )
        .expect("packet-filtered e2e matrix should emit logs");

        assert_eq!(summary.log_entries, 4);
        let raw = fs::read_to_string(&output_path).expect("jsonl output should be readable");
        for line in raw.lines() {
            let value: serde_json::Value =
                serde_json::from_str(line).expect("jsonl line should be valid json");
            assert_eq!(
                value.get("packet_id").and_then(serde_json::Value::as_str),
                Some("FT-P2C-006")
            );
            assert_eq!(
                value.get("suite_id").and_then(serde_json::Value::as_str),
                Some("serialization")
            );
        }

        let _ = fs::remove_file(output_path);
    }

    #[test]
    fn e2e_matrix_packet_filter_includes_device_guard_packet_entries() {
        let cfg = HarnessConfig::default_paths();
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map_or(0, |duration| duration.as_millis());
        let output_path = std::env::temp_dir().join(format!(
            "ft_conformance_e2e_packet_filter_device_guard_{}_{}.jsonl",
            std::process::id(),
            now
        ));

        let summary = emit_e2e_forensics_matrix_filtered(
            &cfg,
            output_path.as_path(),
            &[ExecutionMode::Strict, ExecutionMode::Hardened],
            Some("FT-P2C-007"),
        )
        .expect("packet-filtered e2e matrix should emit logs");

        assert_eq!(summary.log_entries, 20);
        let raw = fs::read_to_string(&output_path).expect("jsonl output should be readable");
        for line in raw.lines() {
            let value: serde_json::Value =
                serde_json::from_str(line).expect("jsonl line should be valid json");
            assert_eq!(
                value.get("packet_id").and_then(serde_json::Value::as_str),
                Some("FT-P2C-007")
            );
            assert_eq!(
                value.get("suite_id").and_then(serde_json::Value::as_str),
                Some("dispatch_key")
            );
            let scenario = value
                .get("scenario_id")
                .and_then(serde_json::Value::as_str)
                .expect("scenario_id must be present");
            assert!(
                scenario.starts_with("ft_p2c_007/dispatch_key/"),
                "FT-P2C-007 projection must retain namespaced scenario IDs"
            );
        }

        let _ = fs::remove_file(output_path);
    }

    #[test]
    fn e2e_matrix_packet_filter_includes_nn_state_packet_entries() {
        let cfg = HarnessConfig::default_paths();
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map_or(0, |duration| duration.as_millis());
        let output_path = std::env::temp_dir().join(format!(
            "ft_conformance_e2e_packet_filter_nn_state_{}_{}.jsonl",
            std::process::id(),
            now
        ));

        let summary = emit_e2e_forensics_matrix_filtered(
            &cfg,
            output_path.as_path(),
            &[ExecutionMode::Strict, ExecutionMode::Hardened],
            Some("FT-P2C-008"),
        )
        .expect("packet-filtered e2e matrix should emit logs");

        assert_eq!(summary.log_entries, 20);
        let raw = fs::read_to_string(&output_path).expect("jsonl output should be readable");
        for line in raw.lines() {
            let value: serde_json::Value =
                serde_json::from_str(line).expect("jsonl line should be valid json");
            assert_eq!(
                value.get("packet_id").and_then(serde_json::Value::as_str),
                Some("FT-P2C-008")
            );
            assert_eq!(
                value.get("suite_id").and_then(serde_json::Value::as_str),
                Some("nn_state")
            );
            let scenario = value
                .get("scenario_id")
                .and_then(serde_json::Value::as_str)
                .expect("scenario_id must be present");
            assert!(
                scenario.starts_with("nn_state/"),
                "FT-P2C-008 logs must remain under nn_state scenario namespace"
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
    fn packet_e2e_microbench_op_schema_produces_percentiles() {
        let report = run_packet_e2e_microbench(&HarnessConfig::default_paths(), 10, "FT-P2C-003")
            .expect("packet e2e microbench should run");
        eprintln!(
            "packet_e2e_microbench_ns packet=FT-P2C-003 p50={} p95={} p99={} mean={}",
            report.p50_ns, report.p95_ns, report.p99_ns, report.mean_ns
        );
        assert_eq!(report.iterations, 10);
        assert!(report.p50_ns > 0);
        assert!(report.p95_ns >= report.p50_ns);
        assert!(report.p99_ns >= report.p95_ns);
    }

    #[test]
    fn packet_e2e_microbench_autograd_scheduler_produces_percentiles() {
        let report = run_packet_e2e_microbench(&HarnessConfig::default_paths(), 10, "FT-P2C-004")
            .expect("packet e2e microbench should run");
        eprintln!(
            "packet_e2e_microbench_ns packet=FT-P2C-004 p50={} p95={} p99={} mean={}",
            report.p50_ns, report.p95_ns, report.p99_ns, report.mean_ns
        );
        assert_eq!(report.iterations, 10);
        assert!(report.p50_ns > 0);
        assert!(report.p95_ns >= report.p50_ns);
        assert!(report.p99_ns >= report.p95_ns);
    }

    #[test]
    fn packet_e2e_microbench_serialization_produces_percentiles() {
        let report = run_packet_e2e_microbench(&HarnessConfig::default_paths(), 10, "FT-P2C-006")
            .expect("packet e2e microbench should run");
        eprintln!(
            "packet_e2e_microbench_ns packet=FT-P2C-006 p50={} p95={} p99={} mean={}",
            report.p50_ns, report.p95_ns, report.p99_ns, report.mean_ns
        );
        assert_eq!(report.iterations, 10);
        assert!(report.p50_ns > 0);
        assert!(report.p95_ns >= report.p50_ns);
        assert!(report.p99_ns >= report.p95_ns);
    }

    #[test]
    fn packet_e2e_microbench_cpu_kernel_produces_percentiles() {
        let report = run_packet_e2e_microbench(&HarnessConfig::default_paths(), 10, "FT-P2C-005")
            .expect("packet e2e microbench should run");
        eprintln!(
            "packet_e2e_microbench_ns packet=FT-P2C-005 p50={} p95={} p99={} mean={}",
            report.p50_ns, report.p95_ns, report.p99_ns, report.mean_ns
        );
        assert_eq!(report.iterations, 10);
        assert!(report.p50_ns > 0);
        assert!(report.p95_ns >= report.p50_ns);
        assert!(report.p99_ns >= report.p95_ns);
    }

    #[test]
    fn packet_e2e_microbench_device_guard_produces_percentiles() {
        let report = run_packet_e2e_microbench(&HarnessConfig::default_paths(), 10, "FT-P2C-007")
            .expect("packet e2e microbench should run");
        eprintln!(
            "packet_e2e_microbench_ns packet=FT-P2C-007 p50={} p95={} p99={} mean={}",
            report.p50_ns, report.p95_ns, report.p99_ns, report.mean_ns
        );
        assert_eq!(report.iterations, 10);
        assert!(report.p50_ns > 0);
        assert!(report.p95_ns >= report.p50_ns);
        assert!(report.p99_ns >= report.p95_ns);
    }

    #[test]
    fn packet_e2e_microbench_nn_state_produces_percentiles() {
        let report = run_packet_e2e_microbench(&HarnessConfig::default_paths(), 10, "FT-P2C-008")
            .expect("packet e2e microbench should run");
        eprintln!(
            "packet_e2e_microbench_ns packet=FT-P2C-008 p50={} p95={} p99={} mean={}",
            report.p50_ns, report.p95_ns, report.p99_ns, report.mean_ns
        );
        assert_eq!(report.iterations, 10);
        assert!(report.p50_ns > 0);
        assert!(report.p95_ns >= report.p50_ns);
        assert!(report.p99_ns >= report.p95_ns);
    }

    #[test]
    fn packet_e2e_microbench_cpu_kernel_legacy_vs_optimized_profiles() {
        let cfg = HarnessConfig::default_paths();
        let legacy = run_packet_e2e_microbench_legacy(&cfg, 10, "FT-P2C-005")
            .expect("legacy packet microbench should run");
        let optimized = run_packet_e2e_microbench(&cfg, 10, "FT-P2C-005")
            .expect("optimized packet microbench should run");

        eprintln!(
            "packet_e2e_microbench_compare_ns packet=FT-P2C-005 legacy_p50={} legacy_p95={} legacy_p99={} legacy_mean={} optimized_p50={} optimized_p95={} optimized_p99={} optimized_mean={}",
            legacy.p50_ns,
            legacy.p95_ns,
            legacy.p99_ns,
            legacy.mean_ns,
            optimized.p50_ns,
            optimized.p95_ns,
            optimized.p99_ns,
            optimized.mean_ns
        );
        assert_eq!(legacy.iterations, optimized.iterations);
        assert!(legacy.p50_ns > 0 && optimized.p50_ns > 0);
    }

    #[test]
    fn packet_e2e_microbench_nn_state_legacy_vs_optimized_profiles() {
        let cfg = HarnessConfig::default_paths();
        let legacy = run_packet_e2e_microbench_legacy(&cfg, 10, "FT-P2C-008")
            .expect("legacy packet microbench should run");
        let optimized = run_packet_e2e_microbench(&cfg, 10, "FT-P2C-008")
            .expect("optimized packet microbench should run");

        eprintln!(
            "packet_e2e_microbench_compare_ns packet=FT-P2C-008 legacy_p50={} legacy_p95={} legacy_p99={} legacy_mean={} optimized_p50={} optimized_p95={} optimized_p99={} optimized_mean={}",
            legacy.p50_ns,
            legacy.p95_ns,
            legacy.p99_ns,
            legacy.mean_ns,
            optimized.p50_ns,
            optimized.p95_ns,
            optimized.p99_ns,
            optimized.mean_ns
        );
        assert_eq!(legacy.iterations, optimized.iterations);
        assert!(legacy.p50_ns > 0 && optimized.p50_ns > 0);
    }

    #[test]
    fn packet_e2e_microbench_device_guard_legacy_vs_optimized_profiles() {
        let cfg = HarnessConfig::default_paths();
        let legacy = run_packet_e2e_microbench_legacy(&cfg, 10, "FT-P2C-007")
            .expect("legacy packet microbench should run");
        let optimized = run_packet_e2e_microbench(&cfg, 10, "FT-P2C-007")
            .expect("optimized packet microbench should run");

        eprintln!(
            "packet_e2e_microbench_compare_ns packet=FT-P2C-007 legacy_p50={} legacy_p95={} legacy_p99={} legacy_mean={} optimized_p50={} optimized_p95={} optimized_p99={} optimized_mean={}",
            legacy.p50_ns,
            legacy.p95_ns,
            legacy.p99_ns,
            legacy.mean_ns,
            optimized.p50_ns,
            optimized.p95_ns,
            optimized.p99_ns,
            optimized.mean_ns
        );
        assert_eq!(legacy.iterations, optimized.iterations);
        assert!(legacy.p50_ns > 0 && optimized.p50_ns > 0);
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
    fn differential_dispatch_adds_metamorphic_and_adversarial_checks() {
        let cfg = HarnessConfig::default_paths();
        let report = run_differential_conformance(&cfg, &[ExecutionMode::Strict])
            .expect("differential report should run");

        assert!(report.checks.iter().any(|check| {
            check.suite == "dispatch_key" && check.comparator == "metamorphic_commutative_local"
        }));
        assert!(report.checks.iter().any(|check| {
            check.suite == "dispatch_key"
                && check.case_name == "adversarial_unknown_key"
                && check.comparator == "adversarial_unknown_key_rejected"
        }));
        assert!(report.checks.iter().any(|check| {
            check.suite == "dispatch_key"
                && check.case_name == "adversarial_autograd_without_cpu"
                && check.comparator == "adversarial_autograd_without_cpu_rejected"
        }));
    }

    #[test]
    fn differential_op_schema_adds_metamorphic_and_adversarial_checks() {
        let cfg = HarnessConfig::default_paths();
        let report = run_differential_conformance(&cfg, &[ExecutionMode::Strict])
            .expect("differential report should run");

        assert!(report.checks.iter().any(|check| {
            check.suite == "op_schema"
                && check.case_name == "operator_name_normalization"
                && check.comparator == "metamorphic_name_normalization"
        }));
        assert!(report.checks.iter().any(|check| {
            check.suite == "op_schema"
                && check.case_name == "malformed_schema_rejected"
                && check.comparator == "adversarial_malformed_schema_rejected"
        }));
        assert!(report.checks.iter().any(|check| {
            check.suite == "op_schema"
                && check.case_name == "dispatch_metadata_incompatible"
                && check.comparator == "adversarial_dispatch_metadata_rejected"
        }));
    }

    #[test]
    fn differential_autograd_scheduler_adds_metamorphic_and_adversarial_checks() {
        let cfg = HarnessConfig::default_paths();
        let report =
            run_differential_conformance(&cfg, &[ExecutionMode::Strict, ExecutionMode::Hardened])
                .expect("differential report should run");

        assert!(report.checks.iter().any(|check| {
            check.suite == "autograd_scheduler"
                && check.comparator == "metamorphic_scale_relation_local"
        }));
        if report.oracle.available {
            assert!(report.checks.iter().any(|check| {
                check.suite == "autograd_scheduler"
                    && check.comparator == "metamorphic_scale_relation_oracle"
            }));
        }
        assert!(report.checks.iter().any(|check| {
            check.suite == "autograd_scheduler"
                && check.mode == "strict"
                && check.comparator == "adversarial_strict_reentrant_overflow_rejected"
        }));
        assert!(report.checks.iter().any(|check| {
            check.suite == "autograd_scheduler"
                && check.mode == "hardened"
                && check.comparator == "adversarial_hardened_reentrant_overflow_guarded"
        }));
    }

    #[test]
    fn differential_serialization_adds_metamorphic_and_adversarial_checks() {
        let cfg = HarnessConfig::default_paths();
        let report =
            run_differential_conformance(&cfg, &[ExecutionMode::Strict, ExecutionMode::Hardened])
                .expect("differential report should run");

        assert!(report.checks.iter().any(|check| {
            check.suite == "serialization"
                && check.case_name == "checkpoint_basic"
                && check.comparator == "metamorphic_entry_order_hash_invariant"
        }));
        assert!(report.checks.iter().any(|check| {
            check.suite == "serialization"
                && check.comparator == "adversarial_unknown_field_rejected"
        }));
        assert!(report.checks.iter().any(|check| {
            check.suite == "serialization"
                && check.comparator == "adversarial_checksum_tamper_rejected"
        }));
        assert!(report.checks.iter().any(|check| {
            check.suite == "serialization"
                && check.mode == "hardened"
                && check.comparator == "policy"
                && check.drift_id.as_deref() == Some("serialization.bounded_malformed_diagnostic")
        }));
    }

    #[test]
    fn differential_nn_state_adds_metamorphic_and_adversarial_checks() {
        let cfg = HarnessConfig::default_paths();
        let report =
            run_differential_conformance(&cfg, &[ExecutionMode::Strict, ExecutionMode::Hardened])
                .expect("differential report should run");

        assert!(report.checks.iter().any(|check| {
            check.suite == "nn_state"
                && check.comparator == "metamorphic_state_export_order_invariant"
        }));
        assert!(report.checks.iter().any(|check| {
            check.suite == "nn_state" && check.comparator == "adversarial_assign_shape_rejected"
        }));
        assert!(report.checks.iter().any(|check| {
            check.suite == "nn_state"
                && check.mode == "hardened"
                && check.comparator == "policy"
                && check.drift_id.as_deref() == Some("nn_state.non_strict_missing_unexpected")
                && check.status == "allowlisted_drift"
        }));
        assert!(report.checks.iter().any(|check| {
            check.suite == "nn_state"
                && check.mode == "strict"
                && check.case_name == "prefix_consumption_maps_ddp_state_dict_keys"
                && check.comparator == "metamorphic_prefix_normalization_idempotent"
                && check.status == "pass"
        }));
        assert!(report.checks.iter().any(|check| {
            check.suite == "nn_state"
                && check.mode == "hardened"
                && check.case_name == "prefix_consumption_maps_ddp_state_dict_keys"
                && check.comparator == "metamorphic_prefix_normalization_idempotent"
                && check.status == "pass"
        }));
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
    fn differential_report_packet_filter_limits_output() {
        let cfg = HarnessConfig::default_paths();
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map_or(0, |duration| duration.as_millis());
        let output_path = std::env::temp_dir().join(format!(
            "ft_conformance_differential_packet_filter_{}_{}.json",
            std::process::id(),
            now
        ));

        let report = emit_differential_report_filtered(
            &cfg,
            output_path.as_path(),
            &[ExecutionMode::Strict, ExecutionMode::Hardened],
            Some("FT-P2C-003"),
        )
        .expect("packet-filtered differential report should emit");
        assert!(report.total_checks > 0);
        assert_eq!(report.total_checks, report.checks.len());
        assert!(
            report
                .checks
                .iter()
                .all(|check| check.packet_id == "FT-P2C-003"),
            "packet filter must only emit FT-P2C-003 checks"
        );

        let _ = fs::remove_file(output_path);
    }

    #[test]
    fn differential_report_packet_filter_projects_ft_p2c_005() {
        let cfg = HarnessConfig::default_paths();
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map_or(0, |duration| duration.as_millis());
        let output_path = std::env::temp_dir().join(format!(
            "ft_conformance_differential_packet_projection_{}_{}.json",
            std::process::id(),
            now
        ));

        let report = emit_differential_report_filtered(
            &cfg,
            output_path.as_path(),
            &[ExecutionMode::Strict, ExecutionMode::Hardened],
            Some("FT-P2C-005"),
        )
        .expect("FT-P2C-005 differential projection should emit");
        assert!(report.total_checks > 0);
        assert_eq!(report.total_checks, report.checks.len());
        assert!(
            report
                .checks
                .iter()
                .all(|check| check.packet_id == "FT-P2C-005"),
            "projection must retag checks to FT-P2C-005"
        );
        assert!(
            report
                .checks
                .iter()
                .any(|check| { check.scenario_id.starts_with("ft_p2c_005/FT-P2C-001/") })
        );
        assert!(
            report
                .checks
                .iter()
                .any(|check| { check.scenario_id.starts_with("ft_p2c_005/FT-P2C-002/") })
        );
        assert!(report.checks.iter().all(|check| {
            check
                .evidence_refs
                .iter()
                .any(|path| path == "artifacts/phase2c/FT-P2C-005/contract_table.md")
        }));
        assert!(
            report
                .checks
                .iter()
                .any(|check| check.comparator.contains("metamorphic")),
            "projection should retain metamorphic comparators"
        );
        assert!(
            report
                .checks
                .iter()
                .any(|check| check.comparator.contains("adversarial")
                    || check.comparator == "fail_closed"),
            "projection should retain adversarial/fail-closed comparators"
        );

        let _ = fs::remove_file(output_path);
    }

    #[test]
    fn allowlist_contains_known_hardened_deviations() {
        let cfg = HarnessConfig::default_paths();
        let allowlist =
            load_allowlist(cfg.allowlist_path.as_path()).expect("allowlist fixture should parse");
        assert!(allowlist.contains("FT-P2C-002", "dispatch.composite_backend_fallback"));
        assert!(allowlist.contains("FT-P2C-004", "autograd.reentrant_depth_bounded_fallback"));
        assert!(allowlist.contains("FT-P2C-008", "nn_state.non_strict_missing_unexpected"));
    }
}

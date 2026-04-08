#![forbid(unsafe_code)]

use std::collections::BTreeMap;
use std::fs;
use std::hint::black_box;
use std::path::{Path, PathBuf};
use std::time::Instant;

use allocation_counter::{AllocationInfo, measure};
use ft_api::FrankenTorchSession;
use ft_autograd::TensorNodeId;
use ft_conformance::perf_slo::{DEFAULT_REPORT_PATH, DEFAULT_SUMMARY_PATH, REPORT_SCHEMA_VERSION};
use ft_core::{DenseTensor, Device, ExecutionMode};
use ft_optim::{Adam, Optimizer};
use ft_serialize::{load_state_dict, save_state_dict};
use serde::{Deserialize, Serialize};

const DEFAULT_ITERATIONS: usize = 100;
const DEFAULT_WARMUP_ITERATIONS: usize = 1;
const CHECKPOINT_BYTES_PER_SECOND_TARGET: f64 = 350_000_000.0;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
enum RunProfile {
    Full,
    Smoke,
}

impl RunProfile {
    fn config(self) -> WorkloadConfig {
        match self {
            Self::Full => WorkloadConfig {
                elementwise_numel: 100_000_000,
                matmul_dim: 1_024,
                representative_batch: 128,
                representative_in_features: 256,
                representative_hidden_features: 256,
                representative_out_features: 128,
                training_batch: 128,
                training_in_features: 128,
                training_hidden_features: 256,
                training_out_features: 64,
                alloc_batch: 128,
                alloc_width: 256,
                alloc_depth: 4,
                alloc_out_features: 64,
                optimizer_total_params: 10_000_000,
                optimizer_param_chunks: 10,
                checkpoint_total_bytes: 2_147_483_648,
                checkpoint_chunks: 8,
            },
            Self::Smoke => WorkloadConfig {
                elementwise_numel: 100_000,
                matmul_dim: 64,
                representative_batch: 8,
                representative_in_features: 32,
                representative_hidden_features: 64,
                representative_out_features: 16,
                training_batch: 8,
                training_in_features: 16,
                training_hidden_features: 32,
                training_out_features: 8,
                alloc_batch: 8,
                alloc_width: 32,
                alloc_depth: 3,
                alloc_out_features: 8,
                optimizer_total_params: 10_000,
                optimizer_param_chunks: 5,
                checkpoint_total_bytes: 16 * 1_024 * 1_024,
                checkpoint_chunks: 4,
            },
        }
    }
}

#[derive(Debug, Clone, Copy)]
struct WorkloadConfig {
    elementwise_numel: usize,
    matmul_dim: usize,
    representative_batch: usize,
    representative_in_features: usize,
    representative_hidden_features: usize,
    representative_out_features: usize,
    training_batch: usize,
    training_in_features: usize,
    training_hidden_features: usize,
    training_out_features: usize,
    alloc_batch: usize,
    alloc_width: usize,
    alloc_depth: usize,
    alloc_out_features: usize,
    optimizer_total_params: usize,
    optimizer_param_chunks: usize,
    checkpoint_total_bytes: u64,
    checkpoint_chunks: usize,
}

#[derive(Debug, Clone)]
struct CliArgs {
    profile: RunProfile,
    output_path: PathBuf,
    summary_path: PathBuf,
    baseline_path: Option<PathBuf>,
    iterations: usize,
    warmup_iterations: usize,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
enum MeasurementStatus {
    Ok,
    Unsupported,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
enum BudgetStatus {
    Pass,
    Fail,
    BaselineRequired,
    Informational,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
enum ReportStatus {
    Pass,
    Partial,
    Fail,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct PerfSloReport {
    schema_version: String,
    generated_unix_ms: u128,
    profile: RunProfile,
    iterations: usize,
    warmup_iterations: usize,
    baseline_report: Option<String>,
    overall_status: ReportStatus,
    measurements: Measurements,
    budgets: Vec<BudgetEvaluation>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct Measurements {
    elementwise_forward: ElementwiseMeasurement,
    representative_backward: RepresentativeBackwardMeasurement,
    matmul_core: MatmulMeasurement,
    optimizer_step: OptimizerMeasurement,
    checkpoint_roundtrip: CheckpointMeasurement,
    training_step_memory: TrainingStepMeasurement,
    backward_heavy_allocation: BackwardAllocationMeasurement,
    benchmark_family_tail: TailStabilityMeasurement,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct LatencySummary {
    samples_ns: Vec<u128>,
    p50_ns: u128,
    p95_ns: u128,
    p99_ns: u128,
    mean_ns: u128,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct ThroughputSummary {
    samples_mb_per_s: Vec<f64>,
    p50_mb_per_s: f64,
    p95_mb_per_s: f64,
    p99_mb_per_s: f64,
    mean_mb_per_s: f64,
}

#[derive(Debug, Clone, Copy, Default, Serialize, Deserialize)]
struct AllocationMetrics {
    count_total: u64,
    count_current: i64,
    count_max: u64,
    bytes_total: u64,
    bytes_current: i64,
    bytes_max: u64,
}

impl AllocationMetrics {
    fn add_assign(&mut self, info: AllocationInfo) {
        self.count_total = self.count_total.saturating_add(info.count_total);
        self.count_current = self.count_current.saturating_add(info.count_current);
        self.count_max = self.count_max.max(info.count_max);
        self.bytes_total = self.bytes_total.saturating_add(info.bytes_total);
        self.bytes_current = self.bytes_current.saturating_add(info.bytes_current);
        self.bytes_max = self.bytes_max.max(info.bytes_max);
    }

    fn allocation_ops_total(self) -> u64 {
        self.count_total
    }
}

#[derive(Debug, Clone, Copy, Default, Serialize, Deserialize)]
struct MemoryMetrics {
    sampled_peak_rss_kb: u64,
    sampled_final_rss_kb: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct ElementwiseMeasurement {
    status: MeasurementStatus,
    dtype: String,
    numel: usize,
    latency: Option<LatencySummary>,
    allocation: AllocationMetrics,
    memory: MemoryMetrics,
    notes: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct RepresentativeBackwardMeasurement {
    status: MeasurementStatus,
    batch: usize,
    in_features: usize,
    hidden_features: usize,
    out_features: usize,
    forward_latency: Option<LatencySummary>,
    backward_latency: Option<LatencySummary>,
    backward_to_forward_p95_ratio: Option<f64>,
    allocation: AllocationMetrics,
    memory: MemoryMetrics,
    notes: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct MatmulMeasurement {
    status: MeasurementStatus,
    dtype: String,
    lhs_shape: [usize; 2],
    rhs_shape: [usize; 2],
    latency: Option<LatencySummary>,
    allocation: AllocationMetrics,
    memory: MemoryMetrics,
    notes: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct OptimizerMeasurement {
    status: MeasurementStatus,
    optimizer: String,
    total_params: usize,
    parameter_tensors: usize,
    latency: Option<LatencySummary>,
    allocation: AllocationMetrics,
    memory: MemoryMetrics,
    notes: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct CheckpointMeasurement {
    status: MeasurementStatus,
    total_payload_bytes: u64,
    tensor_chunks: usize,
    save_latency: Option<LatencySummary>,
    load_latency: Option<LatencySummary>,
    save_throughput: Option<ThroughputSummary>,
    load_throughput: Option<ThroughputSummary>,
    allocation: AllocationMetrics,
    memory: MemoryMetrics,
    notes: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct TrainingStepMeasurement {
    status: MeasurementStatus,
    batch: usize,
    in_features: usize,
    hidden_features: usize,
    out_features: usize,
    latency: Option<LatencySummary>,
    allocation: AllocationMetrics,
    memory: MemoryMetrics,
    notes: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct BackwardAllocationMeasurement {
    status: MeasurementStatus,
    batch: usize,
    width: usize,
    depth: usize,
    out_features: usize,
    latency: Option<LatencySummary>,
    allocation: AllocationMetrics,
    memory: MemoryMetrics,
    notes: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct TailStabilityMeasurement {
    status: MeasurementStatus,
    workload_ids: Vec<String>,
    current_family_p99_ns: Option<u128>,
    baseline_family_p99_ns: Option<u128>,
    regression_pct: Option<f64>,
    notes: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct BudgetEvaluation {
    id: String,
    title: String,
    status: BudgetStatus,
    observed_value: Option<f64>,
    limit_value: f64,
    unit: String,
    comparison: String,
    baseline_value: Option<f64>,
    delta_pct: Option<f64>,
    detail: String,
}

#[derive(Debug, Clone, Copy)]
struct RegressionBudgetArgs<'a> {
    id: &'a str,
    title: &'a str,
    limit_pct: f64,
    unit: &'a str,
    detail: &'a str,
}

fn main() -> Result<(), String> {
    let args = parse_args()?;
    let baseline = args
        .baseline_path
        .as_ref()
        .map(|path| load_report(path.as_path()))
        .transpose()?;
    let config = args.profile.config();

    let mut measurements = Measurements {
        elementwise_forward: measure_elementwise_forward(&args, &config)?,
        representative_backward: measure_representative_backward(&args, &config)?,
        matmul_core: measure_matmul_core(&args, &config)?,
        optimizer_step: measure_optimizer_step(&args, &config)?,
        checkpoint_roundtrip: measure_checkpoint_roundtrip(&args, &config)?,
        training_step_memory: measure_training_step_memory(&args, &config)?,
        backward_heavy_allocation: measure_backward_heavy_allocation(&args, &config)?,
        benchmark_family_tail: TailStabilityMeasurement {
            status: MeasurementStatus::Ok,
            workload_ids: Vec::new(),
            current_family_p99_ns: None,
            baseline_family_p99_ns: None,
            regression_pct: None,
            notes: Vec::new(),
        },
    };
    measurements.benchmark_family_tail = derive_tail_stability(&measurements, baseline.as_ref());

    let budgets = evaluate_budgets(&args, &measurements, baseline.as_ref());
    let report = PerfSloReport {
        schema_version: REPORT_SCHEMA_VERSION.to_string(),
        generated_unix_ms: now_unix_ms(),
        profile: args.profile,
        iterations: args.iterations,
        warmup_iterations: args.warmup_iterations,
        baseline_report: args
            .baseline_path
            .as_ref()
            .map(|path| path.display().to_string()),
        overall_status: overall_status(budgets.as_slice()),
        measurements,
        budgets,
    };

    write_report(
        &report,
        args.output_path.as_path(),
        args.summary_path.as_path(),
    )?;
    println!(
        "{}",
        serde_json::to_string_pretty(&report)
            .map_err(|error| format!("failed to serialize perf SLO report: {error}"))?
    );
    Ok(())
}

fn parse_args() -> Result<CliArgs, String> {
    let mut profile = RunProfile::Full;
    let mut output_path = PathBuf::from(DEFAULT_REPORT_PATH);
    let mut summary_path = PathBuf::from(DEFAULT_SUMMARY_PATH);
    let mut baseline_path = None;
    let mut iterations = DEFAULT_ITERATIONS;
    let mut warmup_iterations = DEFAULT_WARMUP_ITERATIONS;

    let mut args = std::env::args().skip(1);
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--profile" => {
                let raw = args
                    .next()
                    .ok_or_else(|| "--profile requires full or smoke".to_string())?;
                profile = parse_profile(raw.as_str())?;
            }
            "--output" => {
                let raw = args
                    .next()
                    .ok_or_else(|| "--output requires a file path".to_string())?;
                output_path = PathBuf::from(raw);
            }
            "--summary" => {
                let raw = args
                    .next()
                    .ok_or_else(|| "--summary requires a file path".to_string())?;
                summary_path = PathBuf::from(raw);
            }
            "--baseline" => {
                let raw = args
                    .next()
                    .ok_or_else(|| "--baseline requires a file path".to_string())?;
                baseline_path = Some(PathBuf::from(raw));
            }
            "--iterations" => {
                let raw = args
                    .next()
                    .ok_or_else(|| "--iterations requires a positive integer".to_string())?;
                iterations = raw
                    .parse::<usize>()
                    .map_err(|error| format!("invalid --iterations value '{raw}': {error}"))?;
            }
            "--warmup-iterations" => {
                let raw = args.next().ok_or_else(|| {
                    "--warmup-iterations requires a non-negative integer".to_string()
                })?;
                warmup_iterations = raw.parse::<usize>().map_err(|error| {
                    format!("invalid --warmup-iterations value '{raw}': {error}")
                })?;
            }
            other => {
                return Err(format!(
                    "unknown arg '{other}'. usage: run_perf_slos [--profile full|smoke] [--output <path>] [--summary <path>] [--baseline <path>] [--iterations <n>] [--warmup-iterations <n>]"
                ));
            }
        }
    }

    if iterations == 0 {
        return Err("--iterations must be > 0".to_string());
    }

    Ok(CliArgs {
        profile,
        output_path,
        summary_path,
        baseline_path,
        iterations,
        warmup_iterations,
    })
}

fn parse_profile(raw: &str) -> Result<RunProfile, String> {
    match raw {
        "full" => Ok(RunProfile::Full),
        "smoke" => Ok(RunProfile::Smoke),
        _ => Err(format!(
            "invalid --profile value '{raw}'; expected 'full' or 'smoke'"
        )),
    }
}

fn load_report(path: &Path) -> Result<PerfSloReport, String> {
    let raw = fs::read_to_string(path)
        .map_err(|error| format!("failed to read baseline report {}: {error}", path.display()))?;
    serde_json::from_str(raw.as_str()).map_err(|error| {
        format!(
            "failed to parse baseline report {}: {error}",
            path.display()
        )
    })
}

fn write_report(
    report: &PerfSloReport,
    json_path: &Path,
    summary_path: &Path,
) -> Result<(), String> {
    if let Some(parent) = json_path.parent() {
        fs::create_dir_all(parent).map_err(|error| {
            format!(
                "failed to create perf report directory {}: {error}",
                parent.display()
            )
        })?;
    }
    if let Some(parent) = summary_path.parent() {
        fs::create_dir_all(parent).map_err(|error| {
            format!(
                "failed to create perf summary directory {}: {error}",
                parent.display()
            )
        })?;
    }

    let json = serde_json::to_string_pretty(report)
        .map_err(|error| format!("failed to serialize perf report: {error}"))?;
    fs::write(json_path, json).map_err(|error| {
        format!(
            "failed to write perf report {}: {error}",
            json_path.display()
        )
    })?;
    fs::write(summary_path, render_summary(report)).map_err(|error| {
        format!(
            "failed to write perf summary {}: {error}",
            summary_path.display()
        )
    })?;
    Ok(())
}

fn render_summary(report: &PerfSloReport) -> String {
    let mut out = String::new();
    out.push_str("# FrankenTorch Performance SLO Summary\n\n");
    out.push_str(&format!(
        "- schema_version: `{}`\n- generated_unix_ms: `{}`\n- profile: `{}`\n- iterations: `{}`\n- warmup_iterations: `{}`\n- overall_status: `{}`\n",
        report.schema_version,
        report.generated_unix_ms,
        profile_str(report.profile),
        report.iterations,
        report.warmup_iterations,
        report_status_str(report.overall_status)
    ));
    if let Some(path) = &report.baseline_report {
        out.push_str(&format!("- baseline_report: `{path}`\n"));
    }
    out.push_str("\n| Budget | Status | Observed | Limit | Detail |\n");
    out.push_str("|---|---|---|---|---|\n");
    for budget in &report.budgets {
        let observed = budget
            .observed_value
            .map(|value| format!("{value:.3} {}", budget.unit))
            .unwrap_or_else(|| "n/a".to_string());
        let limit = if budget.unit == "n/a" {
            "n/a".to_string()
        } else {
            format!("{:.3} {}", budget.limit_value, budget.unit)
        };
        out.push_str(&format!(
            "| {} | {} | {} | {} | {} |\n",
            budget.title,
            budget_status_str(budget.status),
            observed,
            limit,
            budget.detail.replace('\n', " ")
        ));
    }
    out
}

fn measure_elementwise_forward(
    args: &CliArgs,
    config: &WorkloadConfig,
) -> Result<ElementwiseMeasurement, String> {
    let lhs = DenseTensor::from_contiguous_values_f32(
        vec![1.25_f32; config.elementwise_numel],
        vec![config.elementwise_numel],
        Device::Cpu,
    )
    .map_err(|error| format!("failed to build elementwise lhs tensor: {error}"))?;
    let rhs = DenseTensor::from_contiguous_values_f32(
        vec![0.75_f32; config.elementwise_numel],
        vec![config.elementwise_numel],
        Device::Cpu,
    )
    .map_err(|error| format!("failed to build elementwise rhs tensor: {error}"))?;
    let mut samples = Vec::with_capacity(args.iterations);
    let mut allocation = AllocationMetrics::default();
    let mut memory = MemoryMetrics::default();

    for _ in 0..args.warmup_iterations {
        let _ = run_elementwise_iteration(&lhs, &rhs)?;
    }
    for _ in 0..args.iterations {
        let (elapsed, info) = run_elementwise_iteration(&lhs, &rhs)?;
        samples.push(elapsed);
        allocation.add_assign(info);
        update_memory_metrics(&mut memory, current_rss_kb()?);
    }

    Ok(ElementwiseMeasurement {
        status: MeasurementStatus::Ok,
        dtype: "f32".to_string(),
        numel: config.elementwise_numel,
        latency: Some(latency_summary(samples)),
        allocation,
        memory,
        notes: vec![
            "Measures eager tensor_add on two prebuilt f32 tensors.".to_string(),
            "RSS is sampled from /proc/self/status after each measured iteration.".to_string(),
        ],
    })
}

fn run_elementwise_iteration(
    lhs_storage: &DenseTensor,
    rhs_storage: &DenseTensor,
) -> Result<(u128, AllocationInfo), String> {
    let mut session = FrankenTorchSession::new(ExecutionMode::Strict);
    let lhs = session.tensor_variable_from_storage(lhs_storage.clone(), false);
    let rhs = session.tensor_variable_from_storage(rhs_storage.clone(), false);
    let mut elapsed = 0_u128;
    let mut outcome = Ok(());
    let info = measure(|| {
        let started = Instant::now();
        outcome = session
            .tensor_add(lhs, rhs)
            .map(|out| {
                black_box(out);
            })
            .map_err(|error| format!("elementwise add failed: {error}"));
        elapsed = started.elapsed().as_nanos();
    });
    outcome?;
    Ok((elapsed, info))
}

fn measure_representative_backward(
    args: &CliArgs,
    config: &WorkloadConfig,
) -> Result<RepresentativeBackwardMeasurement, String> {
    let shared = SharedGraphTensors::representative(config)?;
    let mut forward_samples = Vec::with_capacity(args.iterations);
    let mut backward_samples = Vec::with_capacity(args.iterations);
    let mut allocation = AllocationMetrics::default();
    let mut memory = MemoryMetrics::default();

    for _ in 0..args.warmup_iterations {
        let _ = run_representative_backward_iteration(&shared)?;
    }
    for _ in 0..args.iterations {
        let (forward_ns, backward_ns, info) = run_representative_backward_iteration(&shared)?;
        forward_samples.push(forward_ns);
        backward_samples.push(backward_ns);
        allocation.add_assign(info);
        update_memory_metrics(&mut memory, current_rss_kb()?);
    }

    let forward_latency = latency_summary(forward_samples);
    let backward_latency = latency_summary(backward_samples);
    Ok(RepresentativeBackwardMeasurement {
        status: MeasurementStatus::Ok,
        batch: config.representative_batch,
        in_features: config.representative_in_features,
        hidden_features: config.representative_hidden_features,
        out_features: config.representative_out_features,
        forward_latency: Some(forward_latency.clone()),
        backward_latency: Some(backward_latency.clone()),
        backward_to_forward_p95_ratio: Some(ratio(forward_latency.p95_ns, backward_latency.p95_ns)),
        allocation,
        memory,
        notes: vec![
            "Uses a two-layer MLP-shaped graph with MSE loss.".to_string(),
            "Allocation counts cover the tensor_backward section only.".to_string(),
        ],
    })
}

fn run_representative_backward_iteration(
    shared: &SharedGraphTensors,
) -> Result<(u128, u128, AllocationInfo), String> {
    let mut session = FrankenTorchSession::new(ExecutionMode::Strict);
    let x = session.tensor_variable_from_storage(shared.input.clone(), false);
    let target = session.tensor_variable_from_storage(shared.target.clone(), false);
    let w1 = session.tensor_variable_from_storage(shared.weight1.clone(), true);
    let b1 = session.tensor_variable_from_storage(shared.bias1.clone(), true);
    let w2 = session.tensor_variable_from_storage(shared.weight2.clone(), true);
    let b2 = session.tensor_variable_from_storage(shared.bias2.clone(), true);

    let started = Instant::now();
    let pred = forward_graph(&mut session, x, w1, b1, w2, b2)?;
    let loss = session
        .mse_loss(pred, target)
        .map_err(|error| format!("representative mse_loss failed: {error}"))?;
    let forward_ns = started.elapsed().as_nanos();

    let mut backward_ns = 0_u128;
    let mut outcome = Ok(());
    let info = measure(|| {
        let started = Instant::now();
        outcome = session
            .tensor_backward(loss)
            .map(|report| {
                black_box(report);
            })
            .map_err(|error| format!("representative tensor_backward failed: {error}"));
        backward_ns = started.elapsed().as_nanos();
    });
    outcome?;

    Ok((forward_ns, backward_ns, info))
}

fn measure_matmul_core(
    args: &CliArgs,
    config: &WorkloadConfig,
) -> Result<MatmulMeasurement, String> {
    let lhs = DenseTensor::from_contiguous_values_f32(
        vec![0.5_f32; checked_square_numel(config.matmul_dim)?],
        vec![config.matmul_dim, config.matmul_dim],
        Device::Cpu,
    )
    .map_err(|error| format!("failed to build matmul lhs tensor: {error}"))?;
    let rhs = DenseTensor::from_contiguous_values_f32(
        vec![1.5_f32; checked_square_numel(config.matmul_dim)?],
        vec![config.matmul_dim, config.matmul_dim],
        Device::Cpu,
    )
    .map_err(|error| format!("failed to build matmul rhs tensor: {error}"))?;
    let mut samples = Vec::with_capacity(args.iterations);
    let mut allocation = AllocationMetrics::default();
    let mut memory = MemoryMetrics::default();

    for _ in 0..args.warmup_iterations {
        let _ = run_matmul_iteration(&lhs, &rhs)?;
    }
    for _ in 0..args.iterations {
        let (elapsed, info) = run_matmul_iteration(&lhs, &rhs)?;
        samples.push(elapsed);
        allocation.add_assign(info);
        update_memory_metrics(&mut memory, current_rss_kb()?);
    }

    Ok(MatmulMeasurement {
        status: MeasurementStatus::Ok,
        dtype: "f32".to_string(),
        lhs_shape: [config.matmul_dim, config.matmul_dim],
        rhs_shape: [config.matmul_dim, config.matmul_dim],
        latency: Some(latency_summary(samples)),
        allocation,
        memory,
        notes: vec!["Measures eager tensor_matmul on square dense matrices.".to_string()],
    })
}

fn run_matmul_iteration(
    lhs_storage: &DenseTensor,
    rhs_storage: &DenseTensor,
) -> Result<(u128, AllocationInfo), String> {
    let mut session = FrankenTorchSession::new(ExecutionMode::Strict);
    let lhs = session.tensor_variable_from_storage(lhs_storage.clone(), false);
    let rhs = session.tensor_variable_from_storage(rhs_storage.clone(), false);
    let mut elapsed = 0_u128;
    let mut outcome = Ok(());
    let info = measure(|| {
        let started = Instant::now();
        outcome = session
            .tensor_matmul(lhs, rhs)
            .map(|out| {
                black_box(out);
            })
            .map_err(|error| format!("matmul failed: {error}"));
        elapsed = started.elapsed().as_nanos();
    });
    outcome?;
    Ok((elapsed, info))
}

fn measure_optimizer_step(
    args: &CliArgs,
    config: &WorkloadConfig,
) -> Result<OptimizerMeasurement, String> {
    let chunk_len = config
        .optimizer_total_params
        .checked_div(config.optimizer_param_chunks)
        .ok_or_else(|| "optimizer_param_chunks must be > 0".to_string())?;
    let params = (0..config.optimizer_param_chunks)
        .map(|_| {
            DenseTensor::from_contiguous_values(
                vec![0.25_f64; chunk_len],
                vec![chunk_len],
                Device::Cpu,
            )
            .map_err(|error| format!("failed to build optimizer parameter tensor: {error}"))
        })
        .collect::<Result<Vec<_>, _>>()?;
    let gradient = vec![0.01_f64; chunk_len];
    let mut samples = Vec::with_capacity(args.iterations);
    let mut allocation = AllocationMetrics::default();
    let mut memory = MemoryMetrics::default();

    for _ in 0..args.warmup_iterations {
        let _ = run_optimizer_iteration(params.as_slice(), gradient.as_slice())?;
    }
    for _ in 0..args.iterations {
        let (elapsed, info) = run_optimizer_iteration(params.as_slice(), gradient.as_slice())?;
        samples.push(elapsed);
        allocation.add_assign(info);
        update_memory_metrics(&mut memory, current_rss_kb()?);
    }

    Ok(OptimizerMeasurement {
        status: MeasurementStatus::Ok,
        optimizer: "Adam".to_string(),
        total_params: config.optimizer_total_params,
        parameter_tensors: config.optimizer_param_chunks,
        latency: Some(latency_summary(samples)),
        allocation,
        memory,
        notes: vec!["Measures Adam::step with pre-populated persistent gradients.".to_string()],
    })
}

fn run_optimizer_iteration(
    param_templates: &[DenseTensor],
    gradient: &[f64],
) -> Result<(u128, AllocationInfo), String> {
    let mut session = FrankenTorchSession::new(ExecutionMode::Strict);
    let params = param_templates
        .iter()
        .map(|tensor| session.tensor_variable_from_storage(tensor.clone(), true))
        .collect::<Vec<_>>();
    for &param in &params {
        session
            .tensor_set_accumulated_gradient(param, gradient.to_vec())
            .map_err(|error| format!("failed to seed optimizer gradient: {error}"))?;
    }
    let dummy = session
        .tensor_variable(vec![1.0], vec![1], true)
        .map_err(|error| format!("failed to create optimizer dummy tensor: {error}"))?;
    let dummy_sum = session
        .tensor_sum(dummy)
        .map_err(|error| format!("failed to sum optimizer dummy tensor: {error}"))?;
    let report = session
        .tensor_backward(dummy_sum)
        .map_err(|error| format!("failed to build optimizer dummy report: {error}"))?;
    let mut optimizer = Adam::new(params, 1e-3);

    let mut elapsed = 0_u128;
    let mut outcome = Ok(());
    let info = measure(|| {
        let started = Instant::now();
        outcome = optimizer
            .step(&mut session, &report)
            .map_err(|error| format!("adam step failed: {error}"));
        elapsed = started.elapsed().as_nanos();
    });
    outcome?;
    Ok((elapsed, info))
}

fn measure_checkpoint_roundtrip(
    args: &CliArgs,
    config: &WorkloadConfig,
) -> Result<CheckpointMeasurement, String> {
    let available_kb = mem_available_kb().unwrap_or(0);
    let required_kb = config
        .checkpoint_total_bytes
        .saturating_mul(7)
        .checked_div(2)
        .unwrap_or(u64::MAX)
        .checked_div(1_024)
        .unwrap_or(u64::MAX);

    if available_kb > 0 && available_kb < required_kb {
        return Ok(CheckpointMeasurement {
            status: MeasurementStatus::Unsupported,
            total_payload_bytes: config.checkpoint_total_bytes,
            tensor_chunks: config.checkpoint_chunks,
            save_latency: None,
            load_latency: None,
            save_throughput: None,
            load_throughput: None,
            allocation: AllocationMetrics::default(),
            memory: MemoryMetrics::default(),
            notes: vec![format!(
                "Skipping checkpoint measurement: target_bytes={} requires roughly {} KiB available, observed {} KiB.",
                config.checkpoint_total_bytes, required_kb, available_kb
            )],
        });
    }

    let state_dict = build_state_dict(config.checkpoint_total_bytes, config.checkpoint_chunks)?;
    let path = std::env::temp_dir().join(format!(
        "frankentorch_perf_slo_{}_{}.ftsv",
        std::process::id(),
        config.checkpoint_total_bytes
    ));
    let mut save_samples = Vec::with_capacity(args.iterations);
    let mut load_samples = Vec::with_capacity(args.iterations);
    let mut save_throughput_samples = Vec::with_capacity(args.iterations);
    let mut load_throughput_samples = Vec::with_capacity(args.iterations);
    let mut allocation = AllocationMetrics::default();
    let mut memory = MemoryMetrics::default();

    for _ in 0..args.warmup_iterations {
        let _ = run_checkpoint_iteration(&state_dict, path.as_path())?;
    }
    for _ in 0..args.iterations {
        let (save_ns, load_ns, info) = run_checkpoint_iteration(&state_dict, path.as_path())?;
        save_samples.push(save_ns);
        load_samples.push(load_ns);
        save_throughput_samples.push(bytes_per_second_mb(config.checkpoint_total_bytes, save_ns));
        load_throughput_samples.push(bytes_per_second_mb(config.checkpoint_total_bytes, load_ns));
        allocation.add_assign(info);
        update_memory_metrics(&mut memory, current_rss_kb()?);
    }

    if path.exists() {
        fs::remove_file(path.as_path()).map_err(|error| {
            format!(
                "failed to remove temporary checkpoint file {}: {error}",
                path.display()
            )
        })?;
    }

    Ok(CheckpointMeasurement {
        status: MeasurementStatus::Ok,
        total_payload_bytes: config.checkpoint_total_bytes,
        tensor_chunks: config.checkpoint_chunks,
        save_latency: Some(latency_summary(save_samples)),
        load_latency: Some(latency_summary(load_samples)),
        save_throughput: Some(throughput_summary(save_throughput_samples)),
        load_throughput: Some(throughput_summary(load_throughput_samples)),
        allocation,
        memory,
        notes: vec![
            "Uses ft_serialize::save_state_dict and load_state_dict on a dense f32 state dict."
                .to_string(),
            "Target size is a native state-dict payload envelope, not a JSON checkpoint payload."
                .to_string(),
        ],
    })
}

fn run_checkpoint_iteration(
    state_dict: &BTreeMap<String, DenseTensor>,
    path: &Path,
) -> Result<(u128, u128, AllocationInfo), String> {
    let mut save_elapsed = 0_u128;
    let mut load_elapsed = 0_u128;
    let mut outcome = Ok(());
    let info = measure(|| {
        let save_started = Instant::now();
        outcome = save_state_dict(state_dict, path)
            .map_err(|error| format!("save_state_dict failed: {error}"));
        save_elapsed = save_started.elapsed().as_nanos();
        if outcome.is_err() {
            return;
        }

        let load_started = Instant::now();
        outcome = load_state_dict(path)
            .map(|loaded| {
                black_box(loaded.len());
            })
            .map_err(|error| format!("load_state_dict failed: {error}"));
        load_elapsed = load_started.elapsed().as_nanos();
    });
    outcome?;
    Ok((save_elapsed, load_elapsed, info))
}

fn measure_training_step_memory(
    args: &CliArgs,
    config: &WorkloadConfig,
) -> Result<TrainingStepMeasurement, String> {
    let shared = SharedGraphTensors::training(config)?;
    let mut samples = Vec::with_capacity(args.iterations);
    let mut allocation = AllocationMetrics::default();
    let mut memory = MemoryMetrics::default();

    for _ in 0..args.warmup_iterations {
        let _ = run_training_step_iteration(&shared)?;
    }
    for _ in 0..args.iterations {
        let (elapsed, info) = run_training_step_iteration(&shared)?;
        samples.push(elapsed);
        allocation.add_assign(info);
        update_memory_metrics(&mut memory, current_rss_kb()?);
    }

    Ok(TrainingStepMeasurement {
        status: MeasurementStatus::Ok,
        batch: config.training_batch,
        in_features: config.training_in_features,
        hidden_features: config.training_hidden_features,
        out_features: config.training_out_features,
        latency: Some(latency_summary(samples)),
        allocation,
        memory,
        notes: vec!["Profiles a full strict-mode training step.".to_string()],
    })
}

fn run_training_step_iteration(
    shared: &SharedGraphTensors,
) -> Result<(u128, AllocationInfo), String> {
    let mut session = FrankenTorchSession::new(ExecutionMode::Strict);
    let x = session.tensor_variable_from_storage(shared.input.clone(), false);
    let target = session.tensor_variable_from_storage(shared.target.clone(), false);
    let w1 = session.tensor_variable_from_storage(shared.weight1.clone(), true);
    let b1 = session.tensor_variable_from_storage(shared.bias1.clone(), true);
    let w2 = session.tensor_variable_from_storage(shared.weight2.clone(), true);
    let b2 = session.tensor_variable_from_storage(shared.bias2.clone(), true);
    let mut optimizer = Adam::new(vec![w1, b1, w2, b2], 1e-3);

    let mut elapsed = 0_u128;
    let mut outcome = Ok(());
    let info = measure(|| {
        let started = Instant::now();
        outcome = (|| {
            let pred = forward_graph(&mut session, x, w1, b1, w2, b2)?;
            let loss = session
                .mse_loss(pred, target)
                .map_err(|error| format!("training mse_loss failed: {error}"))?;
            let report = session
                .tensor_backward(loss)
                .map_err(|error| format!("training tensor_backward failed: {error}"))?;
            optimizer
                .step(&mut session, &report)
                .map_err(|error| format!("training adam step failed: {error}"))?;
            Ok::<(), String>(())
        })();
        elapsed = started.elapsed().as_nanos();
    });
    outcome?;
    Ok((elapsed, info))
}

fn measure_backward_heavy_allocation(
    args: &CliArgs,
    config: &WorkloadConfig,
) -> Result<BackwardAllocationMeasurement, String> {
    let shared = SharedAllocationGraph::new(config)?;
    let mut samples = Vec::with_capacity(args.iterations);
    let mut allocation = AllocationMetrics::default();
    let mut memory = MemoryMetrics::default();

    for _ in 0..args.warmup_iterations {
        let _ = run_backward_heavy_iteration(&shared)?;
    }
    for _ in 0..args.iterations {
        let (elapsed, info) = run_backward_heavy_iteration(&shared)?;
        samples.push(elapsed);
        allocation.add_assign(info);
        update_memory_metrics(&mut memory, current_rss_kb()?);
    }

    Ok(BackwardAllocationMeasurement {
        status: MeasurementStatus::Ok,
        batch: shared.batch,
        width: shared.width,
        depth: shared.depth,
        out_features: shared.out_features,
        latency: Some(latency_summary(samples)),
        allocation,
        memory,
        notes: vec![
            "Times and counts allocations for tensor_backward on a deeper graph.".to_string(),
        ],
    })
}

fn run_backward_heavy_iteration(
    shared: &SharedAllocationGraph,
) -> Result<(u128, AllocationInfo), String> {
    let mut session = FrankenTorchSession::new(ExecutionMode::Strict);
    let mut activation = session.tensor_variable_from_storage(shared.input.clone(), false);
    for layer in &shared.layers {
        let weight = session.tensor_variable_from_storage(layer.weight.clone(), true);
        let bias = session.tensor_variable_from_storage(layer.bias.clone(), true);
        activation = linear_relu(&mut session, activation, weight, bias)?;
    }
    let target = session.tensor_variable_from_storage(shared.target.clone(), false);
    let loss = session
        .mse_loss(activation, target)
        .map_err(|error| format!("backward-heavy mse_loss failed: {error}"))?;

    let mut elapsed = 0_u128;
    let mut outcome = Ok(());
    let info = measure(|| {
        let started = Instant::now();
        outcome = session
            .tensor_backward(loss)
            .map(|report| {
                black_box(report);
            })
            .map_err(|error| format!("backward-heavy tensor_backward failed: {error}"));
        elapsed = started.elapsed().as_nanos();
    });
    outcome?;
    Ok((elapsed, info))
}

fn derive_tail_stability(
    measurements: &Measurements,
    baseline: Option<&PerfSloReport>,
) -> TailStabilityMeasurement {
    let mut workload_ids = Vec::new();
    let mut current = Vec::new();

    if let Some(latency) = measurements
        .elementwise_forward
        .latency
        .as_ref()
        .filter(|_| measurements.elementwise_forward.status == MeasurementStatus::Ok)
    {
        workload_ids.push("elementwise_forward".to_string());
        current.push(latency.p99_ns);
    }
    if let Some(latency) = measurements
        .representative_backward
        .backward_latency
        .as_ref()
        .filter(|_| measurements.representative_backward.status == MeasurementStatus::Ok)
    {
        workload_ids.push("representative_backward".to_string());
        current.push(latency.p99_ns);
    }
    if let Some(latency) = measurements
        .matmul_core
        .latency
        .as_ref()
        .filter(|_| measurements.matmul_core.status == MeasurementStatus::Ok)
    {
        workload_ids.push("matmul_core".to_string());
        current.push(latency.p99_ns);
    }
    if let Some(latency) = measurements
        .optimizer_step
        .latency
        .as_ref()
        .filter(|_| measurements.optimizer_step.status == MeasurementStatus::Ok)
    {
        workload_ids.push("optimizer_step".to_string());
        current.push(latency.p99_ns);
    }
    if let Some(latency) = measurements
        .training_step_memory
        .latency
        .as_ref()
        .filter(|_| measurements.training_step_memory.status == MeasurementStatus::Ok)
    {
        workload_ids.push("training_step_memory".to_string());
        current.push(latency.p99_ns);
    }
    if let Some(latency) = measurements
        .backward_heavy_allocation
        .latency
        .as_ref()
        .filter(|_| measurements.backward_heavy_allocation.status == MeasurementStatus::Ok)
    {
        workload_ids.push("backward_heavy_allocation".to_string());
        current.push(latency.p99_ns);
    }

    let current_family_p99_ns = current.into_iter().max();
    let baseline_family_p99_ns = baseline.and_then(|report| {
        report
            .measurements
            .benchmark_family_tail
            .current_family_p99_ns
    });

    TailStabilityMeasurement {
        status: MeasurementStatus::Ok,
        workload_ids,
        current_family_p99_ns,
        baseline_family_p99_ns,
        regression_pct: current_family_p99_ns
            .zip(baseline_family_p99_ns)
            .map(|(observed, base)| regression_pct(base, observed)),
        notes: vec![
            "Tail stability is derived from the max p99 across the measured benchmark family."
                .to_string(),
        ],
    }
}

fn evaluate_budgets(
    args: &CliArgs,
    measurements: &Measurements,
    baseline: Option<&PerfSloReport>,
) -> Vec<BudgetEvaluation> {
    if args.profile != RunProfile::Full {
        return smoke_profile_budgets();
    }

    vec![
        evaluate_latency_budget(
            "elementwise_forward",
            "Elementwise forward",
            measurements
                .elementwise_forward
                .latency
                .as_ref()
                .map(|latency| ns_to_ms(latency.p95_ns)),
            200.0,
            measurements.elementwise_forward.status,
            "p95_ms",
        ),
        evaluate_latency_budget(
            "representative_backward",
            "Backward pass",
            measurements
                .representative_backward
                .backward_to_forward_p95_ratio,
            1.35,
            measurements.representative_backward.status,
            "ratio",
        ),
        evaluate_latency_budget(
            "matmul_core",
            "Matmul core",
            measurements
                .matmul_core
                .latency
                .as_ref()
                .map(|latency| ns_to_ms(latency.p95_ns)),
            240.0,
            measurements.matmul_core.status,
            "p95_ms",
        ),
        evaluate_latency_budget(
            "optimizer_step",
            "Optimizer step",
            measurements
                .optimizer_step
                .latency
                .as_ref()
                .map(|latency| ns_to_ms(latency.p95_ns)),
            130.0,
            measurements.optimizer_step.status,
            "p95_ms",
        ),
        evaluate_checkpoint_budget(&measurements.checkpoint_roundtrip),
        evaluate_regression_budget(
            RegressionBudgetArgs {
                id: "training_step_memory",
                title: "Memory footprint",
                limit_pct: 8.0,
                unit: "pct",
                detail: "peak RSS regression versus baseline training-step profile",
            },
            Some(measurements.training_step_memory.memory.sampled_peak_rss_kb as f64),
            baseline.map(|report| {
                report
                    .measurements
                    .training_step_memory
                    .memory
                    .sampled_peak_rss_kb as f64
            }),
            measurements.training_step_memory.status,
        ),
        evaluate_regression_budget(
            RegressionBudgetArgs {
                id: "backward_heavy_allocation",
                title: "Allocation churn",
                limit_pct: 10.0,
                unit: "pct",
                detail: "allocation-op regression versus baseline backward-heavy trace",
            },
            Some(
                measurements
                    .backward_heavy_allocation
                    .allocation
                    .allocation_ops_total() as f64,
            ),
            baseline.map(|report| {
                report
                    .measurements
                    .backward_heavy_allocation
                    .allocation
                    .allocation_ops_total() as f64
            }),
            measurements.backward_heavy_allocation.status,
        ),
        evaluate_regression_budget(
            RegressionBudgetArgs {
                id: "benchmark_family_tail",
                title: "Tail stability",
                limit_pct: 7.0,
                unit: "pct",
                detail: "family p99 regression versus baseline benchmark family",
            },
            measurements
                .benchmark_family_tail
                .current_family_p99_ns
                .map(|value| value as f64),
            measurements
                .benchmark_family_tail
                .baseline_family_p99_ns
                .map(|value| value as f64),
            measurements.benchmark_family_tail.status,
        ),
    ]
}

fn smoke_profile_budgets() -> Vec<BudgetEvaluation> {
    [
        ("elementwise_forward", "Elementwise forward"),
        ("representative_backward", "Backward pass"),
        ("matmul_core", "Matmul core"),
        ("optimizer_step", "Optimizer step"),
        ("checkpoint_roundtrip", "Checkpoint save/load"),
        ("training_step_memory", "Memory footprint"),
        ("backward_heavy_allocation", "Allocation churn"),
        ("benchmark_family_tail", "Tail stability"),
    ]
    .iter()
    .map(|(id, title)| BudgetEvaluation {
        id: (*id).to_string(),
        title: (*title).to_string(),
        status: BudgetStatus::Informational,
        observed_value: None,
        limit_value: 0.0,
        unit: "n/a".to_string(),
        comparison: "n/a".to_string(),
        baseline_value: None,
        delta_pct: None,
        detail: "smoke profile uses reduced envelopes and is not spec-comparable".to_string(),
    })
    .collect()
}

fn evaluate_latency_budget(
    id: &str,
    title: &str,
    observed: Option<f64>,
    limit: f64,
    status: MeasurementStatus,
    unit: &str,
) -> BudgetEvaluation {
    match (status, observed) {
        (MeasurementStatus::Unsupported, _) => BudgetEvaluation {
            id: id.to_string(),
            title: title.to_string(),
            status: BudgetStatus::Fail,
            observed_value: None,
            limit_value: limit,
            unit: unit.to_string(),
            comparison: "<=".to_string(),
            baseline_value: None,
            delta_pct: None,
            detail: "workload could not be measured under the requested envelope".to_string(),
        },
        (_, Some(value)) if value <= limit => BudgetEvaluation {
            id: id.to_string(),
            title: title.to_string(),
            status: BudgetStatus::Pass,
            observed_value: Some(value),
            limit_value: limit,
            unit: unit.to_string(),
            comparison: "<=".to_string(),
            baseline_value: None,
            delta_pct: None,
            detail: "observed metric is within the spec budget".to_string(),
        },
        (_, Some(value)) => BudgetEvaluation {
            id: id.to_string(),
            title: title.to_string(),
            status: BudgetStatus::Fail,
            observed_value: Some(value),
            limit_value: limit,
            unit: unit.to_string(),
            comparison: "<=".to_string(),
            baseline_value: None,
            delta_pct: Some(value - limit),
            detail: "observed metric exceeds the spec budget".to_string(),
        },
        _ => BudgetEvaluation {
            id: id.to_string(),
            title: title.to_string(),
            status: BudgetStatus::Fail,
            observed_value: None,
            limit_value: limit,
            unit: unit.to_string(),
            comparison: "<=".to_string(),
            baseline_value: None,
            delta_pct: None,
            detail: "measurement did not produce an observed value".to_string(),
        },
    }
}

fn evaluate_checkpoint_budget(measurement: &CheckpointMeasurement) -> BudgetEvaluation {
    let limit_ms = bytes_to_ms(
        measurement.total_payload_bytes,
        CHECKPOINT_BYTES_PER_SECOND_TARGET,
    );
    match (
        measurement.status,
        measurement.save_latency.as_ref(),
        measurement.load_latency.as_ref(),
    ) {
        (MeasurementStatus::Unsupported, _, _) => BudgetEvaluation {
            id: "checkpoint_roundtrip".to_string(),
            title: "Checkpoint save/load".to_string(),
            status: BudgetStatus::Fail,
            observed_value: None,
            limit_value: limit_ms,
            unit: "p95_ms".to_string(),
            comparison: "<=".to_string(),
            baseline_value: None,
            delta_pct: None,
            detail: measurement.notes.join(" "),
        },
        (_, Some(save), Some(load)) => {
            let observed_ms = ns_to_ms(save.p95_ns.max(load.p95_ns));
            BudgetEvaluation {
                id: "checkpoint_roundtrip".to_string(),
                title: "Checkpoint save/load".to_string(),
                status: if observed_ms <= limit_ms {
                    BudgetStatus::Pass
                } else {
                    BudgetStatus::Fail
                },
                observed_value: Some(observed_ms),
                limit_value: limit_ms,
                unit: "p95_ms".to_string(),
                comparison: "<=".to_string(),
                baseline_value: None,
                delta_pct: Some(observed_ms - limit_ms),
                detail:
                    "compares the slower of save/load p95 against the latency implied by 350 MB/s"
                        .to_string(),
            }
        }
        _ => BudgetEvaluation {
            id: "checkpoint_roundtrip".to_string(),
            title: "Checkpoint save/load".to_string(),
            status: BudgetStatus::Fail,
            observed_value: None,
            limit_value: limit_ms,
            unit: "p95_ms".to_string(),
            comparison: "<=".to_string(),
            baseline_value: None,
            delta_pct: None,
            detail: "checkpoint measurement did not produce both save and load timings".to_string(),
        },
    }
}

fn evaluate_regression_budget(
    args: RegressionBudgetArgs<'_>,
    observed: Option<f64>,
    baseline: Option<f64>,
    status: MeasurementStatus,
) -> BudgetEvaluation {
    if status == MeasurementStatus::Unsupported {
        return BudgetEvaluation {
            id: args.id.to_string(),
            title: args.title.to_string(),
            status: BudgetStatus::Fail,
            observed_value: observed,
            limit_value: args.limit_pct,
            unit: args.unit.to_string(),
            comparison: "<=".to_string(),
            baseline_value: baseline,
            delta_pct: None,
            detail: "workload could not be measured under the requested envelope".to_string(),
        };
    }

    match (observed, baseline) {
        (Some(value), Some(base)) if base > 0.0 => {
            let delta_pct = regression_pct_f64(base, value);
            BudgetEvaluation {
                id: args.id.to_string(),
                title: args.title.to_string(),
                status: if delta_pct <= args.limit_pct {
                    BudgetStatus::Pass
                } else {
                    BudgetStatus::Fail
                },
                observed_value: Some(value),
                limit_value: args.limit_pct,
                unit: args.unit.to_string(),
                comparison: "<=".to_string(),
                baseline_value: Some(base),
                delta_pct: Some(delta_pct),
                detail: args.detail.to_string(),
            }
        }
        (Some(value), _) => BudgetEvaluation {
            id: args.id.to_string(),
            title: args.title.to_string(),
            status: BudgetStatus::BaselineRequired,
            observed_value: Some(value),
            limit_value: args.limit_pct,
            unit: args.unit.to_string(),
            comparison: "<=".to_string(),
            baseline_value: baseline,
            delta_pct: None,
            detail: format!("{}; provide --baseline to evaluate regression", args.detail),
        },
        _ => BudgetEvaluation {
            id: args.id.to_string(),
            title: args.title.to_string(),
            status: BudgetStatus::Fail,
            observed_value: None,
            limit_value: args.limit_pct,
            unit: args.unit.to_string(),
            comparison: "<=".to_string(),
            baseline_value: baseline,
            delta_pct: None,
            detail: "measurement did not produce an observed value".to_string(),
        },
    }
}

fn overall_status(budgets: &[BudgetEvaluation]) -> ReportStatus {
    if budgets
        .iter()
        .any(|budget| budget.status == BudgetStatus::Fail)
    {
        ReportStatus::Fail
    } else if budgets.iter().any(|budget| {
        matches!(
            budget.status,
            BudgetStatus::BaselineRequired | BudgetStatus::Informational
        )
    }) {
        ReportStatus::Partial
    } else {
        ReportStatus::Pass
    }
}

fn linear(
    session: &mut FrankenTorchSession,
    input: TensorNodeId,
    weight: TensorNodeId,
    bias: TensorNodeId,
) -> Result<TensorNodeId, String> {
    let weight_t = session
        .tensor_transpose(weight, 0, 1)
        .map_err(|error| format!("tensor_transpose failed: {error}"))?;
    let out = session
        .tensor_matmul(input, weight_t)
        .map_err(|error| format!("tensor_matmul failed: {error}"))?;
    let out_shape = session
        .tensor_shape(out)
        .map_err(|error| format!("tensor_shape failed: {error}"))?;
    let expanded_bias = session
        .tensor_expand(bias, out_shape)
        .map_err(|error| format!("tensor_expand failed: {error}"))?;
    session
        .tensor_add(out, expanded_bias)
        .map_err(|error| format!("tensor_add failed: {error}"))
}

fn linear_relu(
    session: &mut FrankenTorchSession,
    input: TensorNodeId,
    weight: TensorNodeId,
    bias: TensorNodeId,
) -> Result<TensorNodeId, String> {
    let out = linear(session, input, weight, bias)?;
    session
        .tensor_relu(out)
        .map_err(|error| format!("tensor_relu failed: {error}"))
}

fn forward_graph(
    session: &mut FrankenTorchSession,
    input: TensorNodeId,
    weight1: TensorNodeId,
    bias1: TensorNodeId,
    weight2: TensorNodeId,
    bias2: TensorNodeId,
) -> Result<TensorNodeId, String> {
    let hidden = linear_relu(session, input, weight1, bias1)?;
    linear(session, hidden, weight2, bias2)
}

fn build_state_dict(
    total_payload_bytes: u64,
    chunk_count: usize,
) -> Result<BTreeMap<String, DenseTensor>, String> {
    if chunk_count == 0 {
        return Err("checkpoint_chunks must be > 0".to_string());
    }
    let bytes_per_chunk =
        total_payload_bytes
            .checked_div(u64::try_from(chunk_count).map_err(|error| {
                format!("failed to convert checkpoint chunk count to u64: {error}")
            })?)
            .ok_or_else(|| "checkpoint byte division by zero".to_string())?;
    let elements_per_chunk = usize::try_from(
        bytes_per_chunk
            .checked_div(4)
            .ok_or_else(|| "checkpoint chunk bytes must be divisible by f32 width".to_string())?,
    )
    .map_err(|error| format!("checkpoint chunk element conversion failed: {error}"))?;

    let mut state_dict = BTreeMap::new();
    for chunk_idx in 0..chunk_count {
        let tensor = DenseTensor::from_contiguous_values_f32(
            vec![0.5_f32; elements_per_chunk],
            vec![elements_per_chunk],
            Device::Cpu,
        )
        .map_err(|error| format!("failed to build checkpoint tensor chunk {chunk_idx}: {error}"))?;
        state_dict.insert(format!("chunk_{chunk_idx:02}"), tensor);
    }
    Ok(state_dict)
}

fn current_rss_kb() -> Result<u64, String> {
    let status = fs::read_to_string("/proc/self/status")
        .map_err(|error| format!("failed to read /proc/self/status: {error}"))?;
    parse_kib_field(status.as_str(), "VmRSS:")
        .ok_or_else(|| "VmRSS not present in /proc/self/status".to_string())
}

fn mem_available_kb() -> Result<u64, String> {
    let meminfo = fs::read_to_string("/proc/meminfo")
        .map_err(|error| format!("failed to read /proc/meminfo: {error}"))?;
    parse_kib_field(meminfo.as_str(), "MemAvailable:")
        .ok_or_else(|| "MemAvailable not present in /proc/meminfo".to_string())
}

fn parse_kib_field(contents: &str, prefix: &str) -> Option<u64> {
    contents.lines().find_map(|line| {
        let stripped = line.strip_prefix(prefix)?.trim();
        stripped
            .split_whitespace()
            .next()
            .and_then(|raw| raw.parse::<u64>().ok())
    })
}

fn update_memory_metrics(metrics: &mut MemoryMetrics, current_rss_kb: u64) {
    metrics.sampled_peak_rss_kb = metrics.sampled_peak_rss_kb.max(current_rss_kb);
    metrics.sampled_final_rss_kb = current_rss_kb;
}

fn checked_square_numel(dim: usize) -> Result<usize, String> {
    dim.checked_mul(dim)
        .ok_or_else(|| format!("square numel overflow for dim={dim}"))
}

fn checked_numel(shape: &[usize]) -> Result<usize, String> {
    shape.iter().try_fold(1usize, |acc, &dim| {
        acc.checked_mul(dim)
            .ok_or_else(|| format!("shape volume overflow for shape={shape:?}"))
    })
}

fn latency_summary(mut samples: Vec<u128>) -> LatencySummary {
    samples.sort_unstable();
    let mean_ns = if samples.is_empty() {
        0
    } else {
        samples.iter().copied().sum::<u128>() / (samples.len() as u128)
    };
    LatencySummary {
        p50_ns: percentile_u128(samples.as_slice(), 50),
        p95_ns: percentile_u128(samples.as_slice(), 95),
        p99_ns: percentile_u128(samples.as_slice(), 99),
        mean_ns,
        samples_ns: samples,
    }
}

fn throughput_summary(mut samples: Vec<f64>) -> ThroughputSummary {
    samples.sort_by(|left, right| left.total_cmp(right));
    let mean_mb_per_s = if samples.is_empty() {
        0.0
    } else {
        samples.iter().sum::<f64>() / (samples.len() as f64)
    };
    ThroughputSummary {
        p50_mb_per_s: percentile_f64(samples.as_slice(), 50),
        p95_mb_per_s: percentile_f64(samples.as_slice(), 95),
        p99_mb_per_s: percentile_f64(samples.as_slice(), 99),
        mean_mb_per_s,
        samples_mb_per_s: samples,
    }
}

fn percentile_u128(samples: &[u128], pct: usize) -> u128 {
    if samples.is_empty() {
        return 0;
    }
    let clamped = pct.min(100);
    let idx = ((samples.len() - 1) * clamped) / 100;
    samples[idx]
}

fn percentile_f64(samples: &[f64], pct: usize) -> f64 {
    if samples.is_empty() {
        return 0.0;
    }
    let clamped = pct.min(100);
    let idx = ((samples.len() - 1) * clamped) / 100;
    samples[idx]
}

#[allow(clippy::cast_precision_loss)]
fn ns_to_ms(value: u128) -> f64 {
    (value as f64) / 1_000_000.0
}

#[allow(clippy::cast_precision_loss)]
fn bytes_to_ms(bytes: u64, bytes_per_second: f64) -> f64 {
    ((bytes as f64) / bytes_per_second) * 1_000.0
}

#[allow(clippy::cast_precision_loss)]
fn bytes_per_second_mb(bytes: u64, elapsed_ns: u128) -> f64 {
    if elapsed_ns == 0 {
        return f64::INFINITY;
    }
    let elapsed_s = (elapsed_ns as f64) / 1_000_000_000.0;
    (bytes as f64) / elapsed_s / 1_000_000.0
}

#[allow(clippy::cast_precision_loss)]
fn ratio(lhs: u128, rhs: u128) -> f64 {
    if lhs == 0 {
        return f64::INFINITY;
    }
    (rhs as f64) / (lhs as f64)
}

#[allow(clippy::cast_precision_loss)]
fn regression_pct(baseline: u128, observed: u128) -> f64 {
    if baseline == 0 {
        if observed == 0 { 0.0 } else { f64::INFINITY }
    } else {
        ((observed as f64) - (baseline as f64)) * 100.0 / (baseline as f64)
    }
}

fn regression_pct_f64(baseline: f64, observed: f64) -> f64 {
    if baseline == 0.0 {
        if observed == 0.0 { 0.0 } else { f64::INFINITY }
    } else {
        ((observed - baseline) * 100.0) / baseline
    }
}

fn now_unix_ms() -> u128 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map_or(0, |duration| duration.as_millis())
}

fn profile_str(profile: RunProfile) -> &'static str {
    match profile {
        RunProfile::Full => "full",
        RunProfile::Smoke => "smoke",
    }
}

fn report_status_str(status: ReportStatus) -> &'static str {
    match status {
        ReportStatus::Pass => "pass",
        ReportStatus::Partial => "partial",
        ReportStatus::Fail => "fail",
    }
}

fn budget_status_str(status: BudgetStatus) -> &'static str {
    match status {
        BudgetStatus::Pass => "pass",
        BudgetStatus::Fail => "fail",
        BudgetStatus::BaselineRequired => "baseline_required",
        BudgetStatus::Informational => "informational",
    }
}

struct SharedGraphTensors {
    input: DenseTensor,
    target: DenseTensor,
    weight1: DenseTensor,
    bias1: DenseTensor,
    weight2: DenseTensor,
    bias2: DenseTensor,
}

impl SharedGraphTensors {
    fn representative(config: &WorkloadConfig) -> Result<Self, String> {
        Self::new(
            config.representative_batch,
            config.representative_in_features,
            config.representative_hidden_features,
            config.representative_out_features,
        )
    }

    fn training(config: &WorkloadConfig) -> Result<Self, String> {
        Self::new(
            config.training_batch,
            config.training_in_features,
            config.training_hidden_features,
            config.training_out_features,
        )
    }

    fn new(
        batch: usize,
        in_features: usize,
        hidden_features: usize,
        out_features: usize,
    ) -> Result<Self, String> {
        Ok(Self {
            input: DenseTensor::from_contiguous_values(
                vec![0.25_f64; checked_numel(&[batch, in_features])?],
                vec![batch, in_features],
                Device::Cpu,
            )
            .map_err(|error| format!("failed to build shared input tensor: {error}"))?,
            target: DenseTensor::from_contiguous_values(
                vec![0.5_f64; checked_numel(&[batch, out_features])?],
                vec![batch, out_features],
                Device::Cpu,
            )
            .map_err(|error| format!("failed to build shared target tensor: {error}"))?,
            weight1: DenseTensor::from_contiguous_values(
                vec![0.02_f64; checked_numel(&[hidden_features, in_features])?],
                vec![hidden_features, in_features],
                Device::Cpu,
            )
            .map_err(|error| format!("failed to build shared weight1 tensor: {error}"))?,
            bias1: DenseTensor::from_contiguous_values(
                vec![0.01_f64; hidden_features],
                vec![1, hidden_features],
                Device::Cpu,
            )
            .map_err(|error| format!("failed to build shared bias1 tensor: {error}"))?,
            weight2: DenseTensor::from_contiguous_values(
                vec![0.03_f64; checked_numel(&[out_features, hidden_features])?],
                vec![out_features, hidden_features],
                Device::Cpu,
            )
            .map_err(|error| format!("failed to build shared weight2 tensor: {error}"))?,
            bias2: DenseTensor::from_contiguous_values(
                vec![0.02_f64; out_features],
                vec![1, out_features],
                Device::Cpu,
            )
            .map_err(|error| format!("failed to build shared bias2 tensor: {error}"))?,
        })
    }
}

struct SharedAllocLayer {
    weight: DenseTensor,
    bias: DenseTensor,
}

struct SharedAllocationGraph {
    batch: usize,
    width: usize,
    depth: usize,
    out_features: usize,
    input: DenseTensor,
    target: DenseTensor,
    layers: Vec<SharedAllocLayer>,
}

impl SharedAllocationGraph {
    fn new(config: &WorkloadConfig) -> Result<Self, String> {
        let mut layers = Vec::with_capacity(config.alloc_depth);
        let mut in_features = config.alloc_width;
        for layer_idx in 0..config.alloc_depth {
            let out_features = if layer_idx + 1 == config.alloc_depth {
                config.alloc_out_features
            } else {
                config.alloc_width
            };
            layers.push(SharedAllocLayer {
                weight: DenseTensor::from_contiguous_values(
                    vec![0.01_f64; checked_numel(&[out_features, in_features])?],
                    vec![out_features, in_features],
                    Device::Cpu,
                )
                .map_err(|error| {
                    format!("failed to build alloc weight layer {layer_idx}: {error}")
                })?,
                bias: DenseTensor::from_contiguous_values(
                    vec![0.02_f64; out_features],
                    vec![1, out_features],
                    Device::Cpu,
                )
                .map_err(|error| {
                    format!("failed to build alloc bias layer {layer_idx}: {error}")
                })?,
            });
            in_features = out_features;
        }

        Ok(Self {
            batch: config.alloc_batch,
            width: config.alloc_width,
            depth: config.alloc_depth,
            out_features: config.alloc_out_features,
            input: DenseTensor::from_contiguous_values(
                vec![0.5_f64; checked_numel(&[config.alloc_batch, config.alloc_width])?],
                vec![config.alloc_batch, config.alloc_width],
                Device::Cpu,
            )
            .map_err(|error| format!("failed to build alloc input tensor: {error}"))?,
            target: DenseTensor::from_contiguous_values(
                vec![0.25_f64; checked_numel(&[config.alloc_batch, config.alloc_out_features])?],
                vec![config.alloc_batch, config.alloc_out_features],
                Device::Cpu,
            )
            .map_err(|error| format!("failed to build alloc target tensor: {error}"))?,
            layers,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::{
        bytes_per_second_mb, bytes_to_ms, parse_kib_field, percentile_f64, percentile_u128,
        regression_pct, regression_pct_f64,
    };

    #[test]
    fn percentile_u128_matches_existing_harness_rule() {
        let samples = [10_u128, 20, 30, 40, 50];
        assert_eq!(percentile_u128(&samples, 50), 30);
        assert_eq!(percentile_u128(&samples, 95), 40);
    }

    #[test]
    fn percentile_f64_uses_same_index_rule() {
        let samples = [1.0_f64, 2.0, 3.0, 4.0, 5.0];
        assert!((percentile_f64(&samples, 50) - 3.0).abs() < f64::EPSILON);
        assert!((percentile_f64(&samples, 95) - 4.0).abs() < f64::EPSILON);
    }

    #[test]
    fn parse_kib_field_extracts_value() {
        let parsed = parse_kib_field("VmRSS:\t  12345 kB\nVmSize:\t  99999 kB\n", "VmRSS:");
        assert_eq!(parsed, Some(12_345));
    }

    #[test]
    fn checkpoint_budget_math_is_stable() {
        let max_ms = bytes_to_ms(350_000_000, 350_000_000.0);
        assert!((max_ms - 1_000.0).abs() < 1e-6);
        let throughput = bytes_per_second_mb(350_000_000, 1_000_000_000);
        assert!((throughput - 350.0).abs() < 1e-6);
    }

    #[test]
    fn regression_helpers_report_zero_for_equal_values() {
        assert!(regression_pct(100, 100).abs() < 1e-9);
        assert!(regression_pct_f64(100.0, 100.0).abs() < 1e-9);
    }
}

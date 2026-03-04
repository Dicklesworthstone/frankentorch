#![forbid(unsafe_code)]

use std::fs;
use std::path::{Path, PathBuf};

use ft_conformance::{HarnessConfig, run_packet_e2e_microbench, run_scalar_microbench};
use ft_core::ExecutionMode;
use serde::{Deserialize, Serialize};

const REPORT_SCHEMA_VERSION: &str = "ft-benchmark-regression-gate-v1";
const DEFAULT_THRESHOLD_PCT: f64 = 5.0;
const DEFAULT_OUTPUT_PATH: &str = "artifacts/phase2c/benchmark_regression_gate_report_v1.json";
const SCALAR_BASELINE_PATH: &str =
    "artifacts/optimization/2026-02-14_foundation_perf_rebaseline.json";
const PACKET_BASELINE_PATHS: &[&str] = &[
    "artifacts/phase2c/FT-P2C-003/optimization_delta_v1.json",
    "artifacts/phase2c/FT-P2C-004/optimization_delta_v1.json",
    "artifacts/phase2c/FT-P2C-005/optimization_delta_v1.json",
    "artifacts/phase2c/FT-P2C-006/optimization_delta_v1.json",
    "artifacts/phase2c/FT-P2C-007/optimization_delta_v1.json",
    "artifacts/phase2c/FT-P2C-008/optimization_delta_v1.json",
];

#[derive(Debug, Clone)]
struct CliArgs {
    threshold_pct: f64,
    output_path: PathBuf,
    skip_scalar: bool,
}

#[derive(Debug, Clone, Serialize)]
struct BenchmarkRegressionReport {
    schema_version: &'static str,
    generated_unix_ms: u128,
    threshold_pct: f64,
    status: &'static str,
    scalar: Option<BenchComparison>,
    packets: Vec<BenchComparison>,
    violations: Vec<RegressionViolation>,
}

#[derive(Debug, Clone, Serialize)]
struct BenchComparison {
    benchmark_id: String,
    baseline_source: String,
    iterations: usize,
    baseline: BenchMetrics,
    observed: BenchMetrics,
    deltas_pct: BenchMetricDeltas,
    regressed_metrics: Vec<String>,
}

#[derive(Debug, Clone, Serialize)]
struct BenchMetricDeltas {
    p50: f64,
    p95: f64,
    p99: f64,
    mean: f64,
}

#[derive(Debug, Clone, Serialize)]
struct BenchMetrics {
    p50_ns: u128,
    p95_ns: u128,
    p99_ns: u128,
    mean_ns: u128,
}

#[derive(Debug, Clone, Serialize)]
struct RegressionViolation {
    benchmark_id: String,
    metric: String,
    baseline_ns: u128,
    observed_ns: u128,
    regression_pct: f64,
    threshold_pct: f64,
    baseline_source: String,
}

#[derive(Debug, Clone, Deserialize)]
struct ScalarBaselineRoot {
    benchmark: ScalarBaselineBenchmark,
}

#[derive(Debug, Clone, Deserialize)]
struct ScalarBaselineBenchmark {
    iterations: usize,
    step_time_tails_ns: ScalarBaselineTails,
}

#[derive(Debug, Clone, Deserialize)]
struct ScalarBaselineTails {
    p50: u128,
    p95: u128,
    p99: u128,
    mean: u128,
}

#[derive(Debug, Clone, Deserialize)]
struct PacketBaselineRoot {
    packet_id: String,
    post: PacketBaselinePost,
}

#[derive(Debug, Clone, Deserialize)]
struct PacketBaselinePost {
    iterations: usize,
    p50_ns: u128,
    p95_ns: u128,
    p99_ns: u128,
    mean_ns: u128,
}

#[derive(Debug, Clone)]
struct BaselineSpec {
    benchmark_id: String,
    source: PathBuf,
    iterations: usize,
    metrics: BenchMetrics,
}

fn main() -> Result<(), String> {
    let args = parse_args()?;
    let repo_root = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..");
    let scalar_baseline_path = repo_root.join(SCALAR_BASELINE_PATH);
    let packet_baseline_paths: Vec<PathBuf> = PACKET_BASELINE_PATHS
        .iter()
        .map(|path| repo_root.join(path))
        .collect();

    let mut comparisons = Vec::new();
    let mut violations = Vec::new();

    let scalar = if args.skip_scalar {
        None
    } else {
        let spec = load_scalar_baseline(scalar_baseline_path.as_path())?;
        let observed = run_scalar_microbench(spec.iterations, ExecutionMode::Strict);
        let observed_metrics = BenchMetrics {
            p50_ns: observed.p50_ns,
            p95_ns: observed.p95_ns,
            p99_ns: observed.p99_ns,
            mean_ns: observed.mean_ns,
        };
        let comparison = compare_metrics(
            spec.benchmark_id.as_str(),
            spec.source.as_path(),
            spec.iterations,
            &spec.metrics,
            &observed_metrics,
            args.threshold_pct,
        );
        violations.extend(comparison_violations(
            &comparison,
            args.threshold_pct,
            spec.source.as_path(),
        ));
        Some(comparison)
    };

    let harness = HarnessConfig::default_paths();
    for path in packet_baseline_paths {
        let spec = load_packet_baseline(path.as_path())?;
        let observed =
            run_packet_e2e_microbench(&harness, spec.iterations, spec.benchmark_id.as_str())
                .map_err(|error| {
                    format!("benchmark '{}' failed to run: {error}", spec.benchmark_id)
                })?;
        let observed_metrics = BenchMetrics {
            p50_ns: observed.p50_ns,
            p95_ns: observed.p95_ns,
            p99_ns: observed.p99_ns,
            mean_ns: observed.mean_ns,
        };
        let comparison = compare_metrics(
            spec.benchmark_id.as_str(),
            spec.source.as_path(),
            spec.iterations,
            &spec.metrics,
            &observed_metrics,
            args.threshold_pct,
        );
        violations.extend(comparison_violations(
            &comparison,
            args.threshold_pct,
            spec.source.as_path(),
        ));
        comparisons.push(comparison);
    }

    let status = if violations.is_empty() {
        "pass"
    } else {
        "fail"
    };
    let report = BenchmarkRegressionReport {
        schema_version: REPORT_SCHEMA_VERSION,
        generated_unix_ms: now_unix_ms(),
        threshold_pct: args.threshold_pct,
        status,
        scalar,
        packets: comparisons,
        violations,
    };

    if let Some(parent) = args.output_path.parent() {
        fs::create_dir_all(parent).map_err(|error| {
            format!(
                "failed to create benchmark gate output dir {}: {error}",
                parent.display()
            )
        })?;
    }

    fs::write(
        args.output_path.as_path(),
        serde_json::to_string_pretty(&report)
            .map_err(|error| format!("failed to serialize benchmark report: {error}"))?,
    )
    .map_err(|error| {
        format!(
            "failed to write benchmark gate report {}: {error}",
            args.output_path.display()
        )
    })?;

    println!(
        "{}",
        serde_json::to_string_pretty(&report)
            .map_err(|error| format!("failed to serialize benchmark report: {error}"))?
    );

    if report.status == "fail" {
        std::process::exit(2);
    }

    Ok(())
}

fn parse_args() -> Result<CliArgs, String> {
    let mut threshold_pct = DEFAULT_THRESHOLD_PCT;
    let mut output_path = PathBuf::from(DEFAULT_OUTPUT_PATH);
    let mut skip_scalar = false;

    let mut args = std::env::args().skip(1);
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--threshold-pct" => {
                let raw = args
                    .next()
                    .ok_or_else(|| "--threshold-pct requires a numeric value".to_string())?;
                threshold_pct = raw
                    .parse::<f64>()
                    .map_err(|error| format!("invalid --threshold-pct value '{raw}': {error}"))?;
            }
            "--output" => {
                let raw = args
                    .next()
                    .ok_or_else(|| "--output requires a file path".to_string())?;
                output_path = PathBuf::from(raw);
            }
            "--skip-scalar" => {
                skip_scalar = true;
            }
            other => {
                return Err(format!(
                    "unknown arg '{other}'. usage: check_benchmark_regression [--threshold-pct <f64>] [--output <path>] [--skip-scalar]"
                ));
            }
        }
    }

    if !threshold_pct.is_finite() || threshold_pct < 0.0 {
        return Err(format!(
            "invalid --threshold-pct value {threshold_pct}; expected finite non-negative value"
        ));
    }

    Ok(CliArgs {
        threshold_pct,
        output_path,
        skip_scalar,
    })
}

fn load_scalar_baseline(path: &Path) -> Result<BaselineSpec, String> {
    let raw = fs::read_to_string(path)
        .map_err(|error| format!("failed to read scalar baseline {}: {error}", path.display()))?;
    let parsed: ScalarBaselineRoot = serde_json::from_str(raw.as_str()).map_err(|error| {
        format!(
            "failed to parse scalar baseline {}: {error}",
            path.display()
        )
    })?;
    Ok(BaselineSpec {
        benchmark_id: "scalar_strict".to_string(),
        source: path.to_path_buf(),
        iterations: parsed.benchmark.iterations,
        metrics: BenchMetrics {
            p50_ns: parsed.benchmark.step_time_tails_ns.p50,
            p95_ns: parsed.benchmark.step_time_tails_ns.p95,
            p99_ns: parsed.benchmark.step_time_tails_ns.p99,
            mean_ns: parsed.benchmark.step_time_tails_ns.mean,
        },
    })
}

fn load_packet_baseline(path: &Path) -> Result<BaselineSpec, String> {
    let raw = fs::read_to_string(path)
        .map_err(|error| format!("failed to read packet baseline {}: {error}", path.display()))?;
    let parsed: PacketBaselineRoot = serde_json::from_str(raw.as_str()).map_err(|error| {
        format!(
            "failed to parse packet baseline {}: {error}",
            path.display()
        )
    })?;
    Ok(BaselineSpec {
        benchmark_id: parsed.packet_id,
        source: path.to_path_buf(),
        iterations: parsed.post.iterations,
        metrics: BenchMetrics {
            p50_ns: parsed.post.p50_ns,
            p95_ns: parsed.post.p95_ns,
            p99_ns: parsed.post.p99_ns,
            mean_ns: parsed.post.mean_ns,
        },
    })
}

fn compare_metrics(
    benchmark_id: &str,
    baseline_source: &Path,
    iterations: usize,
    baseline: &BenchMetrics,
    observed: &BenchMetrics,
    threshold_pct: f64,
) -> BenchComparison {
    let deltas = BenchMetricDeltas {
        p50: regression_pct(baseline.p50_ns, observed.p50_ns),
        p95: regression_pct(baseline.p95_ns, observed.p95_ns),
        p99: regression_pct(baseline.p99_ns, observed.p99_ns),
        mean: regression_pct(baseline.mean_ns, observed.mean_ns),
    };

    let regressed_metrics = [
        ("p50_ns", deltas.p50),
        ("p95_ns", deltas.p95),
        ("p99_ns", deltas.p99),
        ("mean_ns", deltas.mean),
    ]
    .iter()
    .filter_map(|(metric, delta)| {
        if *delta > threshold_pct {
            Some((*metric).to_string())
        } else {
            None
        }
    })
    .collect::<Vec<_>>();

    BenchComparison {
        benchmark_id: benchmark_id.to_string(),
        baseline_source: baseline_source.display().to_string(),
        iterations,
        baseline: baseline.clone(),
        observed: observed.clone(),
        deltas_pct: deltas,
        regressed_metrics,
    }
}

fn comparison_violations(
    comparison: &BenchComparison,
    threshold_pct: f64,
    source: &Path,
) -> Vec<RegressionViolation> {
    [
        (
            "p50_ns",
            comparison.baseline.p50_ns,
            comparison.observed.p50_ns,
            comparison.deltas_pct.p50,
        ),
        (
            "p95_ns",
            comparison.baseline.p95_ns,
            comparison.observed.p95_ns,
            comparison.deltas_pct.p95,
        ),
        (
            "p99_ns",
            comparison.baseline.p99_ns,
            comparison.observed.p99_ns,
            comparison.deltas_pct.p99,
        ),
        (
            "mean_ns",
            comparison.baseline.mean_ns,
            comparison.observed.mean_ns,
            comparison.deltas_pct.mean,
        ),
    ]
    .iter()
    .filter_map(|(metric, baseline_ns, observed_ns, delta)| {
        if *delta > threshold_pct {
            Some(RegressionViolation {
                benchmark_id: comparison.benchmark_id.clone(),
                metric: (*metric).to_string(),
                baseline_ns: *baseline_ns,
                observed_ns: *observed_ns,
                regression_pct: *delta,
                threshold_pct,
                baseline_source: source.display().to_string(),
            })
        } else {
            None
        }
    })
    .collect()
}

fn regression_pct(baseline: u128, observed: u128) -> f64 {
    if baseline == 0 {
        if observed == 0 { 0.0 } else { f64::INFINITY }
    } else {
        ((observed as f64) - (baseline as f64)) * 100.0 / (baseline as f64)
    }
}

fn now_unix_ms() -> u128 {
    let now = std::time::SystemTime::now();
    now.duration_since(std::time::UNIX_EPOCH)
        .map_or(0, |duration| duration.as_millis())
}

#[cfg(test)]
mod tests {
    use super::{BenchMetrics, compare_metrics, regression_pct};
    use std::path::Path;

    #[test]
    fn regression_pct_reports_positive_for_slowdown() {
        let delta = regression_pct(100, 108);
        assert!(delta > 7.99 && delta < 8.01);
    }

    #[test]
    fn compare_metrics_treats_exact_threshold_as_non_regression() {
        let baseline = BenchMetrics {
            p50_ns: 1000,
            p95_ns: 1000,
            p99_ns: 1000,
            mean_ns: 1000,
        };
        let observed = BenchMetrics {
            p50_ns: 1050,
            p95_ns: 1000,
            p99_ns: 950,
            mean_ns: 1049,
        };
        let comparison = compare_metrics(
            "FT-P2C-XYZ",
            Path::new("baseline.json"),
            10,
            &baseline,
            &observed,
            5.0,
        );
        assert!(
            !comparison
                .regressed_metrics
                .iter()
                .any(|metric| metric == "p50_ns"),
            "p50 at exactly 5% should not be marked regressed"
        );
    }

    #[test]
    fn compare_metrics_marks_values_above_threshold() {
        let baseline = BenchMetrics {
            p50_ns: 1000,
            p95_ns: 1000,
            p99_ns: 1000,
            mean_ns: 1000,
        };
        let observed = BenchMetrics {
            p50_ns: 1060,
            p95_ns: 1040,
            p99_ns: 1000,
            mean_ns: 1000,
        };
        let comparison = compare_metrics(
            "FT-P2C-XYZ",
            Path::new("baseline.json"),
            10,
            &baseline,
            &observed,
            5.0,
        );
        assert!(
            comparison
                .regressed_metrics
                .iter()
                .any(|metric| metric == "p50_ns")
        );
        assert!(
            !comparison
                .regressed_metrics
                .iter()
                .any(|metric| metric == "p95_ns")
        );
    }
}

#![forbid(unsafe_code)]

use serde::{Deserialize, Serialize};

pub const REPORT_SCHEMA_VERSION: &str = "ft-perf-slo-report-v1";
pub const GATE_SCHEMA_VERSION: &str = "ft-perf-slo-gate-v1";
pub const DEFAULT_REPORT_PATH: &str = "artifacts/phase2c/performance/perf_slo_measurement_v1.json";
pub const DEFAULT_SUMMARY_PATH: &str = "artifacts/phase2c/performance/PERF_SLO_MEASUREMENT_V1.md";
pub const DEFAULT_BASELINE_PATH: &str = "artifacts/phase2c/performance/perf_slo_baseline_v1.json";
pub const DEFAULT_BASELINE_SUMMARY_PATH: &str =
    "artifacts/phase2c/performance/PERF_SLO_BASELINE_V1.md";
pub const DEFAULT_GATE_OUTPUT_PATH: &str =
    "artifacts/phase2c/performance/perf_slo_gate_report_v1.json";

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PerfProfile {
    Spec,
    Smoke,
}

impl PerfProfile {
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Spec => "spec",
            Self::Smoke => "smoke",
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WorkloadMetadata {
    pub status: String,
    pub sample_count_requested: usize,
    pub sample_count_completed: usize,
    pub target_description: String,
    pub notes: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DistributionSummary {
    pub sample_count: usize,
    pub p50: f64,
    pub p95: f64,
    pub p99: f64,
    pub mean: f64,
    pub min: f64,
    pub max: f64,
}

impl DistributionSummary {
    #[must_use]
    pub fn from_samples(samples: &[f64]) -> Self {
        if samples.is_empty() {
            return Self {
                sample_count: 0,
                p50: 0.0,
                p95: 0.0,
                p99: 0.0,
                mean: 0.0,
                min: 0.0,
                max: 0.0,
            };
        }

        let mut sorted = samples.to_vec();
        sorted.sort_by(|left, right| left.total_cmp(right));
        let mean = sorted.iter().sum::<f64>() / (sorted.len() as f64);

        Self {
            sample_count: sorted.len(),
            p50: percentile(&sorted, 50),
            p95: percentile(&sorted, 95),
            p99: percentile(&sorted, 99),
            mean,
            min: *sorted.first().unwrap_or(&0.0),
            max: *sorted.last().unwrap_or(&0.0),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LatencyMeasurement {
    pub workload: WorkloadMetadata,
    pub duration_ms: DistributionSummary,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BackwardPassMeasurement {
    pub workload: WorkloadMetadata,
    pub forward_ms: DistributionSummary,
    pub backward_ms: DistributionSummary,
    pub p95_backward_to_forward: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CheckpointSaveLoadMeasurement {
    pub workload: WorkloadMetadata,
    pub realized_state_bytes: u64,
    pub save_ms: DistributionSummary,
    pub load_ms: DistributionSummary,
    pub combined_ms: DistributionSummary,
    pub save_throughput_mib_s: DistributionSummary,
    pub load_throughput_mib_s: DistributionSummary,
    pub combined_throughput_mib_s: DistributionSummary,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryFootprintMeasurement {
    pub workload: WorkloadMetadata,
    pub max_rss_kb: DistributionSummary,
    pub peak_allocated_bytes: DistributionSummary,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AllocationChurnMeasurement {
    pub workload: WorkloadMetadata,
    pub allocation_count: DistributionSummary,
    pub allocated_bytes: DistributionSummary,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerfSloReport {
    pub schema_version: String,
    pub generated_unix_ms: u128,
    pub profile: PerfProfile,
    pub elementwise_forward: LatencyMeasurement,
    pub backward_pass: BackwardPassMeasurement,
    pub matmul_core: LatencyMeasurement,
    pub optimizer_step: LatencyMeasurement,
    pub checkpoint_save_load: CheckpointSaveLoadMeasurement,
    pub memory_footprint: MemoryFootprintMeasurement,
    pub allocation_churn: AllocationChurnMeasurement,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GateCheck {
    pub budget_id: String,
    pub status: String,
    pub observed: f64,
    pub threshold: f64,
    pub units: String,
    pub comparator: String,
    pub message: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerfSloGateReport {
    pub schema_version: String,
    pub generated_unix_ms: u128,
    pub current_report_path: String,
    pub baseline_report_path: String,
    pub status: String,
    pub checks: Vec<GateCheck>,
}

#[must_use]
pub fn regression_pct(baseline: f64, observed: f64) -> f64 {
    if baseline == 0.0 {
        0.0
    } else {
        ((observed - baseline) * 100.0) / baseline
    }
}

#[must_use]
pub fn now_unix_ms() -> u128 {
    let now = std::time::SystemTime::now();
    now.duration_since(std::time::UNIX_EPOCH)
        .map_or(0, |duration| duration.as_millis())
}

#[must_use]
pub fn percentile(samples: &[f64], p: usize) -> f64 {
    if samples.is_empty() {
        return 0.0;
    }

    let clamped = p.min(100);
    let idx = ((samples.len() - 1) * clamped) / 100;
    samples[idx]
}

#[cfg(test)]
mod tests {
    use super::{DistributionSummary, regression_pct};

    #[test]
    fn distribution_summary_orders_percentiles() {
        let summary = DistributionSummary::from_samples(&[9.0, 1.0, 5.0, 7.0, 3.0]);
        assert_eq!(summary.sample_count, 5);
        assert_eq!(summary.min, 1.0);
        assert_eq!(summary.max, 9.0);
        assert!(summary.p95 >= summary.p50);
        assert!(summary.p99 >= summary.p95);
    }

    #[test]
    fn regression_pct_handles_zero_baseline() {
        assert_eq!(regression_pct(0.0, 10.0), 0.0);
        assert_eq!(regression_pct(100.0, 110.0), 10.0);
    }
}

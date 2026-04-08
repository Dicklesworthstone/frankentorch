#![forbid(unsafe_code)]

use std::fs;
use std::path::{Path, PathBuf};

use ft_conformance::perf_slo::DEFAULT_REPORT_PATH;
use serde::Deserialize;

#[derive(Debug, Clone)]
struct CliArgs {
    input_path: PathBuf,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize)]
#[serde(rename_all = "snake_case")]
enum BudgetStatus {
    Pass,
    Fail,
    BaselineRequired,
    Informational,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize)]
#[serde(rename_all = "snake_case")]
enum ReportStatus {
    Pass,
    Partial,
    Fail,
}

#[derive(Debug, Clone, Deserialize)]
struct PerfGateReport {
    schema_version: String,
    overall_status: ReportStatus,
    budgets: Vec<BudgetEntry>,
}

#[derive(Debug, Clone, Deserialize)]
struct BudgetEntry {
    id: String,
    title: String,
    status: BudgetStatus,
    observed_value: Option<f64>,
    limit_value: f64,
    unit: String,
    detail: String,
}

fn main() -> Result<(), String> {
    let args = parse_args()?;
    let report = load_report(args.input_path.as_path())?;

    println!("schema_version={}", report.schema_version);
    println!(
        "overall_status={}",
        report_status_str(report.overall_status)
    );
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
        println!(
            "[{}] {} ({}) observed={} limit={} :: {}",
            budget_status_str(budget.status),
            budget.title,
            budget.id,
            observed,
            limit,
            budget.detail
        );
    }

    if report.overall_status == ReportStatus::Fail
        || report
            .budgets
            .iter()
            .any(|budget| budget.status == BudgetStatus::Fail)
    {
        std::process::exit(1);
    }

    Ok(())
}

fn parse_args() -> Result<CliArgs, String> {
    let mut input_path = PathBuf::from(DEFAULT_REPORT_PATH);
    let mut args = std::env::args().skip(1);
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--input" => {
                let raw = args
                    .next()
                    .ok_or_else(|| "--input requires a file path".to_string())?;
                input_path = PathBuf::from(raw);
            }
            other => {
                return Err(format!(
                    "unknown arg '{other}'. usage: check_perf_slos [--input <path>]"
                ));
            }
        }
    }
    Ok(CliArgs { input_path })
}

fn load_report(path: &Path) -> Result<PerfGateReport, String> {
    let raw = fs::read_to_string(path)
        .map_err(|error| format!("failed to read perf report {}: {error}", path.display()))?;
    serde_json::from_str(raw.as_str())
        .map_err(|error| format!("failed to parse perf report {}: {error}", path.display()))
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

#[cfg(test)]
mod tests {
    use super::{BudgetStatus, ReportStatus, budget_status_str, report_status_str};

    #[test]
    fn status_strings_are_stable() {
        assert_eq!(budget_status_str(BudgetStatus::Fail), "fail");
        assert_eq!(
            budget_status_str(BudgetStatus::BaselineRequired),
            "baseline_required"
        );
        assert_eq!(report_status_str(ReportStatus::Partial), "partial");
    }
}

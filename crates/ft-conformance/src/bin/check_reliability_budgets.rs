#![forbid(unsafe_code)]

use std::collections::{BTreeMap, BTreeSet};
use std::fs;
use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};

const RELIABILITY_REPORT_SCHEMA_VERSION: &str = "ft-reliability-gate-report-v1";

#[derive(Debug, Clone, Deserialize)]
struct ReliabilityPolicy {
    #[allow(dead_code)]
    schema_version: String,
    coverage_floors: BTreeMap<String, PacketCoverageFloor>,
    flake_budget: FlakeBudget,
    global_budgets: GlobalBudgets,
    remediation_hints: BTreeMap<String, String>,
    artifact_refs: BTreeMap<String, String>,
}

#[derive(Debug, Clone, Deserialize)]
struct PacketCoverageFloor {
    min_scenarios: usize,
    min_pass_ratio: f64,
    required_suites: Vec<String>,
}

#[derive(Debug, Clone, Deserialize)]
struct FlakeBudget {
    max_flake_suspects: usize,
    #[allow(dead_code)]
    detection_rule: String,
    #[allow(dead_code)]
    retry_limit: usize,
    #[allow(dead_code)]
    quarantine_label: String,
    #[allow(dead_code)]
    de_flake_owner_workflow: Vec<String>,
}

#[derive(Debug, Clone, Deserialize)]
struct GlobalBudgets {
    max_failed_entries: usize,
    max_unknown_reason_codes: usize,
}

#[derive(Debug, Clone, Deserialize)]
struct ForensicsLogEntry {
    #[allow(dead_code)]
    schema_version: String,
    #[allow(dead_code)]
    ts_unix_ms: u128,
    suite_id: String,
    scenario_id: String,
    #[allow(dead_code)]
    fixture_id: String,
    packet_id: String,
    #[allow(dead_code)]
    mode: String,
    #[allow(dead_code)]
    seed: u64,
    #[allow(dead_code)]
    env_fingerprint: String,
    #[allow(dead_code)]
    artifact_refs: Vec<String>,
    #[allow(dead_code)]
    replay_command: String,
    outcome: String,
    reason_code: String,
}

#[derive(Debug, Clone, Serialize)]
struct ReliabilityGateReport {
    schema_version: &'static str,
    generated_unix_ms: u128,
    policy_path: String,
    e2e_path: String,
    status: String,
    summary: GateSummary,
    violations: Vec<BudgetViolation>,
}

#[derive(Debug, Clone, Serialize)]
struct GateSummary {
    total_entries: usize,
    failed_entries: usize,
    flake_suspect_count: usize,
    unknown_reason_code_count: usize,
    packet_metrics: BTreeMap<String, PacketMetrics>,
}

#[derive(Debug, Clone, Serialize)]
struct PacketMetrics {
    total_entries: usize,
    passed_entries: usize,
    pass_ratio: f64,
    suites_present: Vec<String>,
    scenario_ids: Vec<String>,
}

#[derive(Debug, Clone, Serialize)]
struct BudgetViolation {
    budget_id: String,
    category: String,
    message: String,
    packet_id: Option<String>,
    scenario_ids: Vec<String>,
    artifact_refs: Vec<String>,
    remediation_hint: String,
}

fn main() -> Result<(), String> {
    let (policy_path, e2e_path, output_path) = parse_args()?;
    let policy = read_policy(policy_path.as_path())?;
    let entries = read_forensics_jsonl(e2e_path.as_path())?;

    let (summary, violations) = evaluate(policy.clone(), entries.as_slice());
    let status = if violations.is_empty() {
        "pass"
    } else {
        "fail"
    }
    .to_string();

    let report = ReliabilityGateReport {
        schema_version: RELIABILITY_REPORT_SCHEMA_VERSION,
        generated_unix_ms: now_unix_ms(),
        policy_path: policy_path.display().to_string(),
        e2e_path: e2e_path.display().to_string(),
        status,
        summary,
        violations,
    };

    if let Some(parent) = output_path.parent() {
        fs::create_dir_all(parent).map_err(|error| {
            format!(
                "failed to create reliability output dir {}: {error}",
                parent.display()
            )
        })?;
    }

    fs::write(
        output_path.as_path(),
        serde_json::to_string_pretty(&report)
            .map_err(|error| format!("failed to serialize reliability report: {error}"))?,
    )
    .map_err(|error| format!("failed to write reliability report: {error}"))?;

    println!(
        "{}",
        serde_json::to_string_pretty(&report)
            .map_err(|error| format!("failed to serialize reliability report: {error}"))?
    );

    if !report.violations.is_empty() {
        std::process::exit(2);
    }

    Ok(())
}

fn evaluate(
    policy: ReliabilityPolicy,
    entries: &[ForensicsLogEntry],
) -> (GateSummary, Vec<BudgetViolation>) {
    let mut packet_stats: BTreeMap<String, PacketAccumulator> = BTreeMap::new();
    let mut failed_entries = 0usize;

    for entry in entries {
        let packet = packet_stats.entry(entry.packet_id.clone()).or_default();
        packet.total_entries += 1;
        packet.suites_present.insert(entry.suite_id.clone());
        packet.scenario_ids.insert(entry.scenario_id.clone());
        if entry.outcome == "pass" {
            packet.passed_entries += 1;
        } else {
            failed_entries += 1;
        }
    }

    let flake_suspects = detect_flake_suspects(entries);
    let flake_suspect_count = flake_suspects.len();
    let unknown_reason_codes = collect_unknown_reason_codes(entries);

    let packet_metrics = packet_stats
        .iter()
        .map(|(packet_id, acc)| {
            let pass_ratio = if acc.total_entries == 0 {
                0.0
            } else {
                acc.passed_entries as f64 / acc.total_entries as f64
            };
            (
                packet_id.clone(),
                PacketMetrics {
                    total_entries: acc.total_entries,
                    passed_entries: acc.passed_entries,
                    pass_ratio,
                    suites_present: acc.suites_present.iter().cloned().collect(),
                    scenario_ids: acc.scenario_ids.iter().cloned().collect(),
                },
            )
        })
        .collect::<BTreeMap<_, _>>();

    let mut violations = Vec::new();

    for (packet_id, floor) in &policy.coverage_floors {
        let metrics = packet_metrics.get(packet_id);

        if metrics.is_none() {
            violations.push(BudgetViolation {
                budget_id: format!("coverage_floor:{packet_id}:missing_packet"),
                category: "coverage_floor".to_string(),
                message: format!("missing packet {packet_id} in e2e forensic logs"),
                packet_id: Some(packet_id.clone()),
                scenario_ids: Vec::new(),
                artifact_refs: policy_artifact_refs(&policy),
                remediation_hint: remediation_hint(&policy, "coverage_floor"),
            });
            continue;
        }

        let metrics = metrics.expect("checked is_some");

        if metrics.total_entries < floor.min_scenarios {
            violations.push(BudgetViolation {
                budget_id: format!("coverage_floor:{packet_id}:scenario_count"),
                category: "coverage_floor".to_string(),
                message: format!(
                    "packet {packet_id} has {} scenarios, below floor {}",
                    metrics.total_entries, floor.min_scenarios
                ),
                packet_id: Some(packet_id.clone()),
                scenario_ids: metrics.scenario_ids.clone(),
                artifact_refs: policy_artifact_refs(&policy),
                remediation_hint: remediation_hint(&policy, "coverage_floor"),
            });
        }

        if metrics.pass_ratio < floor.min_pass_ratio {
            violations.push(BudgetViolation {
                budget_id: format!("pass_ratio:{packet_id}"),
                category: "pass_ratio".to_string(),
                message: format!(
                    "packet {packet_id} pass ratio {:.3} below floor {:.3}",
                    metrics.pass_ratio, floor.min_pass_ratio
                ),
                packet_id: Some(packet_id.clone()),
                scenario_ids: metrics.scenario_ids.clone(),
                artifact_refs: policy_artifact_refs(&policy),
                remediation_hint: remediation_hint(&policy, "pass_ratio"),
            });
        }

        for required_suite in &floor.required_suites {
            if !metrics
                .suites_present
                .iter()
                .any(|suite| suite == required_suite)
            {
                violations.push(BudgetViolation {
                    budget_id: format!("required_suite:{packet_id}:{required_suite}"),
                    category: "required_suite".to_string(),
                    message: format!("packet {packet_id} missing required suite {required_suite}"),
                    packet_id: Some(packet_id.clone()),
                    scenario_ids: metrics.scenario_ids.clone(),
                    artifact_refs: policy_artifact_refs(&policy),
                    remediation_hint: remediation_hint(&policy, "required_suite"),
                });
            }
        }
    }

    if failed_entries > policy.global_budgets.max_failed_entries {
        violations.push(BudgetViolation {
            budget_id: "global:max_failed_entries".to_string(),
            category: "global_failure_ceiling".to_string(),
            message: format!(
                "failed entries {} exceed budget {}",
                failed_entries, policy.global_budgets.max_failed_entries
            ),
            packet_id: None,
            scenario_ids: entries
                .iter()
                .filter(|entry| entry.outcome != "pass")
                .map(|entry| entry.scenario_id.clone())
                .collect(),
            artifact_refs: policy_artifact_refs(&policy),
            remediation_hint: remediation_hint(&policy, "pass_ratio"),
        });
    }

    if flake_suspects.len() > policy.flake_budget.max_flake_suspects {
        violations.push(BudgetViolation {
            budget_id: "flake_budget:max_flake_suspects".to_string(),
            category: "flake_budget".to_string(),
            message: format!(
                "flake suspects {} exceed budget {}",
                flake_suspects.len(),
                policy.flake_budget.max_flake_suspects
            ),
            packet_id: None,
            scenario_ids: flake_suspects.clone(),
            artifact_refs: policy_artifact_refs(&policy),
            remediation_hint: remediation_hint(&policy, "flake_budget"),
        });
    }

    if unknown_reason_codes.len() > policy.global_budgets.max_unknown_reason_codes {
        violations.push(BudgetViolation {
            budget_id: "global:max_unknown_reason_codes".to_string(),
            category: "reason_taxonomy".to_string(),
            message: format!(
                "unknown reason codes {} exceed budget {}",
                unknown_reason_codes.len(),
                policy.global_budgets.max_unknown_reason_codes
            ),
            packet_id: None,
            scenario_ids: Vec::new(),
            artifact_refs: policy_artifact_refs(&policy),
            remediation_hint: remediation_hint(&policy, "unknown_reason"),
        });
    }

    (
        GateSummary {
            total_entries: entries.len(),
            failed_entries,
            flake_suspect_count,
            unknown_reason_code_count: unknown_reason_codes.len(),
            packet_metrics,
        },
        violations,
    )
}

#[derive(Debug, Default)]
struct PacketAccumulator {
    total_entries: usize,
    passed_entries: usize,
    suites_present: BTreeSet<String>,
    scenario_ids: BTreeSet<String>,
}

fn detect_flake_suspects(entries: &[ForensicsLogEntry]) -> Vec<String> {
    let mut by_scenario: BTreeMap<String, BTreeSet<String>> = BTreeMap::new();
    for entry in entries {
        by_scenario
            .entry(entry.scenario_id.clone())
            .or_default()
            .insert(entry.outcome.clone());
    }

    by_scenario
        .into_iter()
        .filter_map(|(scenario_id, outcomes)| {
            let has_pass = outcomes.iter().any(|outcome| outcome == "pass");
            let has_non_pass = outcomes.iter().any(|outcome| outcome != "pass");
            if has_pass && has_non_pass {
                Some(scenario_id)
            } else {
                None
            }
        })
        .collect()
}

fn collect_unknown_reason_codes(entries: &[ForensicsLogEntry]) -> Vec<String> {
    let mut unknown = BTreeSet::new();
    for entry in entries {
        if !is_known_reason_code(entry.reason_code.as_str()) {
            unknown.insert(entry.reason_code.clone());
        }
    }
    unknown.into_iter().collect()
}

fn is_known_reason_code(reason_code: &str) -> bool {
    if reason_code.is_empty() {
        return false;
    }

    [
        "parity",
        "mismatch",
        "expected_error_observed",
        "legacy_oracle_unavailable",
        "oracle_guard",
        "policy_match",
        "reentrant",
        "dependency",
        "unknown_node",
        "serialization",
        "dispatch",
        "tensor_meta",
        "scheduler",
        "op_schema",
        "nn_state",
        "fail_closed",
        "checksum",
        "version",
        "invalid_json",
    ]
    .iter()
    .any(|token| reason_code.contains(token))
}

fn policy_artifact_refs(policy: &ReliabilityPolicy) -> Vec<String> {
    policy
        .artifact_refs
        .values()
        .cloned()
        .collect::<Vec<String>>()
}

fn remediation_hint(policy: &ReliabilityPolicy, key: &str) -> String {
    policy
        .remediation_hints
        .get(key)
        .cloned()
        .unwrap_or_else(|| "No remediation hint configured".to_string())
}

fn parse_args() -> Result<(PathBuf, PathBuf, PathBuf), String> {
    let mut policy = None;
    let mut e2e = None;
    let mut output = None;

    let mut args = std::env::args().skip(1);
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--policy" => {
                let value = args
                    .next()
                    .ok_or_else(|| "--policy requires a path".to_string())?;
                policy = Some(PathBuf::from(value));
            }
            "--e2e" => {
                let value = args
                    .next()
                    .ok_or_else(|| "--e2e requires a path".to_string())?;
                e2e = Some(PathBuf::from(value));
            }
            "--output" => {
                let value = args
                    .next()
                    .ok_or_else(|| "--output requires a path".to_string())?;
                output = Some(PathBuf::from(value));
            }
            other => {
                return Err(format!(
                    "unknown arg '{other}'. usage: check_reliability_budgets --policy <json> --e2e <jsonl> --output <json>"
                ));
            }
        }
    }

    let policy_path = policy.ok_or_else(|| "missing required --policy argument".to_string())?;
    let e2e_path = e2e.ok_or_else(|| "missing required --e2e argument".to_string())?;
    let output_path = output.ok_or_else(|| "missing required --output argument".to_string())?;

    Ok((policy_path, e2e_path, output_path))
}

fn read_policy(path: &Path) -> Result<ReliabilityPolicy, String> {
    let raw = fs::read_to_string(path)
        .map_err(|error| format!("failed to read policy {}: {error}", path.display()))?;
    serde_json::from_str(&raw)
        .map_err(|error| format!("failed to parse policy {}: {error}", path.display()))
}

fn read_forensics_jsonl(path: &Path) -> Result<Vec<ForensicsLogEntry>, String> {
    let raw = fs::read_to_string(path)
        .map_err(|error| format!("failed to read forensics jsonl {}: {error}", path.display()))?;
    let mut entries = Vec::new();
    for (line_idx, line) in raw.lines().enumerate() {
        if line.trim().is_empty() {
            continue;
        }
        let parsed: ForensicsLogEntry = serde_json::from_str(line).map_err(|error| {
            format!(
                "failed to parse forensics jsonl line {} in {}: {error}",
                line_idx + 1,
                path.display()
            )
        })?;
        entries.push(parsed);
    }
    Ok(entries)
}

fn now_unix_ms() -> u128 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map_or(0, |duration| duration.as_millis())
}

#[cfg(test)]
mod tests {
    use super::{
        ForensicsLogEntry, collect_unknown_reason_codes, detect_flake_suspects,
        is_known_reason_code,
    };

    fn entry(scenario_id: &str, outcome: &str, reason_code: &str) -> ForensicsLogEntry {
        ForensicsLogEntry {
            schema_version: "ft-conformance-log-v1".to_string(),
            ts_unix_ms: 1,
            suite_id: "dispatch_key".to_string(),
            scenario_id: scenario_id.to_string(),
            fixture_id: "dispatch_key_cases.json".to_string(),
            packet_id: "FT-P2C-002".to_string(),
            mode: "strict".to_string(),
            seed: 7,
            env_fingerprint: "det64:test".to_string(),
            artifact_refs: vec!["artifact".to_string()],
            replay_command: "cargo test ...".to_string(),
            outcome: outcome.to_string(),
            reason_code: reason_code.to_string(),
        }
    }

    #[test]
    fn flake_detection_finds_conflicting_outcomes() {
        let entries = vec![
            entry("dispatch_key/strict:a", "pass", "dispatch_parity_ok"),
            entry(
                "dispatch_key/strict:a",
                "fail",
                "dispatch_expectation_mismatch",
            ),
            entry("dispatch_key/strict:b", "pass", "dispatch_parity_ok"),
        ];
        let suspects = detect_flake_suspects(entries.as_slice());
        assert_eq!(suspects, vec!["dispatch_key/strict:a".to_string()]);
    }

    #[test]
    fn known_reason_code_classifier_is_broad_but_bounded() {
        assert!(is_known_reason_code("dispatch_expectation_mismatch"));
        assert!(is_known_reason_code("serialization_parity_ok"));
        assert!(is_known_reason_code("nn_state_hook_trace_ok"));
        assert!(is_known_reason_code("op_schema_adversarial_fail_closed_ok"));
        assert!(!is_known_reason_code("totally_unknown_reason_code"));
    }

    #[test]
    fn unknown_reason_collection_is_deduplicated() {
        let entries = vec![
            entry("x", "pass", "dispatch_parity_ok"),
            entry("y", "fail", "new_unknown_reason"),
            entry("z", "fail", "new_unknown_reason"),
        ];
        let unknown = collect_unknown_reason_codes(entries.as_slice());
        assert_eq!(unknown, vec!["new_unknown_reason".to_string()]);
    }
}

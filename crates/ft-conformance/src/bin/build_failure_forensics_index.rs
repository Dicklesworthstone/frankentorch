#![forbid(unsafe_code)]

use std::collections::BTreeMap;
use std::fs;
use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};

const INDEX_SCHEMA_VERSION: &str = "ft-failure-forensics-index-v1";

#[derive(Debug, Clone, Deserialize)]
struct ForensicsLogEntry {
    #[allow(dead_code)]
    schema_version: String,
    #[allow(dead_code)]
    ts_unix_ms: u128,
    suite_id: String,
    scenario_id: String,
    fixture_id: String,
    packet_id: String,
    mode: String,
    seed: u64,
    env_fingerprint: String,
    artifact_refs: Vec<String>,
    replay_command: String,
    outcome: String,
    reason_code: String,
}

#[derive(Debug, Clone, Deserialize)]
struct CrashTriageSummary {
    #[allow(dead_code)]
    schema_version: String,
    #[allow(dead_code)]
    total_entries: usize,
    #[allow(dead_code)]
    failed_entries: usize,
    incidents: Vec<TriagedIncident>,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
struct TriagedIncident {
    incident_id: String,
    class: String,
    severity: String,
    owner_hint: String,
    packet_id: String,
    suite_id: String,
    scenario_id: String,
    mode: String,
    reason_code: String,
    replay_command: String,
    artifact_refs: Vec<String>,
    seed: u64,
    env_fingerprint: String,
    occurrences: usize,
    last_seen_unix_ms: u128,
}

#[derive(Debug, Clone, Serialize)]
struct FailureForensicsIndex {
    schema_version: &'static str,
    generated_unix_ms: u128,
    source_artifacts: SourceArtifacts,
    summary: IndexSummary,
    suite_evidence_templates: BTreeMap<String, SuiteEvidenceTemplate>,
    failures: Vec<FailureEnvelope>,
    triaged_incidents: Vec<TriagedIncident>,
}

#[derive(Debug, Clone, Serialize)]
struct SourceArtifacts {
    e2e_log_path: String,
    triage_path: String,
    differential_report_path: String,
}

#[derive(Debug, Clone, Serialize)]
struct IndexSummary {
    e2e_entries: usize,
    failed_entries: usize,
    triaged_incidents: usize,
}

#[derive(Debug, Clone, Serialize)]
struct FailureEnvelope {
    failure_id: String,
    packet_id: String,
    suite_id: String,
    scenario_id: String,
    mode: String,
    reason_code: String,
    fixture_id: String,
    replay_command: String,
    seed: u64,
    env_fingerprint: String,
    artifact_refs: Vec<String>,
    evidence_links: Vec<EvidenceLink>,
}

#[derive(Debug, Clone, Serialize)]
struct SuiteEvidenceTemplate {
    unit_property_commands: Vec<String>,
    differential_refs: Vec<String>,
    e2e_refs: Vec<String>,
    performance_refs: Vec<String>,
    raptorq_ref_pattern: String,
}

#[derive(Debug, Clone, Serialize)]
struct EvidenceLink {
    category: String,
    path_or_ref: String,
    exists: bool,
}

fn main() -> Result<(), String> {
    let (e2e_path, triage_path, output_path, differential_report_path) = parse_args()?;

    let entries = read_forensics_jsonl(e2e_path.as_path())?;
    let triage = read_triage_summary(triage_path.as_path())?;

    let suite_templates = build_suite_templates();

    let failures = entries
        .iter()
        .filter(|entry| entry.outcome != "pass")
        .map(|entry| {
            build_failure_envelope(
                entry,
                suite_templates.get(entry.suite_id.as_str()),
                e2e_path.as_path(),
                differential_report_path.as_path(),
            )
        })
        .collect::<Vec<_>>();

    let index = FailureForensicsIndex {
        schema_version: INDEX_SCHEMA_VERSION,
        generated_unix_ms: now_unix_ms(),
        source_artifacts: SourceArtifacts {
            e2e_log_path: e2e_path.display().to_string(),
            triage_path: triage_path.display().to_string(),
            differential_report_path: differential_report_path.display().to_string(),
        },
        summary: IndexSummary {
            e2e_entries: entries.len(),
            failed_entries: failures.len(),
            triaged_incidents: triage.incidents.len(),
        },
        suite_evidence_templates: suite_templates,
        failures,
        triaged_incidents: triage.incidents,
    };

    if let Some(parent) = output_path.parent() {
        fs::create_dir_all(parent).map_err(|error| {
            format!(
                "failed to create failure index output dir {}: {error}",
                parent.display()
            )
        })?;
    }

    fs::write(
        output_path.as_path(),
        serde_json::to_string_pretty(&index)
            .map_err(|error| format!("failed to serialize failure index: {error}"))?,
    )
    .map_err(|error| format!("failed to write failure index: {error}"))?;

    println!(
        "{}",
        serde_json::to_string_pretty(&index)
            .map_err(|error| format!("failed to serialize failure index: {error}"))?
    );

    Ok(())
}

fn build_suite_templates() -> BTreeMap<String, SuiteEvidenceTemplate> {
    let mut map = BTreeMap::new();

    map.insert(
        "scalar_dac".to_string(),
        SuiteEvidenceTemplate {
            unit_property_commands: vec![
                "cargo test -p ft-api session_add_backward_records_evidence -- --nocapture"
                    .to_string(),
                "cargo test -p ft-autograd add_backward_matches_expected_gradient -- --nocapture"
                    .to_string(),
                "cargo test -p ft-autograd mul_backward_matches_expected_gradient -- --nocapture"
                    .to_string(),
            ],
            differential_refs: vec![
                "artifacts/phase2c/conformance/differential_report_v1.json".to_string(),
            ],
            e2e_refs: vec!["artifacts/phase2c/e2e_forensics/e2e_matrix_full_v1.jsonl".to_string()],
            performance_refs: vec![
                "artifacts/phase2c/METHOD_STACK_REPORT_2026-02-14.md".to_string(),
                "artifacts/phase2c/VALIDATOR_OPTIMIZATION_EVIDENCE_20260213.md".to_string(),
            ],
            raptorq_ref_pattern:
                "artifacts/phase2c/{packet_id}/parity_report.{raptorq|decode_proof}.json"
                    .to_string(),
        },
    );

    map.insert(
        "tensor_meta".to_string(),
        SuiteEvidenceTemplate {
            unit_property_commands: vec![
                "cargo test -p ft-core index_rank_and_bounds_are_guarded -- --nocapture"
                    .to_string(),
                "cargo test -p ft-core custom_strides_validate_and_index_into_storage -- --nocapture"
                    .to_string(),
            ],
            differential_refs: vec![
                "artifacts/phase2c/conformance/differential_report_v1.json".to_string(),
            ],
            e2e_refs: vec![
                "artifacts/phase2c/e2e_forensics/e2e_matrix_full_v1.jsonl".to_string(),
            ],
            performance_refs: vec![
                "artifacts/phase2c/METHOD_STACK_REPORT_2026-02-14.md".to_string(),
                "artifacts/phase2c/VALIDATOR_OPTIMIZATION_EVIDENCE_20260213.md".to_string(),
            ],
            raptorq_ref_pattern: "artifacts/phase2c/{packet_id}/parity_report.{raptorq|decode_proof}.json"
                .to_string(),
        },
    );

    map.insert(
        "dispatch_key".to_string(),
        SuiteEvidenceTemplate {
            unit_property_commands: vec![
                "cargo test -p ft-dispatch strict_mode_rejects_composite_fallback -- --nocapture"
                    .to_string(),
                "cargo test -p ft-dispatch hardened_mode_allows_composite_fallback -- --nocapture"
                    .to_string(),
                "cargo test -p ft-dispatch unknown_bits_fail_closed -- --nocapture".to_string(),
            ],
            differential_refs: vec![
                "artifacts/phase2c/conformance/differential_report_v1.json".to_string(),
            ],
            e2e_refs: vec!["artifacts/phase2c/e2e_forensics/e2e_matrix_full_v1.jsonl".to_string()],
            performance_refs: vec![
                "artifacts/phase2c/METHOD_STACK_REPORT_2026-02-14.md".to_string(),
                "artifacts/phase2c/VALIDATOR_OPTIMIZATION_EVIDENCE_20260213.md".to_string(),
            ],
            raptorq_ref_pattern:
                "artifacts/phase2c/{packet_id}/parity_report.{raptorq|decode_proof}.json"
                    .to_string(),
        },
    );

    map.insert(
        "autograd_scheduler".to_string(),
        SuiteEvidenceTemplate {
            unit_property_commands: vec![
                "cargo test -p ft-autograd composite_graph_gradient_is_deterministic -- --nocapture"
                    .to_string(),
                "cargo test -p ft-autograd dependency_scheduler_waits_for_all_children -- --nocapture"
                    .to_string(),
                "cargo test -p ft-autograd strict_mode_reentrant_depth_overflow_fails -- --nocapture"
                    .to_string(),
                "cargo test -p ft-autograd hardened_mode_reentrant_depth_overflow_fallbacks -- --nocapture"
                    .to_string(),
            ],
            differential_refs: vec![
                "artifacts/phase2c/conformance/differential_report_v1.json".to_string(),
            ],
            e2e_refs: vec![
                "artifacts/phase2c/e2e_forensics/e2e_matrix_full_v1.jsonl".to_string(),
            ],
            performance_refs: vec![
                "artifacts/phase2c/METHOD_STACK_REPORT_2026-02-14.md".to_string(),
                "artifacts/phase2c/VALIDATOR_OPTIMIZATION_EVIDENCE_20260213.md".to_string(),
            ],
            raptorq_ref_pattern: "artifacts/phase2c/{packet_id}/parity_report.{raptorq|decode_proof}.json"
                .to_string(),
        },
    );

    map.insert(
        "serialization".to_string(),
        SuiteEvidenceTemplate {
            unit_property_commands: vec![
                "cargo test -p ft-serialize strict_unknown_field_fail_closed -- --nocapture"
                    .to_string(),
                "cargo test -p ft-serialize hardened_malformed_payload_returns_bounded_diagnostic -- --nocapture"
                    .to_string(),
                "cargo test -p ft-serialize version_mismatch_is_fail_closed -- --nocapture"
                    .to_string(),
                "cargo test -p ft-serialize checksum_mismatch_is_fail_closed -- --nocapture"
                    .to_string(),
            ],
            differential_refs: vec![
                "artifacts/phase2c/conformance/differential_report_v1.json".to_string(),
            ],
            e2e_refs: vec![
                "artifacts/phase2c/e2e_forensics/e2e_matrix_full_v1.jsonl".to_string(),
            ],
            performance_refs: vec![
                "artifacts/phase2c/METHOD_STACK_REPORT_2026-02-14.md".to_string(),
                "artifacts/phase2c/VALIDATOR_OPTIMIZATION_EVIDENCE_20260213.md".to_string(),
            ],
            raptorq_ref_pattern: "artifacts/phase2c/{packet_id}/parity_report.{raptorq|decode_proof}.json"
                .to_string(),
        },
    );

    map
}

fn build_failure_envelope(
    entry: &ForensicsLogEntry,
    suite_template: Option<&SuiteEvidenceTemplate>,
    e2e_path: &Path,
    differential_path: &Path,
) -> FailureEnvelope {
    let failure_id = format!(
        "ffx64:{:016x}",
        det_hash64(
            [
                entry.packet_id.as_str(),
                entry.suite_id.as_str(),
                entry.scenario_id.as_str(),
                entry.mode.as_str(),
                entry.reason_code.as_str(),
            ]
            .as_slice()
        )
    );

    let mut evidence_links = Vec::new();

    for unit_cmd in suite_template
        .map(|template| template.unit_property_commands.as_slice())
        .unwrap_or_default()
    {
        evidence_links.push(EvidenceLink {
            category: "unit_property".to_string(),
            path_or_ref: unit_cmd.clone(),
            exists: false,
        });
    }

    evidence_links.push(EvidenceLink {
        category: "differential".to_string(),
        path_or_ref: differential_path.display().to_string(),
        exists: differential_path.exists(),
    });

    evidence_links.push(EvidenceLink {
        category: "e2e".to_string(),
        path_or_ref: e2e_path.display().to_string(),
        exists: e2e_path.exists(),
    });

    for perf_path in [
        "artifacts/phase2c/METHOD_STACK_REPORT_2026-02-14.md",
        "artifacts/phase2c/VALIDATOR_OPTIMIZATION_EVIDENCE_20260213.md",
    ] {
        evidence_links.push(path_link("performance", perf_path));
    }

    for raptorq_path in packet_raptorq_paths(entry.packet_id.as_str()) {
        evidence_links.push(path_link("raptorq", raptorq_path.as_str()));
    }

    FailureEnvelope {
        failure_id,
        packet_id: entry.packet_id.clone(),
        suite_id: entry.suite_id.clone(),
        scenario_id: entry.scenario_id.clone(),
        mode: entry.mode.clone(),
        reason_code: entry.reason_code.clone(),
        fixture_id: entry.fixture_id.clone(),
        replay_command: entry.replay_command.clone(),
        seed: entry.seed,
        env_fingerprint: entry.env_fingerprint.clone(),
        artifact_refs: entry.artifact_refs.clone(),
        evidence_links,
    }
}

fn packet_raptorq_paths(packet_id: &str) -> Vec<String> {
    vec![
        format!("artifacts/phase2c/{packet_id}/parity_report.raptorq.json"),
        format!("artifacts/phase2c/{packet_id}/parity_report.decode_proof.json"),
        format!("artifacts/phase2c/{packet_id}/parity_report.json"),
    ]
}

fn path_link(category: &str, path: &str) -> EvidenceLink {
    EvidenceLink {
        category: category.to_string(),
        path_or_ref: path.to_string(),
        exists: Path::new(path).exists(),
    }
}

fn det_hash64(parts: &[&str]) -> u64 {
    let mut hash = 0xcbf2_9ce4_8422_2325u64;
    for part in parts {
        for byte in part.as_bytes() {
            hash ^= u64::from(*byte);
            hash = hash.wrapping_mul(0x0000_0100_0000_01b3);
        }
        hash ^= 0xff;
        hash = hash.wrapping_mul(0x0000_0100_0000_01b3);
    }
    hash
}

fn parse_args() -> Result<(PathBuf, PathBuf, PathBuf, PathBuf), String> {
    let mut e2e = None;
    let mut triage = None;
    let mut output = None;
    let mut differential = Some(PathBuf::from(
        "artifacts/phase2c/conformance/differential_report_v1.json",
    ));

    let mut args = std::env::args().skip(1);
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--e2e" => {
                let value = args
                    .next()
                    .ok_or_else(|| "--e2e requires a path".to_string())?;
                e2e = Some(PathBuf::from(value));
            }
            "--triage" => {
                let value = args
                    .next()
                    .ok_or_else(|| "--triage requires a path".to_string())?;
                triage = Some(PathBuf::from(value));
            }
            "--output" => {
                let value = args
                    .next()
                    .ok_or_else(|| "--output requires a path".to_string())?;
                output = Some(PathBuf::from(value));
            }
            "--differential" => {
                let value = args
                    .next()
                    .ok_or_else(|| "--differential requires a path".to_string())?;
                differential = Some(PathBuf::from(value));
            }
            other => {
                return Err(format!(
                    "unknown arg '{other}'. usage: build_failure_forensics_index --e2e <jsonl> --triage <json> --output <json> [--differential <json>]"
                ));
            }
        }
    }

    let e2e_path = e2e.ok_or_else(|| "missing required --e2e argument".to_string())?;
    let triage_path = triage.ok_or_else(|| "missing required --triage argument".to_string())?;
    let output_path = output.ok_or_else(|| "missing required --output argument".to_string())?;
    let differential_path =
        differential.ok_or_else(|| "failed to resolve differential report path".to_string())?;

    Ok((e2e_path, triage_path, output_path, differential_path))
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

fn read_triage_summary(path: &Path) -> Result<CrashTriageSummary, String> {
    let raw = fs::read_to_string(path)
        .map_err(|error| format!("failed to read triage summary {}: {error}", path.display()))?;
    serde_json::from_str(&raw)
        .map_err(|error| format!("failed to parse triage summary {}: {error}", path.display()))
}

fn now_unix_ms() -> u128 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map_or(0, |duration| duration.as_millis())
}

#[cfg(test)]
mod tests {
    use super::{
        ForensicsLogEntry, build_failure_envelope, build_suite_templates, det_hash64,
        packet_raptorq_paths,
    };
    use std::path::Path;

    #[test]
    fn det_hash_is_stable() {
        let first = det_hash64(["FT-P2C-001", "suite", "case"].as_slice());
        let second = det_hash64(["FT-P2C-001", "suite", "case"].as_slice());
        assert_eq!(first, second);
    }

    #[test]
    fn packet_raptorq_paths_include_expected_suffixes() {
        let paths = packet_raptorq_paths("FT-P2C-002");
        assert!(
            paths
                .iter()
                .any(|path| path.ends_with("parity_report.raptorq.json"))
        );
        assert!(
            paths
                .iter()
                .any(|path| path.ends_with("parity_report.decode_proof.json"))
        );
    }

    #[test]
    fn failure_envelope_carries_replay_and_environment_fields() {
        let entry = ForensicsLogEntry {
            schema_version: "ft-conformance-log-v1".to_string(),
            ts_unix_ms: 1,
            suite_id: "dispatch_key".to_string(),
            scenario_id: "dispatch_key/strict:composite_route_mode_split".to_string(),
            fixture_id: "dispatch_key_cases.json".to_string(),
            packet_id: "FT-P2C-002".to_string(),
            mode: "strict".to_string(),
            seed: 7,
            env_fingerprint: "det64:test".to_string(),
            artifact_refs: vec!["foo".to_string()],
            replay_command: "cargo test ...".to_string(),
            outcome: "fail".to_string(),
            reason_code: "dispatch_expectation_mismatch".to_string(),
        };

        let templates = build_suite_templates();
        let envelope = build_failure_envelope(
            &entry,
            templates.get("dispatch_key"),
            Path::new("artifacts/phase2c/e2e_forensics/e2e_matrix_full_v1.jsonl"),
            Path::new("artifacts/phase2c/conformance/differential_report_v1.json"),
        );

        assert_eq!(envelope.scenario_id, entry.scenario_id);
        assert_eq!(envelope.replay_command, entry.replay_command);
        assert_eq!(envelope.env_fingerprint, entry.env_fingerprint);
        assert!(!envelope.evidence_links.is_empty());
        assert!(
            envelope
                .evidence_links
                .iter()
                .any(|link| link.category == "raptorq")
        );
    }
}

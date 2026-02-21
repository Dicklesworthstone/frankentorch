#![forbid(unsafe_code)]

use std::collections::BTreeMap;
use std::fs;
use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};

const TRIAGE_SCHEMA_VERSION: &str = "ft-crash-triage-v1";

#[derive(Debug, Clone, Deserialize)]
struct ForensicsLogEntry {
    #[allow(dead_code)]
    schema_version: String,
    ts_unix_ms: u128,
    suite_id: String,
    scenario_id: String,
    #[allow(dead_code)]
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

#[derive(Debug, Clone, Serialize)]
struct CrashTriageSummary {
    schema_version: &'static str,
    generated_unix_ms: u128,
    input_log_path: String,
    packet_filter: Option<String>,
    total_entries: usize,
    failed_entries: usize,
    class_counts: BTreeMap<String, usize>,
    incidents: Vec<CrashIncident>,
}

#[derive(Debug, Clone, Serialize)]
struct CrashIncident {
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

fn main() -> Result<(), String> {
    let (input_path, output_path, packet_filter) = parse_args()?;
    let entries = read_forensics_jsonl(input_path.as_path())?;
    let summary = triage(
        entries.as_slice(),
        packet_filter.clone(),
        input_path.as_path(),
    );

    if let Some(parent) = output_path.parent() {
        fs::create_dir_all(parent).map_err(|error| {
            format!(
                "failed to create triage output dir {}: {error}",
                parent.display()
            )
        })?;
    }

    fs::write(
        output_path.as_path(),
        serde_json::to_string_pretty(&summary)
            .map_err(|error| format!("failed to serialize triage summary: {error}"))?,
    )
    .map_err(|error| format!("failed to write triage summary: {error}"))?;

    println!(
        "{}",
        serde_json::to_string_pretty(&summary)
            .map_err(|error| format!("failed to serialize triage summary: {error}"))?
    );

    Ok(())
}

fn triage(
    entries: &[ForensicsLogEntry],
    packet_filter: Option<String>,
    input_path: &Path,
) -> CrashTriageSummary {
    let mut failed_entries = 0usize;
    let mut class_counts: BTreeMap<String, usize> = BTreeMap::new();
    let mut incidents_by_id: BTreeMap<String, CrashIncident> = BTreeMap::new();

    for entry in entries {
        if let Some(filter) = packet_filter.as_deref()
            && entry.packet_id != filter
        {
            continue;
        }
        if entry.outcome == "pass" {
            continue;
        }

        failed_entries += 1;
        let (class, severity, owner_hint) = classify(entry.reason_code.as_str());
        let incident_id = format!(
            "triage64:{:016x}",
            det_hash64(
                [
                    entry.packet_id.as_str(),
                    entry.suite_id.as_str(),
                    entry.scenario_id.as_str(),
                    entry.mode.as_str(),
                    entry.reason_code.as_str(),
                    class,
                ]
                .as_slice()
            )
        );

        *class_counts.entry(class.to_string()).or_insert(0) += 1;

        if let Some(existing) = incidents_by_id.get_mut(incident_id.as_str()) {
            existing.occurrences += 1;
            existing.last_seen_unix_ms = existing.last_seen_unix_ms.max(entry.ts_unix_ms);
            continue;
        }

        incidents_by_id.insert(
            incident_id.clone(),
            CrashIncident {
                incident_id,
                class: class.to_string(),
                severity: severity.to_string(),
                owner_hint: owner_hint.to_string(),
                packet_id: entry.packet_id.clone(),
                suite_id: entry.suite_id.clone(),
                scenario_id: entry.scenario_id.clone(),
                mode: entry.mode.clone(),
                reason_code: entry.reason_code.clone(),
                replay_command: entry.replay_command.clone(),
                artifact_refs: entry.artifact_refs.clone(),
                seed: entry.seed,
                env_fingerprint: entry.env_fingerprint.clone(),
                occurrences: 1,
                last_seen_unix_ms: entry.ts_unix_ms,
            },
        );
    }

    let mut incidents: Vec<CrashIncident> = incidents_by_id.into_values().collect();
    incidents.sort_by(|left, right| {
        severity_rank(left.severity.as_str())
            .cmp(&severity_rank(right.severity.as_str()))
            .then(left.incident_id.cmp(&right.incident_id))
    });

    CrashTriageSummary {
        schema_version: TRIAGE_SCHEMA_VERSION,
        generated_unix_ms: now_unix_ms(),
        input_log_path: input_path.display().to_string(),
        packet_filter,
        total_entries: entries.len(),
        failed_entries,
        class_counts,
        incidents,
    }
}

fn classify(reason_code: &str) -> (&'static str, &'static str, &'static str) {
    let normalized = reason_code.to_ascii_lowercase();

    if normalized.contains("unavailable") || normalized.contains("timeout") {
        if normalized.contains("oracle") {
            return ("oracle_infra", "medium", "ft-conformance-infra-owners");
        }
    }

    if normalized.contains("reentrant")
        || normalized.contains("dependency")
        || normalized.contains("unknown_node")
        || normalized.contains("scheduler")
    {
        return ("autograd_state", "critical", "ft-autograd-owners");
    }
    if normalized.contains("op_schema")
        || (normalized.contains("schema")
            && (normalized.contains("parse")
                || normalized.contains("variant")
                || normalized.contains("alias")
                || normalized.contains("metadata")))
    {
        return ("op_schema_contract", "high", "ft-dispatch-owners");
    }
    if normalized.contains("dispatch")
        || normalized.contains("keyset")
        || normalized.contains("fallback")
    {
        return ("dispatch_routing", "high", "ft-dispatch-owners");
    }
    if normalized.contains("serialization")
        || normalized.contains("decode")
        || normalized.contains("checksum")
        || normalized.contains("unknown_field")
        || normalized.contains("invalid_json")
        || normalized.contains("version_mismatch")
    {
        return (
            "serialization_parser",
            "high",
            "ft-serialize-and-durability-owners",
        );
    }
    if normalized.contains("tensor_meta")
        || normalized.contains("rank")
        || normalized.contains("index")
        || normalized.contains("stride")
    {
        return ("tensor_meta_state", "high", "ft-core-owners");
    }
    if normalized.contains("scalar")
        || normalized.contains("tensor_binary")
        || normalized.contains("tensor_unary")
        || normalized.contains("tensor_reduction")
        || normalized.contains("tensor_join")
        || normalized.contains("math")
    {
        return ("kernel_math", "critical", "ft-kernel-cpu-owners");
    }
    if normalized.contains("nn_state") {
        return ("nn_module_state", "high", "ft-nn-owners");
    }
    if normalized.contains("optimizer") {
        return ("optimizer_state", "high", "ft-optim-owners");
    }

    ("unclassified", "high", "crash-triage-owner")
}

fn severity_rank(severity: &str) -> u8 {
    match severity {
        "critical" => 0,
        "high" => 1,
        "medium" => 2,
        "low" => 3,
        _ => 4,
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

fn parse_args() -> Result<(PathBuf, PathBuf, Option<String>), String> {
    let mut input = None;
    let mut output = None;
    let mut packet_filter = None;

    let mut args = std::env::args().skip(1);
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--input" => {
                let value = args
                    .next()
                    .ok_or_else(|| "--input requires a path".to_string())?;
                input = Some(PathBuf::from(value));
            }
            "--output" => {
                let value = args
                    .next()
                    .ok_or_else(|| "--output requires a path".to_string())?;
                output = Some(PathBuf::from(value));
            }
            "--packet" => {
                let value = args
                    .next()
                    .ok_or_else(|| "--packet requires a packet id".to_string())?;
                packet_filter = Some(value);
            }
            other => {
                return Err(format!(
                    "unknown arg '{other}'. usage: triage_forensics_failures --input <jsonl> --output <json> [--packet FT-P2C-00X]"
                ));
            }
        }
    }

    let input_path = input
        .ok_or_else(|| "missing required --input argument (forensics jsonl path)".to_string())?;
    let output_path = output
        .ok_or_else(|| "missing required --output argument (triage json path)".to_string())?;
    Ok((input_path, output_path, packet_filter))
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
    use super::{classify, det_hash64, severity_rank};

    #[test]
    fn reason_code_classification_is_stable() {
        assert_eq!(
            classify("legacy_oracle_unavailable"),
            ("oracle_infra", "medium", "ft-conformance-infra-owners")
        );
        assert_eq!(
            classify("scheduler_expectation_mismatch"),
            ("autograd_state", "critical", "ft-autograd-owners")
        );
        assert_eq!(
            classify("dispatch_expectation_mismatch"),
            ("dispatch_routing", "high", "ft-dispatch-owners")
        );
        assert_eq!(
            classify("op_schema_parse_expectation_mismatch"),
            ("op_schema_contract", "high", "ft-dispatch-owners")
        );
        assert_eq!(
            classify("serialization_expectation_mismatch"),
            (
                "serialization_parser",
                "high",
                "ft-serialize-and-durability-owners"
            )
        );
        assert_eq!(
            classify("tensor_meta_expectation_mismatch"),
            ("tensor_meta_state", "high", "ft-core-owners")
        );
    }

    #[test]
    fn det_hash_is_deterministic() {
        let first = det_hash64(["suite", "case", "mode"].as_slice());
        let second = det_hash64(["suite", "case", "mode"].as_slice());
        assert_eq!(first, second);
    }

    #[test]
    fn severity_order_matches_priority() {
        assert!(severity_rank("critical") < severity_rank("high"));
        assert!(severity_rank("high") < severity_rank("medium"));
        assert!(severity_rank("medium") < severity_rank("low"));
    }
}

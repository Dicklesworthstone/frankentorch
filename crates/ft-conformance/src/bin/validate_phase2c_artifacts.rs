#![forbid(unsafe_code)]

use std::collections::BTreeMap;
use std::fs;
use std::path::{Path, PathBuf};

use serde::Serialize;
use serde_json::Value;

const REQUIRED_PACKET_FILES: [&str; 8] = [
    "legacy_anchor_map.md",
    "contract_table.md",
    "fixture_manifest.json",
    "parity_gate.yaml",
    "risk_note.md",
    "parity_report.json",
    "parity_report.raptorq.json",
    "parity_report.decode_proof.json",
];

const REQUIRED_FIXTURE_MANIFEST_KEYS: [&str; 4] = ["packet_id", "fixtures", "modes", "status"];
const REQUIRED_PARITY_REPORT_KEYS: [&str; 6] = [
    "packet_id",
    "suite",
    "strict",
    "hardened",
    "status",
    "generated_from",
];
const REQUIRED_RAPTORQ_KEYS: [&str; 5] = [
    "artifact_id",
    "artifact_type",
    "source_hash",
    "raptorq",
    "scrub",
];

const SECURITY_MATRIX_FILE: &str = "artifacts/phase2c/SECURITY_COMPATIBILITY_THREAT_MATRIX_V1.md";
const HARDENED_ALLOWLIST_FILE: &str = "artifacts/phase2c/HARDENED_DEVIATION_ALLOWLIST_V1.json";
const REQUIRED_ALLOWLIST_TOP_LEVEL_KEYS: [&str; 5] = [
    "schema_version",
    "policy",
    "strict_mode",
    "hardened_mode",
    "packets",
];

#[derive(Debug, Serialize)]
struct ValidationSummary {
    schema_version: &'static str,
    root: String,
    packet_count: usize,
    packets: Vec<PacketValidation>,
    global: GlobalValidation,
    ok: bool,
}

#[derive(Debug, Serialize)]
struct PacketValidation {
    packet_id: String,
    status: &'static str,
    errors: Vec<String>,
    warnings: Vec<String>,
    checks: BTreeMap<String, bool>,
}

#[derive(Debug, Serialize)]
struct GlobalValidation {
    status: &'static str,
    errors: Vec<String>,
    warnings: Vec<String>,
    checks: BTreeMap<String, bool>,
}

fn main() {
    let root = std::env::args()
        .nth(1)
        .map(PathBuf::from)
        .unwrap_or_else(default_root);
    let phase2c_root = root.join("artifacts/phase2c");

    let mut packets = Vec::new();

    let Ok(entries) = fs::read_dir(&phase2c_root) else {
        eprintln!("phase2c root missing: {}", phase2c_root.display());
        std::process::exit(2);
    };

    for entry in entries.flatten() {
        let Ok(file_type) = entry.file_type() else {
            continue;
        };
        if !file_type.is_dir() {
            continue;
        }
        let file_name = entry.file_name();
        let Some(packet_id) = file_name.to_str() else {
            continue;
        };
        if !packet_id.starts_with("FT-P2C-") {
            continue;
        }
        packets.push(validate_packet(packet_id, &entry.path()));
    }

    packets.sort_by(|left, right| left.packet_id.cmp(&right.packet_id));
    let packet_ids: Vec<String> = packets
        .iter()
        .map(|packet| packet.packet_id.clone())
        .collect();
    let global = validate_global(&root, packet_ids.as_slice());
    let ok = packets.iter().all(|packet| packet.status == "READY") && global.status == "READY";

    let summary = ValidationSummary {
        schema_version: "phase2c-packet-lock-v1",
        root: root.display().to_string(),
        packet_count: packets.len(),
        packets,
        global,
        ok,
    };

    println!(
        "{}",
        serde_json::to_string_pretty(&summary)
            .expect("validation summary serialization should succeed")
    );

    if summary.ok {
        std::process::exit(0);
    }
    std::process::exit(1);
}

fn validate_global(root: &Path, packet_ids: &[String]) -> GlobalValidation {
    let mut errors = Vec::new();
    let mut warnings = Vec::new();
    let mut checks = BTreeMap::new();

    let matrix_path = root.join(SECURITY_MATRIX_FILE);
    let matrix_exists = matrix_path.exists();
    checks.insert("security_matrix:file_exists".to_string(), matrix_exists);
    if !matrix_exists {
        errors.push(format!(
            "missing required security matrix file '{}'",
            SECURITY_MATRIX_FILE
        ));
    }

    if let Some(raw) = read_file(&matrix_path) {
        let has_strict = raw.contains("strict mode");
        checks.insert(
            "security_matrix:contains_strict_mode".to_string(),
            has_strict,
        );
        if !has_strict {
            errors.push("security matrix must include strict mode section".to_string());
        }

        let has_hardened = raw.contains("hardened mode");
        checks.insert(
            "security_matrix:contains_hardened_mode".to_string(),
            has_hardened,
        );
        if !has_hardened {
            errors.push("security matrix must include hardened mode section".to_string());
        }

        for packet_id in packet_ids {
            let present = raw.contains(packet_id);
            checks.insert(format!("security_matrix:packet_ref:{packet_id}"), present);
            if !present {
                warnings.push(format!(
                    "security matrix does not explicitly reference packet '{packet_id}'"
                ));
            }
        }
    }

    let allowlist_path = root.join(HARDENED_ALLOWLIST_FILE);
    let allowlist_exists = allowlist_path.exists();
    checks.insert("allowlist:file_exists".to_string(), allowlist_exists);
    if !allowlist_exists {
        errors.push(format!(
            "missing required allowlist file '{}'",
            HARDENED_ALLOWLIST_FILE
        ));
    }

    if let Some(value) = read_json(&allowlist_path) {
        check_required_keys(
            &value,
            HARDENED_ALLOWLIST_FILE,
            REQUIRED_ALLOWLIST_TOP_LEVEL_KEYS.as_slice(),
            &mut errors,
            &mut checks,
        );

        let schema_ok = value
            .get("schema_version")
            .and_then(Value::as_str)
            .is_some_and(|schema| schema == "hardened-deviation-allowlist-v1");
        checks.insert("allowlist:schema_version".to_string(), schema_ok);
        if !schema_ok {
            errors.push(
                "allowlist schema_version must be 'hardened-deviation-allowlist-v1'".to_string(),
            );
        }

        let packets_obj = value.get("packets").and_then(Value::as_object);
        let packets_obj_present = packets_obj.is_some();
        checks.insert("allowlist:packets_object".to_string(), packets_obj_present);
        if let Some(packets_obj) = packets_obj {
            for packet_id in packet_ids {
                let Some(packet_entry) = packets_obj.get(packet_id) else {
                    errors.push(format!("allowlist missing packet entry '{packet_id}'"));
                    checks.insert(format!("allowlist:packet_present:{packet_id}"), false);
                    continue;
                };

                checks.insert(format!("allowlist:packet_present:{packet_id}"), true);

                let disallowed_flag = packet_entry
                    .get("disallowed_by_default")
                    .and_then(Value::as_bool)
                    .is_some_and(|flag| flag);
                checks.insert(
                    format!("allowlist:disallowed_by_default:{packet_id}"),
                    disallowed_flag,
                );
                if !disallowed_flag {
                    errors.push(format!(
                        "allowlist packet '{packet_id}' must set disallowed_by_default=true"
                    ));
                }

                let deviations_is_array = packet_entry
                    .get("allowed_deviations")
                    .and_then(Value::as_array)
                    .is_some();
                checks.insert(
                    format!("allowlist:allowed_deviations_array:{packet_id}"),
                    deviations_is_array,
                );
                if !deviations_is_array {
                    errors.push(format!(
                        "allowlist packet '{packet_id}' must include allowed_deviations array"
                    ));
                }
            }
        }
    } else if allowlist_exists {
        errors.push(format!(
            "{} is present but is not valid json",
            HARDENED_ALLOWLIST_FILE
        ));
    }

    let status = if errors.is_empty() {
        "READY"
    } else {
        "NOT_READY"
    };
    GlobalValidation {
        status,
        errors,
        warnings,
        checks,
    }
}

fn validate_packet(packet_id: &str, packet_path: &Path) -> PacketValidation {
    let mut errors = Vec::new();
    let mut warnings = Vec::new();
    let mut checks = BTreeMap::new();
    let mut packet_files: BTreeMap<&'static str, Option<String>> = BTreeMap::new();

    for file in REQUIRED_PACKET_FILES {
        let file_path = packet_path.join(file);
        let raw = read_file(file_path.as_path());
        let exists = raw.is_some() || file_path.exists();
        checks.insert(format!("file:{file}"), exists);
        if !exists {
            errors.push(format!("missing required artifact file '{file}'"));
        }
        packet_files.insert(file, raw);
    }

    validate_markdown(
        packet_id,
        "legacy_anchor_map.md",
        &packet_files,
        &mut errors,
        &mut checks,
    );
    validate_markdown(
        packet_id,
        "contract_table.md",
        &packet_files,
        &mut errors,
        &mut checks,
    );
    validate_markdown(
        packet_id,
        "risk_note.md",
        &packet_files,
        &mut errors,
        &mut checks,
    );

    validate_fixture_manifest(packet_id, &packet_files, &mut errors, &mut checks);
    validate_parity_report(packet_id, &packet_files, &mut errors, &mut checks);
    validate_raptorq_report(packet_id, &packet_files, &mut errors, &mut checks);
    validate_decode_proof_report(packet_id, &packet_files, &mut errors, &mut checks);
    validate_parity_gate(
        packet_id,
        &packet_files,
        &mut errors,
        &mut warnings,
        &mut checks,
    );

    let status = if errors.is_empty() {
        "READY"
    } else {
        "NOT_READY"
    };

    PacketValidation {
        packet_id: packet_id.to_string(),
        status,
        errors,
        warnings,
        checks,
    }
}

fn validate_markdown(
    packet_id: &str,
    file: &str,
    packet_files: &BTreeMap<&'static str, Option<String>>,
    errors: &mut Vec<String>,
    checks: &mut BTreeMap<String, bool>,
) {
    let Some(raw) = packet_file(packet_files, file) else {
        return;
    };

    let non_empty = !raw.trim().is_empty();
    checks.insert(format!("md_non_empty:{file}"), non_empty);
    if !non_empty {
        errors.push(format!("'{file}' is empty"));
    }

    let has_packet_id = raw.contains(packet_id);
    checks.insert(format!("md_contains_packet_id:{file}"), has_packet_id);
    if !has_packet_id {
        errors.push(format!("'{file}' does not contain packet id '{packet_id}'"));
    }
}

fn validate_fixture_manifest(
    packet_id: &str,
    packet_files: &BTreeMap<&'static str, Option<String>>,
    errors: &mut Vec<String>,
    checks: &mut BTreeMap<String, bool>,
) {
    let Some(raw) = packet_file(packet_files, "fixture_manifest.json") else {
        errors.push("fixture_manifest.json is missing or invalid json".to_string());
        return;
    };
    let Some(value) = parse_json(raw) else {
        errors.push("fixture_manifest.json is missing or invalid json".to_string());
        return;
    };

    check_required_keys(
        &value,
        "fixture_manifest.json",
        REQUIRED_FIXTURE_MANIFEST_KEYS.as_slice(),
        errors,
        checks,
    );

    let packet_matches = value
        .get("packet_id")
        .and_then(Value::as_str)
        .is_some_and(|id| id == packet_id);
    checks.insert(
        "fixture_manifest:packet_id_matches".to_string(),
        packet_matches,
    );
    if !packet_matches {
        errors.push(format!(
            "fixture_manifest.json packet_id does not match directory '{packet_id}'"
        ));
    }
}

fn validate_parity_report(
    packet_id: &str,
    packet_files: &BTreeMap<&'static str, Option<String>>,
    errors: &mut Vec<String>,
    checks: &mut BTreeMap<String, bool>,
) {
    let Some(raw) = packet_file(packet_files, "parity_report.json") else {
        errors.push("parity_report.json is missing or invalid json".to_string());
        return;
    };
    let Some(value) = parse_json(raw) else {
        errors.push("parity_report.json is missing or invalid json".to_string());
        return;
    };

    check_required_keys(
        &value,
        "parity_report.json",
        REQUIRED_PARITY_REPORT_KEYS.as_slice(),
        errors,
        checks,
    );

    let packet_matches = value
        .get("packet_id")
        .and_then(Value::as_str)
        .is_some_and(|id| id == packet_id);
    checks.insert(
        "parity_report:packet_id_matches".to_string(),
        packet_matches,
    );
    if !packet_matches {
        errors.push(format!(
            "parity_report.json packet_id does not match directory '{packet_id}'"
        ));
    }
}

fn validate_raptorq_report(
    packet_id: &str,
    packet_files: &BTreeMap<&'static str, Option<String>>,
    errors: &mut Vec<String>,
    checks: &mut BTreeMap<String, bool>,
) {
    let Some(raw) = packet_file(packet_files, "parity_report.raptorq.json") else {
        errors.push("parity_report.raptorq.json is missing or invalid json".to_string());
        return;
    };
    let Some(value) = parse_json(raw) else {
        errors.push("parity_report.raptorq.json is missing or invalid json".to_string());
        return;
    };

    check_required_keys(
        &value,
        "parity_report.raptorq.json",
        REQUIRED_RAPTORQ_KEYS.as_slice(),
        errors,
        checks,
    );

    let artifact_ok = value
        .get("artifact_id")
        .and_then(Value::as_str)
        .is_some_and(|artifact_id| artifact_id.starts_with(packet_id));
    checks.insert("raptorq:artifact_id_prefix".to_string(), artifact_ok);
    if !artifact_ok {
        errors.push(format!(
            "parity_report.raptorq.json artifact_id must start with '{packet_id}'"
        ));
    }
}

fn validate_decode_proof_report(
    packet_id: &str,
    packet_files: &BTreeMap<&'static str, Option<String>>,
    errors: &mut Vec<String>,
    checks: &mut BTreeMap<String, bool>,
) {
    let Some(raw) = packet_file(packet_files, "parity_report.decode_proof.json") else {
        errors.push("parity_report.decode_proof.json is missing or invalid json".to_string());
        return;
    };
    let Some(value) = parse_json(raw) else {
        errors.push("parity_report.decode_proof.json is missing or invalid json".to_string());
        return;
    };

    let artifact_ok = value
        .get("artifact_id")
        .and_then(Value::as_str)
        .is_some_and(|artifact_id| artifact_id.starts_with(packet_id));
    checks.insert("decode_proof:artifact_id_prefix".to_string(), artifact_ok);
    if !artifact_ok {
        errors.push(format!(
            "parity_report.decode_proof.json artifact_id must start with '{packet_id}'"
        ));
    }

    let has_decode_payload =
        value.get("decode_proof").is_some() || value.get("decode_events").is_some();
    checks.insert(
        "decode_proof:payload_present".to_string(),
        has_decode_payload,
    );
    if !has_decode_payload {
        errors.push(
            "parity_report.decode_proof.json must contain either 'decode_proof' or 'decode_events'"
                .to_string(),
        );
    }
}

fn validate_parity_gate(
    packet_id: &str,
    packet_files: &BTreeMap<&'static str, Option<String>>,
    errors: &mut Vec<String>,
    warnings: &mut Vec<String>,
    checks: &mut BTreeMap<String, bool>,
) {
    let Some(raw) = packet_file(packet_files, "parity_gate.yaml") else {
        return;
    };

    let has_packet = raw
        .lines()
        .map(str::trim)
        .any(|line| line == format!("packet_id: {packet_id}"));
    checks.insert("parity_gate:packet_id_matches".to_string(), has_packet);
    if !has_packet {
        errors.push(format!(
            "parity_gate.yaml missing exact 'packet_id: {packet_id}' line"
        ));
    }

    for required in ["strict:", "hardened:", "checks:", "artifacts:"] {
        let present = raw.contains(required);
        checks.insert(format!("parity_gate:section:{required}"), present);
        if !present {
            errors.push(format!(
                "parity_gate.yaml missing required section '{required}'"
            ));
        }
    }

    if !raw.contains("required_cases_passed_ratio: 1.0") {
        warnings
            .push("parity_gate.yaml does not enforce required_cases_passed_ratio: 1.0".to_string());
    }
}

fn check_required_keys(
    value: &Value,
    name: &str,
    required: &[&str],
    errors: &mut Vec<String>,
    checks: &mut BTreeMap<String, bool>,
) {
    for key in required {
        let present = value.get(*key).is_some();
        checks.insert(format!("{name}:key:{key}"), present);
        if !present {
            errors.push(format!("{name} missing mandatory field '{key}'"));
        }
    }
}

fn read_file(path: &Path) -> Option<String> {
    fs::read_to_string(path).ok()
}

fn read_json(path: &Path) -> Option<Value> {
    let raw = read_file(path)?;
    parse_json(&raw)
}

fn parse_json(raw: &str) -> Option<Value> {
    serde_json::from_str::<Value>(raw).ok()
}

fn packet_file<'a>(
    packet_files: &'a BTreeMap<&'static str, Option<String>>,
    file: &str,
) -> Option<&'a str> {
    packet_files.get(file).and_then(|raw| raw.as_deref())
}

fn default_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..")
}

#[cfg(test)]
mod tests {
    use super::{check_required_keys, validate_markdown};
    use serde_json::json;
    use std::collections::BTreeMap;

    #[test]
    fn required_key_check_reports_missing_fields() {
        let value = json!({"packet_id": "FT-P2C-999"});
        let mut errors = Vec::new();
        let mut checks = BTreeMap::new();

        check_required_keys(
            &value,
            "fixture_manifest.json",
            &["packet_id", "fixtures"],
            &mut errors,
            &mut checks,
        );

        assert!(
            errors
                .iter()
                .any(|err| err.contains("mandatory field 'fixtures'"))
        );
        assert_eq!(
            checks.get("fixture_manifest.json:key:packet_id"),
            Some(&true)
        );
        assert_eq!(
            checks.get("fixture_manifest.json:key:fixtures"),
            Some(&false)
        );
    }

    #[test]
    fn markdown_validation_requires_packet_id_and_content() {
        let mut errors = Vec::new();
        let mut checks = BTreeMap::new();
        let mut packet_files = BTreeMap::new();
        packet_files.insert(
            "legacy_anchor_map.md",
            Some("# FT-P2C-777\ncontent".to_string()),
        );
        validate_markdown(
            "FT-P2C-777",
            "legacy_anchor_map.md",
            &packet_files,
            &mut errors,
            &mut checks,
        );

        assert!(errors.is_empty());
        assert_eq!(checks.get("md_non_empty:legacy_anchor_map.md"), Some(&true));
    }
}

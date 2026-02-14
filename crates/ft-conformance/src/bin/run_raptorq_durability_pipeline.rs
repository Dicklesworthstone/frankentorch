#![forbid(unsafe_code)]

use std::fs;
use std::path::{Path, PathBuf};

use ft_serialize::generate_raptorq_sidecar;
use serde::Serialize;
use serde_json::json;

const MANIFEST_SCHEMA_VERSION: &str = "ft-raptorq-repair-manifest-v1";
const SCRUB_SCHEMA_VERSION: &str = "ft-raptorq-integrity-scrub-v1";
const DECODE_EVENTS_SCHEMA_VERSION: &str = "ft-raptorq-decode-events-v1";
const DEFAULT_REPAIR_SYMBOLS: usize = 4;

#[derive(Debug, Clone)]
struct DurableTarget {
    artifact_id: String,
    artifact_type: String,
    packet_id: Option<String>,
    source_path: PathBuf,
}

#[derive(Debug)]
struct TargetOutcome {
    manifest_entry: Option<RepairManifestEntry>,
    scrub_entry: ScrubEntry,
    decode_event: Option<DecodeEvent>,
}

#[derive(Debug, Serialize)]
struct RepairManifestReport {
    schema_version: &'static str,
    generated_unix_ms: u128,
    repair_symbols_requested: usize,
    summary: RepairManifestSummary,
    entries: Vec<RepairManifestEntry>,
}

#[derive(Debug, Serialize)]
struct RepairManifestSummary {
    total_targets: usize,
    sidecars_emitted: usize,
    failed_targets: usize,
}

#[derive(Debug, Serialize)]
struct RepairManifestEntry {
    artifact_id: String,
    artifact_type: String,
    packet_id: Option<String>,
    source_path: String,
    sidecar_path: String,
    decode_proof_path: String,
    source_hash: String,
    symbol_size: usize,
    source_symbol_count: usize,
    constraints_symbol_count: usize,
    repair_symbol_count: usize,
    repair_manifest_entries: usize,
    proof_hash_hex: String,
    proof_received_symbol_count: usize,
    recovered_bytes: usize,
}

#[derive(Debug, Serialize)]
struct ScrubReport {
    schema_version: &'static str,
    generated_unix_ms: u128,
    summary: ScrubSummary,
    entries: Vec<ScrubEntry>,
}

#[derive(Debug, Serialize)]
struct ScrubSummary {
    total_targets: usize,
    passed: usize,
    failed: usize,
}

#[derive(Debug, Serialize)]
struct ScrubEntry {
    artifact_id: String,
    artifact_type: String,
    packet_id: Option<String>,
    source_path: String,
    source_exists: bool,
    sidecar_path: String,
    decode_proof_path: String,
    sidecar_written: bool,
    decode_proof_written: bool,
    corruption_probe_detected: bool,
    status: String,
    message: String,
}

#[derive(Debug, Serialize)]
struct DecodeEventsReport {
    schema_version: &'static str,
    generated_unix_ms: u128,
    summary: DecodeEventsSummary,
    events: Vec<DecodeEvent>,
}

#[derive(Debug, Serialize)]
struct DecodeEventsSummary {
    total_events: usize,
    corruption_probe_passed: usize,
}

#[derive(Debug, Serialize)]
struct DecodeEvent {
    event_id: String,
    artifact_id: String,
    artifact_type: String,
    packet_id: Option<String>,
    source_hash: String,
    proof_hash_hex: String,
    proof_received_symbol_count: usize,
    recovered_bytes: usize,
    repair_symbol_count: usize,
    corruption_probe_detected: bool,
    sidecar_path: String,
    decode_proof_path: String,
    replay_command: String,
}

#[derive(Debug)]
struct Args {
    phase2c_root: PathBuf,
    repair_symbols: usize,
}

fn main() -> Result<(), String> {
    let args = parse_args()?;
    let now_unix_ms = now_unix_ms();

    let mut targets = default_global_targets(args.phase2c_root.as_path());
    targets.extend(discover_packet_targets(args.phase2c_root.as_path())?);
    targets.sort_by(|left, right| left.artifact_id.cmp(&right.artifact_id));

    let sidecar_root = args.phase2c_root.join("raptorq_sidecars");
    fs::create_dir_all(sidecar_root.as_path()).map_err(|error| {
        format!(
            "failed to create sidecar output dir {}: {error}",
            sidecar_root.display()
        )
    })?;

    let mut manifest_entries = Vec::new();
    let mut scrub_entries = Vec::new();
    let mut decode_events = Vec::new();

    for target in &targets {
        let outcome = process_target(
            target,
            sidecar_root.as_path(),
            now_unix_ms,
            args.repair_symbols,
        )?;
        if let Some(entry) = outcome.manifest_entry {
            manifest_entries.push(entry);
        }
        scrub_entries.push(outcome.scrub_entry);
        if let Some(event) = outcome.decode_event {
            decode_events.push(event);
        }
    }

    let failed_targets = scrub_entries
        .iter()
        .filter(|entry| entry.status != "ok")
        .count();
    let sidecars_emitted = manifest_entries.len();
    let scrub_passed = scrub_entries
        .iter()
        .filter(|entry| entry.status == "ok")
        .count();
    let scrub_failed = scrub_entries.len().saturating_sub(scrub_passed);
    let corruption_probe_passed = decode_events
        .iter()
        .filter(|event| event.corruption_probe_detected)
        .count();

    let manifest = RepairManifestReport {
        schema_version: MANIFEST_SCHEMA_VERSION,
        generated_unix_ms: now_unix_ms,
        repair_symbols_requested: args.repair_symbols,
        summary: RepairManifestSummary {
            total_targets: targets.len(),
            sidecars_emitted,
            failed_targets,
        },
        entries: manifest_entries,
    };

    let scrub_report = ScrubReport {
        schema_version: SCRUB_SCHEMA_VERSION,
        generated_unix_ms: now_unix_ms,
        summary: ScrubSummary {
            total_targets: targets.len(),
            passed: scrub_passed,
            failed: scrub_failed,
        },
        entries: scrub_entries,
    };

    let decode_events_report = DecodeEventsReport {
        schema_version: DECODE_EVENTS_SCHEMA_VERSION,
        generated_unix_ms: now_unix_ms,
        summary: DecodeEventsSummary {
            total_events: decode_events.len(),
            corruption_probe_passed,
        },
        events: decode_events,
    };

    let manifest_path = args
        .phase2c_root
        .join("RAPTORQ_REPAIR_SYMBOL_MANIFEST_V1.json");
    let scrub_path = args
        .phase2c_root
        .join("RAPTORQ_INTEGRITY_SCRUB_REPORT_V1.json");
    let decode_events_path = args
        .phase2c_root
        .join("RAPTORQ_DECODE_PROOF_EVENTS_V1.json");

    write_json_file(manifest_path.as_path(), &manifest)?;
    write_json_file(scrub_path.as_path(), &scrub_report)?;
    write_json_file(decode_events_path.as_path(), &decode_events_report)?;

    println!(
        "{}",
        serde_json::to_string_pretty(&json!({
            "status": if scrub_failed == 0 { "ok" } else { "fail" },
            "phase2c_root": args.phase2c_root.display().to_string(),
            "targets": targets.len(),
            "sidecars_emitted": sidecars_emitted,
            "scrub_failed": scrub_failed,
            "decode_events": decode_events_report.summary.total_events,
            "manifest_path": manifest_path.display().to_string(),
            "scrub_report_path": scrub_path.display().to_string(),
            "decode_events_path": decode_events_path.display().to_string(),
        }))
        .map_err(|error| format!("failed to serialize durability pipeline summary: {error}"))?
    );

    if scrub_failed > 0 {
        return Err(format!(
            "raptorq durability pipeline completed with {scrub_failed} failed target(s)"
        ));
    }

    Ok(())
}

fn process_target(
    target: &DurableTarget,
    sidecar_root: &Path,
    now_unix_ms: u128,
    repair_symbols: usize,
) -> Result<TargetOutcome, String> {
    let source_exists = target.source_path.exists();
    let (sidecar_path, decode_proof_path) = output_paths_for_target(target, sidecar_root);

    if !source_exists {
        return Ok(TargetOutcome {
            manifest_entry: None,
            scrub_entry: ScrubEntry {
                artifact_id: target.artifact_id.clone(),
                artifact_type: target.artifact_type.clone(),
                packet_id: target.packet_id.clone(),
                source_path: target.source_path.display().to_string(),
                source_exists: false,
                sidecar_path: sidecar_path.display().to_string(),
                decode_proof_path: decode_proof_path.display().to_string(),
                sidecar_written: false,
                decode_proof_written: false,
                corruption_probe_detected: false,
                status: "fail".to_string(),
                message: "source artifact missing".to_string(),
            },
            decode_event: None,
        });
    }

    let payload = fs::read_to_string(target.source_path.as_path()).map_err(|error| {
        format!(
            "failed to read source artifact {}: {error}",
            target.source_path.display()
        )
    })?;

    let (sidecar, proof) =
        generate_raptorq_sidecar(payload.as_str(), repair_symbols).map_err(|error| {
            format!(
                "failed to generate sidecar for {}: {error}",
                target.source_path.display()
            )
        })?;

    let tampered_payload = format!("{payload}\n#ft-raptorq-corruption-probe");
    let (tampered_sidecar, _) = generate_raptorq_sidecar(tampered_payload.as_str(), repair_symbols)
        .map_err(|error| {
            format!(
                "failed corruption probe sidecar generation for {}: {error}",
                target.source_path.display()
            )
        })?;
    let corruption_probe_detected = tampered_sidecar.source_hash != sidecar.source_hash;

    if let Some(parent) = sidecar_path.parent() {
        fs::create_dir_all(parent).map_err(|error| {
            format!(
                "failed to create sidecar parent dir {}: {error}",
                parent.display()
            )
        })?;
    }
    if let Some(parent) = decode_proof_path.parent() {
        fs::create_dir_all(parent).map_err(|error| {
            format!(
                "failed to create decode proof parent dir {}: {error}",
                parent.display()
            )
        })?;
    }

    let sidecar_json = json!({
        "artifact_id": target.artifact_id,
        "artifact_type": target.artifact_type,
        "packet_id": target.packet_id,
        "source_path": target.source_path.display().to_string(),
        "source_hash": sidecar.source_hash,
        "raptorq": {
            "schema_version": sidecar.schema_version,
            "symbol_size": sidecar.symbol_size,
            "k": sidecar.source_symbol_count,
            "constraints": sidecar.constraints_symbol_count,
            "repair_symbols": sidecar.repair_symbol_count,
            "seed": sidecar.seed,
            "object_id_high": sidecar.object_id_high,
            "object_id_low": sidecar.object_id_low,
            "repair_manifest": sidecar.repair_manifest,
        },
        "scrub": {
            "status": if corruption_probe_detected { "ok" } else { "fail" },
            "last_ok_unix_ms": now_unix_ms,
            "corruption_probe_detected": corruption_probe_detected,
        },
        "notes": "Generated via ft-serialize::generate_raptorq_sidecar",
    });

    let decode_json = json!({
        "artifact_id": target.artifact_id,
        "artifact_type": target.artifact_type,
        "packet_id": target.packet_id,
        "status": if corruption_probe_detected { "decode_verified" } else { "decode_warning" },
        "decode_proof": {
            "schema_version": proof.schema_version,
            "source_hash": proof.source_hash,
            "proof_hash": proof.proof_hash,
            "proof_hash_hex": proof.proof_hash_hex,
            "received_symbol_count": proof.received_symbol_count,
            "recovered_bytes": proof.recovered_bytes,
        },
        "corruption_probe": {
            "status": if corruption_probe_detected { "hash_mismatch_detected" } else { "hash_mismatch_not_detected" },
            "tampered_source_hash": tampered_sidecar.source_hash,
        },
        "notes": "Proof generated by asupersync inactivation decoder",
    });

    fs::write(
        sidecar_path.as_path(),
        serde_json::to_string_pretty(&sidecar_json)
            .map_err(|error| format!("failed to serialize sidecar json: {error}"))?,
    )
    .map_err(|error| {
        format!(
            "failed to write sidecar file {}: {error}",
            sidecar_path.display()
        )
    })?;

    fs::write(
        decode_proof_path.as_path(),
        serde_json::to_string_pretty(&decode_json)
            .map_err(|error| format!("failed to serialize decode proof json: {error}"))?,
    )
    .map_err(|error| {
        format!(
            "failed to write decode proof file {}: {error}",
            decode_proof_path.display()
        )
    })?;

    let manifest_entry = RepairManifestEntry {
        artifact_id: target.artifact_id.clone(),
        artifact_type: target.artifact_type.clone(),
        packet_id: target.packet_id.clone(),
        source_path: target.source_path.display().to_string(),
        sidecar_path: sidecar_path.display().to_string(),
        decode_proof_path: decode_proof_path.display().to_string(),
        source_hash: sidecar.source_hash.clone(),
        symbol_size: sidecar.symbol_size,
        source_symbol_count: sidecar.source_symbol_count,
        constraints_symbol_count: sidecar.constraints_symbol_count,
        repair_symbol_count: sidecar.repair_symbol_count,
        repair_manifest_entries: sidecar.repair_manifest.len(),
        proof_hash_hex: proof.proof_hash_hex.clone(),
        proof_received_symbol_count: proof.received_symbol_count,
        recovered_bytes: proof.recovered_bytes,
    };

    let scrub_entry = ScrubEntry {
        artifact_id: target.artifact_id.clone(),
        artifact_type: target.artifact_type.clone(),
        packet_id: target.packet_id.clone(),
        source_path: target.source_path.display().to_string(),
        source_exists: true,
        sidecar_path: sidecar_path.display().to_string(),
        decode_proof_path: decode_proof_path.display().to_string(),
        sidecar_written: true,
        decode_proof_written: true,
        corruption_probe_detected,
        status: if corruption_probe_detected {
            "ok".to_string()
        } else {
            "fail".to_string()
        },
        message: if corruption_probe_detected {
            "scrub and corruption probe passed".to_string()
        } else {
            "corruption probe failed to detect hash mismatch".to_string()
        },
    };

    let decode_event = DecodeEvent {
        event_id: format!(
            "rqevt64:{:016x}",
            det_hash64(
                [
                    target.artifact_id.as_str(),
                    sidecar.source_hash.as_str(),
                    proof.proof_hash_hex.as_str(),
                ]
                .as_slice()
            )
        ),
        artifact_id: target.artifact_id.clone(),
        artifact_type: target.artifact_type.clone(),
        packet_id: target.packet_id.clone(),
        source_hash: sidecar.source_hash,
        proof_hash_hex: proof.proof_hash_hex,
        proof_received_symbol_count: proof.received_symbol_count,
        recovered_bytes: proof.recovered_bytes,
        repair_symbol_count: sidecar.repair_symbol_count,
        corruption_probe_detected,
        sidecar_path: sidecar_path.display().to_string(),
        decode_proof_path: decode_proof_path.display().to_string(),
        replay_command: "cargo run -p ft-conformance --bin run_raptorq_durability_pipeline"
            .to_string(),
    };

    Ok(TargetOutcome {
        manifest_entry: Some(manifest_entry),
        scrub_entry,
        decode_event: Some(decode_event),
    })
}

fn output_paths_for_target(target: &DurableTarget, sidecar_root: &Path) -> (PathBuf, PathBuf) {
    if target.packet_id.is_some()
        && target
            .source_path
            .file_name()
            .is_some_and(|name| name == "parity_report.json")
    {
        let packet_dir = target
            .source_path
            .parent()
            .map(Path::to_path_buf)
            .unwrap_or_else(|| sidecar_root.to_path_buf());
        (
            packet_dir.join("parity_report.raptorq.json"),
            packet_dir.join("parity_report.decode_proof.json"),
        )
    } else {
        (
            sidecar_root.join(format!("{}.raptorq.json", target.artifact_id)),
            sidecar_root.join(format!("{}.decode_proof.json", target.artifact_id)),
        )
    }
}

fn default_global_targets(phase2c_root: &Path) -> Vec<DurableTarget> {
    vec![
        DurableTarget {
            artifact_id: "phase2c-conformance-differential-report-v1".to_string(),
            artifact_type: "conformance_fixture_bundle".to_string(),
            packet_id: None,
            source_path: phase2c_root.join("conformance/differential_report_v1.json"),
        },
        DurableTarget {
            artifact_id: "phase2c-benchmark-baseline-validator-optimization-evidence".to_string(),
            artifact_type: "benchmark_baseline_bundle".to_string(),
            packet_id: None,
            source_path: phase2c_root.join("VALIDATOR_OPTIMIZATION_EVIDENCE_20260213.md"),
        },
        DurableTarget {
            artifact_id: "phase2c-migration-manifest-schema-lock-v1".to_string(),
            artifact_type: "migration_manifest".to_string(),
            packet_id: None,
            source_path: phase2c_root.join("SCHEMA_LOCK_V1.md"),
        },
        DurableTarget {
            artifact_id: "phase2c-reproducibility-ledger-v1".to_string(),
            artifact_type: "reproducibility_ledger".to_string(),
            packet_id: None,
            source_path: phase2c_root.join("ESSENCE_EXTRACTION_LEDGER_V1.md"),
        },
        DurableTarget {
            artifact_id: "phase2c-long-lived-state-snapshot-e2e-matrix-full-v1".to_string(),
            artifact_type: "long_lived_state_snapshot".to_string(),
            packet_id: None,
            source_path: phase2c_root.join("e2e_forensics/e2e_matrix_full_v1.jsonl"),
        },
    ]
}

fn discover_packet_targets(phase2c_root: &Path) -> Result<Vec<DurableTarget>, String> {
    let mut targets = Vec::new();
    let dir_entries = fs::read_dir(phase2c_root).map_err(|error| {
        format!(
            "failed to scan phase2c root {}: {error}",
            phase2c_root.display()
        )
    })?;

    for dir_entry in dir_entries {
        let dir_entry =
            dir_entry.map_err(|error| format!("failed to read phase2c dir entry: {error}"))?;
        let path = dir_entry.path();
        if !path.is_dir() {
            continue;
        }
        let packet_name = dir_entry.file_name();
        let packet_name = packet_name.to_string_lossy();
        if !packet_name.starts_with("FT-P2C-") {
            continue;
        }

        targets.push(DurableTarget {
            artifact_id: format!("{packet_name}-parity-report"),
            artifact_type: "packet_parity_report".to_string(),
            packet_id: Some(packet_name.to_string()),
            source_path: path.join("parity_report.json"),
        });
    }

    Ok(targets)
}

fn write_json_file(path: &Path, value: &impl Serialize) -> Result<(), String> {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent).map_err(|error| {
            format!(
                "failed to create parent dir for {}: {error}",
                path.display()
            )
        })?;
    }

    let content = serde_json::to_string_pretty(value)
        .map_err(|error| format!("failed to serialize {}: {error}", path.display()))?;
    fs::write(path, content).map_err(|error| format!("failed to write {}: {error}", path.display()))
}

fn parse_args() -> Result<Args, String> {
    let default_root = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../..")
        .join("artifacts/phase2c");
    let mut phase2c_root = default_root;
    let mut repair_symbols = DEFAULT_REPAIR_SYMBOLS;

    let mut args = std::env::args().skip(1);
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--phase2c-root" => {
                let value = args
                    .next()
                    .ok_or_else(|| "--phase2c-root requires a directory path".to_string())?;
                phase2c_root = PathBuf::from(value);
            }
            "--repair-symbols" => {
                let value = args
                    .next()
                    .ok_or_else(|| "--repair-symbols requires an integer > 0".to_string())?;
                repair_symbols = value.parse::<usize>().map_err(|error| {
                    format!("invalid --repair-symbols value '{value}': {error}")
                })?;
                if repair_symbols == 0 {
                    return Err("--repair-symbols must be > 0".to_string());
                }
            }
            other => {
                return Err(format!(
                    "unknown arg '{other}'. usage: run_raptorq_durability_pipeline [--phase2c-root <path>] [--repair-symbols <n>]"
                ));
            }
        }
    }

    Ok(Args {
        phase2c_root,
        repair_symbols,
    })
}

fn now_unix_ms() -> u128 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map_or(0, |duration| duration.as_millis())
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

#[cfg(test)]
mod tests {
    use super::{DurableTarget, det_hash64, output_paths_for_target};
    use std::path::Path;

    #[test]
    fn det_hash64_is_stable() {
        let a = det_hash64(["x", "y", "z"].as_slice());
        let b = det_hash64(["x", "y", "z"].as_slice());
        assert_eq!(a, b);
    }

    #[test]
    fn packet_targets_write_into_packet_dir() {
        let target = DurableTarget {
            artifact_id: "ft-p2c-001-parity-report".to_string(),
            artifact_type: "packet_parity_report".to_string(),
            packet_id: Some("FT-P2C-001".to_string()),
            source_path: Path::new("artifacts/phase2c/FT-P2C-001/parity_report.json").to_path_buf(),
        };

        let (sidecar, proof) =
            output_paths_for_target(&target, Path::new("artifacts/phase2c/raptorq_sidecars"));
        assert!(sidecar.ends_with("FT-P2C-001/parity_report.raptorq.json"));
        assert!(proof.ends_with("FT-P2C-001/parity_report.decode_proof.json"));
    }
}

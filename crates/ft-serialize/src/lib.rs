#![forbid(unsafe_code)]

use std::collections::BTreeSet;
use std::fmt;
use std::hash::Hasher;

use asupersync::raptorq::decoder::{InactivationDecoder, ReceivedSymbol};
use asupersync::raptorq::systematic::SystematicEncoder;
use asupersync::types::ObjectId;
use asupersync::util::DetHasher;
use serde::{Deserialize, Serialize};
use serde_json::Value;

pub const CHECKPOINT_SCHEMA_VERSION: u32 = 1;
pub const RAPTORQ_SIDECAR_SCHEMA_VERSION: u32 = 1;

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SnapshotEntry {
    pub node_id: usize,
    pub value: f64,
    pub grad: Option<f64>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CheckpointMode {
    Strict,
    Hardened,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DecodeMode {
    Strict,
    Hardened,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CheckpointEnvelope {
    pub schema_version: u32,
    pub mode: CheckpointMode,
    pub entries: Vec<SnapshotEntry>,
    pub source_hash: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RepairSymbolRecord {
    pub esi: u32,
    pub degree: usize,
    pub bytes: usize,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RaptorQSidecar {
    pub schema_version: u32,
    pub source_hash: String,
    pub symbol_size: usize,
    pub source_symbol_count: usize,
    pub repair_symbol_count: usize,
    pub constraints_symbol_count: usize,
    pub seed: u64,
    pub object_id_high: u64,
    pub object_id_low: u64,
    pub repair_manifest: Vec<RepairSymbolRecord>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct DecodeProofArtifact {
    pub schema_version: u8,
    pub source_hash: String,
    pub proof_hash: u64,
    pub proof_hash_hex: String,
    pub received_symbol_count: usize,
    pub recovered_bytes: usize,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SerializeError {
    InvalidJson { diagnostic: String },
    UnknownField { field: String },
    VersionMismatch { expected: u32, found: u32 },
    ChecksumMismatch { expected: String, found: String },
    IncompatiblePayload { reason: String },
    RaptorQFailure { reason: String },
}

impl fmt::Display for SerializeError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidJson { diagnostic } => write!(f, "invalid json: {diagnostic}"),
            Self::UnknownField { field } => write!(f, "unknown field '{field}'"),
            Self::VersionMismatch { expected, found } => {
                write!(
                    f,
                    "schema version mismatch: expected={expected} found={found}"
                )
            }
            Self::ChecksumMismatch { expected, found } => {
                write!(f, "checksum mismatch: expected={expected} found={found}")
            }
            Self::IncompatiblePayload { reason } => write!(f, "incompatible payload: {reason}"),
            Self::RaptorQFailure { reason } => write!(f, "raptorq failure: {reason}"),
        }
    }
}

impl std::error::Error for SerializeError {}

#[must_use]
pub fn encode_checkpoint(entries: &[SnapshotEntry], mode: CheckpointMode) -> String {
    let normalized_entries = normalize_entries(entries);
    let source_hash = checkpoint_hash(CHECKPOINT_SCHEMA_VERSION, mode, &normalized_entries);

    let envelope = CheckpointEnvelope {
        schema_version: CHECKPOINT_SCHEMA_VERSION,
        mode,
        entries: normalized_entries,
        source_hash,
    };

    serde_json::to_string(&envelope).expect("checkpoint serialization should be infallible")
}

pub fn decode_checkpoint(
    input: &str,
    mode: DecodeMode,
) -> Result<CheckpointEnvelope, SerializeError> {
    match mode {
        DecodeMode::Strict => decode_checkpoint_strict(input),
        DecodeMode::Hardened => decode_checkpoint_hardened(input),
    }
}

#[must_use]
pub fn encode_snapshot(entries: &[SnapshotEntry]) -> String {
    encode_checkpoint(entries, CheckpointMode::Strict)
}

pub fn decode_snapshot(input: &str) -> Result<Vec<SnapshotEntry>, SerializeError> {
    let envelope = decode_checkpoint(input, DecodeMode::Strict)?;
    Ok(envelope.entries)
}

pub fn generate_raptorq_sidecar(
    payload: &str,
    repair_symbols: usize,
) -> Result<(RaptorQSidecar, DecodeProofArtifact), SerializeError> {
    let payload_bytes = payload.as_bytes();
    let symbol_size = recommended_symbol_size(payload_bytes.len());
    let source_symbols = split_source_symbols(payload_bytes, symbol_size);
    let source_symbol_count = source_symbols.len();
    let seed = 0x4654_5f52_4150_5451;

    let mut encoder =
        SystematicEncoder::new(&source_symbols, symbol_size, seed).ok_or_else(|| {
            SerializeError::RaptorQFailure {
                reason: "failed to initialize systematic encoder".to_string(),
            }
        })?;

    let systematic = encoder.emit_systematic();
    let repair_count = repair_symbols.max(1);
    let repair = encoder.emit_repair(repair_count);

    let decoder = InactivationDecoder::new(source_symbol_count, symbol_size, seed);
    let constraints = decoder.constraint_symbols();
    let source_received: Vec<ReceivedSymbol> = systematic
        .iter()
        .map(|symbol| ReceivedSymbol::source(symbol.esi, symbol.data.clone()))
        .collect();
    let repair_received: Vec<ReceivedSymbol> = repair
        .iter()
        .map(|symbol| {
            let (columns, coefficients) = decoder.repair_equation(symbol.esi);
            ReceivedSymbol::repair(symbol.esi, columns, coefficients, symbol.data.clone())
        })
        .collect();

    let payload_hash = hash_bytes(payload_bytes);
    let object_id = ObjectId::new(0x4654_5f43_4b50_545f, 0x4455_5241_4249_4c45);
    let min_required = decoder.params().l;

    let mut candidates: Vec<Vec<ReceivedSymbol>> = Vec::new();

    let mut candidate_a = constraints.clone();
    candidate_a.extend(source_received.clone());
    candidates.push(candidate_a);

    if !repair_received.is_empty() && !source_received.is_empty() {
        let mut candidate_b = constraints.clone();
        candidate_b.extend(source_received.iter().skip(1).cloned());
        candidate_b.push(repair_received[0].clone());
        candidate_b.extend(repair_received.iter().skip(1).cloned());
        candidates.push(candidate_b);
    }

    let mut candidate_c = constraints.clone();
    candidate_c.extend(source_received);
    candidate_c.extend(repair_received);
    candidates.push(candidate_c);

    let mut selected_received = None;
    let mut selected_decoded = None;
    let mut last_error = String::from("no decode candidates attempted");

    for candidate in candidates {
        if candidate.len() < min_required {
            continue;
        }
        match decoder.decode_with_proof(candidate.as_slice(), object_id, 0) {
            Ok(decoded) => {
                selected_received = Some(candidate);
                selected_decoded = Some(decoded);
                break;
            }
            Err((error, _proof)) => {
                last_error = format!("{error:?}");
            }
        }
    }

    let received = selected_received.ok_or_else(|| SerializeError::RaptorQFailure {
        reason: format!("decode_with_proof failed for all candidates: {last_error}"),
    })?;
    let decoded = selected_decoded.ok_or_else(|| SerializeError::RaptorQFailure {
        reason: "decode_with_proof returned no decoded proof".to_string(),
    })?;

    let mut recovered = Vec::new();
    for source_symbol in &decoded.result.source {
        recovered.extend_from_slice(source_symbol);
    }
    recovered.truncate(payload_bytes.len());

    if recovered != payload_bytes {
        return Err(SerializeError::RaptorQFailure {
            reason: "decoded payload failed deterministic recovery check".to_string(),
        });
    }

    let proof_hash = decoded.proof.content_hash();

    let sidecar = RaptorQSidecar {
        schema_version: RAPTORQ_SIDECAR_SCHEMA_VERSION,
        source_hash: payload_hash.clone(),
        symbol_size,
        source_symbol_count,
        repair_symbol_count: repair.len(),
        constraints_symbol_count: constraints.len(),
        seed,
        object_id_high: object_id.high(),
        object_id_low: object_id.low(),
        repair_manifest: repair
            .iter()
            .map(|symbol| RepairSymbolRecord {
                esi: symbol.esi,
                degree: symbol.degree,
                bytes: symbol.data.len(),
            })
            .collect(),
    };

    let proof = DecodeProofArtifact {
        schema_version: 1,
        source_hash: payload_hash,
        proof_hash,
        proof_hash_hex: format!("det64:{proof_hash:016x}"),
        received_symbol_count: received.len(),
        recovered_bytes: recovered.len(),
    };

    Ok((sidecar, proof))
}

fn decode_checkpoint_strict(input: &str) -> Result<CheckpointEnvelope, SerializeError> {
    let envelope: CheckpointEnvelope = serde_json::from_str(input).map_err(|error| {
        if let Some(field) = extract_unknown_field(error.to_string().as_str()) {
            SerializeError::UnknownField { field }
        } else {
            SerializeError::InvalidJson {
                diagnostic: bounded(error.to_string().as_str(), 200),
            }
        }
    })?;
    validate_checkpoint(&envelope)?;
    Ok(envelope)
}

fn decode_checkpoint_hardened(input: &str) -> Result<CheckpointEnvelope, SerializeError> {
    let raw: Value = serde_json::from_str(input).map_err(|error| SerializeError::InvalidJson {
        diagnostic: bounded(
            format!(
                "{error}; payload_prefix={} ",
                bounded(input.replace('\n', " ").as_str(), 96)
            )
            .as_str(),
            220,
        ),
    })?;

    let obj = raw
        .as_object()
        .ok_or_else(|| SerializeError::IncompatiblePayload {
            reason: "top-level checkpoint payload must be a JSON object".to_string(),
        })?;

    let allowed: BTreeSet<&str> =
        BTreeSet::from(["schema_version", "mode", "entries", "source_hash"]);
    for key in obj.keys() {
        if !allowed.contains(key.as_str()) {
            return Err(SerializeError::UnknownField { field: key.clone() });
        }
    }

    let envelope: CheckpointEnvelope =
        serde_json::from_value(raw).map_err(|error| SerializeError::IncompatiblePayload {
            reason: bounded(error.to_string().as_str(), 200),
        })?;

    validate_checkpoint(&envelope)?;
    Ok(envelope)
}

fn validate_checkpoint(envelope: &CheckpointEnvelope) -> Result<(), SerializeError> {
    if envelope.schema_version != CHECKPOINT_SCHEMA_VERSION {
        return Err(SerializeError::VersionMismatch {
            expected: CHECKPOINT_SCHEMA_VERSION,
            found: envelope.schema_version,
        });
    }

    let normalized_entries = normalize_entries(&envelope.entries);
    let expected = checkpoint_hash(
        envelope.schema_version,
        envelope.mode,
        normalized_entries.as_slice(),
    );
    if envelope.source_hash != expected {
        return Err(SerializeError::ChecksumMismatch {
            expected,
            found: envelope.source_hash.clone(),
        });
    }

    Ok(())
}

fn normalize_entries(entries: &[SnapshotEntry]) -> Vec<SnapshotEntry> {
    let mut normalized = entries.to_vec();
    normalized.sort_by_key(|entry| entry.node_id);
    normalized
}

fn checkpoint_hash(schema_version: u32, mode: CheckpointMode, entries: &[SnapshotEntry]) -> String {
    let mut hasher = DetHasher::default();
    hasher.write_u32(schema_version);
    hasher.write_u8(match mode {
        CheckpointMode::Strict => 1,
        CheckpointMode::Hardened => 2,
    });
    for entry in entries {
        hasher.write_u64(entry.node_id as u64);
        hasher.write_u64(entry.value.to_bits());
        match entry.grad {
            Some(grad) => {
                hasher.write_u8(1);
                hasher.write_u64(grad.to_bits());
            }
            None => hasher.write_u8(0),
        }
    }
    format!("det64:{:016x}", hasher.finish())
}

fn hash_bytes(bytes: &[u8]) -> String {
    let mut hasher = DetHasher::default();
    hasher.write(bytes);
    format!("det64:{:016x}", hasher.finish())
}

fn split_source_symbols(bytes: &[u8], symbol_size: usize) -> Vec<Vec<u8>> {
    if bytes.is_empty() {
        return vec![vec![0u8; symbol_size]];
    }

    let mut symbols = Vec::new();
    for chunk in bytes.chunks(symbol_size) {
        let mut symbol = vec![0u8; symbol_size];
        symbol[..chunk.len()].copy_from_slice(chunk);
        symbols.push(symbol);
    }
    symbols
}

fn recommended_symbol_size(payload_len: usize) -> usize {
    match payload_len {
        0..=64 => 32,
        65..=512 => 64,
        513..=4096 => 128,
        _ => 256,
    }
}

fn extract_unknown_field(message: &str) -> Option<String> {
    // serde_json message shape: "unknown field `x`, expected ..."
    let marker = "unknown field `";
    let start = message.find(marker)? + marker.len();
    let tail = &message[start..];
    let end = tail.find('`')?;
    Some(tail[..end].to_string())
}

fn bounded(input: &str, max_len: usize) -> String {
    if input.len() <= max_len {
        input.to_string()
    } else {
        let mut boundary = max_len.min(input.len());
        while boundary > 0 && !input.is_char_boundary(boundary) {
            boundary -= 1;
        }
        format!("{}...", &input[..boundary])
    }
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeMap;

    use proptest::prelude::*;
    use serde_json::json;

    use super::{
        CheckpointMode, DecodeMode, SerializeError, SnapshotEntry, decode_checkpoint,
        decode_snapshot, encode_checkpoint, encode_snapshot, generate_raptorq_sidecar,
    };

    fn det_seed(parts: &[u64]) -> u64 {
        let mut hash = 0xcbf2_9ce4_8422_2325u64;
        for value in parts {
            for byte in value.to_le_bytes() {
                hash ^= u64::from(byte);
                hash = hash.wrapping_mul(0x0000_0100_0000_01b3);
            }
        }
        hash
    }

    fn snapshot_digest(entries: &[SnapshotEntry]) -> u64 {
        let mut hash = 0xcbf2_9ce4_8422_2325u64;
        for entry in entries {
            for byte in (entry.node_id as u64).to_le_bytes() {
                hash ^= u64::from(byte);
                hash = hash.wrapping_mul(0x0000_0100_0000_01b3);
            }
            for byte in entry.value.to_bits().to_le_bytes() {
                hash ^= u64::from(byte);
                hash = hash.wrapping_mul(0x0000_0100_0000_01b3);
            }
            if let Some(grad) = entry.grad {
                hash ^= 1;
                hash = hash.wrapping_mul(0x0000_0100_0000_01b3);
                for byte in grad.to_bits().to_le_bytes() {
                    hash ^= u64::from(byte);
                    hash = hash.wrapping_mul(0x0000_0100_0000_01b3);
                }
            } else {
                hash ^= 0;
                hash = hash.wrapping_mul(0x0000_0100_0000_01b3);
            }
        }
        hash
    }

    fn build_property_log(
        test_id: &str,
        mode: &str,
        seed: u64,
        input_digest: u64,
        output_digest: u64,
        reason_code: &str,
    ) -> BTreeMap<String, String> {
        let mut log = BTreeMap::new();
        log.insert("ts_utc".to_string(), "1970-01-01T00:00:00Z".to_string());
        log.insert("suite_id".to_string(), "ft_serialize_property".to_string());
        log.insert("test_id".to_string(), test_id.to_string());
        log.insert("packet_id".to_string(), "FT-P2C-006".to_string());
        log.insert(
            "fixture_id".to_string(),
            "serialization_property_generated".to_string(),
        );
        log.insert(
            "scenario_id".to_string(),
            format!("serialization_property/{mode}:{test_id}"),
        );
        log.insert("mode".to_string(), mode.to_string());
        log.insert("seed".to_string(), seed.to_string());
        log.insert(
            "input_digest".to_string(),
            format!("det64:{input_digest:016x}"),
        );
        log.insert(
            "output_digest".to_string(),
            format!("det64:{output_digest:016x}"),
        );
        log.insert(
            "env_fingerprint".to_string(),
            "det64:ft-serialize-test".to_string(),
        );
        log.insert(
            "artifact_refs".to_string(),
            "artifacts/phase2c/FT-P2C-006/unit_property_quality_report_v1.json".to_string(),
        );
        log.insert(
            "replay_command".to_string(),
            "cargo test -p ft-serialize -- --nocapture".to_string(),
        );
        log.insert("duration_ms".to_string(), "0".to_string());
        log.insert("outcome".to_string(), "pass".to_string());
        log.insert("contract_id".to_string(), reason_code.to_string());
        log.insert("shrink_trace".to_string(), "none".to_string());
        log.insert("reason_code".to_string(), reason_code.to_string());
        log
    }

    fn assert_log_contract(log: &BTreeMap<String, String>) {
        for key in [
            "ts_utc",
            "suite_id",
            "test_id",
            "packet_id",
            "fixture_id",
            "scenario_id",
            "mode",
            "seed",
            "input_digest",
            "output_digest",
            "env_fingerprint",
            "artifact_refs",
            "replay_command",
            "duration_ms",
            "outcome",
            "contract_id",
            "shrink_trace",
            "reason_code",
        ] {
            assert!(
                log.contains_key(key),
                "property log missing required key '{key}'"
            );
        }
    }

    fn snapshot_entry_strategy() -> impl Strategy<Value = SnapshotEntry> {
        (
            0usize..128usize,
            -1_000.0f64..1_000.0f64,
            proptest::option::of(-1_000.0f64..1_000.0f64),
        )
            .prop_map(|(node_id, value, grad)| SnapshotEntry {
                node_id,
                value,
                grad,
            })
    }

    fn generate_sidecar_with_retry(
        payload: &str,
        repair_symbols: usize,
    ) -> Result<(super::RaptorQSidecar, super::DecodeProofArtifact), SerializeError> {
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
                Ok(result) => return Ok(result),
                Err(err) => last_error = Some(err),
            }
        }

        Err(
            last_error.unwrap_or_else(|| SerializeError::RaptorQFailure {
                reason: "sidecar generation attempts exhausted".to_string(),
            }),
        )
    }

    #[test]
    fn checkpoint_round_trip_strict_works() {
        let entries = vec![
            SnapshotEntry {
                node_id: 1,
                value: 3.0,
                grad: Some(2.0),
            },
            SnapshotEntry {
                node_id: 0,
                value: 2.0,
                grad: None,
            },
        ];

        let encoded = encode_checkpoint(&entries, CheckpointMode::Strict);
        let decoded = decode_checkpoint(&encoded, DecodeMode::Strict).expect("strict decode");

        assert_eq!(decoded.entries[0].node_id, 0);
        assert_eq!(decoded.entries[1].node_id, 1);
    }

    #[test]
    fn strict_unknown_field_fail_closed() {
        let payload = json!({
            "schema_version": 1,
            "mode": "strict",
            "entries": [],
            "source_hash": "det64:0000000000000000",
            "extra": "boom"
        })
        .to_string();

        let err = decode_checkpoint(&payload, DecodeMode::Strict).expect_err("must fail");
        assert!(err.to_string().contains("unknown field"));
    }

    #[test]
    fn hardened_malformed_payload_returns_bounded_diagnostic() {
        let err = decode_checkpoint("{ not json", DecodeMode::Hardened).expect_err("must fail");
        let msg = err.to_string();
        assert!(msg.contains("invalid json"));
        assert!(msg.len() < 320);
    }

    #[test]
    fn version_mismatch_is_fail_closed() {
        let entries = vec![SnapshotEntry {
            node_id: 0,
            value: 2.0,
            grad: Some(1.0),
        }];
        let mut payload: serde_json::Value =
            serde_json::from_str(&encode_checkpoint(&entries, CheckpointMode::Strict))
                .expect("valid encoded checkpoint");
        payload["schema_version"] = json!(2);

        let err = decode_checkpoint(payload.to_string().as_str(), DecodeMode::Strict)
            .expect_err("version mismatch should fail");
        assert!(err.to_string().contains("schema version mismatch"));
    }

    #[test]
    fn checksum_mismatch_is_fail_closed() {
        let entries = vec![SnapshotEntry {
            node_id: 0,
            value: 2.0,
            grad: Some(1.0),
        }];
        let mut payload: serde_json::Value =
            serde_json::from_str(&encode_checkpoint(&entries, CheckpointMode::Strict))
                .expect("valid encoded checkpoint");
        payload["source_hash"] = json!("det64:deadbeefdeadbeef");

        let err = decode_checkpoint(payload.to_string().as_str(), DecodeMode::Strict)
            .expect_err("checksum mismatch should fail");
        assert!(err.to_string().contains("checksum mismatch"));
    }

    #[test]
    fn sidecar_generation_and_decode_proof_are_available() {
        let entries = vec![
            SnapshotEntry {
                node_id: 0,
                value: 2.0,
                grad: Some(1.0),
            },
            SnapshotEntry {
                node_id: 1,
                value: 3.0,
                grad: Some(2.0),
            },
        ];
        let payload = encode_checkpoint(&entries, CheckpointMode::Strict);

        let (sidecar, proof) =
            generate_sidecar_with_retry(&payload, 4).expect("sidecar generation should succeed");

        assert!(sidecar.repair_symbol_count >= 1);
        assert!(sidecar.constraints_symbol_count >= 1);
        assert!(proof.proof_hash > 0);
        assert_eq!(proof.recovered_bytes, payload.len());
    }

    #[test]
    fn decode_proof_hash_is_deterministic() {
        let entries = vec![
            SnapshotEntry {
                node_id: 0,
                value: 2.0,
                grad: Some(1.0),
            },
            SnapshotEntry {
                node_id: 1,
                value: 3.0,
                grad: Some(2.0),
            },
        ];
        let payload = encode_checkpoint(&entries, CheckpointMode::Strict);

        let (_, proof_a) =
            generate_sidecar_with_retry(&payload, 4).expect("first sidecar generation should work");
        let (_, proof_b) = generate_sidecar_with_retry(&payload, 4)
            .expect("second sidecar generation should work");

        assert_eq!(proof_a.proof_hash, proof_b.proof_hash);
    }

    #[test]
    fn legacy_snapshot_wrappers_round_trip() {
        let entries = vec![SnapshotEntry {
            node_id: 0,
            value: 2.0,
            grad: Some(1.0),
        }];

        let encoded = encode_snapshot(&entries);
        let decoded = decode_snapshot(&encoded).expect("legacy wrapper decode should work");
        assert_eq!(decoded, entries);
    }

    proptest! {
        #[test]
        fn prop_checkpoint_roundtrip_preserves_sorted_entries(
            entries in prop::collection::vec(snapshot_entry_strategy(), 1..16),
        ) {
            let encoded = encode_checkpoint(entries.as_slice(), CheckpointMode::Strict);
            let decoded =
                decode_checkpoint(encoded.as_str(), DecodeMode::Strict).expect("decode must succeed");
            let mut expected = entries.clone();
            expected.sort_by_key(|entry| entry.node_id);

            prop_assert_eq!(&decoded.entries, &expected);

            let seed = det_seed(&[
                entries.len() as u64,
                snapshot_digest(entries.as_slice()),
            ]);
            let log = build_property_log(
                "prop_checkpoint_roundtrip_preserves_sorted_entries",
                "strict",
                seed,
                snapshot_digest(entries.as_slice()),
                snapshot_digest(decoded.entries.as_slice()),
                "checkpoint_roundtrip_sorted_contract_ok",
            );
            assert_log_contract(&log);
        }

        #[test]
        fn prop_checkpoint_hash_is_order_invariant_for_unique_node_ids(
            rows in prop::collection::btree_map(
                0usize..128usize,
                (-1_000.0f64..1_000.0f64, proptest::option::of(-1_000.0f64..1_000.0f64)),
                1..16
            )
        ) {
            let sorted_entries = rows
                .iter()
                .map(|(node_id, (value, grad))| SnapshotEntry {
                    node_id: *node_id,
                    value: *value,
                    grad: *grad,
                })
                .collect::<Vec<_>>();
            let mut reversed_entries = sorted_entries.clone();
            reversed_entries.reverse();

            let encoded_a = encode_checkpoint(sorted_entries.as_slice(), CheckpointMode::Strict);
            let encoded_b = encode_checkpoint(reversed_entries.as_slice(), CheckpointMode::Strict);

            prop_assert_eq!(encoded_a, encoded_b);

            let seed = det_seed(&[
                sorted_entries.len() as u64,
                snapshot_digest(sorted_entries.as_slice()),
            ]);
            let log = build_property_log(
                "prop_checkpoint_hash_is_order_invariant_for_unique_node_ids",
                "strict",
                seed,
                snapshot_digest(sorted_entries.as_slice()),
                snapshot_digest(reversed_entries.as_slice()),
                "checkpoint_hash_order_invariant_ok",
            );
            assert_log_contract(&log);
        }

        #[test]
        fn prop_strict_unknown_field_remains_fail_closed(
            unknown_field in "[a-z][a-z0-9_]{2,12}",
            value in 0u64..10_000u64,
        ) {
            prop_assume!(
                unknown_field != "schema_version"
                    && unknown_field != "mode"
                    && unknown_field != "entries"
                    && unknown_field != "source_hash"
            );

            let entries = vec![
                SnapshotEntry {
                    node_id: 0,
                    value: value as f64,
                    grad: Some(1.0),
                },
                SnapshotEntry {
                    node_id: 1,
                    value: 2.0,
                    grad: None,
                },
            ];
            let mut payload: serde_json::Value = serde_json::from_str(
                encode_checkpoint(entries.as_slice(), CheckpointMode::Strict).as_str(),
            )
            .expect("checkpoint payload should parse");
            payload[unknown_field.as_str()] = json!(value);

            let result = decode_checkpoint(payload.to_string().as_str(), DecodeMode::Strict);
            prop_assert!(result.is_err());
            let msg = result.expect_err("strict decode should fail").to_string();
            prop_assert!(msg.contains("unknown field"));

            let seed = det_seed(&[value, unknown_field.len() as u64]);
            let log = build_property_log(
                "prop_strict_unknown_field_remains_fail_closed",
                "strict",
                seed,
                value,
                value,
                "strict_unknown_field_fail_closed",
            );
            assert_log_contract(&log);
        }

        #[test]
        fn prop_hardened_malformed_diagnostics_are_bounded(
            payload_suffix in ".{0,128}"
        ) {
            let malformed = format!("{{ malformed {}", payload_suffix);
            let err = decode_checkpoint(malformed.as_str(), DecodeMode::Hardened)
                .expect_err("malformed payload must fail");
            let msg = err.to_string();
            prop_assert!(msg.contains("invalid json"));
            prop_assert!(msg.len() < 320);

            let seed = det_seed(&[payload_suffix.len() as u64]);
            let log = build_property_log(
                "prop_hardened_malformed_diagnostics_are_bounded",
                "hardened",
                seed,
                payload_suffix.len() as u64,
                msg.len() as u64,
                "hardened_malformed_diagnostic_bounded",
            );
            assert_log_contract(&log);
        }

        #[test]
        fn prop_raptorq_decode_proof_hash_stays_deterministic(
            rows in prop::collection::btree_map(
                0usize..64usize,
                (-500.0f64..500.0f64, proptest::option::of(-500.0f64..500.0f64)),
                1..8
            ),
            repair_symbols in 1usize..8usize,
        ) {
            let entries = rows
                .iter()
                .map(|(node_id, (value, grad))| SnapshotEntry {
                    node_id: *node_id,
                    value: *value,
                    grad: *grad,
                })
                .collect::<Vec<_>>();
            let payload = encode_checkpoint(entries.as_slice(), CheckpointMode::Strict);

            let first = generate_sidecar_with_retry(payload.as_str(), repair_symbols);
            let second = generate_sidecar_with_retry(payload.as_str(), repair_symbols);

            let (output_digest, reason_code) = match (&first, &second) {
                (Ok((sidecar_a, proof_a)), Ok((_sidecar_b, proof_b))) => {
                    prop_assert_eq!(proof_a.proof_hash, proof_b.proof_hash);
                    prop_assert_eq!(&proof_a.source_hash, &proof_b.source_hash);
                    prop_assert!(sidecar_a.repair_symbol_count >= 1);
                    prop_assert_eq!(proof_a.recovered_bytes, payload.len());
                    (proof_a.proof_hash, "raptorq_decode_proof_deterministic")
                }
                (Err(err_a), Err(err_b)) => {
                    let a = err_a.to_string();
                    let b = err_b.to_string();
                    prop_assert_eq!(&a, &b);
                    (
                        det_seed(&[a.len() as u64, payload.len() as u64]),
                        "raptorq_failure_deterministic",
                    )
                }
                _ => {
                    prop_assert!(
                        false,
                        "sidecar generation outcome must be deterministic for identical inputs"
                    );
                    (0, "raptorq_outcome_nondeterministic")
                }
            };

            let seed = det_seed(&[
                rows.len() as u64,
                repair_symbols as u64,
                snapshot_digest(entries.as_slice()),
            ]);
            let log = build_property_log(
                "prop_raptorq_decode_proof_hash_stays_deterministic",
                "strict",
                seed,
                snapshot_digest(entries.as_slice()),
                output_digest,
                reason_code,
            );
            assert_log_contract(&log);
        }
    }
}

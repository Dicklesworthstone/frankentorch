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
        format!("{}...", &input[..max_len])
    }
}

#[cfg(test)]
mod tests {
    use serde_json::json;

    use super::{
        CheckpointMode, DecodeMode, SnapshotEntry, decode_checkpoint, decode_snapshot,
        encode_checkpoint, encode_snapshot, generate_raptorq_sidecar,
    };

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
            generate_raptorq_sidecar(&payload, 4).expect("sidecar generation should succeed");

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
            generate_raptorq_sidecar(&payload, 4).expect("first sidecar generation should work");
        let (_, proof_b) =
            generate_raptorq_sidecar(&payload, 4).expect("second sidecar generation should work");

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
}

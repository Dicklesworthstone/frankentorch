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
const MAX_CHECKPOINT_PAYLOAD_BYTES: usize = 1_048_576;

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

pub fn encode_checkpoint(
    entries: &[SnapshotEntry],
    mode: CheckpointMode,
) -> Result<String, SerializeError> {
    let normalized_entries = normalize_entries(entries);
    let source_hash = checkpoint_hash(CHECKPOINT_SCHEMA_VERSION, mode, &normalized_entries);

    let envelope = CheckpointEnvelope {
        schema_version: CHECKPOINT_SCHEMA_VERSION,
        mode,
        entries: normalized_entries,
        source_hash,
    };

    serde_json::to_string(&envelope).map_err(|error| SerializeError::IncompatiblePayload {
        reason: format!("checkpoint encoding failed: {error}"),
    })
}

pub fn decode_checkpoint(
    input: &str,
    mode: DecodeMode,
) -> Result<CheckpointEnvelope, SerializeError> {
    validate_payload_size(input)?;
    match mode {
        DecodeMode::Strict => decode_checkpoint_strict(input),
        DecodeMode::Hardened => decode_checkpoint_hardened(input),
    }
}

pub fn encode_snapshot(entries: &[SnapshotEntry]) -> Result<String, SerializeError> {
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

    let decoder = InactivationDecoder::new(source_symbol_count, symbol_size, seed);
    let params = decoder.params();
    let min_repair_symbols = params.k_prime.saturating_sub(source_symbol_count);
    // Keep a small deterministic headroom so decode can try alternative symbol mixes
    // when the first linear system is singular.
    let repair_count = repair_symbols
        .max(1)
        .max(min_repair_symbols)
        .saturating_add(8);

    let mut encoder =
        SystematicEncoder::new(&source_symbols, symbol_size, seed).ok_or_else(|| {
            SerializeError::RaptorQFailure {
                reason: "failed to initialize systematic encoder".to_string(),
            }
        })?;

    let systematic = encoder.emit_systematic();
    let repair = encoder.emit_repair(repair_count);
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

    let mut min_required_candidate = constraints.clone();
    min_required_candidate.extend(source_received.clone());
    min_required_candidate.extend(repair_received.iter().take(min_repair_symbols).cloned());
    candidates.push(min_required_candidate);

    let max_swaps = source_received
        .len()
        .min(repair_received.len().saturating_sub(min_repair_symbols))
        .min(16);
    for swap in 1..=max_swaps {
        let mut swapped = constraints.clone();
        swapped.extend(source_received.iter().skip(swap).cloned());
        swapped.extend(
            repair_received
                .iter()
                .take(min_repair_symbols + swap)
                .cloned(),
        );
        candidates.push(swapped);
    }

    if repair_received.len() > min_repair_symbols {
        let mut tail_repairs = constraints.clone();
        tail_repairs.extend(source_received.iter().cloned());
        tail_repairs.extend(
            repair_received
                .iter()
                .rev()
                .take(min_repair_symbols)
                .cloned(),
        );
        candidates.push(tail_repairs);
    }

    let mut all_symbols = constraints.clone();
    all_symbols.extend(source_received);
    all_symbols.extend(repair_received);
    candidates.push(all_symbols);

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
    let mut envelope: CheckpointEnvelope = serde_json::from_str(input).map_err(|error| {
        if let Some(field) = extract_unknown_field(error.to_string().as_str()) {
            SerializeError::UnknownField { field }
        } else {
            SerializeError::InvalidJson {
                diagnostic: bounded(error.to_string().as_str(), 200),
            }
        }
    })?;
    validate_checkpoint(&envelope)?;
    envelope.entries = normalize_entries(envelope.entries.as_slice());
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

    let mut envelope: CheckpointEnvelope =
        serde_json::from_value(raw).map_err(|error| SerializeError::IncompatiblePayload {
            reason: bounded(error.to_string().as_str(), 200),
        })?;

    validate_checkpoint(&envelope)?;
    envelope.entries = normalize_entries(envelope.entries.as_slice());
    Ok(envelope)
}

fn validate_payload_size(input: &str) -> Result<(), SerializeError> {
    let actual = input.len();
    if actual > MAX_CHECKPOINT_PAYLOAD_BYTES {
        return Err(SerializeError::IncompatiblePayload {
            reason: format!(
                "checkpoint payload exceeds max bytes: actual={actual} max={MAX_CHECKPOINT_PAYLOAD_BYTES}"
            ),
        });
    }
    Ok(())
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

// ── Tensor State Dict Save/Load ─────────────────────────────────────────

use std::collections::BTreeMap;
use std::path::Path;

use ft_core::{DType, DenseTensor, DenseTensorError, Device, TensorMeta};

/// Magic bytes identifying a FrankenTorch state dict file.
const FT_MAGIC: &[u8; 4] = b"FTSV";
/// Current format version.
const FT_STATE_FORMAT_VERSION: u32 = 1;

/// Errors from tensor state dict save/load operations.
#[derive(Debug, Clone, PartialEq)]
pub enum TensorIOError {
    /// I/O failure (path, message).
    Io { path: String, message: String },
    /// Bad magic bytes in file header.
    InvalidMagic,
    /// Unsupported format version.
    UnsupportedVersion { found: u32, max: u32 },
    /// File is truncated or corrupt.
    Corrupt { reason: String },
    /// Tensor construction error.
    TensorError(DenseTensorError),
}

impl fmt::Display for TensorIOError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Io { path, message } => write!(f, "I/O error at '{path}': {message}"),
            Self::InvalidMagic => write!(f, "invalid magic bytes: not a FrankenTorch state file"),
            Self::UnsupportedVersion { found, max } => {
                write!(
                    f,
                    "unsupported format version {found} (max supported: {max})"
                )
            }
            Self::Corrupt { reason } => write!(f, "corrupt state file: {reason}"),
            Self::TensorError(e) => write!(f, "tensor error: {e}"),
        }
    }
}

impl std::error::Error for TensorIOError {}

impl From<DenseTensorError> for TensorIOError {
    fn from(e: DenseTensorError) -> Self {
        Self::TensorError(e)
    }
}

fn io_err(path: &str, e: std::io::Error) -> TensorIOError {
    TensorIOError::Io {
        path: path.to_string(),
        message: e.to_string(),
    }
}

fn dtype_to_tag(dtype: DType) -> u8 {
    match dtype {
        DType::F64 => 0,
        DType::F32 => 1,
        _ => 255, // unsupported for now
    }
}

fn tag_to_dtype(tag: u8) -> Result<DType, TensorIOError> {
    match tag {
        0 => Ok(DType::F64),
        1 => Ok(DType::F32),
        _ => Err(TensorIOError::Corrupt {
            reason: format!("unknown dtype tag: {tag}"),
        }),
    }
}

/// Save a state dict (map of named tensors) to a file in FrankenTorch native format.
///
/// Format: `FTSV` magic + version(u32) + num_tensors(u64) + per-tensor data.
/// Each tensor: key_len(u64) + key_bytes + ndim(u64) + shape(ndim * u64) + dtype(u8) + values.
pub fn save_state_dict<P: AsRef<Path>>(
    state_dict: &BTreeMap<String, DenseTensor>,
    path: P,
) -> Result<(), TensorIOError> {
    let path_str = path.as_ref().to_string_lossy().to_string();
    let encoded = encode_state_dict_to_bytes(state_dict)?;
    std::fs::write(&path, encoded).map_err(|e| io_err(&path_str, e))?;
    Ok(())
}

fn encode_state_dict_to_bytes(
    state_dict: &BTreeMap<String, DenseTensor>,
) -> Result<Vec<u8>, TensorIOError> {
    let mut encoded = Vec::new();
    // Magic
    encoded.extend_from_slice(FT_MAGIC);
    // Version
    encoded.extend_from_slice(&FT_STATE_FORMAT_VERSION.to_le_bytes());
    // Number of tensors
    let num_tensors = state_dict.len() as u64;
    encoded.extend_from_slice(&num_tensors.to_le_bytes());

    for (key, tensor) in state_dict {
        let meta = tensor.meta();
        let dtype_tag = match meta.dtype() {
            DType::F64 => dtype_to_tag(DType::F64),
            DType::F32 => dtype_to_tag(DType::F32),
            other => {
                return Err(TensorIOError::Corrupt {
                    reason: format!("unsupported dtype for save: {other:?}"),
                });
            }
        };

        // Key
        let key_bytes = key.as_bytes();
        encoded.extend_from_slice(&(key_bytes.len() as u64).to_le_bytes());
        encoded.extend_from_slice(key_bytes);

        // Shape
        let shape = meta.shape();
        encoded.extend_from_slice(&(shape.len() as u64).to_le_bytes());
        for &dim in shape {
            encoded.extend_from_slice(&(dim as u64).to_le_bytes());
        }

        // DType
        encoded.push(dtype_tag);

        // Values
        match meta.dtype() {
            DType::F64 => {
                let values = tensor
                    .contiguous_values()
                    .map_err(TensorIOError::TensorError)?;
                for &v in values {
                    encoded.extend_from_slice(&v.to_le_bytes());
                }
            }
            DType::F32 => {
                let values = tensor
                    .contiguous_values_f32()
                    .map_err(TensorIOError::TensorError)?;
                for &v in values {
                    encoded.extend_from_slice(&v.to_le_bytes());
                }
            }
            other => {
                return Err(TensorIOError::Corrupt {
                    reason: format!("unsupported dtype for save: {other:?}"),
                });
            }
        }
    }

    Ok(encoded)
}

/// Load a state dict from a FrankenTorch native format file.
pub fn load_state_dict<P: AsRef<Path>>(
    path: P,
) -> Result<BTreeMap<String, DenseTensor>, TensorIOError> {
    let path_str = path.as_ref().to_string_lossy().to_string();
    let data = std::fs::read(&path).map_err(|e| io_err(&path_str, e))?;
    load_state_dict_from_bytes(&data)
}

/// Load a state dict from raw bytes.
pub fn load_state_dict_from_bytes(
    data: &[u8],
) -> Result<BTreeMap<String, DenseTensor>, TensorIOError> {
    let mut pos = 0;

    // Magic
    if data.len() < 4 || &data[0..4] != FT_MAGIC {
        return Err(TensorIOError::InvalidMagic);
    }
    pos += 4;

    // Version
    let version = read_u32(data, &mut pos)?;
    if version > FT_STATE_FORMAT_VERSION {
        return Err(TensorIOError::UnsupportedVersion {
            found: version,
            max: FT_STATE_FORMAT_VERSION,
        });
    }

    // Number of tensors — bound by remaining bytes (each tensor needs ≥17 bytes of header)
    let num_tensors = read_u64(data, &mut pos)? as usize;
    let remaining = data.len().saturating_sub(pos);
    if num_tensors > remaining {
        return Err(TensorIOError::Corrupt {
            reason: format!("claimed {num_tensors} tensors but only {remaining} bytes remain"),
        });
    }

    let mut result = BTreeMap::new();
    for _ in 0..num_tensors {
        // Key
        let key_len = read_u64(data, &mut pos)? as usize;
        if pos + key_len > data.len() {
            return Err(TensorIOError::Corrupt {
                reason: "truncated key data".to_string(),
            });
        }
        let key = String::from_utf8(data[pos..pos + key_len].to_vec()).map_err(|_| {
            TensorIOError::Corrupt {
                reason: "invalid UTF-8 in key".to_string(),
            }
        })?;
        pos += key_len;

        // Shape
        let ndim = read_u64(data, &mut pos)? as usize;
        let mut shape = Vec::with_capacity(ndim);
        for _ in 0..ndim {
            shape.push(read_u64(data, &mut pos)? as usize);
        }

        // DType
        if pos >= data.len() {
            return Err(TensorIOError::Corrupt {
                reason: "truncated dtype".to_string(),
            });
        }
        let dtype = tag_to_dtype(data[pos])?;
        pos += 1;

        // Values — use checked arithmetic to reject adversarial shapes
        let numel: usize = shape.iter().try_fold(1usize, |acc, &dim| {
            acc.checked_mul(dim).ok_or_else(|| TensorIOError::Corrupt {
                reason: format!("shape overflow in native state dict tensor '{key}'"),
            })
        })?;
        let meta = TensorMeta::from_shape(shape, dtype, Device::Cpu);

        let tensor = match dtype {
            DType::F64 => {
                let needed = numel.checked_mul(8).ok_or_else(|| TensorIOError::Corrupt {
                    reason: format!("byte count overflow for f64 tensor '{key}'"),
                })?;
                if pos + needed > data.len() {
                    return Err(TensorIOError::Corrupt {
                        reason: "truncated f64 data".to_string(),
                    });
                }
                let mut values = Vec::with_capacity(numel);
                for _ in 0..numel {
                    let bytes = read_fixed_bytes::<8>(data, &mut pos, "truncated f64 data")?;
                    values.push(f64::from_le_bytes(bytes));
                }
                DenseTensor::from_storage(meta, values)?
            }
            DType::F32 => {
                let needed = numel.checked_mul(4).ok_or_else(|| TensorIOError::Corrupt {
                    reason: format!("byte count overflow for f32 tensor '{key}'"),
                })?;
                if pos + needed > data.len() {
                    return Err(TensorIOError::Corrupt {
                        reason: "truncated f32 data".to_string(),
                    });
                }
                let mut values = Vec::with_capacity(numel);
                for _ in 0..numel {
                    let bytes = read_fixed_bytes::<4>(data, &mut pos, "truncated f32 data")?;
                    values.push(f32::from_le_bytes(bytes));
                }
                DenseTensor::from_storage_f32(meta, values)?
            }
            _ => {
                return Err(TensorIOError::Corrupt {
                    reason: format!("unsupported dtype in file: {dtype:?}"),
                });
            }
        };

        result.insert(key, tensor);
    }

    if pos != data.len() {
        return Err(TensorIOError::Corrupt {
            reason: format!(
                "trailing bytes after state dict payload: remaining={}",
                data.len() - pos
            ),
        });
    }

    Ok(result)
}

fn read_u32(data: &[u8], pos: &mut usize) -> Result<u32, TensorIOError> {
    let bytes = read_fixed_bytes::<4>(data, pos, "truncated u32")?;
    Ok(u32::from_le_bytes(bytes))
}

fn read_u64(data: &[u8], pos: &mut usize) -> Result<u64, TensorIOError> {
    let bytes = read_fixed_bytes::<8>(data, pos, "truncated u64")?;
    Ok(u64::from_le_bytes(bytes))
}

fn read_fixed_bytes<const N: usize>(
    data: &[u8],
    pos: &mut usize,
    truncated_reason: &'static str,
) -> Result<[u8; N], TensorIOError> {
    if *pos + N > data.len() {
        return Err(TensorIOError::Corrupt {
            reason: truncated_reason.to_string(),
        });
    }
    let bytes = bytes_to_array::<N>(&data[*pos..*pos + N], truncated_reason)?;
    *pos += N;
    Ok(bytes)
}

fn bytes_to_array<const N: usize>(data: &[u8], reason: &str) -> Result<[u8; N], TensorIOError> {
    data.try_into().map_err(|_| TensorIOError::Corrupt {
        reason: reason.to_string(),
    })
}

// ── SafeTensors Format Support ──────────────────────────────────────────

use std::borrow::Cow;

use ft_core::TensorStorage;
use safetensors::tensor::{self as st_tensor, Dtype as StDtype, SafeTensors};

/// Convert FrankenTorch DType to SafeTensors Dtype.
fn ft_dtype_to_st(dtype: DType) -> Result<StDtype, TensorIOError> {
    match dtype {
        DType::F64 => Ok(StDtype::F64),
        DType::F32 => Ok(StDtype::F32),
        DType::F16 => Ok(StDtype::F16),
        DType::BF16 => Ok(StDtype::BF16),
        DType::I64 | DType::I32 | DType::Bool => Err(TensorIOError::Corrupt {
            reason: format!(
                "integer/bool dtypes ({dtype:?}) are not supported by the DenseTensor SafeTensors bridge"
            ),
        }),
        DType::Complex64 | DType::Complex128 => Err(TensorIOError::Corrupt {
            reason: format!("complex dtypes ({dtype:?}) are not supported by SafeTensors"),
        }),
    }
}

/// Convert SafeTensors Dtype to FrankenTorch DType.
fn st_dtype_to_ft(dtype: StDtype) -> Result<DType, TensorIOError> {
    match dtype {
        StDtype::F64 => Ok(DType::F64),
        StDtype::F32 => Ok(DType::F32),
        StDtype::F16 => Ok(DType::F16),
        StDtype::BF16 => Ok(DType::BF16),
        StDtype::I64 | StDtype::I32 | StDtype::BOOL => Err(TensorIOError::Corrupt {
            reason: format!(
                "integer/bool SafeTensors dtype ({dtype:?}) is unsupported by the DenseTensor loader"
            ),
        }),
        other => Err(TensorIOError::Corrupt {
            reason: format!("unsupported SafeTensors dtype: {other:?}"),
        }),
    }
}

/// Wrapper to implement `safetensors::View` for a `DenseTensor`.
struct TensorViewAdapter<'a> {
    tensor: &'a DenseTensor,
    st_dtype: StDtype,
    storage_start: usize,
    storage_end: usize,
    data_len: usize,
}

impl st_tensor::View for TensorViewAdapter<'_> {
    fn dtype(&self) -> StDtype {
        self.st_dtype
    }

    fn shape(&self) -> &[usize] {
        self.tensor.meta().shape()
    }

    fn data(&self) -> Cow<'_, [u8]> {
        match self.tensor.typed_storage() {
            TensorStorage::F64(v) => {
                let slice = &v[self.storage_start..self.storage_end];
                let mut bytes = Vec::with_capacity(self.data_len);
                for value in slice {
                    bytes.extend_from_slice(&value.to_le_bytes());
                }
                Cow::Owned(bytes)
            }
            TensorStorage::F32(v) => {
                let slice = &v[self.storage_start..self.storage_end];
                let mut bytes = Vec::with_capacity(self.data_len);
                for value in slice {
                    bytes.extend_from_slice(&value.to_le_bytes());
                }
                Cow::Owned(bytes)
            }
            TensorStorage::F16(v) => {
                let slice = &v[self.storage_start..self.storage_end];
                let mut bytes = Vec::with_capacity(self.data_len);
                for value in slice {
                    bytes.extend_from_slice(&value.to_le_bytes());
                }
                Cow::Owned(bytes)
            }
            TensorStorage::BF16(v) => {
                let slice = &v[self.storage_start..self.storage_end];
                let mut bytes = Vec::with_capacity(self.data_len);
                for value in slice {
                    bytes.extend_from_slice(&value.to_le_bytes());
                }
                Cow::Owned(bytes)
            }
            TensorStorage::Complex64(v) => {
                let slice = &v[self.storage_start..self.storage_end];
                let mut bytes = Vec::with_capacity(self.data_len);
                for value in slice {
                    bytes.extend_from_slice(&value.re.to_le_bytes());
                    bytes.extend_from_slice(&value.im.to_le_bytes());
                }
                Cow::Owned(bytes)
            }
            TensorStorage::Complex128(v) => {
                let slice = &v[self.storage_start..self.storage_end];
                let mut bytes = Vec::with_capacity(self.data_len);
                for value in slice {
                    bytes.extend_from_slice(&value.re.to_le_bytes());
                    bytes.extend_from_slice(&value.im.to_le_bytes());
                }
                Cow::Owned(bytes)
            }
        }
    }

    fn data_len(&self) -> usize {
        self.data_len
    }
}

impl<'a> TensorViewAdapter<'a> {
    fn try_new(
        tensor: &'a DenseTensor,
        st_dtype: StDtype,
        name: &str,
    ) -> Result<Self, TensorIOError> {
        let meta = tensor.meta();
        if !meta.is_contiguous() {
            return Err(TensorIOError::Corrupt {
                reason: format!("safetensors requires contiguous tensor '{name}'"),
            });
        }

        let numel = meta.numel();
        if numel == usize::MAX {
            return Err(TensorIOError::Corrupt {
                reason: format!("safetensors shape overflow for tensor '{name}'"),
            });
        }

        let storage_start = meta.storage_offset();
        let storage_end =
            storage_start
                .checked_add(numel)
                .ok_or_else(|| TensorIOError::Corrupt {
                    reason: format!("safetensors storage span overflow for tensor '{name}'"),
                })?;

        let storage_len = tensor.typed_storage().len();
        if storage_end > storage_len {
            return Err(TensorIOError::Corrupt {
                reason: format!(
                    "safetensors storage span out of range for tensor '{name}': end={storage_end} len={storage_len}"
                ),
            });
        }

        let element_size = meta.dtype().element_size();
        let data_len = numel
            .checked_mul(element_size)
            .ok_or_else(|| TensorIOError::Corrupt {
                reason: format!("safetensors byte length overflow for tensor '{name}'"),
            })?;

        Ok(Self {
            tensor,
            st_dtype,
            storage_start,
            storage_end,
            data_len,
        })
    }
}

/// Save a state dict to a file in SafeTensors format.
///
/// SafeTensors is the Hugging Face standard for safe, zero-copy tensor storage.
/// Uses a JSON header with tensor metadata followed by raw binary data.
pub fn save_safetensors<P: AsRef<Path>>(
    state_dict: &BTreeMap<String, DenseTensor>,
    path: P,
    metadata: Option<&std::collections::HashMap<String, String>>,
) -> Result<(), TensorIOError> {
    let path_str = path.as_ref().to_string_lossy().to_string();

    // Build the list of (name, view) pairs
    let views: Vec<(String, TensorViewAdapter<'_>)> = state_dict
        .iter()
        .map(|(name, tensor)| {
            let st_dtype = ft_dtype_to_st(tensor.meta().dtype())?;
            let view = TensorViewAdapter::try_new(tensor, st_dtype, name)?;
            Ok((name.clone(), view))
        })
        .collect::<Result<Vec<_>, TensorIOError>>()?;

    let data =
        st_tensor::serialize(views, metadata.cloned()).map_err(|e| TensorIOError::Corrupt {
            reason: format!("safetensors serialization failed: {e}"),
        })?;

    std::fs::write(&path, data).map_err(|e| io_err(&path_str, e))?;

    Ok(())
}

/// Load a state dict from a SafeTensors format file.
pub fn load_safetensors<P: AsRef<Path>>(
    path: P,
) -> Result<BTreeMap<String, DenseTensor>, TensorIOError> {
    let path_str = path.as_ref().to_string_lossy().to_string();
    let data = std::fs::read(&path).map_err(|e| io_err(&path_str, e))?;
    load_safetensors_from_bytes(&data)
}

/// Load a state dict from SafeTensors-formatted bytes.
pub fn load_safetensors_from_bytes(
    data: &[u8],
) -> Result<BTreeMap<String, DenseTensor>, TensorIOError> {
    let tensors = SafeTensors::deserialize(data).map_err(|e| TensorIOError::Corrupt {
        reason: format!("safetensors deserialization failed: {e}"),
    })?;

    let mut result = BTreeMap::new();

    for (name, view) in tensors.tensors() {
        let dtype = st_dtype_to_ft(view.dtype())?;
        let shape: Vec<usize> = view.shape().to_vec();
        let raw_data = view.data();

        let tensor = match dtype {
            DType::F64 => {
                let numel = validate_safetensors_byte_width(&name, shape.as_slice(), raw_data, 8)?;
                let mut values = Vec::with_capacity(numel);
                for chunk in raw_data.chunks_exact(8) {
                    let bytes = bytes_to_array::<8>(
                        chunk,
                        &format!("invalid f64 payload width in SafeTensors tensor '{name}'"),
                    )?;
                    values.push(f64::from_le_bytes(bytes));
                }
                let meta = TensorMeta::from_shape(shape, DType::F64, Device::Cpu);
                DenseTensor::from_storage(meta, values)?
            }
            DType::F32 => {
                let numel = validate_safetensors_byte_width(&name, shape.as_slice(), raw_data, 4)?;
                let mut values = Vec::with_capacity(numel);
                for chunk in raw_data.chunks_exact(4) {
                    let bytes = bytes_to_array::<4>(
                        chunk,
                        &format!("invalid f32 payload width in SafeTensors tensor '{name}'"),
                    )?;
                    values.push(f32::from_le_bytes(bytes));
                }
                let meta = TensorMeta::from_shape(shape, DType::F32, Device::Cpu);
                DenseTensor::from_storage_f32(meta, values)?
            }
            DType::F16 => {
                let numel = validate_safetensors_byte_width(&name, shape.as_slice(), raw_data, 2)?;
                let mut values = Vec::with_capacity(numel);
                for chunk in raw_data.chunks_exact(2) {
                    let bytes = bytes_to_array::<2>(
                        chunk,
                        &format!("invalid f16 payload width in SafeTensors tensor '{name}'"),
                    )?;
                    values.push(ft_core::Float16::from_le_bytes(bytes));
                }
                let meta = TensorMeta::from_shape(shape, DType::F16, Device::Cpu);
                DenseTensor::from_storage_f16(meta, values)?
            }
            DType::BF16 => {
                let numel = validate_safetensors_byte_width(&name, shape.as_slice(), raw_data, 2)?;
                let mut values = Vec::with_capacity(numel);
                for chunk in raw_data.chunks_exact(2) {
                    let bytes = bytes_to_array::<2>(
                        chunk,
                        &format!("invalid bf16 payload width in SafeTensors tensor '{name}'"),
                    )?;
                    values.push(ft_core::BFloat16::from_le_bytes(bytes));
                }
                let meta = TensorMeta::from_shape(shape, DType::BF16, Device::Cpu);
                DenseTensor::from_storage_bf16(meta, values)?
            }
            _ => {
                return Err(TensorIOError::Corrupt {
                    reason: format!("unsupported dtype in SafeTensors file: {dtype:?}"),
                });
            }
        };

        result.insert(name, tensor);
    }

    Ok(result)
}

fn validate_safetensors_byte_width(
    name: &str,
    shape: &[usize],
    raw_data: &[u8],
    element_width: usize,
) -> Result<usize, TensorIOError> {
    let numel = shape.iter().try_fold(1usize, |acc, &dim| {
        acc.checked_mul(dim).ok_or_else(|| TensorIOError::Corrupt {
            reason: format!("shape overflow in SafeTensors tensor '{name}'"),
        })
    })?;
    let expected_bytes =
        numel
            .checked_mul(element_width)
            .ok_or_else(|| TensorIOError::Corrupt {
                reason: format!("byte width overflow in SafeTensors tensor '{name}'"),
            })?;

    if raw_data.len() != expected_bytes {
        return Err(TensorIOError::Corrupt {
            reason: format!(
                "invalid payload width in SafeTensors tensor '{name}': expected={expected_bytes} actual={}",
                raw_data.len()
            ),
        });
    }

    Ok(numel)
}

/// Load SafeTensors metadata (the string-to-string map) from a file.
pub fn load_safetensors_metadata<P: AsRef<Path>>(
    path: P,
) -> Result<Option<std::collections::HashMap<String, String>>, TensorIOError> {
    let path_str = path.as_ref().to_string_lossy().to_string();
    let data = std::fs::read(&path).map_err(|e| io_err(&path_str, e))?;
    let (_header_size, metadata) =
        SafeTensors::read_metadata(&data).map_err(|e| TensorIOError::Corrupt {
            reason: format!("safetensors metadata read failed: {e}"),
        })?;

    Ok(metadata.metadata().clone())
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

    #[test]
    fn bytes_to_array_reports_corrupt_for_short_slice() {
        let err = super::bytes_to_array::<4>(&[1, 2, 3], "short slice")
            .expect_err("short slice should be rejected");
        assert_eq!(
            err,
            super::TensorIOError::Corrupt {
                reason: "short slice".to_string(),
            }
        );
    }

    #[test]
    fn read_fixed_bytes_reports_corrupt_for_truncated_input() {
        let mut pos = 0;
        let err = super::read_fixed_bytes::<8>(&[1, 2, 3, 4], &mut pos, "truncated bytes")
            .expect_err("truncated bytes should be rejected");
        assert_eq!(
            err,
            super::TensorIOError::Corrupt {
                reason: "truncated bytes".to_string(),
            }
        );
        assert_eq!(pos, 0);
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

        let encoded =
            encode_checkpoint(&entries, CheckpointMode::Strict).expect("strict encode should work");
        let decoded = decode_checkpoint(&encoded, DecodeMode::Strict).expect("strict decode");

        assert_eq!(decoded.entries[0].node_id, 0);
        assert_eq!(decoded.entries[0].value, 2.0);
        assert_eq!(decoded.entries[0].grad, None);
        assert_eq!(decoded.entries[1].node_id, 1);
        assert_eq!(decoded.entries[1].value, 3.0);
        assert_eq!(decoded.entries[1].grad, Some(2.0));
    }

    #[test]
    fn strict_decode_canonicalizes_entry_order() {
        let entries = vec![
            SnapshotEntry {
                node_id: 2,
                value: 20.0,
                grad: None,
            },
            SnapshotEntry {
                node_id: 0,
                value: 0.0,
                grad: Some(1.0),
            },
            SnapshotEntry {
                node_id: 1,
                value: 10.0,
                grad: None,
            },
        ];
        let encoded =
            encode_checkpoint(&entries, CheckpointMode::Strict).expect("strict encode should work");
        let mut payload: serde_json::Value =
            serde_json::from_str(&encoded).expect("valid encoded checkpoint");
        let mut reversed_entries = payload["entries"]
            .as_array()
            .expect("entries must be an array")
            .clone();
        reversed_entries.reverse();
        payload["entries"] = serde_json::Value::Array(reversed_entries);

        let decoded = decode_checkpoint(payload.to_string().as_str(), DecodeMode::Strict)
            .expect("strict decode should canonicalize entry order");
        let node_ids: Vec<usize> = decoded.entries.iter().map(|entry| entry.node_id).collect();
        assert_eq!(node_ids, vec![0, 1, 2]);
    }

    #[test]
    fn hardened_decode_canonicalizes_entry_order() {
        let entries = vec![
            SnapshotEntry {
                node_id: 4,
                value: 4.0,
                grad: None,
            },
            SnapshotEntry {
                node_id: 3,
                value: 3.0,
                grad: Some(0.5),
            },
        ];
        let encoded = encode_checkpoint(&entries, CheckpointMode::Hardened)
            .expect("hardened encode should work");
        let mut payload: serde_json::Value =
            serde_json::from_str(&encoded).expect("valid encoded checkpoint");
        let mut reversed_entries = payload["entries"]
            .as_array()
            .expect("entries must be an array")
            .clone();
        reversed_entries.reverse();
        payload["entries"] = serde_json::Value::Array(reversed_entries);

        let decoded = decode_checkpoint(payload.to_string().as_str(), DecodeMode::Hardened)
            .expect("hardened decode should canonicalize entry order");
        let node_ids: Vec<usize> = decoded.entries.iter().map(|entry| entry.node_id).collect();
        assert_eq!(node_ids, vec![3, 4]);
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
        let encoded =
            encode_checkpoint(&entries, CheckpointMode::Strict).expect("strict encode should work");
        let mut payload: serde_json::Value =
            serde_json::from_str(&encoded).expect("valid encoded checkpoint");
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
        let encoded =
            encode_checkpoint(&entries, CheckpointMode::Strict).expect("strict encode should work");
        let mut payload: serde_json::Value =
            serde_json::from_str(&encoded).expect("valid encoded checkpoint");
        payload["source_hash"] = json!("det64:deadbeefdeadbeef");

        let err = decode_checkpoint(payload.to_string().as_str(), DecodeMode::Strict)
            .expect_err("checksum mismatch should fail");
        assert!(err.to_string().contains("checksum mismatch"));
    }

    #[test]
    fn strict_oversized_payload_is_fail_closed() {
        let payload = "x".repeat(super::MAX_CHECKPOINT_PAYLOAD_BYTES + 1);
        let err = decode_checkpoint(payload.as_str(), DecodeMode::Strict)
            .expect_err("oversized payload must fail");
        assert!(matches!(err, SerializeError::IncompatiblePayload { .. }));
        assert!(err.to_string().contains("exceeds max bytes"));
    }

    #[test]
    fn hardened_oversized_payload_is_fail_closed() {
        let payload = "x".repeat(super::MAX_CHECKPOINT_PAYLOAD_BYTES + 1);
        let err = decode_checkpoint(payload.as_str(), DecodeMode::Hardened)
            .expect_err("oversized payload must fail");
        assert!(matches!(err, SerializeError::IncompatiblePayload { .. }));
        assert!(err.to_string().contains("exceeds max bytes"));
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
        let payload =
            encode_checkpoint(&entries, CheckpointMode::Strict).expect("strict encode should work");

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
        let payload =
            encode_checkpoint(&entries, CheckpointMode::Strict).expect("strict encode should work");

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

        let encoded = encode_snapshot(&entries).expect("strict encode should work");
        let decoded = decode_snapshot(&encoded).expect("legacy wrapper decode should work");
        assert_eq!(decoded, entries);
    }

    proptest! {
        #[test]
        fn prop_checkpoint_roundtrip_preserves_sorted_entries(
            entries in prop::collection::vec(snapshot_entry_strategy(), 1..16),
        ) {
            let encoded = encode_checkpoint(entries.as_slice(), CheckpointMode::Strict)
                .expect("strict encode should work");
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

            let encoded_a = encode_checkpoint(sorted_entries.as_slice(), CheckpointMode::Strict)
                .expect("strict encode should work");
            let encoded_b = encode_checkpoint(reversed_entries.as_slice(), CheckpointMode::Strict)
                .expect("strict encode should work");

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
            let encoded = encode_checkpoint(entries.as_slice(), CheckpointMode::Strict)
                .expect("strict encode should work");
            let mut payload: serde_json::Value =
                serde_json::from_str(encoded.as_str()).expect("checkpoint payload should parse");
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
            let payload = encode_checkpoint(entries.as_slice(), CheckpointMode::Strict)
                .expect("strict encode should work");

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

    // ── bd-437p: hardened encode/decode and boundary cases ──

    #[test]
    fn encode_checkpoint_hardened_mode() {
        let entries = vec![
            SnapshotEntry {
                node_id: 0,
                value: 1.0,
                grad: Some(0.5),
            },
            SnapshotEntry {
                node_id: 1,
                value: 2.0,
                grad: None,
            },
        ];
        let encoded =
            encode_checkpoint(&entries, CheckpointMode::Hardened).expect("hardened encode");
        assert!(encoded.contains("\"mode\":\"hardened\""));
    }

    #[test]
    fn decode_hardened_encoded_checkpoint_in_hardened_mode() {
        let entries = vec![SnapshotEntry {
            node_id: 0,
            value: 42.0,
            grad: Some(1.0),
        }];
        let encoded =
            encode_checkpoint(&entries, CheckpointMode::Hardened).expect("hardened encode");
        let decoded = decode_checkpoint(&encoded, DecodeMode::Hardened).expect("hardened decode");
        assert_eq!(decoded.mode, CheckpointMode::Hardened);
        assert_eq!(decoded.entries.len(), 1);
        assert!((decoded.entries[0].value - 42.0).abs() < 1e-12);
    }

    #[test]
    fn hardened_decode_rejects_non_object_payload() {
        let err = decode_checkpoint("[1, 2, 3]", DecodeMode::Hardened)
            .expect_err("JSON array should fail hardened decode");
        assert!(
            matches!(err, SerializeError::IncompatiblePayload { .. }),
            "expected IncompatiblePayload, got {err:?}"
        );
    }

    #[test]
    fn hardened_decode_rejects_unknown_field() {
        let payload = r#"{"schema_version":1,"mode":"strict","entries":[],"source_hash":"det64:placeholder","extra":1}"#;
        let err = decode_checkpoint(payload, DecodeMode::Hardened)
            .expect_err("unknown field should fail hardened decode");
        assert!(
            matches!(err, SerializeError::UnknownField { ref field } if field == "extra"),
            "expected UnknownField 'extra', got {err:?}"
        );
    }

    #[test]
    fn hardened_decode_provides_bounded_diagnostic_for_malformed_json() {
        let err = decode_checkpoint("{not valid json!!!", DecodeMode::Hardened)
            .expect_err("malformed JSON should fail");
        assert!(
            matches!(err, SerializeError::InvalidJson { .. }),
            "expected InvalidJson, got {err:?}"
        );
    }

    #[test]
    fn serialize_error_display_coverage() {
        let cases: Vec<SerializeError> = vec![
            SerializeError::InvalidJson {
                diagnostic: "bad input".to_string(),
            },
            SerializeError::UnknownField {
                field: "foo".to_string(),
            },
            SerializeError::VersionMismatch {
                expected: 1,
                found: 2,
            },
            SerializeError::ChecksumMismatch {
                expected: "abc".to_string(),
                found: "def".to_string(),
            },
            SerializeError::IncompatiblePayload {
                reason: "test reason".to_string(),
            },
            SerializeError::RaptorQFailure {
                reason: "test failure".to_string(),
            },
        ];
        for err in &cases {
            let msg = format!("{err}");
            assert!(!msg.is_empty(), "Display should produce non-empty output");
        }
    }

    // ── Tensor State Dict Save/Load Tests ──────────────────────────────

    use super::{TensorIOError, load_state_dict, load_state_dict_from_bytes, save_state_dict};
    use ft_core::{DType, DenseTensor, Device, TensorMeta};

    fn make_f64_tensor(values: Vec<f64>, shape: Vec<usize>) -> DenseTensor {
        DenseTensor::from_contiguous_values(values, shape, Device::Cpu).unwrap()
    }

    #[test]
    fn save_load_single_tensor() {
        let dir = std::env::temp_dir().join("ft_test_save_single");
        let _ = std::fs::remove_file(&dir);
        let mut sd = BTreeMap::new();
        sd.insert(
            "weight".to_string(),
            make_f64_tensor(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]),
        );

        save_state_dict(&sd, &dir).unwrap();
        let loaded = load_state_dict(&dir).unwrap();
        let _ = std::fs::remove_file(&dir);

        assert_eq!(loaded.len(), 1);
        assert!(loaded.contains_key("weight"));
        let t = &loaded["weight"];
        assert_eq!(t.meta().shape(), &[2, 2]);
        assert_eq!(t.meta().dtype(), DType::F64);
        assert_eq!(t.contiguous_values().unwrap(), &[1.0, 2.0, 3.0, 4.0]);
    }

    #[test]
    fn save_load_multiple_tensors() {
        let dir = std::env::temp_dir().join("ft_test_save_multi");
        let _ = std::fs::remove_file(&dir);
        let mut sd = BTreeMap::new();
        sd.insert(
            "layer1.weight".to_string(),
            make_f64_tensor(vec![1.0, 2.0], vec![2]),
        );
        sd.insert(
            "layer1.bias".to_string(),
            make_f64_tensor(vec![0.5], vec![1]),
        );
        sd.insert(
            "layer2.weight".to_string(),
            make_f64_tensor(vec![3.0, 4.0, 5.0, 6.0], vec![2, 2]),
        );

        save_state_dict(&sd, &dir).unwrap();
        let loaded = load_state_dict(&dir).unwrap();
        let _ = std::fs::remove_file(&dir);

        assert_eq!(loaded.len(), 3);
        assert_eq!(
            loaded["layer1.weight"].contiguous_values().unwrap(),
            &[1.0, 2.0]
        );
        assert_eq!(loaded["layer1.bias"].contiguous_values().unwrap(), &[0.5]);
        assert_eq!(
            loaded["layer2.weight"].contiguous_values().unwrap(),
            &[3.0, 4.0, 5.0, 6.0]
        );
        assert_eq!(loaded["layer2.weight"].meta().shape(), &[2, 2]);
    }

    #[test]
    fn save_load_empty_state_dict() {
        let dir = std::env::temp_dir().join("ft_test_save_empty");
        let _ = std::fs::remove_file(&dir);
        let sd = BTreeMap::new();

        save_state_dict(&sd, &dir).unwrap();
        let loaded = load_state_dict(&dir).unwrap();
        let _ = std::fs::remove_file(&dir);

        assert!(loaded.is_empty());
    }

    #[test]
    fn save_load_scalar_tensor() {
        let dir = std::env::temp_dir().join("ft_test_save_scalar");
        let _ = std::fs::remove_file(&dir);
        let mut sd = BTreeMap::new();
        sd.insert("lr".to_string(), make_f64_tensor(vec![0.001], vec![1]));

        save_state_dict(&sd, &dir).unwrap();
        let loaded = load_state_dict(&dir).unwrap();
        let _ = std::fs::remove_file(&dir);

        assert_eq!(loaded["lr"].contiguous_values().unwrap(), &[0.001]);
    }

    #[test]
    fn save_load_f32_tensor() {
        let dir = std::env::temp_dir().join("ft_test_save_f32");
        let _ = std::fs::remove_file(&dir);
        let mut sd = BTreeMap::new();
        let meta = TensorMeta::from_shape(vec![3], DType::F32, Device::Cpu);
        let t = DenseTensor::from_storage_f32(meta, vec![1.0f32, 2.0, 3.0]).unwrap();
        sd.insert("w".to_string(), t);

        save_state_dict(&sd, &dir).unwrap();
        let loaded = load_state_dict(&dir).unwrap();
        let _ = std::fs::remove_file(&dir);

        assert_eq!(loaded["w"].meta().dtype(), DType::F32);
        assert_eq!(
            loaded["w"].contiguous_values_f32().unwrap(),
            &[1.0f32, 2.0, 3.0]
        );
    }

    #[test]
    fn load_invalid_magic() {
        let data = b"NOPE0000";
        let result = load_state_dict_from_bytes(data);
        assert!(matches!(result, Err(TensorIOError::InvalidMagic)));
    }

    #[test]
    fn load_truncated_file() {
        let data = b"FTSV";
        let result = load_state_dict_from_bytes(data);
        assert!(matches!(result, Err(TensorIOError::Corrupt { .. })));
    }

    #[test]
    fn load_unsupported_version() {
        let mut data = Vec::new();
        data.extend_from_slice(b"FTSV");
        data.extend_from_slice(&99u32.to_le_bytes()); // future version
        let result = load_state_dict_from_bytes(&data);
        assert!(matches!(
            result,
            Err(TensorIOError::UnsupportedVersion { .. })
        ));
    }

    #[test]
    fn load_state_dict_rejects_trailing_bytes() {
        let mut sd = BTreeMap::new();
        sd.insert("w".to_string(), make_f64_tensor(vec![1.0, 2.0], vec![2]));

        let mut bytes = super::encode_state_dict_to_bytes(&sd).unwrap();
        bytes.extend_from_slice(b"junk");

        let err = load_state_dict_from_bytes(&bytes).expect_err("trailing bytes must fail");
        assert!(matches!(err, TensorIOError::Corrupt { .. }));
        assert!(err.to_string().contains("trailing bytes"));
    }

    #[test]
    fn save_state_dict_unsupported_dtype_leaves_existing_file_untouched() {
        let path = std::env::temp_dir().join("ft_test_save_unsupported_dtype");
        let original = b"sentinel-state";
        std::fs::write(&path, original).unwrap();

        let meta = TensorMeta::from_shape(vec![2], DType::F16, Device::Cpu);
        let tensor = DenseTensor::from_storage_f16(
            meta,
            vec![
                ft_core::Float16::from_f32(1.0),
                ft_core::Float16::from_f32(2.0),
            ],
        )
        .unwrap();
        let mut sd = BTreeMap::new();
        sd.insert("half".to_string(), tensor);

        let err = save_state_dict(&sd, &path).expect_err("unsupported dtype must fail");
        let persisted = std::fs::read(&path).unwrap();
        let _ = std::fs::remove_file(&path);

        assert!(matches!(err, TensorIOError::Corrupt { .. }));
        assert_eq!(persisted, original);
    }

    #[test]
    fn magic_bytes_present() {
        let dir = std::env::temp_dir().join("ft_test_magic");
        let _ = std::fs::remove_file(&dir);
        let sd = BTreeMap::new();
        save_state_dict(&sd, &dir).unwrap();
        let data = std::fs::read(&dir).unwrap();
        let _ = std::fs::remove_file(&dir);
        assert_eq!(&data[0..4], b"FTSV");
    }

    #[test]
    fn save_load_large_tensor() {
        let dir = std::env::temp_dir().join("ft_test_save_large");
        let _ = std::fs::remove_file(&dir);
        let n = 10_000;
        let values: Vec<f64> = (0..n).map(|i| i as f64 * 0.001).collect();
        let mut sd = BTreeMap::new();
        sd.insert(
            "big".to_string(),
            make_f64_tensor(values.clone(), vec![100, 100]),
        );

        save_state_dict(&sd, &dir).unwrap();
        let loaded = load_state_dict(&dir).unwrap();
        let _ = std::fs::remove_file(&dir);

        let loaded_vals = loaded["big"].contiguous_values().unwrap();
        assert_eq!(loaded_vals.len(), n);
        for i in 0..n {
            assert!(
                (loaded_vals[i] - values[i]).abs() < f64::EPSILON,
                "mismatch at {i}"
            );
        }
    }

    #[test]
    fn tensor_io_error_display() {
        let cases = vec![
            TensorIOError::InvalidMagic,
            TensorIOError::UnsupportedVersion { found: 99, max: 1 },
            TensorIOError::Corrupt {
                reason: "test".to_string(),
            },
            TensorIOError::Io {
                path: "/tmp/test".to_string(),
                message: "not found".to_string(),
            },
        ];
        for err in &cases {
            let msg = format!("{err}");
            assert!(!msg.is_empty());
        }
    }

    #[test]
    fn overwrite_existing_file() {
        let dir = std::env::temp_dir().join("ft_test_overwrite");
        let _ = std::fs::remove_file(&dir);

        // Write first version
        let mut sd1 = BTreeMap::new();
        sd1.insert("a".to_string(), make_f64_tensor(vec![1.0], vec![1]));
        save_state_dict(&sd1, &dir).unwrap();

        // Overwrite with different data
        let mut sd2 = BTreeMap::new();
        sd2.insert("b".to_string(), make_f64_tensor(vec![2.0, 3.0], vec![2]));
        save_state_dict(&sd2, &dir).unwrap();

        let loaded = load_state_dict(&dir).unwrap();
        let _ = std::fs::remove_file(&dir);

        // Should only contain the second version
        assert_eq!(loaded.len(), 1);
        assert!(loaded.contains_key("b"));
        assert_eq!(loaded["b"].contiguous_values().unwrap(), &[2.0, 3.0]);
    }

    // ── SafeTensors Format Tests ────────────────────────────────────────

    use super::{
        load_safetensors, load_safetensors_from_bytes, load_safetensors_metadata, save_safetensors,
    };

    #[test]
    fn safetensors_round_trip_f64() {
        let path = std::env::temp_dir().join("ft_test_st_f64.safetensors");
        let _ = std::fs::remove_file(&path);
        let mut sd = BTreeMap::new();
        sd.insert(
            "weight".to_string(),
            make_f64_tensor(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]),
        );

        save_safetensors(&sd, &path, None).unwrap();
        let loaded = load_safetensors(&path).unwrap();
        let _ = std::fs::remove_file(&path);

        assert_eq!(loaded.len(), 1);
        let t = &loaded["weight"];
        assert_eq!(t.meta().shape(), &[2, 2]);
        assert_eq!(t.meta().dtype(), DType::F64);
        assert_eq!(t.contiguous_values().unwrap(), &[1.0, 2.0, 3.0, 4.0]);
    }

    #[test]
    fn safetensors_round_trip_f32() {
        let path = std::env::temp_dir().join("ft_test_st_f32.safetensors");
        let _ = std::fs::remove_file(&path);
        let mut sd = BTreeMap::new();
        let meta = TensorMeta::from_shape(vec![3], DType::F32, Device::Cpu);
        let t = DenseTensor::from_storage_f32(meta, vec![1.0f32, 2.0, 3.0]).unwrap();
        sd.insert("w".to_string(), t);

        save_safetensors(&sd, &path, None).unwrap();
        let loaded = load_safetensors(&path).unwrap();
        let _ = std::fs::remove_file(&path);

        assert_eq!(loaded["w"].meta().dtype(), DType::F32);
        assert_eq!(
            loaded["w"].contiguous_values_f32().unwrap(),
            &[1.0f32, 2.0, 3.0]
        );
    }

    #[test]
    fn safetensors_respects_storage_offset() {
        let path = std::env::temp_dir().join("ft_test_st_offset.safetensors");
        let _ = std::fs::remove_file(&path);
        let mut sd = BTreeMap::new();
        let meta = TensorMeta::from_shape(vec![2], DType::F32, Device::Cpu).with_storage_offset(1);
        let t = DenseTensor::from_storage_f32(meta, vec![0.5f32, 1.25, 2.5]).unwrap();
        sd.insert("x".to_string(), t);

        save_safetensors(&sd, &path, None).unwrap();
        let loaded = load_safetensors(&path).unwrap();
        let _ = std::fs::remove_file(&path);

        assert_eq!(
            loaded["x"].contiguous_values_f32().unwrap(),
            &[1.25f32, 2.5]
        );
    }

    #[test]
    fn safetensors_rejects_noncontiguous_tensor() {
        let path = std::env::temp_dir().join("ft_test_st_noncontig.safetensors");
        let _ = std::fs::remove_file(&path);
        let mut sd = BTreeMap::new();
        let meta =
            TensorMeta::from_shape_and_strides(vec![2, 2], vec![3, 1], 0, DType::F64, Device::Cpu)
                .unwrap();
        let t = DenseTensor::from_storage(meta, vec![1.0, 2.0, 3.0, 4.0, 5.0]).unwrap();
        sd.insert("x".to_string(), t);

        let err = save_safetensors(&sd, &path, None).expect_err("noncontiguous must fail");
        let msg = err.to_string();
        let _ = std::fs::remove_file(&path);

        assert!(
            msg.contains("contiguous"),
            "unexpected error message: {msg}"
        );
    }

    #[test]
    fn safetensors_round_trip_f16() {
        let path = std::env::temp_dir().join("ft_test_st_f16.safetensors");
        let _ = std::fs::remove_file(&path);
        let mut sd = BTreeMap::new();
        let vals: Vec<ft_core::Float16> = vec![1.0f32, 2.0, 3.0]
            .into_iter()
            .map(ft_core::Float16::from_f32)
            .collect();
        let meta = TensorMeta::from_shape(vec![3], DType::F16, Device::Cpu);
        let t = DenseTensor::from_storage_f16(meta, vals.clone()).unwrap();
        sd.insert("h".to_string(), t);

        save_safetensors(&sd, &path, None).unwrap();
        let loaded = load_safetensors(&path).unwrap();
        let _ = std::fs::remove_file(&path);

        assert_eq!(loaded["h"].meta().dtype(), DType::F16);
        let loaded_f32: Vec<f32> = loaded["h"].typed_storage().to_f32_vec();
        assert_eq!(loaded_f32, vec![1.0f32, 2.0, 3.0]);
    }

    #[test]
    fn safetensors_round_trip_bf16() {
        let path = std::env::temp_dir().join("ft_test_st_bf16.safetensors");
        let _ = std::fs::remove_file(&path);
        let mut sd = BTreeMap::new();
        let vals: Vec<ft_core::BFloat16> = vec![1.0f32, 2.0, 3.0]
            .into_iter()
            .map(ft_core::BFloat16::from_f32)
            .collect();
        let meta = TensorMeta::from_shape(vec![3], DType::BF16, Device::Cpu);
        let t = DenseTensor::from_storage_bf16(meta, vals).unwrap();
        sd.insert("b".to_string(), t);

        save_safetensors(&sd, &path, None).unwrap();
        let loaded = load_safetensors(&path).unwrap();
        let _ = std::fs::remove_file(&path);

        assert_eq!(loaded["b"].meta().dtype(), DType::BF16);
        let loaded_f32: Vec<f32> = loaded["b"].typed_storage().to_f32_vec();
        assert_eq!(loaded_f32, vec![1.0f32, 2.0, 3.0]);
    }

    #[test]
    fn safetensors_empty_state_dict() {
        let path = std::env::temp_dir().join("ft_test_st_empty.safetensors");
        let _ = std::fs::remove_file(&path);
        let sd: BTreeMap<String, DenseTensor> = BTreeMap::new();

        save_safetensors(&sd, &path, None).unwrap();
        let loaded = load_safetensors(&path).unwrap();
        let _ = std::fs::remove_file(&path);

        assert!(loaded.is_empty());
    }

    #[test]
    fn safetensors_multiple_tensors() {
        let path = std::env::temp_dir().join("ft_test_st_multi.safetensors");
        let _ = std::fs::remove_file(&path);
        let mut sd = BTreeMap::new();
        sd.insert(
            "layer1.weight".to_string(),
            make_f64_tensor(vec![1.0, 2.0], vec![2]),
        );
        sd.insert(
            "layer1.bias".to_string(),
            make_f64_tensor(vec![0.5], vec![1]),
        );
        sd.insert(
            "layer2.weight".to_string(),
            make_f64_tensor(vec![3.0, 4.0, 5.0, 6.0], vec![2, 2]),
        );

        save_safetensors(&sd, &path, None).unwrap();
        let loaded = load_safetensors(&path).unwrap();
        let _ = std::fs::remove_file(&path);

        assert_eq!(loaded.len(), 3);
        assert_eq!(
            loaded["layer1.weight"].contiguous_values().unwrap(),
            &[1.0, 2.0]
        );
        assert_eq!(loaded["layer1.bias"].contiguous_values().unwrap(), &[0.5]);
        assert_eq!(
            loaded["layer2.weight"].contiguous_values().unwrap(),
            &[3.0, 4.0, 5.0, 6.0]
        );
        assert_eq!(loaded["layer2.weight"].meta().shape(), &[2, 2]);
    }

    #[test]
    fn safetensors_with_metadata() {
        let path = std::env::temp_dir().join("ft_test_st_meta.safetensors");
        let _ = std::fs::remove_file(&path);
        let mut sd = BTreeMap::new();
        sd.insert("x".to_string(), make_f64_tensor(vec![1.0], vec![1]));

        let mut meta = std::collections::HashMap::new();
        meta.insert("format".to_string(), "ft".to_string());
        meta.insert("version".to_string(), "1".to_string());

        save_safetensors(&sd, &path, Some(&meta)).unwrap();
        let loaded_meta = load_safetensors_metadata(&path).unwrap();
        let _ = std::fs::remove_file(&path);

        let loaded_meta = loaded_meta.expect("metadata should be present");
        assert_eq!(loaded_meta.get("format").unwrap(), "ft");
        assert_eq!(loaded_meta.get("version").unwrap(), "1");
    }

    #[test]
    fn safetensors_no_metadata() {
        let path = std::env::temp_dir().join("ft_test_st_nometa.safetensors");
        let _ = std::fs::remove_file(&path);
        let sd: BTreeMap<String, DenseTensor> = BTreeMap::new();

        save_safetensors(&sd, &path, None).unwrap();
        let loaded_meta = load_safetensors_metadata(&path).unwrap();
        let _ = std::fs::remove_file(&path);

        assert!(loaded_meta.is_none());
    }

    #[test]
    fn safetensors_from_bytes_round_trip() {
        let mut sd = BTreeMap::new();
        sd.insert("a".to_string(), make_f64_tensor(vec![1.0, 2.0], vec![2]));

        let path = std::env::temp_dir().join("ft_test_st_bytes.safetensors");
        let _ = std::fs::remove_file(&path);
        save_safetensors(&sd, &path, None).unwrap();
        let bytes = std::fs::read(&path).unwrap();
        let _ = std::fs::remove_file(&path);

        let loaded = load_safetensors_from_bytes(&bytes).unwrap();
        assert_eq!(loaded["a"].contiguous_values().unwrap(), &[1.0, 2.0]);
    }

    #[test]
    fn safetensors_corrupt_bytes() {
        let result = load_safetensors_from_bytes(b"not a safetensors file");
        assert!(result.is_err());
    }

    fn malformed_safetensors_bytes(dtype: &str, shape: &[usize], raw_payload: &[u8]) -> Vec<u8> {
        let header = json!({
            "bad": {
                "dtype": dtype,
                "shape": shape,
                "data_offsets": [0, raw_payload.len()],
            }
        });
        let header_bytes = serde_json::to_vec(&header).expect("header must encode");
        let mut bytes = Vec::with_capacity(8 + header_bytes.len() + raw_payload.len());
        bytes.extend_from_slice(&(header_bytes.len() as u64).to_le_bytes());
        bytes.extend_from_slice(&header_bytes);
        bytes.extend_from_slice(raw_payload);
        bytes
    }

    #[test]
    fn safetensors_rejects_f32_payload_with_extra_trailing_byte() {
        let bytes = malformed_safetensors_bytes("F32", &[1], &[0, 0, 128, 63, 255]);

        let err = load_safetensors_from_bytes(&bytes).expect_err("invalid width must fail");
        let msg = err.to_string();

        assert!(matches!(err, TensorIOError::Corrupt { .. }));
        assert!(
            msg.contains("invalid payload width") || msg.contains("safetensors"),
            "unexpected error: {msg}"
        );
    }

    #[test]
    fn safetensors_rejects_f16_payload_with_odd_byte_count() {
        let bytes = malformed_safetensors_bytes("F16", &[1], &[0, 60, 1]);

        let err = load_safetensors_from_bytes(&bytes).expect_err("invalid width must fail");
        let msg = err.to_string();

        assert!(matches!(err, TensorIOError::Corrupt { .. }));
        assert!(
            msg.contains("invalid payload width") || msg.contains("safetensors"),
            "unexpected error: {msg}"
        );
    }

    #[test]
    fn safetensors_rejects_i64_dtype_for_dense_tensor_loader() {
        let bytes = malformed_safetensors_bytes("I64", &[1], &42_i64.to_le_bytes());

        let err = load_safetensors_from_bytes(&bytes).expect_err("i64 dtype must fail");
        let msg = err.to_string();

        assert!(matches!(err, TensorIOError::Corrupt { .. }));
        assert!(
            msg.contains("integer/bool SafeTensors dtype") || msg.contains("unsupported"),
            "unexpected error: {msg}"
        );
    }

    #[test]
    fn safetensors_rejects_bool_dtype_for_dense_tensor_loader() {
        let bytes = malformed_safetensors_bytes("BOOL", &[1], &[1]);

        let err = load_safetensors_from_bytes(&bytes).expect_err("bool dtype must fail");
        let msg = err.to_string();

        assert!(matches!(err, TensorIOError::Corrupt { .. }));
        assert!(
            msg.contains("integer/bool SafeTensors dtype") || msg.contains("unsupported"),
            "unexpected error: {msg}"
        );
    }

    // ── frankentorch-u0p: Audit edge cases ───────────────────────��─────

    #[test]
    fn native_format_rejects_shape_overflow() {
        // Craft a file with shape dimensions that overflow when multiplied:
        // shape = [usize::MAX, 2] => product overflows
        let mut data = Vec::new();
        data.extend_from_slice(b"FTSV");
        data.extend_from_slice(&1u32.to_le_bytes()); // version
        data.extend_from_slice(&1u64.to_le_bytes()); // num_tensors = 1
        // key: "x"
        data.extend_from_slice(&1u64.to_le_bytes());
        data.push(b'x');
        // ndim = 2
        data.extend_from_slice(&2u64.to_le_bytes());
        // shape[0] = usize::MAX, shape[1] = 2
        data.extend_from_slice(&(usize::MAX as u64).to_le_bytes());
        data.extend_from_slice(&2u64.to_le_bytes());
        // dtype tag = 0 (F64)
        data.push(0);

        let err = load_state_dict_from_bytes(&data).expect_err("shape overflow must fail");
        assert!(matches!(err, TensorIOError::Corrupt { .. }));
        assert!(err.to_string().contains("shape overflow"));
    }

    #[test]
    fn native_format_rejects_excessive_tensor_count() {
        // Craft a file claiming billions of tensors with only a few bytes remaining
        let mut data = Vec::new();
        data.extend_from_slice(b"FTSV");
        data.extend_from_slice(&1u32.to_le_bytes()); // version
        data.extend_from_slice(&u64::MAX.to_le_bytes()); // absurd num_tensors

        let err = load_state_dict_from_bytes(&data).expect_err("excessive count must fail");
        assert!(matches!(err, TensorIOError::Corrupt { .. }));
        assert!(err.to_string().contains("tensors but only"));
    }

    #[test]
    fn native_format_rejects_byte_count_overflow_f64() {
        // Craft a shape that doesn't overflow itself but numel * 8 does
        // shape = [usize::MAX / 4] => numel = usize::MAX/4, numel * 8 overflows
        let mut data = Vec::new();
        data.extend_from_slice(b"FTSV");
        data.extend_from_slice(&1u32.to_le_bytes()); // version
        data.extend_from_slice(&1u64.to_le_bytes()); // num_tensors = 1
        // key: "y"
        data.extend_from_slice(&1u64.to_le_bytes());
        data.push(b'y');
        // ndim = 1
        data.extend_from_slice(&1u64.to_le_bytes());
        // shape[0] = usize::MAX / 4 (won't overflow in shape product but numel*8 will)
        let big_dim = usize::MAX / 4;
        data.extend_from_slice(&(big_dim as u64).to_le_bytes());
        // dtype tag = 0 (F64)
        data.push(0);

        let err = load_state_dict_from_bytes(&data).expect_err("byte overflow must fail");
        assert!(matches!(err, TensorIOError::Corrupt { .. }));
        assert!(err.to_string().contains("overflow"));
    }

    #[test]
    fn native_format_round_trip_zero_element_tensor() {
        let dir = std::env::temp_dir().join("ft_test_zero_elem");
        let _ = std::fs::remove_file(&dir);
        let mut sd = BTreeMap::new();
        sd.insert("empty".to_string(), make_f64_tensor(vec![], vec![0]));

        save_state_dict(&sd, &dir).unwrap();
        let loaded = load_state_dict(&dir).unwrap();
        let _ = std::fs::remove_file(&dir);

        assert_eq!(loaded.len(), 1);
        assert_eq!(loaded["empty"].meta().shape(), &[0]);
        assert!(loaded["empty"].contiguous_values().unwrap().is_empty());
    }

    #[test]
    fn native_format_round_trip_high_dimensional_tensor() {
        let dir = std::env::temp_dir().join("ft_test_high_dim");
        let _ = std::fs::remove_file(&dir);
        let mut sd = BTreeMap::new();
        // 5D tensor: [1, 1, 1, 1, 2]
        sd.insert(
            "deep".to_string(),
            make_f64_tensor(vec![7.0, 8.0], vec![1, 1, 1, 1, 2]),
        );

        save_state_dict(&sd, &dir).unwrap();
        let loaded = load_state_dict(&dir).unwrap();
        let _ = std::fs::remove_file(&dir);

        assert_eq!(loaded["deep"].meta().shape(), &[1, 1, 1, 1, 2]);
        assert_eq!(loaded["deep"].contiguous_values().unwrap(), &[7.0, 8.0]);
    }
}

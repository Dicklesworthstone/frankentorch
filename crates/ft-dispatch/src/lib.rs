#![forbid(unsafe_code)]

use std::fmt;

use ft_core::{Device, ExecutionMode, ScalarTensor, TensorMeta};
use ft_kernel_cpu::{
    KernelError, add_scalar, add_tensor_contiguous_f64, div_scalar, div_tensor_contiguous_f64,
    mul_scalar, mul_tensor_contiguous_f64, sub_scalar, sub_tensor_contiguous_f64,
};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BinaryOp {
    Add,
    Sub,
    Div,
    Mul,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(u8)]
pub enum DispatchKey {
    Undefined = 0,
    BackendSelect = 1,
    CompositeImplicitAutograd = 2,
    CompositeExplicitAutograd = 3,
    CPU = 4,
    AutogradCPU = 5,
}

impl DispatchKey {
    #[must_use]
    pub const fn all() -> &'static [DispatchKey] {
        &[
            DispatchKey::BackendSelect,
            DispatchKey::CompositeImplicitAutograd,
            DispatchKey::CompositeExplicitAutograd,
            DispatchKey::CPU,
            DispatchKey::AutogradCPU,
        ]
    }

    #[must_use]
    pub const fn bit(self) -> u64 {
        1u64 << (self as u8)
    }
}

const TYPE_PRIORITY: [DispatchKey; 5] = [
    DispatchKey::AutogradCPU,
    DispatchKey::CompositeExplicitAutograd,
    DispatchKey::CompositeImplicitAutograd,
    DispatchKey::CPU,
    DispatchKey::BackendSelect,
];

const BACKEND_PRIORITY: [DispatchKey; 1] = [DispatchKey::CPU];

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub struct DispatchKeySet {
    bits: u64,
}

impl DispatchKeySet {
    #[must_use]
    pub const fn empty() -> Self {
        Self { bits: 0 }
    }

    #[must_use]
    pub fn from_keys(keys: &[DispatchKey]) -> Self {
        let mut out = Self::empty();
        for key in keys {
            out.add(*key);
        }
        out
    }

    pub fn from_bits_checked(bits: u64) -> Result<Self, DispatchKeyError> {
        let known_mask = DispatchKey::all()
            .iter()
            .fold(0u64, |mask, key| mask | key.bit());
        let unknown = bits & !known_mask;
        if unknown != 0 {
            return Err(DispatchKeyError::UnknownBits {
                unknown_mask: unknown,
            });
        }
        Ok(Self { bits })
    }

    #[must_use]
    pub const fn bits(self) -> u64 {
        self.bits
    }

    #[must_use]
    pub const fn is_empty(self) -> bool {
        self.bits == 0
    }

    pub fn add(&mut self, key: DispatchKey) {
        self.bits |= key.bit();
    }

    pub fn remove(&mut self, key: DispatchKey) {
        self.bits &= !key.bit();
    }

    #[must_use]
    pub const fn has(self, key: DispatchKey) -> bool {
        (self.bits & key.bit()) != 0
    }

    #[must_use]
    pub const fn union(self, other: Self) -> Self {
        Self {
            bits: self.bits | other.bits,
        }
    }

    #[must_use]
    pub const fn intersection(self, other: Self) -> Self {
        Self {
            bits: self.bits & other.bits,
        }
    }

    pub fn highest_priority_type_id(self) -> Result<DispatchKey, DispatchKeyError> {
        if self.is_empty() {
            return Err(DispatchKeyError::EmptySet);
        }
        TYPE_PRIORITY
            .iter()
            .find(|&&key| self.has(key))
            .copied()
            .ok_or(DispatchKeyError::NoTypeKey)
    }

    pub fn highest_priority_backend_type_id(self) -> Result<DispatchKey, DispatchKeyError> {
        if self.is_empty() {
            return Err(DispatchKeyError::EmptySet);
        }
        BACKEND_PRIORITY
            .iter()
            .find(|&&key| self.has(key))
            .copied()
            .ok_or(DispatchKeyError::NoBackendKey)
    }

    pub fn validate_for_scalar_binary(self) -> Result<(), DispatchKeyError> {
        if self.is_empty() {
            return Err(DispatchKeyError::EmptySet);
        }
        if self.has(DispatchKey::AutogradCPU) && !self.has(DispatchKey::CPU) {
            return Err(DispatchKeyError::IncompatibleSet {
                reason: "AutogradCPU requires CPU backend availability",
            });
        }
        self.highest_priority_type_id()?;
        self.highest_priority_backend_type_id()?;
        Ok(())
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DispatchKeyError {
    EmptySet,
    NoTypeKey,
    NoBackendKey,
    UnknownBits { unknown_mask: u64 },
    IncompatibleSet { reason: &'static str },
}

impl fmt::Display for DispatchKeyError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::EmptySet => write!(f, "dispatch keyset is empty"),
            Self::NoTypeKey => write!(f, "dispatch keyset has no resolvable type key"),
            Self::NoBackendKey => write!(f, "dispatch keyset has no backend key"),
            Self::UnknownBits { unknown_mask } => {
                write!(
                    f,
                    "dispatch keyset has unknown bitmask 0x{unknown_mask:016x}"
                )
            }
            Self::IncompatibleSet { reason } => {
                write!(f, "incompatible dispatch keyset: {reason}")
            }
        }
    }
}

impl std::error::Error for DispatchKeyError {}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DispatchDecision {
    pub op: BinaryOp,
    pub mode: ExecutionMode,
    pub kernel: &'static str,
    pub selected_key: DispatchKey,
    pub backend_key: DispatchKey,
    pub keyset_bits: u64,
    pub fallback_used: bool,
}

#[derive(Debug, Clone, PartialEq)]
pub struct DispatchOutcome {
    pub tensor: ScalarTensor,
    pub decision: DispatchDecision,
}

#[derive(Debug, Clone, PartialEq)]
pub struct TensorDispatchOutcome {
    pub values: Vec<f64>,
    pub decision: DispatchDecision,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum DispatchError {
    Kernel(KernelError),
    Key(DispatchKeyError),
}

impl fmt::Display for DispatchError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Kernel(error) => write!(f, "kernel dispatch failure: {error}"),
            Self::Key(error) => write!(f, "dispatch key failure: {error}"),
        }
    }
}

impl std::error::Error for DispatchError {}

impl From<KernelError> for DispatchError {
    fn from(value: KernelError) -> Self {
        Self::Kernel(value)
    }
}

impl From<DispatchKeyError> for DispatchError {
    fn from(value: DispatchKeyError) -> Self {
        Self::Key(value)
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum OpSchemaError {
    EmptyInput,
    InvalidOperatorName { reason: &'static str },
    InvalidOverloadName { overload: String },
    MalformedSchema { reason: &'static str },
    UnknownDispatchKey { key: String },
    IncompatibleDispatchKeyset(DispatchKeyError),
}

impl fmt::Display for OpSchemaError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::EmptyInput => write!(f, "schema input is empty"),
            Self::InvalidOperatorName { reason } => write!(f, "invalid operator name: {reason}"),
            Self::InvalidOverloadName { overload } => {
                write!(f, "invalid overload name: {overload}")
            }
            Self::MalformedSchema { reason } => write!(f, "malformed schema: {reason}"),
            Self::UnknownDispatchKey { key } => {
                write!(f, "unknown schema dispatch key '{key}'")
            }
            Self::IncompatibleDispatchKeyset(error) => {
                write!(f, "incompatible schema dispatch keyset: {error}")
            }
        }
    }
}

impl std::error::Error for OpSchemaError {}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct OpSchemaName {
    pub base: String,
    pub overload: Option<String>,
    pub is_inplace: bool,
}

impl OpSchemaName {
    #[must_use]
    pub fn unambiguous_name(&self) -> String {
        match &self.overload {
            Some(overload) => format!("{}_{}", self.base, overload),
            None => self.base.clone(),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ParsedOpSchema {
    pub op: OpSchemaName,
    pub arguments: String,
    pub returns: String,
    pub is_out_variant: bool,
    pub schema_digest: u64,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ParsedSchemaInput {
    Name(OpSchemaName),
    Schema(ParsedOpSchema),
}

fn is_valid_ident(value: &str) -> bool {
    let mut chars = value.chars();
    match chars.next() {
        Some(first) if first.is_ascii_alphabetic() || first == '_' => {}
        _ => return false,
    }
    chars.all(|ch| ch.is_ascii_alphanumeric() || ch == '_')
}

fn is_reserved_overload_name(value: &str) -> bool {
    matches!(value, "default") || value.starts_with("__")
}

fn digest64(input: &str) -> u64 {
    let mut hash = 0xcbf2_9ce4_8422_2325u64;
    for byte in input.as_bytes() {
        hash ^= u64::from(*byte);
        hash = hash.wrapping_mul(0x0000_0100_0000_01b3);
    }
    hash
}

pub fn parse_schema_name(input: &str) -> Result<OpSchemaName, OpSchemaError> {
    let trimmed = input.trim();
    if trimmed.is_empty() {
        return Err(OpSchemaError::EmptyInput);
    }

    let op_without_namespace = match trimmed.rsplit_once("::") {
        Some((_, op)) => op,
        None => trimmed,
    };
    if op_without_namespace.is_empty() {
        return Err(OpSchemaError::InvalidOperatorName {
            reason: "missing operator token after namespace",
        });
    }

    let (raw_base, overload) = match op_without_namespace.split_once('.') {
        Some((name, overload_name)) => {
            if overload_name.is_empty() {
                return Err(OpSchemaError::InvalidOverloadName {
                    overload: overload_name.to_string(),
                });
            }
            if is_reserved_overload_name(overload_name) {
                return Err(OpSchemaError::InvalidOverloadName {
                    overload: overload_name.to_string(),
                });
            }
            if !is_valid_ident(overload_name) {
                return Err(OpSchemaError::InvalidOverloadName {
                    overload: overload_name.to_string(),
                });
            }
            (name, Some(overload_name.to_string()))
        }
        None => (op_without_namespace, None),
    };

    if raw_base.is_empty() {
        return Err(OpSchemaError::InvalidOperatorName {
            reason: "empty base operator name",
        });
    }

    let is_inplace = raw_base.ends_with('_');
    let base = if is_inplace {
        raw_base.trim_end_matches('_')
    } else {
        raw_base
    };
    if base.is_empty() || !is_valid_ident(base) {
        return Err(OpSchemaError::InvalidOperatorName {
            reason: "operator base contains invalid characters",
        });
    }

    Ok(OpSchemaName {
        base: base.to_string(),
        overload,
        is_inplace,
    })
}

pub fn parse_schema_or_name(input: &str) -> Result<ParsedSchemaInput, OpSchemaError> {
    let trimmed = input.trim();
    if trimmed.is_empty() {
        return Err(OpSchemaError::EmptyInput);
    }

    if let Some(open_idx) = trimmed.find('(') {
        let arrow_idx = trimmed
            .rfind(") -> ")
            .ok_or(OpSchemaError::MalformedSchema {
                reason: "expected ') ->' separator",
            })?;
        if arrow_idx < open_idx {
            return Err(OpSchemaError::MalformedSchema {
                reason: "found return separator before argument list",
            });
        }
        let op_name = parse_schema_name(&trimmed[..open_idx])?;
        let args = trimmed[open_idx + 1..arrow_idx].trim();
        let returns = trimmed[arrow_idx + 5..].trim();
        if returns.is_empty() {
            return Err(OpSchemaError::MalformedSchema {
                reason: "missing return declaration",
            });
        }
        let is_out_variant = op_name.overload.as_deref() == Some("out")
            || args.contains("Tensor(a!) out")
            || args.contains("Tensor(a!) out)");

        return Ok(ParsedSchemaInput::Schema(ParsedOpSchema {
            op: op_name,
            arguments: args.to_string(),
            returns: returns.to_string(),
            is_out_variant,
            schema_digest: digest64(trimmed),
        }));
    }

    Ok(ParsedSchemaInput::Name(parse_schema_name(trimmed)?))
}

pub fn schema_dispatch_key_from_tag(tag: &str) -> Result<DispatchKey, OpSchemaError> {
    match tag {
        "BackendSelect" => Ok(DispatchKey::BackendSelect),
        "CompositeImplicitAutograd" => Ok(DispatchKey::CompositeImplicitAutograd),
        "CompositeExplicitAutograd" => Ok(DispatchKey::CompositeExplicitAutograd),
        "CPU" => Ok(DispatchKey::CPU),
        "AutogradCPU" => Ok(DispatchKey::AutogradCPU),
        _ => Err(OpSchemaError::UnknownDispatchKey {
            key: tag.to_string(),
        }),
    }
}

pub fn schema_dispatch_keyset_from_tags(tags: &[&str]) -> Result<DispatchKeySet, OpSchemaError> {
    let mut keyset = DispatchKeySet::empty();
    for tag in tags {
        keyset.add(schema_dispatch_key_from_tag(tag)?);
    }
    keyset
        .validate_for_scalar_binary()
        .map_err(OpSchemaError::IncompatibleDispatchKeyset)?;
    Ok(keyset)
}

#[must_use]
fn dispatch_keyset_for_device(device: Device, requires_grad: bool) -> DispatchKeySet {
    let mut keyset = DispatchKeySet::empty();
    keyset.add(DispatchKey::BackendSelect);
    if device == Device::Cpu {
        keyset.add(DispatchKey::CPU);
    }
    if requires_grad {
        keyset.add(DispatchKey::AutogradCPU);
    }
    keyset
}

#[must_use]
pub fn dispatch_keyset_for_tensors(
    lhs: &ScalarTensor,
    _rhs: &ScalarTensor,
    requires_grad: bool,
) -> DispatchKeySet {
    dispatch_keyset_for_device(lhs.meta().device(), requires_grad)
}

#[must_use]
pub fn dispatch_keyset_for_tensor_meta(
    lhs: &TensorMeta,
    _rhs: &TensorMeta,
    requires_grad: bool,
) -> DispatchKeySet {
    dispatch_keyset_for_device(lhs.device(), requires_grad)
}

fn resolve_dispatch_keys(
    mode: ExecutionMode,
    keyset: DispatchKeySet,
) -> Result<(DispatchKey, DispatchKey, DispatchKey, bool), DispatchError> {
    keyset.validate_for_scalar_binary()?;
    let selected_key = keyset.highest_priority_type_id()?;
    let backend_key = keyset.highest_priority_backend_type_id()?;

    let (effective_key, fallback_used) = match selected_key {
        DispatchKey::AutogradCPU | DispatchKey::CPU => (selected_key, false),
        DispatchKey::CompositeExplicitAutograd
        | DispatchKey::CompositeImplicitAutograd
        | DispatchKey::BackendSelect => match mode {
            ExecutionMode::Strict => {
                return Err(DispatchKeyError::IncompatibleSet {
                    reason: "strict mode forbids composite/backend fallback routing",
                }
                .into());
            }
            ExecutionMode::Hardened => (backend_key, true),
        },
        DispatchKey::Undefined => return Err(DispatchKeyError::NoTypeKey.into()),
    };

    if effective_key != backend_key && effective_key != DispatchKey::AutogradCPU {
        return Err(DispatchKeyError::IncompatibleSet {
            reason: "resolved key/backend key drifted to incompatible pair",
        }
        .into());
    }

    Ok((selected_key, backend_key, effective_key, fallback_used))
}

pub fn dispatch_scalar_binary(
    op: BinaryOp,
    mode: ExecutionMode,
    lhs: &ScalarTensor,
    rhs: &ScalarTensor,
    requires_grad: bool,
) -> Result<DispatchOutcome, DispatchError> {
    let keyset = dispatch_keyset_for_tensors(lhs, rhs, requires_grad);
    dispatch_scalar_binary_with_keyset(op, mode, lhs, rhs, keyset)
}

pub fn dispatch_scalar_binary_with_keyset(
    op: BinaryOp,
    mode: ExecutionMode,
    lhs: &ScalarTensor,
    rhs: &ScalarTensor,
    keyset: DispatchKeySet,
) -> Result<DispatchOutcome, DispatchError> {
    let (selected_key, backend_key, effective_key, fallback_used) =
        resolve_dispatch_keys(mode, keyset)?;

    let (tensor, kernel) = match (effective_key, op) {
        (DispatchKey::AutogradCPU, BinaryOp::Add) => {
            (add_scalar(lhs, rhs)?, "autograd_cpu::add_scalar")
        }
        (DispatchKey::AutogradCPU, BinaryOp::Sub) => {
            (sub_scalar(lhs, rhs)?, "autograd_cpu::sub_scalar")
        }
        (DispatchKey::AutogradCPU, BinaryOp::Div) => {
            (div_scalar(lhs, rhs)?, "autograd_cpu::div_scalar")
        }
        (DispatchKey::AutogradCPU, BinaryOp::Mul) => {
            (mul_scalar(lhs, rhs)?, "autograd_cpu::mul_scalar")
        }
        (DispatchKey::CPU, BinaryOp::Add) => (add_scalar(lhs, rhs)?, "cpu::add_scalar"),
        (DispatchKey::CPU, BinaryOp::Sub) => (sub_scalar(lhs, rhs)?, "cpu::sub_scalar"),
        (DispatchKey::CPU, BinaryOp::Div) => (div_scalar(lhs, rhs)?, "cpu::div_scalar"),
        (DispatchKey::CPU, BinaryOp::Mul) => (mul_scalar(lhs, rhs)?, "cpu::mul_scalar"),
        _ => {
            return Err(DispatchKeyError::IncompatibleSet {
                reason: "resolved dispatch key is unsupported for scalar binary ops",
            }
            .into());
        }
    };

    Ok(DispatchOutcome {
        tensor,
        decision: DispatchDecision {
            op,
            mode,
            kernel,
            selected_key,
            backend_key,
            keyset_bits: keyset.bits(),
            fallback_used,
        },
    })
}

pub fn dispatch_tensor_binary_contiguous_f64(
    op: BinaryOp,
    mode: ExecutionMode,
    lhs: &[f64],
    rhs: &[f64],
    lhs_meta: &TensorMeta,
    rhs_meta: &TensorMeta,
    requires_grad: bool,
) -> Result<TensorDispatchOutcome, DispatchError> {
    let keyset = dispatch_keyset_for_tensor_meta(lhs_meta, rhs_meta, requires_grad);
    dispatch_tensor_binary_contiguous_f64_with_keyset(
        op, mode, lhs, rhs, lhs_meta, rhs_meta, keyset,
    )
}

pub fn dispatch_tensor_binary_contiguous_f64_with_keyset(
    op: BinaryOp,
    mode: ExecutionMode,
    lhs: &[f64],
    rhs: &[f64],
    lhs_meta: &TensorMeta,
    rhs_meta: &TensorMeta,
    keyset: DispatchKeySet,
) -> Result<TensorDispatchOutcome, DispatchError> {
    let (selected_key, backend_key, effective_key, fallback_used) =
        resolve_dispatch_keys(mode, keyset)?;

    let (values, kernel) = match (effective_key, op) {
        (DispatchKey::AutogradCPU, BinaryOp::Add) => (
            add_tensor_contiguous_f64(lhs, rhs, lhs_meta, rhs_meta)?,
            "autograd_cpu::add_tensor_contiguous_f64",
        ),
        (DispatchKey::AutogradCPU, BinaryOp::Sub) => (
            sub_tensor_contiguous_f64(lhs, rhs, lhs_meta, rhs_meta)?,
            "autograd_cpu::sub_tensor_contiguous_f64",
        ),
        (DispatchKey::AutogradCPU, BinaryOp::Div) => (
            div_tensor_contiguous_f64(lhs, rhs, lhs_meta, rhs_meta)?,
            "autograd_cpu::div_tensor_contiguous_f64",
        ),
        (DispatchKey::AutogradCPU, BinaryOp::Mul) => (
            mul_tensor_contiguous_f64(lhs, rhs, lhs_meta, rhs_meta)?,
            "autograd_cpu::mul_tensor_contiguous_f64",
        ),
        (DispatchKey::CPU, BinaryOp::Add) => (
            add_tensor_contiguous_f64(lhs, rhs, lhs_meta, rhs_meta)?,
            "cpu::add_tensor_contiguous_f64",
        ),
        (DispatchKey::CPU, BinaryOp::Sub) => (
            sub_tensor_contiguous_f64(lhs, rhs, lhs_meta, rhs_meta)?,
            "cpu::sub_tensor_contiguous_f64",
        ),
        (DispatchKey::CPU, BinaryOp::Div) => (
            div_tensor_contiguous_f64(lhs, rhs, lhs_meta, rhs_meta)?,
            "cpu::div_tensor_contiguous_f64",
        ),
        (DispatchKey::CPU, BinaryOp::Mul) => (
            mul_tensor_contiguous_f64(lhs, rhs, lhs_meta, rhs_meta)?,
            "cpu::mul_tensor_contiguous_f64",
        ),
        _ => {
            return Err(DispatchKeyError::IncompatibleSet {
                reason: "resolved dispatch key is unsupported for contiguous tensor binary ops",
            }
            .into());
        }
    };

    Ok(TensorDispatchOutcome {
        values,
        decision: DispatchDecision {
            op,
            mode,
            kernel,
            selected_key,
            backend_key,
            keyset_bits: keyset.bits(),
            fallback_used,
        },
    })
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeMap;

    use ft_core::{DType, Device, ExecutionMode, ScalarTensor, TensorCompatError, TensorMeta};
    use ft_kernel_cpu::KernelError;
    use proptest::prelude::*;

    use super::{
        BinaryOp, DispatchError, DispatchKey, DispatchKeyError, DispatchKeySet, OpSchemaError,
        ParsedSchemaInput, TYPE_PRIORITY, dispatch_keyset_for_tensor_meta,
        dispatch_keyset_for_tensors, dispatch_scalar_binary, dispatch_scalar_binary_with_keyset,
        dispatch_tensor_binary_contiguous_f64, dispatch_tensor_binary_contiguous_f64_with_keyset,
        parse_schema_name, parse_schema_or_name, schema_dispatch_keyset_from_tags,
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

    fn build_property_log(
        test_id: &str,
        mode: &str,
        seed: u64,
        input_digest: u64,
        output_digest: u64,
        reason_code: &str,
    ) -> BTreeMap<String, String> {
        let mut log = BTreeMap::new();
        let scenario_id = format!("dispatch_key/{mode}:{test_id}");
        log.insert("ts_utc".to_string(), "1970-01-01T00:00:00Z".to_string());
        log.insert("suite_id".to_string(), "ft_dispatch_property".to_string());
        log.insert("test_id".to_string(), test_id.to_string());
        log.insert("packet_id".to_string(), "FT-P2C-002".to_string());
        log.insert(
            "fixture_id".to_string(),
            "ft_dispatch_property_generated".to_string(),
        );
        log.insert("scenario_id".to_string(), scenario_id);
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
            "det64:ft-dispatch-test".to_string(),
        );
        log.insert(
            "artifact_refs".to_string(),
            "artifacts/phase2c/FT-P2C-002/fixture_manifest.json".to_string(),
        );
        log.insert(
            "replay_command".to_string(),
            format!("cargo test -p ft-dispatch {test_id} -- --nocapture"),
        );
        log.insert("duration_ms".to_string(), "0".to_string());
        log.insert("outcome".to_string(), "pass".to_string());
        log.insert("contract_id".to_string(), reason_code.to_string());
        log.insert("shrink_trace".to_string(), "none".to_string());
        log.insert("reason_code".to_string(), reason_code.to_string());
        log
    }

    fn build_packet_007_property_log(
        test_id: &str,
        mode: &str,
        seed: u64,
        input_digest: u64,
        output_digest: u64,
        reason_code: &str,
        scenario_id: &str,
    ) -> BTreeMap<String, String> {
        let mut log = build_property_log(
            test_id,
            mode,
            seed,
            input_digest,
            output_digest,
            reason_code,
        );
        log.insert("packet_id".to_string(), "FT-P2C-007".to_string());
        log.insert("scenario_id".to_string(), scenario_id.to_string());
        log.insert(
            "artifact_refs".to_string(),
            "artifacts/phase2c/FT-P2C-007/contract_table.md".to_string(),
        );
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

    struct SchemaLogParams<'a> {
        test_id: &'a str,
        mode: &'a str,
        seed: u64,
        op_name: &'a str,
        overload_name: &'a str,
        schema_digest: u64,
        dispatch_keyset_bits: u64,
        reason_code: &'a str,
    }

    fn build_schema_property_log(params: SchemaLogParams<'_>) -> BTreeMap<String, String> {
        let mut log = BTreeMap::new();
        let scenario_id = format!("op_schema/{}:{}", params.mode, params.test_id);
        log.insert("ts_utc".to_string(), "1970-01-01T00:00:00Z".to_string());
        log.insert(
            "suite_id".to_string(),
            "ft_dispatch_schema_property".to_string(),
        );
        log.insert("test_id".to_string(), params.test_id.to_string());
        log.insert("packet_id".to_string(), "FT-P2C-003".to_string());
        log.insert(
            "fixture_id".to_string(),
            "ft_dispatch_schema_property_generated".to_string(),
        );
        log.insert("scenario_id".to_string(), scenario_id);
        log.insert("mode".to_string(), params.mode.to_string());
        log.insert("seed".to_string(), params.seed.to_string());
        log.insert("op_name".to_string(), params.op_name.to_string());
        log.insert(
            "overload_name".to_string(),
            params.overload_name.to_string(),
        );
        log.insert(
            "schema_digest".to_string(),
            format!("det64:{:016x}", params.schema_digest),
        );
        log.insert(
            "dispatch_keyset_bits".to_string(),
            format!("0x{:016x}", params.dispatch_keyset_bits),
        );
        log.insert(
            "env_fingerprint".to_string(),
            "det64:ft-dispatch-schema-test".to_string(),
        );
        log.insert(
            "artifact_refs".to_string(),
            "artifacts/phase2c/FT-P2C-003/contract_table.md".to_string(),
        );
        log.insert(
            "replay_command".to_string(),
            format!(
                "cargo test -p ft-dispatch {} -- --nocapture",
                params.test_id
            ),
        );
        log.insert("duration_ms".to_string(), "0".to_string());
        log.insert("outcome".to_string(), "pass".to_string());
        log.insert("contract_id".to_string(), params.reason_code.to_string());
        log.insert("shrink_trace".to_string(), "none".to_string());
        log.insert("reason_code".to_string(), params.reason_code.to_string());
        log
    }

    fn assert_schema_log_contract(log: &BTreeMap<String, String>) {
        for key in [
            "ts_utc",
            "suite_id",
            "test_id",
            "packet_id",
            "fixture_id",
            "scenario_id",
            "mode",
            "seed",
            "op_name",
            "overload_name",
            "schema_digest",
            "dispatch_keyset_bits",
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
                "schema property log missing required key '{key}'"
            );
        }
    }

    #[test]
    fn dispatch_keyset_set_algebra_is_stable() {
        let mut left = DispatchKeySet::from_keys(&[DispatchKey::CPU, DispatchKey::BackendSelect]);
        let right = DispatchKeySet::from_keys(&[DispatchKey::AutogradCPU, DispatchKey::CPU]);

        let union = left.union(right);
        assert!(union.has(DispatchKey::CPU));
        assert!(union.has(DispatchKey::AutogradCPU));
        assert!(union.has(DispatchKey::BackendSelect));

        let intersection = left.intersection(right);
        assert!(intersection.has(DispatchKey::CPU));
        assert!(!intersection.has(DispatchKey::AutogradCPU));

        left.remove(DispatchKey::BackendSelect);
        assert!(!left.has(DispatchKey::BackendSelect));
    }

    #[test]
    fn priority_resolution_prefers_autograd_cpu() {
        let keys = DispatchKeySet::from_keys(&[
            DispatchKey::BackendSelect,
            DispatchKey::CPU,
            DispatchKey::AutogradCPU,
        ]);
        let selected = keys
            .highest_priority_type_id()
            .expect("priority resolution should succeed");
        assert_eq!(selected, DispatchKey::AutogradCPU);
    }

    #[test]
    fn backend_priority_returns_cpu() {
        let keys = DispatchKeySet::from_keys(&[DispatchKey::BackendSelect, DispatchKey::CPU]);
        let backend = keys
            .highest_priority_backend_type_id()
            .expect("backend priority should resolve");
        assert_eq!(backend, DispatchKey::CPU);
    }

    #[test]
    fn unknown_bits_fail_closed() {
        let err =
            DispatchKeySet::from_bits_checked(1u64 << 63).expect_err("unknown bits must fail");
        let msg = err.to_string();
        assert!(msg.contains("unknown bitmask"));
    }

    #[test]
    fn known_bits_parse_successfully() {
        let known_bits = DispatchKey::BackendSelect.bit()
            | DispatchKey::CompositeImplicitAutograd.bit()
            | DispatchKey::CompositeExplicitAutograd.bit()
            | DispatchKey::CPU.bit()
            | DispatchKey::AutogradCPU.bit();
        let parsed = DispatchKeySet::from_bits_checked(known_bits)
            .expect("known dispatch key bits must parse successfully");
        assert_eq!(parsed.bits(), known_bits);
    }

    #[test]
    fn dispatch_keyset_for_tensors_tracks_requires_grad() {
        let lhs = ScalarTensor::new(1.0, DType::F64, Device::Cpu);
        let rhs = ScalarTensor::new(2.0, DType::F64, Device::Cpu);

        let no_grad = dispatch_keyset_for_tensors(&lhs, &rhs, false);
        assert!(no_grad.has(DispatchKey::CPU));
        assert!(no_grad.has(DispatchKey::BackendSelect));
        assert!(!no_grad.has(DispatchKey::AutogradCPU));

        let with_grad = dispatch_keyset_for_tensors(&lhs, &rhs, true);
        assert!(with_grad.has(DispatchKey::CPU));
        assert!(with_grad.has(DispatchKey::BackendSelect));
        assert!(with_grad.has(DispatchKey::AutogradCPU));
    }

    #[test]
    fn validate_requires_cpu_for_autograd() {
        let keyset = DispatchKeySet::from_keys(&[DispatchKey::AutogradCPU]);
        let err = keyset
            .validate_for_scalar_binary()
            .expect_err("autograd without cpu must fail");
        assert!(matches!(err, DispatchKeyError::IncompatibleSet { .. }));
    }

    #[test]
    fn strict_mode_rejects_composite_fallback() {
        let lhs = ScalarTensor::new(2.0, DType::F64, Device::Cpu);
        let rhs = ScalarTensor::new(3.0, DType::F64, Device::Cpu);
        let keyset = DispatchKeySet::from_keys(&[
            DispatchKey::CompositeExplicitAutograd,
            DispatchKey::CPU,
            DispatchKey::BackendSelect,
        ]);

        let err = dispatch_scalar_binary_with_keyset(
            BinaryOp::Add,
            ExecutionMode::Strict,
            &lhs,
            &rhs,
            keyset,
        )
        .expect_err("strict mode must fail closed");
        assert!(err.to_string().contains("strict mode forbids"));
    }

    #[test]
    fn strict_mode_prefers_cpu_over_backendselect() {
        let lhs = ScalarTensor::new(2.0, DType::F64, Device::Cpu);
        let rhs = ScalarTensor::new(3.0, DType::F64, Device::Cpu);
        let keyset = DispatchKeySet::from_keys(&[DispatchKey::BackendSelect, DispatchKey::CPU]);

        let out = dispatch_scalar_binary_with_keyset(
            BinaryOp::Add,
            ExecutionMode::Strict,
            &lhs,
            &rhs,
            keyset,
        )
        .expect("strict mode should resolve directly to cpu when cpu type key exists");
        assert_eq!(out.tensor.value(), 5.0);
        assert_eq!(out.decision.selected_key, DispatchKey::CPU);
        assert_eq!(out.decision.backend_key, DispatchKey::CPU);
        assert!(!out.decision.fallback_used);
        assert_eq!(out.decision.kernel, "cpu::add_scalar");
    }

    #[test]
    fn strict_mode_subtracts_with_cpu_kernel() {
        let lhs = ScalarTensor::new(2.0, DType::F64, Device::Cpu);
        let rhs = ScalarTensor::new(3.0, DType::F64, Device::Cpu);
        let keyset = DispatchKeySet::from_keys(&[DispatchKey::BackendSelect, DispatchKey::CPU]);

        let out = dispatch_scalar_binary_with_keyset(
            BinaryOp::Sub,
            ExecutionMode::Strict,
            &lhs,
            &rhs,
            keyset,
        )
        .expect("strict mode should resolve subtraction directly to cpu");
        assert_eq!(out.tensor.value(), -1.0);
        assert_eq!(out.decision.selected_key, DispatchKey::CPU);
        assert_eq!(out.decision.backend_key, DispatchKey::CPU);
        assert!(!out.decision.fallback_used);
        assert_eq!(out.decision.kernel, "cpu::sub_scalar");
    }

    #[test]
    fn strict_mode_divides_with_cpu_kernel() {
        let lhs = ScalarTensor::new(7.0, DType::F64, Device::Cpu);
        let rhs = ScalarTensor::new(2.0, DType::F64, Device::Cpu);
        let keyset = DispatchKeySet::from_keys(&[DispatchKey::BackendSelect, DispatchKey::CPU]);

        let out = dispatch_scalar_binary_with_keyset(
            BinaryOp::Div,
            ExecutionMode::Strict,
            &lhs,
            &rhs,
            keyset,
        )
        .expect("strict mode should resolve division directly to cpu");
        assert_eq!(out.tensor.value(), 3.5);
        assert_eq!(out.decision.selected_key, DispatchKey::CPU);
        assert_eq!(out.decision.backend_key, DispatchKey::CPU);
        assert!(!out.decision.fallback_used);
        assert_eq!(out.decision.kernel, "cpu::div_scalar");
    }

    #[test]
    fn hardened_mode_allows_composite_fallback() {
        let lhs = ScalarTensor::new(2.0, DType::F64, Device::Cpu);
        let rhs = ScalarTensor::new(3.0, DType::F64, Device::Cpu);
        let keyset = DispatchKeySet::from_keys(&[
            DispatchKey::CompositeExplicitAutograd,
            DispatchKey::CPU,
            DispatchKey::BackendSelect,
        ]);

        let out = dispatch_scalar_binary_with_keyset(
            BinaryOp::Add,
            ExecutionMode::Hardened,
            &lhs,
            &rhs,
            keyset,
        )
        .expect("hardened mode should fallback");
        assert_eq!(out.tensor.value(), 5.0);
        assert!(out.decision.fallback_used);
        assert_eq!(
            out.decision.selected_key,
            DispatchKey::CompositeExplicitAutograd
        );
        assert_eq!(out.decision.backend_key, DispatchKey::CPU);
    }

    #[test]
    fn hardened_mode_prefers_cpu_over_backendselect() {
        let lhs = ScalarTensor::new(2.0, DType::F64, Device::Cpu);
        let rhs = ScalarTensor::new(4.0, DType::F64, Device::Cpu);
        let keyset = DispatchKeySet::from_keys(&[DispatchKey::BackendSelect, DispatchKey::CPU]);

        let out = dispatch_scalar_binary_with_keyset(
            BinaryOp::Mul,
            ExecutionMode::Hardened,
            &lhs,
            &rhs,
            keyset,
        )
        .expect("hardened mode should resolve directly to cpu when cpu type key exists");

        assert_eq!(out.tensor.value(), 8.0);
        assert!(!out.decision.fallback_used);
        assert_eq!(out.decision.selected_key, DispatchKey::CPU);
        assert_eq!(out.decision.backend_key, DispatchKey::CPU);
        assert_eq!(out.decision.kernel, "cpu::mul_scalar");
    }

    #[test]
    fn dispatch_returns_kernel_metadata() {
        let lhs = ScalarTensor::new(1.0, DType::F64, Device::Cpu);
        let rhs = ScalarTensor::new(2.0, DType::F64, Device::Cpu);
        let outcome =
            dispatch_scalar_binary(BinaryOp::Add, ExecutionMode::Strict, &lhs, &rhs, true)
                .expect("dispatch should succeed");

        assert_eq!(outcome.tensor.value(), 3.0);
        assert_eq!(outcome.decision.kernel, "autograd_cpu::add_scalar");
        assert_eq!(outcome.decision.mode, ExecutionMode::Strict);
        assert_eq!(outcome.decision.selected_key, DispatchKey::AutogradCPU);
        assert_eq!(outcome.decision.backend_key, DispatchKey::CPU);
        assert!(!outcome.decision.fallback_used);
    }

    #[test]
    fn dispatch_keyset_for_tensor_meta_tracks_requires_grad() {
        let lhs_meta = TensorMeta::from_shape(vec![2], DType::F64, Device::Cpu);
        let rhs_meta = TensorMeta::from_shape(vec![2], DType::F64, Device::Cpu);

        let no_grad = dispatch_keyset_for_tensor_meta(&lhs_meta, &rhs_meta, false);
        assert!(no_grad.has(DispatchKey::CPU));
        assert!(no_grad.has(DispatchKey::BackendSelect));
        assert!(!no_grad.has(DispatchKey::AutogradCPU));

        let with_grad = dispatch_keyset_for_tensor_meta(&lhs_meta, &rhs_meta, true);
        assert!(with_grad.has(DispatchKey::CPU));
        assert!(with_grad.has(DispatchKey::BackendSelect));
        assert!(with_grad.has(DispatchKey::AutogradCPU));
    }

    #[test]
    fn tensor_dispatch_strict_mode_uses_cpu_add_kernel() {
        let lhs_meta = TensorMeta::from_shape(vec![2, 2], DType::F64, Device::Cpu);
        let rhs_meta = TensorMeta::from_shape(vec![2, 2], DType::F64, Device::Cpu);
        let lhs = vec![1.0, 2.0, 3.0, 4.0];
        let rhs = vec![0.5, 1.5, 2.5, 3.5];
        let keyset = DispatchKeySet::from_keys(&[DispatchKey::BackendSelect, DispatchKey::CPU]);

        let out = dispatch_tensor_binary_contiguous_f64_with_keyset(
            BinaryOp::Add,
            ExecutionMode::Strict,
            &lhs,
            &rhs,
            &lhs_meta,
            &rhs_meta,
            keyset,
        )
        .expect("strict tensor dispatch should resolve to cpu add");

        assert_eq!(out.values, vec![1.5, 3.5, 5.5, 7.5]);
        assert_eq!(out.decision.kernel, "cpu::add_tensor_contiguous_f64");
        assert_eq!(out.decision.selected_key, DispatchKey::CPU);
        assert_eq!(out.decision.backend_key, DispatchKey::CPU);
        assert!(!out.decision.fallback_used);
    }

    #[test]
    fn tensor_dispatch_hardened_mode_allows_composite_fallback() {
        let lhs_meta = TensorMeta::from_shape(vec![2], DType::F64, Device::Cpu);
        let rhs_meta = TensorMeta::from_shape(vec![2], DType::F64, Device::Cpu);
        let lhs = vec![2.0, 4.0];
        let rhs = vec![3.0, 5.0];
        let keyset = DispatchKeySet::from_keys(&[
            DispatchKey::CompositeExplicitAutograd,
            DispatchKey::CPU,
            DispatchKey::BackendSelect,
        ]);

        let out = dispatch_tensor_binary_contiguous_f64_with_keyset(
            BinaryOp::Mul,
            ExecutionMode::Hardened,
            &lhs,
            &rhs,
            &lhs_meta,
            &rhs_meta,
            keyset,
        )
        .expect("hardened tensor dispatch should fallback to cpu");

        assert_eq!(out.values, vec![6.0, 20.0]);
        assert_eq!(
            out.decision.selected_key,
            DispatchKey::CompositeExplicitAutograd
        );
        assert_eq!(out.decision.backend_key, DispatchKey::CPU);
        assert!(out.decision.fallback_used);
    }

    #[test]
    fn tensor_dispatch_strict_mode_rejects_composite_fallback() {
        let lhs_meta = TensorMeta::from_shape(vec![2], DType::F64, Device::Cpu);
        let rhs_meta = TensorMeta::from_shape(vec![2], DType::F64, Device::Cpu);
        let lhs = vec![2.0, 4.0];
        let rhs = vec![3.0, 5.0];
        let keyset = DispatchKeySet::from_keys(&[
            DispatchKey::CompositeExplicitAutograd,
            DispatchKey::CPU,
            DispatchKey::BackendSelect,
        ]);

        let err = dispatch_tensor_binary_contiguous_f64_with_keyset(
            BinaryOp::Mul,
            ExecutionMode::Strict,
            &lhs,
            &rhs,
            &lhs_meta,
            &rhs_meta,
            keyset,
        )
        .expect_err("strict tensor dispatch must fail closed");
        assert!(err.to_string().contains("strict mode forbids"));
    }

    #[test]
    fn tensor_dispatch_propagates_kernel_dtype_mismatch() {
        let lhs_meta = TensorMeta::from_shape(vec![2], DType::F64, Device::Cpu);
        let rhs_meta = TensorMeta::from_shape(vec![2], DType::F32, Device::Cpu);
        let lhs = vec![1.0, 2.0];
        let rhs = vec![3.0, 4.0];

        let err = dispatch_tensor_binary_contiguous_f64(
            BinaryOp::Add,
            ExecutionMode::Strict,
            &lhs,
            &rhs,
            &lhs_meta,
            &rhs_meta,
            false,
        )
        .expect_err("dtype mismatch should bubble as kernel error");

        assert!(matches!(
            err,
            DispatchError::Kernel(KernelError::Incompatible(
                TensorCompatError::DTypeMismatch { .. }
            ))
        ));
    }

    #[test]
    fn tensor_dispatch_propagates_non_contiguous_layout_rejection() {
        let lhs_meta =
            TensorMeta::from_shape_and_strides(vec![2, 2], vec![1, 2], 0, DType::F64, Device::Cpu)
                .expect("test meta should be valid");
        let rhs_meta = TensorMeta::from_shape(vec![2, 2], DType::F64, Device::Cpu);
        let lhs = vec![1.0, 2.0, 3.0, 4.0];
        let rhs = vec![5.0, 6.0, 7.0, 8.0];

        let err = dispatch_tensor_binary_contiguous_f64(
            BinaryOp::Add,
            ExecutionMode::Strict,
            &lhs,
            &rhs,
            &lhs_meta,
            &rhs_meta,
            false,
        )
        .expect_err("non-contiguous layout must fail closed");

        assert!(matches!(
            err,
            DispatchError::Kernel(KernelError::UnsupportedLayout { side: "lhs" })
        ));
    }

    #[test]
    fn schema_row_parse_round_trips_add_tensor_signature() -> Result<(), String> {
        let schema_text = "add.Tensor(Tensor self, Tensor other, *, Scalar alpha=1) -> Tensor";
        let parsed = parse_schema_or_name(schema_text)
            .map_err(|error| format!("schema should parse: {error}"))?;
        let schema = match parsed {
            ParsedSchemaInput::Schema(schema) => schema,
            other => return Err(format!("expected full schema parse, got {other:?}")),
        };
        assert_eq!(schema.op.base, "add");
        assert_eq!(schema.op.overload.as_deref(), Some("Tensor"));
        assert!(!schema.op.is_inplace);
        assert!(!schema.is_out_variant);
        assert!(schema.arguments.contains("Tensor self"));
        assert_eq!(schema.returns, "Tensor");

        let seed = det_seed(&[schema.schema_digest, 0x0034, 0x1405]);
        let log = build_schema_property_log(SchemaLogParams {
            test_id: "schema_row_parse_round_trips_add_tensor_signature",
            mode: "strict",
            seed,
            op_name: &schema.op.base,
            overload_name: schema.op.overload.as_deref().unwrap_or(""),
            schema_digest: schema.schema_digest,
            dispatch_keyset_bits: 0,
            reason_code: "op_schema_row_roundtrip_ok",
        });
        assert_schema_log_contract(&log);
        Ok(())
    }

    #[test]
    fn operator_name_parse_preserves_overload_token() {
        let parsed = parse_schema_name("add.Tensor").expect("name should parse");
        assert_eq!(parsed.base, "add");
        assert_eq!(parsed.overload.as_deref(), Some("Tensor"));
        assert_eq!(parsed.unambiguous_name(), "add_Tensor");
        assert!(!parsed.is_inplace);
    }

    #[test]
    fn base_operator_name_parse_inplace_suffix_contract() {
        let parsed = parse_schema_name("add_").expect("inplace name should parse");
        assert_eq!(parsed.base, "add");
        assert!(parsed.is_inplace);
        assert!(parsed.overload.is_none());
    }

    #[test]
    fn schema_out_variant_requires_mutable_out_alias() -> Result<(), String> {
        let schema_text =
            "add.out(Tensor self, Tensor other, *, Scalar alpha=1, Tensor(a!) out) -> Tensor(a!)";
        let parsed = parse_schema_or_name(schema_text)
            .map_err(|error| format!("schema should parse: {error}"))?;
        let schema = match parsed {
            ParsedSchemaInput::Schema(schema) => schema,
            other => return Err(format!("expected full schema parse, got {other:?}")),
        };
        assert_eq!(schema.op.overload.as_deref(), Some("out"));
        assert!(schema.is_out_variant);
        assert!(schema.arguments.contains("Tensor(a!) out"));
        assert_eq!(schema.returns, "Tensor(a!)");
        Ok(())
    }

    #[test]
    fn parse_schema_or_name_classifies_name_only_vs_full_schema() {
        let name_only = parse_schema_or_name("add.Tensor").expect("name-only form should parse");
        assert!(matches!(name_only, ParsedSchemaInput::Name(_)));

        let full_schema = parse_schema_or_name(
            "add.Tensor(Tensor self, Tensor other, *, Scalar alpha=1) -> Tensor",
        )
        .expect("full schema should parse");
        assert!(matches!(full_schema, ParsedSchemaInput::Schema(_)));
    }

    #[test]
    fn schema_parser_rejects_malformed_tokens() {
        let malformed = "add.Tensor(Tensor self, Tensor other -> Tensor";
        let err = parse_schema_or_name(malformed).expect_err("malformed schema must fail");
        assert!(matches!(err, OpSchemaError::MalformedSchema { .. }));
    }

    #[test]
    fn schema_parser_rejects_illegal_overload_name() {
        let err = parse_schema_name("add.default").expect_err("default overload must fail");
        assert!(matches!(err, OpSchemaError::InvalidOverloadName { .. }));

        let err = parse_schema_name("add.__magic").expect_err("dunder overload must fail");
        assert!(matches!(err, OpSchemaError::InvalidOverloadName { .. }));
    }

    #[test]
    fn schema_dispatch_keyset_rejects_unknown_backend_key() {
        let err = schema_dispatch_keyset_from_tags(&["CPU", "NotARealKey"])
            .expect_err("unknown dispatch key tags must fail");
        assert!(matches!(err, OpSchemaError::UnknownDispatchKey { .. }));
    }

    #[test]
    fn schema_dispatch_keyset_requires_cpu_backend_for_scoped_ops() {
        let err = schema_dispatch_keyset_from_tags(&["AutogradCPU"])
            .expect_err("autograd key without cpu backend should fail");
        assert!(matches!(
            err,
            OpSchemaError::IncompatibleDispatchKeyset(DispatchKeyError::IncompatibleSet { .. })
        ));
    }

    #[test]
    fn property_log_contract_maps_to_dispatch_scenarios() {
        let seed = det_seed(&[0x13, 0x5, 0x2]);
        let log = build_property_log(
            "prop_contract_mapping",
            "strict",
            seed,
            0xdead_beef,
            0xbead_1337,
            "dispatch_property_contract_mapping_ok",
        );
        assert_log_contract(&log);

        let scenario_id = log
            .get("scenario_id")
            .expect("scenario_id must be present in property logs");
        assert!(scenario_id.starts_with("dispatch_key/strict:"));
        assert!(
            scenario_id.contains("prop_contract_mapping"),
            "scenario id should preserve test identity"
        );

        let replay = log
            .get("replay_command")
            .expect("replay command must be present in property logs");
        assert!(replay.contains("cargo test -p ft-dispatch"));
        assert!(replay.contains("prop_contract_mapping"));
    }

    proptest! {
        #[test]
        fn prop_schema_name_roundtrip(
            base in "[a-z][a-z0-9]{0,8}",
            overload in "[A-Z][A-Za-z0-9_]{0,8}",
            has_overload in any::<bool>(),
            inplace in any::<bool>(),
        ) {
            let mut full_name = base.clone();
            if inplace {
                full_name.push('_');
            }
            if has_overload {
                full_name.push('.');
                full_name.push_str(&overload);
            }

            let parsed = parse_schema_name(&full_name)
                .expect("generated schema names should parse");
            prop_assert_eq!(parsed.base.as_str(), base.as_str());
            prop_assert_eq!(parsed.is_inplace, inplace);
            if has_overload {
                prop_assert_eq!(parsed.overload.as_deref(), Some(overload.as_str()));
                prop_assert_eq!(
                    parsed.unambiguous_name(),
                    format!("{}_{}", base, overload),
                );
            } else {
                prop_assert!(parsed.overload.is_none());
                prop_assert_eq!(parsed.unambiguous_name(), base.clone());
            }

            let schema_digest = det_seed(&[
                parsed.base.len() as u64,
                parsed.overload.as_ref().map_or(0, |value| value.len() as u64),
                inplace as u64,
            ]);
            let seed = det_seed(&[schema_digest, has_overload as u64]);
            let log = build_schema_property_log(SchemaLogParams {
                test_id: "prop_schema_name_roundtrip",
                mode: "strict",
                seed,
                op_name: &parsed.base,
                overload_name: parsed.overload.as_deref().unwrap_or(""),
                schema_digest,
                dispatch_keyset_bits: 0,
                reason_code: "op_schema_name_roundtrip_ok",
            });
            assert_schema_log_contract(&log);
        }

        #[test]
        fn prop_known_bits_roundtrip(
            backend_select in any::<bool>(),
            composite_implicit in any::<bool>(),
            composite_explicit in any::<bool>(),
            cpu in any::<bool>(),
            autograd_cpu in any::<bool>(),
        ) {
            let mut keyset = DispatchKeySet::empty();
            if backend_select {
                keyset.add(DispatchKey::BackendSelect);
            }
            if composite_implicit {
                keyset.add(DispatchKey::CompositeImplicitAutograd);
            }
            if composite_explicit {
                keyset.add(DispatchKey::CompositeExplicitAutograd);
            }
            if cpu {
                keyset.add(DispatchKey::CPU);
            }
            if autograd_cpu {
                keyset.add(DispatchKey::AutogradCPU);
            }

            let bits = keyset.bits();
            let reparsed = DispatchKeySet::from_bits_checked(bits)
                .expect("known bit combinations must parse");
            prop_assert_eq!(reparsed.bits(), bits);

            let seed = det_seed(&[bits, backend_select as u64, cpu as u64, autograd_cpu as u64]);
            let log = build_property_log(
                "prop_known_bits_roundtrip",
                "strict",
                seed,
                bits,
                reparsed.bits(),
                "dispatch_known_bits_roundtrip_ok",
            );
            assert_log_contract(&log);
        }

        #[test]
        fn prop_unknown_bits_mask_fail_closed(bit in prop_oneof![Just(0u8), 6u8..=63u8]) {
            let mask = 1u64 << u32::from(bit);
            let err = DispatchKeySet::from_bits_checked(mask)
                .expect_err("unknown bit masks must fail closed");
            match err {
                DispatchKeyError::UnknownBits { unknown_mask } => {
                    prop_assert_eq!(unknown_mask, mask);
                }
                other => prop_assert!(false, "expected UnknownBits, got {other:?}"),
            }

            let seed = det_seed(&[mask, bit as u64]);
            let log = build_property_log(
                "prop_unknown_bits_mask_fail_closed",
                "strict",
                seed,
                mask,
                mask,
                "dispatch_unknown_bits_fail_closed",
            );
            assert_log_contract(&log);
        }

        #[test]
        fn prop_priority_matches_explicit_table(
            backend_select in any::<bool>(),
            composite_implicit in any::<bool>(),
            composite_explicit in any::<bool>(),
            cpu in any::<bool>(),
            autograd_cpu in any::<bool>(),
        ) {
            let mut keyset = DispatchKeySet::empty();
            if backend_select {
                keyset.add(DispatchKey::BackendSelect);
            }
            if composite_implicit {
                keyset.add(DispatchKey::CompositeImplicitAutograd);
            }
            if composite_explicit {
                keyset.add(DispatchKey::CompositeExplicitAutograd);
            }
            if cpu {
                keyset.add(DispatchKey::CPU);
            }
            if autograd_cpu {
                keyset.add(DispatchKey::AutogradCPU);
            }

            let result = keyset.highest_priority_type_id();
            if keyset.is_empty() {
                prop_assert!(matches!(result, Err(DispatchKeyError::EmptySet)));
            } else {
                let expected = TYPE_PRIORITY
                    .iter()
                    .copied()
                    .find(|key| keyset.has(*key))
                    .expect("non-empty keyset should have a type key");
                prop_assert_eq!(result.expect("type key should resolve"), expected);
            }

            let bits = keyset.bits();
            let output = result.map_or(0u64, |key| key as u8 as u64);
            let seed = det_seed(&[bits, output]);
            let log = build_property_log(
                "prop_priority_matches_explicit_table",
                "strict",
                seed,
                bits,
                output,
                "dispatch_type_priority_contract_ok",
            );
            assert_log_contract(&log);
        }

        #[test]
        fn prop_backend_resolution_requires_cpu(
            backend_select in any::<bool>(),
            composite_implicit in any::<bool>(),
            composite_explicit in any::<bool>(),
            cpu in any::<bool>(),
            autograd_cpu in any::<bool>(),
        ) {
            let mut keyset = DispatchKeySet::empty();
            if backend_select {
                keyset.add(DispatchKey::BackendSelect);
            }
            if composite_implicit {
                keyset.add(DispatchKey::CompositeImplicitAutograd);
            }
            if composite_explicit {
                keyset.add(DispatchKey::CompositeExplicitAutograd);
            }
            if cpu {
                keyset.add(DispatchKey::CPU);
            }
            if autograd_cpu {
                keyset.add(DispatchKey::AutogradCPU);
            }

            let result = keyset.highest_priority_backend_type_id();
            if keyset.is_empty() {
                prop_assert!(matches!(result, Err(DispatchKeyError::EmptySet)));
            } else if cpu {
                prop_assert_eq!(result.expect("cpu backend should resolve"), DispatchKey::CPU);
            } else {
                prop_assert!(matches!(result, Err(DispatchKeyError::NoBackendKey)));
            }

            let bits = keyset.bits();
            let output = result.map_or(0u64, |key| key as u8 as u64);
            let seed = det_seed(&[bits, output, cpu as u64]);
            let log = build_property_log(
                "prop_backend_resolution_requires_cpu",
                "strict",
                seed,
                bits,
                output,
                "dispatch_backend_resolution_contract_ok",
            );
            assert_log_contract(&log);
        }

        #[test]
        fn prop_validate_requires_cpu_for_autograd(
            backend_select in any::<bool>(),
            composite_implicit in any::<bool>(),
            composite_explicit in any::<bool>(),
            cpu in any::<bool>(),
            autograd_cpu in any::<bool>(),
        ) {
            let mut keyset = DispatchKeySet::empty();
            if backend_select {
                keyset.add(DispatchKey::BackendSelect);
            }
            if composite_implicit {
                keyset.add(DispatchKey::CompositeImplicitAutograd);
            }
            if composite_explicit {
                keyset.add(DispatchKey::CompositeExplicitAutograd);
            }
            if cpu {
                keyset.add(DispatchKey::CPU);
            }
            if autograd_cpu {
                keyset.add(DispatchKey::AutogradCPU);
            }

            let validation = keyset.validate_for_scalar_binary();
            if keyset.is_empty() {
                prop_assert!(matches!(validation, Err(DispatchKeyError::EmptySet)));
            } else if autograd_cpu && !cpu {
                match validation {
                    Err(DispatchKeyError::IncompatibleSet { .. }) => {}
                    other => prop_assert!(false, "expected IncompatibleSet, got {other:?}"),
                }
            } else if !cpu {
                prop_assert!(matches!(validation, Err(DispatchKeyError::NoBackendKey)));
            } else {
                prop_assert!(validation.is_ok());
            }

            let bits = keyset.bits();
            let outcome = if validation.is_ok() { 1u64 } else { 0u64 };
            let seed = det_seed(&[bits, outcome, autograd_cpu as u64]);
            let log = build_property_log(
                "prop_validate_requires_cpu_for_autograd",
                "strict",
                seed,
                bits,
                outcome,
                "dispatch_validate_contract_ok",
            );
            assert_log_contract(&log);
        }

        #[test]
        fn prop_mode_split_for_composite_keysets(
            lhs_value in -1_000.0f64..1_000.0f64,
            rhs_value in -1_000.0f64..1_000.0f64,
            use_explicit in any::<bool>(),
        ) {
            let lhs = ScalarTensor::new(lhs_value, DType::F64, Device::Cpu);
            let rhs = ScalarTensor::new(rhs_value, DType::F64, Device::Cpu);
            let selected_key = if use_explicit {
                DispatchKey::CompositeExplicitAutograd
            } else {
                DispatchKey::CompositeImplicitAutograd
            };
            let keyset =
                DispatchKeySet::from_keys(&[selected_key, DispatchKey::CPU, DispatchKey::BackendSelect]);

            let strict_err = dispatch_scalar_binary_with_keyset(
                BinaryOp::Add,
                ExecutionMode::Strict,
                &lhs,
                &rhs,
                keyset,
            )
            .expect_err("strict mode must reject composite/backend fallback");
            prop_assert!(strict_err.to_string().contains("strict mode forbids"));

            let hardened_out = dispatch_scalar_binary_with_keyset(
                BinaryOp::Add,
                ExecutionMode::Hardened,
                &lhs,
                &rhs,
                keyset,
            )
            .expect("hardened mode should allow bounded fallback");

            prop_assert!(hardened_out.decision.fallback_used);
            prop_assert_eq!(hardened_out.decision.selected_key, selected_key);
            prop_assert_eq!(hardened_out.decision.backend_key, DispatchKey::CPU);
            prop_assert!((hardened_out.tensor.value() - (lhs_value + rhs_value)).abs() <= 1e-12);

            let seed = det_seed(&[
                lhs_value.to_bits(),
                rhs_value.to_bits(),
                use_explicit as u64,
            ]);
            let log = build_property_log(
                "prop_mode_split_for_composite_keysets",
                "hardened",
                seed,
                lhs.evidence_fingerprint64() ^ rhs.evidence_fingerprint64(),
                hardened_out.decision.keyset_bits,
                "dispatch_mode_split_contract_ok",
            );
            assert_log_contract(&log);
        }

        #[test]
        fn prop_packet_007_autograd_without_cpu_stays_fail_closed(
            backend_select in any::<bool>(),
            composite_implicit in any::<bool>(),
            composite_explicit in any::<bool>(),
        ) {
            let mut keyset = DispatchKeySet::empty();
            keyset.add(DispatchKey::AutogradCPU);
            if backend_select {
                keyset.add(DispatchKey::BackendSelect);
            }
            if composite_implicit {
                keyset.add(DispatchKey::CompositeImplicitAutograd);
            }
            if composite_explicit {
                keyset.add(DispatchKey::CompositeExplicitAutograd);
            }

            let validation = keyset.validate_for_scalar_binary();
            match validation {
                Err(DispatchKeyError::IncompatibleSet { .. }) => {}
                other => prop_assert!(false, "expected IncompatibleSet, got {other:?}"),
            }

            let bits = keyset.bits();
            let seed = det_seed(&[
                bits,
                backend_select as u64,
                composite_implicit as u64,
                composite_explicit as u64,
            ]);
            let strict_log = build_packet_007_property_log(
                "prop_packet_007_autograd_without_cpu_stays_fail_closed",
                "strict",
                seed,
                bits,
                0,
                "dispatch007_autograd_without_cpu_fail_closed",
                "dispatch_key/strict:autograd_without_cpu_fail_closed",
            );
            assert_log_contract(&strict_log);

            let hardened_log = build_packet_007_property_log(
                "prop_packet_007_autograd_without_cpu_stays_fail_closed",
                "hardened",
                seed,
                bits,
                0,
                "dispatch007_autograd_without_cpu_fail_closed",
                "dispatch_key/hardened:autograd_without_cpu_fail_closed",
            );
            assert_log_contract(&hardened_log);
        }

        #[test]
        fn prop_packet_007_dtype_or_device_mismatch_stays_fail_closed(
            lhs_value in -1_000.0f64..1_000.0f64,
            rhs_value in -1_000.0f64..1_000.0f64,
            use_dtype_mismatch in any::<bool>(),
            use_mul in any::<bool>(),
        ) {
            let (lhs, rhs, reason_code, strict_scenario, hardened_scenario) = if use_dtype_mismatch {
                (
                    ScalarTensor::new(lhs_value, DType::F64, Device::Cpu),
                    ScalarTensor::new(rhs_value, DType::F32, Device::Cpu),
                    "dispatch007_dtype_mismatch_fail_closed",
                    "dispatch_key/strict:dtype_mismatch_fail_closed",
                    "dispatch_key/hardened:dtype_mismatch_fail_closed",
                )
            } else {
                (
                    ScalarTensor::new(lhs_value, DType::F64, Device::Cpu),
                    ScalarTensor::new(rhs_value, DType::F64, Device::Cuda),
                    "dispatch007_device_mismatch_fail_closed",
                    "dispatch_key/strict:device_mismatch_fail_closed",
                    "dispatch_key/hardened:device_mismatch_fail_closed",
                )
            };

            let op = if use_mul { BinaryOp::Mul } else { BinaryOp::Add };
            let keyset = DispatchKeySet::from_keys(&[DispatchKey::CPU, DispatchKey::BackendSelect]);

            let strict_err = dispatch_scalar_binary_with_keyset(
                op,
                ExecutionMode::Strict,
                &lhs,
                &rhs,
                keyset,
            )
            .expect_err("strict mode mismatch path must fail closed");
            prop_assert!(strict_err.to_string().contains("incompatible tensors"));

            let hardened_err = dispatch_scalar_binary_with_keyset(
                op,
                ExecutionMode::Hardened,
                &lhs,
                &rhs,
                keyset,
            )
            .expect_err("hardened mode mismatch path must fail closed");
            prop_assert!(hardened_err.to_string().contains("incompatible tensors"));

            let input_digest = lhs.evidence_fingerprint64() ^ rhs.evidence_fingerprint64();
            let output_digest = det_seed(&[input_digest, use_dtype_mismatch as u64, use_mul as u64]);
            let seed = det_seed(&[
                lhs_value.to_bits(),
                rhs_value.to_bits(),
                use_dtype_mismatch as u64,
                use_mul as u64,
            ]);

            let strict_log = build_packet_007_property_log(
                "prop_packet_007_dtype_or_device_mismatch_stays_fail_closed",
                "strict",
                seed,
                input_digest,
                output_digest,
                reason_code,
                strict_scenario,
            );
            assert_log_contract(&strict_log);

            let hardened_log = build_packet_007_property_log(
                "prop_packet_007_dtype_or_device_mismatch_stays_fail_closed",
                "hardened",
                seed,
                input_digest,
                output_digest,
                reason_code,
                hardened_scenario,
            );
            assert_log_contract(&hardened_log);
        }
    }
}

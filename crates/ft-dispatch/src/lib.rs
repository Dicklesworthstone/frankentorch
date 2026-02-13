#![forbid(unsafe_code)]

use std::fmt;

use ft_core::{Device, ExecutionMode, ScalarTensor};
use ft_kernel_cpu::{KernelError, add_scalar, mul_scalar};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BinaryOp {
    Add,
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

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
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

#[must_use]
pub fn dispatch_keyset_for_tensors(
    lhs: &ScalarTensor,
    _rhs: &ScalarTensor,
    requires_grad: bool,
) -> DispatchKeySet {
    let mut keyset = DispatchKeySet::empty();
    keyset.add(DispatchKey::BackendSelect);
    if lhs.meta().device() == Device::Cpu {
        keyset.add(DispatchKey::CPU);
    }
    if requires_grad {
        keyset.add(DispatchKey::AutogradCPU);
    }
    keyset
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

    let (tensor, kernel) = match (effective_key, op) {
        (DispatchKey::AutogradCPU, BinaryOp::Add) => {
            (add_scalar(lhs, rhs)?, "autograd_cpu::add_scalar")
        }
        (DispatchKey::AutogradCPU, BinaryOp::Mul) => {
            (mul_scalar(lhs, rhs)?, "autograd_cpu::mul_scalar")
        }
        (DispatchKey::CPU, BinaryOp::Add) => (add_scalar(lhs, rhs)?, "cpu::add_scalar"),
        (DispatchKey::CPU, BinaryOp::Mul) => (mul_scalar(lhs, rhs)?, "cpu::mul_scalar"),
        _ => {
            return Err(DispatchKeyError::IncompatibleSet {
                reason: "resolved dispatch key is unsupported for scalar binary ops",
            }
            .into());
        }
    };

    if effective_key != backend_key && effective_key != DispatchKey::AutogradCPU {
        return Err(DispatchKeyError::IncompatibleSet {
            reason: "resolved key/backend key drifted to incompatible pair",
        }
        .into());
    }

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

#[cfg(test)]
mod tests {
    use ft_core::{DType, Device, ExecutionMode, ScalarTensor};

    use super::{
        BinaryOp, DispatchKey, DispatchKeySet, dispatch_scalar_binary,
        dispatch_scalar_binary_with_keyset,
    };

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
}

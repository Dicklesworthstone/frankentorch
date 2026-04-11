#![forbid(unsafe_code)]

use std::fmt;
use std::hash::{Hash, Hasher};
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

/// Half-precision float types re-exported from the `half` crate.
/// Named `Float16`/`BFloat16` to avoid conflict with Rust 2024 primitive `f16`.
pub type Float16 = half::f16;
pub type BFloat16 = half::bf16;

/// Complex number types re-exported from the `num_complex` crate.
pub type Complex64 = num_complex::Complex<f32>;
pub type Complex128 = num_complex::Complex<f64>;

static NEXT_TENSOR_ID: AtomicU64 = AtomicU64::new(1);
static NEXT_STORAGE_ID: AtomicU64 = AtomicU64::new(1);

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum DType {
    F64,
    F32,
    F16,
    BF16,
    I64,
    I32,
    Bool,
    Complex64,
    Complex128,
}

impl DType {
    /// Size of one element in bytes.
    #[must_use]
    pub fn element_size(self) -> usize {
        match self {
            Self::Complex128 => 16,
            Self::F64 | Self::I64 | Self::Complex64 => 8,
            Self::F32 | Self::I32 => 4,
            Self::F16 | Self::BF16 => 2,
            Self::Bool => 1,
        }
    }

    /// Returns true for floating-point dtypes.
    #[must_use]
    pub fn is_floating_point(self) -> bool {
        matches!(self, Self::F64 | Self::F32 | Self::F16 | Self::BF16)
    }

    /// Returns true for half-precision floating-point dtypes (F16 or BF16).
    #[must_use]
    pub fn is_half(self) -> bool {
        matches!(self, Self::F16 | Self::BF16)
    }

    /// Returns true for integer dtypes (not bool).
    #[must_use]
    pub fn is_integer(self) -> bool {
        matches!(self, Self::I32 | Self::I64)
    }

    /// Returns true for the boolean dtype.
    #[must_use]
    pub fn is_bool(self) -> bool {
        matches!(self, Self::Bool)
    }

    /// Returns true for complex dtypes (Complex64 or Complex128).
    #[must_use]
    pub fn is_complex(self) -> bool {
        matches!(self, Self::Complex64 | Self::Complex128)
    }

    /// Promote two floating-point or complex dtypes: F32+F64→F64, same→same.
    /// Half-precision types promote to F32. F16+BF16→F32.
    /// Complex types: Complex64+F32→Complex64, Complex64+F64→Complex128, etc.
    /// Returns `None` for non-floating-point/non-complex dtypes.
    #[must_use]
    pub fn promote(self, other: Self) -> Option<Self> {
        match (self, other) {
            // Complex + Complex
            (Self::Complex128, Self::Complex128) => Some(Self::Complex128),
            (Self::Complex64, Self::Complex64) => Some(Self::Complex64),
            (Self::Complex128, Self::Complex64) | (Self::Complex64, Self::Complex128) => {
                Some(Self::Complex128)
            }
            // Complex + real float → complex (widen component if needed)
            (Self::Complex128, Self::F64 | Self::F32 | Self::F16 | Self::BF16)
            | (Self::F64 | Self::F32 | Self::F16 | Self::BF16, Self::Complex128) => {
                Some(Self::Complex128)
            }
            (Self::Complex64, Self::F64) | (Self::F64, Self::Complex64) => Some(Self::Complex128),
            (Self::Complex64, Self::F32 | Self::F16 | Self::BF16)
            | (Self::F32 | Self::F16 | Self::BF16, Self::Complex64) => Some(Self::Complex64),
            // Real floats
            (Self::F64, Self::F64) => Some(Self::F64),
            (Self::F32, Self::F32) => Some(Self::F32),
            (Self::F64, Self::F32) | (Self::F32, Self::F64) => Some(Self::F64),
            // Half-precision: same type stays, mixed half → F32
            (Self::F16, Self::F16) => Some(Self::F16),
            (Self::BF16, Self::BF16) => Some(Self::BF16),
            (Self::F16, Self::BF16) | (Self::BF16, Self::F16) => Some(Self::F32),
            // Half + wider float → wider float
            (Self::F16 | Self::BF16, Self::F32) | (Self::F32, Self::F16 | Self::BF16) => {
                Some(Self::F32)
            }
            (Self::F16 | Self::BF16, Self::F64) | (Self::F64, Self::F16 | Self::BF16) => {
                Some(Self::F64)
            }
            _ => None,
        }
    }

    /// Promote two dtypes following PyTorch's promotion hierarchy:
    /// Bool → I32 → I64 → F16/BF16 → F32 → F64 → Complex64 → Complex128.
    ///
    /// Any pair of dtypes returns the wider type in this hierarchy.
    /// Int + Float always promotes to the float type (or wider float).
    /// F16 + BF16 promotes to F32 (matching PyTorch semantics).
    /// Real + Complex promotes to Complex (widening component type if needed).
    /// This matches PyTorch's `torch.promote_types()`.
    #[must_use]
    pub fn promote_types(self, other: Self) -> Self {
        if self == other {
            return self;
        }
        // Special case: F16 + BF16 → F32 (PyTorch semantics)
        if matches!(
            (self, other),
            (Self::F16, Self::BF16) | (Self::BF16, Self::F16)
        ) {
            return Self::F32;
        }
        // Special case: Complex64 + F64 → Complex128 (widen component)
        if matches!(
            (self, other),
            (Self::Complex64, Self::F64) | (Self::F64, Self::Complex64)
        ) {
            return Self::Complex128;
        }
        // Assign a rank following PyTorch's promotion hierarchy.
        let rank = |d: Self| -> u8 {
            match d {
                Self::Bool => 0,
                Self::I32 => 1,
                Self::I64 => 2,
                Self::F16 | Self::BF16 => 3,
                Self::F32 => 4,
                Self::F64 => 5,
                Self::Complex64 => 6,
                Self::Complex128 => 7,
            }
        };
        if rank(self) >= rank(other) {
            self
        } else {
            other
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Device {
    Cpu,
    Cuda,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ExecutionMode {
    Strict,
    Hardened,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TensorMeta {
    shape: Vec<usize>,
    strides: Vec<usize>,
    storage_offset: usize,
    dtype: DType,
    device: Device,
}

impl TensorMeta {
    #[must_use]
    pub fn scalar(dtype: DType, device: Device) -> Self {
        Self {
            shape: Vec::new(),
            strides: Vec::new(),
            storage_offset: 0,
            dtype,
            device,
        }
    }

    #[must_use]
    pub fn from_shape(shape: Vec<usize>, dtype: DType, device: Device) -> Self {
        let strides = contiguous_strides(&shape);
        Self {
            shape,
            strides,
            storage_offset: 0,
            dtype,
            device,
        }
    }

    pub fn from_shape_and_strides(
        shape: Vec<usize>,
        strides: Vec<usize>,
        storage_offset: usize,
        dtype: DType,
        device: Device,
    ) -> Result<Self, TensorMetaError> {
        let meta = Self {
            shape,
            strides,
            storage_offset,
            dtype,
            device,
        };
        meta.validate()?;
        Ok(meta)
    }

    #[must_use]
    pub fn with_storage_offset(mut self, storage_offset: usize) -> Self {
        self.storage_offset = storage_offset;
        self
    }

    #[must_use]
    pub fn with_dtype(mut self, dtype: DType) -> Self {
        self.dtype = dtype;
        self
    }

    pub fn validate(&self) -> Result<(), TensorMetaError> {
        if self.shape.len() != self.strides.len() {
            return Err(TensorMetaError::RankStrideMismatch {
                rank: self.shape.len(),
                strides: self.strides.len(),
            });
        }

        let mut max_linear_offset = 0usize;
        for (size, stride) in self.shape.iter().copied().zip(self.strides.iter().copied()) {
            if size == 0 {
                continue;
            }

            let span = stride
                .checked_mul(size.saturating_sub(1))
                .ok_or(TensorMetaError::StrideOverflow { size, stride })?;
            max_linear_offset = max_linear_offset.checked_add(span).ok_or(
                TensorMetaError::StorageOffsetOverflow {
                    storage_offset: self.storage_offset,
                    max_linear_offset,
                },
            )?;
        }

        let _ = self.storage_offset.checked_add(max_linear_offset).ok_or(
            TensorMetaError::StorageOffsetOverflow {
                storage_offset: self.storage_offset,
                max_linear_offset,
            },
        )?;

        Ok(())
    }

    #[must_use]
    pub fn shape(&self) -> &[usize] {
        &self.shape
    }

    #[must_use]
    pub fn strides(&self) -> &[usize] {
        &self.strides
    }

    #[must_use]
    pub fn storage_offset(&self) -> usize {
        self.storage_offset
    }

    #[must_use]
    pub fn dtype(&self) -> DType {
        self.dtype
    }

    #[must_use]
    pub fn device(&self) -> Device {
        self.device
    }

    #[must_use]
    pub fn numel(&self) -> usize {
        if self.shape.is_empty() {
            return 1;
        }
        if self.shape.iter().copied().any(|dim| dim == 0) {
            return 0;
        }
        let mut product = 1usize;
        for dim in self.shape.iter().copied() {
            let Some(next) = product.checked_mul(dim) else {
                return usize::MAX;
            };
            product = next;
        }
        product
    }

    #[must_use]
    pub fn is_contiguous(&self) -> bool {
        if self.shape.len() != self.strides.len() {
            return false;
        }

        let mut expected_stride = 1usize;
        for (size, stride) in self
            .shape
            .iter()
            .copied()
            .zip(self.strides.iter().copied())
            .rev()
        {
            // Match PyTorch semantics: singleton dimensions are contiguous
            // regardless of stride.
            if size == 1 {
                continue;
            }
            if stride != expected_stride {
                return false;
            }
            let Some(next_expected) = expected_stride.checked_mul(size) else {
                return false;
            };
            expected_stride = next_expected;
        }
        true
    }

    pub fn storage_index_for(&self, index: &[usize]) -> Result<usize, TensorMetaError> {
        if index.len() != self.shape.len() {
            return Err(TensorMetaError::IndexRankMismatch {
                expected: self.shape.len(),
                actual: index.len(),
            });
        }

        let mut linear = self.storage_offset;
        for (dim, ((idx, dim_size), stride)) in index
            .iter()
            .copied()
            .zip(self.shape.iter().copied())
            .zip(self.strides.iter().copied())
            .enumerate()
        {
            if idx >= dim_size {
                return Err(TensorMetaError::IndexOutOfBounds {
                    dim,
                    index: idx,
                    size: dim_size,
                });
            }

            let step = idx
                .checked_mul(stride)
                .ok_or(TensorMetaError::StrideOverflow { size: idx, stride })?;
            linear = linear
                .checked_add(step)
                .ok_or(TensorMetaError::StorageOffsetOverflow {
                    storage_offset: self.storage_offset,
                    max_linear_offset: step,
                })?;
        }

        Ok(linear)
    }

    #[must_use]
    pub fn fingerprint64(&self) -> u64 {
        let mut hasher = DetHasher::new();
        self.shape.hash(&mut hasher);
        self.strides.hash(&mut hasher);
        self.storage_offset.hash(&mut hasher);
        self.dtype.hash(&mut hasher);
        self.device.hash(&mut hasher);
        hasher.finish()
    }
}

struct DetHasher(u64);

impl DetHasher {
    fn new() -> Self {
        Self(0xcbf2_9ce4_8422_2325)
    }
}

impl Hasher for DetHasher {
    fn finish(&self) -> u64 {
        self.0
    }

    fn write(&mut self, bytes: &[u8]) {
        for &byte in bytes {
            self.0 ^= u64::from(byte);
            self.0 = self.0.wrapping_mul(0x0000_0100_0000_01b3);
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TensorMetaError {
    RankStrideMismatch {
        rank: usize,
        strides: usize,
    },
    StrideOverflow {
        size: usize,
        stride: usize,
    },
    StorageOffsetOverflow {
        storage_offset: usize,
        max_linear_offset: usize,
    },
    IndexRankMismatch {
        expected: usize,
        actual: usize,
    },
    IndexOutOfBounds {
        dim: usize,
        index: usize,
        size: usize,
    },
}

impl fmt::Display for TensorMetaError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::RankStrideMismatch { rank, strides } => {
                write!(f, "shape rank {rank} does not match strides rank {strides}")
            }
            Self::StrideOverflow { size, stride } => {
                write!(f, "stride overflow for size={size}, stride={stride}")
            }
            Self::StorageOffsetOverflow {
                storage_offset,
                max_linear_offset,
            } => write!(
                f,
                "storage offset overflow for storage_offset={storage_offset}, max_linear_offset={max_linear_offset}"
            ),
            Self::IndexRankMismatch { expected, actual } => {
                write!(
                    f,
                    "index rank mismatch expected={expected}, actual={actual}"
                )
            }
            Self::IndexOutOfBounds { dim, index, size } => {
                write!(
                    f,
                    "index out of bounds at dim={dim}: index={index}, size={size}"
                )
            }
        }
    }
}

impl std::error::Error for TensorMetaError {}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TensorCompatError {
    DTypeMismatch { lhs: DType, rhs: DType },
    DeviceMismatch { lhs: Device, rhs: Device },
}

impl fmt::Display for TensorCompatError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::DTypeMismatch { lhs, rhs } => {
                write!(f, "dtype mismatch: lhs={lhs:?}, rhs={rhs:?}")
            }
            Self::DeviceMismatch { lhs, rhs } => {
                write!(f, "device mismatch: lhs={lhs:?}, rhs={rhs:?}")
            }
        }
    }
}

impl std::error::Error for TensorCompatError {}

#[derive(Debug, Clone, PartialEq)]
pub struct ScalarTensor {
    id: u64,
    storage_id: u64,
    meta: TensorMeta,
    value: f64,
    version: u64,
}

impl ScalarTensor {
    #[must_use]
    pub fn new(value: f64, dtype: DType, device: Device) -> Self {
        Self {
            id: NEXT_TENSOR_ID.fetch_add(1, Ordering::Relaxed),
            storage_id: NEXT_STORAGE_ID.fetch_add(1, Ordering::Relaxed),
            meta: TensorMeta::scalar(dtype, device),
            value,
            version: 0,
        }
    }

    #[must_use]
    pub fn with_value(&self, value: f64) -> Self {
        Self {
            id: NEXT_TENSOR_ID.fetch_add(1, Ordering::Relaxed),
            storage_id: NEXT_STORAGE_ID.fetch_add(1, Ordering::Relaxed),
            meta: self.meta.clone(),
            value,
            version: self.version.saturating_add(1),
        }
    }

    pub fn alias_view(&self, storage_offset: usize) -> Result<Self, TensorMetaError> {
        let meta = self.meta.clone().with_storage_offset(storage_offset);
        meta.validate()?;
        Ok(Self {
            id: NEXT_TENSOR_ID.fetch_add(1, Ordering::Relaxed),
            storage_id: self.storage_id,
            meta,
            value: self.value,
            version: self.version,
        })
    }

    pub fn set_in_place(&mut self, value: f64) {
        self.value = value;
        self.version = self.version.saturating_add(1);
    }

    #[must_use]
    pub fn id(&self) -> u64 {
        self.id
    }

    #[must_use]
    pub fn storage_id(&self) -> u64 {
        self.storage_id
    }

    #[must_use]
    pub fn value(&self) -> f64 {
        self.value
    }

    #[must_use]
    pub fn meta(&self) -> &TensorMeta {
        &self.meta
    }

    #[must_use]
    pub fn version(&self) -> u64 {
        self.version
    }

    #[must_use]
    pub fn evidence_fingerprint64(&self) -> u64 {
        let mut hasher = DetHasher::new();
        self.id.hash(&mut hasher);
        self.storage_id.hash(&mut hasher);
        self.version.hash(&mut hasher);
        self.meta.fingerprint64().hash(&mut hasher);
        self.value.to_bits().hash(&mut hasher);
        hasher.finish()
    }
}

// ── Typed Tensor Storage ────────────────────────────────────────────────

#[derive(Debug, Clone, PartialEq)]
pub enum TensorStorage {
    F32(Arc<Vec<f32>>),
    F64(Arc<Vec<f64>>),
    F16(Arc<Vec<Float16>>),
    BF16(Arc<Vec<BFloat16>>),
    Complex64(Arc<Vec<Complex64>>),
    Complex128(Arc<Vec<Complex128>>),
}

impl TensorStorage {
    #[must_use]
    pub fn len(&self) -> usize {
        match self {
            Self::F32(v) => v.len(),
            Self::F64(v) => v.len(),
            Self::F16(v) => v.len(),
            Self::BF16(v) => v.len(),
            Self::Complex64(v) => v.len(),
            Self::Complex128(v) => v.len(),
        }
    }

    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    #[must_use]
    pub fn dtype(&self) -> DType {
        match self {
            Self::F32(_) => DType::F32,
            Self::F64(_) => DType::F64,
            Self::F16(_) => DType::F16,
            Self::BF16(_) => DType::BF16,
            Self::Complex64(_) => DType::Complex64,
            Self::Complex128(_) => DType::Complex128,
        }
    }

    #[must_use]
    pub fn as_f64(&self) -> Option<&[f64]> {
        match self {
            Self::F64(v) => Some(v.as_slice()),
            _ => None,
        }
    }

    #[must_use]
    pub fn as_f32(&self) -> Option<&[f32]> {
        match self {
            Self::F32(v) => Some(v.as_slice()),
            _ => None,
        }
    }

    #[must_use]
    pub fn as_f16(&self) -> Option<&[Float16]> {
        match self {
            Self::F16(v) => Some(v.as_slice()),
            _ => None,
        }
    }

    #[must_use]
    pub fn as_bf16(&self) -> Option<&[BFloat16]> {
        match self {
            Self::BF16(v) => Some(v.as_slice()),
            _ => None,
        }
    }

    #[must_use]
    pub fn as_complex64(&self) -> Option<&[Complex64]> {
        match self {
            Self::Complex64(v) => Some(v.as_slice()),
            _ => None,
        }
    }

    #[must_use]
    pub fn as_complex128(&self) -> Option<&[Complex128]> {
        match self {
            Self::Complex128(v) => Some(v.as_slice()),
            _ => None,
        }
    }

    /// Convert storage to f64 values, promoting from any float type.
    /// Complex types extract the real part.
    #[must_use]
    pub fn to_f64_vec(&self) -> Vec<f64> {
        match self {
            Self::F64(v) => v.as_ref().clone(),
            Self::F32(v) => v.iter().map(|&x| f64::from(x)).collect(),
            Self::F16(v) => v.iter().map(|&x| f64::from(x.to_f32())).collect(),
            Self::BF16(v) => v.iter().map(|&x| f64::from(x.to_f32())).collect(),
            Self::Complex64(v) => v.iter().map(|z| f64::from(z.re)).collect(),
            Self::Complex128(v) => v.iter().map(|z| z.re).collect(),
        }
    }

    /// Convert storage to f32 values, promoting from half or demoting from f64.
    /// Complex types extract the real part.
    #[must_use]
    pub fn to_f32_vec(&self) -> Vec<f32> {
        match self {
            Self::F32(v) => v.as_ref().clone(),
            Self::F64(v) => v.iter().map(|&x| x as f32).collect(),
            Self::F16(v) => v.iter().map(|&x| x.to_f32()).collect(),
            Self::BF16(v) => v.iter().map(|&x| x.to_f32()).collect(),
            Self::Complex64(v) => v.iter().map(|z| z.re).collect(),
            Self::Complex128(v) => v.iter().map(|z| z.re as f32).collect(),
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct DenseTensor {
    id: u64,
    storage_id: u64,
    meta: TensorMeta,
    storage: TensorStorage,
    version: u64,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum DenseTensorError {
    Meta(TensorMetaError),
    UnsupportedDType(DType),
    UnsupportedLayout,
    UnsupportedStorageAccess { dtype: DType },
    StorageSpanOverflow { storage_offset: usize, numel: usize },
    InsufficientStorage { needed: usize, actual: usize },
    ShapeOverflow { shape: Vec<usize> },
}

impl fmt::Display for DenseTensorError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Meta(error) => write!(f, "invalid tensor metadata: {error}"),
            Self::UnsupportedDType(dtype) => write!(
                f,
                "unsupported tensor dtype for this storage type: {dtype:?}"
            ),
            Self::UnsupportedLayout => {
                write!(f, "dense tensor requires contiguous layout")
            }
            Self::UnsupportedStorageAccess { dtype } => write!(
                f,
                "raw f64 storage access is unsupported for tensor dtype {dtype:?}; use typed_storage() or contiguous_values_as_f64()"
            ),
            Self::StorageSpanOverflow {
                storage_offset,
                numel,
            } => write!(
                f,
                "dense tensor storage span overflow for storage_offset={storage_offset}, numel={numel}"
            ),
            Self::InsufficientStorage { needed, actual } => write!(
                f,
                "dense tensor storage length is insufficient: needed={needed}, actual={actual}"
            ),
            Self::ShapeOverflow { shape } => {
                write!(f, "dense tensor shape volume overflow for shape={shape:?}")
            }
        }
    }
}

impl std::error::Error for DenseTensorError {}

impl From<TensorMetaError> for DenseTensorError {
    fn from(value: TensorMetaError) -> Self {
        Self::Meta(value)
    }
}

fn contiguous_required_len(meta: &TensorMeta) -> Result<usize, DenseTensorError> {
    meta.storage_offset()
        .checked_add(meta.numel())
        .ok_or(DenseTensorError::StorageSpanOverflow {
            storage_offset: meta.storage_offset(),
            numel: meta.numel(),
        })
}

fn checked_shape_numel(shape: &[usize]) -> Result<usize, DenseTensorError> {
    if shape.is_empty() {
        return Ok(1);
    }
    let mut product = 1usize;
    for &dim in shape {
        if dim == 0 {
            return Ok(0);
        }
        product = product
            .checked_mul(dim)
            .ok_or_else(|| DenseTensorError::ShapeOverflow {
                shape: shape.to_vec(),
            })?;
    }
    Ok(product)
}

impl DenseTensor {
    pub fn from_typed_storage(
        meta: TensorMeta,
        storage: TensorStorage,
    ) -> Result<Self, DenseTensorError> {
        meta.validate()?;
        if meta.dtype() != storage.dtype() {
            return Err(DenseTensorError::UnsupportedDType(meta.dtype()));
        }
        if !meta.dtype().is_floating_point() && !meta.dtype().is_complex() {
            return Err(DenseTensorError::UnsupportedDType(meta.dtype()));
        }

        let needed = Self::storage_span_required_len(&meta)?;
        if storage.len() < needed {
            return Err(DenseTensorError::InsufficientStorage {
                needed,
                actual: storage.len(),
            });
        }

        Ok(Self {
            id: NEXT_TENSOR_ID.fetch_add(1, Ordering::Relaxed),
            storage_id: NEXT_STORAGE_ID.fetch_add(1, Ordering::Relaxed),
            meta,
            storage,
            version: 0,
        })
    }

    pub fn from_storage(meta: TensorMeta, storage: Vec<f64>) -> Result<Self, DenseTensorError> {
        if meta.dtype() != DType::F64 {
            return Err(DenseTensorError::UnsupportedDType(meta.dtype()));
        }
        Self::from_typed_storage(meta, TensorStorage::F64(Arc::new(storage)))
    }

    pub fn from_storage_f32(meta: TensorMeta, storage: Vec<f32>) -> Result<Self, DenseTensorError> {
        if meta.dtype() != DType::F32 {
            return Err(DenseTensorError::UnsupportedDType(meta.dtype()));
        }
        Self::from_typed_storage(meta, TensorStorage::F32(Arc::new(storage)))
    }

    pub fn from_storage_f16(
        meta: TensorMeta,
        storage: Vec<Float16>,
    ) -> Result<Self, DenseTensorError> {
        if meta.dtype() != DType::F16 {
            return Err(DenseTensorError::UnsupportedDType(meta.dtype()));
        }
        Self::from_typed_storage(meta, TensorStorage::F16(Arc::new(storage)))
    }

    pub fn from_storage_bf16(
        meta: TensorMeta,
        storage: Vec<BFloat16>,
    ) -> Result<Self, DenseTensorError> {
        if meta.dtype() != DType::BF16 {
            return Err(DenseTensorError::UnsupportedDType(meta.dtype()));
        }
        Self::from_typed_storage(meta, TensorStorage::BF16(Arc::new(storage)))
    }

    pub fn from_contiguous_values(
        values: Vec<f64>,
        shape: Vec<usize>,
        device: Device,
    ) -> Result<Self, DenseTensorError> {
        let meta = TensorMeta::from_shape(shape, DType::F64, device);
        Self::from_storage(meta, values)
    }

    pub fn from_contiguous_values_f32(
        values: Vec<f32>,
        shape: Vec<usize>,
        device: Device,
    ) -> Result<Self, DenseTensorError> {
        let meta = TensorMeta::from_shape(shape, DType::F32, device);
        Self::from_storage_f32(meta, values)
    }

    pub fn from_contiguous_values_f16(
        values: Vec<Float16>,
        shape: Vec<usize>,
        device: Device,
    ) -> Result<Self, DenseTensorError> {
        let meta = TensorMeta::from_shape(shape, DType::F16, device);
        Self::from_storage_f16(meta, values)
    }

    pub fn from_contiguous_values_bf16(
        values: Vec<BFloat16>,
        shape: Vec<usize>,
        device: Device,
    ) -> Result<Self, DenseTensorError> {
        let meta = TensorMeta::from_shape(shape, DType::BF16, device);
        Self::from_storage_bf16(meta, values)
    }

    fn contiguous_required_len(meta: &TensorMeta) -> Result<usize, DenseTensorError> {
        contiguous_required_len(meta)
    }

    fn storage_span_required_len(meta: &TensorMeta) -> Result<usize, DenseTensorError> {
        let mut max_linear_offset = 0usize;
        for (size, stride) in meta
            .shape()
            .iter()
            .copied()
            .zip(meta.strides().iter().copied())
        {
            if size == 0 {
                continue;
            }
            let span = stride.checked_mul(size.saturating_sub(1)).ok_or(
                DenseTensorError::StorageSpanOverflow {
                    storage_offset: meta.storage_offset(),
                    numel: meta.numel(),
                },
            )?;
            max_linear_offset = max_linear_offset.checked_add(span).ok_or(
                DenseTensorError::StorageSpanOverflow {
                    storage_offset: meta.storage_offset(),
                    numel: meta.numel(),
                },
            )?;
        }

        if meta.numel() == 0 {
            return Ok(meta.storage_offset());
        }

        let max_index = meta.storage_offset().checked_add(max_linear_offset).ok_or(
            DenseTensorError::StorageSpanOverflow {
                storage_offset: meta.storage_offset(),
                numel: meta.numel(),
            },
        )?;
        max_index
            .checked_add(1)
            .ok_or(DenseTensorError::StorageSpanOverflow {
                storage_offset: meta.storage_offset(),
                numel: meta.numel(),
            })
    }

    pub fn dispatch_values(&self) -> Result<&[f64], DenseTensorError> {
        let start = self.meta.storage_offset();
        let end = Self::storage_span_required_len(&self.meta)?;
        match &self.storage {
            TensorStorage::F64(v) => Ok(&v[start..end]),
            _ => Err(DenseTensorError::UnsupportedDType(self.meta.dtype())),
        }
    }

    pub fn contiguous_values(&self) -> Result<&[f64], DenseTensorError> {
        if !self.meta.is_contiguous() {
            return Err(DenseTensorError::UnsupportedLayout);
        }
        self.dispatch_values()
    }

    pub fn contiguous_values_f32(&self) -> Result<&[f32], DenseTensorError> {
        if !self.meta.is_contiguous() {
            return Err(DenseTensorError::UnsupportedLayout);
        }
        let start = self.meta.storage_offset();
        let end = Self::storage_span_required_len(&self.meta)?;
        match &self.storage {
            TensorStorage::F32(v) => Ok(&v[start..end]),
            _ => Err(DenseTensorError::UnsupportedDType(self.meta.dtype())),
        }
    }

    /// Returns contiguous values as f64, converting from any float type.
    /// Used by backward pass to keep gradient computation in f64.
    pub fn contiguous_values_as_f64(&self) -> Result<Vec<f64>, DenseTensorError> {
        if !self.meta.is_contiguous() {
            return Err(DenseTensorError::UnsupportedLayout);
        }
        let start = self.meta.storage_offset();
        let end = Self::storage_span_required_len(&self.meta)?;
        match &self.storage {
            TensorStorage::F64(v) => Ok(v[start..end].to_vec()),
            TensorStorage::F32(v) => Ok(v[start..end].iter().map(|&x| f64::from(x)).collect()),
            TensorStorage::F16(v) => Ok(v[start..end]
                .iter()
                .map(|&x| f64::from(x.to_f32()))
                .collect()),
            TensorStorage::BF16(v) => Ok(v[start..end]
                .iter()
                .map(|&x| f64::from(x.to_f32()))
                .collect()),
            TensorStorage::Complex64(v) => {
                Ok(v[start..end].iter().map(|z| f64::from(z.re)).collect())
            }
            TensorStorage::Complex128(v) => Ok(v[start..end].iter().map(|z| z.re).collect()),
        }
    }

    #[must_use]
    pub fn typed_storage(&self) -> &TensorStorage {
        &self.storage
    }

    /// Returns the raw f64 storage slice.
    ///
    /// This is only available for `F64` tensors. Callers that need a dtype-agnostic
    /// path should use `typed_storage()` or `contiguous_values_as_f64()`.
    pub fn storage(&self) -> Result<&[f64], DenseTensorError> {
        match &self.storage {
            TensorStorage::F64(v) => Ok(v.as_slice()),
            other => Err(DenseTensorError::UnsupportedStorageAccess {
                dtype: other.dtype(),
            }),
        }
    }

    #[must_use]
    pub fn meta(&self) -> &TensorMeta {
        &self.meta
    }

    #[must_use]
    pub fn id(&self) -> u64 {
        self.id
    }

    #[must_use]
    pub fn storage_id(&self) -> u64 {
        self.storage_id
    }

    #[must_use]
    pub fn version(&self) -> u64 {
        self.version
    }

    /// Cast this tensor to a different floating-point or complex dtype.
    pub fn to_dtype(&self, dtype: DType) -> Result<Self, DenseTensorError> {
        if !dtype.is_floating_point() && !dtype.is_complex() {
            return Err(DenseTensorError::UnsupportedDType(dtype));
        }
        if self.meta.dtype() == dtype {
            return Ok(self.clone());
        }
        let new_meta = self.meta.clone().with_dtype(dtype);
        let as_f32 = || -> Vec<f32> { self.storage.to_f32_vec() };
        let as_f64 = || -> Vec<f64> { self.storage.to_f64_vec() };
        let new_storage = match dtype {
            DType::F64 => TensorStorage::F64(Arc::new(as_f64())),
            DType::F32 => TensorStorage::F32(Arc::new(as_f32())),
            DType::F16 => {
                let vals: Vec<Float16> = as_f32().into_iter().map(Float16::from_f32).collect();
                TensorStorage::F16(Arc::new(vals))
            }
            DType::BF16 => {
                let vals: Vec<BFloat16> = as_f32().into_iter().map(BFloat16::from_f32).collect();
                TensorStorage::BF16(Arc::new(vals))
            }
            DType::Complex64 => {
                let vals: Vec<Complex64> = as_f32()
                    .into_iter()
                    .map(|r| Complex64::new(r, 0.0))
                    .collect();
                TensorStorage::Complex64(Arc::new(vals))
            }
            DType::Complex128 => {
                let vals: Vec<Complex128> = as_f64()
                    .into_iter()
                    .map(|r| Complex128::new(r, 0.0))
                    .collect();
                TensorStorage::Complex128(Arc::new(vals))
            }
            _ => return Err(DenseTensorError::UnsupportedDType(dtype)),
        };
        Self::from_typed_storage(new_meta, new_storage)
    }

    /// Update the contiguous values in-place and bump the version counter.
    ///
    /// The new values must exactly match the length of the contiguous slice.
    pub fn update_contiguous_values(&mut self, new_values: &[f64]) -> Result<(), DenseTensorError> {
        if !self.meta.is_contiguous() {
            return Err(DenseTensorError::UnsupportedLayout);
        }
        let start = self.meta.storage_offset();
        let end = Self::contiguous_required_len(&self.meta)?;
        match &mut self.storage {
            TensorStorage::F64(v) => {
                let buf = Arc::make_mut(v);
                let slice = &mut buf[start..end];
                if new_values.len() != slice.len() {
                    return Err(DenseTensorError::InsufficientStorage {
                        needed: slice.len(),
                        actual: new_values.len(),
                    });
                }
                slice.copy_from_slice(new_values);
            }
            _ => {
                return Err(DenseTensorError::UnsupportedDType(self.meta.dtype()));
            }
        }
        self.version += 1;
        Ok(())
    }

    /// Update the contiguous f32 values in-place and bump the version counter.
    pub fn update_contiguous_values_f32(
        &mut self,
        new_values: &[f32],
    ) -> Result<(), DenseTensorError> {
        if !self.meta.is_contiguous() {
            return Err(DenseTensorError::UnsupportedLayout);
        }
        let start = self.meta.storage_offset();
        let end = Self::contiguous_required_len(&self.meta)?;
        match &mut self.storage {
            TensorStorage::F32(v) => {
                let buf = Arc::make_mut(v);
                let slice = &mut buf[start..end];
                if new_values.len() != slice.len() {
                    return Err(DenseTensorError::InsufficientStorage {
                        needed: slice.len(),
                        actual: new_values.len(),
                    });
                }
                slice.copy_from_slice(new_values);
            }
            _ => {
                return Err(DenseTensorError::UnsupportedDType(self.meta.dtype()));
            }
        }
        self.version += 1;
        Ok(())
    }

    /// Create a view of this tensor with a new shape.
    /// The view shares the same underlying storage (zero-copy).
    /// Only works for contiguous tensors where the new shape has the same numel.
    pub fn view(&self, new_shape: Vec<usize>) -> Result<Self, DenseTensorError> {
        if !self.meta.is_contiguous() {
            return Err(DenseTensorError::UnsupportedLayout);
        }
        let new_numel = checked_shape_numel(&new_shape)?;
        if new_numel != self.meta.numel() {
            return Err(DenseTensorError::InsufficientStorage {
                needed: new_numel,
                actual: self.meta.numel(),
            });
        }
        let new_meta = TensorMeta::from_shape(new_shape, self.meta.dtype(), self.meta.device());
        Ok(Self {
            id: NEXT_TENSOR_ID.fetch_add(1, Ordering::Relaxed),
            storage_id: self.storage_id, // same storage
            meta: new_meta,
            storage: self.storage.clone(), // Arc clone = cheap refcount bump
            version: self.version,
        })
    }

    /// Returns true if this tensor shares storage with another.
    #[must_use]
    pub fn shares_storage_with(&self, other: &Self) -> bool {
        self.storage_id == other.storage_id
    }
}

// ── Integer Tensor Types ───────────────────────────────────────────────

/// Dense tensor backed by `Vec<i64>` storage.
///
/// Integer tensors do NOT participate in autograd (`requires_grad` is always false).
/// They are used for indexing, class labels, and shape operations.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DenseI64Tensor {
    id: u64,
    storage_id: u64,
    meta: TensorMeta,
    storage: Vec<i64>,
    version: u64,
}

impl DenseI64Tensor {
    pub fn from_storage(meta: TensorMeta, storage: Vec<i64>) -> Result<Self, DenseTensorError> {
        meta.validate()?;
        if meta.dtype() != DType::I64 {
            return Err(DenseTensorError::UnsupportedDType(meta.dtype()));
        }
        if !meta.is_contiguous() {
            return Err(DenseTensorError::UnsupportedLayout);
        }
        let needed = contiguous_required_len(&meta)?;
        if storage.len() < needed {
            return Err(DenseTensorError::InsufficientStorage {
                needed,
                actual: storage.len(),
            });
        }
        Ok(Self {
            id: NEXT_TENSOR_ID.fetch_add(1, Ordering::Relaxed),
            storage_id: NEXT_STORAGE_ID.fetch_add(1, Ordering::Relaxed),
            meta,
            storage,
            version: 0,
        })
    }

    pub fn from_contiguous_values(
        values: Vec<i64>,
        shape: Vec<usize>,
        device: Device,
    ) -> Result<Self, DenseTensorError> {
        let meta = TensorMeta::from_shape(shape, DType::I64, device);
        Self::from_storage(meta, values)
    }

    #[must_use]
    pub fn id(&self) -> u64 {
        self.id
    }

    #[must_use]
    pub fn storage_id(&self) -> u64 {
        self.storage_id
    }

    #[must_use]
    pub fn meta(&self) -> &TensorMeta {
        &self.meta
    }

    #[must_use]
    pub fn storage(&self) -> &[i64] {
        &self.storage
    }

    /// Return the contiguous values slice.
    pub fn contiguous_values(&self) -> Result<&[i64], DenseTensorError> {
        if !self.meta.is_contiguous() {
            return Err(DenseTensorError::UnsupportedLayout);
        }
        let start = self.meta.storage_offset();
        let end = contiguous_required_len(&self.meta)?;
        Ok(&self.storage[start..end])
    }

    #[must_use]
    pub fn version(&self) -> u64 {
        self.version
    }
}

/// Dense tensor backed by `Vec<i32>` storage.
///
/// Integer tensors do NOT participate in autograd (`requires_grad` is always false).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DenseI32Tensor {
    id: u64,
    storage_id: u64,
    meta: TensorMeta,
    storage: Vec<i32>,
    version: u64,
}

impl DenseI32Tensor {
    pub fn from_storage(meta: TensorMeta, storage: Vec<i32>) -> Result<Self, DenseTensorError> {
        meta.validate()?;
        if meta.dtype() != DType::I32 {
            return Err(DenseTensorError::UnsupportedDType(meta.dtype()));
        }
        if !meta.is_contiguous() {
            return Err(DenseTensorError::UnsupportedLayout);
        }
        let needed = contiguous_required_len(&meta)?;
        if storage.len() < needed {
            return Err(DenseTensorError::InsufficientStorage {
                needed,
                actual: storage.len(),
            });
        }
        Ok(Self {
            id: NEXT_TENSOR_ID.fetch_add(1, Ordering::Relaxed),
            storage_id: NEXT_STORAGE_ID.fetch_add(1, Ordering::Relaxed),
            meta,
            storage,
            version: 0,
        })
    }

    pub fn from_contiguous_values(
        values: Vec<i32>,
        shape: Vec<usize>,
        device: Device,
    ) -> Result<Self, DenseTensorError> {
        let meta = TensorMeta::from_shape(shape, DType::I32, device);
        Self::from_storage(meta, values)
    }

    #[must_use]
    pub fn id(&self) -> u64 {
        self.id
    }

    #[must_use]
    pub fn storage_id(&self) -> u64 {
        self.storage_id
    }

    #[must_use]
    pub fn meta(&self) -> &TensorMeta {
        &self.meta
    }

    #[must_use]
    pub fn storage(&self) -> &[i32] {
        &self.storage
    }

    /// Return the contiguous values slice.
    pub fn contiguous_values(&self) -> Result<&[i32], DenseTensorError> {
        if !self.meta.is_contiguous() {
            return Err(DenseTensorError::UnsupportedLayout);
        }
        let start = self.meta.storage_offset();
        let end = contiguous_required_len(&self.meta)?;
        Ok(&self.storage[start..end])
    }

    #[must_use]
    pub fn version(&self) -> u64 {
        self.version
    }
}

/// Dense tensor backed by `Vec<u8>` storage (0=false, 1=true).
///
/// Bool tensors do NOT participate in autograd (`requires_grad` is always false).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DenseBoolTensor {
    id: u64,
    storage_id: u64,
    meta: TensorMeta,
    storage: Vec<u8>,
    version: u64,
}

impl DenseBoolTensor {
    pub fn from_storage(meta: TensorMeta, storage: Vec<u8>) -> Result<Self, DenseTensorError> {
        meta.validate()?;
        if meta.dtype() != DType::Bool {
            return Err(DenseTensorError::UnsupportedDType(meta.dtype()));
        }
        if !meta.is_contiguous() {
            return Err(DenseTensorError::UnsupportedLayout);
        }
        let needed = contiguous_required_len(&meta)?;
        if storage.len() < needed {
            return Err(DenseTensorError::InsufficientStorage {
                needed,
                actual: storage.len(),
            });
        }
        Ok(Self {
            id: NEXT_TENSOR_ID.fetch_add(1, Ordering::Relaxed),
            storage_id: NEXT_STORAGE_ID.fetch_add(1, Ordering::Relaxed),
            meta,
            storage,
            version: 0,
        })
    }

    pub fn from_bools(
        values: &[bool],
        shape: Vec<usize>,
        device: Device,
    ) -> Result<Self, DenseTensorError> {
        let storage: Vec<u8> = values.iter().map(|&b| u8::from(b)).collect();
        let meta = TensorMeta::from_shape(shape, DType::Bool, device);
        Self::from_storage(meta, storage)
    }

    #[must_use]
    pub fn id(&self) -> u64 {
        self.id
    }

    #[must_use]
    pub fn storage_id(&self) -> u64 {
        self.storage_id
    }

    #[must_use]
    pub fn meta(&self) -> &TensorMeta {
        &self.meta
    }

    #[must_use]
    pub fn storage(&self) -> &[u8] {
        &self.storage
    }

    /// Return the contiguous values slice as u8 (0=false, 1=true).
    pub fn contiguous_values(&self) -> Result<&[u8], DenseTensorError> {
        if !self.meta.is_contiguous() {
            return Err(DenseTensorError::UnsupportedLayout);
        }
        let start = self.meta.storage_offset();
        let end = contiguous_required_len(&self.meta)?;
        Ok(&self.storage[start..end])
    }

    /// Return the contiguous values as a Vec<bool>.
    pub fn contiguous_bools(&self) -> Result<Vec<bool>, DenseTensorError> {
        let values = self.contiguous_values()?;
        Ok(values.iter().map(|&v| v != 0).collect())
    }

    #[must_use]
    pub fn version(&self) -> u64 {
        self.version
    }
}

pub fn ensure_compatible(lhs: &ScalarTensor, rhs: &ScalarTensor) -> Result<(), TensorCompatError> {
    if lhs.meta().dtype() != rhs.meta().dtype() {
        return Err(TensorCompatError::DTypeMismatch {
            lhs: lhs.meta().dtype(),
            rhs: rhs.meta().dtype(),
        });
    }

    if lhs.meta().device() != rhs.meta().device() {
        return Err(TensorCompatError::DeviceMismatch {
            lhs: lhs.meta().device(),
            rhs: rhs.meta().device(),
        });
    }

    Ok(())
}

#[must_use]
pub fn contiguous_strides(shape: &[usize]) -> Vec<usize> {
    if shape.is_empty() {
        return Vec::new();
    }

    let mut strides = vec![1; shape.len()];
    let mut running = 1usize;
    for idx in (0..shape.len()).rev() {
        strides[idx] = running;
        running = running.saturating_mul(shape[idx]);
    }
    strides
}

// ── Sparse Tensor Types ────────────────────────────────────────────────

/// Error type for sparse tensor operations.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SparseTensorError {
    /// Indices tensor has wrong rank (expected 2 for COO).
    InvalidIndicesRank { expected: usize, actual: usize },
    /// Indices tensor has wrong dtype (must be I64).
    InvalidIndicesDType { actual: DType },
    /// Indices tensor is on a different device than values.
    DeviceMismatch { expected: Device, actual: Device },
    /// Indices sparse_dim doesn't match the dense shape.
    SparseDimMismatch {
        indices_sparse_dim: usize,
        expected: usize,
    },
    /// Number of non-zero entries doesn't match between indices and values.
    NnzMismatch {
        indices_nnz: usize,
        values_nnz: usize,
    },
    /// Values tensor shape doesn't match expected [nnz, *dense_dims].
    InvalidValuesShape {
        expected: Vec<usize>,
        actual: Vec<usize>,
    },
    /// Index out of bounds for the dense shape.
    IndexOutOfBounds { dim: usize, index: i64, size: usize },
    /// Negative index in indices tensor.
    NegativeIndex { dim: usize, index: i64 },
    /// CSR crow_indices has wrong length (expected nrows + 1).
    InvalidCrowIndicesLen { expected: usize, actual: usize },
    /// CSR col_indices has wrong length (expected nnz).
    InvalidColIndicesLen { expected: usize, actual: usize },
    /// CSR crow_indices values are not monotonically increasing.
    NonMonotonicCrowIndices { row: usize, prev: i64, curr: i64 },
    /// CSR crow_indices contains an invalid value.
    InvalidCrowIndexValue { index: usize, value: i64 },
    /// CSR column index out of bounds.
    ColIndexOutOfBounds { index: i64, ncols: usize },
    /// Dense tensor error during conversion.
    DenseTensor(DenseTensorError),
    /// Only 2D sparse CSR tensors are supported.
    UnsupportedRank { rank: usize },
}

impl fmt::Display for SparseTensorError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidIndicesRank { expected, actual } => {
                write!(f, "indices tensor has rank {actual}, expected {expected}")
            }
            Self::InvalidIndicesDType { actual } => {
                write!(f, "indices tensor has dtype {actual:?}, expected I64")
            }
            Self::DeviceMismatch { expected, actual } => {
                write!(
                    f,
                    "sparse tensor device mismatch: expected {expected:?}, got {actual:?}"
                )
            }
            Self::SparseDimMismatch {
                indices_sparse_dim,
                expected,
            } => {
                write!(
                    f,
                    "indices sparse_dim is {indices_sparse_dim}, expected {expected}"
                )
            }
            Self::NnzMismatch {
                indices_nnz,
                values_nnz,
            } => {
                write!(
                    f,
                    "indices nnz ({indices_nnz}) != values nnz ({values_nnz})"
                )
            }
            Self::InvalidValuesShape { expected, actual } => {
                write!(f, "values shape is {actual:?}, expected {expected:?}")
            }
            Self::IndexOutOfBounds { dim, index, size } => {
                write!(
                    f,
                    "index {index} at dim {dim} out of bounds for size {size}"
                )
            }
            Self::NegativeIndex { dim, index } => {
                write!(f, "negative index {index} at dim {dim}")
            }
            Self::InvalidCrowIndicesLen { expected, actual } => {
                write!(f, "crow_indices length is {actual}, expected {expected}")
            }
            Self::InvalidColIndicesLen { expected, actual } => {
                write!(f, "col_indices length is {actual}, expected {expected}")
            }
            Self::NonMonotonicCrowIndices { row, prev, curr } => {
                write!(
                    f,
                    "crow_indices not monotonic at row {row}: {prev} > {curr}"
                )
            }
            Self::InvalidCrowIndexValue { index, value } => {
                write!(f, "crow_indices[{index}] has invalid value {value}")
            }
            Self::ColIndexOutOfBounds { index, ncols } => {
                write!(f, "column index {index} out of bounds for {ncols} columns")
            }
            Self::DenseTensor(err) => write!(f, "dense tensor error: {err}"),
            Self::UnsupportedRank { rank } => {
                write!(f, "sparse CSR only supports 2D tensors, got rank {rank}")
            }
        }
    }
}

impl std::error::Error for SparseTensorError {}

impl From<DenseTensorError> for SparseTensorError {
    fn from(value: DenseTensorError) -> Self {
        Self::DenseTensor(value)
    }
}

/// Sparse tensor in COO (Coordinate) format.
///
/// COO format stores sparse tensors as a pair of tensors:
/// - `indices`: shape [sparse_dim, nnz], dtype I64 — the coordinates of non-zero elements
/// - `values`: shape [nnz, *dense_dims] — the values at those coordinates
///
/// For a sparse tensor with shape [3, 4, 5] and sparse_dim=2:
/// - indices has shape [2, nnz] (row and column indices)
/// - values has shape [nnz, 5] (the remaining dense dimension)
///
/// The `coalesced` flag indicates whether duplicate indices have been merged.
/// A coalesced tensor has unique, sorted indices.
#[derive(Debug, Clone, PartialEq)]
pub struct SparseCOOTensor {
    id: u64,
    /// The indices of non-zero elements, shape [sparse_dim, nnz], dtype I64.
    indices: DenseI64Tensor,
    /// The values at the indexed positions, shape [nnz, *dense_dims].
    values: DenseTensor,
    /// The full dense shape this sparse tensor represents.
    dense_shape: Vec<usize>,
    /// Number of sparse dimensions (indices.shape[0]).
    sparse_dim: usize,
    /// Whether the tensor is coalesced (no duplicate indices, sorted).
    coalesced: bool,
    device: Device,
    version: u64,
}

impl SparseCOOTensor {
    /// Create a new sparse COO tensor.
    ///
    /// # Arguments
    /// * `indices` - shape [sparse_dim, nnz], dtype I64
    /// * `values` - shape [nnz, *dense_dims], any floating-point dtype
    /// * `dense_shape` - the full shape if this were a dense tensor
    /// * `coalesced` - whether indices are unique and sorted
    ///
    /// # Errors
    /// Returns error if indices/values shapes don't match or indices are out of bounds.
    pub fn new(
        indices: DenseI64Tensor,
        values: DenseTensor,
        dense_shape: Vec<usize>,
        coalesced: bool,
    ) -> Result<Self, SparseTensorError> {
        // Validate indices shape: must be [sparse_dim, nnz]
        let indices_shape = indices.meta().shape();
        if indices_shape.len() != 2 {
            return Err(SparseTensorError::InvalidIndicesRank {
                expected: 2,
                actual: indices_shape.len(),
            });
        }

        let sparse_dim = indices_shape[0];
        let nnz = indices_shape[1];

        // sparse_dim must not exceed dense_shape rank
        if sparse_dim > dense_shape.len() {
            return Err(SparseTensorError::SparseDimMismatch {
                indices_sparse_dim: sparse_dim,
                expected: dense_shape.len(),
            });
        }

        // Validate values shape: must be [nnz, *dense_dims]
        let values_shape = values.meta().shape();
        if values_shape.is_empty() {
            return Err(SparseTensorError::InvalidValuesShape {
                expected: vec![nnz],
                actual: values_shape.to_vec(),
            });
        }

        if values_shape[0] != nnz {
            return Err(SparseTensorError::NnzMismatch {
                indices_nnz: nnz,
                values_nnz: values_shape[0],
            });
        }

        // dense_dims are the remaining dimensions after sparse_dim
        let dense_dims = &dense_shape[sparse_dim..];
        let expected_values_shape: Vec<usize> = std::iter::once(nnz)
            .chain(dense_dims.iter().copied())
            .collect();

        if values_shape != expected_values_shape.as_slice() {
            return Err(SparseTensorError::InvalidValuesShape {
                expected: expected_values_shape,
                actual: values_shape.to_vec(),
            });
        }

        let device = values.meta().device();
        if indices.meta().device() != device {
            return Err(SparseTensorError::DeviceMismatch {
                expected: device,
                actual: indices.meta().device(),
            });
        }

        // Validate indices are within bounds even when not coalesced.
        let indices_values = indices.storage();
        for d in 0..sparse_dim {
            let dim_size = dense_shape[d];
            for i in 0..nnz {
                let idx = indices_values[d * nnz + i];
                if idx < 0 {
                    return Err(SparseTensorError::NegativeIndex { dim: d, index: idx });
                }
                if (idx as usize) >= dim_size {
                    return Err(SparseTensorError::IndexOutOfBounds {
                        dim: d,
                        index: idx,
                        size: dim_size,
                    });
                }
            }
        }

        Ok(Self {
            id: NEXT_TENSOR_ID.fetch_add(1, Ordering::Relaxed),
            indices,
            values,
            dense_shape,
            sparse_dim,
            coalesced,
            device,
            version: 0,
        })
    }

    /// Create a sparse COO tensor from coordinate lists.
    ///
    /// # Arguments
    /// * `coords` - list of coordinate tuples, each of length sparse_dim
    /// * `values` - flat values for each coordinate
    /// * `dense_shape` - the full dense shape
    /// * `dtype` - dtype for the values
    /// * `device` - device for the tensor
    pub fn from_coords(
        coords: &[Vec<i64>],
        values: Vec<f64>,
        dense_shape: Vec<usize>,
        dtype: DType,
        device: Device,
    ) -> Result<Self, SparseTensorError> {
        let nnz = coords.len();
        if nnz == 0 {
            // Empty sparse tensor
            let sparse_dim = dense_shape.len();
            let indices =
                DenseI64Tensor::from_contiguous_values(vec![], vec![sparse_dim, 0], device)?;
            let values_tensor = DenseTensor::from_contiguous_values(vec![], vec![0], device)?;
            return Self::new(indices, values_tensor, dense_shape, true);
        }

        let sparse_dim = coords[0].len();

        // Build indices tensor [sparse_dim, nnz]
        let mut indices_data = vec![0i64; sparse_dim * nnz];
        for (i, coord) in coords.iter().enumerate() {
            if coord.len() != sparse_dim {
                return Err(SparseTensorError::SparseDimMismatch {
                    indices_sparse_dim: coord.len(),
                    expected: sparse_dim,
                });
            }
            for (d, &idx) in coord.iter().enumerate() {
                indices_data[d * nnz + i] = idx;
            }
        }

        let indices =
            DenseI64Tensor::from_contiguous_values(indices_data, vec![sparse_dim, nnz], device)?;

        let values_tensor = DenseTensor::from_contiguous_values(values, vec![nnz], device)?;
        let values_tensor = values_tensor.to_dtype(dtype)?;

        Self::new(indices, values_tensor, dense_shape, false)
    }

    #[must_use]
    pub fn id(&self) -> u64 {
        self.id
    }

    #[must_use]
    pub fn indices(&self) -> &DenseI64Tensor {
        &self.indices
    }

    #[must_use]
    pub fn values(&self) -> &DenseTensor {
        &self.values
    }

    #[must_use]
    pub fn dense_shape(&self) -> &[usize] {
        &self.dense_shape
    }

    #[must_use]
    pub fn sparse_dim(&self) -> usize {
        self.sparse_dim
    }

    #[must_use]
    pub fn nnz(&self) -> usize {
        self.indices.meta().shape().get(1).copied().unwrap_or(0)
    }

    #[must_use]
    pub fn is_coalesced(&self) -> bool {
        self.coalesced
    }

    #[must_use]
    pub fn dtype(&self) -> DType {
        self.values.meta().dtype()
    }

    #[must_use]
    pub fn device(&self) -> Device {
        self.device
    }

    #[must_use]
    pub fn version(&self) -> u64 {
        self.version
    }

    /// Convert this sparse tensor to a dense tensor.
    ///
    /// This allocates a new dense tensor with zeros and fills in the non-zero values.
    pub fn to_dense(&self) -> Result<DenseTensor, SparseTensorError> {
        let numel = checked_shape_numel(&self.dense_shape)?;
        let mut dense_data = vec![0.0f64; numel];

        let nnz = self.nnz();
        if nnz == 0 {
            return Ok(DenseTensor::from_contiguous_values(
                dense_data,
                self.dense_shape.clone(),
                self.device,
            )?);
        }

        let indices_data = self.indices.storage();
        let values_data = self.values.contiguous_values_as_f64()?;
        let strides = contiguous_strides(&self.dense_shape);

        // For now, only support fully sparse tensors (sparse_dim == rank)
        // where values are scalars
        if self.sparse_dim == self.dense_shape.len() {
            for i in 0..nnz {
                let mut linear_idx = 0usize;
                for d in 0..self.sparse_dim {
                    let idx = indices_data[d * nnz + i];
                    if idx < 0 || (idx as usize) >= self.dense_shape[d] {
                        return Err(SparseTensorError::IndexOutOfBounds {
                            dim: d,
                            index: idx,
                            size: self.dense_shape[d],
                        });
                    }
                    linear_idx += (idx as usize) * strides[d];
                }
                dense_data[linear_idx] += values_data[i];
            }
        } else {
            // Hybrid sparse-dense: values have shape [nnz, *dense_dims]
            let dense_dims = &self.dense_shape[self.sparse_dim..];
            let dense_numel = checked_shape_numel(dense_dims)?;

            for i in 0..nnz {
                let mut sparse_linear_idx = 0usize;
                for d in 0..self.sparse_dim {
                    let idx = indices_data[d * nnz + i];
                    if idx < 0 || (idx as usize) >= self.dense_shape[d] {
                        return Err(SparseTensorError::IndexOutOfBounds {
                            dim: d,
                            index: idx,
                            size: self.dense_shape[d],
                        });
                    }
                    sparse_linear_idx += (idx as usize) * strides[d];
                }
                // Copy the dense slice
                for j in 0..dense_numel {
                    dense_data[sparse_linear_idx + j] += values_data[i * dense_numel + j];
                }
            }
        }

        let result =
            DenseTensor::from_contiguous_values(dense_data, self.dense_shape.clone(), self.device)?;
        Ok(result.to_dtype(self.dtype())?)
    }
}

/// Sparse tensor in CSR (Compressed Sparse Row) format.
///
/// CSR format is efficient for row-wise operations on 2D sparse matrices:
/// - `crow_indices`: shape [nrows + 1], dtype I64 — row pointers
/// - `col_indices`: shape [nnz], dtype I64 — column indices for each non-zero
/// - `values`: shape [nnz] — the non-zero values
///
/// For row `i`, the non-zeros are at positions crow_indices[i]..crow_indices[i+1]
/// in col_indices and values.
#[derive(Debug, Clone, PartialEq)]
pub struct SparseCSRTensor {
    id: u64,
    /// Row pointers, shape [nrows + 1], dtype I64.
    crow_indices: DenseI64Tensor,
    /// Column indices for each non-zero, shape [nnz], dtype I64.
    col_indices: DenseI64Tensor,
    /// Values at each non-zero position, shape [nnz].
    values: DenseTensor,
    /// The 2D shape [nrows, ncols].
    shape: [usize; 2],
    device: Device,
    version: u64,
}

impl SparseCSRTensor {
    /// Create a new sparse CSR tensor.
    ///
    /// # Arguments
    /// * `crow_indices` - shape [nrows + 1], dtype I64
    /// * `col_indices` - shape [nnz], dtype I64
    /// * `values` - shape [nnz], any floating-point dtype
    /// * `shape` - [nrows, ncols]
    ///
    /// # Errors
    /// Returns error if shapes don't match or indices are invalid.
    pub fn new(
        crow_indices: DenseI64Tensor,
        col_indices: DenseI64Tensor,
        values: DenseTensor,
        shape: [usize; 2],
    ) -> Result<Self, SparseTensorError> {
        let [nrows, ncols] = shape;

        // Validate crow_indices: shape [nrows + 1]
        let crow_shape = crow_indices.meta().shape();
        if crow_shape.len() != 1 {
            return Err(SparseTensorError::UnsupportedRank {
                rank: crow_shape.len(),
            });
        }
        let expected_crow_len = nrows.checked_add(1).ok_or_else(|| {
            SparseTensorError::DenseTensor(DenseTensorError::ShapeOverflow {
                shape: vec![nrows, ncols],
            })
        })?;
        if crow_shape[0] != expected_crow_len {
            return Err(SparseTensorError::InvalidCrowIndicesLen {
                expected: expected_crow_len,
                actual: crow_shape[0],
            });
        }

        let crow_data = crow_indices.storage();

        if crow_data[0] != 0 {
            return Err(SparseTensorError::InvalidCrowIndexValue {
                index: 0,
                value: crow_data[0],
            });
        }

        let nnz = usize::try_from(crow_data[nrows]).map_err(|_| {
            SparseTensorError::InvalidCrowIndexValue {
                index: nrows,
                value: crow_data[nrows],
            }
        })?;

        // Validate col_indices: shape [nnz]
        let col_shape = col_indices.meta().shape();
        if col_shape.len() != 1 {
            return Err(SparseTensorError::UnsupportedRank {
                rank: col_shape.len(),
            });
        }
        if col_shape[0] != nnz {
            return Err(SparseTensorError::InvalidColIndicesLen {
                expected: nnz,
                actual: col_shape[0],
            });
        }

        // Validate values: shape [nnz]
        let values_shape = values.meta().shape();
        if values_shape.len() != 1 {
            return Err(SparseTensorError::UnsupportedRank {
                rank: values_shape.len(),
            });
        }
        if values_shape[0] != nnz {
            return Err(SparseTensorError::NnzMismatch {
                indices_nnz: nnz,
                values_nnz: values_shape[0],
            });
        }

        let device = values.meta().device();
        if crow_indices.meta().device() != device {
            return Err(SparseTensorError::DeviceMismatch {
                expected: device,
                actual: crow_indices.meta().device(),
            });
        }
        if col_indices.meta().device() != device {
            return Err(SparseTensorError::DeviceMismatch {
                expected: device,
                actual: col_indices.meta().device(),
            });
        }

        // Validate crow_indices is monotonically increasing and within bounds.
        for i in 0..=nrows {
            let value = crow_data[i];
            if value < 0 {
                return Err(SparseTensorError::InvalidCrowIndexValue { index: i, value });
            }
            if i > 0 && crow_data[i - 1] > value {
                return Err(SparseTensorError::NonMonotonicCrowIndices {
                    row: i - 1,
                    prev: crow_data[i - 1],
                    curr: value,
                });
            }
        }

        // Validate col_indices are in bounds
        let col_data = col_indices.storage();
        for &col in col_data {
            if col < 0 || (col as usize) >= ncols {
                return Err(SparseTensorError::ColIndexOutOfBounds { index: col, ncols });
            }
        }

        Ok(Self {
            id: NEXT_TENSOR_ID.fetch_add(1, Ordering::Relaxed),
            crow_indices,
            col_indices,
            values,
            shape,
            device,
            version: 0,
        })
    }

    #[must_use]
    pub fn id(&self) -> u64 {
        self.id
    }

    #[must_use]
    pub fn crow_indices(&self) -> &DenseI64Tensor {
        &self.crow_indices
    }

    #[must_use]
    pub fn col_indices(&self) -> &DenseI64Tensor {
        &self.col_indices
    }

    #[must_use]
    pub fn values(&self) -> &DenseTensor {
        &self.values
    }

    #[must_use]
    pub fn shape(&self) -> [usize; 2] {
        self.shape
    }

    #[must_use]
    pub fn nrows(&self) -> usize {
        self.shape[0]
    }

    #[must_use]
    pub fn ncols(&self) -> usize {
        self.shape[1]
    }

    #[must_use]
    pub fn nnz(&self) -> usize {
        self.col_indices
            .meta()
            .shape()
            .first()
            .copied()
            .unwrap_or(0)
    }

    #[must_use]
    pub fn dtype(&self) -> DType {
        self.values.meta().dtype()
    }

    #[must_use]
    pub fn device(&self) -> Device {
        self.device
    }

    #[must_use]
    pub fn version(&self) -> u64 {
        self.version
    }

    /// Convert this sparse CSR tensor to a dense tensor.
    pub fn to_dense(&self) -> Result<DenseTensor, SparseTensorError> {
        let [nrows, ncols] = self.shape;
        let numel = nrows
            .checked_mul(ncols)
            .ok_or_else(|| DenseTensorError::ShapeOverflow {
                shape: vec![nrows, ncols],
            })?;
        let mut dense_data = vec![0.0f64; numel];

        let crow_data = self.crow_indices.storage();
        let col_data = self.col_indices.storage();
        let values_data = self.values.contiguous_values_as_f64()?;

        for row in 0..nrows {
            let start = crow_data[row] as usize;
            let end = crow_data[row + 1] as usize;
            for idx in start..end {
                let col = col_data[idx] as usize;
                dense_data[row * ncols + col] = values_data[idx];
            }
        }

        let result =
            DenseTensor::from_contiguous_values(dense_data, vec![nrows, ncols], self.device)?;
        Ok(result.to_dtype(self.dtype())?)
    }
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeMap;

    use proptest::prelude::*;

    use std::sync::Arc;

    use super::{
        BFloat16, DType, DenseBoolTensor, DenseI32Tensor, DenseI64Tensor, DenseTensor,
        DenseTensorError, Device, Float16, ScalarTensor, SparseCOOTensor, SparseCSRTensor,
        SparseTensorError, TensorMeta, TensorMetaError, TensorStorage, contiguous_strides,
        ensure_compatible,
    };

    fn det_seed(parts: &[usize]) -> u64 {
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
        let scenario_id = format!("ft_core_property/{mode}:{test_id}");
        log.insert("ts_utc".to_string(), "1970-01-01T00:00:00Z".to_string());
        log.insert("suite_id".to_string(), "ft_core_property".to_string());
        log.insert("test_id".to_string(), test_id.to_string());
        log.insert("packet_id".to_string(), "FT-P2C-001".to_string());
        log.insert(
            "fixture_id".to_string(),
            "ft_core_property_generated".to_string(),
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
            "det64:ft-core-test".to_string(),
        );
        log.insert(
            "artifact_refs".to_string(),
            "artifacts/phase2c/FT-P2C-001/fixture_manifest.json".to_string(),
        );
        log.insert(
            "replay_command".to_string(),
            format!("cargo test -p ft-core {test_id} -- --nocapture"),
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

    #[test]
    fn scalar_meta_is_valid() {
        let meta = TensorMeta::scalar(DType::F64, Device::Cpu);
        assert!(meta.validate().is_ok());
        assert!(meta.shape().is_empty());
        assert!(meta.strides().is_empty());
        assert_eq!(meta.numel(), 1);
        assert!(meta.is_contiguous());
    }

    #[test]
    fn dense_tensor_from_contiguous_values_accepts_matching_storage() {
        let tensor = DenseTensor::from_contiguous_values(vec![1.0, 2.0, 3.0], vec![3], Device::Cpu)
            .expect("contiguous dense tensor should build");
        assert_eq!(
            tensor.contiguous_values().expect("slice should resolve"),
            &[1.0, 2.0, 3.0]
        );
        assert_eq!(tensor.meta().shape(), &[3]);
        assert_eq!(tensor.meta().dtype(), DType::F64);
    }

    #[test]
    fn dense_tensor_contiguous_view_rejects_non_contiguous_layout() {
        let meta =
            TensorMeta::from_shape_and_strides(vec![2, 2], vec![4, 1], 0, DType::F64, Device::Cpu)
                .expect("meta should validate");
        let tensor = DenseTensor::from_storage(meta, vec![1.0; 6])
            .expect("non-contiguous metadata should still construct tensor storage");
        let err = tensor
            .contiguous_values()
            .expect_err("non-contiguous layout must fail contiguous view");
        assert!(matches!(err, DenseTensorError::UnsupportedLayout));
    }

    #[test]
    fn dense_tensor_rejects_insufficient_storage_for_offset() {
        let meta = TensorMeta::from_shape(vec![3], DType::F64, Device::Cpu).with_storage_offset(2);
        let err = DenseTensor::from_storage(meta, vec![1.0, 2.0, 3.0])
            .expect_err("offset span must require enough storage");
        assert!(matches!(
            err,
            DenseTensorError::InsufficientStorage {
                needed: 5,
                actual: 3
            }
        ));
    }

    #[test]
    fn dense_tensor_rejects_insufficient_storage_for_strided_layout() {
        let meta =
            TensorMeta::from_shape_and_strides(vec![2, 2], vec![4, 1], 0, DType::F64, Device::Cpu)
                .expect("strided meta should validate");
        let err = DenseTensor::from_storage(meta, vec![1.0; 5])
            .expect_err("strided span must require enough backing storage");
        assert!(matches!(
            err,
            DenseTensorError::InsufficientStorage {
                needed: 6,
                actual: 5
            }
        ));
    }

    #[test]
    fn shape_builds_contiguous_strides() {
        let meta = TensorMeta::from_shape(vec![2, 3, 4], DType::F64, Device::Cpu);
        assert_eq!(meta.strides(), &[12, 4, 1]);
        assert_eq!(meta.numel(), 24);
        assert!(meta.is_contiguous());
    }

    #[test]
    fn singleton_dim_stride_variation_is_still_contiguous() {
        let broadcast_meta =
            TensorMeta::from_shape_and_strides(vec![1, 3], vec![0, 1], 0, DType::F64, Device::Cpu)
                .expect("broadcast shape should validate");
        assert!(
            broadcast_meta.is_contiguous(),
            "singleton stride should not break contiguous semantics"
        );

        let interior_singleton_meta = TensorMeta::from_shape_and_strides(
            vec![2, 1, 4],
            vec![4, 99, 1],
            0,
            DType::F64,
            Device::Cpu,
        )
        .expect("interior singleton stride should validate");
        assert!(
            interior_singleton_meta.is_contiguous(),
            "interior singleton stride should be ignored for contiguity"
        );
    }

    #[test]
    fn non_singleton_stride_mismatch_is_not_contiguous() {
        let meta =
            TensorMeta::from_shape_and_strides(vec![2, 3], vec![0, 1], 0, DType::F64, Device::Cpu)
                .expect("meta should validate");
        assert!(
            !meta.is_contiguous(),
            "non-singleton zero stride should fail contiguous check"
        );
    }

    #[test]
    fn custom_strides_validate_and_index_into_storage() {
        let meta =
            TensorMeta::from_shape_and_strides(vec![2, 2], vec![4, 1], 3, DType::F64, Device::Cpu)
                .expect("meta should validate");

        assert_eq!(meta.storage_index_for(&[0, 0]).expect("index 0,0"), 3);
        assert_eq!(meta.storage_index_for(&[1, 1]).expect("index 1,1"), 8);
    }

    #[test]
    fn index_rank_and_bounds_are_guarded() {
        let meta = TensorMeta::from_shape(vec![2, 3], DType::F64, Device::Cpu);

        let rank_err = meta
            .storage_index_for(&[1])
            .expect_err("rank mismatch should fail");
        assert!(matches!(
            rank_err,
            TensorMetaError::IndexRankMismatch {
                expected: 2,
                actual: 1
            }
        ));

        let oob_err = meta
            .storage_index_for(&[2, 0])
            .expect_err("out-of-bounds index should fail");
        assert!(matches!(
            oob_err,
            TensorMetaError::IndexOutOfBounds {
                dim: 0,
                index: 2,
                size: 2
            }
        ));
    }

    #[test]
    fn validate_rejects_stride_overflow() {
        let err = TensorMeta::from_shape_and_strides(
            vec![3],
            vec![usize::MAX],
            0,
            DType::F64,
            Device::Cpu,
        )
        .expect_err("overflowing stride span must fail validation");

        assert!(matches!(
            err,
            TensorMetaError::StrideOverflow { size: 3, stride } if stride == usize::MAX
        ));
    }

    #[test]
    fn storage_index_for_rejects_storage_offset_overflow() {
        let meta = TensorMeta::from_shape(vec![2], DType::F64, Device::Cpu)
            .with_storage_offset(usize::MAX);
        let err = meta
            .storage_index_for(&[1])
            .expect_err("overflowing storage offset accumulation must fail");

        assert!(matches!(
            err,
            TensorMetaError::StorageOffsetOverflow {
                storage_offset,
                max_linear_offset: 1
            } if storage_offset == usize::MAX
        ));
    }

    #[test]
    fn compatibility_checks_dtype_and_device() {
        let lhs = ScalarTensor::new(1.0, DType::F64, Device::Cpu);
        let rhs = ScalarTensor::new(2.0, DType::F64, Device::Cpu);
        assert!(ensure_compatible(&lhs, &rhs).is_ok());
    }

    #[test]
    fn compatibility_checks_reject_dtype_mismatch() {
        let lhs = ScalarTensor::new(1.0, DType::F64, Device::Cpu);
        let rhs = ScalarTensor::new(2.0, DType::F32, Device::Cpu);
        let err = ensure_compatible(&lhs, &rhs).expect_err("dtype mismatch must fail");
        assert!(matches!(
            err,
            super::TensorCompatError::DTypeMismatch {
                lhs: DType::F64,
                rhs: DType::F32
            }
        ));
    }

    #[test]
    fn compatibility_checks_reject_device_mismatch() {
        let lhs = ScalarTensor::new(1.0, DType::F64, Device::Cpu);
        let rhs = ScalarTensor::new(2.0, DType::F64, Device::Cuda);
        let err = ensure_compatible(&lhs, &rhs).expect_err("device mismatch must fail");
        assert!(matches!(
            err,
            super::TensorCompatError::DeviceMismatch {
                lhs: Device::Cpu,
                rhs: Device::Cuda
            }
        ));
    }

    #[test]
    fn contiguous_stride_helper_handles_scalar() {
        assert_eq!(contiguous_strides(&[]), Vec::<usize>::new());
    }

    #[test]
    fn numel_saturates_on_overflow() {
        let meta = TensorMeta::from_shape(vec![usize::MAX, 2], DType::F64, Device::Cpu);
        assert_eq!(meta.numel(), usize::MAX);
    }

    #[test]
    fn numel_zero_dimension_short_circuits_before_overflow() {
        let meta = TensorMeta::from_shape(vec![usize::MAX, 2, 0], DType::F64, Device::Cpu);
        assert_eq!(meta.numel(), 0);
    }

    #[test]
    fn dense_view_rejects_shape_overflow() {
        let tensor = DenseTensor::from_contiguous_values(vec![1.0], vec![1], Device::Cpu).unwrap();
        let err = tensor
            .view(vec![usize::MAX, 2])
            .expect_err("overflowing view shape must fail");
        assert!(matches!(err, DenseTensorError::ShapeOverflow { .. }));
    }

    #[test]
    fn sparse_coo_to_dense_rejects_shape_overflow() {
        let coords = vec![vec![0, 0]];
        let sparse = SparseCOOTensor::from_coords(
            &coords,
            vec![1.0],
            vec![usize::MAX, 2],
            DType::F64,
            Device::Cpu,
        )
        .expect("sparse COO should build with oversized shape");
        let err = sparse
            .to_dense()
            .expect_err("overflowing dense shape must fail");
        assert!(matches!(
            err,
            SparseTensorError::DenseTensor(DenseTensorError::ShapeOverflow { .. })
        ));
    }

    #[test]
    fn out_of_place_result_gets_new_storage_and_version_bump() {
        let source = ScalarTensor::new(2.0, DType::F64, Device::Cpu);
        let derived = source.with_value(5.0);

        assert_ne!(source.id(), derived.id());
        assert_ne!(source.storage_id(), derived.storage_id());
        assert_eq!(derived.version(), source.version() + 1);
    }

    #[test]
    fn alias_view_shares_storage_identity() {
        let source = ScalarTensor::new(2.0, DType::F64, Device::Cpu);
        let alias = source.alias_view(0).expect("alias with zero offset");

        assert_ne!(source.id(), alias.id());
        assert_eq!(source.storage_id(), alias.storage_id());
        assert_eq!(source.version(), alias.version());
        assert_eq!(source.value(), alias.value());
    }

    #[test]
    fn in_place_updates_bump_version_and_fingerprint() {
        let mut tensor = ScalarTensor::new(2.0, DType::F64, Device::Cpu);
        let before = tensor.evidence_fingerprint64();
        tensor.set_in_place(7.0);
        let after = tensor.evidence_fingerprint64();

        assert_eq!(tensor.value(), 7.0);
        assert_eq!(tensor.version(), 1);
        assert_ne!(before, after);
    }

    #[test]
    fn meta_fingerprint_changes_when_offset_changes() {
        let a = TensorMeta::from_shape(vec![2, 2], DType::F64, Device::Cpu);
        let b = a.clone().with_storage_offset(1);
        assert_ne!(a.fingerprint64(), b.fingerprint64());
    }

    proptest! {
        #[test]
        fn prop_contiguous_stride_contract(shape in prop::collection::vec(1usize..=4, 1..=4)) {
            let strides = contiguous_strides(shape.as_slice());
            prop_assert_eq!(strides.len(), shape.len());
            prop_assert_eq!(strides.last().copied(), Some(1));

            let seed = det_seed(shape.as_slice());
            let log = build_property_log(
                "prop_contiguous_stride_contract",
                "strict",
                seed,
                seed,
                det_seed(strides.as_slice()),
                "contiguous_stride_contract_ok",
            );
            assert_log_contract(&log);
        }

        #[test]
        fn prop_numel_matches_shape_product(shape in prop::collection::vec(1usize..=6, 1..=4)) {
            let meta = TensorMeta::from_shape(shape.clone(), DType::F64, Device::Cpu);
            let expected: usize = shape.iter().copied().product();
            prop_assert_eq!(meta.numel(), expected);

            let seed = det_seed(shape.as_slice());
            let log = build_property_log(
                "prop_numel_matches_shape_product",
                "strict",
                seed,
                seed,
                expected as u64,
                "numel_product_contract_ok",
            );
            assert_log_contract(&log);
        }

        #[test]
        fn prop_contiguous_index_bounds(shape in prop::collection::vec(1usize..=4, 1..=4)) {
            let meta = TensorMeta::from_shape(shape.clone(), DType::F64, Device::Cpu);
            let zero_index = vec![0; shape.len()];
            let max_index = shape.iter().map(|dim| dim - 1).collect::<Vec<_>>();

            let zero_linear = meta.storage_index_for(zero_index.as_slice()).expect("zero index must be valid");
            let max_linear = meta.storage_index_for(max_index.as_slice()).expect("max index must be valid");

            prop_assert_eq!(zero_linear, 0);
            prop_assert!(max_linear < meta.numel());

            let seed = det_seed(shape.as_slice());
            let log = build_property_log(
                "prop_contiguous_index_bounds",
                "strict",
                seed,
                seed,
                max_linear as u64,
                "index_bounds_contract_ok",
            );
            assert_log_contract(&log);
        }

        #[test]
        fn prop_with_value_bumps_version_and_storage(
            source_value in -1_000.0f64..1_000.0f64,
            derived_value in -1_000.0f64..1_000.0f64,
        ) {
            let source = ScalarTensor::new(source_value, DType::F64, Device::Cpu);
            let derived = source.with_value(derived_value);

            prop_assert_eq!(derived.version(), source.version() + 1);
            prop_assert_ne!(derived.storage_id(), source.storage_id());

            let source_seed = source_value.to_bits() as usize;
            let derived_seed = derived_value.to_bits() as usize;
            let seed = det_seed([source_seed, derived_seed].as_slice());
            let log = build_property_log(
                "prop_with_value_bumps_version_and_storage",
                "strict",
                seed,
                source.evidence_fingerprint64(),
                derived.evidence_fingerprint64(),
                "version_storage_contract_ok",
            );
            assert_log_contract(&log);
        }

        #[test]
        fn prop_rank_stride_mismatch_fail_closed(
            shape in prop::collection::vec(1usize..=4, 1..=4),
            extra in 1usize..=3,
        ) {
            let strides = vec![1usize; shape.len() + extra];
            let err = TensorMeta::from_shape_and_strides(
                shape.clone(),
                strides,
                0,
                DType::F64,
                Device::Cpu,
            )
            .expect_err("rank/stride mismatch must fail");

            match err {
                TensorMetaError::RankStrideMismatch { .. } => {}
                other => prop_assert!(false, "expected RankStrideMismatch, got {other:?}"),
            }

            let seed = det_seed(shape.as_slice());
            let log = build_property_log(
                "prop_rank_stride_mismatch_fail_closed",
                "strict",
                seed,
                seed,
                0,
                "rank_stride_mismatch_fail_closed",
            );
            assert_log_contract(&log);
        }
    }

    // ── bd-2wwr: DenseTensor accessors and methods ──

    #[test]
    fn dense_tensor_id_storage_id_version_accessors() {
        let dt = DenseTensor::from_contiguous_values(vec![1.0, 2.0, 3.0], vec![3], Device::Cpu)
            .expect("create dense tensor");
        // id and storage_id should be nonzero (unique)
        assert!(dt.id() > 0);
        assert!(dt.storage_id() > 0);
        assert_eq!(dt.version(), 0);
    }

    #[test]
    fn dense_tensor_storage_accessor() {
        let vals = vec![10.0, 20.0, 30.0, 40.0];
        let dt = DenseTensor::from_contiguous_values(vals.clone(), vec![4], Device::Cpu)
            .expect("create dense tensor");
        assert_eq!(
            dt.storage().expect("f64 storage should be accessible"),
            &[10.0, 20.0, 30.0, 40.0]
        );
    }

    #[test]
    fn dense_tensor_storage_accessor_rejects_non_f64_dtype() {
        let dt = DenseTensor::from_storage_f32(
            TensorMeta::from_shape(vec![2], DType::F32, Device::Cpu),
            vec![1.0f32, 2.0],
        )
        .expect("create f32 dense tensor");
        let err = dt
            .storage()
            .expect_err("raw f64 storage access should reject f32 tensors");
        assert!(matches!(
            err,
            DenseTensorError::UnsupportedStorageAccess { dtype: DType::F32 }
        ));
    }

    #[test]
    fn dense_tensor_replace_storage_success() {
        let mut dt = DenseTensor::from_contiguous_values(vec![1.0, 2.0, 3.0], vec![3], Device::Cpu)
            .expect("create dense tensor");
        assert_eq!(dt.version(), 0);

        dt.update_contiguous_values(&[4.0, 5.0, 6.0])
            .expect("replace with same-length storage should succeed");
        assert_eq!(dt.version(), 1);
        assert_eq!(
            dt.storage().expect("f64 storage should be accessible"),
            &[4.0, 5.0, 6.0]
        );
    }

    #[test]
    fn dense_tensor_replace_storage_wrong_length() {
        let mut dt = DenseTensor::from_contiguous_values(vec![1.0, 2.0, 3.0], vec![3], Device::Cpu)
            .expect("create dense tensor");
        let err = dt
            .update_contiguous_values(&[1.0, 2.0])
            .expect_err("replace with different-length storage should fail");
        assert!(
            matches!(
                err,
                DenseTensorError::InsufficientStorage {
                    needed: 3,
                    actual: 2
                }
            ),
            "expected InsufficientStorage, got {err:?}"
        );
        // version should NOT have been bumped
        assert_eq!(dt.version(), 0);
    }

    #[test]
    fn dense_tensor_dispatch_values_returns_storage_slice() {
        let dt = DenseTensor::from_contiguous_values(vec![5.0, 6.0, 7.0], vec![3], Device::Cpu)
            .expect("create dense tensor");
        let vals = dt
            .dispatch_values()
            .expect("dispatch_values should succeed");
        assert_eq!(vals, &[5.0, 6.0, 7.0]);
    }

    #[test]
    fn dense_tensor_from_storage_rejects_dtype_mismatch() {
        // from_storage takes Vec<f64>, so passing F32 meta is a mismatch
        let meta = TensorMeta::from_shape(vec![2], DType::F32, Device::Cpu);
        let err = DenseTensor::from_storage(meta, vec![1.0, 2.0])
            .expect_err("F32 meta with f64 storage should be rejected");
        assert!(
            matches!(err, DenseTensorError::UnsupportedDType(DType::F32)),
            "expected UnsupportedDType(F32), got {err:?}"
        );

        // from_storage_f32 with F64 meta is also a mismatch
        let meta = TensorMeta::from_shape(vec![2], DType::F64, Device::Cpu);
        let err = DenseTensor::from_storage_f32(meta, vec![1.0f32, 2.0])
            .expect_err("F64 meta with f32 storage should be rejected");
        assert!(
            matches!(err, DenseTensorError::UnsupportedDType(DType::F64)),
            "expected UnsupportedDType(F64), got {err:?}"
        );
    }

    #[test]
    fn dense_tensor_from_storage_rejects_insufficient_storage() {
        let meta = TensorMeta::from_shape(vec![5], DType::F64, Device::Cpu);
        let err = DenseTensor::from_storage(meta, vec![1.0, 2.0])
            .expect_err("too-short storage should be rejected");
        assert!(
            matches!(
                err,
                DenseTensorError::InsufficientStorage {
                    needed: 5,
                    actual: 2
                }
            ),
            "expected InsufficientStorage, got {err:?}"
        );
    }

    #[test]
    fn dense_tensor_replace_storage_bumps_version_multiple() {
        let mut dt = DenseTensor::from_contiguous_values(vec![0.0, 0.0], vec![2], Device::Cpu)
            .expect("create dense tensor");
        for i in 1..=5 {
            dt.update_contiguous_values(&[i as f64, i as f64 * 2.0])
                .expect("replace should succeed");
        }
        assert_eq!(dt.version(), 5);
    }

    // ── Integer DType tests ───────────────────────────────────────────

    #[test]
    fn dtype_element_sizes() {
        assert_eq!(DType::F64.element_size(), 8);
        assert_eq!(DType::F32.element_size(), 4);
        assert_eq!(DType::I64.element_size(), 8);
        assert_eq!(DType::I32.element_size(), 4);
    }

    #[test]
    fn dtype_is_floating_point() {
        assert!(DType::F64.is_floating_point());
        assert!(DType::F32.is_floating_point());
        assert!(!DType::I64.is_floating_point());
        assert!(!DType::I32.is_floating_point());
    }

    #[test]
    fn dtype_is_integer() {
        assert!(!DType::F64.is_integer());
        assert!(!DType::F32.is_integer());
        assert!(DType::I64.is_integer());
        assert!(DType::I32.is_integer());
    }

    #[test]
    fn tensor_meta_with_integer_dtypes() {
        let meta_i64 = TensorMeta::from_shape(vec![2, 3], DType::I64, Device::Cpu);
        assert_eq!(meta_i64.dtype(), DType::I64);
        assert_eq!(meta_i64.numel(), 6);
        assert_eq!(meta_i64.strides(), &[3, 1]);
        assert!(meta_i64.is_contiguous());

        let meta_i32 = TensorMeta::from_shape(vec![4], DType::I32, Device::Cpu);
        assert_eq!(meta_i32.dtype(), DType::I32);
        assert_eq!(meta_i32.numel(), 4);
    }

    #[test]
    fn dense_i64_tensor_from_contiguous_values() {
        let dt = DenseI64Tensor::from_contiguous_values(vec![1, 2, 3, 4], vec![2, 2], Device::Cpu)
            .expect("create i64 tensor");
        assert_eq!(dt.meta().shape(), &[2, 2]);
        assert_eq!(dt.meta().dtype(), DType::I64);
        assert_eq!(dt.contiguous_values().expect("values"), &[1i64, 2, 3, 4]);
        assert!(dt.id() > 0);
        assert_eq!(dt.version(), 0);
    }

    #[test]
    fn dense_i32_tensor_from_contiguous_values() {
        let dt = DenseI32Tensor::from_contiguous_values(vec![10, 20, 30], vec![3], Device::Cpu)
            .expect("create i32 tensor");
        assert_eq!(dt.meta().shape(), &[3]);
        assert_eq!(dt.meta().dtype(), DType::I32);
        assert_eq!(dt.contiguous_values().expect("values"), &[10i32, 20, 30]);
    }

    #[test]
    fn dense_i64_tensor_rejects_wrong_dtype() {
        let meta = TensorMeta::from_shape(vec![2], DType::F64, Device::Cpu);
        let err = DenseI64Tensor::from_storage(meta, vec![1, 2])
            .expect_err("wrong dtype should be rejected");
        assert!(matches!(
            err,
            DenseTensorError::UnsupportedDType(DType::F64)
        ));
    }

    #[test]
    fn dense_i32_tensor_rejects_wrong_dtype() {
        let meta = TensorMeta::from_shape(vec![2], DType::I64, Device::Cpu);
        let err = DenseI32Tensor::from_storage(meta, vec![1, 2])
            .expect_err("wrong dtype should be rejected");
        assert!(matches!(
            err,
            DenseTensorError::UnsupportedDType(DType::I64)
        ));
    }

    #[test]
    fn dense_i64_tensor_rejects_insufficient_storage() {
        let meta = TensorMeta::from_shape(vec![5], DType::I64, Device::Cpu);
        let err = DenseI64Tensor::from_storage(meta, vec![1, 2])
            .expect_err("insufficient storage should fail");
        assert!(matches!(
            err,
            DenseTensorError::InsufficientStorage {
                needed: 5,
                actual: 2
            }
        ));
    }

    #[test]
    fn dense_i64_tensor_rejects_storage_span_overflow() {
        let meta = TensorMeta::from_shape(vec![1], DType::I64, Device::Cpu)
            .with_storage_offset(usize::MAX);
        let err =
            DenseI64Tensor::from_storage(meta, vec![1]).expect_err("overflowing span must fail");
        assert!(matches!(
            err,
            DenseTensorError::StorageSpanOverflow {
                storage_offset,
                numel: 1
            } if storage_offset == usize::MAX
        ));
    }

    #[test]
    fn dense_i64_tensor_negative_values() {
        let dt = DenseI64Tensor::from_contiguous_values(
            vec![-100, 0, i64::MAX, i64::MIN],
            vec![4],
            Device::Cpu,
        )
        .expect("create with extreme values");
        let vals = dt.contiguous_values().expect("values");
        assert_eq!(vals[0], -100);
        assert_eq!(vals[2], i64::MAX);
        assert_eq!(vals[3], i64::MIN);
    }

    #[test]
    fn dense_i32_tensor_extreme_values() {
        let dt = DenseI32Tensor::from_contiguous_values(
            vec![i32::MAX, i32::MIN, 0, -1],
            vec![4],
            Device::Cpu,
        )
        .expect("create with extreme values");
        let vals = dt.contiguous_values().expect("values");
        assert_eq!(vals[0], i32::MAX);
        assert_eq!(vals[1], i32::MIN);
    }

    #[test]
    fn dense_i64_tensor_scalar() {
        let dt = DenseI64Tensor::from_contiguous_values(vec![42], vec![], Device::Cpu)
            .expect("scalar i64 tensor");
        assert_eq!(dt.meta().shape(), &[] as &[usize]);
        assert_eq!(dt.meta().numel(), 1);
        assert_eq!(dt.contiguous_values().expect("values"), &[42]);
    }

    #[test]
    fn dense_i64_tensor_empty() {
        let dt = DenseI64Tensor::from_contiguous_values(vec![], vec![0], Device::Cpu)
            .expect("empty i64 tensor");
        assert_eq!(dt.meta().numel(), 0);
        assert_eq!(dt.contiguous_values().expect("values"), &[] as &[i64]);
    }

    #[test]
    fn dense_i64_tensor_storage_accessor() {
        let dt = DenseI64Tensor::from_contiguous_values(vec![5, 6, 7], vec![3], Device::Cpu)
            .expect("create i64 tensor");
        assert_eq!(dt.storage(), &[5i64, 6, 7]);
    }

    #[test]
    fn dense_i32_tensor_storage_accessor() {
        let dt = DenseI32Tensor::from_contiguous_values(vec![8, 9], vec![2], Device::Cpu)
            .expect("create i32 tensor");
        assert_eq!(dt.storage(), &[8i32, 9]);
    }

    #[test]
    fn dense_i32_tensor_rejects_storage_span_overflow() {
        let meta = TensorMeta::from_shape(vec![1], DType::I32, Device::Cpu)
            .with_storage_offset(usize::MAX);
        let err =
            DenseI32Tensor::from_storage(meta, vec![1]).expect_err("overflowing span must fail");
        assert!(matches!(
            err,
            DenseTensorError::StorageSpanOverflow {
                storage_offset,
                numel: 1
            } if storage_offset == usize::MAX
        ));
    }

    // ---- Bool DType tests (bd-2do9.2) ----

    #[test]
    fn bool_dtype_element_size_is_1() {
        assert_eq!(DType::Bool.element_size(), 1);
    }

    #[test]
    fn bool_dtype_is_not_floating_point() {
        assert!(!DType::Bool.is_floating_point());
    }

    #[test]
    fn bool_dtype_is_not_integer() {
        assert!(!DType::Bool.is_integer());
    }

    #[test]
    fn bool_dtype_is_bool() {
        assert!(DType::Bool.is_bool());
        assert!(!DType::F64.is_bool());
        assert!(!DType::F32.is_bool());
        assert!(!DType::I64.is_bool());
        assert!(!DType::I32.is_bool());
    }

    #[test]
    fn dense_bool_tensor_from_bools() {
        let t = DenseBoolTensor::from_bools(&[true, false, true, false], vec![4], Device::Cpu)
            .expect("create bool tensor");
        assert_eq!(t.meta().dtype(), DType::Bool);
        assert_eq!(t.meta().shape(), &[4]);
        assert_eq!(t.contiguous_values().unwrap(), &[1u8, 0, 1, 0]);
        assert_eq!(
            t.contiguous_bools().unwrap(),
            vec![true, false, true, false]
        );
    }

    #[test]
    fn dense_bool_tensor_2d() {
        let t = DenseBoolTensor::from_bools(
            &[true, true, false, false, true, false],
            vec![2, 3],
            Device::Cpu,
        )
        .expect("create 2d bool tensor");
        assert_eq!(t.meta().shape(), &[2, 3]);
        assert_eq!(t.meta().numel(), 6);
        assert_eq!(t.contiguous_values().unwrap(), &[1u8, 1, 0, 0, 1, 0]);
    }

    #[test]
    fn dense_bool_tensor_rejects_wrong_dtype() {
        let meta = TensorMeta::from_shape(vec![2], DType::F64, Device::Cpu);
        let err =
            DenseBoolTensor::from_storage(meta, vec![0, 1]).expect_err("wrong dtype should fail");
        assert!(matches!(
            err,
            DenseTensorError::UnsupportedDType(DType::F64)
        ));
    }

    #[test]
    fn dense_bool_tensor_rejects_insufficient_storage() {
        let meta = TensorMeta::from_shape(vec![5], DType::Bool, Device::Cpu);
        let err = DenseBoolTensor::from_storage(meta, vec![0, 1])
            .expect_err("insufficient storage should fail");
        assert!(matches!(
            err,
            DenseTensorError::InsufficientStorage {
                needed: 5,
                actual: 2
            }
        ));
    }

    #[test]
    fn dense_bool_tensor_rejects_storage_span_overflow() {
        let meta = TensorMeta::from_shape(vec![1], DType::Bool, Device::Cpu)
            .with_storage_offset(usize::MAX);
        let err =
            DenseBoolTensor::from_storage(meta, vec![1]).expect_err("overflowing span must fail");
        assert!(matches!(
            err,
            DenseTensorError::StorageSpanOverflow {
                storage_offset,
                numel: 1
            } if storage_offset == usize::MAX
        ));
    }

    #[test]
    fn dense_i64_tensor_contiguous_values_rejects_storage_span_overflow() {
        let tensor = DenseI64Tensor {
            id: 1,
            storage_id: 1,
            meta: TensorMeta::from_shape(vec![1], DType::I64, Device::Cpu)
                .with_storage_offset(usize::MAX),
            storage: vec![1],
            version: 0,
        };

        let err = tensor
            .contiguous_values()
            .expect_err("overflowing span must fail");
        assert!(matches!(
            err,
            DenseTensorError::StorageSpanOverflow {
                storage_offset,
                numel: 1
            } if storage_offset == usize::MAX
        ));
    }

    #[test]
    fn dense_i32_tensor_contiguous_values_rejects_storage_span_overflow() {
        let tensor = DenseI32Tensor {
            id: 1,
            storage_id: 1,
            meta: TensorMeta::from_shape(vec![1], DType::I32, Device::Cpu)
                .with_storage_offset(usize::MAX),
            storage: vec![1],
            version: 0,
        };

        let err = tensor
            .contiguous_values()
            .expect_err("overflowing span must fail");
        assert!(matches!(
            err,
            DenseTensorError::StorageSpanOverflow {
                storage_offset,
                numel: 1
            } if storage_offset == usize::MAX
        ));
    }

    #[test]
    fn dense_bool_tensor_contiguous_values_rejects_storage_span_overflow() {
        let tensor = DenseBoolTensor {
            id: 1,
            storage_id: 1,
            meta: TensorMeta::from_shape(vec![1], DType::Bool, Device::Cpu)
                .with_storage_offset(usize::MAX),
            storage: vec![1],
            version: 0,
        };

        let err = tensor
            .contiguous_values()
            .expect_err("overflowing span must fail");
        assert!(matches!(
            err,
            DenseTensorError::StorageSpanOverflow {
                storage_offset,
                numel: 1
            } if storage_offset == usize::MAX
        ));
    }

    #[test]
    fn dense_bool_tensor_all_true() {
        let t = DenseBoolTensor::from_bools(&[true, true, true], vec![3], Device::Cpu)
            .expect("all-true tensor");
        assert!(t.contiguous_bools().unwrap().iter().all(|&b| b));
    }

    #[test]
    fn dense_bool_tensor_all_false() {
        let t = DenseBoolTensor::from_bools(&[false, false, false], vec![3], Device::Cpu)
            .expect("all-false tensor");
        assert!(t.contiguous_bools().unwrap().iter().all(|&b| !b));
    }

    #[test]
    fn dense_bool_tensor_empty() {
        let t = DenseBoolTensor::from_bools(&[], vec![0], Device::Cpu).expect("empty bool tensor");
        assert_eq!(t.meta().numel(), 0);
        assert_eq!(t.contiguous_values().unwrap(), &[] as &[u8]);
    }

    #[test]
    fn dense_bool_tensor_scalar() {
        let t =
            DenseBoolTensor::from_bools(&[true], vec![], Device::Cpu).expect("scalar bool tensor");
        assert_eq!(t.meta().numel(), 1);
        assert_eq!(t.contiguous_bools().unwrap(), vec![true]);
    }

    #[test]
    fn dense_bool_tensor_storage_accessor() {
        let t = DenseBoolTensor::from_bools(&[false, true], vec![2], Device::Cpu)
            .expect("create bool tensor");
        assert_eq!(t.storage(), &[0u8, 1]);
    }

    #[test]
    fn dense_bool_tensor_has_unique_ids() {
        let t1 = DenseBoolTensor::from_bools(&[true], vec![1], Device::Cpu).expect("t1");
        let t2 = DenseBoolTensor::from_bools(&[false], vec![1], Device::Cpu).expect("t2");
        assert_ne!(t1.id(), t2.id());
        assert_ne!(t1.storage_id(), t2.storage_id());
    }

    #[test]
    fn dense_bool_tensor_version_starts_at_zero() {
        let t =
            DenseBoolTensor::from_bools(&[true, false], vec![2], Device::Cpu).expect("bool tensor");
        assert_eq!(t.version(), 0);
    }

    // ── bd-2do9.3: TensorStorage and F32 DenseTensor tests ──────────────

    #[test]
    fn tensor_storage_f32_basic_ops() {
        let s = TensorStorage::F32(Arc::new(vec![1.0f32, 2.0, 3.0]));
        assert_eq!(s.len(), 3);
        assert!(!s.is_empty());
        assert_eq!(s.dtype(), DType::F32);
        assert!(s.as_f32().is_some());
        assert!(s.as_f64().is_none());
        assert_eq!(s.as_f32().unwrap(), &[1.0f32, 2.0, 3.0]);
    }

    #[test]
    fn tensor_storage_f64_basic_ops() {
        let s = TensorStorage::F64(Arc::new(vec![1.0, 2.0]));
        assert_eq!(s.len(), 2);
        assert_eq!(s.dtype(), DType::F64);
        assert!(s.as_f64().is_some());
        assert!(s.as_f32().is_none());
    }

    #[test]
    fn tensor_storage_empty() {
        let s = TensorStorage::F32(Arc::new(Vec::new()));
        assert!(s.is_empty());
        assert_eq!(s.len(), 0);
    }

    #[test]
    fn tensor_storage_to_f64_vec_from_f32() {
        let s = TensorStorage::F32(Arc::new(vec![1.5f32, 2.5]));
        let v = s.to_f64_vec();
        assert_eq!(v, vec![1.5f64, 2.5]);
    }

    #[test]
    fn tensor_storage_to_f32_vec_from_f64() {
        let s = TensorStorage::F64(Arc::new(vec![1.5, 2.5]));
        let v = s.to_f32_vec();
        assert_eq!(v, vec![1.5f32, 2.5]);
    }

    #[test]
    fn dense_tensor_f32_creation() {
        let dt =
            DenseTensor::from_contiguous_values_f32(vec![1.0f32, 2.0, 3.0], vec![3], Device::Cpu)
                .expect("create f32 dense tensor");
        assert_eq!(dt.meta().dtype(), DType::F32);
        assert_eq!(dt.contiguous_values_f32().unwrap(), &[1.0f32, 2.0, 3.0]);
        assert!(dt.contiguous_values().is_err()); // f64 access on f32 tensor fails
    }

    #[test]
    fn dense_tensor_f32_contiguous_values_as_f64() {
        let dt =
            DenseTensor::from_contiguous_values_f32(vec![1.5f32, 2.5, 3.5], vec![3], Device::Cpu)
                .expect("create f32 dense tensor");
        let f64_vals = dt.contiguous_values_as_f64().unwrap();
        assert_eq!(f64_vals, vec![1.5f64, 2.5, 3.5]);
    }

    #[test]
    fn dense_tensor_f64_contiguous_values_as_f64() {
        let dt = DenseTensor::from_contiguous_values(vec![1.0, 2.0], vec![2], Device::Cpu)
            .expect("create f64 dense tensor");
        let f64_vals = dt.contiguous_values_as_f64().unwrap();
        assert_eq!(f64_vals, vec![1.0, 2.0]);
    }

    #[test]
    fn dense_tensor_to_dtype_f64_to_f32() {
        let dt = DenseTensor::from_contiguous_values(vec![1.5, 2.5], vec![2], Device::Cpu)
            .expect("create f64 tensor");
        let f32_dt = dt.to_dtype(DType::F32).expect("cast to f32");
        assert_eq!(f32_dt.meta().dtype(), DType::F32);
        assert_eq!(f32_dt.contiguous_values_f32().unwrap(), &[1.5f32, 2.5]);
    }

    #[test]
    fn dense_tensor_to_dtype_f32_to_f64() {
        let dt = DenseTensor::from_contiguous_values_f32(vec![1.5f32, 2.5], vec![2], Device::Cpu)
            .expect("create f32 tensor");
        let f64_dt = dt.to_dtype(DType::F64).expect("cast to f64");
        assert_eq!(f64_dt.meta().dtype(), DType::F64);
        assert_eq!(f64_dt.contiguous_values().unwrap(), &[1.5, 2.5]);
    }

    #[test]
    fn dense_tensor_to_dtype_same_is_clone() {
        let dt = DenseTensor::from_contiguous_values(vec![1.0, 2.0], vec![2], Device::Cpu)
            .expect("create tensor");
        let same = dt.to_dtype(DType::F64).expect("same dtype");
        assert_eq!(same.contiguous_values().unwrap(), &[1.0, 2.0]);
    }

    #[test]
    fn dense_tensor_to_dtype_rejects_non_float() {
        let dt = DenseTensor::from_contiguous_values(vec![1.0], vec![1], Device::Cpu)
            .expect("create tensor");
        assert!(dt.to_dtype(DType::I64).is_err());
    }

    #[test]
    fn dtype_promote_f32_f32() {
        assert_eq!(DType::F32.promote(DType::F32), Some(DType::F32));
    }

    #[test]
    fn dtype_promote_f64_f64() {
        assert_eq!(DType::F64.promote(DType::F64), Some(DType::F64));
    }

    #[test]
    fn dtype_promote_mixed() {
        assert_eq!(DType::F32.promote(DType::F64), Some(DType::F64));
        assert_eq!(DType::F64.promote(DType::F32), Some(DType::F64));
    }

    #[test]
    fn dtype_promote_non_float_returns_none() {
        assert_eq!(DType::F32.promote(DType::I64), None);
        assert_eq!(DType::I64.promote(DType::F64), None);
    }

    #[test]
    fn tensor_meta_with_dtype() {
        let meta = TensorMeta::from_shape(vec![2, 3], DType::F64, Device::Cpu);
        let meta_f32 = meta.clone().with_dtype(DType::F32);
        assert_eq!(meta_f32.dtype(), DType::F32);
        assert_eq!(meta_f32.shape(), meta.shape());
        assert_eq!(meta_f32.strides(), meta.strides());
    }

    #[test]
    fn dense_tensor_typed_storage_accessor() {
        let dt = DenseTensor::from_contiguous_values_f32(vec![1.0f32, 2.0], vec![2], Device::Cpu)
            .expect("create f32 tensor");
        assert_eq!(dt.typed_storage().dtype(), DType::F32);
        assert_eq!(dt.typed_storage().as_f32().unwrap(), &[1.0f32, 2.0]);
    }

    #[test]
    fn dense_tensor_f32_update_contiguous_values() {
        let mut dt =
            DenseTensor::from_contiguous_values_f32(vec![1.0f32, 2.0, 3.0], vec![3], Device::Cpu)
                .expect("create f32 tensor");
        dt.update_contiguous_values_f32(&[4.0f32, 5.0, 6.0])
            .expect("update should succeed");
        assert_eq!(dt.version(), 1);
        assert_eq!(dt.contiguous_values_f32().unwrap(), &[4.0f32, 5.0, 6.0]);
    }

    #[test]
    fn dense_tensor_from_typed_storage_f32() {
        let meta = TensorMeta::from_shape(vec![2], DType::F32, Device::Cpu);
        let storage = TensorStorage::F32(Arc::new(vec![1.0f32, 2.0]));
        let dt = DenseTensor::from_typed_storage(meta, storage).expect("create from typed storage");
        assert_eq!(dt.meta().dtype(), DType::F32);
    }

    #[test]
    fn dense_tensor_from_typed_storage_rejects_non_float() {
        let meta = TensorMeta::from_shape(vec![2], DType::I64, Device::Cpu);
        let storage = TensorStorage::F64(Arc::new(vec![1.0, 2.0]));
        let err = DenseTensor::from_typed_storage(meta, storage)
            .expect_err("non-float dtype should be rejected");
        assert!(matches!(
            err,
            DenseTensorError::UnsupportedDType(DType::I64)
        ));
    }

    // ── promote_types tests ───────────────────────────────────────────

    #[test]
    fn promote_types_same_dtype_is_identity() {
        for dtype in [
            DType::Bool,
            DType::I32,
            DType::I64,
            DType::F16,
            DType::BF16,
            DType::F32,
            DType::F64,
        ] {
            assert_eq!(dtype.promote_types(dtype), dtype);
        }
    }

    #[test]
    fn promote_types_is_symmetric() {
        let dtypes = [
            DType::Bool,
            DType::I32,
            DType::I64,
            DType::F16,
            DType::BF16,
            DType::F32,
            DType::F64,
        ];
        for &a in &dtypes {
            for &b in &dtypes {
                assert_eq!(
                    a.promote_types(b),
                    b.promote_types(a),
                    "promote_types({a:?}, {b:?}) != promote_types({b:?}, {a:?})"
                );
            }
        }
    }

    #[test]
    fn promote_types_bool_with_integers() {
        assert_eq!(DType::Bool.promote_types(DType::I32), DType::I32);
        assert_eq!(DType::Bool.promote_types(DType::I64), DType::I64);
    }

    #[test]
    fn promote_types_bool_with_floats() {
        assert_eq!(DType::Bool.promote_types(DType::F32), DType::F32);
        assert_eq!(DType::Bool.promote_types(DType::F64), DType::F64);
    }

    #[test]
    fn promote_types_int_with_int() {
        assert_eq!(DType::I32.promote_types(DType::I64), DType::I64);
    }

    #[test]
    fn promote_types_int_with_float() {
        // Int + Float → Float (matching PyTorch: int64 + float32 → float32)
        assert_eq!(DType::I32.promote_types(DType::F32), DType::F32);
        assert_eq!(DType::I32.promote_types(DType::F64), DType::F64);
        assert_eq!(DType::I64.promote_types(DType::F32), DType::F32);
        assert_eq!(DType::I64.promote_types(DType::F64), DType::F64);
    }

    #[test]
    fn promote_types_float_with_float() {
        assert_eq!(DType::F32.promote_types(DType::F64), DType::F64);
    }

    #[test]
    fn promote_types_full_table() {
        // Exhaustive: every pair produces expected result
        let expected: &[(DType, DType, DType)] = &[
            (DType::Bool, DType::Bool, DType::Bool),
            (DType::Bool, DType::I32, DType::I32),
            (DType::Bool, DType::I64, DType::I64),
            (DType::Bool, DType::F16, DType::F16),
            (DType::Bool, DType::BF16, DType::BF16),
            (DType::Bool, DType::F32, DType::F32),
            (DType::Bool, DType::F64, DType::F64),
            (DType::I32, DType::I32, DType::I32),
            (DType::I32, DType::I64, DType::I64),
            (DType::I32, DType::F16, DType::F16),
            (DType::I32, DType::BF16, DType::BF16),
            (DType::I32, DType::F32, DType::F32),
            (DType::I32, DType::F64, DType::F64),
            (DType::I64, DType::I64, DType::I64),
            (DType::I64, DType::F16, DType::F16),
            (DType::I64, DType::BF16, DType::BF16),
            (DType::I64, DType::F32, DType::F32),
            (DType::I64, DType::F64, DType::F64),
            (DType::F16, DType::F16, DType::F16),
            (DType::F16, DType::BF16, DType::F32), // mixed half → F32
            (DType::F16, DType::F32, DType::F32),
            (DType::F16, DType::F64, DType::F64),
            (DType::BF16, DType::BF16, DType::BF16),
            (DType::BF16, DType::F32, DType::F32),
            (DType::BF16, DType::F64, DType::F64),
            (DType::F32, DType::F32, DType::F32),
            (DType::F32, DType::F64, DType::F64),
            (DType::F64, DType::F64, DType::F64),
        ];
        for &(a, b, result) in expected {
            assert_eq!(
                a.promote_types(b),
                result,
                "promote_types({a:?}, {b:?}) should be {result:?}"
            );
            assert_eq!(
                b.promote_types(a),
                result,
                "promote_types({b:?}, {a:?}) should be {result:?} (symmetry)"
            );
        }
    }

    // ── F16 / BF16 tests ─────────────────────────────────────────────

    #[test]
    fn dtype_f16_properties() {
        assert_eq!(DType::F16.element_size(), 2);
        assert!(DType::F16.is_floating_point());
        assert!(DType::F16.is_half());
        assert!(!DType::F16.is_integer());
        assert!(!DType::F16.is_bool());
    }

    #[test]
    fn dtype_bf16_properties() {
        assert_eq!(DType::BF16.element_size(), 2);
        assert!(DType::BF16.is_floating_point());
        assert!(DType::BF16.is_half());
        assert!(!DType::BF16.is_integer());
        assert!(!DType::BF16.is_bool());
    }

    #[test]
    fn f16_roundtrip_preserves_value() {
        let original: Vec<f32> = vec![1.0, 0.5, -3.25, 0.0, 100.0];
        let f16_vals: Vec<Float16> = original.iter().map(|&x| Float16::from_f32(x)).collect();
        let roundtrip: Vec<f32> = f16_vals.iter().map(|&x| x.to_f32()).collect();
        for (orig, rt) in original.iter().zip(roundtrip.iter()) {
            assert!(
                (orig - rt).abs() < 0.01,
                "f16 roundtrip failed: {orig} -> {rt}"
            );
        }
    }

    #[test]
    fn bf16_roundtrip_preserves_value() {
        let original: Vec<f32> = vec![1.0, 0.5, -3.25, 0.0, 100.0];
        let bf16_vals: Vec<BFloat16> = original.iter().map(|&x| BFloat16::from_f32(x)).collect();
        let roundtrip: Vec<f32> = bf16_vals.iter().map(|&x| x.to_f32()).collect();
        for (orig, rt) in original.iter().zip(roundtrip.iter()) {
            assert!(
                (orig - rt).abs() < 1.0,
                "bf16 roundtrip failed: {orig} -> {rt}"
            );
        }
    }

    #[test]
    fn f16_overflow_above_65504() {
        let big = Float16::from_f32(70000.0);
        assert!(
            big.to_f32().is_infinite(),
            "f16 should overflow to inf for values > 65504"
        );
    }

    #[test]
    fn f16_inf_neg_inf() {
        let pos_inf = Float16::from_f32(f32::INFINITY);
        let neg_inf = Float16::from_f32(f32::NEG_INFINITY);
        assert!(pos_inf.to_f32().is_infinite() && pos_inf.to_f32() > 0.0);
        assert!(neg_inf.to_f32().is_infinite() && neg_inf.to_f32() < 0.0);
    }

    #[test]
    fn f16_zero_and_neg_zero() {
        let zero = Float16::from_f32(0.0);
        let neg_zero = Float16::from_f32(-0.0);
        assert_eq!(zero.to_f32(), 0.0);
        assert_eq!(neg_zero.to_f32(), -0.0);
        // Both should compare equal as floats
        assert_eq!(zero.to_f32(), neg_zero.to_f32());
    }

    #[test]
    fn bf16_same_exponent_range_as_f32() {
        // BF16 has same 8-bit exponent as f32, so same max magnitude
        let big = BFloat16::from_f32(1e38);
        assert!(
            big.to_f32().is_finite(),
            "bf16 should handle 1e38 (within f32 range)"
        );
    }

    #[test]
    fn tensor_storage_f16_basic() {
        let vals: Vec<Float16> = vec![1.0f32, 2.0, 3.0]
            .into_iter()
            .map(Float16::from_f32)
            .collect();
        let s = TensorStorage::F16(Arc::new(vals));
        assert_eq!(s.len(), 3);
        assert_eq!(s.dtype(), DType::F16);
        assert!(!s.is_empty());
        // Conversion to f32
        let f32_vals = s.to_f32_vec();
        assert!((f32_vals[0] - 1.0).abs() < 0.01);
        assert!((f32_vals[1] - 2.0).abs() < 0.01);
        assert!((f32_vals[2] - 3.0).abs() < 0.01);
        // Conversion to f64
        let f64_vals = s.to_f64_vec();
        assert!((f64_vals[0] - 1.0).abs() < 0.01);
    }

    #[test]
    fn tensor_storage_bf16_basic() {
        let vals: Vec<BFloat16> = vec![1.0f32, 2.0, 3.0]
            .into_iter()
            .map(BFloat16::from_f32)
            .collect();
        let s = TensorStorage::BF16(Arc::new(vals));
        assert_eq!(s.len(), 3);
        assert_eq!(s.dtype(), DType::BF16);
        let f32_vals = s.to_f32_vec();
        assert!((f32_vals[0] - 1.0).abs() < 0.01);
        assert!((f32_vals[1] - 2.0).abs() < 0.01);
    }

    #[test]
    fn dense_tensor_f16_create_and_read() {
        let vals: Vec<Float16> = vec![1.0f32, 2.0, 3.0, 4.0]
            .into_iter()
            .map(Float16::from_f32)
            .collect();
        let dt = DenseTensor::from_contiguous_values_f16(vals, vec![2, 2], Device::Cpu).unwrap();
        assert_eq!(dt.meta().dtype(), DType::F16);
        assert_eq!(dt.meta().shape(), &[2, 2]);
        // contiguous_values_as_f64 should work
        let f64_vals = dt.contiguous_values_as_f64().unwrap();
        assert!((f64_vals[0] - 1.0).abs() < 0.01);
        assert!((f64_vals[3] - 4.0).abs() < 0.01);
    }

    #[test]
    fn dense_tensor_bf16_create_and_read() {
        let vals: Vec<BFloat16> = vec![1.0f32, 2.0, 3.0]
            .into_iter()
            .map(BFloat16::from_f32)
            .collect();
        let dt = DenseTensor::from_contiguous_values_bf16(vals, vec![3], Device::Cpu).unwrap();
        assert_eq!(dt.meta().dtype(), DType::BF16);
        let f64_vals = dt.contiguous_values_as_f64().unwrap();
        assert!((f64_vals[0] - 1.0).abs() < 0.01);
    }

    #[test]
    fn to_dtype_f32_to_f16() {
        let dt =
            DenseTensor::from_contiguous_values_f32(vec![1.0f32, 2.5, -3.0], vec![3], Device::Cpu)
                .unwrap();
        let f16_dt = dt.to_dtype(DType::F16).unwrap();
        assert_eq!(f16_dt.meta().dtype(), DType::F16);
        let vals = f16_dt.contiguous_values_as_f64().unwrap();
        assert!((vals[0] - 1.0).abs() < 0.01);
        assert!((vals[1] - 2.5).abs() < 0.01);
        assert!((vals[2] + 3.0).abs() < 0.01);
    }

    #[test]
    fn to_dtype_f16_to_f32() {
        let vals: Vec<Float16> = vec![1.0f32, 2.5, -3.0]
            .into_iter()
            .map(Float16::from_f32)
            .collect();
        let dt = DenseTensor::from_contiguous_values_f16(vals, vec![3], Device::Cpu).unwrap();
        let f32_dt = dt.to_dtype(DType::F32).unwrap();
        assert_eq!(f32_dt.meta().dtype(), DType::F32);
        let vals = f32_dt.contiguous_values_f32().unwrap();
        assert!((vals[0] - 1.0).abs() < 0.01);
        assert!((vals[1] - 2.5).abs() < 0.01);
    }

    #[test]
    fn to_dtype_f16_to_f64() {
        let vals: Vec<Float16> = vec![1.0f32, 2.0]
            .into_iter()
            .map(Float16::from_f32)
            .collect();
        let dt = DenseTensor::from_contiguous_values_f16(vals, vec![2], Device::Cpu).unwrap();
        let f64_dt = dt.to_dtype(DType::F64).unwrap();
        assert_eq!(f64_dt.meta().dtype(), DType::F64);
        let vals = f64_dt.contiguous_values().unwrap();
        assert!((vals[0] - 1.0).abs() < 0.01);
    }

    #[test]
    fn to_dtype_bf16_roundtrip() {
        let dt =
            DenseTensor::from_contiguous_values_f32(vec![1.0f32, 0.5, -2.0], vec![3], Device::Cpu)
                .unwrap();
        let bf16_dt = dt.to_dtype(DType::BF16).unwrap();
        assert_eq!(bf16_dt.meta().dtype(), DType::BF16);
        let back_to_f32 = bf16_dt.to_dtype(DType::F32).unwrap();
        let vals = back_to_f32.contiguous_values_f32().unwrap();
        assert!((vals[0] - 1.0).abs() < 0.1);
        assert!((vals[1] - 0.5).abs() < 0.1);
        assert!((vals[2] + 2.0).abs() < 0.1);
    }

    #[test]
    fn promote_f16_with_f32() {
        assert_eq!(DType::F16.promote(DType::F32), Some(DType::F32));
        assert_eq!(DType::F32.promote(DType::F16), Some(DType::F32));
    }

    #[test]
    fn promote_f16_with_f64() {
        assert_eq!(DType::F16.promote(DType::F64), Some(DType::F64));
        assert_eq!(DType::F64.promote(DType::F16), Some(DType::F64));
    }

    #[test]
    fn promote_f16_with_bf16() {
        assert_eq!(DType::F16.promote(DType::BF16), Some(DType::F32));
        assert_eq!(DType::BF16.promote(DType::F16), Some(DType::F32));
    }

    #[test]
    fn promote_bf16_with_f32() {
        assert_eq!(DType::BF16.promote(DType::F32), Some(DType::F32));
        assert_eq!(DType::F32.promote(DType::BF16), Some(DType::F32));
    }

    #[test]
    fn f16_dispatch_values_returns_error() {
        let vals: Vec<Float16> = vec![1.0f32, 2.0]
            .into_iter()
            .map(Float16::from_f32)
            .collect();
        let dt = DenseTensor::from_contiguous_values_f16(vals, vec![2], Device::Cpu).unwrap();
        assert!(dt.dispatch_values().is_err());
        assert!(dt.contiguous_values().is_err());
    }

    #[test]
    fn f16_subnormal_handling() {
        // Smallest positive f16 subnormal: ~5.96e-8
        let tiny = Float16::from_f32(5.96e-8);
        let rt = tiny.to_f32();
        assert!(
            (0.0..1e-5).contains(&rt),
            "f16 subnormal should be small positive or zero"
        );
    }

    // ── Complex dtype tests ─────────────────────────────────────────

    #[test]
    fn complex_dtype_element_sizes() {
        assert_eq!(DType::Complex64.element_size(), 8);
        assert_eq!(DType::Complex128.element_size(), 16);
    }

    #[test]
    fn complex_dtype_predicates() {
        assert!(DType::Complex64.is_complex());
        assert!(DType::Complex128.is_complex());
        assert!(!DType::F64.is_complex());
        assert!(!DType::Complex64.is_floating_point());
        assert!(!DType::Complex128.is_integer());
        assert!(!DType::Complex64.is_bool());
    }

    #[test]
    fn complex_promote_types_hierarchy() {
        // Complex128 is the widest type
        assert_eq!(
            DType::Complex128.promote_types(DType::Complex64),
            DType::Complex128
        );
        assert_eq!(
            DType::Complex128.promote_types(DType::F64),
            DType::Complex128
        );
        assert_eq!(
            DType::Complex128.promote_types(DType::F32),
            DType::Complex128
        );

        // Complex64 + F64 widens to Complex128 (f64 component)
        assert_eq!(
            DType::Complex64.promote_types(DType::F64),
            DType::Complex128
        );
        assert_eq!(
            DType::F64.promote_types(DType::Complex64),
            DType::Complex128
        );

        // Complex64 + F32 stays Complex64
        assert_eq!(DType::Complex64.promote_types(DType::F32), DType::Complex64);

        // Complex64 + integer → Complex64
        assert_eq!(DType::Complex64.promote_types(DType::I32), DType::Complex64);
        assert_eq!(
            DType::Complex64.promote_types(DType::Bool),
            DType::Complex64
        );
    }

    #[test]
    fn complex_promote_float_function() {
        // promote() handles complex types
        assert_eq!(
            DType::Complex128.promote(DType::Complex64),
            Some(DType::Complex128)
        );
        assert_eq!(
            DType::Complex64.promote(DType::F64),
            Some(DType::Complex128)
        );
        assert_eq!(DType::Complex64.promote(DType::F32), Some(DType::Complex64));
        assert_eq!(
            DType::Complex128.promote(DType::F64),
            Some(DType::Complex128)
        );
    }

    #[test]
    fn complex_storage_basic() {
        use super::Complex128;

        let vals = vec![Complex128::new(1.0, 2.0), Complex128::new(3.0, 4.0)];
        let storage = TensorStorage::Complex128(Arc::new(vals));
        assert_eq!(storage.len(), 2);
        assert_eq!(storage.dtype(), DType::Complex128);

        let slice = storage.as_complex128().unwrap();
        assert_eq!(slice[0].re, 1.0);
        assert_eq!(slice[0].im, 2.0);

        // to_f64_vec extracts real parts
        let f64s = storage.to_f64_vec();
        assert_eq!(f64s, vec![1.0, 3.0]);
    }

    #[test]
    fn complex64_storage_basic() {
        use super::Complex64;

        let vals = vec![Complex64::new(1.0, -1.0), Complex64::new(0.0, 5.0)];
        let storage = TensorStorage::Complex64(Arc::new(vals));
        assert_eq!(storage.len(), 2);
        assert_eq!(storage.dtype(), DType::Complex64);

        let slice = storage.as_complex64().unwrap();
        assert_eq!(slice[1].im, 5.0);
    }

    #[test]
    fn complex_dense_tensor_creation() {
        use super::Complex128;

        let vals = vec![
            Complex128::new(1.0, 0.0),
            Complex128::new(0.0, 1.0),
            Complex128::new(-1.0, 0.0),
        ];
        let meta = TensorMeta::from_shape(vec![3], DType::Complex128, Device::Cpu);
        let storage = TensorStorage::Complex128(Arc::new(vals));
        let dt = DenseTensor::from_typed_storage(meta, storage);
        assert!(dt.is_ok(), "complex tensor creation should succeed");

        let t = dt.unwrap();
        assert_eq!(t.meta().dtype(), DType::Complex128);
        assert_eq!(t.meta().shape(), &[3]);
    }

    #[test]
    fn complex_to_dtype_from_real() {
        // Create an f64 tensor, cast to Complex128
        let meta = TensorMeta::from_shape(vec![2], DType::F64, Device::Cpu);
        let storage = TensorStorage::F64(Arc::new(vec![3.0, 4.0]));
        let dt = DenseTensor::from_typed_storage(meta, storage).unwrap();

        let complex = dt.to_dtype(DType::Complex128).unwrap();
        assert_eq!(complex.meta().dtype(), DType::Complex128);

        let c_slice = complex.typed_storage().as_complex128().unwrap();
        assert_eq!(c_slice[0].re, 3.0);
        assert_eq!(c_slice[0].im, 0.0);
        assert_eq!(c_slice[1].re, 4.0);
        assert_eq!(c_slice[1].im, 0.0);
    }

    // ── Sparse Tensor Tests ────────────────────────────────────────────────

    #[test]
    fn sparse_coo_creation() {
        // Create a 3x4 sparse matrix with 2 non-zero elements
        // indices: [[0, 2], [1, 3]] (row 0 col 1, row 2 col 3)
        // values: [1.0, 2.0]
        let indices =
            DenseI64Tensor::from_contiguous_values(vec![0, 2, 1, 3], vec![2, 2], Device::Cpu)
                .unwrap();

        let values =
            DenseTensor::from_contiguous_values(vec![1.0, 2.0], vec![2], Device::Cpu).unwrap();

        let sparse = SparseCOOTensor::new(indices, values, vec![3, 4], true).unwrap();

        assert_eq!(sparse.dense_shape(), &[3, 4]);
        assert_eq!(sparse.sparse_dim(), 2);
        assert_eq!(sparse.nnz(), 2);
        assert!(sparse.is_coalesced());
        assert_eq!(sparse.dtype(), DType::F64);
    }

    #[test]
    fn sparse_coo_to_dense_roundtrip() {
        // Create a 2x3 sparse matrix
        // [[1, 0, 2],
        //  [0, 3, 0]]
        let coords = vec![vec![0, 0], vec![0, 2], vec![1, 1]];
        let values = vec![1.0, 2.0, 3.0];

        let sparse =
            SparseCOOTensor::from_coords(&coords, values, vec![2, 3], DType::F64, Device::Cpu)
                .unwrap();

        let dense = sparse.to_dense().unwrap();

        let expected = vec![1.0, 0.0, 2.0, 0.0, 3.0, 0.0];
        let actual = dense.contiguous_values().unwrap();
        assert_eq!(actual, expected.as_slice());
    }

    #[test]
    fn sparse_coo_to_dense_sums_duplicate_indices() {
        // Duplicate entry at (0, 1) should accumulate.
        let coords = vec![vec![0, 1], vec![0, 1]];
        let values = vec![1.0, 2.5];

        let sparse =
            SparseCOOTensor::from_coords(&coords, values, vec![2, 2], DType::F64, Device::Cpu)
                .unwrap();

        let dense = sparse.to_dense().unwrap();
        let expected = vec![0.0, 3.5, 0.0, 0.0];
        let actual = dense.contiguous_values().unwrap();
        assert_eq!(actual, expected.as_slice());
    }

    #[test]
    fn sparse_coo_to_dense_sums_duplicate_dense_blocks() {
        // Duplicate entry at (0, 1, :) should accumulate elementwise.
        let indices =
            DenseI64Tensor::from_contiguous_values(vec![0, 0, 1, 1], vec![2, 2], Device::Cpu)
                .unwrap();
        let values =
            DenseTensor::from_contiguous_values(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2], Device::Cpu)
                .unwrap();

        let sparse = SparseCOOTensor::new(indices, values, vec![2, 2, 2], false).unwrap();
        let dense = sparse.to_dense().unwrap();

        let expected = vec![0.0, 0.0, 4.0, 6.0, 0.0, 0.0, 0.0, 0.0];
        let actual = dense.contiguous_values().unwrap();
        assert_eq!(actual, expected.as_slice());
    }

    #[test]
    fn sparse_coo_empty() {
        let sparse =
            SparseCOOTensor::from_coords(&[], vec![], vec![5, 5], DType::F64, Device::Cpu).unwrap();

        assert_eq!(sparse.nnz(), 0);
        assert_eq!(sparse.dense_shape(), &[5, 5]);

        let dense = sparse.to_dense().unwrap();
        let values = dense.contiguous_values().unwrap();
        assert!(values.iter().all(|&v| v == 0.0));
    }

    #[test]
    fn sparse_coo_index_out_of_bounds() {
        // Index [5, 0] is out of bounds for shape [3, 4] (row 5 >= 3)
        // indices shape [2, 1]: 2 sparse dims, 1 non-zero
        let indices =
            DenseI64Tensor::from_contiguous_values(vec![5, 0], vec![2, 1], Device::Cpu).unwrap();
        let values = DenseTensor::from_contiguous_values(vec![1.0], vec![1], Device::Cpu).unwrap();

        let result = SparseCOOTensor::new(indices, values, vec![3, 4], true);
        assert!(matches!(
            result,
            Err(SparseTensorError::IndexOutOfBounds { .. })
        ));
    }

    #[test]
    fn sparse_coo_uncoalesced_out_of_bounds_rejected() {
        let indices =
            DenseI64Tensor::from_contiguous_values(vec![5, 0], vec![2, 1], Device::Cpu).unwrap();
        let values = DenseTensor::from_contiguous_values(vec![1.0], vec![1], Device::Cpu).unwrap();

        let result = SparseCOOTensor::new(indices, values, vec![3, 4], false);
        assert!(matches!(
            result,
            Err(SparseTensorError::IndexOutOfBounds { .. })
        ));
    }

    #[test]
    fn sparse_coo_device_mismatch() {
        let indices =
            DenseI64Tensor::from_contiguous_values(vec![0, 1], vec![2, 1], Device::Cpu).unwrap();
        let values = DenseTensor::from_contiguous_values(vec![1.0], vec![1], Device::Cuda).unwrap();

        let result = SparseCOOTensor::new(indices, values, vec![2, 2], true);
        assert!(matches!(
            result,
            Err(SparseTensorError::DeviceMismatch { .. })
        ));
    }

    #[test]
    fn sparse_csr_creation() {
        // Create a 3x4 sparse matrix:
        // [[1, 0, 2, 0],
        //  [0, 0, 0, 3],
        //  [4, 0, 0, 0]]
        // crow_indices: [0, 2, 3, 4] (row 0 has 2 elems, row 1 has 1, row 2 has 1)
        // col_indices: [0, 2, 3, 0]
        // values: [1, 2, 3, 4]
        let crow =
            DenseI64Tensor::from_contiguous_values(vec![0, 2, 3, 4], vec![4], Device::Cpu).unwrap();
        let col =
            DenseI64Tensor::from_contiguous_values(vec![0, 2, 3, 0], vec![4], Device::Cpu).unwrap();
        let values =
            DenseTensor::from_contiguous_values(vec![1.0, 2.0, 3.0, 4.0], vec![4], Device::Cpu)
                .unwrap();

        let csr = SparseCSRTensor::new(crow, col, values, [3, 4]).unwrap();

        assert_eq!(csr.shape(), [3, 4]);
        assert_eq!(csr.nrows(), 3);
        assert_eq!(csr.ncols(), 4);
        assert_eq!(csr.nnz(), 4);
    }

    #[test]
    fn sparse_csr_to_dense() {
        // Same matrix as above
        let crow =
            DenseI64Tensor::from_contiguous_values(vec![0, 2, 3, 4], vec![4], Device::Cpu).unwrap();
        let col =
            DenseI64Tensor::from_contiguous_values(vec![0, 2, 3, 0], vec![4], Device::Cpu).unwrap();
        let values =
            DenseTensor::from_contiguous_values(vec![1.0, 2.0, 3.0, 4.0], vec![4], Device::Cpu)
                .unwrap();

        let csr = SparseCSRTensor::new(crow, col, values, [3, 4]).unwrap();
        let dense = csr.to_dense().unwrap();

        let expected = vec![1.0, 0.0, 2.0, 0.0, 0.0, 0.0, 0.0, 3.0, 4.0, 0.0, 0.0, 0.0];
        let actual = dense.contiguous_values().unwrap();
        assert_eq!(actual, expected.as_slice());
    }

    #[test]
    fn sparse_csr_col_out_of_bounds() {
        // Column index 5 is out of bounds for 4 columns
        let crow =
            DenseI64Tensor::from_contiguous_values(vec![0, 1], vec![2], Device::Cpu).unwrap();
        let col = DenseI64Tensor::from_contiguous_values(vec![5], vec![1], Device::Cpu).unwrap();
        let values = DenseTensor::from_contiguous_values(vec![1.0], vec![1], Device::Cpu).unwrap();

        let result = SparseCSRTensor::new(crow, col, values, [1, 4]);
        assert!(matches!(
            result,
            Err(SparseTensorError::ColIndexOutOfBounds { .. })
        ));
    }

    #[test]
    fn sparse_csr_device_mismatch() {
        let crow =
            DenseI64Tensor::from_contiguous_values(vec![0, 1], vec![2], Device::Cuda).unwrap();
        let col = DenseI64Tensor::from_contiguous_values(vec![0], vec![1], Device::Cpu).unwrap();
        let values = DenseTensor::from_contiguous_values(vec![1.0], vec![1], Device::Cpu).unwrap();

        let result = SparseCSRTensor::new(crow, col, values, [1, 2]);
        assert!(matches!(
            result,
            Err(SparseTensorError::DeviceMismatch { .. })
        ));
    }

    #[test]
    fn sparse_csr_non_monotonic_crow() {
        // crow_indices [0, 2, 1] is not monotonic at row 1: 2 > 1
        // nnz = crow[nrows] = crow[2] = 1, so col and values need 1 element
        let crow =
            DenseI64Tensor::from_contiguous_values(vec![0, 2, 1], vec![3], Device::Cpu).unwrap();
        let col = DenseI64Tensor::from_contiguous_values(vec![0], vec![1], Device::Cpu).unwrap();
        let values = DenseTensor::from_contiguous_values(vec![1.0], vec![1], Device::Cpu).unwrap();

        let result = SparseCSRTensor::new(crow, col, values, [2, 3]);
        assert!(matches!(
            result,
            Err(SparseTensorError::NonMonotonicCrowIndices { .. })
        ));
    }

    #[test]
    fn sparse_csr_rejects_nonzero_crow_start() {
        let crow =
            DenseI64Tensor::from_contiguous_values(vec![1, 1], vec![2], Device::Cpu).unwrap();
        let col = DenseI64Tensor::from_contiguous_values(vec![0], vec![1], Device::Cpu).unwrap();
        let values = DenseTensor::from_contiguous_values(vec![1.0], vec![1], Device::Cpu).unwrap();

        let result = SparseCSRTensor::new(crow, col, values, [1, 3]);
        assert!(matches!(
            result,
            Err(SparseTensorError::InvalidCrowIndexValue { .. })
        ));
    }

    #[test]
    fn sparse_csr_rejects_negative_crow_tail() {
        let crow =
            DenseI64Tensor::from_contiguous_values(vec![0, -1], vec![2], Device::Cpu).unwrap();
        let col = DenseI64Tensor::from_contiguous_values(vec![], vec![0], Device::Cpu).unwrap();
        let values = DenseTensor::from_contiguous_values(vec![], vec![0], Device::Cpu).unwrap();

        let result = SparseCSRTensor::new(crow, col, values, [1, 3]);
        assert!(matches!(
            result,
            Err(SparseTensorError::InvalidCrowIndexValue { .. })
        ));
    }
}

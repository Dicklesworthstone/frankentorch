#![forbid(unsafe_code)]

use std::fmt;
use std::hash::{Hash, Hasher};
use std::sync::atomic::{AtomicU64, Ordering};

static NEXT_TENSOR_ID: AtomicU64 = AtomicU64::new(1);
static NEXT_STORAGE_ID: AtomicU64 = AtomicU64::new(1);

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum DType {
    F64,
    F32,
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
        self.shape
            .iter()
            .copied()
            .try_fold(1usize, usize::checked_mul)
            .unwrap_or(usize::MAX)
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
        let mut hasher = std::collections::hash_map::DefaultHasher::new();
        self.shape.hash(&mut hasher);
        self.strides.hash(&mut hasher);
        self.storage_offset.hash(&mut hasher);
        self.dtype.hash(&mut hasher);
        self.device.hash(&mut hasher);
        hasher.finish()
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
        let mut hasher = std::collections::hash_map::DefaultHasher::new();
        self.id.hash(&mut hasher);
        self.storage_id.hash(&mut hasher);
        self.version.hash(&mut hasher);
        self.meta.fingerprint64().hash(&mut hasher);
        self.value.to_bits().hash(&mut hasher);
        hasher.finish()
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct DenseTensor {
    id: u64,
    storage_id: u64,
    meta: TensorMeta,
    storage: Vec<f64>,
    version: u64,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum DenseTensorError {
    Meta(TensorMetaError),
    UnsupportedDType(DType),
    UnsupportedLayout,
    StorageSpanOverflow { storage_offset: usize, numel: usize },
    InsufficientStorage { needed: usize, actual: usize },
}

impl fmt::Display for DenseTensorError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Meta(error) => write!(f, "invalid tensor metadata: {error}"),
            Self::UnsupportedDType(dtype) => write!(
                f,
                "unsupported dense tensor dtype: expected F64, got {dtype:?}"
            ),
            Self::UnsupportedLayout => {
                write!(f, "dense tensor requires contiguous layout")
            }
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
        }
    }
}

impl std::error::Error for DenseTensorError {}

impl From<TensorMetaError> for DenseTensorError {
    fn from(value: TensorMetaError) -> Self {
        Self::Meta(value)
    }
}

impl DenseTensor {
    pub fn from_storage(meta: TensorMeta, storage: Vec<f64>) -> Result<Self, DenseTensorError> {
        meta.validate()?;
        if meta.dtype() != DType::F64 {
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

    pub fn from_contiguous_values(
        values: Vec<f64>,
        shape: Vec<usize>,
        device: Device,
    ) -> Result<Self, DenseTensorError> {
        let meta = TensorMeta::from_shape(shape, DType::F64, device);
        Self::from_storage(meta, values)
    }

    fn contiguous_required_len(meta: &TensorMeta) -> Result<usize, DenseTensorError> {
        meta.storage_offset().checked_add(meta.numel()).ok_or(
            DenseTensorError::StorageSpanOverflow {
                storage_offset: meta.storage_offset(),
                numel: meta.numel(),
            },
        )
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
        Ok(&self.storage[start..end])
    }

    pub fn contiguous_values(&self) -> Result<&[f64], DenseTensorError> {
        if !self.meta.is_contiguous() {
            return Err(DenseTensorError::UnsupportedLayout);
        }
        self.dispatch_values()
    }

    #[must_use]
    pub fn storage(&self) -> &[f64] {
        &self.storage
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

    /// Update the contiguous values in-place and bump the version counter.
    ///
    /// The new values must exactly match the length of the contiguous slice.
    pub fn update_contiguous_values(&mut self, new_values: &[f64]) -> Result<(), DenseTensorError> {
        if !self.meta.is_contiguous() {
            return Err(DenseTensorError::UnsupportedLayout);
        }
        let start = self.meta.storage_offset();
        let end = Self::contiguous_required_len(&self.meta)?;
        let slice = &mut self.storage[start..end];
        if new_values.len() != slice.len() {
            return Err(DenseTensorError::InsufficientStorage {
                needed: slice.len(),
                actual: new_values.len(),
            });
        }
        slice.copy_from_slice(new_values);
        self.version += 1;
        Ok(())
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

#[cfg(test)]
mod tests {
    use std::collections::BTreeMap;

    use proptest::prelude::*;

    use super::{
        DType, DenseTensor, DenseTensorError, Device, ScalarTensor, TensorMeta, TensorMetaError,
        contiguous_strides, ensure_compatible,
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
        assert_eq!(dt.storage(), &[10.0, 20.0, 30.0, 40.0]);
    }

    #[test]
    fn dense_tensor_replace_storage_success() {
        let mut dt = DenseTensor::from_contiguous_values(vec![1.0, 2.0, 3.0], vec![3], Device::Cpu)
            .expect("create dense tensor");
        assert_eq!(dt.version(), 0);

        dt.update_contiguous_values(&[4.0, 5.0, 6.0])
            .expect("replace with same-length storage should succeed");
        assert_eq!(dt.version(), 1);
        assert_eq!(dt.storage(), &[4.0, 5.0, 6.0]);
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
    fn dense_tensor_from_storage_rejects_unsupported_dtype() {
        let meta = TensorMeta::from_shape(vec![2], DType::F32, Device::Cpu);
        let err = DenseTensor::from_storage(meta, vec![1.0, 2.0])
            .expect_err("F32 dtype should be rejected");
        assert!(
            matches!(err, DenseTensorError::UnsupportedDType(DType::F32)),
            "expected UnsupportedDType(F32), got {err:?}"
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
}

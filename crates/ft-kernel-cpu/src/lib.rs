#![forbid(unsafe_code)]

use std::fmt;

use ft_core::{ScalarTensor, TensorCompatError, TensorMeta, ensure_compatible};

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum KernelError {
    Incompatible(TensorCompatError),
    ShapeMismatch {
        lhs: Vec<usize>,
        rhs: Vec<usize>,
    },
    UnsupportedLayout {
        side: &'static str,
    },
    StorageSpanOverflow {
        side: &'static str,
        storage_offset: usize,
        numel: usize,
    },
    InsufficientStorage {
        side: &'static str,
        needed: usize,
        available: usize,
    },
}

impl fmt::Display for KernelError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Incompatible(error) => write!(f, "incompatible tensors: {error}"),
            Self::ShapeMismatch { lhs, rhs } => {
                write!(f, "shape mismatch: lhs={lhs:?}, rhs={rhs:?}")
            }
            Self::UnsupportedLayout { side } => {
                write!(f, "unsupported non-contiguous layout on {side}")
            }
            Self::StorageSpanOverflow {
                side,
                storage_offset,
                numel,
            } => write!(
                f,
                "storage span overflow on {side}: storage_offset={storage_offset}, numel={numel}"
            ),
            Self::InsufficientStorage {
                side,
                needed,
                available,
            } => write!(
                f,
                "insufficient storage on {side}: needed={needed}, available={available}"
            ),
        }
    }
}

impl std::error::Error for KernelError {}

impl From<TensorCompatError> for KernelError {
    fn from(value: TensorCompatError) -> Self {
        Self::Incompatible(value)
    }
}

pub fn add_scalar(lhs: &ScalarTensor, rhs: &ScalarTensor) -> Result<ScalarTensor, KernelError> {
    ensure_compatible(lhs, rhs)?;
    Ok(lhs.with_value(lhs.value() + rhs.value()))
}

pub fn sub_scalar(lhs: &ScalarTensor, rhs: &ScalarTensor) -> Result<ScalarTensor, KernelError> {
    ensure_compatible(lhs, rhs)?;
    Ok(lhs.with_value(lhs.value() - rhs.value()))
}

pub fn div_scalar(lhs: &ScalarTensor, rhs: &ScalarTensor) -> Result<ScalarTensor, KernelError> {
    ensure_compatible(lhs, rhs)?;
    Ok(lhs.with_value(lhs.value() / rhs.value()))
}

pub fn mul_scalar(lhs: &ScalarTensor, rhs: &ScalarTensor) -> Result<ScalarTensor, KernelError> {
    ensure_compatible(lhs, rhs)?;
    Ok(lhs.with_value(lhs.value() * rhs.value()))
}

fn ensure_dtype_device_and_layout(lhs: &TensorMeta, rhs: &TensorMeta) -> Result<(), KernelError> {
    if lhs.dtype() != rhs.dtype() {
        return Err(KernelError::Incompatible(
            TensorCompatError::DTypeMismatch {
                lhs: lhs.dtype(),
                rhs: rhs.dtype(),
            },
        ));
    }

    if lhs.device() != rhs.device() {
        return Err(KernelError::Incompatible(
            TensorCompatError::DeviceMismatch {
                lhs: lhs.device(),
                rhs: rhs.device(),
            },
        ));
    }

    if !lhs.is_contiguous() {
        return Err(KernelError::UnsupportedLayout { side: "lhs" });
    }

    if !rhs.is_contiguous() {
        return Err(KernelError::UnsupportedLayout { side: "rhs" });
    }

    Ok(())
}

fn ensure_meta_compatible(lhs: &TensorMeta, rhs: &TensorMeta) -> Result<(), KernelError> {
    ensure_dtype_device_and_layout(lhs, rhs)?;

    if lhs.shape() != rhs.shape() {
        return Err(KernelError::ShapeMismatch {
            lhs: lhs.shape().to_vec(),
            rhs: rhs.shape().to_vec(),
        });
    }

    Ok(())
}

fn matmul_dims(
    lhs_meta: &TensorMeta,
    rhs_meta: &TensorMeta,
) -> Result<(usize, usize, usize), KernelError> {
    if lhs_meta.shape().len() != 2 || rhs_meta.shape().len() != 2 {
        return Err(KernelError::ShapeMismatch {
            lhs: lhs_meta.shape().to_vec(),
            rhs: rhs_meta.shape().to_vec(),
        });
    }

    let m = lhs_meta.shape()[0];
    let k = lhs_meta.shape()[1];
    let rhs_k = rhs_meta.shape()[0];
    let n = rhs_meta.shape()[1];
    if k != rhs_k {
        return Err(KernelError::ShapeMismatch {
            lhs: lhs_meta.shape().to_vec(),
            rhs: rhs_meta.shape().to_vec(),
        });
    }

    Ok((m, k, n))
}

fn contiguous_required_len(meta: &TensorMeta, side: &'static str) -> Result<usize, KernelError> {
    let numel = meta.numel();
    if numel == 0 {
        return Ok(0);
    }

    meta.storage_offset()
        .checked_add(numel)
        .ok_or(KernelError::StorageSpanOverflow {
            side,
            storage_offset: meta.storage_offset(),
            numel,
        })
}

fn ensure_storage_len(
    buffer: &[f64],
    meta: &TensorMeta,
    side: &'static str,
) -> Result<(), KernelError> {
    let needed = contiguous_required_len(meta, side)?;
    if buffer.len() < needed {
        return Err(KernelError::InsufficientStorage {
            side,
            needed,
            available: buffer.len(),
        });
    }
    Ok(())
}

fn elementwise_contiguous_f64<F>(
    lhs: &[f64],
    rhs: &[f64],
    lhs_meta: &TensorMeta,
    rhs_meta: &TensorMeta,
    op: F,
) -> Result<Vec<f64>, KernelError>
where
    F: Fn(f64, f64) -> f64,
{
    ensure_meta_compatible(lhs_meta, rhs_meta)?;
    ensure_storage_len(lhs, lhs_meta, "lhs")?;
    ensure_storage_len(rhs, rhs_meta, "rhs")?;

    let numel = lhs_meta.numel();
    if numel == 0 {
        return Ok(Vec::new());
    }

    let lhs_start = lhs_meta.storage_offset();
    let rhs_start = rhs_meta.storage_offset();
    let lhs_window = &lhs[lhs_start..lhs_start + numel];
    let rhs_window = &rhs[rhs_start..rhs_start + numel];

    Ok(lhs_window
        .iter()
        .zip(rhs_window.iter())
        .map(|(left, right)| op(*left, *right))
        .collect())
}

pub fn add_tensor_contiguous_f64(
    lhs: &[f64],
    rhs: &[f64],
    lhs_meta: &TensorMeta,
    rhs_meta: &TensorMeta,
) -> Result<Vec<f64>, KernelError> {
    elementwise_contiguous_f64(lhs, rhs, lhs_meta, rhs_meta, |left, right| left + right)
}

pub fn sub_tensor_contiguous_f64(
    lhs: &[f64],
    rhs: &[f64],
    lhs_meta: &TensorMeta,
    rhs_meta: &TensorMeta,
) -> Result<Vec<f64>, KernelError> {
    elementwise_contiguous_f64(lhs, rhs, lhs_meta, rhs_meta, |left, right| left - right)
}

pub fn div_tensor_contiguous_f64(
    lhs: &[f64],
    rhs: &[f64],
    lhs_meta: &TensorMeta,
    rhs_meta: &TensorMeta,
) -> Result<Vec<f64>, KernelError> {
    elementwise_contiguous_f64(lhs, rhs, lhs_meta, rhs_meta, |left, right| left / right)
}

pub fn mul_tensor_contiguous_f64(
    lhs: &[f64],
    rhs: &[f64],
    lhs_meta: &TensorMeta,
    rhs_meta: &TensorMeta,
) -> Result<Vec<f64>, KernelError> {
    elementwise_contiguous_f64(lhs, rhs, lhs_meta, rhs_meta, |left, right| left * right)
}

pub fn matmul_tensor_contiguous_f64(
    lhs: &[f64],
    rhs: &[f64],
    lhs_meta: &TensorMeta,
    rhs_meta: &TensorMeta,
) -> Result<Vec<f64>, KernelError> {
    ensure_dtype_device_and_layout(lhs_meta, rhs_meta)?;
    let (m, k, n) = matmul_dims(lhs_meta, rhs_meta)?;

    ensure_storage_len(lhs, lhs_meta, "lhs")?;
    ensure_storage_len(rhs, rhs_meta, "rhs")?;

    let lhs_start = lhs_meta.storage_offset();
    let rhs_start = rhs_meta.storage_offset();
    let mut out = vec![0.0; m.saturating_mul(n)];

    for row in 0..m {
        let out_row_base = row * n;
        let lhs_row_base = lhs_start + row * k;
        for col in 0..n {
            let mut acc = 0.0;
            for inner in 0..k {
                let lhs_idx = lhs_row_base + inner;
                let rhs_idx = rhs_start + inner * n + col;
                acc += lhs[lhs_idx] * rhs[rhs_idx];
            }
            out[out_row_base + col] = acc;
        }
    }

    Ok(out)
}

#[cfg(test)]
mod tests {
    use ft_core::{DType, Device, ScalarTensor, TensorCompatError, TensorMeta};

    use super::{
        KernelError, add_scalar, add_tensor_contiguous_f64, div_scalar, div_tensor_contiguous_f64,
        matmul_tensor_contiguous_f64, mul_scalar, mul_tensor_contiguous_f64, sub_scalar,
        sub_tensor_contiguous_f64,
    };

    #[test]
    fn add_scalar_returns_expected_value() {
        let lhs = ScalarTensor::new(2.0, DType::F64, Device::Cpu);
        let rhs = ScalarTensor::new(3.5, DType::F64, Device::Cpu);

        let out = add_scalar(&lhs, &rhs).expect("add should succeed");
        assert_eq!(out.value(), 5.5);
    }

    #[test]
    fn mul_scalar_returns_expected_value() {
        let lhs = ScalarTensor::new(2.0, DType::F64, Device::Cpu);
        let rhs = ScalarTensor::new(4.0, DType::F64, Device::Cpu);

        let out = mul_scalar(&lhs, &rhs).expect("mul should succeed");
        assert_eq!(out.value(), 8.0);
    }

    #[test]
    fn sub_scalar_returns_expected_value() {
        let lhs = ScalarTensor::new(2.0, DType::F64, Device::Cpu);
        let rhs = ScalarTensor::new(4.0, DType::F64, Device::Cpu);

        let out = sub_scalar(&lhs, &rhs).expect("sub should succeed");
        assert_eq!(out.value(), -2.0);
    }

    #[test]
    fn div_scalar_returns_expected_value() {
        let lhs = ScalarTensor::new(7.0, DType::F64, Device::Cpu);
        let rhs = ScalarTensor::new(2.0, DType::F64, Device::Cpu);

        let out = div_scalar(&lhs, &rhs).expect("div should succeed");
        assert_eq!(out.value(), 3.5);
    }

    #[test]
    fn add_scalar_rejects_dtype_mismatch() {
        let lhs = ScalarTensor::new(2.0, DType::F64, Device::Cpu);
        let rhs = ScalarTensor::new(4.0, DType::F32, Device::Cpu);

        let err = add_scalar(&lhs, &rhs).expect_err("dtype mismatch must fail closed");
        assert!(matches!(
            err,
            KernelError::Incompatible(TensorCompatError::DTypeMismatch { .. })
        ));
    }

    #[test]
    fn mul_scalar_rejects_device_mismatch() {
        let lhs = ScalarTensor::new(2.0, DType::F64, Device::Cpu);
        let rhs = ScalarTensor::new(4.0, DType::F64, Device::Cuda);

        let err = mul_scalar(&lhs, &rhs).expect_err("device mismatch must fail closed");
        assert!(matches!(
            err,
            KernelError::Incompatible(TensorCompatError::DeviceMismatch { .. })
        ));
    }

    #[test]
    fn sub_scalar_rejects_dtype_mismatch() {
        let lhs = ScalarTensor::new(2.0, DType::F64, Device::Cpu);
        let rhs = ScalarTensor::new(4.0, DType::F32, Device::Cpu);

        let err = sub_scalar(&lhs, &rhs).expect_err("dtype mismatch must fail closed");
        assert!(matches!(
            err,
            KernelError::Incompatible(TensorCompatError::DTypeMismatch { .. })
        ));
    }

    #[test]
    fn div_scalar_rejects_device_mismatch() {
        let lhs = ScalarTensor::new(2.0, DType::F64, Device::Cpu);
        let rhs = ScalarTensor::new(4.0, DType::F64, Device::Cuda);

        let err = div_scalar(&lhs, &rhs).expect_err("device mismatch must fail closed");
        assert!(matches!(
            err,
            KernelError::Incompatible(TensorCompatError::DeviceMismatch { .. })
        ));
    }

    #[test]
    fn add_tensor_contiguous_returns_expected_values() {
        let meta = TensorMeta::from_shape(vec![2, 2], DType::F64, Device::Cpu);
        let lhs = vec![1.0, 2.0, 3.0, 4.0];
        let rhs = vec![0.5, 1.5, 2.5, 3.5];

        let out = add_tensor_contiguous_f64(&lhs, &rhs, &meta, &meta)
            .expect("contiguous add should succeed");
        assert_eq!(out, vec![1.5, 3.5, 5.5, 7.5]);
    }

    #[test]
    fn sub_tensor_contiguous_respects_storage_offsets() {
        let lhs_meta =
            TensorMeta::from_shape(vec![3], DType::F64, Device::Cpu).with_storage_offset(1);
        let rhs_meta =
            TensorMeta::from_shape(vec![3], DType::F64, Device::Cpu).with_storage_offset(2);
        let lhs = vec![99.0, 5.0, 7.0, 9.0];
        let rhs = vec![11.0, 22.0, 3.0, 4.0, 5.0];

        let out = sub_tensor_contiguous_f64(&lhs, &rhs, &lhs_meta, &rhs_meta)
            .expect("offset add should succeed");
        assert_eq!(out, vec![2.0, 3.0, 4.0]);
    }

    #[test]
    fn mul_tensor_contiguous_rejects_shape_mismatch() {
        let lhs_meta = TensorMeta::from_shape(vec![4], DType::F64, Device::Cpu);
        let rhs_meta = TensorMeta::from_shape(vec![2, 2], DType::F64, Device::Cpu);
        let lhs = vec![1.0, 2.0, 3.0, 4.0];
        let rhs = vec![1.0, 2.0, 3.0, 4.0];

        let err = mul_tensor_contiguous_f64(&lhs, &rhs, &lhs_meta, &rhs_meta)
            .expect_err("shape mismatch must fail closed");
        assert!(matches!(err, KernelError::ShapeMismatch { .. }));
    }

    #[test]
    fn div_tensor_contiguous_rejects_non_contiguous_layout() {
        let lhs_meta =
            TensorMeta::from_shape_and_strides(vec![2, 2], vec![1, 2], 0, DType::F64, Device::Cpu)
                .expect("test meta should be valid");
        let rhs_meta = TensorMeta::from_shape(vec![2, 2], DType::F64, Device::Cpu);
        let lhs = vec![2.0, 4.0, 6.0, 8.0];
        let rhs = vec![1.0, 2.0, 3.0, 4.0];

        let err = div_tensor_contiguous_f64(&lhs, &rhs, &lhs_meta, &rhs_meta)
            .expect_err("non-contiguous input must be rejected");
        assert!(matches!(
            err,
            KernelError::UnsupportedLayout { side: "lhs" }
        ));
    }

    #[test]
    fn add_tensor_contiguous_rejects_insufficient_storage() {
        let lhs_meta =
            TensorMeta::from_shape(vec![3], DType::F64, Device::Cpu).with_storage_offset(1);
        let rhs_meta =
            TensorMeta::from_shape(vec![3], DType::F64, Device::Cpu).with_storage_offset(1);
        let lhs = vec![10.0, 11.0, 12.0];
        let rhs = vec![20.0, 21.0, 22.0, 23.0];

        let err = add_tensor_contiguous_f64(&lhs, &rhs, &lhs_meta, &rhs_meta)
            .expect_err("insufficient storage should fail closed");
        assert!(matches!(
            err,
            KernelError::InsufficientStorage {
                side: "lhs",
                needed: 4,
                available: 3
            }
        ));
    }

    #[test]
    fn add_tensor_contiguous_rejects_dtype_mismatch() {
        let lhs_meta = TensorMeta::from_shape(vec![2], DType::F64, Device::Cpu);
        let rhs_meta = TensorMeta::from_shape(vec![2], DType::F32, Device::Cpu);
        let lhs = vec![1.0, 2.0];
        let rhs = vec![3.0, 4.0];

        let err = add_tensor_contiguous_f64(&lhs, &rhs, &lhs_meta, &rhs_meta)
            .expect_err("dtype mismatch must fail closed");
        assert!(matches!(
            err,
            KernelError::Incompatible(TensorCompatError::DTypeMismatch { .. })
        ));
    }

    #[test]
    fn add_tensor_contiguous_rejects_device_mismatch() {
        let lhs_meta = TensorMeta::from_shape(vec![2], DType::F64, Device::Cpu);
        let rhs_meta = TensorMeta::from_shape(vec![2], DType::F64, Device::Cuda);
        let lhs = vec![1.0, 2.0];
        let rhs = vec![3.0, 4.0];

        let err = add_tensor_contiguous_f64(&lhs, &rhs, &lhs_meta, &rhs_meta)
            .expect_err("device mismatch must fail closed");
        assert!(matches!(
            err,
            KernelError::Incompatible(TensorCompatError::DeviceMismatch { .. })
        ));
    }

    #[test]
    fn add_tensor_contiguous_empty_shape_returns_empty_output() {
        let meta =
            TensorMeta::from_shape(vec![0, 3], DType::F64, Device::Cpu).with_storage_offset(8);
        let lhs = Vec::new();
        let rhs = Vec::new();

        let out = add_tensor_contiguous_f64(&lhs, &rhs, &meta, &meta)
            .expect("empty tensors should succeed without touching storage");
        assert!(out.is_empty());
    }

    #[test]
    fn matmul_tensor_contiguous_returns_expected_values() {
        let lhs_meta = TensorMeta::from_shape(vec![2, 3], DType::F64, Device::Cpu);
        let rhs_meta = TensorMeta::from_shape(vec![3, 2], DType::F64, Device::Cpu);
        let lhs = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let rhs = vec![7.0, 8.0, 9.0, 10.0, 11.0, 12.0];

        let out = matmul_tensor_contiguous_f64(&lhs, &rhs, &lhs_meta, &rhs_meta)
            .expect("contiguous matmul should succeed");
        assert_eq!(out, vec![58.0, 64.0, 139.0, 154.0]);
    }

    #[test]
    fn matmul_tensor_contiguous_rejects_rank_mismatch() {
        let lhs_meta = TensorMeta::from_shape(vec![6], DType::F64, Device::Cpu);
        let rhs_meta = TensorMeta::from_shape(vec![3, 2], DType::F64, Device::Cpu);
        let lhs = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let rhs = vec![7.0, 8.0, 9.0, 10.0, 11.0, 12.0];

        let err = matmul_tensor_contiguous_f64(&lhs, &rhs, &lhs_meta, &rhs_meta)
            .expect_err("rank mismatch must fail closed");
        assert!(matches!(err, KernelError::ShapeMismatch { .. }));
    }

    #[test]
    fn matmul_tensor_contiguous_rejects_inner_dimension_mismatch() {
        let lhs_meta = TensorMeta::from_shape(vec![2, 2], DType::F64, Device::Cpu);
        let rhs_meta = TensorMeta::from_shape(vec![3, 2], DType::F64, Device::Cpu);
        let lhs = vec![1.0, 2.0, 3.0, 4.0];
        let rhs = vec![7.0, 8.0, 9.0, 10.0, 11.0, 12.0];

        let err = matmul_tensor_contiguous_f64(&lhs, &rhs, &lhs_meta, &rhs_meta)
            .expect_err("inner-dimension mismatch must fail closed");
        assert!(matches!(err, KernelError::ShapeMismatch { .. }));
    }
}

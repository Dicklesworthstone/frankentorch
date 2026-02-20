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
    InvalidDimension {
        dim: usize,
        ndim: usize,
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
            Self::InvalidDimension { dim, ndim } => write!(
                f,
                "invalid reduction dimension {dim} for tensor with {ndim} dimensions"
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

pub fn neg_scalar(input: &ScalarTensor) -> ScalarTensor {
    input.with_value(-input.value())
}

pub fn abs_scalar(input: &ScalarTensor) -> ScalarTensor {
    input.with_value(input.value().abs())
}

pub fn exp_scalar(input: &ScalarTensor) -> ScalarTensor {
    input.with_value(input.value().exp())
}

pub fn log_scalar(input: &ScalarTensor) -> ScalarTensor {
    input.with_value(input.value().ln())
}

pub fn relu_scalar(input: &ScalarTensor) -> ScalarTensor {
    input.with_value(if input.value() > 0.0 {
        input.value()
    } else {
        0.0
    })
}

pub fn sigmoid_scalar(input: &ScalarTensor) -> ScalarTensor {
    input.with_value(1.0 / (1.0 + (-input.value()).exp()))
}

pub fn tanh_scalar(input: &ScalarTensor) -> ScalarTensor {
    input.with_value(input.value().tanh())
}

pub fn sqrt_scalar(input: &ScalarTensor) -> ScalarTensor {
    input.with_value(input.value().sqrt())
}

pub fn reciprocal_scalar(input: &ScalarTensor) -> ScalarTensor {
    input.with_value(1.0 / input.value())
}

pub fn sin_scalar(input: &ScalarTensor) -> ScalarTensor {
    input.with_value(input.value().sin())
}

pub fn cos_scalar(input: &ScalarTensor) -> ScalarTensor {
    input.with_value(input.value().cos())
}

pub fn tan_scalar(input: &ScalarTensor) -> ScalarTensor {
    input.with_value(input.value().tan())
}

pub fn floor_scalar(input: &ScalarTensor) -> ScalarTensor {
    input.with_value(input.value().floor())
}

pub fn ceil_scalar(input: &ScalarTensor) -> ScalarTensor {
    input.with_value(input.value().ceil())
}

pub fn round_scalar(input: &ScalarTensor) -> ScalarTensor {
    input.with_value(input.value().round())
}

pub fn log2_scalar(input: &ScalarTensor) -> ScalarTensor {
    input.with_value(input.value().log2())
}

pub fn log10_scalar(input: &ScalarTensor) -> ScalarTensor {
    input.with_value(input.value().log10())
}

pub fn log1p_scalar(input: &ScalarTensor) -> ScalarTensor {
    input.with_value(input.value().ln_1p())
}

pub fn expm1_scalar(input: &ScalarTensor) -> ScalarTensor {
    input.with_value(input.value().exp_m1())
}

pub fn sign_scalar(input: &ScalarTensor) -> ScalarTensor {
    input.with_value(input.value().signum())
}

pub fn trunc_scalar(input: &ScalarTensor) -> ScalarTensor {
    input.with_value(input.value().trunc())
}

pub fn frac_scalar(input: &ScalarTensor) -> ScalarTensor {
    input.with_value(input.value().fract())
}

pub fn asin_scalar(input: &ScalarTensor) -> ScalarTensor {
    input.with_value(input.value().asin())
}

pub fn acos_scalar(input: &ScalarTensor) -> ScalarTensor {
    input.with_value(input.value().acos())
}

pub fn atan_scalar(input: &ScalarTensor) -> ScalarTensor {
    input.with_value(input.value().atan())
}

pub fn sinh_scalar(input: &ScalarTensor) -> ScalarTensor {
    input.with_value(input.value().sinh())
}

pub fn cosh_scalar(input: &ScalarTensor) -> ScalarTensor {
    input.with_value(input.value().cosh())
}

fn gelu_value(x: f64) -> f64 {
    // GELU(x) = 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
    let c = std::f64::consts::FRAC_2_SQRT_PI * std::f64::consts::FRAC_1_SQRT_2; // sqrt(2/pi)
    let k = c * (x + 0.044715 * x * x * x);
    0.5 * x * (1.0 + k.tanh())
}

pub fn gelu_scalar(input: &ScalarTensor) -> ScalarTensor {
    input.with_value(gelu_value(input.value()))
}

fn silu_value(x: f64) -> f64 {
    x / (1.0 + (-x).exp())
}

pub fn silu_scalar(input: &ScalarTensor) -> ScalarTensor {
    input.with_value(silu_value(input.value()))
}

fn leaky_relu_value(x: f64) -> f64 {
    if x >= 0.0 { x } else { 0.01 * x }
}

pub fn leaky_relu_scalar(input: &ScalarTensor) -> ScalarTensor {
    input.with_value(leaky_relu_value(input.value()))
}

fn elu_value(x: f64) -> f64 {
    if x > 0.0 { x } else { x.exp() - 1.0 }
}

pub fn elu_scalar(input: &ScalarTensor) -> ScalarTensor {
    input.with_value(elu_value(input.value()))
}

pub fn pow_scalar(input: &ScalarTensor, exponent: f64) -> ScalarTensor {
    input.with_value(input.value().powf(exponent))
}

pub fn clamp_scalar(input: &ScalarTensor, min_val: f64, max_val: f64) -> ScalarTensor {
    input.with_value(input.value().clamp(min_val, max_val))
}

pub fn min_scalar(lhs: &ScalarTensor, rhs: &ScalarTensor) -> Result<ScalarTensor, KernelError> {
    ensure_compatible(lhs, rhs)?;
    Ok(lhs.with_value(lhs.value().min(rhs.value())))
}

pub fn max_scalar(lhs: &ScalarTensor, rhs: &ScalarTensor) -> Result<ScalarTensor, KernelError> {
    ensure_compatible(lhs, rhs)?;
    Ok(lhs.with_value(lhs.value().max(rhs.value())))
}

pub fn eq_scalar(lhs: &ScalarTensor, rhs: &ScalarTensor) -> Result<ScalarTensor, KernelError> {
    ensure_compatible(lhs, rhs)?;
    Ok(lhs.with_value(if lhs.value() == rhs.value() { 1.0 } else { 0.0 }))
}

pub fn ne_scalar(lhs: &ScalarTensor, rhs: &ScalarTensor) -> Result<ScalarTensor, KernelError> {
    ensure_compatible(lhs, rhs)?;
    Ok(lhs.with_value(if lhs.value() != rhs.value() { 1.0 } else { 0.0 }))
}

pub fn lt_scalar(lhs: &ScalarTensor, rhs: &ScalarTensor) -> Result<ScalarTensor, KernelError> {
    ensure_compatible(lhs, rhs)?;
    Ok(lhs.with_value(if lhs.value() < rhs.value() { 1.0 } else { 0.0 }))
}

pub fn gt_scalar(lhs: &ScalarTensor, rhs: &ScalarTensor) -> Result<ScalarTensor, KernelError> {
    ensure_compatible(lhs, rhs)?;
    Ok(lhs.with_value(if lhs.value() > rhs.value() { 1.0 } else { 0.0 }))
}

pub fn le_scalar(lhs: &ScalarTensor, rhs: &ScalarTensor) -> Result<ScalarTensor, KernelError> {
    ensure_compatible(lhs, rhs)?;
    Ok(lhs.with_value(if lhs.value() <= rhs.value() { 1.0 } else { 0.0 }))
}

pub fn ge_scalar(lhs: &ScalarTensor, rhs: &ScalarTensor) -> Result<ScalarTensor, KernelError> {
    ensure_compatible(lhs, rhs)?;
    Ok(lhs.with_value(if lhs.value() >= rhs.value() { 1.0 } else { 0.0 }))
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

fn ensure_unary_layout_and_storage(buffer: &[f64], meta: &TensorMeta) -> Result<(), KernelError> {
    if !meta.is_contiguous() {
        return Err(KernelError::UnsupportedLayout { side: "input" });
    }
    ensure_storage_len(buffer, meta, "input")
}

fn unary_contiguous_f64<F>(input: &[f64], meta: &TensorMeta, op: F) -> Result<Vec<f64>, KernelError>
where
    F: Fn(f64) -> f64,
{
    ensure_unary_layout_and_storage(input, meta)?;

    let numel = meta.numel();
    if numel == 0 {
        return Ok(Vec::new());
    }

    let start = meta.storage_offset();
    let window = &input[start..start + numel];

    Ok(window.iter().map(|value| op(*value)).collect())
}

pub fn neg_tensor_contiguous_f64(
    input: &[f64],
    meta: &TensorMeta,
) -> Result<Vec<f64>, KernelError> {
    unary_contiguous_f64(input, meta, |value| -value)
}

pub fn abs_tensor_contiguous_f64(
    input: &[f64],
    meta: &TensorMeta,
) -> Result<Vec<f64>, KernelError> {
    unary_contiguous_f64(input, meta, |value| value.abs())
}

pub fn exp_tensor_contiguous_f64(
    input: &[f64],
    meta: &TensorMeta,
) -> Result<Vec<f64>, KernelError> {
    unary_contiguous_f64(input, meta, |value| value.exp())
}

pub fn log_tensor_contiguous_f64(
    input: &[f64],
    meta: &TensorMeta,
) -> Result<Vec<f64>, KernelError> {
    unary_contiguous_f64(input, meta, |value| value.ln())
}

pub fn relu_tensor_contiguous_f64(
    input: &[f64],
    meta: &TensorMeta,
) -> Result<Vec<f64>, KernelError> {
    unary_contiguous_f64(input, meta, |value| if value > 0.0 { value } else { 0.0 })
}

pub fn sigmoid_tensor_contiguous_f64(
    input: &[f64],
    meta: &TensorMeta,
) -> Result<Vec<f64>, KernelError> {
    unary_contiguous_f64(input, meta, |value| 1.0 / (1.0 + (-value).exp()))
}

pub fn tanh_tensor_contiguous_f64(
    input: &[f64],
    meta: &TensorMeta,
) -> Result<Vec<f64>, KernelError> {
    unary_contiguous_f64(input, meta, |value| value.tanh())
}

pub fn sqrt_tensor_contiguous_f64(
    input: &[f64],
    meta: &TensorMeta,
) -> Result<Vec<f64>, KernelError> {
    unary_contiguous_f64(input, meta, |value| value.sqrt())
}

pub fn reciprocal_tensor_contiguous_f64(
    input: &[f64],
    meta: &TensorMeta,
) -> Result<Vec<f64>, KernelError> {
    unary_contiguous_f64(input, meta, |value| 1.0 / value)
}

pub fn sin_tensor_contiguous_f64(
    input: &[f64],
    meta: &TensorMeta,
) -> Result<Vec<f64>, KernelError> {
    unary_contiguous_f64(input, meta, |value| value.sin())
}

pub fn cos_tensor_contiguous_f64(
    input: &[f64],
    meta: &TensorMeta,
) -> Result<Vec<f64>, KernelError> {
    unary_contiguous_f64(input, meta, |value| value.cos())
}

pub fn tan_tensor_contiguous_f64(
    input: &[f64],
    meta: &TensorMeta,
) -> Result<Vec<f64>, KernelError> {
    unary_contiguous_f64(input, meta, |value| value.tan())
}

pub fn floor_tensor_contiguous_f64(
    input: &[f64],
    meta: &TensorMeta,
) -> Result<Vec<f64>, KernelError> {
    unary_contiguous_f64(input, meta, |value| value.floor())
}

pub fn ceil_tensor_contiguous_f64(
    input: &[f64],
    meta: &TensorMeta,
) -> Result<Vec<f64>, KernelError> {
    unary_contiguous_f64(input, meta, |value| value.ceil())
}

pub fn round_tensor_contiguous_f64(
    input: &[f64],
    meta: &TensorMeta,
) -> Result<Vec<f64>, KernelError> {
    unary_contiguous_f64(input, meta, |value| value.round())
}

pub fn log2_tensor_contiguous_f64(
    input: &[f64],
    meta: &TensorMeta,
) -> Result<Vec<f64>, KernelError> {
    unary_contiguous_f64(input, meta, |value| value.log2())
}

pub fn log10_tensor_contiguous_f64(
    input: &[f64],
    meta: &TensorMeta,
) -> Result<Vec<f64>, KernelError> {
    unary_contiguous_f64(input, meta, |value| value.log10())
}

pub fn log1p_tensor_contiguous_f64(
    input: &[f64],
    meta: &TensorMeta,
) -> Result<Vec<f64>, KernelError> {
    unary_contiguous_f64(input, meta, |value| value.ln_1p())
}

pub fn expm1_tensor_contiguous_f64(
    input: &[f64],
    meta: &TensorMeta,
) -> Result<Vec<f64>, KernelError> {
    unary_contiguous_f64(input, meta, |value| value.exp_m1())
}

pub fn sign_tensor_contiguous_f64(
    input: &[f64],
    meta: &TensorMeta,
) -> Result<Vec<f64>, KernelError> {
    unary_contiguous_f64(input, meta, |value| value.signum())
}

pub fn trunc_tensor_contiguous_f64(
    input: &[f64],
    meta: &TensorMeta,
) -> Result<Vec<f64>, KernelError> {
    unary_contiguous_f64(input, meta, |value| value.trunc())
}

pub fn frac_tensor_contiguous_f64(
    input: &[f64],
    meta: &TensorMeta,
) -> Result<Vec<f64>, KernelError> {
    unary_contiguous_f64(input, meta, |value| value.fract())
}

pub fn asin_tensor_contiguous_f64(
    input: &[f64],
    meta: &TensorMeta,
) -> Result<Vec<f64>, KernelError> {
    unary_contiguous_f64(input, meta, |value| value.asin())
}

pub fn acos_tensor_contiguous_f64(
    input: &[f64],
    meta: &TensorMeta,
) -> Result<Vec<f64>, KernelError> {
    unary_contiguous_f64(input, meta, |value| value.acos())
}

pub fn atan_tensor_contiguous_f64(
    input: &[f64],
    meta: &TensorMeta,
) -> Result<Vec<f64>, KernelError> {
    unary_contiguous_f64(input, meta, |value| value.atan())
}

pub fn sinh_tensor_contiguous_f64(
    input: &[f64],
    meta: &TensorMeta,
) -> Result<Vec<f64>, KernelError> {
    unary_contiguous_f64(input, meta, |value| value.sinh())
}

pub fn cosh_tensor_contiguous_f64(
    input: &[f64],
    meta: &TensorMeta,
) -> Result<Vec<f64>, KernelError> {
    unary_contiguous_f64(input, meta, |value| value.cosh())
}

pub fn gelu_tensor_contiguous_f64(
    input: &[f64],
    meta: &TensorMeta,
) -> Result<Vec<f64>, KernelError> {
    unary_contiguous_f64(input, meta, gelu_value)
}

pub fn silu_tensor_contiguous_f64(
    input: &[f64],
    meta: &TensorMeta,
) -> Result<Vec<f64>, KernelError> {
    unary_contiguous_f64(input, meta, silu_value)
}

pub fn leaky_relu_tensor_contiguous_f64(
    input: &[f64],
    meta: &TensorMeta,
) -> Result<Vec<f64>, KernelError> {
    unary_contiguous_f64(input, meta, leaky_relu_value)
}

pub fn elu_tensor_contiguous_f64(
    input: &[f64],
    meta: &TensorMeta,
) -> Result<Vec<f64>, KernelError> {
    unary_contiguous_f64(input, meta, elu_value)
}

pub fn pow_tensor_contiguous_f64(
    input: &[f64],
    meta: &TensorMeta,
    exponent: f64,
) -> Result<Vec<f64>, KernelError> {
    ensure_unary_layout_and_storage(input, meta)?;

    let numel = meta.numel();
    if numel == 0 {
        return Ok(Vec::new());
    }

    let start = meta.storage_offset();
    let window = &input[start..start + numel];

    Ok(window.iter().map(|value| value.powf(exponent)).collect())
}

pub fn clamp_tensor_contiguous_f64(
    input: &[f64],
    meta: &TensorMeta,
    min_val: f64,
    max_val: f64,
) -> Result<Vec<f64>, KernelError> {
    ensure_unary_layout_and_storage(input, meta)?;

    let numel = meta.numel();
    if numel == 0 {
        return Ok(Vec::new());
    }

    let start = meta.storage_offset();
    let window = &input[start..start + numel];

    Ok(window
        .iter()
        .map(|value| value.clamp(min_val, max_val))
        .collect())
}

pub fn min_tensor_contiguous_f64(
    lhs: &[f64],
    rhs: &[f64],
    lhs_meta: &TensorMeta,
    rhs_meta: &TensorMeta,
) -> Result<Vec<f64>, KernelError> {
    elementwise_contiguous_f64(lhs, rhs, lhs_meta, rhs_meta, |l, r| l.min(r))
}

pub fn max_tensor_contiguous_f64(
    lhs: &[f64],
    rhs: &[f64],
    lhs_meta: &TensorMeta,
    rhs_meta: &TensorMeta,
) -> Result<Vec<f64>, KernelError> {
    elementwise_contiguous_f64(lhs, rhs, lhs_meta, rhs_meta, |l, r| l.max(r))
}

pub fn eq_tensor_contiguous_f64(
    lhs: &[f64],
    rhs: &[f64],
    lhs_meta: &TensorMeta,
    rhs_meta: &TensorMeta,
) -> Result<Vec<f64>, KernelError> {
    elementwise_contiguous_f64(
        lhs,
        rhs,
        lhs_meta,
        rhs_meta,
        |l, r| {
            if l == r { 1.0 } else { 0.0 }
        },
    )
}

pub fn ne_tensor_contiguous_f64(
    lhs: &[f64],
    rhs: &[f64],
    lhs_meta: &TensorMeta,
    rhs_meta: &TensorMeta,
) -> Result<Vec<f64>, KernelError> {
    elementwise_contiguous_f64(
        lhs,
        rhs,
        lhs_meta,
        rhs_meta,
        |l, r| {
            if l != r { 1.0 } else { 0.0 }
        },
    )
}

pub fn lt_tensor_contiguous_f64(
    lhs: &[f64],
    rhs: &[f64],
    lhs_meta: &TensorMeta,
    rhs_meta: &TensorMeta,
) -> Result<Vec<f64>, KernelError> {
    elementwise_contiguous_f64(
        lhs,
        rhs,
        lhs_meta,
        rhs_meta,
        |l, r| {
            if l < r { 1.0 } else { 0.0 }
        },
    )
}

pub fn gt_tensor_contiguous_f64(
    lhs: &[f64],
    rhs: &[f64],
    lhs_meta: &TensorMeta,
    rhs_meta: &TensorMeta,
) -> Result<Vec<f64>, KernelError> {
    elementwise_contiguous_f64(
        lhs,
        rhs,
        lhs_meta,
        rhs_meta,
        |l, r| {
            if l > r { 1.0 } else { 0.0 }
        },
    )
}

pub fn le_tensor_contiguous_f64(
    lhs: &[f64],
    rhs: &[f64],
    lhs_meta: &TensorMeta,
    rhs_meta: &TensorMeta,
) -> Result<Vec<f64>, KernelError> {
    elementwise_contiguous_f64(
        lhs,
        rhs,
        lhs_meta,
        rhs_meta,
        |l, r| {
            if l <= r { 1.0 } else { 0.0 }
        },
    )
}

pub fn ge_tensor_contiguous_f64(
    lhs: &[f64],
    rhs: &[f64],
    lhs_meta: &TensorMeta,
    rhs_meta: &TensorMeta,
) -> Result<Vec<f64>, KernelError> {
    elementwise_contiguous_f64(
        lhs,
        rhs,
        lhs_meta,
        rhs_meta,
        |l, r| {
            if l >= r { 1.0 } else { 0.0 }
        },
    )
}

pub fn sum_tensor_contiguous_f64(input: &[f64], meta: &TensorMeta) -> Result<f64, KernelError> {
    ensure_unary_layout_and_storage(input, meta)?;
    let offset = meta.storage_offset();
    let numel = meta.numel();
    Ok(input[offset..offset + numel].iter().sum())
}

pub fn mean_tensor_contiguous_f64(input: &[f64], meta: &TensorMeta) -> Result<f64, KernelError> {
    ensure_unary_layout_and_storage(input, meta)?;
    let offset = meta.storage_offset();
    let numel = meta.numel();
    if numel == 0 {
        return Ok(f64::NAN);
    }
    let sum: f64 = input[offset..offset + numel].iter().sum();
    Ok(sum / numel as f64)
}

pub fn sum_dim_tensor_contiguous_f64(
    input: &[f64],
    meta: &TensorMeta,
    dim: usize,
) -> Result<Vec<f64>, KernelError> {
    ensure_unary_layout_and_storage(input, meta)?;
    let shape = meta.shape();
    let ndim = shape.len();
    if dim >= ndim {
        return Err(KernelError::InvalidDimension { dim, ndim });
    }
    let offset = meta.storage_offset();
    let reduce_size = shape[dim];
    let outer_size: usize = shape[..dim].iter().product();
    let inner_size: usize = shape[dim + 1..].iter().product();
    let out_numel = outer_size * inner_size;
    let mut output = vec![0.0; out_numel];
    let data = &input[offset..];

    for outer in 0..outer_size {
        for inner in 0..inner_size {
            let mut sum = 0.0;
            for r in 0..reduce_size {
                sum += data[outer * reduce_size * inner_size + r * inner_size + inner];
            }
            output[outer * inner_size + inner] = sum;
        }
    }

    Ok(output)
}

pub fn mean_dim_tensor_contiguous_f64(
    input: &[f64],
    meta: &TensorMeta,
    dim: usize,
) -> Result<Vec<f64>, KernelError> {
    ensure_unary_layout_and_storage(input, meta)?;
    let shape = meta.shape();
    let ndim = shape.len();
    if dim >= ndim {
        return Err(KernelError::InvalidDimension { dim, ndim });
    }
    let reduce_size = shape[dim];
    if reduce_size == 0 {
        let outer_size: usize = shape[..dim].iter().product();
        let inner_size: usize = shape[dim + 1..].iter().product();
        return Ok(vec![f64::NAN; outer_size * inner_size]);
    }
    let mut output = sum_dim_tensor_contiguous_f64(input, meta, dim)?;
    let scale = 1.0 / reduce_size as f64;
    for v in &mut output {
        *v *= scale;
    }
    Ok(output)
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

pub fn prod_dim_tensor_contiguous_f64(
    input: &[f64],
    meta: &TensorMeta,
    dim: usize,
) -> Result<Vec<f64>, KernelError> {
    ensure_unary_layout_and_storage(input, meta)?;
    let shape = meta.shape();
    let ndim = shape.len();
    if dim >= ndim {
        return Err(KernelError::InvalidDimension { dim, ndim });
    }
    let offset = meta.storage_offset();
    let reduce_size = shape[dim];
    let outer_size: usize = shape[..dim].iter().product();
    let inner_size: usize = shape[dim + 1..].iter().product();
    let out_numel = outer_size * inner_size;
    let mut output = vec![1.0; out_numel];
    let data = &input[offset..];

    for outer in 0..outer_size {
        for inner in 0..inner_size {
            let mut prod = 1.0;
            for r in 0..reduce_size {
                prod *= data[outer * reduce_size * inner_size + r * inner_size + inner];
            }
            output[outer * inner_size + inner] = prod;
        }
    }

    Ok(output)
}

pub fn var_dim_tensor_contiguous_f64(
    input: &[f64],
    meta: &TensorMeta,
    dim: usize,
) -> Result<Vec<f64>, KernelError> {
    ensure_unary_layout_and_storage(input, meta)?;
    let shape = meta.shape();
    let ndim = shape.len();
    if dim >= ndim {
        return Err(KernelError::InvalidDimension { dim, ndim });
    }
    let offset = meta.storage_offset();
    let reduce_size = shape[dim];
    let outer_size: usize = shape[..dim].iter().product();
    let inner_size: usize = shape[dim + 1..].iter().product();
    let out_numel = outer_size * inner_size;
    let data = &input[offset..];

    if reduce_size < 2 {
        return Ok(vec![f64::NAN; out_numel]);
    }

    let mut output = vec![0.0; out_numel];
    let correction = (reduce_size - 1) as f64; // Bessel's correction

    for outer in 0..outer_size {
        for inner in 0..inner_size {
            // Compute mean
            let mut sum = 0.0;
            for r in 0..reduce_size {
                sum += data[outer * reduce_size * inner_size + r * inner_size + inner];
            }
            let mean = sum / reduce_size as f64;
            // Compute variance
            let mut var_sum = 0.0;
            for r in 0..reduce_size {
                let diff = data[outer * reduce_size * inner_size + r * inner_size + inner] - mean;
                var_sum += diff * diff;
            }
            output[outer * inner_size + inner] = var_sum / correction;
        }
    }

    Ok(output)
}

pub fn std_dim_tensor_contiguous_f64(
    input: &[f64],
    meta: &TensorMeta,
    dim: usize,
) -> Result<Vec<f64>, KernelError> {
    let mut output = var_dim_tensor_contiguous_f64(input, meta, dim)?;
    for v in &mut output {
        *v = v.sqrt();
    }
    Ok(output)
}

pub fn softmax_dim_tensor_contiguous_f64(
    input: &[f64],
    meta: &TensorMeta,
    dim: usize,
) -> Result<Vec<f64>, KernelError> {
    ensure_unary_layout_and_storage(input, meta)?;
    let shape = meta.shape();
    let ndim = shape.len();
    if dim >= ndim {
        return Err(KernelError::InvalidDimension { dim, ndim });
    }
    let offset = meta.storage_offset();
    let reduce_size = shape[dim];
    let outer_size: usize = shape[..dim].iter().product();
    let inner_size: usize = shape[dim + 1..].iter().product();
    let numel: usize = shape.iter().product();
    let mut output = vec![0.0; numel];
    let data = &input[offset..];

    for outer in 0..outer_size {
        for inner in 0..inner_size {
            // Find max for numerical stability
            let mut max_val = f64::NEG_INFINITY;
            for r in 0..reduce_size {
                let idx = outer * reduce_size * inner_size + r * inner_size + inner;
                if data[idx] > max_val {
                    max_val = data[idx];
                }
            }
            // Compute exp(x - max) and sum
            let mut sum = 0.0;
            for r in 0..reduce_size {
                let idx = outer * reduce_size * inner_size + r * inner_size + inner;
                let e = (data[idx] - max_val).exp();
                output[idx] = e;
                sum += e;
            }
            // Normalize
            for r in 0..reduce_size {
                let idx = outer * reduce_size * inner_size + r * inner_size + inner;
                output[idx] /= sum;
            }
        }
    }

    Ok(output)
}

pub fn log_softmax_dim_tensor_contiguous_f64(
    input: &[f64],
    meta: &TensorMeta,
    dim: usize,
) -> Result<Vec<f64>, KernelError> {
    ensure_unary_layout_and_storage(input, meta)?;
    let shape = meta.shape();
    let ndim = shape.len();
    if dim >= ndim {
        return Err(KernelError::InvalidDimension { dim, ndim });
    }
    let offset = meta.storage_offset();
    let reduce_size = shape[dim];
    let outer_size: usize = shape[..dim].iter().product();
    let inner_size: usize = shape[dim + 1..].iter().product();
    let numel: usize = shape.iter().product();
    let mut output = vec![0.0; numel];
    let data = &input[offset..];

    for outer in 0..outer_size {
        for inner in 0..inner_size {
            // Find max for numerical stability
            let mut max_val = f64::NEG_INFINITY;
            for r in 0..reduce_size {
                let idx = outer * reduce_size * inner_size + r * inner_size + inner;
                if data[idx] > max_val {
                    max_val = data[idx];
                }
            }
            // Compute log(sum(exp(x - max)))
            let mut sum_exp = 0.0;
            for r in 0..reduce_size {
                let idx = outer * reduce_size * inner_size + r * inner_size + inner;
                sum_exp += (data[idx] - max_val).exp();
            }
            let log_sum_exp = max_val + sum_exp.ln();
            // output = x - log_sum_exp
            for r in 0..reduce_size {
                let idx = outer * reduce_size * inner_size + r * inner_size + inner;
                output[idx] = data[idx] - log_sum_exp;
            }
        }
    }

    Ok(output)
}

pub fn cat_tensor_contiguous_f64(
    inputs: &[(&[f64], &TensorMeta)],
    dim: usize,
) -> Result<Vec<f64>, KernelError> {
    if inputs.is_empty() {
        return Err(KernelError::ShapeMismatch {
            lhs: vec![],
            rhs: vec![],
        });
    }
    let first_shape = inputs[0].1.shape();
    let ndim = first_shape.len();
    if dim >= ndim {
        return Err(KernelError::InvalidDimension { dim, ndim });
    }
    // Validate shapes match except along cat dim
    for (data, meta) in inputs {
        ensure_unary_layout_and_storage(data, meta)?;
        let shape = meta.shape();
        if shape.len() != ndim {
            return Err(KernelError::ShapeMismatch {
                lhs: first_shape.to_vec(),
                rhs: shape.to_vec(),
            });
        }
        for d in 0..ndim {
            if d != dim && shape[d] != first_shape[d] {
                return Err(KernelError::ShapeMismatch {
                    lhs: first_shape.to_vec(),
                    rhs: shape.to_vec(),
                });
            }
        }
    }

    let outer_size: usize = first_shape[..dim].iter().product();
    let inner_size: usize = first_shape[dim + 1..].iter().product();
    let total_cat_size: usize = inputs.iter().map(|(_, m)| m.shape()[dim]).sum();
    let out_numel = outer_size * total_cat_size * inner_size;
    let mut output = Vec::with_capacity(out_numel);

    for outer in 0..outer_size {
        for (data, meta) in inputs {
            let shape = meta.shape();
            let cat_size = shape[dim];
            let offset = meta.storage_offset();
            let d = &data[offset..];
            for r in 0..cat_size {
                for inner in 0..inner_size {
                    output.push(d[outer * cat_size * inner_size + r * inner_size + inner]);
                }
            }
        }
    }

    Ok(output)
}

pub fn stack_tensor_contiguous_f64(
    inputs: &[(&[f64], &TensorMeta)],
    dim: usize,
) -> Result<Vec<f64>, KernelError> {
    if inputs.is_empty() {
        return Err(KernelError::ShapeMismatch {
            lhs: vec![],
            rhs: vec![],
        });
    }
    let first_shape = inputs[0].1.shape();
    let ndim = first_shape.len();
    // dim can be 0..=ndim (inserting a new dimension)
    if dim > ndim {
        return Err(KernelError::InvalidDimension {
            dim,
            ndim: ndim + 1,
        });
    }
    // Validate all shapes are identical
    for (data, meta) in inputs {
        ensure_unary_layout_and_storage(data, meta)?;
        if meta.shape() != first_shape {
            return Err(KernelError::ShapeMismatch {
                lhs: first_shape.to_vec(),
                rhs: meta.shape().to_vec(),
            });
        }
    }

    let num_inputs = inputs.len();
    let outer_size: usize = first_shape[..dim].iter().product();
    let inner_size: usize = first_shape[dim..].iter().product();
    let out_numel = outer_size * num_inputs * inner_size;
    let mut output = Vec::with_capacity(out_numel);

    for outer in 0..outer_size {
        for (data, meta) in inputs {
            let offset = meta.storage_offset();
            let d = &data[offset..];
            for inner in 0..inner_size {
                output.push(d[outer * inner_size + inner]);
            }
        }
    }

    Ok(output)
}

#[cfg(test)]
mod tests {
    use ft_core::{DType, Device, ScalarTensor, TensorCompatError, TensorMeta};

    use super::{
        KernelError, abs_scalar, abs_tensor_contiguous_f64, add_scalar, add_tensor_contiguous_f64,
        cat_tensor_contiguous_f64, clamp_scalar, clamp_tensor_contiguous_f64, div_scalar,
        div_tensor_contiguous_f64, eq_scalar, eq_tensor_contiguous_f64, exp_scalar,
        exp_tensor_contiguous_f64, ge_scalar, ge_tensor_contiguous_f64, gt_scalar,
        gt_tensor_contiguous_f64, le_scalar, le_tensor_contiguous_f64, log_scalar,
        log_softmax_dim_tensor_contiguous_f64, log_tensor_contiguous_f64, lt_scalar,
        lt_tensor_contiguous_f64, matmul_tensor_contiguous_f64, max_scalar,
        max_tensor_contiguous_f64, mean_dim_tensor_contiguous_f64, mean_tensor_contiguous_f64,
        min_scalar, min_tensor_contiguous_f64, mul_scalar, mul_tensor_contiguous_f64, ne_scalar,
        ne_tensor_contiguous_f64, neg_scalar, neg_tensor_contiguous_f64, pow_scalar,
        pow_tensor_contiguous_f64, prod_dim_tensor_contiguous_f64, reciprocal_scalar,
        reciprocal_tensor_contiguous_f64, relu_scalar, relu_tensor_contiguous_f64, sigmoid_scalar,
        sigmoid_tensor_contiguous_f64, softmax_dim_tensor_contiguous_f64, sqrt_scalar,
        sqrt_tensor_contiguous_f64, stack_tensor_contiguous_f64, std_dim_tensor_contiguous_f64,
        sub_scalar, sub_tensor_contiguous_f64, sum_dim_tensor_contiguous_f64,
        sum_tensor_contiguous_f64, tanh_scalar, tanh_tensor_contiguous_f64,
        var_dim_tensor_contiguous_f64,
    };

    #[test]
    fn neg_scalar_returns_expected_value() {
        let input = ScalarTensor::new(3.5, DType::F64, Device::Cpu);
        let out = neg_scalar(&input);
        assert_eq!(out.value(), -3.5);
    }

    #[test]
    fn neg_scalar_double_negation_identity() {
        let input = ScalarTensor::new(-7.0, DType::F64, Device::Cpu);
        let out = neg_scalar(&neg_scalar(&input));
        assert_eq!(out.value(), -7.0);
    }

    #[test]
    fn neg_scalar_zero_is_zero() {
        let input = ScalarTensor::new(0.0, DType::F64, Device::Cpu);
        let out = neg_scalar(&input);
        assert_eq!(out.value(), 0.0);
    }

    #[test]
    fn neg_tensor_contiguous_returns_expected_values() {
        let meta = TensorMeta::from_shape(vec![2, 2], DType::F64, Device::Cpu);
        let input = vec![1.0, -2.0, 3.0, -4.0];

        let out = neg_tensor_contiguous_f64(&input, &meta).expect("contiguous neg should succeed");
        assert_eq!(out, vec![-1.0, 2.0, -3.0, 4.0]);
    }

    #[test]
    fn neg_tensor_contiguous_respects_storage_offset() {
        let meta = TensorMeta::from_shape(vec![3], DType::F64, Device::Cpu).with_storage_offset(1);
        let input = vec![99.0, 5.0, -7.0, 9.0];

        let out = neg_tensor_contiguous_f64(&input, &meta).expect("offset neg should succeed");
        assert_eq!(out, vec![-5.0, 7.0, -9.0]);
    }

    #[test]
    fn neg_tensor_contiguous_empty_shape_returns_empty_output() {
        let meta =
            TensorMeta::from_shape(vec![0, 3], DType::F64, Device::Cpu).with_storage_offset(8);
        let input = Vec::new();

        let out =
            neg_tensor_contiguous_f64(&input, &meta).expect("empty tensor neg should succeed");
        assert!(out.is_empty());
    }

    #[test]
    fn neg_tensor_contiguous_rejects_non_contiguous_layout() {
        let meta =
            TensorMeta::from_shape_and_strides(vec![2, 2], vec![1, 2], 0, DType::F64, Device::Cpu)
                .expect("test meta should be valid");
        let input = vec![1.0, 2.0, 3.0, 4.0];

        let err = neg_tensor_contiguous_f64(&input, &meta)
            .expect_err("non-contiguous input must be rejected");
        assert!(matches!(
            err,
            KernelError::UnsupportedLayout { side: "input" }
        ));
    }

    #[test]
    fn neg_tensor_contiguous_rejects_insufficient_storage() {
        let meta = TensorMeta::from_shape(vec![3], DType::F64, Device::Cpu).with_storage_offset(1);
        let input = vec![10.0, 11.0, 12.0];

        let err = neg_tensor_contiguous_f64(&input, &meta)
            .expect_err("insufficient storage should fail closed");
        assert!(matches!(
            err,
            KernelError::InsufficientStorage {
                side: "input",
                needed: 4,
                available: 3
            }
        ));
    }

    #[test]
    fn abs_scalar_returns_expected_value() {
        let input = ScalarTensor::new(-3.5, DType::F64, Device::Cpu);
        let out = abs_scalar(&input);
        assert_eq!(out.value(), 3.5);
    }

    #[test]
    fn abs_scalar_positive_unchanged() {
        let input = ScalarTensor::new(7.0, DType::F64, Device::Cpu);
        let out = abs_scalar(&input);
        assert_eq!(out.value(), 7.0);
    }

    #[test]
    fn abs_scalar_zero_is_zero() {
        let input = ScalarTensor::new(0.0, DType::F64, Device::Cpu);
        let out = abs_scalar(&input);
        assert_eq!(out.value(), 0.0);
    }

    #[test]
    fn abs_tensor_contiguous_returns_expected_values() {
        let meta = TensorMeta::from_shape(vec![2, 2], DType::F64, Device::Cpu);
        let input = vec![-1.0, 2.0, -3.0, 4.0];

        let out = abs_tensor_contiguous_f64(&input, &meta).expect("contiguous abs should succeed");
        assert_eq!(out, vec![1.0, 2.0, 3.0, 4.0]);
    }

    #[test]
    fn abs_tensor_contiguous_empty_returns_empty() {
        let meta = TensorMeta::from_shape(vec![0], DType::F64, Device::Cpu);
        let input = Vec::new();

        let out = abs_tensor_contiguous_f64(&input, &meta).expect("empty abs should succeed");
        assert!(out.is_empty());
    }

    #[test]
    fn exp_scalar_returns_expected_value() {
        let input = ScalarTensor::new(0.0, DType::F64, Device::Cpu);
        let out = exp_scalar(&input);
        assert_eq!(out.value(), 1.0);
    }

    #[test]
    fn exp_scalar_of_one() {
        let input = ScalarTensor::new(1.0, DType::F64, Device::Cpu);
        let out = exp_scalar(&input);
        assert!((out.value() - std::f64::consts::E).abs() < 1e-10);
    }

    #[test]
    fn log_scalar_returns_expected_value() {
        let input = ScalarTensor::new(1.0, DType::F64, Device::Cpu);
        let out = log_scalar(&input);
        assert_eq!(out.value(), 0.0);
    }

    #[test]
    fn log_scalar_of_e() {
        let input = ScalarTensor::new(std::f64::consts::E, DType::F64, Device::Cpu);
        let out = log_scalar(&input);
        assert!((out.value() - 1.0).abs() < 1e-10);
    }

    #[test]
    fn exp_log_roundtrip() {
        let input = ScalarTensor::new(2.5, DType::F64, Device::Cpu);
        let exp_result = exp_scalar(&input);
        let log_result = log_scalar(&exp_result);
        assert!((log_result.value() - 2.5).abs() < 1e-10);
    }

    #[test]
    fn exp_tensor_contiguous_returns_expected_values() {
        let meta = TensorMeta::from_shape(vec![3], DType::F64, Device::Cpu);
        let input = vec![0.0, 1.0, 2.0];

        let out = exp_tensor_contiguous_f64(&input, &meta).expect("contiguous exp should succeed");
        assert_eq!(out.len(), 3);
        assert!((out[0] - 1.0).abs() < 1e-10);
        assert!((out[1] - std::f64::consts::E).abs() < 1e-10);
    }

    #[test]
    fn log_tensor_contiguous_returns_expected_values() {
        let meta = TensorMeta::from_shape(vec![2], DType::F64, Device::Cpu);
        let input = vec![1.0, std::f64::consts::E];

        let out = log_tensor_contiguous_f64(&input, &meta).expect("contiguous log should succeed");
        assert!((out[0] - 0.0).abs() < 1e-10);
        assert!((out[1] - 1.0).abs() < 1e-10);
    }

    #[test]
    fn relu_scalar_positive_passes_through() {
        let input = ScalarTensor::new(3.5, DType::F64, Device::Cpu);
        let out = relu_scalar(&input);
        assert_eq!(out.value(), 3.5);
    }

    #[test]
    fn relu_scalar_negative_returns_zero() {
        let input = ScalarTensor::new(-2.0, DType::F64, Device::Cpu);
        let out = relu_scalar(&input);
        assert_eq!(out.value(), 0.0);
    }

    #[test]
    fn relu_scalar_zero_returns_zero() {
        let input = ScalarTensor::new(0.0, DType::F64, Device::Cpu);
        let out = relu_scalar(&input);
        assert_eq!(out.value(), 0.0);
    }

    #[test]
    fn relu_tensor_contiguous_returns_expected_values() {
        let meta = TensorMeta::from_shape(vec![4], DType::F64, Device::Cpu);
        let input = vec![-1.0, 0.0, 2.0, -3.0];

        let out =
            relu_tensor_contiguous_f64(&input, &meta).expect("contiguous relu should succeed");
        assert_eq!(out, vec![0.0, 0.0, 2.0, 0.0]);
    }

    #[test]
    fn sigmoid_scalar_at_zero_returns_half() {
        let input = ScalarTensor::new(0.0, DType::F64, Device::Cpu);
        let out = sigmoid_scalar(&input);
        assert_eq!(out.value(), 0.5);
    }

    #[test]
    fn sigmoid_scalar_large_positive_approaches_one() {
        let input = ScalarTensor::new(10.0, DType::F64, Device::Cpu);
        let out = sigmoid_scalar(&input);
        assert!((out.value() - 1.0).abs() < 1e-4);
    }

    #[test]
    fn sigmoid_scalar_large_negative_approaches_zero() {
        let input = ScalarTensor::new(-10.0, DType::F64, Device::Cpu);
        let out = sigmoid_scalar(&input);
        assert!(out.value().abs() < 1e-4);
    }

    #[test]
    fn sigmoid_tensor_contiguous_returns_expected_values() {
        let meta = TensorMeta::from_shape(vec![3], DType::F64, Device::Cpu);
        let input = vec![0.0, 10.0, -10.0];

        let out = sigmoid_tensor_contiguous_f64(&input, &meta)
            .expect("contiguous sigmoid should succeed");
        assert!((out[0] - 0.5).abs() < 1e-10);
        assert!((out[1] - 1.0).abs() < 1e-4);
        assert!(out[2].abs() < 1e-4);
    }

    #[test]
    fn tanh_scalar_at_zero_returns_zero() {
        let input = ScalarTensor::new(0.0, DType::F64, Device::Cpu);
        let out = tanh_scalar(&input);
        assert_eq!(out.value(), 0.0);
    }

    #[test]
    fn tanh_scalar_large_positive_approaches_one() {
        let input = ScalarTensor::new(10.0, DType::F64, Device::Cpu);
        let out = tanh_scalar(&input);
        assert!((out.value() - 1.0).abs() < 1e-6);
    }

    #[test]
    fn tanh_scalar_is_odd_function() {
        let pos = ScalarTensor::new(1.5, DType::F64, Device::Cpu);
        let neg = ScalarTensor::new(-1.5, DType::F64, Device::Cpu);
        let out_pos = tanh_scalar(&pos);
        let out_neg = tanh_scalar(&neg);
        assert!((out_pos.value() + out_neg.value()).abs() < 1e-10);
    }

    #[test]
    fn tanh_tensor_contiguous_returns_expected_values() {
        let meta = TensorMeta::from_shape(vec![3], DType::F64, Device::Cpu);
        let input = vec![0.0, 10.0, -10.0];

        let out =
            tanh_tensor_contiguous_f64(&input, &meta).expect("contiguous tanh should succeed");
        assert!((out[0]).abs() < 1e-10);
        assert!((out[1] - 1.0).abs() < 1e-6);
        assert!((out[2] + 1.0).abs() < 1e-6);
    }

    #[test]
    fn sum_tensor_contiguous_returns_expected_value() {
        let meta = TensorMeta::from_shape(vec![4], DType::F64, Device::Cpu);
        let input = vec![1.0, 2.0, 3.0, 4.0];
        let out = sum_tensor_contiguous_f64(&input, &meta).expect("contiguous sum should succeed");
        assert_eq!(out, 10.0);
    }

    #[test]
    fn sum_tensor_contiguous_empty_returns_zero() {
        let meta = TensorMeta::from_shape(vec![0], DType::F64, Device::Cpu);
        let input: Vec<f64> = vec![];
        let out = sum_tensor_contiguous_f64(&input, &meta).expect("empty sum should succeed");
        assert_eq!(out, 0.0);
    }

    #[test]
    fn sum_tensor_contiguous_respects_storage_offset() {
        let meta = TensorMeta::from_shape_and_strides(vec![2], vec![1], 2, DType::F64, Device::Cpu)
            .expect("offset meta should validate");
        let storage = vec![100.0, 200.0, 3.0, 7.0];
        let out = sum_tensor_contiguous_f64(&storage, &meta).expect("offset sum should succeed");
        assert_eq!(out, 10.0);
    }

    #[test]
    fn mean_tensor_contiguous_returns_expected_value() {
        let meta = TensorMeta::from_shape(vec![4], DType::F64, Device::Cpu);
        let input = vec![1.0, 2.0, 3.0, 4.0];
        let out =
            mean_tensor_contiguous_f64(&input, &meta).expect("contiguous mean should succeed");
        assert_eq!(out, 2.5);
    }

    #[test]
    fn mean_tensor_contiguous_empty_returns_nan() {
        let meta = TensorMeta::from_shape(vec![0], DType::F64, Device::Cpu);
        let input: Vec<f64> = vec![];
        let out = mean_tensor_contiguous_f64(&input, &meta).expect("empty mean should succeed");
        assert!(out.is_nan());
    }

    #[test]
    fn mean_tensor_contiguous_single_element() {
        let meta = TensorMeta::from_shape(vec![1], DType::F64, Device::Cpu);
        let input = vec![42.0];
        let out = mean_tensor_contiguous_f64(&input, &meta).expect("single mean should succeed");
        assert_eq!(out, 42.0);
    }

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

    #[test]
    fn eq_scalar_returns_expected_values() {
        let a = ScalarTensor::new(3.0, DType::F64, Device::Cpu);
        let b = ScalarTensor::new(3.0, DType::F64, Device::Cpu);
        let c = ScalarTensor::new(4.0, DType::F64, Device::Cpu);
        assert_eq!(eq_scalar(&a, &b).expect("eq should succeed").value(), 1.0);
        assert_eq!(eq_scalar(&a, &c).expect("eq should succeed").value(), 0.0);
    }

    #[test]
    fn ne_scalar_returns_expected_values() {
        let a = ScalarTensor::new(3.0, DType::F64, Device::Cpu);
        let b = ScalarTensor::new(3.0, DType::F64, Device::Cpu);
        let c = ScalarTensor::new(4.0, DType::F64, Device::Cpu);
        assert_eq!(ne_scalar(&a, &b).expect("ne should succeed").value(), 0.0);
        assert_eq!(ne_scalar(&a, &c).expect("ne should succeed").value(), 1.0);
    }

    #[test]
    fn eq_ne_scalar_respect_ieee_special_values() {
        let pos_inf = ScalarTensor::new(f64::INFINITY, DType::F64, Device::Cpu);
        let neg_inf = ScalarTensor::new(f64::NEG_INFINITY, DType::F64, Device::Cpu);
        let nan = ScalarTensor::new(f64::NAN, DType::F64, Device::Cpu);

        assert_eq!(
            eq_scalar(&pos_inf, &pos_inf)
                .expect("eq(+inf,+inf) should succeed")
                .value(),
            1.0
        );
        assert_eq!(
            eq_scalar(&neg_inf, &neg_inf)
                .expect("eq(-inf,-inf) should succeed")
                .value(),
            1.0
        );
        assert_eq!(
            ne_scalar(&pos_inf, &pos_inf)
                .expect("ne(+inf,+inf) should succeed")
                .value(),
            0.0
        );
        assert_eq!(
            ne_scalar(&neg_inf, &neg_inf)
                .expect("ne(-inf,-inf) should succeed")
                .value(),
            0.0
        );
        assert_eq!(
            eq_scalar(&nan, &nan)
                .expect("eq(nan,nan) should succeed")
                .value(),
            0.0
        );
        assert_eq!(
            ne_scalar(&nan, &nan)
                .expect("ne(nan,nan) should succeed")
                .value(),
            1.0
        );
        assert_eq!(
            eq_scalar(&pos_inf, &neg_inf)
                .expect("eq(+inf,-inf) should succeed")
                .value(),
            0.0
        );
        assert_eq!(
            ne_scalar(&pos_inf, &neg_inf)
                .expect("ne(+inf,-inf) should succeed")
                .value(),
            1.0
        );
    }

    #[test]
    fn lt_gt_scalar_returns_expected_values() {
        let a = ScalarTensor::new(2.0, DType::F64, Device::Cpu);
        let b = ScalarTensor::new(3.0, DType::F64, Device::Cpu);
        assert_eq!(lt_scalar(&a, &b).expect("lt should succeed").value(), 1.0);
        assert_eq!(lt_scalar(&b, &a).expect("lt should succeed").value(), 0.0);
        assert_eq!(gt_scalar(&b, &a).expect("gt should succeed").value(), 1.0);
        assert_eq!(gt_scalar(&a, &b).expect("gt should succeed").value(), 0.0);
    }

    #[test]
    fn le_ge_scalar_returns_expected_values() {
        let a = ScalarTensor::new(2.0, DType::F64, Device::Cpu);
        let b = ScalarTensor::new(2.0, DType::F64, Device::Cpu);
        let c = ScalarTensor::new(3.0, DType::F64, Device::Cpu);
        assert_eq!(le_scalar(&a, &b).expect("le should succeed").value(), 1.0);
        assert_eq!(le_scalar(&a, &c).expect("le should succeed").value(), 1.0);
        assert_eq!(le_scalar(&c, &a).expect("le should succeed").value(), 0.0);
        assert_eq!(ge_scalar(&a, &b).expect("ge should succeed").value(), 1.0);
        assert_eq!(ge_scalar(&c, &a).expect("ge should succeed").value(), 1.0);
        assert_eq!(ge_scalar(&a, &c).expect("ge should succeed").value(), 0.0);
    }

    #[test]
    fn eq_tensor_contiguous_returns_expected_values() {
        let meta = TensorMeta::from_shape(vec![4], DType::F64, Device::Cpu);
        let lhs = vec![1.0, 2.0, 3.0, 4.0];
        let rhs = vec![1.0, 5.0, 3.0, 0.0];
        let out = eq_tensor_contiguous_f64(&lhs, &rhs, &meta, &meta).expect("eq should succeed");
        assert_eq!(out, vec![1.0, 0.0, 1.0, 0.0]);
    }

    #[test]
    fn ne_tensor_contiguous_returns_expected_values() {
        let meta = TensorMeta::from_shape(vec![4], DType::F64, Device::Cpu);
        let lhs = vec![1.0, 2.0, 3.0, 4.0];
        let rhs = vec![1.0, 5.0, 3.0, 0.0];
        let out = ne_tensor_contiguous_f64(&lhs, &rhs, &meta, &meta).expect("ne should succeed");
        assert_eq!(out, vec![0.0, 1.0, 0.0, 1.0]);
    }

    #[test]
    fn eq_ne_tensor_contiguous_respect_ieee_special_values() {
        let meta = TensorMeta::from_shape(vec![4], DType::F64, Device::Cpu);
        let lhs = vec![f64::INFINITY, f64::NEG_INFINITY, f64::NAN, 1.0];
        let rhs = vec![f64::INFINITY, f64::NEG_INFINITY, f64::NAN, 2.0];

        let eq_out =
            eq_tensor_contiguous_f64(&lhs, &rhs, &meta, &meta).expect("eq tensor should succeed");
        let ne_out =
            ne_tensor_contiguous_f64(&lhs, &rhs, &meta, &meta).expect("ne tensor should succeed");

        assert_eq!(eq_out, vec![1.0, 1.0, 0.0, 0.0]);
        assert_eq!(ne_out, vec![0.0, 0.0, 1.0, 1.0]);
    }

    #[test]
    fn lt_tensor_contiguous_returns_expected_values() {
        let meta = TensorMeta::from_shape(vec![3], DType::F64, Device::Cpu);
        let lhs = vec![1.0, 3.0, 2.0];
        let rhs = vec![2.0, 3.0, 1.0];
        let out = lt_tensor_contiguous_f64(&lhs, &rhs, &meta, &meta).expect("lt should succeed");
        assert_eq!(out, vec![1.0, 0.0, 0.0]);
    }

    #[test]
    fn gt_tensor_contiguous_returns_expected_values() {
        let meta = TensorMeta::from_shape(vec![3], DType::F64, Device::Cpu);
        let lhs = vec![1.0, 3.0, 2.0];
        let rhs = vec![2.0, 3.0, 1.0];
        let out = gt_tensor_contiguous_f64(&lhs, &rhs, &meta, &meta).expect("gt should succeed");
        assert_eq!(out, vec![0.0, 0.0, 1.0]);
    }

    #[test]
    fn le_tensor_contiguous_returns_expected_values() {
        let meta = TensorMeta::from_shape(vec![3], DType::F64, Device::Cpu);
        let lhs = vec![1.0, 3.0, 2.0];
        let rhs = vec![2.0, 3.0, 1.0];
        let out = le_tensor_contiguous_f64(&lhs, &rhs, &meta, &meta).expect("le should succeed");
        assert_eq!(out, vec![1.0, 1.0, 0.0]);
    }

    #[test]
    fn ge_tensor_contiguous_returns_expected_values() {
        let meta = TensorMeta::from_shape(vec![3], DType::F64, Device::Cpu);
        let lhs = vec![1.0, 3.0, 2.0];
        let rhs = vec![2.0, 3.0, 1.0];
        let out = ge_tensor_contiguous_f64(&lhs, &rhs, &meta, &meta).expect("ge should succeed");
        assert_eq!(out, vec![0.0, 1.0, 1.0]);
    }

    #[test]
    fn sqrt_scalar_returns_expected_value() {
        let input = ScalarTensor::new(9.0, DType::F64, Device::Cpu);
        let out = sqrt_scalar(&input);
        assert_eq!(out.value(), 3.0);
    }

    #[test]
    fn sqrt_scalar_of_zero() {
        let input = ScalarTensor::new(0.0, DType::F64, Device::Cpu);
        let out = sqrt_scalar(&input);
        assert_eq!(out.value(), 0.0);
    }

    #[test]
    fn sqrt_scalar_of_one() {
        let input = ScalarTensor::new(1.0, DType::F64, Device::Cpu);
        let out = sqrt_scalar(&input);
        assert_eq!(out.value(), 1.0);
    }

    #[test]
    fn sqrt_tensor_contiguous_returns_expected_values() {
        let meta = TensorMeta::from_shape(vec![4], DType::F64, Device::Cpu);
        let input = vec![0.0, 1.0, 4.0, 9.0];
        let out =
            sqrt_tensor_contiguous_f64(&input, &meta).expect("contiguous sqrt should succeed");
        assert_eq!(out, vec![0.0, 1.0, 2.0, 3.0]);
    }

    #[test]
    fn reciprocal_scalar_returns_expected_value() {
        let input = ScalarTensor::new(4.0, DType::F64, Device::Cpu);
        let out = reciprocal_scalar(&input);
        assert_eq!(out.value(), 0.25);
    }

    #[test]
    fn reciprocal_scalar_of_one() {
        let input = ScalarTensor::new(1.0, DType::F64, Device::Cpu);
        let out = reciprocal_scalar(&input);
        assert_eq!(out.value(), 1.0);
    }

    #[test]
    fn reciprocal_scalar_negative() {
        let input = ScalarTensor::new(-2.0, DType::F64, Device::Cpu);
        let out = reciprocal_scalar(&input);
        assert_eq!(out.value(), -0.5);
    }

    #[test]
    fn reciprocal_tensor_contiguous_returns_expected_values() {
        let meta = TensorMeta::from_shape(vec![3], DType::F64, Device::Cpu);
        let input = vec![1.0, 2.0, 4.0];
        let out = reciprocal_tensor_contiguous_f64(&input, &meta)
            .expect("contiguous reciprocal should succeed");
        assert_eq!(out, vec![1.0, 0.5, 0.25]);
    }

    #[test]
    fn pow_scalar_returns_expected_value() {
        let input = ScalarTensor::new(3.0, DType::F64, Device::Cpu);
        let out = pow_scalar(&input, 2.0);
        assert_eq!(out.value(), 9.0);
    }

    #[test]
    fn pow_scalar_zero_exponent() {
        let input = ScalarTensor::new(5.0, DType::F64, Device::Cpu);
        let out = pow_scalar(&input, 0.0);
        assert_eq!(out.value(), 1.0);
    }

    #[test]
    fn pow_scalar_fractional_exponent() {
        let input = ScalarTensor::new(4.0, DType::F64, Device::Cpu);
        let out = pow_scalar(&input, 0.5);
        assert!((out.value() - 2.0).abs() < 1e-10);
    }

    #[test]
    fn pow_tensor_contiguous_returns_expected_values() {
        let meta = TensorMeta::from_shape(vec![3], DType::F64, Device::Cpu);
        let input = vec![1.0, 2.0, 3.0];
        let out =
            pow_tensor_contiguous_f64(&input, &meta, 2.0).expect("contiguous pow should succeed");
        assert_eq!(out, vec![1.0, 4.0, 9.0]);
    }

    #[test]
    fn pow_tensor_contiguous_negative_exponent() {
        let meta = TensorMeta::from_shape(vec![2], DType::F64, Device::Cpu);
        let input = vec![2.0, 4.0];
        let out = pow_tensor_contiguous_f64(&input, &meta, -1.0)
            .expect("pow with negative exponent should succeed");
        assert_eq!(out, vec![0.5, 0.25]);
    }

    #[test]
    fn clamp_scalar_within_range() {
        let input = ScalarTensor::new(3.0, DType::F64, Device::Cpu);
        let out = clamp_scalar(&input, 1.0, 5.0);
        assert_eq!(out.value(), 3.0);
    }

    #[test]
    fn clamp_scalar_below_min() {
        let input = ScalarTensor::new(-1.0, DType::F64, Device::Cpu);
        let out = clamp_scalar(&input, 0.0, 5.0);
        assert_eq!(out.value(), 0.0);
    }

    #[test]
    fn clamp_scalar_above_max() {
        let input = ScalarTensor::new(10.0, DType::F64, Device::Cpu);
        let out = clamp_scalar(&input, 0.0, 5.0);
        assert_eq!(out.value(), 5.0);
    }

    #[test]
    fn clamp_tensor_contiguous_returns_expected_values() {
        let meta = TensorMeta::from_shape(vec![5], DType::F64, Device::Cpu);
        let input = vec![-2.0, 0.0, 3.0, 5.0, 8.0];
        let out = clamp_tensor_contiguous_f64(&input, &meta, 0.0, 5.0)
            .expect("contiguous clamp should succeed");
        assert_eq!(out, vec![0.0, 0.0, 3.0, 5.0, 5.0]);
    }

    #[test]
    fn min_scalar_returns_expected_value() {
        let a = ScalarTensor::new(3.0, DType::F64, Device::Cpu);
        let b = ScalarTensor::new(5.0, DType::F64, Device::Cpu);
        let out = min_scalar(&a, &b).expect("min should succeed");
        assert_eq!(out.value(), 3.0);
    }

    #[test]
    fn max_scalar_returns_expected_value() {
        let a = ScalarTensor::new(3.0, DType::F64, Device::Cpu);
        let b = ScalarTensor::new(5.0, DType::F64, Device::Cpu);
        let out = max_scalar(&a, &b).expect("max should succeed");
        assert_eq!(out.value(), 5.0);
    }

    #[test]
    fn min_tensor_contiguous_returns_expected_values() {
        let meta = TensorMeta::from_shape(vec![3], DType::F64, Device::Cpu);
        let lhs = vec![1.0, 5.0, 3.0];
        let rhs = vec![2.0, 4.0, 3.0];
        let out = min_tensor_contiguous_f64(&lhs, &rhs, &meta, &meta).expect("min should succeed");
        assert_eq!(out, vec![1.0, 4.0, 3.0]);
    }

    #[test]
    fn max_tensor_contiguous_returns_expected_values() {
        let meta = TensorMeta::from_shape(vec![3], DType::F64, Device::Cpu);
        let lhs = vec![1.0, 5.0, 3.0];
        let rhs = vec![2.0, 4.0, 3.0];
        let out = max_tensor_contiguous_f64(&lhs, &rhs, &meta, &meta).expect("max should succeed");
        assert_eq!(out, vec![2.0, 5.0, 3.0]);
    }

    #[test]
    fn sum_dim_reduces_along_dim0() {
        // shape [2, 3]: [[1, 2, 3], [4, 5, 6]]
        let meta = TensorMeta::from_shape(vec![2, 3], DType::F64, Device::Cpu);
        let input = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let out = sum_dim_tensor_contiguous_f64(&input, &meta, 0).expect("sum_dim 0");
        // reduce rows: [1+4, 2+5, 3+6] = [5, 7, 9]
        assert_eq!(out, vec![5.0, 7.0, 9.0]);
    }

    #[test]
    fn sum_dim_reduces_along_dim1() {
        // shape [2, 3]: [[1, 2, 3], [4, 5, 6]]
        let meta = TensorMeta::from_shape(vec![2, 3], DType::F64, Device::Cpu);
        let input = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let out = sum_dim_tensor_contiguous_f64(&input, &meta, 1).expect("sum_dim 1");
        // reduce cols: [1+2+3, 4+5+6] = [6, 15]
        assert_eq!(out, vec![6.0, 15.0]);
    }

    #[test]
    fn sum_dim_3d_reduces_middle_dim() {
        // shape [2, 3, 2]: 12 elements
        let meta = TensorMeta::from_shape(vec![2, 3, 2], DType::F64, Device::Cpu);
        let input = vec![
            1.0, 2.0, 3.0, 4.0, 5.0, 6.0, // first [3,2] slice
            7.0, 8.0, 9.0, 10.0, 11.0, 12.0, // second [3,2] slice
        ];
        let out = sum_dim_tensor_contiguous_f64(&input, &meta, 1).expect("sum_dim 1 on 3d");
        // Output shape [2, 2]: reduce along dim 1 (size 3)
        // [0,0]: 1+3+5=9, [0,1]: 2+4+6=12, [1,0]: 7+9+11=27, [1,1]: 8+10+12=30
        assert_eq!(out, vec![9.0, 12.0, 27.0, 30.0]);
    }

    #[test]
    fn sum_dim_invalid_dim_returns_error() {
        let meta = TensorMeta::from_shape(vec![2, 3], DType::F64, Device::Cpu);
        let input = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let err = sum_dim_tensor_contiguous_f64(&input, &meta, 2).unwrap_err();
        assert!(matches!(
            err,
            KernelError::InvalidDimension { dim: 2, ndim: 2 }
        ));
    }

    #[test]
    fn mean_dim_reduces_along_dim0() {
        let meta = TensorMeta::from_shape(vec![2, 3], DType::F64, Device::Cpu);
        let input = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let out = mean_dim_tensor_contiguous_f64(&input, &meta, 0).expect("mean_dim 0");
        // [2.5, 3.5, 4.5]
        assert_eq!(out, vec![2.5, 3.5, 4.5]);
    }

    #[test]
    fn mean_dim_reduces_along_dim1() {
        let meta = TensorMeta::from_shape(vec![2, 3], DType::F64, Device::Cpu);
        let input = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let out = mean_dim_tensor_contiguous_f64(&input, &meta, 1).expect("mean_dim 1");
        // [6/3, 15/3] = [2.0, 5.0]
        assert_eq!(out, vec![2.0, 5.0]);
    }

    #[test]
    fn sum_dim_1d_reduces_to_scalar() {
        let meta = TensorMeta::from_shape(vec![4], DType::F64, Device::Cpu);
        let input = vec![1.0, 2.0, 3.0, 4.0];
        let out = sum_dim_tensor_contiguous_f64(&input, &meta, 0).expect("sum_dim on 1d");
        // Reduces [4] along dim 0 -> scalar-like vec of len 1? No, outer=1, inner=1 -> output len 1
        assert_eq!(out, vec![10.0]);
    }

    #[test]
    fn prod_dim_reduces_along_dim0() {
        let meta = TensorMeta::from_shape(vec![2, 3], DType::F64, Device::Cpu);
        let input = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let out = prod_dim_tensor_contiguous_f64(&input, &meta, 0).expect("prod_dim 0");
        // [1*4, 2*5, 3*6] = [4, 10, 18]
        assert_eq!(out, vec![4.0, 10.0, 18.0]);
    }

    #[test]
    fn prod_dim_reduces_along_dim1() {
        let meta = TensorMeta::from_shape(vec![2, 3], DType::F64, Device::Cpu);
        let input = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let out = prod_dim_tensor_contiguous_f64(&input, &meta, 1).expect("prod_dim 1");
        // [1*2*3, 4*5*6] = [6, 120]
        assert_eq!(out, vec![6.0, 120.0]);
    }

    #[test]
    fn prod_dim_invalid_dim_returns_error() {
        let meta = TensorMeta::from_shape(vec![2, 3], DType::F64, Device::Cpu);
        let input = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let err = prod_dim_tensor_contiguous_f64(&input, &meta, 2).unwrap_err();
        assert!(matches!(
            err,
            KernelError::InvalidDimension { dim: 2, ndim: 2 }
        ));
    }

    #[test]
    fn var_dim_reduces_along_dim0() {
        let meta = TensorMeta::from_shape(vec![3, 2], DType::F64, Device::Cpu);
        // col0: [1,3,5] mean=3, var=((1-3)^2+(3-3)^2+(5-3)^2)/2 = (4+0+4)/2 = 4
        // col1: [2,4,6] mean=4, var=((2-4)^2+(4-4)^2+(6-4)^2)/2 = (4+0+4)/2 = 4
        let input = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let out = var_dim_tensor_contiguous_f64(&input, &meta, 0).expect("var_dim 0");
        assert!((out[0] - 4.0).abs() < 1e-12);
        assert!((out[1] - 4.0).abs() < 1e-12);
    }

    #[test]
    fn var_dim_reduces_along_dim1() {
        let meta = TensorMeta::from_shape(vec![2, 3], DType::F64, Device::Cpu);
        // row0: [1,2,3] mean=2, var=((1-2)^2+(2-2)^2+(3-2)^2)/2 = (1+0+1)/2 = 1
        // row1: [4,5,6] mean=5, var=((4-5)^2+(5-5)^2+(6-5)^2)/2 = (1+0+1)/2 = 1
        let input = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let out = var_dim_tensor_contiguous_f64(&input, &meta, 1).expect("var_dim 1");
        assert!((out[0] - 1.0).abs() < 1e-12);
        assert!((out[1] - 1.0).abs() < 1e-12);
    }

    #[test]
    fn std_dim_reduces_along_dim0() {
        let meta = TensorMeta::from_shape(vec![3, 2], DType::F64, Device::Cpu);
        let input = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let out = std_dim_tensor_contiguous_f64(&input, &meta, 0).expect("std_dim 0");
        assert!((out[0] - 2.0).abs() < 1e-12); // sqrt(4) = 2
        assert!((out[1] - 2.0).abs() < 1e-12);
    }

    #[test]
    fn softmax_dim_sums_to_one() {
        let meta = TensorMeta::from_shape(vec![2, 3], DType::F64, Device::Cpu);
        let input = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let out = softmax_dim_tensor_contiguous_f64(&input, &meta, 1).expect("softmax");
        // Each row should sum to 1
        let row0_sum: f64 = out[0..3].iter().sum();
        let row1_sum: f64 = out[3..6].iter().sum();
        assert!((row0_sum - 1.0).abs() < 1e-12);
        assert!((row1_sum - 1.0).abs() < 1e-12);
        // All values should be positive
        assert!(out.iter().all(|&v| v > 0.0));
    }

    #[test]
    fn softmax_dim_preserves_relative_order() {
        let meta = TensorMeta::from_shape(vec![1, 3], DType::F64, Device::Cpu);
        let input = vec![1.0, 3.0, 2.0];
        let out = softmax_dim_tensor_contiguous_f64(&input, &meta, 1).expect("softmax");
        // input[1] > input[2] > input[0], so output should preserve order
        assert!(out[1] > out[2]);
        assert!(out[2] > out[0]);
    }

    #[test]
    fn log_softmax_dim_values() {
        let meta = TensorMeta::from_shape(vec![1, 3], DType::F64, Device::Cpu);
        let input = vec![1.0, 2.0, 3.0];
        let out = log_softmax_dim_tensor_contiguous_f64(&input, &meta, 1).expect("log_softmax");
        let sm = softmax_dim_tensor_contiguous_f64(&input, &meta, 1).expect("softmax");
        // log_softmax should equal log(softmax)
        for i in 0..3 {
            assert!((out[i] - sm[i].ln()).abs() < 1e-12);
        }
    }

    #[test]
    fn softmax_dim_invalid_dim() {
        let meta = TensorMeta::from_shape(vec![2, 3], DType::F64, Device::Cpu);
        let input = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let err = softmax_dim_tensor_contiguous_f64(&input, &meta, 2).unwrap_err();
        assert!(matches!(
            err,
            KernelError::InvalidDimension { dim: 2, ndim: 2 }
        ));
    }

    #[test]
    fn cat_along_dim0() {
        let m0 = TensorMeta::from_shape(vec![2, 3], DType::F64, Device::Cpu);
        let d0 = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let m1 = TensorMeta::from_shape(vec![1, 3], DType::F64, Device::Cpu);
        let d1 = vec![7.0, 8.0, 9.0];
        let out = cat_tensor_contiguous_f64(&[(&d0, &m0), (&d1, &m1)], 0).expect("cat dim 0");
        assert_eq!(out, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]);
    }

    #[test]
    fn cat_along_dim1() {
        let m0 = TensorMeta::from_shape(vec![2, 2], DType::F64, Device::Cpu);
        let d0 = vec![1.0, 2.0, 3.0, 4.0];
        let m1 = TensorMeta::from_shape(vec![2, 3], DType::F64, Device::Cpu);
        let d1 = vec![5.0, 6.0, 7.0, 8.0, 9.0, 10.0];
        let out = cat_tensor_contiguous_f64(&[(&d0, &m0), (&d1, &m1)], 1).expect("cat dim 1");
        // row0: [1,2,5,6,7], row1: [3,4,8,9,10]
        assert_eq!(out, vec![1.0, 2.0, 5.0, 6.0, 7.0, 3.0, 4.0, 8.0, 9.0, 10.0]);
    }

    #[test]
    fn stack_along_dim0() {
        let m0 = TensorMeta::from_shape(vec![3], DType::F64, Device::Cpu);
        let d0 = vec![1.0, 2.0, 3.0];
        let m1 = TensorMeta::from_shape(vec![3], DType::F64, Device::Cpu);
        let d1 = vec![4.0, 5.0, 6.0];
        let out = stack_tensor_contiguous_f64(&[(&d0, &m0), (&d1, &m1)], 0).expect("stack dim 0");
        // shape [2,3]: [[1,2,3],[4,5,6]]
        assert_eq!(out, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    }

    #[test]
    fn stack_along_dim1() {
        let m0 = TensorMeta::from_shape(vec![2, 2], DType::F64, Device::Cpu);
        let d0 = vec![1.0, 2.0, 3.0, 4.0];
        let m1 = TensorMeta::from_shape(vec![2, 2], DType::F64, Device::Cpu);
        let d1 = vec![5.0, 6.0, 7.0, 8.0];
        let out = stack_tensor_contiguous_f64(&[(&d0, &m0), (&d1, &m1)], 1).expect("stack dim 1");
        // shape [2,2,2]: [[[1,2],[5,6]],[[3,4],[7,8]]]
        assert_eq!(out, vec![1.0, 2.0, 5.0, 6.0, 3.0, 4.0, 7.0, 8.0]);
    }
}

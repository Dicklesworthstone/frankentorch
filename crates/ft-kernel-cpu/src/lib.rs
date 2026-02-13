#![forbid(unsafe_code)]

use std::fmt;

use ft_core::{ScalarTensor, TensorCompatError, ensure_compatible};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum KernelError {
    Incompatible(TensorCompatError),
}

impl fmt::Display for KernelError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Incompatible(error) => write!(f, "incompatible tensors: {error}"),
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

pub fn mul_scalar(lhs: &ScalarTensor, rhs: &ScalarTensor) -> Result<ScalarTensor, KernelError> {
    ensure_compatible(lhs, rhs)?;
    Ok(lhs.with_value(lhs.value() * rhs.value()))
}

#[cfg(test)]
mod tests {
    use ft_core::{DType, Device, ScalarTensor};

    use super::{add_scalar, mul_scalar};

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
}

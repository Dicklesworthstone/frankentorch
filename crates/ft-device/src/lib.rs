#![forbid(unsafe_code)]

use std::fmt;

use ft_core::{Device, ScalarTensor};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DeviceError {
    Mismatch { expected: Device, actual: Device },
}

impl fmt::Display for DeviceError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Mismatch { expected, actual } => {
                write!(f, "device mismatch: expected {expected:?}, got {actual:?}")
            }
        }
    }
}

impl std::error::Error for DeviceError {}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DeviceGuard {
    device: Device,
}

impl DeviceGuard {
    #[must_use]
    pub fn new(device: Device) -> Self {
        Self { device }
    }

    #[must_use]
    pub fn device(&self) -> Device {
        self.device
    }

    pub fn ensure_tensor_device(&self, tensor: &ScalarTensor) -> Result<(), DeviceError> {
        let actual = tensor.meta().device();
        if actual != self.device {
            return Err(DeviceError::Mismatch {
                expected: self.device,
                actual,
            });
        }
        Ok(())
    }
}

pub fn ensure_same_device(lhs: &ScalarTensor, rhs: &ScalarTensor) -> Result<Device, DeviceError> {
    let lhs_device = lhs.meta().device();
    let rhs_device = rhs.meta().device();
    if lhs_device != rhs_device {
        return Err(DeviceError::Mismatch {
            expected: lhs_device,
            actual: rhs_device,
        });
    }
    Ok(lhs_device)
}

#[cfg(test)]
mod tests {
    use ft_core::{DType, Device, ScalarTensor};

    use super::{DeviceGuard, ensure_same_device};

    #[test]
    fn guard_accepts_matching_device() {
        let tensor = ScalarTensor::new(1.0, DType::F64, Device::Cpu);
        let guard = DeviceGuard::new(Device::Cpu);
        assert!(guard.ensure_tensor_device(&tensor).is_ok());
    }

    #[test]
    fn same_device_check_returns_cpu() {
        let lhs = ScalarTensor::new(1.0, DType::F64, Device::Cpu);
        let rhs = ScalarTensor::new(2.0, DType::F64, Device::Cpu);
        let device = ensure_same_device(&lhs, &rhs).expect("devices should match");
        assert_eq!(device, Device::Cpu);
    }
}

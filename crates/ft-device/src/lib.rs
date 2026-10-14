#![forbid(unsafe_code)]

use std::cell::RefCell;
use std::fmt;

use ft_core::{DenseTensor, Device, ScalarTensor, TensorMeta};

thread_local! {
    static ACTIVE_DEVICE_STACK: RefCell<Vec<Device>> = const { RefCell::new(Vec::new()) };
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DeviceError {
    Mismatch { expected: Device, actual: Device },
    EmptyDeviceCheck,
}

impl fmt::Display for DeviceError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Mismatch { expected, actual } => {
                f.write_str(device_mismatch_message(*expected, *actual))
            }
            Self::EmptyDeviceCheck => {
                f.write_str("device check requires at least one tensor metadata entry")
            }
        }
    }
}

impl std::error::Error for DeviceError {}

fn device_mismatch_message(expected: Device, actual: Device) -> &'static str {
    match (expected, actual) {
        (Device::Cpu, Device::Cpu) => "device mismatch: expected Cpu, got Cpu",
        (Device::Cpu, Device::Cuda) => "device mismatch: expected Cpu, got Cuda",
        (Device::Cpu, Device::Metal) => "device mismatch: expected Cpu, got Metal",
        (Device::Cuda, Device::Cpu) => "device mismatch: expected Cuda, got Cpu",
        (Device::Cuda, Device::Cuda) => "device mismatch: expected Cuda, got Cuda",
        (Device::Cuda, Device::Metal) => "device mismatch: expected Cuda, got Metal",
        (Device::Metal, Device::Cpu) => "device mismatch: expected Metal, got Cpu",
        (Device::Metal, Device::Cuda) => "device mismatch: expected Metal, got Cuda",
        (Device::Metal, Device::Metal) => "device mismatch: expected Metal, got Metal",
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DeviceGuard {
    device: Device,
}

impl Default for DeviceGuard {
    fn default() -> Self {
        Self::new(current_device())
    }
}

impl DeviceGuard {
    #[must_use]
    pub const fn new(device: Device) -> Self {
        Self { device }
    }

    #[must_use]
    pub const fn device(&self) -> Device {
        self.device
    }

    pub fn ensure_meta_device(&self, meta: &TensorMeta) -> Result<(), DeviceError> {
        let actual = meta.device();
        if actual != self.device {
            return Err(DeviceError::Mismatch {
                expected: self.device,
                actual,
            });
        }
        Ok(())
    }

    pub fn ensure_dense_tensor_device(&self, tensor: &DenseTensor) -> Result<(), DeviceError> {
        self.ensure_meta_device(tensor.meta())
    }

    pub fn ensure_tensor_device(&self, tensor: &ScalarTensor) -> Result<(), DeviceError> {
        self.ensure_meta_device(tensor.meta())
    }
}

/// Returns the currently active thread-local device (defaults to [`Device::Cpu`]).
#[must_use]
pub fn current_device() -> Device {
    ACTIVE_DEVICE_STACK
        .with(|stack| stack.borrow().last().copied())
        .unwrap_or(Device::Cpu)
}

/// Returns a [`DeviceGuard`] bound to [`current_device()`].
#[must_use]
pub fn active_device_guard() -> DeviceGuard {
    DeviceGuard::new(current_device())
}

struct ScopedDeviceStackGuard;

impl Drop for ScopedDeviceStackGuard {
    fn drop(&mut self) {
        ACTIVE_DEVICE_STACK.with(|stack| {
            stack.borrow_mut().pop();
        });
    }
}

/// Executes `f` with `device` pushed onto the thread-local active device guard stack.
///
/// Even if `f` panics, the guard is popped when unwinding.
pub fn with_device_guard<R>(device: Device, f: impl FnOnce(DeviceGuard) -> R) -> R {
    ACTIVE_DEVICE_STACK.with(|stack| {
        stack.borrow_mut().push(device);
    });
    let _pop_on_drop = ScopedDeviceStackGuard;
    f(DeviceGuard::new(device))
}

pub fn ensure_matches_active_device(meta: &TensorMeta) -> Result<Device, DeviceError> {
    let guard = active_device_guard();
    guard.ensure_meta_device(meta)?;
    Ok(guard.device())
}

pub fn ensure_same_meta_device(lhs: &TensorMeta, rhs: &TensorMeta) -> Result<Device, DeviceError> {
    let lhs_device = lhs.device();
    let rhs_device = rhs.device();
    if lhs_device != rhs_device {
        return Err(DeviceError::Mismatch {
            expected: lhs_device,
            actual: rhs_device,
        });
    }
    Ok(lhs_device)
}

pub fn ensure_same_dense_device(
    lhs: &DenseTensor,
    rhs: &DenseTensor,
) -> Result<Device, DeviceError> {
    ensure_same_meta_device(lhs.meta(), rhs.meta())
}

pub fn ensure_same_device_slice(metas: &[&TensorMeta]) -> Result<Device, DeviceError> {
    let Some((&first, rest)) = metas.split_first() else {
        return Err(DeviceError::EmptyDeviceCheck);
    };
    let expected = first.device();
    for &meta in rest {
        let actual = meta.device();
        if actual != expected {
            return Err(DeviceError::Mismatch { expected, actual });
        }
    }
    Ok(expected)
}

pub fn ensure_same_device(lhs: &ScalarTensor, rhs: &ScalarTensor) -> Result<Device, DeviceError> {
    ensure_same_meta_device(lhs.meta(), rhs.meta())
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeMap;

    use ft_core::{DType, Device, ScalarTensor};

    use super::{DeviceError, DeviceGuard, ensure_same_device};

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

    fn build_packet_007_log(
        test_id: &str,
        scenario_id: &str,
        mode: &str,
        seed: u64,
        input_digest: u64,
        output_digest: u64,
        reason_code: &str,
    ) -> BTreeMap<String, String> {
        let mut log = BTreeMap::new();
        log.insert("ts_utc".to_string(), "1970-01-01T00:00:00Z".to_string());
        log.insert("suite_id".to_string(), "ft_device_unit".to_string());
        log.insert("test_id".to_string(), test_id.to_string());
        log.insert("packet_id".to_string(), "FT-P2C-007".to_string());
        log.insert("fixture_id".to_string(), "ft_device_packet_007".to_string());
        log.insert("scenario_id".to_string(), scenario_id.to_string());
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
            "det64:ft-device-test".to_string(),
        );
        log.insert(
            "artifact_refs".to_string(),
            "artifacts/phase2c/FT-P2C-007/contract_table.md".to_string(),
        );
        log.insert(
            "replay_command".to_string(),
            format!("cargo test -p ft-device {test_id} -- --nocapture"),
        );
        log.insert("duration_ms".to_string(), "0".to_string());
        log.insert("outcome".to_string(), "pass".to_string());
        log.insert("reason_code".to_string(), reason_code.to_string());
        log
    }

    fn assert_packet_007_log_contract(log: &BTreeMap<String, String>) {
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
            "reason_code",
        ] {
            assert!(
                log.contains_key(key),
                "missing required packet log field '{key}'"
            );
        }
    }

    #[test]
    fn guard_accepts_matching_device() {
        let tensor = ScalarTensor::new(1.0, DType::F64, Device::Cpu);
        let guard = DeviceGuard::new(Device::Cpu);
        assert!(guard.ensure_tensor_device(&tensor).is_ok());

        let input_digest = tensor.evidence_fingerprint64();
        let output_digest = det_seed(&[input_digest, 1]);
        let seed = det_seed(&[input_digest, output_digest, 7]);
        let log = build_packet_007_log(
            "guard_accepts_matching_device",
            "dispatch_key/strict:strict_cpu_route",
            "strict",
            seed,
            input_digest,
            output_digest,
            "device_guard_match_ok",
        );
        assert_packet_007_log_contract(&log);
    }

    #[test]
    fn same_device_check_returns_cpu() {
        let lhs = ScalarTensor::new(1.0, DType::F64, Device::Cpu);
        let rhs = ScalarTensor::new(2.0, DType::F64, Device::Cpu);
        let device = ensure_same_device(&lhs, &rhs).expect("devices should match");
        assert_eq!(device, Device::Cpu);

        let input_digest = lhs.evidence_fingerprint64() ^ rhs.evidence_fingerprint64();
        let output_digest = det_seed(&[input_digest, 2]);
        let seed = det_seed(&[input_digest, output_digest, 11]);
        let log = build_packet_007_log(
            "same_device_check_returns_cpu",
            "dispatch_key/strict:strict_cpu_route",
            "strict",
            seed,
            input_digest,
            output_digest,
            "same_device_contract_ok",
        );
        assert_packet_007_log_contract(&log);
    }

    #[test]
    fn guard_rejects_mismatched_device() {
        let tensor = ScalarTensor::new(1.0, DType::F64, Device::Cuda);
        let guard = DeviceGuard::new(Device::Cpu);
        let err = guard
            .ensure_tensor_device(&tensor)
            .expect_err("mismatched device should fail closed");
        assert!(matches!(
            err,
            DeviceError::Mismatch {
                expected: Device::Cpu,
                actual: Device::Cuda
            }
        ));

        let input_digest = tensor.evidence_fingerprint64();
        let output_digest = det_seed(&[input_digest, 3]);
        let seed = det_seed(&[input_digest, output_digest, 13]);
        let log = build_packet_007_log(
            "guard_rejects_mismatched_device",
            "dispatch_key/strict:device_mismatch_fail_closed",
            "strict",
            seed,
            input_digest,
            output_digest,
            "device_guard_mismatch_fail_closed",
        );
        assert_packet_007_log_contract(&log);
    }

    #[test]
    fn same_device_check_rejects_cross_device_pair() {
        let lhs = ScalarTensor::new(1.0, DType::F64, Device::Cpu);
        let rhs = ScalarTensor::new(2.0, DType::F64, Device::Cuda);
        let err = ensure_same_device(&lhs, &rhs).expect_err("cross-device pair must fail closed");
        assert!(matches!(
            err,
            DeviceError::Mismatch {
                expected: Device::Cpu,
                actual: Device::Cuda
            }
        ));

        let input_digest = lhs.evidence_fingerprint64() ^ rhs.evidence_fingerprint64();
        let output_digest = det_seed(&[input_digest, 4]);
        let seed = det_seed(&[input_digest, output_digest, 17]);
        let log = build_packet_007_log(
            "same_device_check_rejects_cross_device_pair",
            "dispatch_key/strict:device_mismatch_fail_closed",
            "strict",
            seed,
            input_digest,
            output_digest,
            "same_device_fail_closed",
        );
        assert_packet_007_log_contract(&log);
    }

    // ── bd-hcvr: DeviceGuard::device accessor and Cuda paths ──

    #[test]
    fn guard_device_accessor_returns_correct_device() {
        let cpu_guard = DeviceGuard::new(Device::Cpu);
        assert_eq!(cpu_guard.device(), Device::Cpu);

        let cuda_guard = DeviceGuard::new(Device::Cuda);
        assert_eq!(cuda_guard.device(), Device::Cuda);
    }

    #[test]
    fn cuda_guard_accepts_cuda_tensor() {
        let tensor = ScalarTensor::new(42.0, DType::F64, Device::Cuda);
        let guard = DeviceGuard::new(Device::Cuda);
        assert!(guard.ensure_tensor_device(&tensor).is_ok());
    }

    #[test]
    fn cuda_guard_rejects_cpu_tensor() {
        let tensor = ScalarTensor::new(1.0, DType::F64, Device::Cpu);
        let guard = DeviceGuard::new(Device::Cuda);
        let err = guard
            .ensure_tensor_device(&tensor)
            .expect_err("cpu tensor on cuda guard should fail");
        assert!(matches!(
            err,
            DeviceError::Mismatch {
                expected: Device::Cuda,
                actual: Device::Cpu
            }
        ));
    }

    #[test]
    fn same_device_check_cuda_pair() {
        let lhs = ScalarTensor::new(1.0, DType::F64, Device::Cuda);
        let rhs = ScalarTensor::new(2.0, DType::F64, Device::Cuda);
        let device = ensure_same_device(&lhs, &rhs).expect("cuda pair should match");
        assert_eq!(device, Device::Cuda);
    }

    #[test]
    fn device_error_display() {
        fn assert_mismatch_display(expected: Device, actual: Device, message: &str) {
            let err = DeviceError::Mismatch { expected, actual };
            assert_eq!(err.to_string(), message);
        }

        assert_mismatch_display(
            Device::Cpu,
            Device::Cpu,
            "device mismatch: expected Cpu, got Cpu",
        );
        assert_mismatch_display(
            Device::Cpu,
            Device::Cuda,
            "device mismatch: expected Cpu, got Cuda",
        );
        assert_mismatch_display(
            Device::Cuda,
            Device::Cpu,
            "device mismatch: expected Cuda, got Cpu",
        );
        assert_mismatch_display(
            Device::Cuda,
            Device::Cuda,
            "device mismatch: expected Cuda, got Cuda",
        );
    }

    #[test]
    fn device_contract_is_exhaustive_over_devices_and_dtypes() {
        let devices = [Device::Cpu, Device::Cuda];
        let dtypes = [
            DType::F64,
            DType::F32,
            DType::F16,
            DType::BF16,
            DType::QInt8,
            DType::QUInt8,
            DType::I64,
            DType::I32,
            DType::Bool,
            DType::Complex64,
            DType::Complex128,
        ];

        for &dtype in &dtypes {
            for &expected in &devices {
                let guard = DeviceGuard::new(expected);
                for &actual in &devices {
                    let tensor = ScalarTensor::new(1.0, dtype, actual);
                    let result = guard.ensure_tensor_device(&tensor);
                    if expected == actual {
                        assert_eq!(result, Ok(()), "guard={expected:?} dtype={dtype:?}");
                    } else {
                        let expected_error = DeviceError::Mismatch { expected, actual };
                        assert_eq!(
                            result,
                            Err(expected_error),
                            "guard={expected:?} tensor={actual:?} dtype={dtype:?}"
                        );
                    }
                }
            }

            for &lhs_device in &devices {
                for &rhs_device in &devices {
                    let lhs = ScalarTensor::new(1.0, dtype, lhs_device);
                    let rhs = ScalarTensor::new(2.0, dtype, rhs_device);
                    let result = ensure_same_device(&lhs, &rhs);
                    if lhs_device == rhs_device {
                        assert_eq!(
                            result,
                            Ok(lhs_device),
                            "lhs={lhs_device:?} rhs={rhs_device:?} dtype={dtype:?}"
                        );
                    } else {
                        let expected_error = DeviceError::Mismatch {
                            expected: lhs_device,
                            actual: rhs_device,
                        };
                        assert_eq!(
                            result,
                            Err(expected_error),
                            "lhs={lhs_device:?} rhs={rhs_device:?} dtype={dtype:?}"
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn device_guard_golden_summary_matches_fixture() {
        let cpu_guard = DeviceGuard::new(Device::Cpu);
        let cuda_guard = DeviceGuard::new(Device::Cuda);
        let cpu_tensor = ScalarTensor::new(1.0, DType::F64, Device::Cpu);
        let cuda_tensor = ScalarTensor::new(2.0, DType::F64, Device::Cuda);

        let accept_cpu = cpu_guard
            .ensure_tensor_device(&cpu_tensor)
            .map(|()| "ok".to_string())
            .unwrap_or_else(|err| err.to_string());
        let reject_cuda = cpu_guard
            .ensure_tensor_device(&cuda_tensor)
            .map(|()| "ok".to_string())
            .unwrap_or_else(|err| err.to_string());
        let same_cpu = ensure_same_device(&cpu_tensor, &cpu_tensor)
            .map(|device| format!("{device:?}"))
            .unwrap_or_else(|err| err.to_string());
        let same_cross = ensure_same_device(&cpu_tensor, &cuda_tensor)
            .map(|device| format!("{device:?}"))
            .unwrap_or_else(|err| err.to_string());

        let summary = format!(
            "ft_device_guard_pass23\n\
             cpu_guard_device={:?}\n\
             cuda_guard_device={:?}\n\
             cpu_guard_accept_cpu={accept_cpu}\n\
             cpu_guard_reject_cuda={reject_cuda}\n\
             same_cpu_pair={same_cpu}\n\
             same_cross_pair={same_cross}\n",
            cpu_guard.device(),
            cuda_guard.device()
        );
        assert_eq!(
            summary,
            include_str!(
                "../../../artifacts/optimization/golden_outputs/ft_device_guard_pass23.txt"
            )
        );
    }

    #[test]
    fn dense_tensor_and_metal_device_guard_and_scoped_stack_work() {
        use super::{
            active_device_guard, current_device, ensure_matches_active_device,
            ensure_same_dense_device, ensure_same_device_slice, ensure_same_meta_device,
            with_device_guard,
        };
        use ft_core::{DenseTensor, TensorMeta};

        assert_eq!(current_device(), Device::Cpu);
        assert_eq!(Device::parse("metal"), Some(Device::Metal));
        assert_eq!(Device::parse("MPS:0"), Some(Device::Metal));
        assert_eq!(Device::Metal.to_string(), "metal");
        assert!(Device::Metal.is_metal());
        assert!(Device::Metal.is_accelerator());

        let cpu_dense = DenseTensor::from_contiguous_values(vec![1.0, 2.0], vec![2], Device::Cpu)
            .expect("cpu dense tensor");
        let metal_dense =
            DenseTensor::from_contiguous_values(vec![3.0, 4.0], vec![2], Device::Metal)
                .expect("metal dense tensor");

        let metal_guard = DeviceGuard::new(Device::Metal);
        assert_eq!(metal_guard.ensure_dense_tensor_device(&metal_dense), Ok(()));
        assert_eq!(
            metal_guard.ensure_dense_tensor_device(&cpu_dense),
            Err(DeviceError::Mismatch {
                expected: Device::Metal,
                actual: Device::Cpu,
            })
        );
        assert_eq!(
            ensure_same_dense_device(&metal_dense, &metal_dense),
            Ok(Device::Metal)
        );
        assert_eq!(
            ensure_same_dense_device(&cpu_dense, &metal_dense),
            Err(DeviceError::Mismatch {
                expected: Device::Cpu,
                actual: Device::Metal,
            })
        );
        assert_eq!(
            ensure_same_meta_device(cpu_dense.meta(), metal_dense.meta()),
            Err(DeviceError::Mismatch {
                expected: Device::Cpu,
                actual: Device::Metal,
            })
        );
        assert_eq!(
            ensure_same_device_slice(&[]),
            Err(DeviceError::EmptyDeviceCheck)
        );
        let m1 = TensorMeta::from_shape(vec![2], DType::F32, Device::Metal);
        let m2 = TensorMeta::from_shape(vec![2, 2], DType::F32, Device::Metal);
        assert_eq!(ensure_same_device_slice(&[&m1, &m2]), Ok(Device::Metal));

        with_device_guard(Device::Metal, |guard| {
            assert_eq!(guard.device(), Device::Metal);
            assert_eq!(current_device(), Device::Metal);
            assert_eq!(active_device_guard().device(), Device::Metal);
            assert_eq!(
                ensure_matches_active_device(metal_dense.meta()),
                Ok(Device::Metal)
            );
            assert_eq!(
                ensure_matches_active_device(cpu_dense.meta()),
                Err(DeviceError::Mismatch {
                    expected: Device::Metal,
                    actual: Device::Cpu,
                })
            );

            with_device_guard(Device::Cuda, |inner| {
                assert_eq!(inner.device(), Device::Cuda);
                assert_eq!(current_device(), Device::Cuda);
            });
            assert_eq!(current_device(), Device::Metal);
        });
        assert_eq!(current_device(), Device::Cpu);
    }
}

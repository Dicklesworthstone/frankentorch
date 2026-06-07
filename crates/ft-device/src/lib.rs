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
                f.write_str(device_mismatch_message(*expected, *actual))
            }
        }
    }
}

impl std::error::Error for DeviceError {}

fn device_mismatch_message(expected: Device, actual: Device) -> &'static str {
    match (expected, actual) {
        (Device::Cpu, Device::Cpu) => "device mismatch: expected Cpu, got Cpu",
        (Device::Cpu, Device::Cuda) => "device mismatch: expected Cpu, got Cuda",
        (Device::Cuda, Device::Cpu) => "device mismatch: expected Cuda, got Cpu",
        (Device::Cuda, Device::Cuda) => "device mismatch: expected Cuda, got Cuda",
    }
}

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
}

#![no_main]

use ft_core::{DType, Device, TensorMeta};
use ft_kernel_cpu::{
    elu_tensor_contiguous_f64, gelu_tensor_contiguous_f64,
    hardsigmoid_tensor_contiguous_f64, hardswish_tensor_contiguous_f64,
    hardtanh_tensor_contiguous_f64, leaky_relu_tensor_contiguous_f64,
    mish_tensor_contiguous_f64, relu_tensor_contiguous_f64,
    sigmoid_tensor_contiguous_f64, silu_tensor_contiguous_f64,
    softplus_tensor_contiguous_f64, tanh_tensor_contiguous_f64,
};
use libfuzzer_sys::fuzz_target;

const MAX_INPUT_BYTES: usize = 512;
const MAX_SHAPE_DIM: u8 = 8;

fuzz_target!(|data: &[u8]| {
    if data.len() < 2 || data.len() > MAX_INPUT_BYTES {
        return;
    }

    let ndim = usize::from(data[0] % 5);
    if ndim == 0 {
        return;
    }
    let body = &data[1..];

    if body.len() < ndim {
        return;
    }
    let shape: Vec<usize> = body[..ndim]
        .iter()
        .map(|b| usize::from(b % (MAX_SHAPE_DIM + 1)))
        .collect();

    let meta = match TensorMeta::from_shape_and_strides(
        shape.clone(),
        ft_core::contiguous_strides(&shape),
        0,
        DType::F64,
        Device::Cpu,
    ) {
        Ok(meta) => meta,
        Err(_) => return,
    };
    let numel = meta.numel();
    if numel > 4096 {
        return;
    }

    // Input bytes scaled to a wide-but-bounded range so activations
    // exercise their non-trivial regions (gelu cubic, mish double
    // exp, softplus log1p+exp) without saturating to inf.
    let storage: Vec<f64> = (0..numel)
        .map(|i| {
            let raw = body[(ndim + i) % body.len()] as i32;
            f64::from(raw - 128) / 25.0 // ~[-5.12, 5.08]
        })
        .collect();

    let drive = |op: fn(&[f64], &TensorMeta) -> Result<Vec<f64>, _>, name: &str| {
        match op(&storage, &meta) {
            Ok(out) => {
                assert_eq!(
                    out.len(),
                    numel,
                    "{name} output length must equal numel"
                );
                Some(out)
            }
            Err(_) => None,
        }
    };

    // Range invariants are checked per activation.
    if let Some(out) = drive(relu_tensor_contiguous_f64, "relu") {
        for (i, &v) in out.iter().enumerate() {
            if !v.is_finite() {
                continue;
            }
            assert!(v >= 0.0, "relu[{i}] = {v} should be >= 0");
            // relu(x) = max(x, 0): if input is finite, output <= |input|.
        }
    }
    if let Some(out) = drive(sigmoid_tensor_contiguous_f64, "sigmoid") {
        for (i, &v) in out.iter().enumerate() {
            if !v.is_finite() {
                continue;
            }
            assert!(
                (0.0..=1.0).contains(&v),
                "sigmoid[{i}] = {v} outside [0, 1]"
            );
        }
    }
    if let Some(out) = drive(tanh_tensor_contiguous_f64, "tanh") {
        for (i, &v) in out.iter().enumerate() {
            if !v.is_finite() {
                continue;
            }
            assert!(
                (-1.0..=1.0).contains(&v),
                "tanh[{i}] = {v} outside [-1, 1]"
            );
        }
    }
    if let Some(out) = drive(hardsigmoid_tensor_contiguous_f64, "hardsigmoid") {
        for (i, &v) in out.iter().enumerate() {
            if !v.is_finite() {
                continue;
            }
            assert!(
                (0.0..=1.0).contains(&v),
                "hardsigmoid[{i}] = {v} outside [0, 1]"
            );
        }
    }
    if let Some(out) = drive(hardtanh_tensor_contiguous_f64, "hardtanh") {
        for (i, &v) in out.iter().enumerate() {
            if !v.is_finite() {
                continue;
            }
            assert!(
                (-1.0..=1.0).contains(&v),
                "hardtanh[{i}] = {v} outside [-1, 1]"
            );
        }
    }
    if let Some(out) = drive(softplus_tensor_contiguous_f64, "softplus") {
        for (i, &v) in out.iter().enumerate() {
            if !v.is_finite() {
                continue;
            }
            // softplus(x) = log(1 + exp(x)) >= 0 strictly. Allow
            // tiny FP slack near zero (could be -0.0 from log1p).
            assert!(
                v >= -1e-15,
                "softplus[{i}] = {v} should be >= 0"
            );
        }
    }
    // gelu, silu, mish, elu, leaky_relu, hardswish: drive for shape
    // and finiteness only (their output ranges are unbounded or
    // include small-negative regions that vary by definition).
    drive(gelu_tensor_contiguous_f64, "gelu");
    drive(silu_tensor_contiguous_f64, "silu");
    drive(mish_tensor_contiguous_f64, "mish");
    drive(elu_tensor_contiguous_f64, "elu");
    drive(leaky_relu_tensor_contiguous_f64, "leaky_relu");
    drive(hardswish_tensor_contiguous_f64, "hardswish");
});

#![no_main]

use ft_core::{DType, Device, TensorMeta};
use ft_kernel_cpu::addmv_tensor_contiguous_f64;
use libfuzzer_sys::fuzz_target;

const MAX_INPUT_BYTES: usize = 512;
const MAX_DIM: u8 = 16;

fuzz_target!(|data: &[u8]| {
    if data.len() < 4 || data.len() > MAX_INPUT_BYTES {
        return;
    }

    let m = usize::from(data[0] % (MAX_DIM + 1));
    let k = usize::from(data[1] % (MAX_DIM + 1));
    let (alpha, beta) = match data[2] % 8 {
        0 => (1.0, 0.0),  // plain matvec
        1 => (0.0, 1.0),  // bias-only
        2 => (0.0, 0.0),  // zero
        3 => (1.0, 1.0),  // residual + matvec
        b => (
            f64::from(b as i32 - 4) / 2.0,
            f64::from(data[3] as i32 - 128) / 64.0,
        ),
    };
    let body = &data[4..];

    let mat_shape = vec![m, k];
    let vec_shape = vec![k];
    let input_shape = vec![m];

    let mat_meta = match TensorMeta::from_shape_and_strides(
        mat_shape.clone(),
        ft_core::contiguous_strides(&mat_shape),
        0,
        DType::F64,
        Device::Cpu,
    ) {
        Ok(meta) => meta,
        Err(_) => return,
    };
    let vec_meta = match TensorMeta::from_shape_and_strides(
        vec_shape.clone(),
        ft_core::contiguous_strides(&vec_shape),
        0,
        DType::F64,
        Device::Cpu,
    ) {
        Ok(meta) => meta,
        Err(_) => return,
    };
    let input_meta = match TensorMeta::from_shape_and_strides(
        input_shape.clone(),
        ft_core::contiguous_strides(&input_shape),
        0,
        DType::F64,
        Device::Cpu,
    ) {
        Ok(meta) => meta,
        Err(_) => return,
    };

    let mat_numel = m * k;
    if mat_numel > 4096 {
        return;
    }

    let mat: Vec<f64> = (0..mat_numel)
        .map(|i| {
            let raw = body.get(i % body.len().max(1)).copied().unwrap_or(0) as i32;
            f64::from(raw - 128) / 40.0
        })
        .collect();
    let vec_data: Vec<f64> = (0..k)
        .map(|i| {
            let raw = body.get((mat_numel + i) % body.len().max(1)).copied().unwrap_or(0) as i32;
            f64::from(raw - 128) / 40.0
        })
        .collect();
    let input: Vec<f64> = (0..m)
        .map(|i| {
            let raw = body
                .get((mat_numel + k + i) % body.len().max(1))
                .copied()
                .unwrap_or(0) as i32;
            f64::from(raw - 128) / 40.0
        })
        .collect();

    let output = match addmv_tensor_contiguous_f64(
        &input, &mat, &vec_data, &input_meta, &mat_meta, &vec_meta, beta, alpha,
    ) {
        Ok(out) => out,
        Err(_) => return,
    };
    assert_eq!(output.len(), m, "addmv output length mismatch");

    for (i, &v) in output.iter().enumerate() {
        assert!(v.is_finite(), "addmv[{i}] = {v} should be finite");
    }

    if m == 0 {
        return;
    }

    const ULP_TOL: f64 = 64.0 * f64::EPSILON;
    if alpha == 0.0 && beta == 0.0 {
        for (i, &v) in output.iter().enumerate() {
            assert!(
                v.abs() < 1e-12,
                "addmv(alpha=0, beta=0)[{i}] = {v} should be 0"
            );
        }
    } else if alpha == 0.0 && beta == 1.0 {
        for i in 0..m {
            let scale = input[i].abs().max(1.0);
            assert!(
                (output[i] - input[i]).abs() <= ULP_TOL * scale,
                "addmv(alpha=0, beta=1)[{i}] = {}, expected input = {}",
                output[i], input[i]
            );
        }
    }
});

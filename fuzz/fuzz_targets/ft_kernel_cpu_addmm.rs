#![no_main]

use ft_core::{DType, Device, TensorMeta};
use ft_kernel_cpu::addmm_tensor_contiguous_f64;
use libfuzzer_sys::fuzz_target;

const MAX_INPUT_BYTES: usize = 512;
const MAX_DIM: u8 = 12;

fuzz_target!(|data: &[u8]| {
    if data.len() < 5 || data.len() > MAX_INPUT_BYTES {
        return;
    }

    let m = usize::from(data[0] % (MAX_DIM + 1));
    let k = usize::from(data[1] % (MAX_DIM + 1));
    let n = usize::from(data[2] % (MAX_DIM + 1));
    // Alpha/beta selector with high-probability boundary cases.
    let (alpha, beta) = match data[3] % 8 {
        0 => (1.0, 0.0),  // identical to plain matmul
        1 => (0.0, 1.0),  // identical to bias-only
        2 => (0.0, 0.0),  // zero output
        3 => (1.0, 1.0),  // standard residual + matmul
        b => (
            f64::from(b as i32 - 4) / 2.0,
            f64::from(data[4] as i32 - 128) / 64.0,
        ),
    };
    let input_2d = data[4] & 1 == 0;
    let body = &data[5..];

    let mat1_shape = vec![m, k];
    let mat2_shape = vec![k, n];
    let input_shape = if input_2d { vec![m, n] } else { vec![n] };

    let mat1_meta = match TensorMeta::from_shape_and_strides(
        mat1_shape.clone(),
        ft_core::contiguous_strides(&mat1_shape),
        0,
        DType::F64,
        Device::Cpu,
    ) {
        Ok(meta) => meta,
        Err(_) => return,
    };
    let mat2_meta = match TensorMeta::from_shape_and_strides(
        mat2_shape.clone(),
        ft_core::contiguous_strides(&mat2_shape),
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

    let mat1_numel = m * k;
    let mat2_numel = k * n;
    let out_numel = m * n;
    let input_numel = input_meta.numel();
    if mat1_numel > 4096 || mat2_numel > 4096 || out_numel > 4096 || input_numel > 4096 {
        return;
    }

    let mat1: Vec<f64> = (0..mat1_numel)
        .map(|i| {
            let raw = body.get(i % body.len().max(1)).copied().unwrap_or(0) as i32;
            f64::from(raw - 128) / 50.0
        })
        .collect();
    let mat2: Vec<f64> = (0..mat2_numel)
        .map(|i| {
            let raw = body.get((mat1_numel + i) % body.len().max(1)).copied().unwrap_or(0) as i32;
            f64::from(raw - 128) / 50.0
        })
        .collect();
    let input: Vec<f64> = (0..input_numel)
        .map(|i| {
            let raw = body
                .get((mat1_numel + mat2_numel + i) % body.len().max(1))
                .copied()
                .unwrap_or(0) as i32;
            f64::from(raw - 128) / 50.0
        })
        .collect();

    let output = match addmm_tensor_contiguous_f64(
        &input, &mat1, &mat2, &input_meta, &mat1_meta, &mat2_meta, beta, alpha,
    ) {
        Ok(out) => out,
        Err(_) => return,
    };
    assert_eq!(output.len(), out_numel, "addmm output length mismatch");

    for (i, &v) in output.iter().enumerate() {
        assert!(v.is_finite(), "addmm[{i}] = {v} should be finite");
    }

    if out_numel == 0 {
        return;
    }

    // Boundary identities.
    const ULP_TOL: f64 = 64.0 * f64::EPSILON; // looser — beta*input has FP noise
    if alpha == 0.0 && beta == 0.0 {
        for (i, &v) in output.iter().enumerate() {
            assert!(
                v.abs() < 1e-12,
                "addmm(alpha=0, beta=0)[{i}] = {v} should be 0"
            );
        }
    } else if alpha == 0.0 && beta == 1.0 {
        for r in 0..m {
            for c in 0..n {
                let expected = if input_2d {
                    input[r * n + c]
                } else {
                    input[c]
                };
                let got = output[r * n + c];
                let scale = expected.abs().max(1.0);
                assert!(
                    (got - expected).abs() <= ULP_TOL * scale,
                    "addmm(alpha=0, beta=1)[{r},{c}] = {got}, expected input = {expected}"
                );
            }
        }
    }
});

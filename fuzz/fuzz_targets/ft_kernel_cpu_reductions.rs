#![no_main]

use ft_core::{DType, Device, TensorMeta};
use ft_kernel_cpu::{
    max_dim_tensor_contiguous_f64, mean_dim_tensor_contiguous_f64,
    min_dim_tensor_contiguous_f64, prod_dim_tensor_contiguous_f64,
    sum_dim_tensor_contiguous_f64,
};
use libfuzzer_sys::fuzz_target;

const MAX_INPUT_BYTES: usize = 512;
const MAX_SHAPE_DIM: u8 = 8;

fuzz_target!(|data: &[u8]| {
    if data.len() < 2 || data.len() > MAX_INPUT_BYTES {
        return;
    }

    let ndim = usize::from(data[0] % 7);
    if ndim == 0 {
        return;
    }
    let dim = usize::from(data[1] % (ndim as u8).max(1)) % ndim;
    let body = &data[2..];

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

    // Bounded values so prod_dim doesn't immediately blow to inf
    // and so sum/mean stay finite.
    let storage: Vec<f64> = (0..numel)
        .map(|i| {
            let raw = body[(ndim + i) % body.len()] as i32;
            f64::from(raw - 128) / 96.0
        })
        .collect();

    // Independent expected out_numel.
    let mut out_numel: usize = 1;
    let mut zero_seen = false;
    for (d, &s) in shape.iter().enumerate() {
        if s == 0 {
            zero_seen = true;
            break;
        }
        if d != dim {
            out_numel = match out_numel.checked_mul(s) {
                Some(v) => v,
                None => return,
            };
        }
    }
    if zero_seen {
        // Some reductions return Vec::new(), some may error. Skip.
        return;
    }

    let sum_out = match sum_dim_tensor_contiguous_f64(&storage, &meta, dim) {
        Ok(v) => v,
        Err(_) => return,
    };
    let mean_out = match mean_dim_tensor_contiguous_f64(&storage, &meta, dim) {
        Ok(v) => v,
        Err(_) => return,
    };
    let prod_out = match prod_dim_tensor_contiguous_f64(&storage, &meta, dim) {
        Ok(v) => v,
        Err(_) => return,
    };
    let (max_vals, max_idx) = match max_dim_tensor_contiguous_f64(&storage, &meta, dim) {
        Ok(v) => v,
        Err(_) => return,
    };
    let (min_vals, min_idx) = match min_dim_tensor_contiguous_f64(&storage, &meta, dim) {
        Ok(v) => v,
        Err(_) => return,
    };

    // Length contracts.
    assert_eq!(sum_out.len(), out_numel, "sum_dim length");
    assert_eq!(mean_out.len(), out_numel, "mean_dim length");
    assert_eq!(prod_out.len(), out_numel, "prod_dim length");
    assert_eq!(max_vals.len(), out_numel, "max_dim values length");
    assert_eq!(max_idx.len(), out_numel, "max_dim indices length");
    assert_eq!(min_vals.len(), out_numel, "min_dim values length");
    assert_eq!(min_idx.len(), out_numel, "min_dim indices length");

    // Cross-op invariant: max_vals >= mean >= min_vals (within FP
    // slack) wherever all three are finite. This couples three
    // independent reductions — any drift in any one breaks the
    // chain.
    const EPS: f64 = 1e-9;
    for i in 0..out_numel {
        if !max_vals[i].is_finite() || !mean_out[i].is_finite() || !min_vals[i].is_finite() {
            continue;
        }
        assert!(
            max_vals[i] >= mean_out[i] - EPS,
            "max[{i}]={} < mean[{i}]={}", max_vals[i], mean_out[i]
        );
        assert!(
            mean_out[i] >= min_vals[i] - EPS,
            "mean[{i}]={} < min[{i}]={}", mean_out[i], min_vals[i]
        );
    }

    // Argmax/argmin range: indices must be in [0, dim_size).
    let dim_size = shape[dim];
    if dim_size == 0 {
        return;
    }
    for (i, &idx) in max_idx.iter().enumerate() {
        let idx_u = idx as usize;
        assert!(
            (idx - idx_u as f64).abs() < f64::EPSILON,
            "max idx[{i}] = {idx} not integer"
        );
        assert!(
            idx_u < dim_size,
            "max idx[{i}] = {idx_u} >= dim_size {dim_size}"
        );
    }
    for (i, &idx) in min_idx.iter().enumerate() {
        let idx_u = idx as usize;
        assert!(
            (idx - idx_u as f64).abs() < f64::EPSILON,
            "min idx[{i}] = {idx} not integer"
        );
        assert!(
            idx_u < dim_size,
            "min idx[{i}] = {idx_u} >= dim_size {dim_size}"
        );
    }

    // dim_size==1 invariant: every reduction equals the input.
    if dim_size == 1 {
        let inner_size: usize = shape[dim + 1..].iter().product();
        let outer_size: usize = shape[..dim].iter().product();
        for outer in 0..outer_size {
            for inner in 0..inner_size {
                let flat_in = outer * inner_size + inner; // dim_size=1 collapses
                let flat_out = outer * inner_size + inner;
                let v = storage[flat_in];
                if !v.is_finite() {
                    continue;
                }
                let scale = v.abs().max(1.0);
                assert!(
                    (sum_out[flat_out] - v).abs() <= 4.0 * f64::EPSILON * scale,
                    "sum(dim_size=1)[{flat_out}] = {} != input {v}", sum_out[flat_out]
                );
                assert!(
                    (mean_out[flat_out] - v).abs() <= 4.0 * f64::EPSILON * scale,
                    "mean(dim_size=1)[{flat_out}] = {} != input {v}", mean_out[flat_out]
                );
                assert!(
                    (prod_out[flat_out] - v).abs() <= 4.0 * f64::EPSILON * scale,
                    "prod(dim_size=1)[{flat_out}] = {} != input {v}", prod_out[flat_out]
                );
                assert!(
                    max_vals[flat_out] == v,
                    "max(dim_size=1)[{flat_out}] = {} != input {v}", max_vals[flat_out]
                );
                assert!(
                    min_vals[flat_out] == v,
                    "min(dim_size=1)[{flat_out}] = {} != input {v}", min_vals[flat_out]
                );
            }
        }
    }
});

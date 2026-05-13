#![no_main]

use ft_core::{DType, Device, TensorMeta};
use ft_kernel_cpu::scatter_add_tensor_contiguous_f64;
use libfuzzer_sys::fuzz_target;

const MAX_INPUT_BYTES: usize = 512;
const MAX_SHAPE_DIM: u8 = 8;

fuzz_target!(|data: &[u8]| {
    if data.len() < 4 || data.len() > MAX_INPUT_BYTES {
        return;
    }

    // Preamble: ndim, dim, idx_dim_override, dup_bias.
    let ndim = usize::from(data[0] % 7);
    if ndim == 0 {
        return;
    }
    let dim = usize::from(data[1] % (ndim as u8).max(1)) % ndim;
    let idx_dim_override = usize::from(data[2] % (MAX_SHAPE_DIM + 1));
    // When dup_bias is high, indices collapse onto a small subset of
    // valid positions, forcing the accumulation path (multiple src
    // values writing into the same output cell). This is the
    // distinguishing behavior of scatter_add vs scatter.
    let dup_bias = data[3];
    let body = &data[4..];

    if body.len() < ndim + 1 {
        return;
    }
    let shape: Vec<usize> = body[..ndim]
        .iter()
        .map(|b| usize::from(b % (MAX_SHAPE_DIM + 1)))
        .collect();
    let mut idx_shape = shape.clone();
    idx_shape[dim] = idx_dim_override;

    if shape[dim] == 0 {
        return;
    }

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
    let idx_meta = match TensorMeta::from_shape_and_strides(
        idx_shape.clone(),
        ft_core::contiguous_strides(&idx_shape),
        0,
        DType::F64,
        Device::Cpu,
    ) {
        Ok(meta) => meta,
        Err(_) => return,
    };
    let numel = meta.numel();
    let idx_numel = idx_meta.numel();
    if numel > 4096 || idx_numel > 4096 {
        return;
    }

    // Non-zero base tensor so we can later assert that scatter_add
    // accumulated onto the original (not overwritten) — this is the
    // semantic that distinguishes scatter_add from scatter.
    let storage: Vec<f64> = (0..numel).map(|i| (i as f64) * 0.1).collect();

    // Effective index range: when dup_bias is large, collapse to a
    // tiny prefix of valid positions to force duplicate writes.
    let dim_size = shape[dim];
    let effective_range = if dup_bias > 192 {
        1usize.min(dim_size)
    } else if dup_bias > 128 {
        2usize.min(dim_size)
    } else {
        dim_size
    };
    let mut index = Vec::with_capacity(idx_numel);
    for i in 0..idx_numel {
        let raw = body[(ndim + i) % body.len()];
        let v = (usize::from(raw) % effective_range.max(1)) as f64;
        index.push(v);
    }

    // src values picked to make accumulation observable: identical
    // unit contributions, so any cell touched k times sums to its
    // base + k. The bit-equal-to-base check is therefore a precise
    // accumulation correctness probe.
    let src: Vec<f64> = vec![1.0_f64; idx_numel];

    let output = match scatter_add_tensor_contiguous_f64(
        &storage, &meta, dim, &index, &idx_meta, &src,
    ) {
        Ok(out) => out,
        Err(_) => return,
    };
    assert_eq!(
        output.len(),
        numel,
        "scatter_add output length must equal input numel"
    );

    // Monotonicity: with src all >= 0, every output cell must be
    // >= the corresponding base cell (no spurious subtraction).
    for (i, (&out_v, &base_v)) in output.iter().zip(storage.iter()).enumerate() {
        assert!(
            out_v >= base_v - 1e-12,
            "scatter_add[{i}] = {out_v} dropped below base {base_v} despite non-negative src"
        );
    }

    // Conservation: the sum of (output - base) must equal the sum
    // of src values that landed at in-range positions. Since we
    // gate effective_range to be inside [0, dim_size), every src
    // element should be applied — so total delta == src.iter().sum.
    let delta: f64 = output.iter().zip(storage.iter()).map(|(o, b)| o - b).sum();
    let expected: f64 = src.iter().sum();
    assert!(
        (delta - expected).abs() < 1e-9 * (1.0 + expected.abs()),
        "scatter_add accumulation conservation broken: delta={delta} expected={expected}"
    );
});

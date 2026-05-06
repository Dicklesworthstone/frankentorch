#![no_main]

use ft_core::{DType, Device, TensorMeta};
use ft_kernel_cpu::where_tensor_contiguous_f64;
use libfuzzer_sys::fuzz_target;

const MAX_INPUT_BYTES: usize = 512;
const MAX_SHAPE_DIM: u8 = 8;

fuzz_target!(|data: &[u8]| {
    if data.len() < 3 || data.len() > MAX_INPUT_BYTES {
        return;
    }

    // Preamble: ndim, length_mode (0 = all match, 1 = truncated
    // condition, 2 = truncated y), then shape bytes. The
    // truncation modes deliberately exercise the kernel's
    // length-validation paths (condition.len() < offset+numel
    // and y.len() < offset+numel) which return ShapeMismatch.
    let ndim = usize::from(data[0] % 7);
    if ndim == 0 {
        return;
    }
    let length_mode = data[1] % 3;
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

    let x = vec![1.0_f64; numel];
    let y = vec![2.0_f64; numel];
    // Mix true and false in the condition so the kernel
    // exercises both selection branches per call.
    let condition: Vec<f64> = (0..numel)
        .map(|i| if i % 2 == 0 { 1.0 } else { 0.0 })
        .collect();

    // Truncate one of the operands to drive the length-check
    // paths. truncate_by_one only fires when numel >= 1 (already
    // checked); for numel == 0 the kernel short-circuits before
    // length checks.
    let (cond_slice, y_slice): (&[f64], &[f64]) = match length_mode {
        // All operands have correct length — kernel must succeed.
        0 => (&condition[..], &y[..]),
        // Condition truncated: kernel must reject with
        // ShapeMismatch and not panic. Only meaningful when
        // numel >= 1.
        1 if numel >= 1 => (&condition[..numel - 1], &y[..]),
        // Y truncated: same shape-mismatch path on the y-length
        // check.
        2 if numel >= 1 => (&condition[..], &y[..numel - 1]),
        _ => return,
    };

    match where_tensor_contiguous_f64(cond_slice, &x, y_slice, &meta) {
        Ok(output) => {
            // Successful path: output length must equal numel
            // and every cell must be either 1.0 (from x where
            // condition true) or 2.0 (from y where condition
            // false) — never anything else.
            assert_eq!(
                output.len(),
                numel,
                "where output length must equal numel"
            );
            for (i, val) in output.iter().enumerate() {
                assert!(
                    *val == 1.0 || *val == 2.0,
                    "where output[{i}] = {val}, expected 1.0 or 2.0"
                );
            }
        }
        Err(_) => {
            // Length-mismatch paths return Err; that's fine.
            // The fuzz target's contract is no panic.
        }
    }
});

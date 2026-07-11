//! Same-worker RAYON A/B for narrow_tensor_contiguous_{f64,f32} (serial per-element push ->
//! parallel copy_from_slice block copy). Kernels called directly; 1t ≈ old serial, Nt parallel.
//! Run: RAYON_NUM_THREADS=N cargo run --release -p ft-api --example narrow_ab

use std::time::Instant;

use ft_core::{DType, TensorMeta};

const R: usize = 8000;
const C: usize = 8000;
const START: usize = 2000;
const LEN: usize = 4000;

fn best<F: FnMut()>(mut f: F) -> f64 {
    let mut b = f64::INFINITY;
    for _ in 0..9 {
        let t = Instant::now();
        f();
        let e = t.elapsed().as_secs_f64() * 1e3;
        if e < b {
            b = e;
        }
    }
    b
}

// Reproduction of the ORIGINAL serial per-element push path (pre-change baseline).
fn narrow_old_push(
    data: &[f64],
    outer: usize,
    dim_size: usize,
    inner: usize,
    start: usize,
    length: usize,
) -> Vec<f64> {
    let mut output = Vec::with_capacity(outer * length * inner);
    for o in 0..outer {
        for r in 0..length {
            for i in 0..inner {
                output.push(data[o * dim_size * inner + (start + r) * inner + i]);
            }
        }
    }
    output
}

fn main() {
    let n = R * C;
    let d64: Vec<f64> = (0..n).map(|i| (i % 9973) as f64 - 4986.0).collect();
    let d32: Vec<f32> = (0..n).map(|i| (i % 9973) as f32 - 4986.0).collect();
    let m64 = TensorMeta::from_shape(vec![R, C], DType::F64, ft_core::Device::Cpu);
    let m32 = TensorMeta::from_shape(vec![R, C], DType::F32, ft_core::Device::Cpu);

    // ORIGINAL per-element-push serial baseline (f64), single-thread by construction.
    let old = best(|| {
        let _ = narrow_old_push(&d64, R, C, 1, START, LEN);
    });
    println!(
        "narrow OLD push-serial f64: {old:.3} ms  (threads={})",
        rayon::current_num_threads()
    );

    // Narrow the LAST dim (inner_size=1, outer_size=R rows, each a contiguous LEN-run).
    let f64t = best(|| {
        let _ = ft_kernel_cpu::narrow_tensor_contiguous_f64(&d64, &m64, 1, START, LEN).unwrap();
    });
    let f32t = best(|| {
        let _ = ft_kernel_cpu::narrow_tensor_contiguous_f32(&d32, &m32, 1, START, LEN).unwrap();
    });

    let t = rayon::current_num_threads();
    println!("narrow dim=1 [{R},{C}]->len{LEN} f64: {f64t:.3} ms  (threads={t})");
    println!("narrow dim=1 [{R},{C}]->len{LEN} f32: {f32t:.3} ms  (threads={t})");
}

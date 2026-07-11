//! Same-worker RAYON A/B for the f32 cat + stack kernel parallelization (asymmetry method:
//! f64 siblings landed; these f32 ones were the lone remaining serial outliers). Kernels called
//! directly; 1t time ≈ old serial, Nt is the parallel path.
//! Run: RAYON_NUM_THREADS=N cargo run --release -p ft-api --example catstack_f32_ab

use std::time::Instant;

use ft_core::{DType, TensorMeta};

const R: usize = 4000;
const C: usize = 4000;
const K: usize = 4;

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

fn main() {
    let inputs: Vec<(Vec<f32>, TensorMeta)> = (0..K)
        .map(|k| {
            let data: Vec<f32> = (0..R * C)
                .map(|i| ((i.wrapping_mul(2654435761usize).wrapping_add(k)) % 9973) as f32 - 4986.0)
                .collect();
            let meta = TensorMeta::from_shape(vec![R, C], DType::F32, ft_core::Device::Cpu);
            (data, meta)
        })
        .collect();
    let refs: Vec<(&[f32], &TensorMeta)> = inputs.iter().map(|(d, m)| (d.as_slice(), m)).collect();

    let cat = best(|| {
        let _ = ft_kernel_cpu::cat_tensor_contiguous_f32(&refs, 1).unwrap();
    });
    let stack = best(|| {
        let _ = ft_kernel_cpu::stack_tensor_contiguous_f32(&refs, 1).unwrap();
    });

    let t = rayon::current_num_threads();
    println!("cat   dim=1 [{R},{C}]x{K} f32: {cat:.3} ms  (threads={t})");
    println!("stack dim=1 [{R},{K},{C}] f32: {stack:.3} ms  (threads={t})");
}

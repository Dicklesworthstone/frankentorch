//! Same-worker RAYON A/B for stack_tensor_contiguous_f64 outer-row parallelization.
//! Calls the ft_kernel_cpu kernel directly; 1t time ≈ old serial, Nt is the parallel path.
//! Run: RAYON_NUM_THREADS=N cargo run --release -p ft-api --example stack_ab

use std::time::Instant;

use ft_core::{DType, TensorMeta};

const R: usize = 4000;
const C: usize = 4000;
const K: usize = 4; // stack 4 inputs along a new dim

fn main() {
    // K inputs of shape [R, C], stacked along dim=1 → output [R, K, C].
    let inputs: Vec<(Vec<f64>, TensorMeta)> = (0..K)
        .map(|k| {
            let data: Vec<f64> = (0..R * C)
                .map(|i| ((i.wrapping_mul(2654435761usize).wrapping_add(k)) % 9973) as f64 - 4986.0)
                .collect();
            let meta = TensorMeta::from_shape(vec![R, C], DType::F64, ft_core::Device::Cpu);
            (data, meta)
        })
        .collect();
    let refs: Vec<(&[f64], &TensorMeta)> = inputs.iter().map(|(d, m)| (d.as_slice(), m)).collect();

    let mut best = f64::INFINITY;
    for _ in 0..9 {
        let t = Instant::now();
        let _ = ft_kernel_cpu::stack_tensor_contiguous_f64(&refs, 1).unwrap();
        let e = t.elapsed().as_secs_f64() * 1e3;
        if e < best {
            best = e;
        }
    }
    println!(
        "stack dim=1 [{R},{K},{C}] f64: {best:.3} ms  (threads={})",
        rayon::current_num_threads()
    );
}

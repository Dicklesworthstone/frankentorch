//! Same-worker RAYON A/B for cummax last-dim parallelization (kernel called directly).
//! Run: RAYON_NUM_THREADS=N cargo run --release -p ft-api --example cummax_ab

use ft_core::{DType, TensorMeta};
use std::time::Instant;

const R: usize = 4000;
const C: usize = 4000;

fn main() {
    let data: Vec<f64> = (0..R * C)
        .map(|i| ((i * 2654435761usize) % 9973) as f64 - 4986.0)
        .collect();
    let meta = TensorMeta::from_shape(vec![R, C], DType::F64, ft_core::Device::Cpu);
    let mut best = f64::INFINITY;
    for _ in 0..9 {
        let t = Instant::now();
        let _ = ft_kernel_cpu::cummax_dim_tensor_contiguous_f64(&data, &meta, 1).unwrap();
        let e = t.elapsed().as_secs_f64() * 1e3;
        if e < best {
            best = e;
        }
    }
    println!(
        "cummax dim=1 [{R},{C}] f64: {best:.3} ms  (threads={})",
        rayon::current_num_threads()
    );
}

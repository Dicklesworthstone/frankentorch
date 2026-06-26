//! Same-worker RAYON A/B for the lerp kernel parallelization (kernel called directly).
//! Run: RAYON_NUM_THREADS=N cargo run --release -p ft-api --example lerp_ab
use std::time::Instant;
use ft_core::{DType, TensorMeta};
const R: usize = 4000;
const C: usize = 4000;
fn main() {
    let s: Vec<f64> = (0..R*C).map(|i| (i % 17) as f64 - 8.0).collect();
    let e: Vec<f64> = (0..R*C).map(|i| (i % 13) as f64 - 6.0).collect();
    let meta = TensorMeta::from_shape(vec![R, C], DType::F64, ft_core::Device::Cpu);
    let mut best = f64::INFINITY;
    for _ in 0..9 {
        let t = Instant::now();
        let _ = ft_kernel_cpu::lerp_tensor_contiguous_f64(&s, &e, 0.3, &meta).unwrap();
        let el = t.elapsed().as_secs_f64()*1e3; if el<best {best=el;}
    }
    println!("lerp [{R},{C}] f64: {best:.3} ms (threads={})", rayon::current_num_threads());
}

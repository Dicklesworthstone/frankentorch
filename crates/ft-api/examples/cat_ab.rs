//! Same-worker RAYON A/B for the cat kernel parallelization (kernel called directly).
//! Run: RAYON_NUM_THREADS=N cargo run --release -p ft-api --example cat_ab
use ft_core::{DType, TensorMeta};
use std::time::Instant;
const R: usize = 4000;
const C: usize = 4000;
fn main() {
    let a: Vec<f64> = (0..R * C).map(|i| (i % 17) as f64).collect();
    let b: Vec<f64> = (0..R * C).map(|i| (i % 13) as f64).collect();
    let meta = TensorMeta::from_shape(vec![R, C], DType::F64, ft_core::Device::Cpu);
    let inputs = [(&a[..], &meta), (&b[..], &meta)];
    let mut best = f64::INFINITY;
    for _ in 0..9 {
        let t = Instant::now();
        let _ = ft_kernel_cpu::cat_tensor_contiguous_f64(&inputs, 1).unwrap();
        let el = t.elapsed().as_secs_f64() * 1e3;
        if el < best {
            best = el;
        }
    }
    println!(
        "cat dim=1 [{R},{C}]x2 f64: {best:.3} ms (threads={})",
        rayon::current_num_threads()
    );
}

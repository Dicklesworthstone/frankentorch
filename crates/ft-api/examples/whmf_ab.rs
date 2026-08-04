//! Same-worker RAYON_NUM_THREADS A/B for the where/masked_fill KERNEL parallelization. Calls
//! the ft_kernel_cpu kernels directly (the ft-api no-grad fast paths bypass them). Run twice:
//! RAYON_NUM_THREADS=1 (≈ old serial) vs =64 (parallel); the 1t/Nt ratio is the win.
//! Run: RAYON_NUM_THREADS=N cargo run --release -p ft-api --example whmf_ab

use std::time::Instant;

use ft_core::{DType, TensorMeta};

const R: usize = 4000;
const C: usize = 4000;

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
    let n = R * C;
    let cond: Vec<f64> = (0..n).map(|i| (i % 2) as f64).collect();
    let x: Vec<f64> = (0..n).map(|i| (i % 17) as f64 - 8.0).collect();
    let y: Vec<f64> = (0..n).map(|i| (i % 13) as f64 - 6.0).collect();
    let mask: Vec<f64> = (0..n).map(|i| (i % 3 == 0) as i32 as f64).collect();
    let meta = TensorMeta::from_shape(vec![R, C], DType::F64, ft_core::Device::Cpu);

    let wh = best(|| {
        let _ = ft_kernel_cpu::where_tensor_contiguous_f64(&cond, &x, &y, &meta).unwrap();
    });
    let mf = best(|| {
        let _ = ft_kernel_cpu::masked_fill_tensor_contiguous_f64(&x, &meta, &mask, 0.0).unwrap();
    });

    let t = rayon::current_num_threads();
    println!("where      [{R},{C}] f64: {wh:.3} ms  (threads={t})");
    println!("masked_fill[{R},{C}] f64: {mf:.3} ms  (threads={t})");
}

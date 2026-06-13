use ft_core::{DType, Device, TensorMeta};
use std::time::Instant;
fn t<F: FnMut()>(mut f: F) -> f64 {
    for _ in 0..3 {
        f();
    }
    let mut b = f64::MAX;
    for _ in 0..10 {
        let s = Instant::now();
        for _ in 0..10 {
            f();
        }
        b = b.min(s.elapsed().as_secs_f64() / 10.0 * 1000.0);
    }
    b
}
fn main() {
    println!("threads={}", rayon::current_num_threads());
    let n = 2048;
    let d: Vec<f64> = (0..n * n)
        .map(|i| ((i * 2654435761usize) % 10007) as f64 * 0.001)
        .collect();
    let m = TensorMeta::from_shape(vec![n, n], DType::F64, Device::Cpu);
    println!(
        "sum_d1  {:.3} ms",
        t(|| {
            let _ = ft_kernel_cpu::sum_dim_tensor_contiguous_f64(&d, &m, 1).unwrap();
        })
    );
    println!(
        "sum_d0  {:.3} ms",
        t(|| {
            let _ = ft_kernel_cpu::sum_dim_tensor_contiguous_f64(&d, &m, 0).unwrap();
        })
    );
    println!(
        "mean_d1 {:.3} ms",
        t(|| {
            let _ = ft_kernel_cpu::mean_dim_tensor_contiguous_f64(&d, &m, 1).unwrap();
        })
    );
    println!(
        "prod_d1 {:.3} ms",
        t(|| {
            let _ = ft_kernel_cpu::prod_dim_tensor_contiguous_f64(&d, &m, 1).unwrap();
        })
    );
}

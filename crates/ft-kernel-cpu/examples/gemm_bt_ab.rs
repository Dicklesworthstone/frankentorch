// Linear-layer (x @ W^T) throughput aid via the public transposed matmul entry.
use ft_kernel_cpu::matmul_rhs_transposed_contiguous_f64 as bt;
use std::time::Instant;
fn main() {
    println!("threads = {}", rayon::current_num_threads());
    for &(m, k, n) in &[
        (512usize, 1024, 1024),
        (1024, 1024, 1024),
        (2048, 512, 2048),
    ] {
        let a: Vec<f64> = (0..m * k).map(|i| (i as f64 * 0.001).sin()).collect();
        let w: Vec<f64> = (0..n * k).map(|i| (i as f64 * 0.0017).cos()).collect();
        for _ in 0..3 {
            let _ = bt(m, k, n, &a, &w).unwrap();
        }
        let mut best = f64::MAX;
        for _ in 0..6 {
            let s = Instant::now();
            for _ in 0..10 {
                let _ = bt(m, k, n, &a, &w).unwrap();
            }
            best = best.min(s.elapsed().as_secs_f64() / 10.0 * 1000.0);
        }
        let g = 2.0 * m as f64 * k as f64 * n as f64 / 1e9;
        println!(
            "linear x[{m},{k}] @ W[{n},{k}]^T: {best:.3} ms  {:.0} GFLOP/s",
            g / (best / 1e3)
        );
    }
}

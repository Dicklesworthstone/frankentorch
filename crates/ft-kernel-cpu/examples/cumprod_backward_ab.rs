//! Anchored A/B for cumprod_backward lane parallelization. Calls the PRODUCTION
//! kernel under a 1-thread pool (serial) vs full pool (parallel) in ONE process,
//! asserts bit-for-bit equality, reports the ratio.
//!   cargo run -q --release -p ft-kernel-cpu --example cumprod_backward_ab
use ft_core::{DType, Device, TensorMeta};
use ft_kernel_cpu::{cumprod_backward_tensor_contiguous_f64, cumprod_tensor_contiguous_f64};

fn run_best(
    threads: usize,
    go: &[f64],
    inp: &[f64],
    out: &[f64],
    meta: &TensorMeta,
) -> (f64, Vec<f64>) {
    let pool = rayon::ThreadPoolBuilder::new().num_threads(threads).build().unwrap();
    pool.install(|| {
        let r = cumprod_backward_tensor_contiguous_f64(go, inp, out, meta, 1).unwrap();
        let mut best = f64::INFINITY;
        for _ in 0..20 {
            let t = std::time::Instant::now();
            std::hint::black_box(
                cumprod_backward_tensor_contiguous_f64(go, inp, out, meta, 1).unwrap(),
            );
            best = best.min(t.elapsed().as_secs_f64() * 1e3);
        }
        (best, r)
    })
}

fn main() {
    let (rows, cols) = (8192usize, 1024);
    let n = rows * cols;
    // Non-zero inputs near 1.0 so cumprod stays finite (common, non-zero branch).
    let inp: Vec<f64> = (0..n).map(|i| 1.0 + (((i % 17) as f64) - 8.0) * 1e-3).collect();
    let meta = TensorMeta::from_shape(vec![rows, cols], DType::F64, Device::Cpu);
    let out = cumprod_tensor_contiguous_f64(&inp, &meta, 1).unwrap();
    let go: Vec<f64> = (0..n).map(|i| 1.0 + ((i % 13) as f64) * 1e-3).collect();
    let nthreads = rayon::current_num_threads();

    let (ser, r_ser) = run_best(1, &go, &inp, &out, &meta);
    let (par, r_par) = run_best(nthreads.max(2), &go, &inp, &out, &meta);
    assert_eq!(r_ser.len(), r_par.len());
    for (a, b) in r_ser.iter().zip(r_par.iter()) {
        assert_eq!(a.to_bits(), b.to_bits(), "parallel != serial bit-exact");
    }
    println!(
        "cumprod_backward [{rows},{cols}] dim=1 (bit-exact OK): serial(1t) {ser:.2}ms  parallel({nthreads}t) {par:.2}ms  =>  {:.2}x",
        ser / par
    );
}

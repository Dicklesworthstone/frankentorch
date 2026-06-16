//! Diligence probe for bead frankentorch-z6sjf: does the current 2-D-tiled dgemm
//! scale across cores? Times tensor_matmul (square + tall-skinny "linear" shapes)
//! under a 1-thread vs all-cores rayon pool, in ONE process, op-only (fresh session
//! per iter, input build outside the timed region). Near-linear speedup ⇒ GEMM
//! multi-core scaling is already good (no parallelization gap; any win must come
//! from a better microkernel, which matrixmultiply already provides). Sublinear ⇒
//! a real parallelization lever remains.
//!   cargo run -q --release -p ft-api --example matmul_scaling_probe
use ft_api::FrankenTorchSession;
use ft_core::ExecutionMode;
use rayon::ThreadPool;
use std::time::Instant;

fn time_mm(pool: &ThreadPool, av: &[f64], bv: &[f64], m: usize, k: usize, n: usize) -> f64 {
    let mut best = f64::INFINITY;
    for _ in 0..12 {
        let mut s = FrankenTorchSession::new(ExecutionMode::Strict);
        let a = s.tensor_variable(av.to_vec(), vec![m, k], false).unwrap();
        let b = s.tensor_variable(bv.to_vec(), vec![k, n], false).unwrap();
        let t = Instant::now();
        pool.install(|| {
            s.tensor_matmul(a, b).unwrap();
        });
        best = best.min(t.elapsed().as_secs_f64() * 1e3);
    }
    best
}

fn main() {
    let pool1 = rayon::ThreadPoolBuilder::new().num_threads(1).build().unwrap();
    let pooln = rayon::ThreadPoolBuilder::new().build().unwrap();
    let nt = pooln.current_num_threads();
    let mut out = String::new();

    for &(m, k, n, label) in &[
        (2048usize, 2048usize, 2048usize, "square[2048^3]"),
        (8192, 1024, 1024, "linear[8192x1024x1024]"),
        (1024, 1024, 4096, "wide[1024x1024x4096]"),
    ] {
        let av: Vec<f64> = (0..m * k).map(|i| (i % 1009) as f64 * 0.001).collect();
        let bv: Vec<f64> = (0..k * n).map(|i| (i % 1013) as f64 * 0.001).collect();
        let s1 = time_mm(&pool1, &av, &bv, m, k, n);
        let sn = time_mm(&pooln, &av, &bv, m, k, n);
        let gflops = 2.0 * (m as f64) * (k as f64) * (n as f64) / 1e9;
        let line = format!(
            "{label}: 1t {s1:.2} ms ({:.1} GF/s) / {nt}t {sn:.2} ms ({:.1} GF/s) / scaling {:.2}x of {nt}",
            gflops / (s1 / 1e3),
            gflops / (sn / 1e3),
            s1 / sn
        );
        eprintln!("{line}");
        out.push_str(&line);
        out.push('\n');
    }
    let _ = std::fs::write("artifacts/perf/matmul_scaling_probe.txt", &out);
}

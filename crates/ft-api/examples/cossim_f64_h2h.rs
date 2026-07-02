// Same-process A/B for the fused cosine_similarity f64 last-dim path.
// FT_ORIG unset -> fused per-row native; FT_ORIG set -> composed mul+sum_dim+sqrt+...+div.
// Input created BEFORE Instant::now(). torch F.cosine_similarity f64 ~33ms at [4096,4096] dim=1.
use ft_api::FrankenTorchSession;
use ft_core::ExecutionMode;
use std::time::Instant;

fn bench<F: FnMut() -> u128>(iters: usize, mut f: F) -> u128 {
    let mut best = u128::MAX;
    for _ in 0..iters {
        let t = f();
        if t < best {
            best = t;
        }
    }
    best
}

fn main() {
    let orig = std::env::var("FT_ORIG").is_ok();
    let tag = if orig { "ORIG(compose)" } else { "FUSED" };
    let (rows, cols) = (4096usize, 4096usize);
    let n = rows * cols;
    let x1: Vec<f64> = (0..n).map(|i| ((i % 97) as f64) * 0.031 - 1.4).collect();
    let x2: Vec<f64> = (0..n).map(|i| ((i * 7 % 89) as f64) * 0.027 - 1.1).collect();
    let mut s = FrankenTorchSession::new(ExecutionMode::Strict);
    let a = s.tensor_variable(x1.clone(), vec![rows, cols], false).unwrap();
    let b = s.tensor_variable(x2.clone(), vec![rows, cols], false).unwrap();
    let _ = s.tensor_cosine_similarity(a, b, 1, 1e-8).unwrap();
    let t = bench(9, || {
        let t0 = Instant::now();
        let o = s.tensor_cosine_similarity(a, b, 1, 1e-8).unwrap();
        let e = t0.elapsed().as_micros();
        std::hint::black_box(o);
        e
    });
    println!("[{tag}] cosine_similarity f64 [4096,4096] dim=1: {:.2} ms", t as f64 / 1000.0);
}

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
    let tag = if std::env::var("FT_ORIG").is_ok() {
        "ORIG(clone)"
    } else {
        "FUSED(borrow)"
    };
    let (rows, cols) = (4096usize, 4096usize);
    let n = rows * cols;
    let data: Vec<f64> = (0..n)
        .map(|i| ((i * 2654435761usize) % 100000) as f64 * 0.001)
        .collect();
    let mut s = FrankenTorchSession::new(ExecutionMode::Strict);
    let a = s
        .tensor_variable(data.clone(), vec![rows, cols], false)
        .unwrap();
    let _ = s.tensor_max_dim(a, 1).unwrap();
    let t = bench(9, || {
        let t0 = Instant::now();
        let o = s.tensor_max_dim(a, 1).unwrap();
        let e = t0.elapsed().as_micros();
        std::hint::black_box(o);
        e
    });
    println!(
        "[{tag}] max_dim f64 [4096,4096] dim=1: {:.2} ms",
        t as f64 / 1000.0
    );
}

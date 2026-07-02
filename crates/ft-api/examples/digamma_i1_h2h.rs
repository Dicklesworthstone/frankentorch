use ft_api::FrankenTorchSession;
use ft_core::ExecutionMode;
use std::time::Instant;
fn bench<F: FnMut() -> u128>(iters: usize, mut f: F) -> u128 {
    let mut best = u128::MAX;
    for _ in 0..iters { let t = f(); if t < best { best = t; } }
    best
}
fn main() {
    let tag = if std::env::var("FT_ORIG").is_ok() { "ORIG(apply_fn)" } else { "FUSED" };
    let (rows, cols) = (4096usize, 4096usize);
    let n = rows * cols;
    for which in ["digamma", "i1"] {
        let data: Vec<f64> = if which == "digamma" {
            (0..n).map(|i| 0.05 + (i % 900) as f64 * 0.01).collect()
        } else {
            (0..n).map(|i| ((i % 800) as f64 / 100.0) - 4.0).collect()
        };
        let mut s = FrankenTorchSession::new(ExecutionMode::Strict);
        let a = s.tensor_variable(data.clone(), vec![rows, cols], false).unwrap();
        let call = |s: &mut FrankenTorchSession, a| if which == "digamma" { s.tensor_digamma(a).unwrap() } else { s.tensor_i1(a).unwrap() };
        let _ = call(&mut s, a);
        let t = bench(9, || { let t0 = Instant::now(); let o = call(&mut s, a); let e = t0.elapsed().as_micros(); std::hint::black_box(o); e });
        println!("[{tag}] {which} f64 [4096,4096]: {:.2} ms", t as f64 / 1000.0);
    }
}

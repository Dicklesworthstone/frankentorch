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
    let a: Vec<f64> = (0..n).map(|i| 0.3 + (i % 50) as f64 * 0.1).collect();
    let x: Vec<f64> = (0..n).map(|i| (i % 90) as f64 * 0.1).collect();
    for which in ["igamma", "igammac"] {
        let mut s = FrankenTorchSession::new(ExecutionMode::Strict);
        let ta = s
            .tensor_variable(a.clone(), vec![rows, cols], false)
            .unwrap();
        let tx = s
            .tensor_variable(x.clone(), vec![rows, cols], false)
            .unwrap();
        let call = |s: &mut FrankenTorchSession, ta, tx| {
            if which == "igamma" {
                s.tensor_igamma(ta, tx).unwrap()
            } else {
                s.tensor_igammac(ta, tx).unwrap()
            }
        };
        let _ = call(&mut s, ta, tx);
        let t = bench(9, || {
            let t0 = Instant::now();
            let o = call(&mut s, ta, tx);
            let e = t0.elapsed().as_micros();
            std::hint::black_box(o);
            e
        });
        println!(
            "[{tag}] {which} f64 [4096,4096]: {:.2} ms",
            t as f64 / 1000.0
        );
    }
}

// Same-process A/B for the fused tanhshrink f64 path.
// FT_ORIG unset -> fused x - tanh(x) one pass; FT_ORIG set -> composed tanh + sub.
// try_f64_unary_native has no env gate, so ORIG here forces the compose by requesting grad off but
// routing through a manual compose is not possible from outside; instead this bench compares FUSED
// (current) vs the composed path measured by calling tensor_tanh + tensor_sub directly.
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
    let (rows, cols) = (4096usize, 4096usize);
    let n = rows * cols;
    let data: Vec<f64> = (0..n).map(|i| ((i % 211) as f64) * 0.017 - 1.8).collect();

    // FUSED
    {
        let mut s = FrankenTorchSession::new(ExecutionMode::Strict);
        let a = s.tensor_variable(data.clone(), vec![rows, cols], false).unwrap();
        let _ = s.tensor_tanhshrink(a).unwrap();
        let t = bench(9, || {
            let t0 = Instant::now();
            let o = s.tensor_tanhshrink(a).unwrap();
            let e = t0.elapsed().as_micros();
            std::hint::black_box(o);
            e
        });
        println!("[FUSED] tanhshrink f64 [4096,4096]: {:.2} ms", t as f64 / 1000.0);
    }
    // ORIG compose (tanh + sub explicitly)
    {
        let mut s = FrankenTorchSession::new(ExecutionMode::Strict);
        let a = s.tensor_variable(data.clone(), vec![rows, cols], false).unwrap();
        let t0w = s.tensor_tanh(a).unwrap();
        let _ = s.tensor_sub(a, t0w).unwrap();
        let t = bench(9, || {
            let t0 = Instant::now();
            let tv = s.tensor_tanh(a).unwrap();
            let o = s.tensor_sub(a, tv).unwrap();
            let e = t0.elapsed().as_micros();
            std::hint::black_box(o);
            e
        });
        println!("[ORIG(compose)] tanhshrink f64 [4096,4096]: {:.2} ms", t as f64 / 1000.0);
    }
}

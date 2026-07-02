// A/B for the fused pow_tensor f64 path (torch.float_power with a tensor exponent).
// FUSED = tensor_pow_tensor (fast path). ORIG = explicit compose log+mul+exp.
// Inputs materialized BEFORE Instant::now(). torch.float_power(base, exp_tensor) ~73ms [4096,4096].
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
    let base: Vec<f64> = (0..n).map(|i| 0.05 + (i % 97) as f64 * 0.04).collect(); // positive
    let expo: Vec<f64> = (0..n).map(|i| 0.5 + (i % 23) as f64 * 0.1).collect();

    // FUSED
    {
        let mut s = FrankenTorchSession::new(ExecutionMode::Strict);
        let x = s.tensor_variable(base.clone(), vec![rows, cols], false).unwrap();
        let e = s.tensor_variable(expo.clone(), vec![rows, cols], false).unwrap();
        let _ = s.tensor_pow_tensor(x, e).unwrap();
        let t = bench(9, || {
            let t0 = Instant::now();
            let o = s.tensor_pow_tensor(x, e).unwrap();
            let el = t0.elapsed().as_micros();
            std::hint::black_box(o);
            el
        });
        println!("[FUSED] pow_tensor f64 [4096,4096]: {:.2} ms", t as f64 / 1000.0);
    }
    // ORIG compose (log + mul + exp, no 0^0 mask since base>0)
    {
        let mut s = FrankenTorchSession::new(ExecutionMode::Strict);
        let x = s.tensor_variable(base.clone(), vec![rows, cols], false).unwrap();
        let e = s.tensor_variable(expo.clone(), vec![rows, cols], false).unwrap();
        let lg = s.tensor_log(x).unwrap();
        let ml = s.tensor_mul(e, lg).unwrap();
        let _ = s.tensor_exp(ml).unwrap();
        let t = bench(9, || {
            let t0 = Instant::now();
            let lg = s.tensor_log(x).unwrap();
            let ml = s.tensor_mul(e, lg).unwrap();
            let o = s.tensor_exp(ml).unwrap();
            let el = t0.elapsed().as_micros();
            std::hint::black_box(o);
            el
        });
        println!("[ORIG(compose)] pow_tensor f64 [4096,4096]: {:.2} ms", t as f64 / 1000.0);
    }
}

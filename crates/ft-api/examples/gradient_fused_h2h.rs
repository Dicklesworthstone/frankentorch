// Same-process A/B for the fused tensor_gradient_dim (torch.gradient equivalent).
// FT_ORIG unset -> fused single-pass central difference; FT_ORIG set -> composed narrow+sub+mul+cat.
// Input created BEFORE Instant::now(). torch.gradient baseline ~46ms at [4096,4096] f64.
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
    let data64: Vec<f64> = (0..n).map(|i| ((i % 101) as f64) * 0.037 - 1.7).collect();
    let data32: Vec<f32> = data64.iter().map(|&x| x as f32).collect();

    // f64 dim=1 (last, contiguous rows)
    {
        let mut s = FrankenTorchSession::new(ExecutionMode::Strict);
        let a = s.tensor_variable(data64.clone(), vec![rows, cols], false).unwrap();
        let _ = s.tensor_gradient_dim(a, 1, 0.5, 1).unwrap();
        let t = bench(9, || {
            let t0 = Instant::now();
            let o = s.tensor_gradient_dim(a, 1, 0.5, 1).unwrap();
            let e = t0.elapsed().as_micros();
            std::hint::black_box(o);
            e
        });
        println!("[{tag}] gradient f64 [4096,4096] dim=1 eo=1: {:.2} ms", t as f64 / 1000.0);
    }
    // f64 dim=0 (interior stride)
    {
        let mut s = FrankenTorchSession::new(ExecutionMode::Strict);
        let a = s.tensor_variable(data64.clone(), vec![rows, cols], false).unwrap();
        let _ = s.tensor_gradient_dim(a, 0, 0.5, 2).unwrap();
        let t = bench(9, || {
            let t0 = Instant::now();
            let o = s.tensor_gradient_dim(a, 0, 0.5, 2).unwrap();
            let e = t0.elapsed().as_micros();
            std::hint::black_box(o);
            e
        });
        println!("[{tag}] gradient f64 [4096,4096] dim=0 eo=2: {:.2} ms", t as f64 / 1000.0);
    }
    // f32 dim=1
    {
        let mut s = FrankenTorchSession::new(ExecutionMode::Strict);
        let a = s.tensor_variable_f32(data32.clone(), vec![rows, cols], false).unwrap();
        let _ = s.tensor_gradient_dim(a, 1, 0.5, 1).unwrap();
        let t = bench(9, || {
            let t0 = Instant::now();
            let o = s.tensor_gradient_dim(a, 1, 0.5, 1).unwrap();
            let e = t0.elapsed().as_micros();
            std::hint::black_box(o);
            e
        });
        println!("[{tag}] gradient f32 [4096,4096] dim=1 eo=1: {:.2} ms", t as f64 / 1000.0);
    }
}

// Same-process A/B for the fused diff n>=2 path.
// FT_ORIG unset -> fused iterated-contiguous-pass; FT_ORIG set -> composed n tape subs of views.
// Input created BEFORE Instant::now(). torch.diff(n=3) baseline ~32ms at [4096,4096] f64.
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
    let d64: Vec<f64> = (0..n).map(|i| ((i % 89) as f64) * 0.041 - 1.3).collect();
    let d32: Vec<f32> = d64.iter().map(|&x| x as f32).collect();
    for order in [2usize, 3] {
        let mut s = FrankenTorchSession::new(ExecutionMode::Strict);
        let a = s.tensor_variable(d64.clone(), vec![rows, cols], false).unwrap();
        let _ = s.tensor_diff_full(a, order, 1, None, None).unwrap();
        let t = bench(9, || {
            let t0 = Instant::now();
            let o = s.tensor_diff_full(a, order, 1, None, None).unwrap();
            let e = t0.elapsed().as_micros();
            std::hint::black_box(o);
            e
        });
        println!("[{tag}] diff f64 [4096,4096] dim=1 n={order}: {:.2} ms", t as f64 / 1000.0);
    }
    {
        let mut s = FrankenTorchSession::new(ExecutionMode::Strict);
        let a = s.tensor_variable_f32(d32.clone(), vec![rows, cols], false).unwrap();
        let _ = s.tensor_diff_full(a, 3, 1, None, None).unwrap();
        let t = bench(9, || {
            let t0 = Instant::now();
            let o = s.tensor_diff_full(a, 3, 1, None, None).unwrap();
            let e = t0.elapsed().as_micros();
            std::hint::black_box(o);
            e
        });
        println!("[{tag}] diff f32 [4096,4096] dim=1 n=3: {:.2} ms", t as f64 / 1000.0);
    }
}

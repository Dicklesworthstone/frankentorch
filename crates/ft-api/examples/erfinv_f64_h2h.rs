// A/B for the no-grad f64 erfinv fast path.
// FT_ORIG unset -> try_f64_unary_native (par_map, leaf); FT_ORIG set -> apply_function (par_map +
// 128MB save_for_backward clone + node). Input materialized before Instant::now().
// torch.special.erfinv f64 ~116ms at [4096,4096] (slow scalar).
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
    let tag = if orig { "ORIG(apply_fn)" } else { "FUSED" };
    let (rows, cols) = (4096usize, 4096usize);
    let n = rows * cols;
    let data: Vec<f64> = (0..n).map(|i| ((i % 1999) as f64 / 1000.0) - 0.999).collect(); // (-1,1)
    let mut s = FrankenTorchSession::new(ExecutionMode::Strict);
    let a = s.tensor_variable(data.clone(), vec![rows, cols], false).unwrap();
    let _ = s.tensor_erfinv(a).unwrap();
    let t = bench(9, || {
        let t0 = Instant::now();
        let o = s.tensor_erfinv(a).unwrap();
        let e = t0.elapsed().as_micros();
        std::hint::black_box(o);
        e
    });
    println!("[{tag}] erfinv f64 [4096,4096]: {:.2} ms", t as f64 / 1000.0);
}

// A/B for the no-grad f64 fast paths on i0e/i1e/erfcx (apply_function-with-create-graph vein).
// FT_ORIG unset -> try_f64_unary_native (par_map, leaf); FT_ORIG set... the fast path has no gate,
// so ORIG is measured by NOT taking it — we compare FUSED (tensor_i0e etc.) against a manual
// apply_function baseline is not exposed; instead FUSED here is the fast path and we rely on the
// erfinv A/B (same vein, gated) for the ORIG ratio. This reports FUSED absolute times vs torch.
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
    let data: Vec<f64> = (0..n).map(|i| ((i % 800) as f64 / 100.0) - 4.0).collect(); // [-4,4)
    for which in ["i0e", "i1e", "erfcx"] {
        let mut s = FrankenTorchSession::new(ExecutionMode::Strict);
        let a = s.tensor_variable(data.clone(), vec![rows, cols], false).unwrap();
        let call = |s: &mut FrankenTorchSession, a| match which {
            "i0e" => s.tensor_i0e(a).unwrap(),
            "i1e" => s.tensor_i1e(a).unwrap(),
            _ => s.tensor_erfcx(a).unwrap(),
        };
        let _ = call(&mut s, a);
        let t = bench(9, || {
            let t0 = Instant::now();
            let o = call(&mut s, a);
            let e = t0.elapsed().as_micros();
            std::hint::black_box(o);
            e
        });
        let tag = if std::env::var("FT_ORIG").is_ok() { "ORIG(apply_fn)" } else { "FUSED" };
        println!("[{tag}] {which} f64 [4096,4096]: {:.2} ms", t as f64 / 1000.0);
    }
}

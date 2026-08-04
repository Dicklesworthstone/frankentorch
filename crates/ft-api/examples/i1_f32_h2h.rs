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
    let data: Vec<f32> = (0..n).map(|i| ((i % 800) as f32 / 100.0) - 4.0).collect();
    for which in ["i1", "special_i1"] {
        let mut s = FrankenTorchSession::new(ExecutionMode::Strict);
        let a = s
            .tensor_variable_f32(data.clone(), vec![rows, cols], false)
            .unwrap();
        let call = |s: &mut FrankenTorchSession, a| {
            if which == "i1" {
                s.tensor_i1(a).unwrap()
            } else {
                s.tensor_special_i1(a).unwrap()
            }
        };
        let o0 = call(&mut s, a);
        let dt = s.tensor_dtype(o0).unwrap();
        let t = bench(9, || {
            let t0 = Instant::now();
            let o = call(&mut s, a);
            let e = t0.elapsed().as_micros();
            std::hint::black_box(o);
            e
        });
        println!(
            "[FUSED-f32] {which} f32 [4096,4096]: {:.2} ms  dtype={:?}",
            t as f64 / 1000.0,
            dt
        );
    }
}

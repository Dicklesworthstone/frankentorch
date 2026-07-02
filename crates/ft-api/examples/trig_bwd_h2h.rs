use ft_api::FrankenTorchSession;
use ft_core::ExecutionMode;
use std::time::Instant;

fn fingerprint(g: &[f64]) -> u64 {
    let mut h: u64 = 1469598103934665603;
    for &x in g {
        h = h.wrapping_mul(1099511628211) ^ x.to_bits();
    }
    h
}

// Ops whose backward = parallel tensor_backward_zip_map derivative map +
// accumulate_tensor_gradient (now Rayon-parallel). frankentorch-accum-par.
fn main() {
    let tag = std::env::var("FT_TAG").unwrap_or_else(|_| "FT".into());
    let (rows, cols) = (4096usize, 4096usize);
    let n = rows * cols;
    let data: Vec<f64> = (0..n).map(|i| ((i % 971) as f64) * 0.006 - 3.0).collect();
    for op in ["sin", "cos", "sinh", "cosh"] {
        let mut best = f64::INFINITY;
        let mut best_fwd = f64::INFINITY;
        let mut best_bwd = f64::INFINITY;
        let mut fp = 0u64;
        for _ in 0..7 {
            let mut s = FrankenTorchSession::new(ExecutionMode::Strict);
            let a = s.tensor_variable(data.clone(), vec![rows, cols], true).unwrap();
            let t0 = Instant::now();
            let y = match op {
                "sin" => s.tensor_sin(a).unwrap(),
                "cos" => s.tensor_cos(a).unwrap(),
                "sinh" => s.tensor_sinh(a).unwrap(),
                _ => s.tensor_cosh(a).unwrap(),
            };
            let t1 = Instant::now();
            let loss = s.tensor_sum(y).unwrap();
            s.tensor_backward(loss).unwrap();
            let t2 = Instant::now();
            let ms = (t2 - t0).as_secs_f64() * 1e3;
            if ms < best {
                best = ms;
                best_fwd = (t1 - t0).as_secs_f64() * 1e3;
                best_bwd = (t2 - t1).as_secs_f64() * 1e3;
                fp = fingerprint(&s.tensor_grad(a).unwrap().unwrap());
            }
            std::hint::black_box(&s);
        }
        println!("[{tag}] {op} f64 [4096,4096]: total {best:.2} ms (fwd {best_fwd:.2} + sum/bwd {best_bwd:.2}) | grad_fp=0x{fp:016x}");
    }
}

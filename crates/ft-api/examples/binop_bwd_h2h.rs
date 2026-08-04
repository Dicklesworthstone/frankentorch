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

fn main() {
    let tag = std::env::var("FT_TAG").unwrap_or_else(|_| "FT".into());
    let (rows, cols) = (4096usize, 4096usize);
    let n = rows * cols;
    let da: Vec<f64> = (0..n).map(|i| 0.5 + ((i % 971) as f64) * 0.01).collect();
    let db: Vec<f64> = (0..n).map(|i| 1.0 + ((i % 617) as f64) * 0.013).collect();
    for op in ["add", "sub", "mul", "div"] {
        let mut best = f64::INFINITY;
        let mut best_bwd = f64::INFINITY;
        let mut fpa = 0u64;
        for _ in 0..7 {
            let mut s = FrankenTorchSession::new(ExecutionMode::Strict);
            let a = s
                .tensor_variable(da.clone(), vec![rows, cols], true)
                .unwrap();
            let b = s
                .tensor_variable(db.clone(), vec![rows, cols], true)
                .unwrap();
            let t0 = Instant::now();
            let c = match op {
                "add" => s.tensor_add(a, b).unwrap(),
                "sub" => s.tensor_sub(a, b).unwrap(),
                "mul" => s.tensor_mul(a, b).unwrap(),
                _ => s.tensor_div(a, b).unwrap(),
            };
            let t1 = Instant::now();
            let loss = s.tensor_sum(c).unwrap();
            s.tensor_backward(loss).unwrap();
            let t2 = Instant::now();
            let ms = (t2 - t0).as_secs_f64() * 1e3;
            if ms < best {
                best = ms;
                best_bwd = (t2 - t1).as_secs_f64() * 1e3;
                fpa = fingerprint(&s.tensor_grad(a).unwrap().unwrap());
            }
            std::hint::black_box(&s);
        }
        println!(
            "[{tag}] {op} fwd+bwd f64 [4096,4096]: total {best:.2} ms (sum/bwd {best_bwd:.2}) | grad_a_fp=0x{fpa:016x}"
        );
    }
}

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

// exp/log/sigmoid/tanh/relu backward (fused zip_map). frankentorch-act-bwd-fused.
fn main() {
    let tag = std::env::var("FT_TAG").unwrap_or_else(|_| "FT".into());
    let (rows, cols) = (4096usize, 4096usize);
    let n = rows * cols;
    let data: Vec<f64> = (0..n).map(|i| 0.5 + ((i % 971) as f64) * 0.0026).collect();
    for op in ["exp", "log", "sigmoid", "tanh", "relu"] {
        let mut best = f64::INFINITY;
        let mut fp = 0u64;
        for _ in 0..7 {
            let mut s = FrankenTorchSession::new(ExecutionMode::Strict);
            let a = s
                .tensor_variable(data.clone(), vec![rows, cols], true)
                .unwrap();
            let t0 = Instant::now();
            let y = match op {
                "exp" => s.tensor_exp(a).unwrap(),
                "log" => s.tensor_log(a).unwrap(),
                "sigmoid" => s.tensor_sigmoid(a).unwrap(),
                "tanh" => s.tensor_tanh(a).unwrap(),
                _ => s.tensor_relu(a).unwrap(),
            };
            let loss = s.tensor_sum(y).unwrap();
            s.tensor_backward(loss).unwrap();
            let ms = t0.elapsed().as_secs_f64() * 1e3;
            if ms < best {
                best = ms;
                fp = fingerprint(&s.tensor_grad(a).unwrap().unwrap());
            }
            std::hint::black_box(&s);
        }
        println!("[{tag}] {op} fwd+bwd f64 [4096,4096]: {best:.2} ms | grad_fp=0x{fp:016x}");
    }
}

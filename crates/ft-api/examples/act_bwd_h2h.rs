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
    // positive range so rsqrt/softplus are well-defined
    let data: Vec<f64> = (0..n).map(|i| 0.5 + ((i % 971) as f64) * 0.01).collect();
    for op in ["rsqrt", "silu", "erf", "elu", "softplus"] {
        let mut best = f64::INFINITY;
        let mut fp = 0u64;
        for _ in 0..7 {
            let mut s = FrankenTorchSession::new(ExecutionMode::Strict);
            let a = s.tensor_variable(data.clone(), vec![rows, cols], true).unwrap();
            let t0 = Instant::now();
            let y = match op {
                "rsqrt" => s.tensor_rsqrt(a).unwrap(),
                "silu" => s.tensor_silu(a).unwrap(),
                "erf" => s.tensor_erf(a).unwrap(),
                "elu" => s.tensor_elu(a).unwrap(),
                _ => s.tensor_softplus(a).unwrap(),
            };
            let loss = s.tensor_sum(y).unwrap();
            s.tensor_backward(loss).unwrap();
            let ms = t0.elapsed().as_secs_f64() * 1e3;
            if ms < best {
                best = ms;
                let g = s.tensor_grad(a).unwrap().unwrap();
                fp = fingerprint(&g);
            }
            std::hint::black_box(&s);
        }
        println!("[{tag}] {op} fwd+bwd f64 [4096,4096]: {best:.2} ms | grad_fp=0x{fp:016x}");
    }
}

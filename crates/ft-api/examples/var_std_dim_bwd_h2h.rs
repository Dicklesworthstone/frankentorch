use ft_api::FrankenTorchSession;
use ft_core::ExecutionMode;
use std::time::Instant;

// Head-to-head timing + grad fingerprint for the fused var_dim / std_dim backward
// (frankentorch-normp-bwd-fused). Compare vs torch via the companion scratch script.
fn fingerprint(g: &[f64]) -> u64 {
    let mut h: u64 = 1469598103934665603;
    for &x in g {
        h = h.wrapping_mul(1099511628211) ^ x.to_bits();
    }
    h
}

fn main() {
    let (rows, cols) = (4096usize, 4096usize);
    let n = rows * cols;
    let dim = 1usize;
    let data: Vec<f64> = (0..n).map(|i| ((i % 971) as f64) * 0.01 - 4.0).collect();
    for op in ["var", "std"] {
        let mut best = f64::INFINITY;
        let mut fp = 0u64;
        let mut gsum = 0.0f64;
        let mut samples = [0.0f64; 3];
        for _ in 0..7 {
            let mut s = FrankenTorchSession::new(ExecutionMode::Strict);
            let a = s
                .tensor_variable(data.clone(), vec![rows, cols], true)
                .unwrap();
            let t0 = Instant::now();
            let red = if op == "var" {
                s.tensor_var_dim(a, dim, 1).unwrap()
            } else {
                s.tensor_std_dim(a, dim, 1).unwrap()
            };
            let loss = s.tensor_sum(red).unwrap();
            s.tensor_backward(loss).unwrap();
            let ms = t0.elapsed().as_secs_f64() * 1e3;
            if ms < best {
                best = ms;
                let g = s.tensor_grad(a).unwrap().unwrap();
                fp = fingerprint(&g);
                gsum = g.iter().sum();
                samples = [g[0], g[n / 2], g[n - 1]];
            }
            std::hint::black_box(&s);
        }
        println!(
            "[FUSED(par)] {op}_dim(dim={dim},corr=1) fwd+bwd f64 [4096,4096]: {best:.2} ms | grad_fp=0x{fp:016x} grad_sum={gsum:.17e}"
        );
        println!(
            "        samples g[0]=0x{:016x} g[n/2]=0x{:016x} g[n-1]=0x{:016x}",
            samples[0].to_bits(),
            samples[1].to_bits(),
            samples[2].to_bits()
        );
    }
}

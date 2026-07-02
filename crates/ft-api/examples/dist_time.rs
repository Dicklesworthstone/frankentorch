//! Distribution throughput: FT parallel inverse-CDF transform vs torch (serial RNG).
//! frankentorch-randn-par. Bit-exact to serial (verified via A/B fingerprint).
use ft_api::FrankenTorchSession;
use ft_core::ExecutionMode;
use std::time::Instant;
fn main() {
    let n = 1usize << 24;
    for op in ["exponential", "cauchy", "weibull"] {
        let mut best = f64::INFINITY;
        for _ in 0..8 {
            let mut s = FrankenTorchSession::new(ExecutionMode::Strict);
            let t = Instant::now();
            let x = match op {
                "exponential" => s.tensor_exponential(1.0, vec![n], false).unwrap(),
                "cauchy" => s.tensor_cauchy(0.0, 1.0, vec![n], false).unwrap(),
                _ => s.tensor_weibull(1.0, 1.5, vec![n], false).unwrap(),
            };
            best = best.min(t.elapsed().as_secs_f64() * 1e3);
            std::hint::black_box(&s.tensor_values(x).unwrap());
        }
        println!("FT {op} [{n}] f64: {best:.3} ms");
    }
}

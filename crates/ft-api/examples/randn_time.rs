//! randn throughput: FT parallel Box-Muller transform vs torch (single-threaded RNG).
//! frankentorch-randn-par. Bit-exact to the serial path (verified via A/B fingerprint).
use ft_api::FrankenTorchSession;
use ft_core::ExecutionMode;
use std::time::Instant;
fn main() {
    let n = 1usize << 24; // 16.7M
    let mut best = f64::INFINITY;
    for _ in 0..10 {
        let mut s = FrankenTorchSession::new(ExecutionMode::Strict);
        let t = Instant::now();
        let x = s.tensor_randn(vec![n], false).unwrap();
        best = best.min(t.elapsed().as_secs_f64() * 1e3);
        std::hint::black_box(&s.tensor_values(x).unwrap());
    }
    println!("FT randn [{n}] f64: {best:.3} ms");
}

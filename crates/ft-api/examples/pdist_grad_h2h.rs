use ft_api::FrankenTorchSession;
use ft_core::ExecutionMode;
use std::time::Instant;
fn main() {
    let (n, m) = (512usize, 128usize);
    let data: Vec<f64> = (0..n*m).map(|i| ((i as u64).wrapping_mul(0x9e3779b97f4a7c15) >> 40) as f64 / (1u64<<24) as f64).collect();
    let threads = rayon::current_num_threads();
    let (mut bf, mut bb) = (f64::INFINITY, f64::INFINITY);
    for _ in 0..7 {
        let mut s = FrankenTorchSession::new(ExecutionMode::Strict);
        let x = s.tensor_variable(data.clone(), vec![n, m], true).unwrap();
        let t0 = Instant::now();
        let d = s.tensor_pdist(x, 2.0).unwrap();
        let loss = s.tensor_sum(d).unwrap();
        let t1 = Instant::now();
        let rep = s.tensor_backward(loss).unwrap();
        let _g = s.tensor_gradient(&rep, x).unwrap();
        let t2 = Instant::now();
        bf = bf.min((t1-t0).as_secs_f64()*1e3); bb = bb.min((t2-t1).as_secs_f64()*1e3);
        std::hint::black_box(&s);
    }
    println!("[pdist n={n} m={m} p=2] threads={threads}: fwd {bf:.2} bwd {bb:.2} fwd+bwd {:.2} ms", bf+bb);
}

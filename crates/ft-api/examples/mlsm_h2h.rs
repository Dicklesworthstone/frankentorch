use ft_api::FrankenTorchSession;
use ft_core::ExecutionMode;
use std::time::Instant;
fn main() {
    let (n, c) = (256usize, 1024usize);
    let input: Vec<f64> = (0..n*c).map(|i| ((i as u64).wrapping_mul(0x9e3779b97f4a7c15) >> 40) as f64 / (1u64<<24) as f64 * 6.0 - 3.0).collect();
    let target: Vec<f64> = (0..n*c).map(|i| ((i*7+1)%2) as f64).collect();
    let threads = rayon::current_num_threads();
    let (mut bf, mut bb) = (f64::INFINITY, f64::INFINITY);
    for _ in 0..7 {
        let mut s = FrankenTorchSession::new(ExecutionMode::Strict);
        let inp = s.tensor_variable(input.clone(), vec![n, c], true).unwrap();
        let tgt = s.tensor_variable(target.clone(), vec![n, c], false).unwrap();
        let t0 = Instant::now();
        let loss = s.tensor_multilabel_soft_margin_loss(inp, tgt, None, "mean").unwrap();
        let t1 = Instant::now();
        let rep = s.tensor_backward(loss).unwrap();
        let _g = s.tensor_gradient(&rep, inp).unwrap();
        let t2 = Instant::now();
        bf = bf.min((t1-t0).as_secs_f64()*1e3); bb = bb.min((t2-t1).as_secs_f64()*1e3);
        std::hint::black_box(&s);
    }
    println!("[mlsm n={n} c={c}] threads={threads}: fwd {bf:.2} bwd {bb:.2} fwd+bwd {:.2} ms", bf+bb);
}

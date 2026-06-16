//! A/B for the fused parallel Adamax step (frankentorch-optpar). Adam-style fused
//! in-place callback (no grad/param clones + write-back); per-element m EMA + u
//! inf-norm + step fan over rayon above OPTIM_PARALLEL_THRESHOLD. Bit-exact (ft-optim
//! lib tests). 1-thread vs all-cores rayon pools, op-only.
//!   cargo run -q --release -p ft-optim --example adamax_par_ab
use ft_api::FrankenTorchSession;
use ft_core::ExecutionMode;
use ft_optim::{Adamax, Optimizer};
use rayon::ThreadPool;
use std::time::Instant;

fn time_step(pool: &ThreadPool, n: usize) -> f64 {
    let mut best = f64::INFINITY;
    for _ in 0..20 {
        let mut s = FrankenTorchSession::new(ExecutionMode::Strict);
        let x = s.tensor_variable(vec![0.5_f64; n], vec![n], true).unwrap();
        let sq = s.tensor_mul(x, x).unwrap();
        let loss = s.tensor_sum(sq).unwrap();
        let report = s.tensor_backward(loss).unwrap();
        let mut opt = Adamax::new(vec![x], 0.01);
        let t = Instant::now();
        pool.install(|| {
            opt.step(&mut s, &report).unwrap();
        });
        best = best.min(t.elapsed().as_secs_f64() * 1e3);
    }
    best
}

fn main() {
    let pool1 = rayon::ThreadPoolBuilder::new().num_threads(1).build().unwrap();
    let pooln = rayon::ThreadPoolBuilder::new().build().unwrap();
    let nt = pooln.current_num_threads();
    let n = 1usize << 21; // 2M params
    let s1 = time_step(&pool1, n);
    let sn = time_step(&pooln, n);
    eprintln!(
        "adamax step [{n} params]: serial {s1:.3} ms / parallel({nt}t) {sn:.3} ms / {:.2}x",
        s1 / sn
    );
}

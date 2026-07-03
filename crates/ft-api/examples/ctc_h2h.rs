// ctc_loss batch-parallel A/B: run this binary with RAYON_NUM_THREADS=1 (serial-
// equivalent) vs many threads (parallel) in the same process/worker to measure the
// batch-parallelization speedup. Inputs are materialized BEFORE Instant::now() so
// the tensor_variable copy never lands in the timed region.
use ft_api::FrankenTorchSession;
use ft_core::ExecutionMode;
use std::time::Instant;

fn main() {
    let t_len = 120usize; // timesteps
    let n = 32usize; // batch
    let c = 60usize; // classes (incl blank=0)
    let tgt_len = 24usize; // target length per sample

    // Deterministic pseudo-random log-prob-ish values.
    let lp: Vec<f64> = (0..t_len * n * c)
        .map(|i| {
            let x = (i as u64).wrapping_mul(0x9e3779b97f4a7c15).wrapping_add(11);
            ((x >> 40) as f64 / (1u64 << 24) as f64) - 4.0
        })
        .collect();
    // Concatenated targets (labels in 1..c), lengths.
    let targets: Vec<f64> = (0..n * tgt_len)
        .map(|i| (1 + (i % (c - 1))) as f64)
        .collect();
    let in_lens: Vec<f64> = vec![t_len as f64; n];
    let tgt_lens: Vec<f64> = vec![tgt_len as f64; n];

    let threads = rayon::current_num_threads();
    let mut best_fwd = f64::INFINITY;
    let mut best_bwd = f64::INFINITY;
    for _ in 0..7 {
        let mut s = FrankenTorchSession::new(ExecutionMode::Strict);
        let log_probs = s.tensor_variable(lp.clone(), vec![t_len, n, c], true).unwrap();
        let tg = s.tensor_variable(targets.clone(), vec![n * tgt_len], false).unwrap();
        let il = s.tensor_variable(in_lens.clone(), vec![n], false).unwrap();
        let tl = s.tensor_variable(tgt_lens.clone(), vec![n], false).unwrap();
        let t0 = Instant::now();
        let loss = s
            .tensor_ctc_loss(log_probs, tg, il, tl, 0, "mean", false)
            .unwrap();
        let t1 = Instant::now();
        let report = s.tensor_backward(loss).unwrap();
        let _g = s.tensor_gradient(&report, log_probs).unwrap();
        let t2 = Instant::now();
        best_fwd = best_fwd.min((t1 - t0).as_secs_f64() * 1e3);
        best_bwd = best_bwd.min((t2 - t1).as_secs_f64() * 1e3);
        std::hint::black_box(&s);
    }
    println!(
        "[ctc T={t_len} N={n} C={c} L={tgt_len}] threads={threads}: fwd {best_fwd:.2} ms  bwd {best_bwd:.2} ms  fwd+bwd {:.2} ms",
        best_fwd + best_bwd
    );
}

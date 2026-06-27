//! f64 SDPA training-step tape-retention A/B (cc). The gauntlet sdpa_grad bench (kgs4.113)
//! reuses ONE session across steps; FT's tape retains every step's intermediates (gmuml), so
//! later steps slow down. Converting the f64 SDPA grad path (scaled_dot_product_attention,
//! the 5083 entry) to borrowed-inputs drops 3 full q/k/v clones per step from the retained
//! ctx. This harness measures the worker-immune within-process early-vs-late ratio + total on
//! a reused session — the borrowed path should degrade less and run faster overall.
//!
//! Run: cargo run --release -p ft-api --example sdpa_f64_train_retention_ab

use std::time::Instant;

use ft_api::FrankenTorchSession;
use ft_core::ExecutionMode;

const BH: usize = 16;
const SEQ: usize = 512;
const D: usize = 64;

fn seq_vals(n: usize, shift: f64) -> Vec<f64> {
    (0..n)
        .map(|i| (((i as f64) * 0.017 + shift).sin()) * 0.2)
        .collect()
}

fn main() {
    let steps: usize = std::env::var("STEPS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(200);
    let total = BH * SEQ * D;
    let shape = vec![BH, SEQ, D];
    let mut s = FrankenTorchSession::new(ExecutionMode::Strict);
    // Param leaves created ONCE (requires_grad), as in a real training loop.
    let q = s
        .tensor_variable(seq_vals(total, 0.0), shape.clone(), true)
        .unwrap();
    let k = s
        .tensor_variable(seq_vals(total, 1.0), shape.clone(), true)
        .unwrap();
    let v = s
        .tensor_variable(seq_vals(total, 2.0), shape, true)
        .unwrap();

    let mut times = Vec::with_capacity(steps);
    let mut chk = 0.0;
    for _ in 0..steps {
        let t = Instant::now();
        let out = s
            .scaled_dot_product_attention(q, k, v, None, 0.0, false)
            .unwrap();
        let loss = s.tensor_sum(out).unwrap();
        let report = s.tensor_backward(loss).unwrap();
        chk = report.gradient(q).unwrap().iter().map(|x| x.abs()).sum();
        times.push(t.elapsed().as_secs_f64() * 1e3);
    }
    let avg = |lo: usize, hi: usize| times[lo..hi].iter().sum::<f64>() / (hi - lo) as f64;
    let early = avg(10, 30);
    let late = avg(steps - 30, steps - 10);
    let tot: f64 = times.iter().sum();
    println!("f64 SDPA train-step reused-session [{BH},{SEQ},{D}], {steps} steps:");
    println!(
        "  early[10..30] {early:.3} ms   late[{}..{}] {late:.3} ms   late/early {:.3}",
        steps - 30,
        steps - 10,
        late / early
    );
    println!("  total {tot:.1} ms   checksum {chk:.6e}");
}

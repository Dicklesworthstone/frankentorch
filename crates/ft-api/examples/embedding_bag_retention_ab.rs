//! embedding_bag (sum mode) training-step A/B (cc). The f64 embedding_bag grad path saved the
//! ENTIRE [num_embeddings, embedding_dim] weight table into ctx every forward, but the sum/mean
//! backward only needs indices/offsets/grad — so for the common modes that was a dead clone of
//! the whole embedding table per step. Gating the save on mode=="max" removes it. This harness
//! times the f64 sum-mode train step (reused session) so the clone-elision shows up.
//!
//! Run: cargo run --release -p ft-api --example embedding_bag_retention_ab

use std::time::Instant;

use ft_api::FrankenTorchSession;
use ft_core::ExecutionMode;

const VOCAB: usize = 50_000;
const DIM: usize = 128;
const NUM_BAGS: usize = 256;
const BAG: usize = 12;

fn main() {
    let steps: usize = std::env::var("STEPS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(150);
    let mut s = FrankenTorchSession::new(ExecutionMode::Strict);
    let wvals: Vec<f64> = (0..VOCAB * DIM)
        .map(|i| ((i as f64) * 0.0001).sin() * 0.1)
        .collect();
    let weight = s.tensor_variable(wvals, vec![VOCAB, DIM], true).unwrap();
    // input indices [NUM_BAGS*BAG], offsets [NUM_BAGS]
    let input: Vec<f64> = (0..NUM_BAGS * BAG)
        .map(|i| ((i * 2654435761) % VOCAB) as f64)
        .collect();
    let offsets: Vec<f64> = (0..NUM_BAGS).map(|b| (b * BAG) as f64).collect();
    let idx = s
        .tensor_variable(input, vec![NUM_BAGS * BAG], false)
        .unwrap();
    let off = s.tensor_variable(offsets, vec![NUM_BAGS], false).unwrap();

    let mut times = Vec::with_capacity(steps);
    let mut chk = 0.0;
    for _ in 0..steps {
        let t = Instant::now();
        let out = s.tensor_embedding_bag(idx, weight, off, "sum").unwrap();
        let loss = s.tensor_sum(out).unwrap();
        let report = s.tensor_backward(loss).unwrap();
        chk = report
            .gradient(weight)
            .unwrap()
            .iter()
            .map(|x| x.abs())
            .sum();
        times.push(t.elapsed().as_secs_f64() * 1e3);
    }
    let avg = |lo: usize, hi: usize| times[lo..hi].iter().sum::<f64>() / (hi - lo) as f64;
    let early = avg(5, 20);
    let late = avg(steps - 20, steps - 5);
    let tot: f64 = times.iter().sum();
    println!(
        "embedding_bag sum train [vocab={VOCAB},dim={DIM},bags={NUM_BAGS},bag={BAG}], {steps} steps:"
    );
    println!(
        "  early {early:.3} ms   late {late:.3} ms   late/early {:.3}   total {tot:.1} ms   checksum {chk:.6e}",
        late / early
    );
}

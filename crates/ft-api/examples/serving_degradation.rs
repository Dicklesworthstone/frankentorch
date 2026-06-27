//! Reused-session serving degradation (BlackThrush): a real vs-PyTorch axis the
//! fresh-session gauntlet can't see. PyTorch frees tensors between inferences (flat
//! throughput); FT's TensorTape retains nodes (gmuml, index-based ids, no Drop) so a
//! long-running FT server's per-inference cost should grow as retained heap climbs.
//! Run N no-grad SDPA inferences in ONE session and watch per-iter time early vs late.
//!
//! Run: cargo run --release -p ft-api --example serving_degradation

use std::time::Instant;

use ft_api::FrankenTorchSession;
use ft_core::ExecutionMode;

const BH: usize = 16;
const SEQ: usize = 512;
const D: usize = 64;

fn vals(n: usize, shift: f64) -> Vec<f64> {
    (0..n)
        .map(|i| (((i as f64) * 0.017 + shift).sin()) * 0.2)
        .collect()
}

fn main() {
    let n: usize = std::env::var("N")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(150);
    let total = BH * SEQ * D;
    let q = vals(total, 0.0);
    let k = vals(total, 1.0);
    let v = vals(total, 2.0);
    let shape = vec![BH, SEQ, D];

    // ONE long-lived session (a server). Each inference creates fresh leaves + an SDPA
    // output node -> ~4 x 4MB = 16MB retained per inference (no-grad, never freed).
    let mut s = FrankenTorchSession::new(ExecutionMode::Strict);
    let mut per_iter = Vec::with_capacity(n);
    for _ in 0..n {
        let t = Instant::now();
        let qn = s.tensor_variable(q.clone(), shape.clone(), false).unwrap();
        let kn = s.tensor_variable(k.clone(), shape.clone(), false).unwrap();
        let vn = s.tensor_variable(v.clone(), shape.clone(), false).unwrap();
        let out = s
            .scaled_dot_product_attention(qn, kn, vn, None, 0.0, false)
            .unwrap();
        let _: f64 = s.tensor_values(out).unwrap().iter().sum();
        per_iter.push(t.elapsed().as_secs_f64() * 1e3);
    }

    let avg = |sl: &[f64]| sl.iter().sum::<f64>() / sl.len() as f64;
    let first10 = avg(&per_iter[..10.min(n)]);
    let last10 = avg(&per_iter[n.saturating_sub(10)..]);
    println!("reused-session SDPA serving [{BH},{SEQ},{D}], {n} inferences in ONE session:");
    println!("  iter[0..10] avg   : {first10:8.3} ms");
    println!(
        "  iter[{}..{}] avg : {:8.3} ms",
        n.saturating_sub(10),
        n,
        last10
    );
    println!("  retained heap est : ~{} MB ({n} x ~16MB)", n * 16);
    println!(
        "  degradation       : {:.2}x (last10/first10)",
        last10 / first10
    );
    if last10 > 1.5 * first10 {
        println!("  => SERVING DEGRADATION confirmed (gmuml tape-retention). PyTorch stays flat.");
        println!(
            "     Mitigation: compact the no-grad tape between inferences (compact_nograd_tensor_since / truncate_graph_to), or RAII handles."
        );
    } else {
        println!(
            "  => flat at this scale (~{} MB retained); gmuml threshold not reached.",
            n * 16
        );
    }
}

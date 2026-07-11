//! Does FT's no-grad op cost degrade as the session tape grows? (BlackThrush)
//! Tests the gmuml tape-retention hypothesis for the unexplained MHA composed-inference
//! overhead: MHA runs ~18 ops in ONE session; if each no-grad op degrades ~linearly with
//! tape size (nodes never freed), that compounds. Time the k-th op in a chain of N.
//!
//! Run: cargo run --release -p ft-api --example nograd_tape_degradation

use std::time::Instant;

use ft_api::FrankenTorchSession;
use ft_core::ExecutionMode;

const SZ: usize = 512 * 512; // 256K elem, ~2MB — small enough that tape-degradation dominates

fn main() {
    let n: usize = std::env::var("N")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(40);
    let base: Vec<f64> = (0..SZ).map(|i| ((i % 251) as f64) * 0.001 - 0.12).collect();

    // One long-lived session: each tensor_relu adds a node; tape grows to N+1.
    let mut s = FrankenTorchSession::new(ExecutionMode::Strict);
    let mut cur = s
        .tensor_variable(base.clone(), vec![512, 512], false)
        .unwrap();
    let mut per_op = Vec::with_capacity(n);
    for _ in 0..n {
        let t = Instant::now();
        cur = s.tensor_relu(cur).unwrap();
        per_op.push(t.elapsed().as_secs_f64() * 1e3);
    }

    // Control: the SAME op in a FRESH session each time (tape stays size 1-2) — no growth.
    let mut fresh = Vec::with_capacity(n);
    for _ in 0..n {
        let mut s2 = FrankenTorchSession::new(ExecutionMode::Strict);
        let x = s2
            .tensor_variable(base.clone(), vec![512, 512], false)
            .unwrap();
        let t = Instant::now();
        let _ = s2.tensor_relu(x).unwrap();
        fresh.push(t.elapsed().as_secs_f64() * 1e3);
    }

    println!("no-grad relu [512,512], chain of {n} ops in ONE session vs fresh sessions:");
    println!("  one-session op[0]   : {:.3} ms", per_op[0]);
    println!("  one-session op[{}]  : {:.3} ms", n / 2, per_op[n / 2]);
    println!("  one-session op[{}]  : {:.3} ms", n - 1, per_op[n - 1]);
    let chain_total: f64 = per_op.iter().sum();
    println!(
        "  one-session TOTAL   : {:.3} ms  (sum of {n} ops)",
        chain_total
    );
    let fresh_med = {
        let mut f = fresh.clone();
        f.sort_by(|a, b| a.partial_cmp(b).unwrap());
        f[f.len() / 2]
    };
    println!("  fresh-session median: {fresh_med:.3} ms/op (no tape growth)");
    let slope = (per_op[n - 1] - per_op[0]) / (n as f64 - 1.0);
    println!(
        "  degradation slope   : {:.4} ms/op-added   ({}x op[last]/op[0])",
        slope,
        if per_op[0] > 0.0 {
            per_op[n - 1] / per_op[0]
        } else {
            0.0
        }
    );
    if per_op[n - 1] > 2.0 * per_op[0].max(fresh_med) {
        println!("  => CONFIRMED: no-grad op cost grows with tape size (gmuml tape-retention).");
    } else {
        println!("  => no significant tape-growth degradation.");
    }
}

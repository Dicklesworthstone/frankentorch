//! Fair f64 no-grad SDPA inference head-to-head vs PyTorch (cc).
//! kgs4.153 measured the unmasked f64 flash win through `sdpa_inference_headtohead`, whose
//! FT loop re-creates q/k/v + reads the output every iter (its own PHASES row shows
//! tensor_variable ~4.6ms + read ~2.4ms ≫ the ~4.5ms op), so the recorded 2.4-2.7x is
//! understated. This benchmark matches PyTorch's harness: q/k/v built ONCE, time only
//! `scaled_dot_product_attention` + `tensor_values` read per iter. Also a direct empirical
//! check of the session-reuse tape-retention concern (reused session across iters).
//!
//! Run: PYTORCH_PYTHON=/path/to/python cargo run --release -p ft-api --example sdpa_f64_fair_inference

use std::process::Command;
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

fn bench_ft(causal: bool, iters: usize) -> (f64, f64) {
    let total = BH * SEQ * D;
    let shape = vec![BH, SEQ, D];
    let mut s = FrankenTorchSession::new(ExecutionMode::Strict);
    let q = s
        .tensor_variable(seq_vals(total, 0.0), shape.clone(), false)
        .unwrap();
    let k = s
        .tensor_variable(seq_vals(total, 1.0), shape.clone(), false)
        .unwrap();
    let v = s
        .tensor_variable(seq_vals(total, 2.0), shape, false)
        .unwrap();
    let op = |s: &mut FrankenTorchSession| -> f64 {
        let out = s
            .scaled_dot_product_attention(q, k, v, None, 0.0, causal)
            .unwrap();
        s.tensor_values(out).unwrap().iter().map(|x| x.abs()).sum()
    };
    for _ in 0..3 {
        let _ = op(&mut s);
    }
    let mut times = Vec::with_capacity(iters);
    let mut sum = 0.0;
    for _ in 0..iters {
        let t = Instant::now();
        sum = op(&mut s);
        times.push(t.elapsed().as_secs_f64() * 1e3);
    }
    times.sort_by(|a, b| a.partial_cmp(b).unwrap());
    (times[0], sum)
}

const PY: &str = r#"
import os, time
import torch
import torch.nn.functional as F
BH, SEQ, D = 16, 512, 64
total = BH*SEQ*D
causal = os.environ.get("FT_CAUSAL","0") == "1"
def dv(shift):
    return (torch.arange(total, dtype=torch.float64).mul_(0.017).add_(shift).sin_().mul_(0.2).reshape(BH,SEQ,D))
iters = int(os.environ.get("FT_GAUNTLET_ITERS","25"))
torch.set_num_threads(int(os.environ.get("FT_TORCH_THREADS","8")))
q, k, v = dv(0.0), dv(1.0), dv(2.0)
with torch.no_grad():
    chk = 0.0
    for _ in range(3):
        chk = F.scaled_dot_product_attention(q,k,v,is_causal=causal).abs().sum().item()
    ts=[]
    for _ in range(iters):
        t0=time.perf_counter()
        F.scaled_dot_product_attention(q,k,v,is_causal=causal).abs().sum().item()
        ts.append((time.perf_counter()-t0)*1e3)
ts.sort()
print("ELAPSED_MS", ts[0]); print("CHECKSUM", chk)
"#;

fn report(label: &str, causal: bool, iters: usize) {
    let (ft, ft_sum) = bench_ft(causal, iters);
    print!("  {label:12} FT {ft:8.3} ms");
    let python = std::env::var("PYTORCH_PYTHON").unwrap_or_else(|_| "python3".to_string());
    match Command::new(&python)
        .arg("-c")
        .arg(PY)
        .env("FT_GAUNTLET_ITERS", iters.to_string())
        .env("FT_CAUSAL", if causal { "1" } else { "0" })
        .output()
    {
        Ok(o) if o.status.success() => {
            let s = String::from_utf8_lossy(&o.stdout);
            let p: Option<f64> = s.lines().find_map(|l| {
                l.strip_prefix("ELAPSED_MS ")
                    .and_then(|v| v.trim().parse().ok())
            });
            let ps: Option<f64> = s.lines().find_map(|l| {
                l.strip_prefix("CHECKSUM ")
                    .and_then(|v| v.trim().parse().ok())
            });
            match (p, ps) {
                (Some(p), Some(ps)) => {
                    let rel = (ft_sum - ps).abs() / (ps.abs() + 1e-12);
                    let r = p / ft;
                    let verdict = if r >= 1.0 {
                        format!("FT {r:.2}x FASTER")
                    } else {
                        format!("FT {:.2}x slower", 1.0 / r)
                    };
                    println!(
                        "   PyTorch {p:8.3} ms  => {verdict}  (rel-diff {rel:.2e} {})",
                        if rel < 1e-9 { "MATCH" } else { "MISMATCH!" }
                    );
                }
                _ => println!("   PyTorch (parse failed)"),
            }
        }
        Ok(o) => eprintln!("pytorch failed: {}", String::from_utf8_lossy(&o.stderr)),
        Err(e) => eprintln!("launch failed: {e}"),
    }
}

fn main() {
    let iters: usize = std::env::var("ITERS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(25);
    println!("f64 SDPA INFERENCE (no-grad, fair op+read) [{BH},{SEQ},{D}], {iters} iters MIN:");
    report("non-causal", false, iters);
    report("causal", true, iters);
}

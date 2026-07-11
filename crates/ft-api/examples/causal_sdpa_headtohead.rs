//! Causal-SDPA head-to-head vs PyTorch (BlackThrush). The non-causal SDPA win
//! (~2.3x, kgs4.113) came from FT's fused flash-attn beating PyTorch's unfused CPU
//! f64 SDPA. Causal attention is THE transformer use-case — this checks whether the
//! win extends to is_causal=true (PyTorch CPU causal SDPA is likely the same unfused
//! math path + a triangular mask).
//!
//! Run: PYTORCH_PYTHON=/path/to/python cargo run --release -p ft-api --example causal_sdpa_headtohead

use std::process::Command;
use std::time::Instant;

use ft_api::FrankenTorchSession;
use ft_core::ExecutionMode;

const BH: usize = 16;
const SEQ: usize = 512;
const D: usize = 64;

fn seq(n: usize, shift: f64) -> Vec<f64> {
    (0..n)
        .map(|i| (((i as f64) * 0.017 + shift).sin()) * 0.2)
        .collect()
}

fn ft_causal_sdpa_step() -> f64 {
    let total = BH * SEQ * D;
    let shape = vec![BH, SEQ, D];
    let mut s = FrankenTorchSession::new(ExecutionMode::Strict);
    let q = s
        .tensor_variable(seq(total, 0.0), shape.clone(), true)
        .unwrap();
    let k = s
        .tensor_variable(seq(total, 1.0), shape.clone(), true)
        .unwrap();
    let v = s.tensor_variable(seq(total, 2.0), shape, true).unwrap();
    // is_causal = true
    let out = s
        .scaled_dot_product_attention(q, k, v, None, 0.0, true)
        .unwrap();
    let loss = s.tensor_sum(out).unwrap();
    let report = s.tensor_backward(loss).unwrap();
    report.gradient(q).unwrap().iter().sum()
}

const PY: &str = r#"
import os, time
import torch
import torch.nn.functional as F
BH, SEQ, D = 16, 512, 64
total = BH*SEQ*D
def dv(shift):
    return (torch.arange(total, dtype=torch.float64).mul_(0.017).add_(shift).sin_().mul_(0.2).reshape(BH,SEQ,D))
iters = int(os.environ.get("FT_GAUNTLET_ITERS","20"))
torch.set_num_threads(int(os.environ.get("FT_TORCH_THREADS","8")))
bq, bk, bv = dv(0.0), dv(1.0), dv(2.0)
# warmup
for _ in range(3):
    q = bq.detach().clone().requires_grad_(True); k = bk.detach().clone().requires_grad_(True); v = bv.detach().clone().requires_grad_(True)
    F.scaled_dot_product_attention(q,k,v,is_causal=True).sum().backward()
ts=[]
for _ in range(iters):
    q = bq.detach().clone().requires_grad_(True); k = bk.detach().clone().requires_grad_(True); v = bv.detach().clone().requires_grad_(True)
    t0=time.perf_counter()
    F.scaled_dot_product_attention(q,k,v,is_causal=True).sum().backward()
    ts.append((time.perf_counter()-t0)*1e3)
ts.sort()
print("ELAPSED_MS", ts[len(ts)//2])
"#;

fn bench_ft(iters: usize) -> f64 {
    let mut times = Vec::with_capacity(iters);
    for _ in 0..3 {
        let _ = ft_causal_sdpa_step();
    }
    for _ in 0..iters {
        let t = Instant::now();
        let _ = ft_causal_sdpa_step();
        times.push(t.elapsed().as_secs_f64() * 1e3);
    }
    times.sort_by(|a, b| a.partial_cmp(b).unwrap());
    times[times.len() / 2]
}

fn main() {
    let iters: usize = std::env::var("ITERS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(20);
    let ft_ms = bench_ft(iters);

    let python = std::env::var("PYTORCH_PYTHON").unwrap_or_else(|_| "python3".to_string());
    let out = Command::new(&python)
        .arg("-c")
        .arg(PY)
        .env("FT_GAUNTLET_ITERS", iters.to_string())
        .output();

    let py_ms = match out {
        Ok(o) if o.status.success() => {
            let s = String::from_utf8_lossy(&o.stdout);
            s.lines().find_map(|l| {
                l.strip_prefix("ELAPSED_MS ")
                    .and_then(|v| v.trim().parse::<f64>().ok())
            })
        }
        Ok(o) => {
            eprintln!("pytorch failed: {}", String::from_utf8_lossy(&o.stderr));
            None
        }
        Err(e) => {
            eprintln!("could not launch python ({python}): {e}");
            None
        }
    };

    println!("causal SDPA [{BH},{SEQ},{D}] f64 train step, {iters} iters median:");
    println!("  FrankenTorch : {ft_ms:8.3} ms");
    match py_ms {
        Some(p) => {
            println!("  PyTorch      : {p:8.3} ms");
            let r = p / ft_ms;
            if r >= 1.0 {
                println!("  => FrankenTorch is {r:.2}x FASTER (causal SDPA)");
            } else {
                println!("  => FrankenTorch is {:.2}x slower (causal SDPA)", 1.0 / r);
            }
        }
        None => {
            println!("  PyTorch      : (unavailable — set PYTORCH_PYTHON to a torch interpreter)")
        }
    }
}

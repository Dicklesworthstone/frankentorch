//! f32 SDPA head-to-head vs PyTorch (BlackThrush). The f64 SDPA wins (~2.3x non-causal,
//! ~1.24-2x causal) come from PyTorch's CPU SDPA having NO fused f64 path (math fallback).
//! PyTorch's CPU flash-attn kernel DOES cover f32 — so this checks whether the win is
//! f64-specific (expected FT loss/tie on f32) or general. f32 is the common inference dtype.
//!
//! Run: PYTORCH_PYTHON=/path/to/python cargo run --release -p ft-api --example sdpa_f32_headtohead

use std::process::Command;
use std::time::Instant;

use ft_api::FrankenTorchSession;
use ft_core::ExecutionMode;

const BH: usize = 16;
const SEQ: usize = 512;
const D: usize = 64;

fn seq_f32(n: usize, shift: f32) -> Vec<f32> {
    (0..n)
        .map(|i| (((i as f32) * 0.017 + shift).sin()) * 0.2)
        .collect()
}

fn ft_f32_sdpa_step(causal: bool) -> f64 {
    let total = BH * SEQ * D;
    let shape = vec![BH, SEQ, D];
    let mut s = FrankenTorchSession::new(ExecutionMode::Strict);
    let q = s
        .tensor_variable_f32(seq_f32(total, 0.0), shape.clone(), true)
        .unwrap();
    let k = s
        .tensor_variable_f32(seq_f32(total, 1.0), shape.clone(), true)
        .unwrap();
    let v = s
        .tensor_variable_f32(seq_f32(total, 2.0), shape, true)
        .unwrap();
    let out = s
        .scaled_dot_product_attention(q, k, v, None, 0.0, causal)
        .unwrap();
    let loss = s.tensor_sum(out).unwrap();
    let report = s.tensor_backward(loss).unwrap();
    report.gradient(q).unwrap().iter().map(|x| x.abs()).sum()
}

const PY: &str = r#"
import os, time
import torch
import torch.nn.functional as F
BH, SEQ, D = 16, 512, 64
total = BH*SEQ*D
causal = os.environ.get("FT_CAUSAL","0") == "1"
def dv(shift):
    return (torch.arange(total, dtype=torch.float32).mul_(0.017).add_(shift).sin_().mul_(0.2).reshape(BH,SEQ,D))
iters = int(os.environ.get("FT_GAUNTLET_ITERS","20"))
torch.set_num_threads(int(os.environ.get("FT_TORCH_THREADS","8")))
bq, bk, bv = dv(0.0), dv(1.0), dv(2.0)
for _ in range(3):
    q = bq.detach().clone().requires_grad_(True); k = bk.detach().clone().requires_grad_(True); v = bv.detach().clone().requires_grad_(True)
    F.scaled_dot_product_attention(q,k,v,is_causal=causal).sum().backward()
ts=[]
for _ in range(iters):
    q = bq.detach().clone().requires_grad_(True); k = bk.detach().clone().requires_grad_(True); v = bv.detach().clone().requires_grad_(True)
    t0=time.perf_counter()
    F.scaled_dot_product_attention(q,k,v,is_causal=causal).sum().backward()
    ts.append((time.perf_counter()-t0)*1e3)
ts.sort()
print("ELAPSED_MS", ts[len(ts)//2])
"#;

fn bench_ft(causal: bool, iters: usize) -> f64 {
    for _ in 0..3 {
        let _ = ft_f32_sdpa_step(causal);
    }
    let mut times = Vec::with_capacity(iters);
    for _ in 0..iters {
        let t = Instant::now();
        let _ = ft_f32_sdpa_step(causal);
        times.push(t.elapsed().as_secs_f64() * 1e3);
    }
    times.sort_by(|a, b| a.partial_cmp(b).unwrap());
    times[times.len() / 2]
}

fn py_ms(causal: bool, iters: usize) -> Option<f64> {
    let python = std::env::var("PYTORCH_PYTHON").unwrap_or_else(|_| "python3".to_string());
    let out = Command::new(&python)
        .arg("-c")
        .arg(PY)
        .env("FT_GAUNTLET_ITERS", iters.to_string())
        .env("FT_CAUSAL", if causal { "1" } else { "0" })
        .output()
        .ok()?;
    if !out.status.success() {
        eprintln!("pytorch failed: {}", String::from_utf8_lossy(&out.stderr));
        return None;
    }
    String::from_utf8_lossy(&out.stdout).lines().find_map(|l| {
        l.strip_prefix("ELAPSED_MS ")
            .and_then(|v| v.trim().parse::<f64>().ok())
    })
}

fn report(label: &str, causal: bool, iters: usize) {
    let ft = bench_ft(causal, iters);
    print!("  {label:18} FT {ft:8.3} ms");
    match py_ms(causal, iters) {
        Some(p) => {
            let r = p / ft;
            if r >= 1.0 {
                println!("   PyTorch {p:8.3} ms  => FT {r:.2}x FASTER");
            } else {
                println!("   PyTorch {p:8.3} ms  => FT {:.2}x slower", 1.0 / r);
            }
        }
        None => println!("   PyTorch (unavailable)"),
    }
}

fn main() {
    let iters: usize = std::env::var("ITERS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(20);
    println!("f32 SDPA [{BH},{SEQ},{D}] train step, {iters} iters median:");
    report("non-causal", false, iters);
    report("causal", true, iters);
}

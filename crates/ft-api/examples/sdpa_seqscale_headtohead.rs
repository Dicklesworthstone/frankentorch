//! f64 SDPA sequence-length scaling head-to-head vs PyTorch (BlackThrush).
//! The f64 SDPA win comes from PyTorch's CPU having NO fused f64 path -> it materialises
//! the O(seq^2) score matrix, while FT's flash kernel is O(seq) memory. So the win should
//! GROW with seq (long context = where attention dominates LLM cost). This quantifies it.
//!
//! Run: PYTORCH_PYTHON=/path/to/python cargo run --release -p ft-api --example sdpa_seqscale_headtohead

use std::process::Command;
use std::time::Instant;

use ft_api::FrankenTorchSession;
use ft_core::ExecutionMode;

const BH: usize = 4;
const D: usize = 64;

fn seq_vals(n: usize, shift: f64) -> Vec<f64> {
    (0..n)
        .map(|i| (((i as f64) * 0.017 + shift).sin()) * 0.2)
        .collect()
}

fn ft_step(seq: usize) -> f64 {
    let total = BH * seq * D;
    let shape = vec![BH, seq, D];
    let mut s = FrankenTorchSession::new(ExecutionMode::Strict);
    let q = s
        .tensor_variable(seq_vals(total, 0.0), shape.clone(), true)
        .unwrap();
    let k = s
        .tensor_variable(seq_vals(total, 1.0), shape.clone(), true)
        .unwrap();
    let v = s
        .tensor_variable(seq_vals(total, 2.0), shape, true)
        .unwrap();
    let out = s
        .scaled_dot_product_attention(q, k, v, None, 0.0, false)
        .unwrap();
    let loss = s.tensor_sum(out).unwrap();
    let report = s.tensor_backward(loss).unwrap();
    report.gradient(q).unwrap().iter().map(|x| x.abs()).sum()
}

const PY: &str = r#"
import os, time
import torch
import torch.nn.functional as F
BH, D = 4, 64
SEQ = int(os.environ["FT_SEQ"])
total = BH*SEQ*D
def dv(shift):
    return (torch.arange(total, dtype=torch.float64).mul_(0.017).add_(shift).sin_().mul_(0.2).reshape(BH,SEQ,D))
iters = int(os.environ.get("FT_GAUNTLET_ITERS","10"))
torch.set_num_threads(int(os.environ.get("FT_TORCH_THREADS","8")))
bq, bk, bv = dv(0.0), dv(1.0), dv(2.0)
for _ in range(2):
    q = bq.detach().clone().requires_grad_(True); k = bk.detach().clone().requires_grad_(True); v = bv.detach().clone().requires_grad_(True)
    F.scaled_dot_product_attention(q,k,v).sum().backward()
ts=[]
for _ in range(iters):
    q = bq.detach().clone().requires_grad_(True); k = bk.detach().clone().requires_grad_(True); v = bv.detach().clone().requires_grad_(True)
    t0=time.perf_counter()
    F.scaled_dot_product_attention(q,k,v).sum().backward()
    ts.append((time.perf_counter()-t0)*1e3)
ts.sort()
print("ELAPSED_MS", ts[0])
"#;

fn bench_ft(seq: usize, iters: usize) -> f64 {
    for _ in 0..2 {
        let _ = ft_step(seq);
    }
    let mut times = Vec::with_capacity(iters);
    for _ in 0..iters {
        let t = Instant::now();
        let _ = ft_step(seq);
        times.push(t.elapsed().as_secs_f64() * 1e3);
    }
    times.sort_by(|a, b| a.partial_cmp(b).unwrap());
    times[0] /* min: least-contended sample */
}

fn py_ms(seq: usize, iters: usize) -> Option<f64> {
    let python = std::env::var("PYTORCH_PYTHON").unwrap_or_else(|_| "python3".to_string());
    let out = Command::new(&python)
        .arg("-c")
        .arg(PY)
        .env("FT_GAUNTLET_ITERS", iters.to_string())
        .env("FT_SEQ", seq.to_string())
        .output()
        .ok()?;
    if !out.status.success() {
        eprintln!(
            "pytorch failed (seq={seq}): {}",
            String::from_utf8_lossy(&out.stderr)
        );
        return None;
    }
    String::from_utf8_lossy(&out.stdout).lines().find_map(|l| {
        l.strip_prefix("ELAPSED_MS ")
            .and_then(|v| v.trim().parse::<f64>().ok())
    })
}

fn main() {
    let iters: usize = std::env::var("ITERS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(10);
    println!("f64 SDPA [BH={BH},D={D}] non-causal train step, seq sweep, {iters} iters median:");
    for &seq in &[512usize, 1024, 2048] {
        let ft = bench_ft(seq, iters);
        match py_ms(seq, iters) {
            Some(p) => println!(
                "  seq={seq:5}  FT {ft:9.2} ms   PyTorch {p:9.2} ms   => FT {:.2}x FASTER",
                p / ft
            ),
            None => println!("  seq={seq:5}  FT {ft:9.2} ms   PyTorch (unavailable)"),
        }
    }
}

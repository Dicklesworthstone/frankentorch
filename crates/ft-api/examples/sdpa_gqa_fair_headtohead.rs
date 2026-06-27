//! Fair masked f64 GQA SDPA head-to-head vs PyTorch (cc).
//! kgs4.151 measured GQA through a harness that re-created q/k/v + read the output INSIDE the
//! timed loop, while PyTorch reuses pre-built tensors — that ~3ms create/read overhead made
//! the grouped flash kernel look like a 32t loss. This benchmark matches PyTorch's harness:
//! build q/k/v ONCE, time only `tensor_scaled_dot_product_attention_gqa` + output read per
//! iter (PyTorch builds qg/kg/vg once and times `F.sdpa(...,enable_gqa=True).abs().sum()`).
//!
//! Run: PYTORCH_PYTHON=/path/to/python cargo run --release -p ft-api --example sdpa_gqa_fair_headtohead

use std::process::Command;
use std::time::Instant;

use ft_api::FrankenTorchSession;
use ft_core::ExecutionMode;

const B: usize = 2;
const HQ: usize = 8;
const HKV: usize = 2;
const SEQ: usize = 512;
const D: usize = 64;

fn seq_vals(n: usize, shift: f64) -> Vec<f64> {
    (0..n)
        .map(|i| (((i as f64) * 0.017 + shift).sin()) * 0.2)
        .collect()
}
fn mask_vals() -> Vec<f64> {
    (0..SEQ * SEQ)
        .map(|idx| if idx % 3 == 0 { -0.5 } else { 0.0 })
        .collect()
}

// Inputs built ONCE; timed region = GQA op + f64 output read, matching PyTorch's harness.
fn bench_ft(iters: usize) -> (f64, f64) {
    let q_total = B * HQ * SEQ * D;
    let kv_total = B * HKV * SEQ * D;
    let mut s = FrankenTorchSession::new(ExecutionMode::Strict);
    let q = s
        .tensor_variable(seq_vals(q_total, 0.0), vec![B, HQ, SEQ, D], false)
        .unwrap();
    let k = s
        .tensor_variable(seq_vals(kv_total, 1.0), vec![B, HKV, SEQ, D], false)
        .unwrap();
    let v = s
        .tensor_variable(seq_vals(kv_total, 2.0), vec![B, HKV, SEQ, D], false)
        .unwrap();
    let m = s
        .tensor_variable(mask_vals(), vec![SEQ, SEQ], false)
        .unwrap();
    let op = |s: &mut FrankenTorchSession| -> f64 {
        let out = s
            .tensor_scaled_dot_product_attention_gqa(q, k, v, Some(m), false, None, true)
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
B, HQ, HKV, SEQ, D = 2, 8, 2, 512, 64
def qv(shift):
    return (torch.arange(B*HQ*SEQ*D, dtype=torch.float64).mul_(0.017).add_(shift).sin_().mul_(0.2).reshape(B,HQ,SEQ,D))
def kvv(shift):
    return (torch.arange(B*HKV*SEQ*D, dtype=torch.float64).mul_(0.017).add_(shift).sin_().mul_(0.2).reshape(B,HKV,SEQ,D))
m = torch.tensor([(-0.5 if idx%3==0 else 0.0) for idx in range(SEQ*SEQ)], dtype=torch.float64).reshape(SEQ,SEQ)
iters = int(os.environ.get("FT_GAUNTLET_ITERS","25"))
torch.set_num_threads(int(os.environ.get("FT_TORCH_THREADS","8")))
q, k, v = qv(0.0), kvv(1.0), kvv(2.0)
with torch.no_grad():
    chk = 0.0
    for _ in range(3):
        chk = F.scaled_dot_product_attention(q,k,v,attn_mask=m,enable_gqa=True).abs().sum().item()
    ts=[]
    for _ in range(iters):
        t0=time.perf_counter()
        F.scaled_dot_product_attention(q,k,v,attn_mask=m,enable_gqa=True).abs().sum().item()
        ts.append((time.perf_counter()-t0)*1e3)
ts.sort()
print("ELAPSED_MS", ts[0])
print("CHECKSUM", chk)
"#;

fn main() {
    let iters: usize = std::env::var("ITERS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(25);
    println!(
        "fair masked f64 GQA SDPA [B{B},HQ{HQ},HKV{HKV},S{SEQ},D{D}], {iters} iters MIN (op+read only):"
    );
    let (ft, ft_sum) = bench_ft(iters);
    print!("  FrankenTorch GQA {ft:8.3} ms");
    let python = std::env::var("PYTORCH_PYTHON").unwrap_or_else(|_| "python3".to_string());
    match Command::new(&python)
        .arg("-c")
        .arg(PY)
        .env("FT_GAUNTLET_ITERS", iters.to_string())
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

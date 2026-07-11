//! 4-D SDPA head-to-head vs PyTorch (BlackThrush) — the STANDARD [B,H,seq,d] layout real
//! transformers use. The prior SDPA "wins" were all measured at the gauntlet's unusual 3-D
//! [bh,seq,d] shape where PyTorch f64 is anomalously slow (22.96ms); at 4-D PyTorch is ~5x
//! faster (4.53ms). This confirms FT LOSES real-world 4-D SDPA (FT folds to bh -> same kernel).
//! f64, no-grad (inference). frankentorch-sdpamask correction (2026-06-21at).
//!
//! Run: PYTORCH_PYTHON=/path/to/python cargo run --release -p ft-api --example sdpa_4d_headtohead

use std::process::Command;
use std::time::Instant;

use ft_api::FrankenTorchSession;
use ft_core::ExecutionMode;

const B: usize = 2;
const H: usize = 8;
const SEQ: usize = 512;
const D: usize = 64;

fn vals(n: usize, shift: f64) -> Vec<f64> {
    (0..n)
        .map(|i| (((i as f64) * 0.017 + shift).sin()) * 0.2)
        .collect()
}

fn ft_4d() -> (f64, f64) {
    let n = B * H * SEQ * D;
    let mut s = FrankenTorchSession::new(ExecutionMode::Strict);
    let q = s
        .tensor_variable(vals(n, 0.0), vec![B, H, SEQ, D], false)
        .unwrap();
    let k = s
        .tensor_variable(vals(n, 1.0), vec![B, H, SEQ, D], false)
        .unwrap();
    let v = s
        .tensor_variable(vals(n, 2.0), vec![B, H, SEQ, D], false)
        .unwrap();
    let t = Instant::now();
    let out = s
        .scaled_dot_product_attention(q, k, v, None, 0.0, false)
        .unwrap();
    let vals = s.tensor_values(out).unwrap();
    (
        vals.iter().map(|x| x.abs()).sum(),
        t.elapsed().as_secs_f64() * 1e3,
    )
}

const PY: &str = r#"
import os, time
import torch
import torch.nn.functional as F
B, H, SEQ, D = 2, 8, 512, 64
n = B*H*SEQ*D
def dv(s): return torch.arange(n, dtype=torch.float64).mul_(0.017).add_(s).sin_().mul_(0.2).reshape(B,H,SEQ,D)
iters = int(os.environ.get("FT_GAUNTLET_ITERS","15"))
torch.set_num_threads(int(os.environ.get("FT_TORCH_THREADS","8")))
q,k,v = dv(0.0),dv(1.0),dv(2.0)
with torch.no_grad():
    for _ in range(3): F.scaled_dot_product_attention(q,k,v).abs().sum().item()
    ts=[]
    for _ in range(iters):
        t=time.perf_counter(); F.scaled_dot_product_attention(q,k,v).abs().sum().item(); ts.append((time.perf_counter()-t)*1e3)
ts.sort(); print("ELAPSED_MS", ts[0])
"#;

fn main() {
    let iters: usize = std::env::var("ITERS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(15);
    for _ in 0..3 {
        let _ = ft_4d();
    }
    let mut times = Vec::with_capacity(iters);
    for _ in 0..iters {
        let (_, t) = ft_4d();
        times.push(t);
    }
    times.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let ft = times[0];
    let python = std::env::var("PYTORCH_PYTHON").unwrap_or_else(|_| "python3".to_string());
    let py = Command::new(&python)
        .arg("-c")
        .arg(PY)
        .env("FT_GAUNTLET_ITERS", iters.to_string())
        .output()
        .ok()
        .filter(|o| o.status.success())
        .and_then(|o| {
            String::from_utf8_lossy(&o.stdout).lines().find_map(|l| {
                l.strip_prefix("ELAPSED_MS ")
                    .and_then(|v| v.trim().parse::<f64>().ok())
            })
        });
    println!("4-D SDPA [B{B},H{H},{SEQ},{D}] (standard layout) f64 no-grad, {iters} iters MIN:");
    println!("  FrankenTorch : {ft:8.3} ms");
    match py {
        Some(p) => {
            let r = p / ft;
            if r >= 1.0 {
                println!("  PyTorch      : {p:8.3} ms  => FT {r:.2}x FASTER");
            } else {
                println!(
                    "  PyTorch      : {p:8.3} ms  => FT {:.2}x SLOWER (real-world 4-D: FT loses, vendor-walled)",
                    1.0 / r
                );
            }
        }
        None => println!("  PyTorch      : (unavailable)"),
    }
}

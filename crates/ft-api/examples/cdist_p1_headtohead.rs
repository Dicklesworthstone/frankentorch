//! cdist p=1 (manhattan pairwise distance) f64 forward head-to-head vs PyTorch (BlackThrush).
//! Probes a NEW op class (not attention): FT ships a fused powf-elided cdist p=1 kernel (a5kk8);
//! PyTorch's torch.cdist p=1 may materialise/parallelise differently. f64, no-grad (the common
//! distance-matrix use). Mechanism check per 2026-06-21ac: FT wins where its parallel+fused f64
//! path beats PyTorch's f64 path.
//!
//! Run: PYTORCH_PYTHON=/path/to/python cargo run --release -p ft-api --example cdist_p1_headtohead

use std::process::Command;
use std::time::Instant;

use ft_api::FrankenTorchSession;
use ft_core::ExecutionMode;

const N: usize = 1024;
const M: usize = 1024;
const D: usize = 128;

fn vals(n: usize, shift: f64) -> Vec<f64> {
    (0..n)
        .map(|i| (((i as f64) * 0.013 + shift).sin()) * 0.5)
        .collect()
}

fn ft_cdist_p1() -> f64 {
    let mut s = FrankenTorchSession::new(ExecutionMode::Strict);
    // no-grad forward (requires_grad = false) -> fused cdist path
    let x1 = s
        .tensor_variable(vals(N * D, 0.0), vec![N, D], false)
        .unwrap();
    let x2 = s
        .tensor_variable(vals(M * D, 1.0), vec![M, D], false)
        .unwrap();
    let out = s.tensor_cdist(x1, x2, 1.0).unwrap();
    // checksum to force materialization + validate
    s.tensor_values(out).unwrap().iter().sum()
}

const PY: &str = r#"
import os, time
import torch
N, M, D = 1024, 1024, 128
def vals(n, shift):
    return (torch.arange(n, dtype=torch.float64).mul_(0.013).add_(shift).sin_().mul_(0.5))
iters = int(os.environ.get("FT_GAUNTLET_ITERS","15"))
torch.set_num_threads(int(os.environ.get("FT_TORCH_THREADS","8")))
x1 = vals(N*D, 0.0).reshape(N, D)
x2 = vals(M*D, 1.0).reshape(M, D)
with torch.no_grad():
    for _ in range(3):
        torch.cdist(x1, x2, p=1.0).sum().item()
    ts=[]
    for _ in range(iters):
        t0=time.perf_counter()
        r = torch.cdist(x1, x2, p=1.0).sum().item()
        ts.append((time.perf_counter()-t0)*1e3)
ts.sort()
print("ELAPSED_MS", ts[0])
"#;

fn main() {
    let iters: usize = std::env::var("ITERS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(15);
    for _ in 0..3 {
        let _ = ft_cdist_p1();
    }
    let mut times = Vec::with_capacity(iters);
    let mut checksum = 0.0;
    for _ in 0..iters {
        let t = Instant::now();
        checksum = ft_cdist_p1();
        times.push(t.elapsed().as_secs_f64() * 1e3);
    }
    times.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let ft = times[0]; // min: least-contended

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

    println!(
        "cdist p=1 f64 [{N},{D}] x [{M},{D}] -> [{N},{M}] forward (no-grad), {iters} iters MIN:"
    );
    println!("  FrankenTorch : {ft:8.3} ms   (checksum {checksum:.4e})");
    match py {
        Some(p) => {
            let r = p / ft;
            if r >= 1.0 {
                println!("  PyTorch      : {p:8.3} ms   => FT {r:.2}x FASTER");
            } else {
                println!("  PyTorch      : {p:8.3} ms   => FT {:.2}x slower", 1.0 / r);
            }
        }
        None => println!("  PyTorch      : (unavailable)"),
    }
}

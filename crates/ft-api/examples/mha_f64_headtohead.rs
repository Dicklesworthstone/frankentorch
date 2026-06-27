//! Multi-head attention (MHA) f64 inference head-to-head vs PyTorch (BlackThrush).
//! The practical transformer block = in-proj (GEMM) + SDPA + out-proj (GEMM). FT's MHA
//! (need_weights=false) routes through the now clone-free, f64-WINNING SDPA internally; the
//! projections are tensor_matmul (matrixmultiply, parallel) vs PyTorch's MKL. Tests whether
//! the SDPA f64 win survives the full attention layer. f64, no-grad inference.
//!
//! Run: PYTORCH_PYTHON=/path/to/python cargo run --release -p ft-api --example mha_f64_headtohead

use std::process::Command;
use std::time::Instant;

use ft_api::FrankenTorchSession;
use ft_core::ExecutionMode;

const SEQ: usize = 256;
const BATCH: usize = 8;
const EMBED: usize = 512;
const HEADS: usize = 8;

fn vals(n: usize, shift: f64) -> Vec<f64> {
    (0..n)
        .map(|i| (((i as f64) * 0.011 + shift).sin()) * 0.1)
        .collect()
}

struct Inputs {
    q: Vec<f64>,
    k: Vec<f64>,
    v: Vec<f64>,
    inw: Vec<f64>,
    inb: Vec<f64>,
    outw: Vec<f64>,
    outb: Vec<f64>,
}

// Pre-computed (no sin()-generation inside the timed region — matches the gauntlet harness).
fn ft_mha(inp: &Inputs) -> f64 {
    let qkv_shape = vec![SEQ, BATCH, EMBED];
    let mut s = FrankenTorchSession::new(ExecutionMode::Strict);
    let q = s
        .tensor_variable(inp.q.clone(), qkv_shape.clone(), false)
        .unwrap();
    let k = s
        .tensor_variable(inp.k.clone(), qkv_shape.clone(), false)
        .unwrap();
    let v = s.tensor_variable(inp.v.clone(), qkv_shape, false).unwrap();
    let inw = s
        .tensor_variable(inp.inw.clone(), vec![3 * EMBED, EMBED], false)
        .unwrap();
    let inb = s
        .tensor_variable(inp.inb.clone(), vec![3 * EMBED], false)
        .unwrap();
    let outw = s
        .tensor_variable(inp.outw.clone(), vec![EMBED, EMBED], false)
        .unwrap();
    let outb = s
        .tensor_variable(inp.outb.clone(), vec![EMBED], false)
        .unwrap();
    let (out, _) = s
        .functional_multi_head_attention_forward(
            q,
            k,
            v,
            inw,
            Some(inb),
            outw,
            Some(outb),
            HEADS,
            0.0,
            None,
            None,
            false,
            false,
            false,
        )
        .unwrap();
    s.tensor_values(out).unwrap().iter().map(|x| x.abs()).sum()
}

const PY: &str = r#"
import os, time
import torch
import torch.nn as nn
SEQ, BATCH, EMBED, HEADS = 256, 8, 512, 8
iters = int(os.environ.get("FT_GAUNTLET_ITERS","12"))
torch.set_num_threads(int(os.environ.get("FT_TORCH_THREADS","8")))
torch.manual_seed(0)
mha = nn.MultiheadAttention(EMBED, HEADS, batch_first=False, dtype=torch.float64)
def dv(shift):
    n = SEQ*BATCH*EMBED
    return (torch.arange(n, dtype=torch.float64).mul_(0.011).add_(shift).sin_().mul_(0.1).reshape(SEQ,BATCH,EMBED))
q, k, v = dv(0.0), dv(1.0), dv(2.0)
with torch.no_grad():
    for _ in range(3):
        mha(q, k, v, need_weights=False)
    ts=[]
    for _ in range(iters):
        t0=time.perf_counter()
        o,_ = mha(q, k, v, need_weights=False)
        o.abs().sum().item()
        ts.append((time.perf_counter()-t0)*1e3)
ts.sort()
print("ELAPSED_MS", ts[0])
"#;

fn main() {
    let iters: usize = std::env::var("ITERS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(12);
    let n = SEQ * BATCH * EMBED;
    let inp = Inputs {
        q: vals(n, 0.0),
        k: vals(n, 1.0),
        v: vals(n, 2.0),
        inw: vals(3 * EMBED * EMBED, 0.3),
        inb: vals(3 * EMBED, 0.4),
        outw: vals(EMBED * EMBED, 0.5),
        outb: vals(EMBED, 0.6),
    };
    for _ in 0..3 {
        let _ = ft_mha(&inp);
    }
    let mut times = Vec::with_capacity(iters);
    for _ in 0..iters {
        let t = Instant::now();
        let _ = ft_mha(&inp);
        times.push(t.elapsed().as_secs_f64() * 1e3);
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

    println!(
        "MHA f64 inference (no-grad) [seq={SEQ},batch={BATCH},embed={EMBED},heads={HEADS}], {iters} iters MIN:"
    );
    println!("  FrankenTorch : {ft:8.3} ms");
    match py {
        Some(p) => {
            let r = p / ft;
            if r >= 1.0 {
                println!("  PyTorch      : {p:8.3} ms  => FT {r:.2}x FASTER");
            } else {
                println!("  PyTorch      : {p:8.3} ms  => FT {:.2}x slower", 1.0 / r);
            }
        }
        None => println!("  PyTorch      : (unavailable)"),
    }
}

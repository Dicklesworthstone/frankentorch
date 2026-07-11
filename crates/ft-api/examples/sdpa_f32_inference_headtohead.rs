//! f32 SDPA INFERENCE (no-grad forward) head-to-head vs PyTorch (cc).
//! f32 mirror of `sdpa_inference_headtohead` — the dominant production dtype for serving /
//! autoregressive decode. FT's no-grad f32 fast-path flash kernel (`sdpa_forward_f32`,
//! reached via `scaled_dot_product_attention` with requires_grad=false) vs PyTorch f32 CPU
//! SDPA. Non-causal + causal. Measures the full through-session no-grad API path and prints
//! the FT/PyTorch output-sum relative diff as a correctness check. (NOTE: read the f32 output
//! with `tensor_values_f32`, not the f64 `tensor_values`, which rejects an F32 tensor.)
//!
//! Run: PYTORCH_PYTHON=/path/to/python cargo run --release -p ft-api --example sdpa_f32_inference_headtohead

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

// Full no-grad f32 SDPA through the public session API (requires_grad=false -> no tape,
// f32 fast-path flash kernel `sdpa_forward_f32`). The dominant serving path. Inputs are
// created ONCE and the timed region is op+read only — matching PyTorch's harness, which
// builds q/k/v once and times `F.scaled_dot_product_attention(...).abs().sum()` per iter.
fn bench_ft(qb: &[f32], kb: &[f32], vb: &[f32], causal: bool, iters: usize) -> (f64, f64) {
    let shape = vec![BH, SEQ, D];
    let mut s = FrankenTorchSession::new(ExecutionMode::Strict);
    let q = s
        .tensor_variable_f32(qb.to_vec(), shape.clone(), false)
        .unwrap();
    let k = s
        .tensor_variable_f32(kb.to_vec(), shape.clone(), false)
        .unwrap();
    let v = s.tensor_variable_f32(vb.to_vec(), shape, false).unwrap();
    let op = |s: &mut FrankenTorchSession| -> f64 {
        let out = s
            .scaled_dot_product_attention(q, k, v, None, 0.0, causal)
            .unwrap();
        s.tensor_values_f32(out)
            .unwrap()
            .iter()
            .map(|x| x.abs() as f64)
            .sum()
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
    return (torch.arange(total, dtype=torch.float32).mul_(0.017).add_(shift).sin_().mul_(0.2).reshape(BH,SEQ,D))
iters = int(os.environ.get("FT_GAUNTLET_ITERS","20"))
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
print("ELAPSED_MS", ts[0])
print("CHECKSUM", chk)
"#;

fn py(causal: bool, iters: usize) -> Option<(f64, f64)> {
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
    let s = String::from_utf8_lossy(&out.stdout);
    let ms = s.lines().find_map(|l| {
        l.strip_prefix("ELAPSED_MS ")
            .and_then(|v| v.trim().parse().ok())
    })?;
    let chk = s.lines().find_map(|l| {
        l.strip_prefix("CHECKSUM ")
            .and_then(|v| v.trim().parse().ok())
    })?;
    Some((ms, chk))
}

fn report(qb: &[f32], kb: &[f32], vb: &[f32], label: &str, causal: bool, iters: usize) {
    let (ft, ft_sum) = bench_ft(qb, kb, vb, causal, iters);
    print!("  {label:12} FT {ft:8.3} ms");
    match py(causal, iters) {
        Some((p, ps)) => {
            let rel = (ft_sum - ps).abs() / (ps.abs() + 1e-9);
            let r = p / ft;
            let verdict = if r >= 1.0 {
                format!("FT {r:.2}x FASTER")
            } else {
                format!("FT {:.2}x slower", 1.0 / r)
            };
            println!(
                "   PyTorch {p:8.3} ms  => {verdict}  (rel-diff {rel:.2e} {})",
                if rel < 1e-4 { "MATCH" } else { "MISMATCH!" }
            );
        }
        None => println!("   PyTorch (unavailable)"),
    }
}

fn main() {
    let iters: usize = std::env::var("ITERS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(20);
    println!("f32 SDPA INFERENCE (no-grad, through-session) [{BH},{SEQ},{D}], {iters} iters MIN:");
    let total = BH * SEQ * D;
    let q = seq_f32(total, 0.0);
    let k = seq_f32(total, 1.0);
    let v = seq_f32(total, 2.0);
    // RAW kernel row (no session/API overhead) to localize the kernel vs full-path cost.
    let scale = 1.0 / (D as f32).sqrt();
    for _ in 0..3 {
        let _ = ft_kernel_cpu::sdpa_forward_f32(&q, &k, &v, BH, SEQ, SEQ, D, D, scale, false);
    }
    let mut kt = Vec::new();
    for _ in 0..iters {
        let t = Instant::now();
        let o = ft_kernel_cpu::sdpa_forward_f32(&q, &k, &v, BH, SEQ, SEQ, D, D, scale, false);
        std::hint::black_box(&o);
        kt.push(t.elapsed().as_secs_f64() * 1e3);
    }
    kt.sort_by(|a, b| a.partial_cmp(b).unwrap());
    println!("  RAW sdpa_forward_f32 kernel only: {:8.3} ms (min)", kt[0]);
    report(&q, &k, &v, "non-causal", false, iters);
    report(&q, &k, &v, "causal", true, iters);
}

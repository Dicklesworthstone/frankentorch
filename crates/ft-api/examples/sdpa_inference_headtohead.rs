//! f64 SDPA INFERENCE (no-grad forward) head-to-head vs PyTorch (BlackThrush).
//! The f64 SDPA train-step win (~2.1x) extends here to the dominant production use:
//! no-grad attention (serving / autoregressive decode). FT's no-grad fast-path flash
//! kernel vs PyTorch's f64 SDPA math path. Both non-causal + causal.
//!
//! Run: PYTORCH_PYTHON=/path/to/python cargo run --release -p ft-api --example sdpa_inference_headtohead

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

// Pre-computed inputs are CLONED in (no sin()-generation inside the timed region,
// matching the gauntlet harness — else input-gen dominates the measurement).
fn ft_infer(qb: &[f64], kb: &[f64], vb: &[f64], causal: bool) -> f64 {
    let shape = vec![BH, SEQ, D];
    let mut s = FrankenTorchSession::new(ExecutionMode::Strict);
    // requires_grad = false -> no tape, no-grad fast-path flash kernel
    let q = s
        .tensor_variable(qb.to_vec(), shape.clone(), false)
        .unwrap();
    let k = s
        .tensor_variable(kb.to_vec(), shape.clone(), false)
        .unwrap();
    let v = s.tensor_variable(vb.to_vec(), shape, false).unwrap();
    let out = s
        .scaled_dot_product_attention(q, k, v, None, 0.0, causal)
        .unwrap();
    s.tensor_values(out).unwrap().iter().map(|x| x.abs()).sum()
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
iters = int(os.environ.get("FT_GAUNTLET_ITERS","15"))
torch.set_num_threads(int(os.environ.get("FT_TORCH_THREADS","8")))
q, k, v = dv(0.0), dv(1.0), dv(2.0)
with torch.no_grad():
    for _ in range(3):
        F.scaled_dot_product_attention(q,k,v,is_causal=causal).abs().sum().item()
    ts=[]
    for _ in range(iters):
        t0=time.perf_counter()
        F.scaled_dot_product_attention(q,k,v,is_causal=causal).abs().sum().item()
        ts.append((time.perf_counter()-t0)*1e3)
ts.sort()
print("ELAPSED_MS", ts[0])
"#;

fn bench_ft(qb: &[f64], kb: &[f64], vb: &[f64], causal: bool, iters: usize) -> f64 {
    for _ in 0..3 {
        let _ = ft_infer(qb, kb, vb, causal);
    }
    let mut times = Vec::with_capacity(iters);
    for _ in 0..iters {
        let t = Instant::now();
        let _ = ft_infer(qb, kb, vb, causal);
        times.push(t.elapsed().as_secs_f64() * 1e3);
    }
    times.sort_by(|a, b| a.partial_cmp(b).unwrap());
    times[0]
}

fn py(causal: bool, iters: usize) -> Option<f64> {
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

fn report(qb: &[f64], kb: &[f64], vb: &[f64], label: &str, causal: bool, iters: usize) {
    let ft = bench_ft(qb, kb, vb, causal, iters);
    print!("  {label:12} FT {ft:8.3} ms");
    match py(causal, iters) {
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
        .unwrap_or(15);
    println!("f64 SDPA INFERENCE (no-grad) [{BH},{SEQ},{D}], {iters} iters MIN:");
    // localize: RAW kernel (no session/API) vs the full no-grad API path
    let total = BH * SEQ * D;
    let q = seq_vals(total, 0.0);
    let k = seq_vals(total, 1.0);
    let v = seq_vals(total, 2.0);
    let scale = 1.0 / (D as f64).sqrt();
    for _ in 0..3 {
        let _ = ft_kernel_cpu::sdpa_forward_f64(&q, &k, &v, BH, SEQ, SEQ, D, D, scale, false);
    }
    let mut kt = Vec::new();
    for _ in 0..iters {
        let t = Instant::now();
        let o = ft_kernel_cpu::sdpa_forward_f64(&q, &k, &v, BH, SEQ, SEQ, D, D, scale, false);
        std::hint::black_box(&o);
        kt.push(t.elapsed().as_secs_f64() * 1e3);
    }
    kt.sort_by(|a, b| a.partial_cmp(b).unwrap());
    println!("  RAW sdpa_forward_f64 kernel only: {:8.3} ms (min)", kt[0]);

    // PHASE BREAKDOWN of one no-grad API call (min over iters) — localize the ~17ms overhead.
    let shape = vec![BH, SEQ, D];
    let (mut t_new, mut t_var, mut t_sdpa, mut t_read) = (f64::MAX, f64::MAX, f64::MAX, f64::MAX);
    for _ in 0..iters {
        let a = Instant::now();
        let mut s = FrankenTorchSession::new(ExecutionMode::Strict);
        t_new = t_new.min(a.elapsed().as_secs_f64() * 1e3);
        let a = Instant::now();
        let qn = s.tensor_variable(q.clone(), shape.clone(), false).unwrap();
        let kn = s.tensor_variable(k.clone(), shape.clone(), false).unwrap();
        let vn = s.tensor_variable(v.clone(), shape.clone(), false).unwrap();
        t_var = t_var.min(a.elapsed().as_secs_f64() * 1e3);
        let a = Instant::now();
        let out = s
            .scaled_dot_product_attention(qn, kn, vn, None, 0.0, false)
            .unwrap();
        t_sdpa = t_sdpa.min(a.elapsed().as_secs_f64() * 1e3);
        let a = Instant::now();
        let r: f64 = s.tensor_values(out).unwrap().iter().sum();
        t_read = t_read.min(a.elapsed().as_secs_f64() * 1e3);
        std::hint::black_box(r);
    }
    println!(
        "  PHASES (min ms): session_new {t_new:.3}  3x tensor_variable {t_var:.3}  sdpa {t_sdpa:.3}  read_out {t_read:.3}"
    );
    report(&q, &k, &v, "non-causal", false, iters);
    report(&q, &k, &v, "causal", true, iters);
}

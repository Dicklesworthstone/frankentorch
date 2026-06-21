//! cumsum/cumprod BACKWARD (full fwd+bwd step) head-to-head vs PyTorch dim=0 (BlackThrush).
//! PyTorch's dim=0 grad step is strided-slow (cumsum 463ms, cumprod 522ms vs ~95/154ms last-dim).
//! cumsum fwd+bwd kernels are cache-reordered (shipped); cumprod fwd is reordered but its BACKWARD
//! is not yet. f64 [262144,64]. Measures the full step (forward + backward).
//!
//! Run: PYTORCH_PYTHON=/path/to/python cargo run --release -p ft-api --example scan_bwd_headtohead

use std::process::Command;
use std::time::Instant;

use ft_api::FrankenTorchSession;
use ft_core::ExecutionMode;

const R: usize = 262144;
const C: usize = 64;

fn ft_step(op: &str, iters: usize) -> (f64, f64) {
    // bounded values for cumprod (alternating ~1.0)
    let data: Vec<f64> = (0..R * C)
        .map(|i| if op == "prod" { if i % 2 == 0 { 1.0001 } else { 0.9999 } } else { ((i as f64) * 0.001).sin() })
        .collect();
    let mut best = f64::INFINITY;
    let mut gchk = 0.0;
    for _ in 0..iters + 3 {
        let mut s = FrankenTorchSession::new(ExecutionMode::Strict);
        let x = s.tensor_variable(data.clone(), vec![R, C], true).unwrap();
        let t = Instant::now();
        let y = if op == "prod" { s.tensor_cumprod(x, 0).unwrap() } else { s.tensor_cumsum(x, 0).unwrap() };
        let loss = s.tensor_sum(y).unwrap();
        let report = s.tensor_backward(loss).unwrap();
        let g = s.tensor_gradient(&report, x).unwrap();
        let el = t.elapsed().as_secs_f64() * 1e3;
        if el < best { best = el; gchk = g.iter().map(|v| v.abs()).sum(); }
    }
    (gchk, best)
}

fn py(op: &str) -> Option<(f64, f64)> {
    let python = std::env::var("PYTORCH_PYTHON").unwrap_or_else(|_| "python3".to_string());
    let src = format!(r#"
import time, torch
torch.set_num_threads(8)
R,C=262144,64
if "{op}"=="prod":
    base = torch.where(torch.arange(R*C)%2==0, torch.tensor(1.0001,dtype=torch.float64), torch.tensor(0.9999,dtype=torch.float64)).reshape(R,C)
    f = torch.cumprod
else:
    base = torch.arange(R*C, dtype=torch.float64).mul_(0.001).sin_().reshape(R,C)
    f = torch.cumsum
def step():
    x = base.clone().requires_grad_(True)
    y = f(x, dim=0); y.sum().backward(); return x.grad
for _ in range(3): step()
ts=[]; g=None
for _ in range(12):
    t=time.perf_counter(); g=step(); ts.append((time.perf_counter()-t)*1e3)
print("MS", sorted(ts)[0]); print("GCHK", g.abs().sum().item())
"#);
    let o = Command::new(&python).arg("-c").arg(src).output().ok()?;
    if !o.status.success() { eprintln!("py: {}", String::from_utf8_lossy(&o.stderr)); return None; }
    let s = String::from_utf8_lossy(&o.stdout);
    let g = |p: &str| s.lines().find_map(|l| l.strip_prefix(p).and_then(|v| v.trim().parse::<f64>().ok()));
    Some((g("MS ")?, g("GCHK ")?))
}

fn main() {
    println!("cumsum/cumprod fwd+bwd step dim=0 [{R},{C}] f64, 12 iters MIN:");
    for op in ["sum", "prod"] {
        let (g, ms) = ft_step(op, 12);
        print!("  cum{op}: FT {ms:8.3} ms", );
        if let Some((p, pg)) = py(op) {
            let rel = (g - pg).abs() / (pg.abs() + 1e-9);
            let r = p / ms;
            let v = if r >= 1.0 { format!("FT {r:.2}x FASTER") } else { format!("FT {:.2}x slower", 1.0 / r) };
            println!("  | PyTorch {p:8.3} ms => {v}  (grad {})", if rel < 1e-9 { "MATCH" } else { "MISMATCH!" });
        } else { println!("  | PyTorch (unavailable)"); }
    }
}

//! End-to-end tensor_cummin_dim API head-to-head vs PyTorch (BlackThrush). Confirms the
//! cummin_dim kernel win survives the API wrapper (input read + leaf). f64 no-grad, dim=0.
//!
//! Run: PYTORCH_PYTHON=/path/to/python cargo run --release -p ft-api --example cummin_dim_api_headtohead

use std::process::Command;
use std::time::Instant;

use ft_api::FrankenTorchSession;
use ft_core::ExecutionMode;

const R: usize = 262144;
const C: usize = 64;

fn main() {
    let n = R * C;
    let data: Vec<f64> = (0..n).map(|i| ((i as f64) * 0.001).sin()).collect();
    let mut best = f64::INFINITY;
    let (mut vsum, mut isum) = (0.0, 0.0);
    for _ in 0..18 {
        let mut s = FrankenTorchSession::new(ExecutionMode::Strict);
        let x = s.tensor_variable(data.clone(), vec![R, C], false).unwrap();
        let t = Instant::now();
        let (v, i) = s.tensor_cummin_dim(x, 0).unwrap();
        let el = t.elapsed().as_secs_f64() * 1e3;
        if el < best {
            best = el;
            vsum = s.tensor_values(v).unwrap().iter().sum();
            isum = s.tensor_values(i).unwrap().iter().sum();
        }
    }
    let python = std::env::var("PYTORCH_PYTHON").unwrap_or_else(|_| "python3".to_string());
    let py = r#"
import time, torch
torch.set_num_threads(8)
R,C=262144,64
x = torch.arange(R*C, dtype=torch.float64).mul_(0.001).sin_().reshape(R,C)
for _ in range(3): torch.cummin(x, dim=0)
ts=[]
for _ in range(15):
    t=time.perf_counter(); torch.cummin(x, dim=0); ts.append((time.perf_counter()-t)*1e3)
v,i = torch.cummin(x, dim=0)
print("MS", sorted(ts)[0]); print("VSUM", v.sum().item()); print("ISUM", float(i.sum().item()))
"#;
    let out = Command::new(&python).arg("-c").arg(py).output();
    println!("tensor_cummin_dim API dim=0 [{R},{C}] f64 no-grad, 15 iters MIN:");
    println!("  FrankenTorch : {best:8.3} ms   vsum {vsum:.6e}  isum {isum:.6e}");
    let Ok(o) = out else {
        return;
    };
    if !o.status.success() {
        return;
    }
    let s = String::from_utf8_lossy(&o.stdout);
    let g = |p: &str| {
        s.lines()
            .find_map(|l| l.strip_prefix(p).and_then(|v| v.trim().parse::<f64>().ok()))
    };
    if let (Some(p), Some(pv), Some(pi)) = (g("MS "), g("VSUM "), g("ISUM ")) {
        let vrel = (vsum - pv).abs() / (pv.abs() + 1e-12);
        println!("  PyTorch      : {p:8.3} ms   vsum {pv:.6e}  isum {pi:.6e}");
        println!(
            "  CORRECTNESS  : values rel {vrel:.2e} ({}) ; indices ({})",
            if vrel < 1e-9 { "MATCH" } else { "MISMATCH!" },
            if (isum - pi).abs() < 0.5 {
                "MATCH"
            } else {
                "MISMATCH!"
            }
        );
        let r = p / best;
        if r >= 1.0 {
            println!("  => FT {r:.2}x FASTER (cummin end-to-end)");
        } else {
            println!("  => FT {:.2}x slower", 1.0 / r);
        }
    }
}

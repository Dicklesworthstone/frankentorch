//! f32 dim-aware cummax/cummin API head-to-head vs PyTorch (BlackThrush). Confirms the f32 fast
//! kernels (cummax_dim/cummin_dim _f32) win the strided dim=0 case. torch sums the values in f64
//! (.double().sum()) to match FT's f64-accumulated checksum of the identical f32 values.
//!
//! Run: PYTORCH_PYTHON=/path/to/python cargo run --release -p ft-api --example cummax_dim_f32_headtohead

use std::process::Command;
use std::time::Instant;

use ft_api::FrankenTorchSession;
use ft_core::{DType, ExecutionMode};

const R: usize = 262144;
const C: usize = 64;

fn ft(op: &str) -> (f64, f64, f64) {
    let data: Vec<f64> = (0..R * C).map(|i| ((i as f64) * 0.001).sin()).collect();
    let mut best = f64::INFINITY;
    let (mut vs, mut is_) = (0.0, 0.0);
    for _ in 0..18 {
        let mut s = FrankenTorchSession::new(ExecutionMode::Strict);
        let xf = s.tensor_variable(data.clone(), vec![R, C], false).unwrap();
        let x = s.tensor_to_dtype(xf, DType::F32).unwrap();
        let t = Instant::now();
        let (v, i) = if op == "max" {
            s.tensor_cummax_dim(x, 0).unwrap()
        } else {
            s.tensor_cummin_dim(x, 0).unwrap()
        };
        let el = t.elapsed().as_secs_f64() * 1e3;
        if el < best {
            best = el;
            vs = s.tensor_values_lossy_f64(v).unwrap().iter().sum();
            is_ = s.tensor_values_lossy_f64(i).unwrap().iter().sum();
        }
    }
    (vs, is_, best)
}

fn py(op: &str) -> Option<(f64, f64, f64)> {
    let python = std::env::var("PYTORCH_PYTHON").unwrap_or_else(|_| "python3".to_string());
    let src = format!(
        r#"
import time, torch
torch.set_num_threads(8)
R,C=262144,64
x = torch.arange(R*C, dtype=torch.float64).mul_(0.001).sin_().reshape(R,C).float()  # match FT: f64 sin -> round f32
f = torch.cummax if "{op}"=="max" else torch.cummin
for _ in range(3): f(x,dim=0)
ts=[]
for _ in range(15):
    t=time.perf_counter(); f(x,dim=0); ts.append((time.perf_counter()-t)*1e3)
v,i = f(x,dim=0)
print("MS", sorted(ts)[0]); print("VSUM", v.double().sum().item()); print("ISUM", float(i.sum().item()))
"#
    );
    let o = Command::new(&python).arg("-c").arg(src).output().ok()?;
    if !o.status.success() {
        eprintln!("py: {}", String::from_utf8_lossy(&o.stderr));
        return None;
    }
    let s = String::from_utf8_lossy(&o.stdout);
    let g = |p: &str| {
        s.lines()
            .find_map(|l| l.strip_prefix(p).and_then(|v| v.trim().parse::<f64>().ok()))
    };
    Some((g("MS ")?, g("VSUM ")?, g("ISUM ")?))
}

fn main() {
    println!("f32 dim-aware cummax/cummin dim=0 [{R},{C}], 15 iters MIN:");
    for op in ["max", "min"] {
        let (vs, is_, ms) = ft(op);
        print!("  cum{op}: FT {ms:8.3} ms");
        if let Some((p, pv, pi)) = py(op) {
            let vrel = (vs - pv).abs() / (pv.abs() + 1e-9);
            let r = p / ms;
            let verdict = if r >= 1.0 {
                format!("FT {r:.2}x FASTER")
            } else {
                format!("FT {:.2}x slower", 1.0 / r)
            };
            println!(
                "  | PyTorch {p:8.3} ms => {verdict}  (values {} ; indices {})",
                if vrel < 1e-6 { "MATCH" } else { "MISMATCH!" },
                if (is_ - pi).abs() < 0.5 {
                    "MATCH"
                } else {
                    "MISMATCH!"
                }
            );
        } else {
            println!("  | PyTorch (unavailable)");
        }
    }
}

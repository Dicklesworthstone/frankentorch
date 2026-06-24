//! softmax + log_softmax along dim=0 (f64 + f32) head-to-head vs PyTorch. dim=0 had outer_size==1
//! → the strided softmax path ran serial over the inner columns. The column-parallel transpose
//! trick parallelizes the exp-bound work. f64 + f32, dim=0.
//!
//! Run: PYTORCH_PYTHON=/path/to/python cargo run --release -p ft-api --example softmax_family_dim0_h2h

use std::process::Command;
use std::time::Instant;

use ft_api::FrankenTorchSession;
use ft_core::ExecutionMode;

const R: usize = 4096;
const C: usize = 4096;

fn bench_ft(op: &str, dtype: &str, data: &[f64]) -> (f64, f64) {
    let mut best = f64::INFINITY;
    let mut chk = 0.0;
    for _ in 0..12 {
        let mut s = FrankenTorchSession::new(ExecutionMode::Strict);
        let x = s.tensor_variable(data.to_vec(), vec![R, C], false).unwrap();
        let x = if dtype == "f32" {
            s.tensor_to_f32(x).unwrap()
        } else {
            x
        };
        let t = Instant::now();
        let out = if op == "softmax" {
            s.tensor_softmax(x, 0).unwrap()
        } else {
            s.tensor_log_softmax(x, 0).unwrap()
        };
        let el = t.elapsed().as_secs_f64() * 1e3;
        if el < best {
            best = el;
            chk = s.tensor_values(out).unwrap().iter().sum();
        }
    }
    (best, chk)
}

fn main() {
    let data: Vec<f64> = (0..R * C)
        .map(|i| ((i as f64) * 0.0007).sin() * 4.0)
        .collect();
    let python = std::env::var("PYTORCH_PYTHON").unwrap_or_else(|_| "python3".to_string());
    for op in ["softmax", "log_softmax"] {
        for dtype in ["f64"] {
            let (ft, chk) = bench_ft(op, dtype, &data);
            let py = format!(
                r#"
import time, torch
torch.set_num_threads(8)
R,C={R},{C}
x = (torch.arange(R*C, dtype=torch.float64).mul_(0.0007).sin_().mul_(4.0)).reshape(R,C)
x = x.to(torch.{dt})
fn = torch.{fn}
for _ in range(3): fn(x, dim=0)
ts=[]; chk=0.0
for _ in range(12):
    t=time.perf_counter(); o=fn(x, dim=0); ts.append((time.perf_counter()-t)*1e3); chk=o.double().sum().item()
print("MS", sorted(ts)[0]); print("CHK", chk)
"#,
                dt = if dtype == "f32" { "float32" } else { "float64" },
                fn = if op == "softmax" { "softmax" } else { "log_softmax" },
            );
            let out = Command::new(&python).arg("-c").arg(&py).output();
            print!("{op:11} {dtype}  dim=0 [{R},{C}]: FT {ft:8.3} ms");
            if let Ok(o) = out
                && o.status.success()
            {
                let s = String::from_utf8_lossy(&o.stdout);
                let g = |p: &str| {
                    s.lines()
                        .find_map(|l| l.strip_prefix(p).and_then(|v| v.trim().parse::<f64>().ok()))
                };
                if let (Some(p), Some(pc)) = (g("MS "), g("CHK ")) {
                    let rel = (chk - pc).abs() / (pc.abs() + 1e-9);
                    let r = p / ft;
                    let verdict = if r >= 1.0 {
                        format!("FT {r:.2}x FASTER")
                    } else {
                        format!("FT {:.2}x slower", 1.0 / r)
                    };
                    println!("  | torch {p:8.3} ms  => {verdict}  (sum rel {rel:.1e})");
                } else {
                    println!("  (torch parse fail)");
                }
            } else {
                println!("  (torch fail)");
            }
        }
    }
}

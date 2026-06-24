//! softmax along a LEADING dim (dim=0) head-to-head vs PyTorch. softmax_dim's general strided path
//! parallelizes over OUTER blocks, so dim=0 (outer_size==1) runs serial: it loops over inner columns
//! sequentially, each doing gather/max/exp/sum/scatter over the reduce dim. exp is compute-bound, so
//! parallelizing the independent inner columns should scale. f64 no-grad, dim=0.
//!
//! Run: PYTORCH_PYTHON=/path/to/python cargo run --release -p ft-api --example softmax_dim0_h2h

use std::process::Command;
use std::time::Instant;

use ft_api::FrankenTorchSession;
use ft_core::ExecutionMode;

const R: usize = 4096;
const C: usize = 4096;

fn main() {
    let n = R * C;
    let data: Vec<f64> = (0..n).map(|i| ((i as f64) * 0.0007).sin() * 4.0).collect();
    let mut best = f64::INFINITY;
    let mut chk = 0.0;
    for _ in 0..12 {
        let mut s = FrankenTorchSession::new(ExecutionMode::Strict);
        let x = s.tensor_variable(data.clone(), vec![R, C], false).unwrap();
        let t = Instant::now();
        let out = s.tensor_softmax(x, 0).unwrap();
        let el = t.elapsed().as_secs_f64() * 1e3;
        if el < best {
            best = el;
            chk = s.tensor_values(out).unwrap().iter().sum();
        }
    }
    let python = std::env::var("PYTORCH_PYTHON").unwrap_or_else(|_| "python3".to_string());
    let py = r#"
import time, torch
torch.set_num_threads(8)
R,C=4096,4096
x = (torch.arange(R*C, dtype=torch.float64).mul_(0.0007).sin_().mul_(4.0)).reshape(R,C)
for _ in range(3): torch.softmax(x, dim=0)
ts=[]; chk=0.0
for _ in range(12):
    t=time.perf_counter(); o=torch.softmax(x, dim=0); ts.append((time.perf_counter()-t)*1e3); chk=o.sum().item()
print("MS", sorted(ts)[0]); print("CHK", chk)
"#;
    let out = Command::new(&python).arg("-c").arg(py).output();
    println!("softmax(x, dim=0) [{R},{C}] f64 no-grad, 12 iters MIN:");
    println!("  FrankenTorch : {best:8.3} ms   chk {chk:.6e}");
    if let Ok(o) = out
        && o.status.success()
    {
        let s = String::from_utf8_lossy(&o.stdout);
        let g = |p: &str| {
            s.lines()
                .find_map(|l| l.strip_prefix(p).and_then(|v| v.trim().parse::<f64>().ok()))
        };
        if let (Some(p), Some(pc)) = (g("MS "), g("CHK ")) {
            let rel = (chk - pc).abs() / (pc.abs() + 1e-12);
            println!("  PyTorch      : {p:8.3} ms   chk {pc:.6e}");
            println!(
                "  CORRECTNESS  : sum rel {rel:.2e} ({})",
                if rel < 1e-9 { "MATCH" } else { "MISMATCH!" }
            );
            let r = p / best;
            if r >= 1.0 {
                println!("  => FT {r:.2}x FASTER");
            } else {
                println!("  => FT {:.2}x slower", 1.0 / r);
            }
        }
    }
}

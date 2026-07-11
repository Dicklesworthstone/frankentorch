//! torch.count_nonzero(x) head-to-head vs PyTorch. FT currently CLONES the full input via
//! tensor_values and counts SERIALLY (.iter().filter().count()). Both are wasteful: count is
//! order-independent, so we can borrow zero-copy + rayon-count. This bench measures the gap.
//!
//! Run: PYTORCH_PYTHON=/path/to/python cargo run --release -p ft-api --example count_nonzero_h2h

use std::io::Write;
use std::process::{Command, Stdio};
use std::time::Instant;

use ft_api::FrankenTorchSession;
use ft_core::ExecutionMode;

const R: usize = 4000;
const C: usize = 4000;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // ~half nonzero (every other element zero).
    let data: Vec<f64> = (0..R * C)
        .map(|i| if i % 2 == 0 { 0.0 } else { (i % 7 + 1) as f64 })
        .collect();
    let mut best = f64::INFINITY;
    let mut chk = 0.0;
    for _ in 0..8 {
        let mut s = FrankenTorchSession::new(ExecutionMode::Strict);
        let x = s.tensor_variable(data.clone(), vec![R, C], false)?;
        let t = Instant::now();
        let c = s.tensor_count_nonzero(x)?;
        let el = t.elapsed().as_secs_f64() * 1e3;
        if el < best {
            best = el;
            chk = s.tensor_values(c)?[0];
        }
    }
    let python = std::env::var("PYTORCH_PYTHON").unwrap_or_else(|_| "python3".to_string());
    let py = r#"
import time, torch
torch.set_num_threads(8)
R,C=4000,4000
idx=torch.arange(R*C,dtype=torch.int64)
x=torch.where(idx%2==0, torch.zeros(R*C,dtype=torch.float64), ((idx%7)+1).double()).reshape(R,C)
for _ in range(2): torch.count_nonzero(x)
ts=[];chk=0.0
for _ in range(8):
    t=time.perf_counter(); c=torch.count_nonzero(x); ts.append((time.perf_counter()-t)*1e3); chk=float(c.item())
print("MS", sorted(ts)[0]); print("CHK", chk)
"#;
    let mut child = Command::new(&python)
        .arg("-")
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .spawn()?;
    child
        .stdin
        .as_mut()
        .ok_or_else(|| std::io::Error::other("no stdin"))?
        .write_all(py.as_bytes())?;
    let out = child.wait_with_output();
    println!("count_nonzero(x) [{R},{C}] f64 no-grad, 8 iters MIN:");
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
            let rel = (chk - pc).abs() / (pc.abs() + 1e-9);
            println!("  PyTorch      : {p:8.3} ms   chk {pc:.6e}  (count rel {rel:.1e})");
            let r = p / best;
            if r >= 1.0 {
                println!("  => FT {r:.2}x FASTER");
            } else {
                println!("  => FT {:.2}x SLOWER", 1.0 / r);
            }
        }
    }
    Ok(())
}

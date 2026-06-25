//! torch.mode(x, dim=-1) head-to-head vs PyTorch on f32 BOUNDED-INTEGER data.
//! f32 is the common ML dtype; torch.mode(f32) is sort-based and slow, and FT's f32 mode
//! otherwise upcasts to f64 and sorts. FT's f32 counting fast path: histogram for the
//! mode INDEX, then gather the value from the f32 input (keeps f32 dtype). Bit-identical.
//!
//! Run: PYTORCH_PYTHON=/path/to/python cargo run --release -p ft-api --example mode_f32_h2h

use std::io::Write;
use std::process::{Command, Stdio};
use std::time::Instant;

use ft_api::FrankenTorchSession;
use ft_core::ExecutionMode;

const R: usize = 4096;
const C: usize = 4096;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let data: Vec<f32> = (0..R * C)
        .map(|i| ((i * 1_103_515_245usize + 12345) % 97) as f32)
        .collect();
    let mut best = f64::INFINITY;
    let mut chk = 0.0_f64;
    for _ in 0..8 {
        let mut s = FrankenTorchSession::new(ExecutionMode::Strict);
        let x = s.tensor_variable_f32(data.clone(), vec![R, C], false)?;
        let t = Instant::now();
        let (vals, _idx) = s.tensor_mode(x)?;
        let el = t.elapsed().as_secs_f64() * 1e3;
        if el < best {
            best = el;
            chk = s.tensor_values_f32(vals)?.iter().map(|&v| f64::from(v)).sum();
        }
    }
    let python = std::env::var("PYTORCH_PYTHON").unwrap_or_else(|_| "python3".to_string());
    let py = r#"
import time, torch
torch.set_num_threads(8)
R,C=4096,4096
idx = torch.arange(R*C, dtype=torch.int64)
x = (((idx * 1103515245 + 12345) % 97)).float().reshape(R,C)
for _ in range(2): torch.mode(x, dim=-1)
ts=[]; chk=0.0
for _ in range(8):
    t=time.perf_counter(); v,i=torch.mode(x, dim=-1); ts.append((time.perf_counter()-t)*1e3); chk=v.double().sum().item()
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
        .ok_or_else(|| std::io::Error::other("python stdin unavailable"))?
        .write_all(py.as_bytes())?;
    let out = child.wait_with_output();
    println!("mode(x, dim=-1) [{R},{C}] f32 no-grad, 8 iters MIN:");
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
            println!("  PyTorch      : {p:8.3} ms   chk {pc:.6e}  (value-sum rel {rel:.1e})");
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

//! Single-plane rfft2 [R,C] no-grad — row-phase parallelization (try). FT_ORIG=1 = serial row phase.
//! Internal A/B (same process = contention-robust ratio). Also vs torch. Inputs OUTSIDE the timer.
use ft_api::FrankenTorchSession;
use ft_core::ExecutionMode;
use std::io::Write;
use std::process::{Command, Stdio};
use std::time::Instant;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let python = std::env::var("PYTORCH_PYTHON").unwrap_or_else(|_| "python3".to_string());
    let (r, c) = (4096usize, 4096usize); // single 2D plane
    let n = r * c;
    let xd: Vec<f64> = (0..n).map(|i| ((i % 101) as f64) * 0.01 - 0.5).collect();
    let orig = std::env::var("FT_ORIG").is_ok();
    let mut best = f64::INFINITY;
    for _ in 0..7 {
        let mut s = FrankenTorchSession::new(ExecutionMode::Strict);
        let x = s.tensor_variable(xd.clone(), vec![r, c], false).unwrap();
        let t = Instant::now();
        let _ = s.tensor_rfft2(x).unwrap();
        let e = t.elapsed().as_secs_f64() * 1e3;
        if e < best { best = e; }
    }
    let label = if orig { "FT_ORIG(serial-row)" } else { "FT_PARALLEL" };
    let py = format!(
        r#"
import time,torch
torch.set_num_threads(8)
r,c={r},{c}
x=((torch.arange(r*c,dtype=torch.int64)%101).double())*0.01-0.5; x=x.reshape(r,c)
def t(fn,reps=7):
    for _ in range(2): fn()
    ts=[]
    for _ in range(reps): s=time.perf_counter(); fn(); ts.append((time.perf_counter()-s)*1e3)
    return min(ts)
print("PT rfft2 %.4f"%t(lambda:torch.fft.rfft2(x)))
"#
    );
    let mut ch = Command::new(&python).arg("-").stdin(Stdio::piped()).stdout(Stdio::piped()).spawn()?;
    ch.stdin.as_mut().unwrap().write_all(py.as_bytes())?;
    let pt = String::from_utf8_lossy(&ch.wait_with_output()?.stdout).to_string();
    let ptw = pt.lines().find_map(|l| l.strip_prefix("PT rfft2 ")).and_then(|s| s.trim().parse::<f64>().ok()).unwrap_or(f64::NAN);
    let v = |ft: f64, p: f64| if p >= ft { format!("FT {:.2}x FASTER", p / ft) } else { format!("FT {:.2}x SLOWER", ft / p) };
    println!("  rfft2[{r}x{c}] {label} {best:.3}ms  PT {ptw:.3}ms => {}", v(best, ptw));
    Ok(())
}

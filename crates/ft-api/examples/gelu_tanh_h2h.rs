//! gelu_tanh (GPT-2/BERT GELU, F.gelu(x, approximate='tanh')) — fused one-pass vs torch. Set FT_ORIG=1
//! (fast path disabled + rebuilt) to time the ~9-op compose. 16M f64 no-grad, inputs OUTSIDE the timer.
use ft_api::FrankenTorchSession;
use ft_core::ExecutionMode;
use std::io::Write;
use std::process::{Command, Stdio};
use std::time::Instant;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let python = std::env::var("PYTORCH_PYTHON").unwrap_or_else(|_| "python3".to_string());

    // parity: small
    let ns = 9usize;
    let sv: Vec<f64> = (0..ns).map(|i| (i as f64) - 4.0).collect();
    let py_s = format!(
        r#"
import torch, torch.nn.functional as F
x=torch.tensor({sv:?},dtype=torch.float64)
o=F.gelu(x,approximate='tanh')
print("VALS"," ".join("%.17g"%v for v in o.tolist()))
"#
    );
    let mut ch = Command::new(&python).arg("-").stdin(Stdio::piped()).stdout(Stdio::piped()).spawn()?;
    ch.stdin.as_mut().unwrap().write_all(py_s.as_bytes())?;
    let pt = String::from_utf8_lossy(&ch.wait_with_output()?.stdout).to_string();
    let pv: Vec<f64> = pt.lines().find_map(|l| l.strip_prefix("VALS ")).map(|s| s.split_whitespace().filter_map(|t| t.parse().ok()).collect()).unwrap_or_default();
    let mut s = FrankenTorchSession::new(ExecutionMode::Strict);
    let x = s.tensor_variable(sv.clone(), vec![ns], false)?;
    let o = s.tensor_gelu_tanh(x)?;
    let fv = s.tensor_values(o)?;
    // gelu_tanh is an approx op — report max abs diff + bit-mismatch count (torch may use a fused kernel).
    let maxad = fv.iter().zip(&pv).map(|(a, b)| (a - b).abs()).fold(0.0f64, f64::max);
    let mm = fv.iter().zip(&pv).filter(|(a, b)| a.to_bits() != b.to_bits()).count();
    println!("parity: {mm}/{} bit-mismatch, max_abs_diff={maxad:.2e}", pv.len());

    // perf: 16M f64 no-grad
    let n = 16_000_000usize;
    let xd: Vec<f64> = (0..n).map(|i| ((i % 4000) as f64) * 0.005 - 10.0).collect();
    let orig = std::env::var("FT_ORIG").is_ok();
    let mut best = f64::INFINITY;
    for _ in 0..7 {
        let mut s = FrankenTorchSession::new(ExecutionMode::Strict);
        let x = s.tensor_variable(xd.clone(), vec![n], false).unwrap();
        let t = Instant::now();
        let _ = s.tensor_gelu_tanh(x).unwrap();
        let e = t.elapsed().as_secs_f64() * 1e3;
        if e < best { best = e; }
    }
    let label = if orig { "FT_ORIG(compose)" } else { "FT_FUSED" };
    let py_b = format!(
        r#"
import time,torch,torch.nn.functional as F
torch.set_num_threads(8)
n={n}
x=((torch.arange(n,dtype=torch.int64)%4000).double())*0.005-10.0
def t(fn,reps=7):
    for _ in range(2): fn()
    ts=[]
    for _ in range(reps): s=time.perf_counter(); fn(); ts.append((time.perf_counter()-s)*1e3)
    return min(ts)
print("PT gelu %.4f"%t(lambda:F.gelu(x,approximate='tanh')))
"#
    );
    let mut ch = Command::new(&python).arg("-").stdin(Stdio::piped()).stdout(Stdio::piped()).spawn()?;
    ch.stdin.as_mut().unwrap().write_all(py_b.as_bytes())?;
    let pt = String::from_utf8_lossy(&ch.wait_with_output()?.stdout).to_string();
    let ptw = pt.lines().find_map(|l| l.strip_prefix("PT gelu ")).and_then(|s| s.trim().parse::<f64>().ok()).unwrap_or(f64::NAN);
    let v = |ft: f64, p: f64| if p >= ft { format!("FT {:.2}x FASTER", p / ft) } else { format!("FT {:.2}x SLOWER", ft / p) };
    println!("  gelu_tanh(16M) {label} {best:.3}ms  PT {ptw:.3}ms  => {}", v(best, ptw));
    Ok(())
}

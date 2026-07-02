//! torch.lerp(start, end, weight_tensor) — tensor-weight linear interpolation (EMA/diffusion). Fused
//! no-grad same-shape path (tensor_lerp_weighted) vs torch. Set FT_ORIG=1 (fast path disabled +
//! rebuilt) to time the sub+mul+add compose. Inputs materialized OUTSIDE the timer.
use ft_api::FrankenTorchSession;
use ft_core::ExecutionMode;
use std::io::Write;
use std::process::{Command, Stdio};
use std::time::Instant;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let python = std::env::var("PYTORCH_PYTHON").unwrap_or_else(|_| "python3".to_string());

    // ---- parity: small ----
    let ns = 7usize;
    let sv: Vec<f64> = (0..ns).map(|i| (i as f64) * 0.5 - 3.0).collect();
    let ev: Vec<f64> = (0..ns).map(|i| (i as f64) * -0.25 + 2.0).collect();
    let wv: Vec<f64> = (0..ns).map(|i| ((i % 5) as f64) * 0.2).collect();
    let py_s = format!(
        r#"
import torch
s=torch.tensor({sv:?},dtype=torch.float64); e=torch.tensor({ev:?},dtype=torch.float64); w=torch.tensor({wv:?},dtype=torch.float64)
o=torch.lerp(s,e,w)
print("VALS"," ".join("%.17g"%v for v in o.tolist()))
"#
    );
    let mut ch = Command::new(&python).arg("-").stdin(Stdio::piped()).stdout(Stdio::piped()).spawn()?;
    ch.stdin.as_mut().unwrap().write_all(py_s.as_bytes())?;
    let pt = String::from_utf8_lossy(&ch.wait_with_output()?.stdout).to_string();
    let pv: Vec<f64> = pt.lines().find_map(|l| l.strip_prefix("VALS ")).map(|s| s.split_whitespace().filter_map(|t| t.parse().ok()).collect()).unwrap_or_default();
    let mut s = FrankenTorchSession::new(ExecutionMode::Strict);
    let a = s.tensor_variable(sv.clone(), vec![ns], false)?;
    let b = s.tensor_variable(ev.clone(), vec![ns], false)?;
    let w = s.tensor_variable(wv.clone(), vec![ns], false)?;
    let o = s.tensor_lerp_weighted(a, b, w)?;
    let fv = s.tensor_values(o)?;
    let mm = fv.iter().zip(&pv).filter(|(x, y)| x.to_bits() != y.to_bits()).count() + fv.len().abs_diff(pv.len());
    println!("parity: {mm}/{} value-bit mismatches", pv.len());

    // ---- perf: 16M f64 no-grad ----
    let n = 16_000_000usize;
    let sd: Vec<f64> = (0..n).map(|i| (i % 97) as f64 * 0.5 - 10.0).collect();
    let ed: Vec<f64> = (0..n).map(|i| (i % 89) as f64 * -0.25 + 4.0).collect();
    let wd: Vec<f64> = (0..n).map(|i| (i % 13) as f64 * 0.07).collect();
    let orig = std::env::var("FT_ORIG").is_ok();
    let mut best = f64::INFINITY;
    for _ in 0..9 {
        let mut s = FrankenTorchSession::new(ExecutionMode::Strict);
        let a = s.tensor_variable(sd.clone(), vec![n], false).unwrap();
        let b = s.tensor_variable(ed.clone(), vec![n], false).unwrap();
        let w = s.tensor_variable(wd.clone(), vec![n], false).unwrap();
        let t = Instant::now();
        let _ = s.tensor_lerp_weighted(a, b, w).unwrap();
        let e = t.elapsed().as_secs_f64() * 1e3;
        if e < best { best = e; }
    }
    let label = if orig { "FT_ORIG(compose)" } else { "FT_FUSED" };

    let py_b = format!(
        r#"
import time,torch
torch.set_num_threads(8)
n={n}
s=(torch.arange(n,dtype=torch.int64)%97).double()*0.5-10.0
e=(torch.arange(n,dtype=torch.int64)%89).double()*-0.25+4.0
w=(torch.arange(n,dtype=torch.int64)%13).double()*0.07
def t(fn,reps=9):
    for _ in range(2): fn()
    ts=[]
    for _ in range(reps): st=time.perf_counter(); fn(); ts.append((time.perf_counter()-st)*1e3)
    return min(ts)
print("PT lerp %.4f"%t(lambda:torch.lerp(s,e,w)))
"#
    );
    let mut ch = Command::new(&python).arg("-").stdin(Stdio::piped()).stdout(Stdio::piped()).spawn()?;
    ch.stdin.as_mut().unwrap().write_all(py_b.as_bytes())?;
    let pt = String::from_utf8_lossy(&ch.wait_with_output()?.stdout).to_string();
    let ptw = pt.lines().find_map(|l| l.strip_prefix("PT lerp ")).and_then(|s| s.trim().parse::<f64>().ok()).unwrap_or(f64::NAN);
    let v = |ft: f64, p: f64| if p >= ft { format!("FT {:.2}x FASTER", p / ft) } else { format!("FT {:.2}x SLOWER", ft / p) };
    println!("  lerp_weighted(16M) {label} {best:.3}ms  PT {ptw:.3}ms  => {}", v(best, ptw));
    Ok(())
}

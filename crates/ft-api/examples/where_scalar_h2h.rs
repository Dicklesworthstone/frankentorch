//! f64 where_scalar no-grad fast path vs torch (was: full(x)+full(y)+where = 3 passes).
//! relu = anchor. where_scalar(cond, x, y): cond != 0 ? x : y.
use ft_api::FrankenTorchSession;
use ft_core::ExecutionMode;
use std::io::Write;
use std::process::{Command, Stdio};
use std::time::Instant;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let python = std::env::var("PYTORCH_PYTHON").unwrap_or_else(|_| "python3".to_string());
    let (x, y) = (3.5_f64, -1.25_f64);

    // parity: small cond (0/1)
    let cond: Vec<f64> = vec![0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0, 0.0];
    let py_s = format!(
        r#"
import torch
c=torch.tensor({cond:?},dtype=torch.float64)
o=torch.where(c!=0,torch.tensor({x},dtype=torch.float64),torch.tensor({y},dtype=torch.float64))
print("VALS"," ".join("%.17g"%v for v in o.tolist()))
"#,
        cond = cond, x = x, y = y
    );
    let mut ch = Command::new(&python).arg("-").stdin(Stdio::piped()).stdout(Stdio::piped()).spawn()?;
    ch.stdin.as_mut().unwrap().write_all(py_s.as_bytes())?;
    let pt = String::from_utf8_lossy(&ch.wait_with_output()?.stdout).to_string();
    let pv: Vec<f64> = pt.lines().find_map(|l| l.strip_prefix("VALS ")).map(|s| s.split_whitespace().filter_map(|t| t.parse().ok()).collect()).unwrap_or_default();
    let mut s = FrankenTorchSession::new(ExecutionMode::Strict);
    let c = s.tensor_variable(cond.clone(), vec![cond.len()], false)?;
    let o = s.tensor_where_scalar(c, x, y)?;
    let fv = s.tensor_values(o)?;
    let mm = fv.iter().zip(&pv).filter(|(a, b)| a.to_bits() != b.to_bits()).count() + fv.len().abs_diff(pv.len());
    println!("parity: {mm}/{} value-bit mismatches", pv.len());

    // perf 16M f64, time only the op
    let n = 16_000_000usize;
    let cd: Vec<f64> = (0..n).map(|i| if i % 2 == 0 { 1.0 } else { 0.0 }).collect();
    let t = |which: u8| { let mut b = f64::INFINITY; for _ in 0..9 { let mut s = FrankenTorchSession::new(ExecutionMode::Strict); let c = s.tensor_variable(cd.clone(), vec![n], false).unwrap(); let t = Instant::now(); let _ = if which == 0 { s.tensor_relu(c) } else { s.tensor_where_scalar(c, x, y) }; let e = t.elapsed().as_secs_f64()*1e3; if e<b{b=e;} } b };
    let (tr, tw) = (t(0), t(1));
    let py_b = format!(r#"
import time,torch
torch.set_num_threads(8)
n={n}
c=((torch.arange(n,dtype=torch.int64)%2)==0).double()
X=torch.tensor({x},dtype=torch.float64); Y=torch.tensor({y},dtype=torch.float64)
def t(fn,reps=9):
    for _ in range(2): fn()
    ts=[]
    for _ in range(reps): s=time.perf_counter(); fn(); ts.append((time.perf_counter()-s)*1e3)
    return min(ts)
print("PT relu %.4f"%t(lambda:torch.relu(c)))
print("PT ws %.4f"%t(lambda:torch.where(c!=0,X,Y)))
"#, n = n, x = x, y = y);
    let mut ch = Command::new(&python).arg("-").stdin(Stdio::piped()).stdout(Stdio::piped()).spawn()?;
    ch.stdin.as_mut().unwrap().write_all(py_b.as_bytes())?;
    let pt = String::from_utf8_lossy(&ch.wait_with_output()?.stdout).to_string();
    let g = |k: &str| pt.lines().find_map(|l| { let mut it = l.strip_prefix("PT ")?.split_whitespace(); if it.next()? == k { it.next()?.parse::<f64>().ok() } else { None } }).unwrap_or(f64::NAN);
    let v = |ft: f64, p: f64| if p >= ft { format!("FT {:.2}x FASTER", p / ft) } else { format!("FT {:.2}x SLOWER", ft / p) };
    println!("  relu_anchor  FT {tr:.3} PT {:.3}  => {}", g("relu"), v(tr, g("relu")));
    println!("  where_scalar FT {tw:.3} PT {:.3}  => {}", g("ws"), v(tw, g("ws")));
    Ok(())
}

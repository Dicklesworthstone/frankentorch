//! f32 smooth_l1_loss (was ERRORING on f32) vs torch. beta=1.0. relu = anchor.
use ft_api::FrankenTorchSession;
use ft_core::{DType, ExecutionMode};
use std::io::Write;
use std::process::{Command, Stdio};
use std::time::Instant;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let python = std::env::var("PYTORCH_PYTHON").unwrap_or_else(|_| "python3".to_string());
    let beta = 1.0_f64;

    // parity reduction='none' (bit-exact per-element)
    let xs: Vec<f32> = vec![0.3, 1.5, -0.7, 2.0, -2.5, 0.0, 0.9, -1.1, 0.5, -0.5];
    let ts: Vec<f32> = vec![0.0, 0.0, 0.5, -1.0, 1.0, 0.2, 1.0, 0.0, 0.49, -0.49];
    let py_s = format!(
        r#"
import torch
x=torch.tensor({xs:?},dtype=torch.float32); t=torch.tensor({ts:?},dtype=torch.float32)
o=torch.nn.functional.smooth_l1_loss(x,t,beta={beta},reduction='none')
print("VALS"," ".join("%.9g"%v for v in o.tolist()))
"#,
        xs = xs, ts = ts, beta = beta
    );
    let mut ch = Command::new(&python).arg("-").stdin(Stdio::piped()).stdout(Stdio::piped()).spawn()?;
    ch.stdin.as_mut().unwrap().write_all(py_s.as_bytes())?;
    let pt = String::from_utf8_lossy(&ch.wait_with_output()?.stdout).to_string();
    let pv: Vec<f64> = pt.lines().find_map(|l| l.strip_prefix("VALS ")).map(|s| s.split_whitespace().filter_map(|t| t.parse().ok()).collect()).unwrap_or_default();
    let mut s = FrankenTorchSession::new(ExecutionMode::Strict);
    let x = s.tensor_variable_f32(xs.clone(), vec![xs.len()], false)?;
    let t = s.tensor_variable_f32(ts.clone(), vec![ts.len()], false)?;
    let o = s.tensor_smooth_l1_loss(x, t, "none", beta)?;
    let dt = s.tensor_dtype(o)?;
    let fv = s.tensor_values_lossy_f64(o)?;
    let mm = fv.iter().zip(&pv).filter(|(a, b)| (**a as f32).to_bits() != (**b as f32).to_bits()).count() + fv.len().abs_diff(pv.len());
    println!("parity f32: dtype={dt:?} (now WORKS, was ERROR) {mm}/{} per-element mismatches", pv.len());

    // perf 16M f32, reduction='mean'
    let n = 16_000_000usize;
    let xb: Vec<f32> = (0..n).map(|i| ((i % 9973) as f32 - 5000.0) * 0.001).collect();
    let tb: Vec<f32> = (0..n).map(|i| ((i % 7919) as f32 - 4000.0) * 0.0011).collect();
    let tt = |which: u8| { let mut b = f64::INFINITY; for _ in 0..9 { let mut s = FrankenTorchSession::new(ExecutionMode::Strict); let x = s.tensor_variable_f32(xb.clone(), vec![n], false).unwrap(); let t = s.tensor_variable_f32(tb.clone(), vec![n], false).unwrap(); let ti = Instant::now(); let _ = if which == 0 { s.tensor_relu(x) } else { s.tensor_smooth_l1_loss(x, t, "mean", beta) }; let e = ti.elapsed().as_secs_f64()*1e3; if e<b{b=e;} } b };
    let (tr, th) = (tt(0), tt(1));
    let py_b = format!(r#"
import time,torch
torch.set_num_threads(8)
n={n}
x=(((torch.arange(n,dtype=torch.int64)%9973).float()-5000.0)*0.001)
t=(((torch.arange(n,dtype=torch.int64)%7919).float()-4000.0)*0.0011)
def tm(fn,reps=9):
    for _ in range(2): fn()
    ts=[]
    for _ in range(reps): s=time.perf_counter(); fn(); ts.append((time.perf_counter()-s)*1e3)
    return min(ts)
print("PT relu %.4f"%tm(lambda:torch.relu(x)))
print("PT sl1 %.4f"%tm(lambda:torch.nn.functional.smooth_l1_loss(x,t,beta={beta},reduction='mean')))
"#, n = n, beta = beta);
    let mut ch = Command::new(&python).arg("-").stdin(Stdio::piped()).stdout(Stdio::piped()).spawn()?;
    ch.stdin.as_mut().unwrap().write_all(py_b.as_bytes())?;
    let pt = String::from_utf8_lossy(&ch.wait_with_output()?.stdout).to_string();
    let g = |k: &str| pt.lines().find_map(|l| { let mut it = l.strip_prefix("PT ")?.split_whitespace(); if it.next()? == k { it.next()?.parse::<f64>().ok() } else { None } }).unwrap_or(f64::NAN);
    let v = |ft: f64, p: f64| if p >= ft { format!("FT {:.2}x FASTER", p / ft) } else { format!("FT {:.2}x SLOWER", ft / p) };
    println!("  relu_anchor  FT {tr:.3} PT {:.3}  => {}", g("relu"), v(tr, g("relu")));
    println!("  smooth_l1    FT {th:.3} PT {:.3}  => {}", g("sl1"), v(th, g("sl1")));
    Ok(())
}

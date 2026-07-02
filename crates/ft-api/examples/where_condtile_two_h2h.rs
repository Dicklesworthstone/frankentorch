//! where(mask[1,1,S,S], a[B,H,S,S], b[B,H,S,S]) — select between two full score tensors with a
//! broadcast attention mask. Fused two-full cond-tile path (try_where_cond_tile_two) vs torch.
//! Set FT_ORIG=1 (with the fast path disabled + rebuilt) to time the broadcast fallthrough.
//! Inputs are materialized OUTSIDE the timer (a 8.4M clone is ~ms and would swamp the op).
use ft_api::FrankenTorchSession;
use ft_core::ExecutionMode;
use std::io::Write;
use std::process::{Command, Stdio};
use std::time::Instant;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let python = std::env::var("PYTORCH_PYTHON").unwrap_or_else(|_| "python3".to_string());

    // ---- parity: small, mask [1,1,s,s] into a/b [b,h,s,s] ----
    let (bb, hh, ss) = (3usize, 2usize, 5usize);
    let n_small = bb * hh * ss * ss;
    let cn = ss * ss;
    let mask: Vec<f64> = (0..cn).map(|i| ((i % 3) as f64) - 1.0).collect(); // -1/0/1
    let av: Vec<f64> = (0..n_small).map(|i| (i as f64) * 0.25 - 30.0).collect();
    let bv: Vec<f64> = (0..n_small).map(|i| (i as f64) * -0.5 + 7.0).collect();
    let py_s = format!(
        r#"
import torch
m=torch.tensor({mask:?},dtype=torch.float64).reshape(1,1,{ss},{ss})
a=torch.tensor({av:?},dtype=torch.float64).reshape({bb},{hh},{ss},{ss})
b=torch.tensor({bv:?},dtype=torch.float64).reshape({bb},{hh},{ss},{ss})
o=torch.where(m!=0,a,b).reshape(-1)
print("VALS"," ".join("%.17g"%v for v in o.tolist()))
"#
    );
    let mut ch = Command::new(&python).arg("-").stdin(Stdio::piped()).stdout(Stdio::piped()).spawn()?;
    ch.stdin.as_mut().unwrap().write_all(py_s.as_bytes())?;
    let pt = String::from_utf8_lossy(&ch.wait_with_output()?.stdout).to_string();
    let pv: Vec<f64> = pt.lines().find_map(|l| l.strip_prefix("VALS ")).map(|s| s.split_whitespace().filter_map(|t| t.parse().ok()).collect()).unwrap_or_default();
    let mut s = FrankenTorchSession::new(ExecutionMode::Strict);
    let m = s.tensor_variable(mask.clone(), vec![1, 1, ss, ss], false)?;
    let a = s.tensor_variable(av.clone(), vec![bb, hh, ss, ss], false)?;
    let b = s.tensor_variable(bv.clone(), vec![bb, hh, ss, ss], false)?;
    let o = s.tensor_where(m, a, b)?;
    let fv = s.tensor_values(o)?;
    let mm = fv.iter().zip(&pv).filter(|(x, y)| x.to_bits() != y.to_bits()).count() + fv.len().abs_diff(pv.len());
    println!("parity: {mm}/{} value-bit mismatches", pv.len());

    // ---- perf: mask[1,1,256,256] into a/b [16,8,256,256] = 8.4M f64 no-grad ----
    let (b_, h_, s_) = (16usize, 8usize, 256usize);
    let n = b_ * h_ * s_ * s_;
    let ci = s_ * s_;
    let md: Vec<f64> = (0..ci).map(|i| if i % 2 == 0 { 1.0 } else { 0.0 }).collect();
    let ad: Vec<f64> = (0..n).map(|i| (i % 97) as f64 * 0.5 - 10.0).collect();
    let bd: Vec<f64> = (0..n).map(|i| (i % 91) as f64 * -0.25 + 3.0).collect();
    let orig = std::env::var("FT_ORIG").is_ok();
    let mut best = f64::INFINITY;
    for _ in 0..9 {
        let mut s = FrankenTorchSession::new(ExecutionMode::Strict);
        let m = s.tensor_variable(md.clone(), vec![1, 1, s_, s_], false).unwrap();
        let a = s.tensor_variable(ad.clone(), vec![b_, h_, s_, s_], false).unwrap();
        let b = s.tensor_variable(bd.clone(), vec![b_, h_, s_, s_], false).unwrap();
        let t = Instant::now();
        let _ = s.tensor_where(m, a, b).unwrap();
        let e = t.elapsed().as_secs_f64() * 1e3;
        if e < best { best = e; }
    }
    let label = if orig { "FT_ORIG(fallthrough)" } else { "FT_FUSED" };

    let py_b = format!(
        r#"
import time,torch
torch.set_num_threads(8)
b,h,s={b_},{h_},{s_}; n=b*h*s*s; ci=s*s
m=((torch.arange(ci,dtype=torch.int64)%2)==0).double().reshape(1,1,s,s)
a=(torch.arange(n,dtype=torch.int64)%97).double()*0.5-10.0; a=a.reshape(b,h,s,s)
bb=(torch.arange(n,dtype=torch.int64)%91).double()*-0.25+3.0; bb=bb.reshape(b,h,s,s)
def t(fn,reps=9):
    for _ in range(2): fn()
    ts=[]
    for _ in range(reps): st=time.perf_counter(); fn(); ts.append((time.perf_counter()-st)*1e3)
    return min(ts)
print("PT where %.4f"%t(lambda:torch.where(m!=0,a,bb)))
"#
    );
    let mut ch = Command::new(&python).arg("-").stdin(Stdio::piped()).stdout(Stdio::piped()).spawn()?;
    ch.stdin.as_mut().unwrap().write_all(py_b.as_bytes())?;
    let pt = String::from_utf8_lossy(&ch.wait_with_output()?.stdout).to_string();
    let ptw = pt.lines().find_map(|l| l.strip_prefix("PT where ")).and_then(|s| s.trim().parse::<f64>().ok()).unwrap_or(f64::NAN);
    let v = |ft: f64, p: f64| if p >= ft { format!("FT {:.2}x FASTER", p / ft) } else { format!("FT {:.2}x SLOWER", ft / p) };
    println!("  where(mask,a,b) {label} {best:.3}ms  PT {ptw:.3}ms  => {}", v(best, ptw));
    Ok(())
}

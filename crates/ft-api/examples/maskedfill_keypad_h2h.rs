//! masked_fill(scores[B,H,S,S], key_pad_mask[B,1,1,S], -inf) — the KEY-PADDING attention mask (batch
//! varies, heads/query broadcast — a MIDDLE/leading broadcast, NOT a trailing tile). Fused general-
//! broadcast path (try_masked_fill_broadcast) vs torch. Set FT_ORIG=1 (fast path disabled + rebuilt)
//! to time the fallthrough (full(value) const + broadcast + where). f64 (f32 fallthrough crashes).
//! Inputs materialized OUTSIDE the timer.
use ft_api::FrankenTorchSession;
use ft_core::ExecutionMode;
use std::io::Write;
use std::process::{Command, Stdio};
use std::time::Instant;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let python = std::env::var("PYTORCH_PYTHON").unwrap_or_else(|_| "python3".to_string());

    // ---- parity: small, mask [b,1,1,s] into scores [b,h,s,s] ----
    let (bb, hh, ss) = (3usize, 2usize, 5usize);
    let n_small = bb * hh * ss * ss;
    let mask_s: Vec<f64> = (0..bb * ss).map(|i| (i % 2) as f64).collect(); // 0/1
    let sc: Vec<f64> = (0..n_small).map(|i| (i as f64) * 0.25 - 30.0).collect();
    let val = -7.25_f64;
    let py_s = format!(
        r#"
import torch
m=torch.tensor({mask_s:?},dtype=torch.float64).reshape({bb},1,1,{ss}).bool()
x=torch.tensor({sc:?},dtype=torch.float64).reshape({bb},{hh},{ss},{ss})
o=x.masked_fill(m,{val}).reshape(-1)
print("VALS"," ".join("%.17g"%v for v in o.tolist()))
"#
    );
    let mut ch = Command::new(&python)
        .arg("-")
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .spawn()?;
    ch.stdin.as_mut().unwrap().write_all(py_s.as_bytes())?;
    let pt = String::from_utf8_lossy(&ch.wait_with_output()?.stdout).to_string();
    let pv: Vec<f64> = pt
        .lines()
        .find_map(|l| l.strip_prefix("VALS "))
        .map(|s| {
            s.split_whitespace()
                .filter_map(|t| t.parse().ok())
                .collect()
        })
        .unwrap_or_default();
    let mut s = FrankenTorchSession::new(ExecutionMode::Strict);
    let m = s.tensor_variable(mask_s.clone(), vec![bb, 1, 1, ss], false)?;
    let x = s.tensor_variable(sc.clone(), vec![bb, hh, ss, ss], false)?;
    let o = s.tensor_masked_fill(x, m, val)?;
    let fv = s.tensor_values(o)?;
    let mm = fv
        .iter()
        .zip(&pv)
        .filter(|(a, b)| a.to_bits() != b.to_bits())
        .count()
        + fv.len().abs_diff(pv.len());
    println!("parity: {mm}/{} value-bit mismatches", pv.len());

    // ---- perf: mask[16,1,1,256] into scores[16,8,256,256] = 8.4M f64 no-grad ----
    let (b_, h_, s_) = (16usize, 8usize, 256usize);
    let n = b_ * h_ * s_ * s_;
    let md: Vec<f64> = (0..b_ * s_).map(|i| (i % 4 == 0) as u8 as f64).collect();
    let xd: Vec<f64> = (0..n).map(|i| (i % 97) as f64 * 0.5 - 10.0).collect();
    let orig = std::env::var("FT_ORIG").is_ok();
    let mut best = f64::INFINITY;
    for _ in 0..9 {
        let mut s = FrankenTorchSession::new(ExecutionMode::Strict);
        let m = s
            .tensor_variable(md.clone(), vec![b_, 1, 1, s_], false)
            .unwrap();
        let x = s
            .tensor_variable(xd.clone(), vec![b_, h_, s_, s_], false)
            .unwrap();
        let t = Instant::now();
        let _ = s.tensor_masked_fill(x, m, val).unwrap();
        let e = t.elapsed().as_secs_f64() * 1e3;
        if e < best {
            best = e;
        }
    }
    let label = if orig {
        "FT_ORIG(fallthrough)"
    } else {
        "FT_FUSED"
    };

    let py_b = format!(
        r#"
import time,torch
torch.set_num_threads(8)
b,h,s={b_},{h_},{s_}; n=b*h*s*s
m=((torch.arange(b*s,dtype=torch.int64)%4)==0).reshape(b,1,1,s)
x=(torch.arange(n,dtype=torch.int64)%97).double()*0.5-10.0; x=x.reshape(b,h,s,s)
def t(fn,reps=9):
    for _ in range(2): fn()
    ts=[]
    for _ in range(reps): st=time.perf_counter(); fn(); ts.append((time.perf_counter()-st)*1e3)
    return min(ts)
print("PT mf %.4f"%t(lambda:x.masked_fill(m,{val})))
"#
    );
    let mut ch = Command::new(&python)
        .arg("-")
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .spawn()?;
    ch.stdin.as_mut().unwrap().write_all(py_b.as_bytes())?;
    let pt = String::from_utf8_lossy(&ch.wait_with_output()?.stdout).to_string();
    let ptw = pt
        .lines()
        .find_map(|l| l.strip_prefix("PT mf "))
        .and_then(|s| s.trim().parse::<f64>().ok())
        .unwrap_or(f64::NAN);
    let v = |ft: f64, p: f64| {
        if p >= ft {
            format!("FT {:.2}x FASTER", p / ft)
        } else {
            format!("FT {:.2}x SLOWER", ft / p)
        }
    };
    println!(
        "  masked_fill(kpad) {label} {best:.3}ms  PT {ptw:.3}ms  => {}",
        v(best, ptw)
    );
    Ok(())
}

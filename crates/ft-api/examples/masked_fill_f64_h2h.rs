//! f64 masked_fill no-grad fast path vs torch (was: f64 fell to full(128MB)+where composed).
//! add_f64 = SIMD anchor. masked_fill(x, mask, value): mask != 0 ? value : x.
use ft_api::FrankenTorchSession;
use ft_core::{DType, ExecutionMode};
use std::io::Write;
use std::process::{Command, Stdio};
use std::time::Instant;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let python = std::env::var("PYTORCH_PYTHON").unwrap_or_else(|_| "python3".to_string());
    let val = -123.5_f64;

    // parity small
    let inp: Vec<f64> = vec![1.0, 2.0, -3.0, 0.0, -0.0, 5.5, 7.0, -8.0];
    let msk: Vec<f64> = vec![0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0];
    let py_s = format!(
        r#"
import torch
a=torch.tensor({inp:?},dtype=torch.float64)
m=torch.tensor({msk:?},dtype=torch.float64).bool()
o=a.masked_fill(m,{val})
print("VALS"," ".join("%.17g"%v for v in o.tolist()))
"#,
        inp = inp,
        msk = msk,
        val = val
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
    let x = s.tensor_variable(inp.clone(), vec![inp.len()], false)?;
    let m = s.tensor_variable(msk.clone(), vec![msk.len()], false)?;
    let o = s.tensor_masked_fill(x, m, val)?;
    let dt = s.tensor_dtype(o)?;
    let fv = s.tensor_values(o)?;
    let mm = fv
        .iter()
        .zip(&pv)
        .filter(|(a, b)| a.to_bits() != b.to_bits())
        .count()
        + fv.len().abs_diff(pv.len());
    println!(
        "parity: dtype={dt:?} (f64={})  value-bit mismatches: {mm}/{}",
        dt == DType::F64,
        pv.len()
    );

    // perf: 16M f64, time only the op
    let n = 16_000_000usize;
    let a: Vec<f64> = (0..n)
        .map(|i| ((i % 9973) as f64 - 5000.0) * 0.001)
        .collect();
    let mv: Vec<f64> = (0..n).map(|i| if i % 3 == 0 { 1.0 } else { 0.0 }).collect();
    let ft_add = {
        let mut best = f64::INFINITY;
        for _ in 0..9 {
            let mut s = FrankenTorchSession::new(ExecutionMode::Strict);
            let x = s.tensor_variable(a.clone(), vec![n], false).unwrap();
            let y = s.tensor_variable(mv.clone(), vec![n], false).unwrap();
            let t = Instant::now();
            let _ = s.tensor_add(x, y);
            let e = t.elapsed().as_secs_f64() * 1e3;
            if e < best {
                best = e;
            }
        }
        best
    };
    let ft_mf = {
        let mut best = f64::INFINITY;
        for _ in 0..9 {
            let mut s = FrankenTorchSession::new(ExecutionMode::Strict);
            let x = s.tensor_variable(a.clone(), vec![n], false).unwrap();
            let m = s.tensor_variable(mv.clone(), vec![n], false).unwrap();
            let t = Instant::now();
            let _ = s.tensor_masked_fill(x, m, val);
            let e = t.elapsed().as_secs_f64() * 1e3;
            if e < best {
                best = e;
            }
        }
        best
    };
    let py_b = format!(
        r#"
import time,torch
torch.set_num_threads(8)
n={n}
a=(((torch.arange(n,dtype=torch.int64)%9973).double()-5000.0)*0.001)
m=((torch.arange(n,dtype=torch.int64)%3)==0)
def t(fn,reps=9):
    for _ in range(2): fn()
    ts=[]
    for _ in range(reps): s=time.perf_counter(); fn(); ts.append((time.perf_counter()-s)*1e3)
    return min(ts)
print("PT add %.4f"%t(lambda:a+a))
print("PT masked_fill %.4f"%t(lambda:a.masked_fill(m,{val})))
"#,
        n = n,
        val = val
    );
    let mut ch = Command::new(&python)
        .arg("-")
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .spawn()?;
    ch.stdin.as_mut().unwrap().write_all(py_b.as_bytes())?;
    let pt = String::from_utf8_lossy(&ch.wait_with_output()?.stdout).to_string();
    let g = |n: &str| {
        pt.lines()
            .find_map(|l| {
                let mut it = l.strip_prefix("PT ")?.split_whitespace();
                if it.next()? == n {
                    it.next()?.parse::<f64>().ok()
                } else {
                    None
                }
            })
            .unwrap_or(f64::NAN)
    };
    let (pa, pm) = (g("add"), g("masked_fill"));
    println!(
        "  add_anchor   FT {ft_add:.3} PT {pa:.3}  => FT {:.2}x {}",
        (pa / ft_add).max(ft_add / pa),
        if pa >= ft_add { "FASTER" } else { "SLOWER" }
    );
    println!(
        "  masked_fill  FT {ft_mf:.3} PT {pm:.3}  => FT {:.2}x {}",
        (pm / ft_mf).max(ft_mf / pm),
        if pm >= ft_mf { "FASTER" } else { "SLOWER" }
    );
    Ok(())
}

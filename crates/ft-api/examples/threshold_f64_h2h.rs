//! f64 threshold no-grad fast path vs torch (was: f64 fell through to ~6-pass composed path).
//! relu_f64 = anchor. threshold(x, t=0.5, v=-1.0): x>t ? x : v ; NaN -> NaN (current torch).
use ft_api::FrankenTorchSession;
use ft_core::{DType, ExecutionMode};
use std::io::Write;
use std::process::{Command, Stdio};
use std::time::Instant;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let python = std::env::var("PYTORCH_PYTHON").unwrap_or_else(|_| "python3".to_string());
    let (th, va) = (0.5_f64, -1.0_f64);

    // parity: finite + boundary
    let small: Vec<f64> = vec![
        0.0, -0.0, 0.5, 0.50001, 0.49999, -0.5, 1.0, -1.0, 2.0, -2.0, 0.3,
    ];
    let py_s = format!(
        r#"
import torch
a=torch.tensor({small:?},dtype=torch.float64)
o=torch.nn.functional.threshold(a,{th},{va})
print("VALS"," ".join("%.17g"%v for v in o.tolist()))
"#,
        small = small,
        th = th,
        va = va
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
    let x = s.tensor_variable(small.clone(), vec![small.len()], false)?;
    let o = s.tensor_threshold(x, th, va)?;
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
        .map(|i| ((i % 9973) as f64 - 5000.0) * 0.0003)
        .collect();
    let op = |which: u8| {
        let mut best = f64::INFINITY;
        for _ in 0..9 {
            let mut s = FrankenTorchSession::new(ExecutionMode::Strict);
            let x = s.tensor_variable(a.clone(), vec![n], false).unwrap();
            let t = Instant::now();
            let _ = if which == 0 {
                s.tensor_relu(x)
            } else {
                s.tensor_threshold(x, th, va)
            };
            let e = t.elapsed().as_secs_f64() * 1e3;
            if e < best {
                best = e;
            }
        }
        best
    };
    let (ft_r, ft_t) = (op(0), op(1));
    let py_b = format!(
        r#"
import time,torch
torch.set_num_threads(8)
n={n}
a=(((torch.arange(n,dtype=torch.int64)%9973).double()-5000.0)*0.0003)
def t(fn,reps=9):
    for _ in range(2): fn()
    ts=[]
    for _ in range(reps): s=time.perf_counter(); fn(); ts.append((time.perf_counter()-s)*1e3)
    return min(ts)
print("PT relu %.4f"%t(lambda:torch.relu(a)))
print("PT threshold %.4f"%t(lambda:torch.nn.functional.threshold(a,{th},{va})))
"#,
        n = n,
        th = th,
        va = va
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
    let (pr, ptv) = (g("relu"), g("threshold"));
    println!(
        "  relu_anchor  FT {ft_r:.3} PT {pr:.3}  => FT {:.2}x {}",
        (pr / ft_r).max(ft_r / pr),
        if pr >= ft_r { "FASTER" } else { "SLOWER" }
    );
    println!(
        "  threshold    FT {ft_t:.3} PT {ptv:.3}  => FT {:.2}x {}",
        (ptv / ft_t).max(ft_t / ptv),
        if ptv >= ft_t { "FASTER" } else { "SLOWER" }
    );
    Ok(())
}

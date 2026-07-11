//! f64 hardshrink no-grad fast path vs torch (was: f64 fell through to the ~5-pass composed
//! path, ~23x SLOWER). relu_f64 = SIMD anchor (gate trust on it FASTER).
use ft_api::FrankenTorchSession;
use ft_core::{DType, ExecutionMode};
use std::io::Write;
use std::process::{Command, Stdio};
use std::time::Instant;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let python = std::env::var("PYTORCH_PYTHON").unwrap_or_else(|_| "python3".to_string());

    // parity: small + dtype check vs torch (lambd 0.5). NaN excluded here (it can't round-trip
    // through the python list literal); NaN-keep matches the bit-exact-verified f32 form (t503).
    let small: Vec<f64> = vec![
        0.0, -0.0, 0.5, -0.5, 0.50001, -0.50001, 1.0, -1.0, 0.3, -0.3, 2.0,
    ];
    let py_s = format!(
        r#"
import torch
a=torch.tensor({small:?},dtype=torch.float64)
o=torch.nn.functional.hardshrink(a,0.5)
print("VALS"," ".join(("nan" if (v!=v) else "%.17g"%v) for v in o.tolist()))
"#,
        small = small
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
                .map(|t| {
                    if t == "nan" {
                        f64::NAN
                    } else {
                        t.parse().unwrap()
                    }
                })
                .collect()
        })
        .unwrap_or_default();
    let mut s = FrankenTorchSession::new(ExecutionMode::Strict);
    let x = s.tensor_variable(small.clone(), vec![small.len()], false)?;
    let o = s.tensor_hardshrink(x, 0.5)?;
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

    // perf: 16M f64
    let n = 16_000_000usize;
    let a: Vec<f64> = (0..n)
        .map(|i| ((i % 9973) as f64 - 5000.0) * 0.0003)
        .collect();
    // Time ONLY the op (tensor_variable / 128MB alloc built BEFORE the timer).
    let ft_relu = {
        let mut best = f64::INFINITY;
        for _ in 0..9 {
            let mut s = FrankenTorchSession::new(ExecutionMode::Strict);
            let x = s.tensor_variable(a.clone(), vec![n], false).unwrap();
            let t = Instant::now();
            let _ = s.tensor_relu(x);
            let e = t.elapsed().as_secs_f64() * 1e3;
            if e < best {
                best = e;
            }
        }
        best
    };
    let ft_hs = {
        let mut best = f64::INFINITY;
        for _ in 0..9 {
            let mut s = FrankenTorchSession::new(ExecutionMode::Strict);
            let x = s.tensor_variable(a.clone(), vec![n], false).unwrap();
            let t = Instant::now();
            let _ = s.tensor_hardshrink(x, 0.5);
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
a=(((torch.arange(n,dtype=torch.int64)%9973).double()-5000.0)*0.0003)
def t(fn,reps=9):
    for _ in range(2): fn()
    ts=[]
    for _ in range(reps): s=time.perf_counter(); fn(); ts.append((time.perf_counter()-s)*1e3)
    return min(ts)
print("PT relu %.4f"%t(lambda:torch.relu(a)))
print("PT hardshrink %.4f"%t(lambda:torch.nn.functional.hardshrink(a,0.5)))
"#,
        n = n
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
    let (ptr, pth) = (g("relu"), g("hardshrink"));
    let vr = ptr / ft_relu;
    let vh = pth / ft_hs;
    println!(
        "  relu_anchor  FT {ft_relu:.3} PT {ptr:.3}  => FT {vr:.2}x {}",
        if vr >= 1.0 { "FASTER" } else { "SLOWER" }
    );
    println!(
        "  hardshrink   FT {ft_hs:.3} PT {pth:.3}  => FT {vh:.2}x {}",
        if vh >= 1.0 { "FASTER" } else { "SLOWER" }
    );
    Ok(())
}

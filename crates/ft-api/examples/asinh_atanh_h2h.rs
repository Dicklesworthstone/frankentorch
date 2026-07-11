//! asinh/atanh F64 no-grad — fused try_f64_unary_native vs torch. Set FT_ORIG=1 (fast path disabled +
//! rebuilt) to time the multi-op compose. 16M f64, inputs OUTSIDE the timer.
use ft_api::FrankenTorchSession;
use ft_core::ExecutionMode;
use std::io::Write;
use std::process::{Command, Stdio};
use std::time::Instant;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let python = std::env::var("PYTORCH_PYTHON").unwrap_or_else(|_| "python3".to_string());
    let n = 16_000_000usize;
    let asinh_x: Vec<f64> = (0..n).map(|i| ((i % 4000) as f64) * 0.01 - 20.0).collect();
    let atanh_x: Vec<f64> = (0..n)
        .map(|i| ((i % 1999) as f64) / 1000.0 - 0.999)
        .collect(); // (-1,1)
    let orig = std::env::var("FT_ORIG").is_ok();
    let run = |which: &str| -> f64 {
        let mut best = f64::INFINITY;
        for _ in 0..7 {
            let mut s = FrankenTorchSession::new(ExecutionMode::Strict);
            let v = if which == "asinh" { &asinh_x } else { &atanh_x };
            let x = s.tensor_variable(v.clone(), vec![n], false).unwrap();
            let t = Instant::now();
            let _ = if which == "asinh" {
                s.tensor_asinh(x).unwrap()
            } else {
                s.tensor_atanh(x).unwrap()
            };
            let e = t.elapsed().as_secs_f64() * 1e3;
            if e < best {
                best = e;
            }
        }
        best
    };
    let (fa, ft2) = (run("asinh"), run("atanh"));
    let label = if orig { "FT_ORIG(compose)" } else { "FT_FUSED" };
    let py = format!(
        r#"
import time,torch
torch.set_num_threads(8)
n={n}
a=((torch.arange(n,dtype=torch.int64)%4000).double())*0.01-20.0
b=((torch.arange(n,dtype=torch.int64)%1999).double())/1000.0-0.999
def t(fn,reps=7):
    for _ in range(2): fn()
    ts=[]
    for _ in range(reps): s=time.perf_counter(); fn(); ts.append((time.perf_counter()-s)*1e3)
    return min(ts)
print("PT asinh %.4f"%t(lambda:torch.asinh(a)))
print("PT atanh %.4f"%t(lambda:torch.atanh(b)))
"#
    );
    let mut ch = Command::new(&python)
        .arg("-")
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .spawn()?;
    ch.stdin.as_mut().unwrap().write_all(py.as_bytes())?;
    let pt = String::from_utf8_lossy(&ch.wait_with_output()?.stdout).to_string();
    let g = |k: &str| {
        pt.lines()
            .find_map(|l| {
                let mut it = l.strip_prefix("PT ")?.split_whitespace();
                if it.next()? == k {
                    it.next()?.parse::<f64>().ok()
                } else {
                    None
                }
            })
            .unwrap_or(f64::NAN)
    };
    let v = |ft: f64, p: f64| {
        if p >= ft {
            format!("FT {:.2}x FASTER", p / ft)
        } else {
            format!("FT {:.2}x SLOWER", ft / p)
        }
    };
    println!(
        "  asinh(16M) {label} {fa:.3}ms  PT {:.3}ms => {}",
        g("asinh"),
        v(fa, g("asinh"))
    );
    println!(
        "  atanh(16M) {label} {ft2:.3}ms  PT {:.3}ms => {}",
        g("atanh"),
        v(ft2, g("atanh"))
    );
    Ok(())
}

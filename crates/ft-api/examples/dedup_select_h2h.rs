//! nanmedian / mode / unique f32 survey vs torch.
use ft_api::FrankenTorchSession;
use ft_core::ExecutionMode;
use std::io::Write;
use std::process::{Command, Stdio};
use std::time::Instant;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let n = 8_000_000usize;
    let med: Vec<f32> = (0..n)
        .map(|i| {
            let v = ((i * 2654435761usize % 1_000_003) as f32) * 0.001 - 500.0;
            if i % 9999 == 0 { f32::NAN } else { v }
        })
        .collect();
    let modev: Vec<f32> = (0..n).map(|i| (i % 1000) as f32).collect();
    let uniq: Vec<f32> = (0..n).map(|i| (i % 100_000) as f32).collect();
    let python = std::env::var("PYTORCH_PYTHON").unwrap_or_else(|_| "python3".to_string());
    let py = r#"
import time,torch
torch.set_num_threads(8)
n=8_000_000
med=(((torch.arange(n,dtype=torch.int64)*2654435761)%1_000_003).float()*0.001-500.0)
med[torch.arange(0,n,9999)]=float('nan')
modev=(torch.arange(n,dtype=torch.int64)%1000).float()
uniq=(torch.arange(n,dtype=torch.int64)%100_000).float()
def t(fn,nn=7):
    for _ in range(2): fn()
    ts=[]
    for _ in range(nn): s=time.perf_counter(); fn(); ts.append((time.perf_counter()-s)*1e3)
    return min(ts)
print("PT nanmedian %.4f"%t(lambda:torch.nanmedian(med)))
print("PT mode %.4f"%t(lambda:torch.mode(modev)))
print("PT unique %.4f"%t(lambda:torch.unique(uniq,sorted=True)))
"#;
    let mut ch = Command::new(&python)
        .arg("-")
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .spawn()?;
    ch.stdin.as_mut().unwrap().write_all(py.as_bytes())?;
    let o = ch.wait_with_output();
    let pt = String::from_utf8_lossy(&o.unwrap().stdout).to_string();
    let rep = |name: &str, ft: f64| {
        if let Some(p) = pt.lines().find_map(|l| {
            let mut it = l.strip_prefix("PT ")?.split_whitespace();
            if it.next()? == name {
                it.next()?.parse::<f64>().ok()
            } else {
                None
            }
        }) {
            let r = p / ft;
            let tag = if r >= 1.0 {
                format!("FT {r:.2}x FASTER")
            } else {
                format!("FT {:.2}x SLOWER", 1.0 / r)
            };
            println!("  {name:<12} {ft:8.3} {p:8.3}   {tag}");
        }
    };
    println!("op            FT(ms)    PT(ms)   verdict");
    {
        let mut best = f64::INFINITY;
        for _ in 0..7 {
            let mut s = FrankenTorchSession::new(ExecutionMode::Strict);
            let x = s.tensor_variable_f32(med.clone(), vec![n], false).unwrap();
            let t = Instant::now();
            let _ = s.tensor_nanmedian(x);
            let e = t.elapsed().as_secs_f64() * 1e3;
            if e < best {
                best = e;
            }
        }
        rep("nanmedian", best);
    }
    {
        let mut best = f64::INFINITY;
        for _ in 0..7 {
            let mut s = FrankenTorchSession::new(ExecutionMode::Strict);
            let x = s
                .tensor_variable_f32(modev.clone(), vec![n], false)
                .unwrap();
            let t = Instant::now();
            let _ = s.tensor_mode(x);
            let e = t.elapsed().as_secs_f64() * 1e3;
            if e < best {
                best = e;
            }
        }
        rep("mode", best);
    }
    {
        let mut best = f64::INFINITY;
        for _ in 0..7 {
            let mut s = FrankenTorchSession::new(ExecutionMode::Strict);
            let x = s.tensor_variable_f32(uniq.clone(), vec![n], false).unwrap();
            let t = Instant::now();
            let _ = s.tensor_unique(x, true, false, false);
            let e = t.elapsed().as_secs_f64() * 1e3;
            if e < best {
                best = e;
            }
        }
        rep("unique", best);
    }
    Ok(())
}

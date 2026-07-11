//! bincount / isin / histogram f32 survey vs torch.
use ft_api::FrankenTorchSession;
use ft_core::ExecutionMode;
use std::io::Write;
use std::process::{Command, Stdio};
use std::time::Instant;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let n = 8_000_000usize;
    let bc: Vec<f32> = (0..n).map(|i| (i % 4096) as f32).collect(); // bincount: ints 0..4096
    let big: Vec<f32> = (0..n)
        .map(|i| ((i * 2654435761usize % 1_000_003) as f32) * 0.001 - 500.0)
        .collect();
    let elements: Vec<f32> = (0..2_000_000).map(|i| (i % 50_000) as f32).collect();
    let test: Vec<f32> = (0..50_000).map(|i| (i * 2) as f32).collect(); // even numbers
    let python = std::env::var("PYTORCH_PYTHON").unwrap_or_else(|_| "python3".to_string());
    let py = r#"
import time,torch
torch.set_num_threads(8)
n=8_000_000
bc=(torch.arange(n,dtype=torch.int64)%4096)
big=(((torch.arange(n,dtype=torch.int64)*2654435761)%1_000_003).float()*0.001-500.0)
elements=(torch.arange(2_000_000,dtype=torch.int64)%50_000).float()
test=(torch.arange(50_000,dtype=torch.int64)*2).float()
def t(fn,n=7):
    for _ in range(2): fn()
    ts=[]
    for _ in range(n): s=time.perf_counter(); fn(); ts.append((time.perf_counter()-s)*1e3)
    return min(ts)
print("PT bincount %.4f"%t(lambda:torch.bincount(bc)))
print("PT isin %.4f"%t(lambda:torch.isin(elements,test)))
print("PT histogram %.4f"%t(lambda:torch.histogram(big,bins=256,range=(-500.0,500.0))))
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
    // bincount
    {
        let mut best = f64::INFINITY;
        for _ in 0..7 {
            let mut s = FrankenTorchSession::new(ExecutionMode::Strict);
            let x = s.tensor_variable_f32(bc.clone(), vec![n], false).unwrap();
            let t = Instant::now();
            let _ = s.tensor_bincount(x, None, 0);
            let e = t.elapsed().as_secs_f64() * 1e3;
            if e < best {
                best = e;
            }
        }
        rep("bincount", best);
    }
    // isin
    {
        let mut best = f64::INFINITY;
        for _ in 0..7 {
            let mut s = FrankenTorchSession::new(ExecutionMode::Strict);
            let el = s
                .tensor_variable_f32(elements.clone(), vec![elements.len()], false)
                .unwrap();
            let te = s
                .tensor_variable_f32(test.clone(), vec![test.len()], false)
                .unwrap();
            let t = Instant::now();
            let _ = s.tensor_isin(el, te);
            let e = t.elapsed().as_secs_f64() * 1e3;
            if e < best {
                best = e;
            }
        }
        rep("isin", best);
    }
    // histogram
    {
        let mut best = f64::INFINITY;
        for _ in 0..7 {
            let mut s = FrankenTorchSession::new(ExecutionMode::Strict);
            let x = s.tensor_variable_f32(big.clone(), vec![n], false).unwrap();
            let t = Instant::now();
            let _ = s.tensor_histogram(x, 256, -500.0, 500.0, None, false);
            let e = t.elapsed().as_secs_f64() * 1e3;
            if e < best {
                best = e;
            }
        }
        rep("histogram", best);
    }
    Ok(())
}

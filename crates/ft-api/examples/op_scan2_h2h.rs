//! Broad multi-op FT vs PyTorch scan ([4000,4000] f64 no-grad) to find the next
//! torch-slow + FT-fixable gap (the method that found count_nonzero + prod/var/std).
//! Run: PYTORCH_PYTHON=/path/to/python cargo run --release -p ft-api --example op_scan2_h2h

use std::io::Write;
use std::process::{Command, Stdio};
use std::time::Instant;

use ft_api::FrankenTorchSession;
use ft_core::ExecutionMode;

const R: usize = 4000;
const C: usize = 4000;

type UnaryOp = fn(&mut FrankenTorchSession, ft_autograd::TensorNodeId);

fn time_ft<F: Fn(&mut FrankenTorchSession, ft_autograd::TensorNodeId)>(data: &[f64], f: F) -> f64 {
    let mut best = f64::INFINITY;
    for _ in 0..5 {
        let mut s = FrankenTorchSession::new(ExecutionMode::Strict);
        let x = s.tensor_variable(data.to_vec(), vec![R, C], false).unwrap();
        let t = Instant::now();
        f(&mut s, x);
        let el = t.elapsed().as_secs_f64() * 1e3;
        if el < best {
            best = el;
        }
    }
    best
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let data: Vec<f64> = (0..R * C).map(|i| ((i % 17) as f64) - 8.0).collect();
    let ops: Vec<(&str, UnaryOp)> = vec![
        ("trace", |s, x| {
            let _ = s.tensor_trace(x);
        }),
        ("diff", |s, x| {
            let _ = s.tensor_diff(x, 1);
        }),
        ("flip", |s, x| {
            let _ = s.tensor_flip(x, &[1]);
        }),
        ("roll", |s, x| {
            let _ = s.tensor_roll(x, 1, 1);
        }),
        ("cumprod", |s, x| {
            let _ = s.tensor_cumprod(x, 1);
        }),
        ("cumsum", |s, x| {
            let _ = s.tensor_cumsum(x, 1);
        }),
        ("cov", |s, x| {
            let _ = s.tensor_cov(x);
        }),
    ];
    let python = std::env::var("PYTORCH_PYTHON").unwrap_or_else(|_| "python3".to_string());
    let py = r#"
import time, torch
torch.set_num_threads(8)
R,C=4000,4000
idx=torch.arange(R*C,dtype=torch.int64)
x=((idx%17).double()-8.0).reshape(R,C)
def t(fn,n=5):
    for _ in range(2):
        try: fn()
        except Exception as e: return float('nan')
    ts=[]
    for _ in range(n):
        s=time.perf_counter(); fn(); ts.append((time.perf_counter()-s)*1e3)
    return min(ts)
for name,fn in [("trace",lambda:torch.trace(x)),("diff",lambda:torch.diff(x,dim=1)),
                ("flip",lambda:torch.flip(x,[1])),("roll",lambda:torch.roll(x,1,1)),
                ("cumprod",lambda:torch.cumprod(x,1)),("cumsum",lambda:torch.cumsum(x,1)),
                ("cov",lambda:torch.cov(x))]:
    print("PT %s %.4f"%(name,t(fn)))
"#;
    let mut child = Command::new(&python)
        .arg("-")
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .spawn()?;
    child
        .stdin
        .as_mut()
        .ok_or_else(|| std::io::Error::other("no stdin"))?
        .write_all(py.as_bytes())?;
    let out = child.wait_with_output();
    let pt = out
        .ok()
        .filter(|o| o.status.success())
        .map(|o| String::from_utf8_lossy(&o.stdout).to_string())
        .unwrap_or_default();
    println!("op (f64 [{R},{C}])  FT(ms)    PT(ms)   ratio(PT/FT, <1=FT slower)");
    for (name, f) in &ops {
        let ftv = time_ft(&data, *f);
        let p = pt.lines().find_map(|l| {
            let mut it = l.strip_prefix("PT ")?.split_whitespace();
            if it.next()? == *name {
                it.next()?.parse::<f64>().ok()
            } else {
                None
            }
        });
        if let Some(p) = p {
            let r = p / ftv;
            let tag = if r >= 1.0 {
                format!("FT {r:.2}x FASTER")
            } else {
                format!("FT {:.2}x SLOWER", 1.0 / r)
            };
            println!("  {name:<8} {ftv:9.3} {p:8.3}   {tag}");
        }
    }
    Ok(())
}

//! group_norm (no-grad) FT vs PyTorch. cat_anchor = landed-win sanity. Build input outside timer.
//! Run: PYTORCH_PYTHON=/path/to/python cargo run --release -p ft-api --example groupnorm_h2h

use std::io::Write;
use std::process::{Command, Stdio};
use std::time::Instant;

use ft_api::FrankenTorchSession;
use ft_core::ExecutionMode;

const N: usize = 16;
const CH: usize = 256;
const H: usize = 64;
const W: usize = 64;
const G: usize = 32;
const R: usize = 4000;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let numel = N * CH * H * W;
    let x: Vec<f64> = (0..numel).map(|i| (i % 9973) as f64 - 4986.0).collect();
    let mat: Vec<f64> = (0..R * R).map(|i| (i % 17) as f64).collect();

    let python = std::env::var("PYTORCH_PYTHON").unwrap_or_else(|_| "python3".to_string());
    let py = r#"
import time, torch
import torch.nn.functional as Fn
torch.set_num_threads(8)
N,CH,H,W,G,R=16,256,64,64,32,4000
x=((torch.arange(N*CH*H*W,dtype=torch.int64)%9973).double()-4986.0).reshape(N,CH,H,W)
m=((torch.arange(R*R,dtype=torch.int64)%17).double()).reshape(R,R)
def t(fn,n=7):
    for _ in range(2):
        try: fn()
        except Exception: return float('nan')
    ts=[]
    for _ in range(n):
        s=time.perf_counter(); fn(); ts.append((time.perf_counter()-s)*1e3)
    return min(ts)
for name,fn in [("cat_anchor",lambda:torch.cat([m,m],1)),("group_norm",lambda:Fn.group_norm(x,G))]:
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
    let report = |name: &str, ftv: f64| {
        if let Some(p) = pt.lines().find_map(|l| {
            let mut it = l.strip_prefix("PT ")?.split_whitespace();
            if it.next()? == name {
                it.next()?.parse::<f64>().ok()
            } else {
                None
            }
        }) {
            let r = p / ftv;
            let tag = if r >= 1.0 {
                format!("FT {r:.2}x FASTER")
            } else {
                format!("FT {:.2}x SLOWER", 1.0 / r)
            };
            println!("  {name:<12} {ftv:8.3} {p:8.3}   {tag}");
        }
    };
    println!("op            FT(ms)    PT(ms)   verdict");
    // cat anchor
    let mut b = f64::INFINITY;
    for _ in 0..7 {
        let mut s = FrankenTorchSession::new(ExecutionMode::Strict);
        let xm = s.tensor_variable(mat.clone(), vec![R, R], false).unwrap();
        let t = Instant::now();
        let _ = s.tensor_cat(&[xm, xm], 1);
        let e = t.elapsed().as_secs_f64() * 1e3;
        if e < b {
            b = e;
        }
    }
    report("cat_anchor", b);
    let mut b = f64::INFINITY;
    for _ in 0..7 {
        let mut s = FrankenTorchSession::new(ExecutionMode::Strict);
        let xn = s
            .tensor_variable(x.clone(), vec![N, CH, H, W], false)
            .unwrap();
        let t = Instant::now();
        let _ = s.functional_group_norm(xn, G, None, None, 1e-5);
        let e = t.elapsed().as_secs_f64() * 1e3;
        if e < b {
            b = e;
        }
    }
    report("group_norm", b);
    Ok(())
}

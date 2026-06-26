//! Embedding lookup (no-grad gather) FT vs PyTorch. cat_anchor = landed-win sanity.
//! Build input/weight OUTSIDE the timer; only the op is timed.
//! Run: PYTORCH_PYTHON=/path/to/python cargo run --release -p ft-api --example embedding_h2h

use std::io::Write;
use std::process::{Command, Stdio};
use std::time::Instant;

use ft_api::FrankenTorchSession;
use ft_core::ExecutionMode;

const N: usize = 250_000; // num indices
const V: usize = 30_000; // vocab
const D: usize = 512; // embedding dim
const R: usize = 4000;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let idx: Vec<f64> = (0..N).map(|i| (i.wrapping_mul(2654435761) % V) as f64).collect();
    let weight: Vec<f64> = (0..V * D).map(|i| (i % 97) as f64 * 0.01).collect();
    let mat: Vec<f64> = (0..R * R).map(|i| (i % 17) as f64).collect();

    let python = std::env::var("PYTORCH_PYTHON").unwrap_or_else(|_| "python3".to_string());
    let py = r#"
import time, torch
torch.set_num_threads(8)
N,V,D,R=250000,30000,512,4000
ix=torch.tensor([(i*2654435761)%V for i in range(N)],dtype=torch.int64)
w=((torch.arange(V*D,dtype=torch.int64)%97).double()*0.01).reshape(V,D)
x=((torch.arange(R*R,dtype=torch.int64)%17).double()).reshape(R,R)
import torch.nn.functional as Fn
def t(fn,n=7):
    for _ in range(2):
        try: fn()
        except Exception: return float('nan')
    ts=[]
    for _ in range(n):
        a=time.perf_counter(); fn(); ts.append((time.perf_counter()-a)*1e3)
    return min(ts)
for name,fn in [("cat_anchor",lambda:torch.cat([x,x],1)),("embedding",lambda:Fn.embedding(ix,w))]:
    print("PT %s %.4f"%(name,t(fn)))
"#;
    let mut child = Command::new(&python).arg("-").stdin(Stdio::piped()).stdout(Stdio::piped()).spawn()?;
    child.stdin.as_mut().ok_or_else(|| std::io::Error::other("no stdin"))?.write_all(py.as_bytes())?;
    let out = child.wait_with_output();
    let pt = out.ok().filter(|o| o.status.success()).map(|o| String::from_utf8_lossy(&o.stdout).to_string()).unwrap_or_default();
    let getpt = |name: &str| pt.lines().find_map(|l| { let mut it=l.strip_prefix("PT ")?.split_whitespace(); if it.next()?==name {it.next()?.parse::<f64>().ok()} else {None} });

    let bench = |label: &str, ftms: f64| {
        if let Some(p) = getpt(label) {
            let r = p / ftms;
            let tag = if r >= 1.0 { format!("FT {r:.2}x FASTER") } else { format!("FT {:.2}x SLOWER", 1.0 / r) };
            println!("  {label:<12} {ftms:8.3} {p:8.3}   {tag}");
        }
    };

    println!("op            FT(ms)    PT(ms)   verdict");
    // cat anchor
    let mut best = f64::INFINITY;
    for _ in 0..7 {
        let mut s = FrankenTorchSession::new(ExecutionMode::Strict);
        let x = s.tensor_variable(mat.clone(), vec![R, R], false).unwrap();
        let t = Instant::now();
        let _ = s.tensor_cat(&[x, x], 1);
        let e = t.elapsed().as_secs_f64() * 1e3; if e < best { best = e; }
    }
    bench("cat_anchor", best);

    let mut best = f64::INFINITY;
    for _ in 0..7 {
        let mut s = FrankenTorchSession::new(ExecutionMode::Strict);
        let ix = s.tensor_variable(idx.clone(), vec![N], false).unwrap();
        let w = s.tensor_variable(weight.clone(), vec![V, D], false).unwrap();
        let t = Instant::now();
        let _ = s.tensor_embedding(ix, w, None);
        let e = t.elapsed().as_secs_f64() * 1e3; if e < best { best = e; }
    }
    bench("embedding", best);
    Ok(())
}

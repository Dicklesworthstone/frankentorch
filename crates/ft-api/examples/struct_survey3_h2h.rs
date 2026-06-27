//! Structural-op survey #3 FT vs PyTorch (materialize ops: one_hot/pad/tile).
//! cat_anchor = landed-win sanity (must show FASTER). Builds input OUTSIDE the timer.
//! Run: PYTORCH_PYTHON=/path/to/python cargo run --release -p ft-api --example struct_survey3_h2h

use std::io::Write;
use std::process::{Command, Stdio};
use std::time::Instant;

use ft_api::FrankenTorchSession;
use ft_autograd::TensorNodeId;
use ft_core::ExecutionMode;

const R: usize = 4000;
const C: usize = 4000;
const ON: usize = 1_000_000; // one_hot rows
const OC: usize = 64; // one_hot classes
const TH: usize = 2000; // tile base

fn timed<B, F>(build: B, op: F) -> f64
where
    B: Fn() -> (FrankenTorchSession, TensorNodeId),
    F: Fn(&mut FrankenTorchSession, TensorNodeId),
{
    let mut best = f64::INFINITY;
    for _ in 0..7 {
        let (mut s, x) = build();
        let t = Instant::now();
        op(&mut s, x);
        let e = t.elapsed().as_secs_f64() * 1e3;
        if e < best { best = e; }
    }
    best
}

fn report(name: &str, ftv: f64, pt: &str) {
    let p = pt.lines().find_map(|l| { let mut it=l.strip_prefix("PT ")?.split_whitespace(); if it.next()?==name {it.next()?.parse::<f64>().ok()} else {None} });
    if let Some(p) = p {
        let r = p / ftv;
        let tag = if r >= 1.0 { format!("FT {r:.2}x FASTER") } else { format!("FT {:.2}x SLOWER", 1.0 / r) };
        println!("  {name:<12} {ftv:8.3} {p:8.3}   {tag}");
    } else {
        println!("  {name:<12} {ftv:8.3}      nan   (torch failed)");
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mat: Vec<f64> = (0..R * C).map(|i| ((i % 17) as f64) - 8.0).collect();
    let idx: Vec<f64> = (0..ON).map(|i| (i % OC) as f64).collect();
    let tbase: Vec<f64> = (0..TH * TH).map(|i| (i % 17) as f64).collect();

    let python = std::env::var("PYTORCH_PYTHON").unwrap_or_else(|_| "python3".to_string());
    let py = r#"
import time, torch
torch.set_num_threads(8)
R,C,ON,OC,TH=4000,4000,1000000,64,2000
x=((torch.arange(R*C,dtype=torch.int64)%17).double()-8.0).reshape(R,C)
ix=(torch.arange(ON,dtype=torch.int64)%OC)
tb=((torch.arange(TH*TH,dtype=torch.int64)%17).double()).reshape(TH,TH)
def t(fn,n=7):
    for _ in range(2):
        try: fn()
        except Exception: return float('nan')
    ts=[]
    for _ in range(n):
        a=time.perf_counter(); fn(); ts.append((time.perf_counter()-a)*1e3)
    return min(ts)
import torch.nn.functional as Fn
for name,fn in [("cat_anchor",lambda:torch.cat([x,x],1)),
                ("one_hot",lambda:Fn.one_hot(ix,OC)),
                ("pad",lambda:Fn.pad(x,(50,50,50,50))),
                ("tile",lambda:tb.tile((2,2)))]:
    print("PT %s %.4f"%(name,t(fn)))
"#;
    let mut child = Command::new(&python).arg("-").stdin(Stdio::piped()).stdout(Stdio::piped()).spawn()?;
    child.stdin.as_mut().ok_or_else(|| std::io::Error::other("no stdin"))?.write_all(py.as_bytes())?;
    let out = child.wait_with_output();
    let pt = out.ok().filter(|o| o.status.success()).map(|o| String::from_utf8_lossy(&o.stdout).to_string()).unwrap_or_default();

    println!("op            FT(ms)    PT(ms)   verdict");
    report("cat_anchor", timed(
        || { let mut s = FrankenTorchSession::new(ExecutionMode::Strict); let x = s.tensor_variable(mat.clone(), vec![R, C], false).unwrap(); (s, x) },
        |s, x| { let _ = s.tensor_cat(&[x, x], 1); }), &pt);
    report("one_hot", timed(
        || { let mut s = FrankenTorchSession::new(ExecutionMode::Strict); let x = s.tensor_variable(idx.clone(), vec![ON], false).unwrap(); (s, x) },
        |s, x| { let _ = s.tensor_one_hot(x, OC); }), &pt);
    report("pad", timed(
        || { let mut s = FrankenTorchSession::new(ExecutionMode::Strict); let x = s.tensor_variable(mat.clone(), vec![R, C], false).unwrap(); (s, x) },
        |s, x| { let _ = s.tensor_pad(x, &[50, 50, 50, 50], 0.0); }), &pt);
    report("tile", timed(
        || { let mut s = FrankenTorchSession::new(ExecutionMode::Strict); let x = s.tensor_variable(tbase.clone(), vec![TH, TH], false).unwrap(); (s, x) },
        |s, x| { let _ = s.tensor_tile(x, &[2, 2]); }), &pt);
    Ok(())
}

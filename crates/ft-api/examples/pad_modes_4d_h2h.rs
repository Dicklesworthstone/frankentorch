//! 4D reflect/replicate/circular pad FT vs PyTorch (torch needs >=3D for these modes).
//! Probes whether the no-grad reflect/replicate gather path (which clones the input via
//! tensor_values) loses to torch — input is [N,C,H,W] f64, pad 8 on each spatial side.
//! `constant_4d` + `cat` are landed-win ANCHORS for worker health.
//! Run: PYTORCH_PYTHON=/path/to/python cargo run --release -p ft-api --example pad_modes_4d_h2h

use std::io::Write;
use std::process::{Command, Stdio};
use std::time::Instant;

use ft_api::FrankenTorchSession;
use ft_core::ExecutionMode;

const N: usize = 8;
const CH: usize = 32;
const H: usize = 256;
const W: usize = 256;
const P: usize = 8;

type UnaryOp = fn(&mut FrankenTorchSession, ft_autograd::TensorNodeId);

fn time_ft<F: Fn(&mut FrankenTorchSession, ft_autograd::TensorNodeId)>(data: &[f64], f: F) -> f64 {
    let mut best = f64::INFINITY;
    for _ in 0..5 {
        let mut s = FrankenTorchSession::new(ExecutionMode::Strict);
        let x = s.tensor_variable(data.to_vec(), vec![N, CH, H, W], false).unwrap();
        let t = Instant::now();
        f(&mut s, x);
        let el = t.elapsed().as_secs_f64() * 1e3;
        if el < best { best = el; }
    }
    best
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let data: Vec<f64> = (0..N * CH * H * W).map(|i| ((i % 17) as f64) - 8.0).collect();
    let ops: Vec<(&str, UnaryOp)> = vec![
        ("cat_anchor", |s, x| { let _ = s.tensor_cat(&[x, x], 1); }),
        ("constant_4d", |s, x| { let _ = s.tensor_pad(x, &[P, P, P, P], 0.0); }),
        ("reflect_4d", |s, x| { let _ = s.tensor_pad_mode(x, &[P, P, P, P], "reflect", 0.0); }),
        ("replicate_4d", |s, x| { let _ = s.tensor_pad_mode(x, &[P, P, P, P], "replicate", 0.0); }),
        ("circular_4d", |s, x| { let _ = s.tensor_pad_mode(x, &[P, P, P, P], "circular", 0.0); }),
    ];
    let python = std::env::var("PYTORCH_PYTHON").unwrap_or_else(|_| "python3".to_string());
    let py = r#"
import time, torch
import torch.nn.functional as F
torch.set_num_threads(8)
N,CH,H,W,P=8,32,256,256,8
idx=torch.arange(N*CH*H*W,dtype=torch.int64)
x=((idx%17).double()-8.0).reshape(N,CH,H,W)
def t(fn,n=5):
    for _ in range(2):
        try: fn()
        except Exception as e: return float('nan')
    ts=[]
    for _ in range(n):
        s=time.perf_counter(); fn(); ts.append((time.perf_counter()-s)*1e3)
    return min(ts)
for name,fn in [("cat_anchor",lambda:torch.cat([x,x],1)),
                ("constant_4d",lambda:F.pad(x,(P,P,P,P),mode='constant',value=0.0)),
                ("reflect_4d",lambda:F.pad(x,(P,P,P,P),mode='reflect')),
                ("replicate_4d",lambda:F.pad(x,(P,P,P,P),mode='replicate')),
                ("circular_4d",lambda:F.pad(x,(P,P,P,P),mode='circular'))]:
    print("PT %s %.4f"%(name,t(fn)))
"#;
    let mut child = Command::new(&python).arg("-").stdin(Stdio::piped()).stdout(Stdio::piped()).spawn()?;
    child.stdin.as_mut().ok_or_else(|| std::io::Error::other("no stdin"))?.write_all(py.as_bytes())?;
    let out = child.wait_with_output();
    let pt = out.ok().filter(|o| o.status.success()).map(|o| String::from_utf8_lossy(&o.stdout).to_string()).unwrap_or_default();
    println!("op            FT(ms)    PT(ms)   ratio(PT/FT, <1=FT slower)");
    for (name, f) in &ops {
        let ftv = time_ft(&data, *f);
        let p = pt.lines().find_map(|l| {
            let mut it = l.strip_prefix("PT ")?.split_whitespace();
            if it.next()? == *name { it.next()?.parse::<f64>().ok() } else { None }
        });
        if let Some(p) = p {
            let r = p / ftv;
            let tag = if r >= 1.0 { format!("FT {r:.2}x FASTER") } else { format!("FT {:.2}x SLOWER", 1.0 / r) };
            println!("  {name:<14} {ftv:8.3} {p:8.3}   {tag}");
        }
    }
    Ok(())
}

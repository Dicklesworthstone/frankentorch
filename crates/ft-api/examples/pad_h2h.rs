//! Padding op scan FT vs PyTorch ([4000,4000] f64 no-grad, pad 16 each side) — hunting the
//! per-element division-unravel anti-pattern in constant/reflect/replicate pad. `cat` is a
//! landed-win ANCHOR for worker-health (discard the run if it regresses far from ~3-6x).
//! Run: PYTORCH_PYTHON=/path/to/python cargo run --release -p ft-api --example pad_h2h

use std::io::Write;
use std::process::{Command, Stdio};
use std::time::Instant;

use ft_api::FrankenTorchSession;
use ft_core::ExecutionMode;

const R: usize = 4000;
const C: usize = 4000;
const P: usize = 16;

type UnaryOp = fn(&mut FrankenTorchSession, ft_autograd::TensorNodeId);

fn time_ft<F: Fn(&mut FrankenTorchSession, ft_autograd::TensorNodeId)>(data: &[f64], f: F) -> f64 {
    let mut best = f64::INFINITY;
    for _ in 0..7 {
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

fn time_ft_f32_constant_pad() -> f64 {
    let data: Vec<f32> = (0..R * C).map(|i| ((i % 17) as f32) - 8.0).collect();
    let mut best = f64::INFINITY;
    for _ in 0..7 {
        let mut s = FrankenTorchSession::new(ExecutionMode::Strict);
        let x = s
            .tensor_variable_f32(data.clone(), vec![R, C], false)
            .unwrap();
        let t = Instant::now();
        let _ = s.tensor_pad(x, &[P, P, P, P], 0.0);
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
        ("cat_anchor", |s, x| {
            let _ = s.tensor_cat(&[x, x], 1);
        }),
        ("constant_pad", |s, x| {
            let _ = s.tensor_pad(x, &[P, P, P, P], 0.0);
        }),
        ("reflect_pad", |s, x| {
            let _ = s.tensor_pad_mode(x, &[P, P, P, P], "reflect", 0.0);
        }),
        ("replicate_pad", |s, x| {
            let _ = s.tensor_pad_mode(x, &[P, P, P, P], "replicate", 0.0);
        }),
    ];
    let python = std::env::var("PYTORCH_PYTHON").unwrap_or_else(|_| "python3".to_string());
    let py = r#"
import time, torch
import torch.nn.functional as F
torch.set_num_threads(8)
R,C,P=4000,4000,16
idx=torch.arange(R*C,dtype=torch.int64)
x=((idx%17).double()-8.0).reshape(R,C)
xf=((idx%17).float()-8.0).reshape(R,C)
def t(fn,n=7):
    for _ in range(2):
        try: fn()
        except Exception as e: return float('nan')
    ts=[]
    for _ in range(n):
        s=time.perf_counter(); fn(); ts.append((time.perf_counter()-s)*1e3)
    return min(ts)
for name,fn in [("cat_anchor",lambda:torch.cat([x,x],1)),
                ("constant_pad",lambda:F.pad(x,(P,P,P,P),mode='constant',value=0.0)),
                ("constant_pad_f32",lambda:F.pad(xf,(P,P,P,P),mode='constant',value=0.0)),
                ("reflect_pad",lambda:F.pad(x,(P,P,P,P),mode='reflect')),
                ("replicate_pad",lambda:F.pad(x,(P,P,P,P),mode='replicate'))]:
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
    println!("op            FT(ms)    PT(ms)   ratio(PT/FT, <1=FT slower)");
    let lookup = |name: &str| -> Option<f64> {
        pt.lines().find_map(|l| {
            let mut it = l.strip_prefix("PT ")?.split_whitespace();
            if it.next()? == name {
                it.next()?.parse::<f64>().ok()
            } else {
                None
            }
        })
    };
    let report = |name: &str, ftv: f64, p: Option<f64>| {
        if let Some(p) = p {
            let r = p / ftv;
            let tag = if r >= 1.0 {
                format!("FT {r:.2}x FASTER")
            } else {
                format!("FT {:.2}x SLOWER", 1.0 / r)
            };
            println!("  {name:<16} {ftv:8.3} {p:8.3}   {tag}");
        }
    };
    for (name, f) in &ops {
        let ftv = time_ft(&data, *f);
        report(name, ftv, lookup(name));
    }
    report(
        "constant_pad_f32",
        time_ft_f32_constant_pad(),
        lookup("constant_pad_f32"),
    );
    Ok(())
}

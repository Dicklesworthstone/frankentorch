//! Structural f32 survey: masked_fill/pad/rot90/diag_embed vs torch.
use ft_api::FrankenTorchSession;
use ft_autograd::TensorNodeId;
use ft_core::ExecutionMode;
use std::io::Write;
use std::process::{Command, Stdio};
use std::time::Instant;
const R: usize = 4000;
const C: usize = 4000;

fn t1<F: Fn(&mut FrankenTorchSession, TensorNodeId)>(a: &[f32], shape: Vec<usize>, f: F) -> f64 {
    let mut best = f64::INFINITY;
    for _ in 0..7 {
        let mut s = FrankenTorchSession::new(ExecutionMode::Strict);
        let x = s
            .tensor_variable_f32(a.to_vec(), shape.clone(), false)
            .unwrap();
        let t = Instant::now();
        f(&mut s, x);
        let e = t.elapsed().as_secs_f64() * 1e3;
        if e < best {
            best = e;
        }
    }
    best
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let a: Vec<f32> = (0..R * C).map(|i| (i % 1000) as f32 * 0.01).collect();
    let mask: Vec<f32> = (0..R * C).map(|i| (i % 2) as f32).collect();
    let de: Vec<f32> = (0..200 * 500).map(|i| (i % 97) as f32 * 0.1).collect();
    let python = std::env::var("PYTORCH_PYTHON").unwrap_or_else(|_| "python3".to_string());
    let py = r#"
import time,torch
torch.set_num_threads(8)
import torch.nn.functional as F
R,C=4000,4000
a=((torch.arange(R*C,dtype=torch.int64)%1000).float()*0.01).reshape(R,C)
mask=((torch.arange(R*C,dtype=torch.int64)%2)==1).reshape(R,C)
de=((torch.arange(200*500,dtype=torch.int64)%97).float()*0.1).reshape(200,500)
def t(fn,n=7):
    for _ in range(2): fn()
    ts=[]
    for _ in range(n): s=time.perf_counter(); fn(); ts.append((time.perf_counter()-s)*1e3)
    return min(ts)
for name,fn in [("add_anchor",lambda:a+a),
                ("masked_fill",lambda:a.masked_fill(mask,-1.5)),
                ("pad",lambda:F.pad(a,(8,8,8,8),value=2.0)),
                ("rot90",lambda:torch.rot90(a,1,[0,1])),
                ("diag_embed",lambda:torch.diag_embed(de))]:
    print("PT %s %.4f"%(name,t(fn)))
"#;
    let mut ch = Command::new(&python)
        .arg("-")
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .spawn()?;
    ch.stdin.as_mut().unwrap().write_all(py.as_bytes())?;
    let o = ch.wait_with_output();
    let pt = String::from_utf8_lossy(&o.unwrap().stdout).to_string();
    let rep = |n: &str, ft: f64| {
        if let Some(p) = pt.lines().find_map(|l| {
            let mut it = l.strip_prefix("PT ")?.split_whitespace();
            if it.next()? == n {
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
            println!("  {n:<12} {ft:8.3} {p:8.3}   {tag}");
        }
    };
    println!("op            FT(ms)    PT(ms)   verdict");
    rep(
        "add_anchor",
        t1(&a, vec![R, C], |s, x| {
            let _ = s.tensor_add(x, x);
        }),
    );
    {
        let mut best = f64::INFINITY;
        for _ in 0..7 {
            let mut s = FrankenTorchSession::new(ExecutionMode::Strict);
            let x = s.tensor_variable_f32(a.clone(), vec![R, C], false).unwrap();
            let mk = s
                .tensor_variable_f32(mask.clone(), vec![R, C], false)
                .unwrap();
            let t = Instant::now();
            let _ = s.tensor_masked_fill(x, mk, -1.5);
            let e = t.elapsed().as_secs_f64() * 1e3;
            if e < best {
                best = e;
            }
        }
        rep("masked_fill", best);
    }
    rep(
        "pad",
        t1(&a, vec![R, C], |s, x| {
            let _ = s.tensor_pad(x, &[8, 8, 8, 8], 2.0);
        }),
    );
    rep(
        "rot90",
        t1(&a, vec![R, C], |s, x| {
            let _ = s.tensor_rot90(x, 1, [0, 1]);
        }),
    );
    rep(
        "diag_embed",
        t1(&de, vec![200, 500], |s, x| {
            let _ = s.tensor_diag_embed(x, 0);
        }),
    );
    Ok(())
}

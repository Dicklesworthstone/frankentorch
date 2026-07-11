//! prod/var/std on f32 [4000,4000] FT vs PyTorch — confirms the f32 no-grad bypass +
//! parallel kernel close the same 45-49x gap the f64 fix did.
//! Run: PYTORCH_PYTHON=/path/to/python cargo run --release -p ft-api --example reduction_f32_h2h

use std::io::Write;
use std::process::{Command, Stdio};
use std::time::Instant;

use ft_api::FrankenTorchSession;
use ft_core::ExecutionMode;

const R: usize = 4000;
const C: usize = 4000;

type UnaryOp = fn(&mut FrankenTorchSession, ft_autograd::TensorNodeId);

fn time_ft<F: Fn(&mut FrankenTorchSession, ft_autograd::TensorNodeId) -> ()>(
    data: &[f32],
    f: F,
) -> f64 {
    let mut best = f64::INFINITY;
    for _ in 0..6 {
        let mut s = FrankenTorchSession::new(ExecutionMode::Strict);
        let x = s
            .tensor_variable_f32(data.to_vec(), vec![R, C], false)
            .unwrap();
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
    let data: Vec<f32> = (0..R * C).map(|i| ((i % 17) as f32) - 8.0).collect();
    let ops: Vec<(&str, UnaryOp)> = vec![
        ("prod", |s, x| {
            let _ = s.tensor_prod(x);
        }),
        ("var", |s, x| {
            let _ = s.tensor_var(x, 1);
        }),
        ("std", |s, x| {
            let _ = s.tensor_std(x, 1);
        }),
    ];
    let python = std::env::var("PYTORCH_PYTHON").unwrap_or_else(|_| "python3".to_string());
    let py = r#"
import time, torch
torch.set_num_threads(8)
R,C=4000,4000
idx=torch.arange(R*C,dtype=torch.int64)
x=((idx%17).float()-8.0).reshape(R,C)
def t(fn,n=6):
    for _ in range(2): fn()
    ts=[]
    for _ in range(n):
        s=time.perf_counter(); fn(); ts.append((time.perf_counter()-s)*1e3)
    return min(ts)
for name,fn in [("prod",lambda:torch.prod(x)),("var",lambda:torch.var(x)),("std",lambda:torch.std(x))]:
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
    println!("op (f32 [{R},{C}])   FT(ms)   PT(ms)   ratio");
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
            println!("  {name:<6} {ftv:10.3} {p:8.3}   {tag}");
        }
    }
    Ok(())
}

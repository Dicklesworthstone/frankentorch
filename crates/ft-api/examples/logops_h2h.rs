//! Two-input log-family op scan FT vs PyTorch ([4000,4000] f64 no-grad). Both operands
//! POSITIVE (xlogy/xlog1py need y>0). Hunting composed / apply_function serial+clone ops.
//! Run: PYTORCH_PYTHON=/path/to/python cargo run --release -p ft-api --example logops_h2h

use std::io::Write;
use std::process::{Command, Stdio};
use std::time::Instant;

use ft_api::FrankenTorchSession;
use ft_core::ExecutionMode;

const R: usize = 4000;
const C: usize = 4000;

fn time_ft<
    F: Fn(&mut FrankenTorchSession, ft_autograd::TensorNodeId, ft_autograd::TensorNodeId),
>(
    a: &[f64],
    b: &[f64],
    f: F,
) -> f64 {
    let mut best = f64::INFINITY;
    for _ in 0..7 {
        let mut s = FrankenTorchSession::new(ExecutionMode::Strict);
        let x = s.tensor_variable(a.to_vec(), vec![R, C], false).unwrap();
        let y = s.tensor_variable(b.to_vec(), vec![R, C], false).unwrap();
        let t = Instant::now();
        f(&mut s, x, y);
        let el = t.elapsed().as_secs_f64() * 1e3;
        if el < best {
            best = el;
        }
    }
    best
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let a: Vec<f64> = (0..R * C).map(|i| 0.2 + ((i % 11) as f64) * 0.3).collect();
    let b: Vec<f64> = (0..R * C).map(|i| 0.3 + ((i % 7) as f64) * 0.4).collect();
    type Op = fn(&mut FrankenTorchSession, ft_autograd::TensorNodeId, ft_autograd::TensorNodeId);
    let ops: Vec<(&str, Op)> = vec![
        ("cat_anchor", |s, x, _| {
            let _ = s.tensor_cat(&[x, x], 1);
        }),
        ("xlogy", |s, x, y| {
            let _ = s.tensor_xlogy(x, y);
        }),
        ("xlog1py", |s, x, y| {
            let _ = s.tensor_xlog1py(x, y);
        }),
        ("logaddexp", |s, x, y| {
            let _ = s.tensor_logaddexp(x, y);
        }),
        ("logaddexp2", |s, x, y| {
            let _ = s.tensor_logaddexp2(x, y);
        }),
    ];
    let python = std::env::var("PYTORCH_PYTHON").unwrap_or_else(|_| "python3".to_string());
    let py = r#"
import time, torch
torch.set_num_threads(8)
R,C=4000,4000
idx=torch.arange(R*C,dtype=torch.int64)
x=(0.2+(idx%11).double()*0.3).reshape(R,C)
y=(0.3+(idx%7).double()*0.4).reshape(R,C)
def t(fn,n=7):
    for _ in range(2):
        try: fn()
        except Exception as e: return float('nan')
    ts=[]
    for _ in range(n):
        s=time.perf_counter(); fn(); ts.append((time.perf_counter()-s)*1e3)
    return min(ts)
for name,fn in [("cat_anchor",lambda:torch.cat([x,x],1)),
                ("xlogy",lambda:torch.xlogy(x,y)),
                ("xlog1py",lambda:torch.special.xlog1py(x,y)),
                ("logaddexp",lambda:torch.logaddexp(x,y)),
                ("logaddexp2",lambda:torch.logaddexp2(x,y))]:
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
    for (name, f) in &ops {
        let ftv = time_ft(&a, &b, *f);
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
            println!("  {name:<14} {ftv:8.3} {p:8.3}   {tag}");
        }
    }
    Ok(())
}

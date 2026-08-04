//! cummin (flattened [16M]) FT vs PyTorch. cat_anchor = landed-win sanity.
//! Run: PYTORCH_PYTHON=/path/to/python cargo run --release -p ft-api --example cummin_h2h

use std::io::Write;
use std::process::{Command, Stdio};
use std::time::Instant;

use ft_api::FrankenTorchSession;
use ft_autograd::TensorNodeId;
use ft_core::ExecutionMode;

const R: usize = 4000;
const C: usize = 4000;

fn timed<F: Fn(&mut FrankenTorchSession, TensorNodeId)>(mat: &[f64], f: F) -> f64 {
    let mut best = f64::INFINITY;
    for _ in 0..7 {
        let mut s = FrankenTorchSession::new(ExecutionMode::Strict);
        let x = s.tensor_variable(mat.to_vec(), vec![R, C], false).unwrap();
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
    let mat: Vec<f64> = (0..R * C)
        .map(|i| ((i.wrapping_mul(2654435761) % 9973) as f64) - 4986.0)
        .collect();
    let python = std::env::var("PYTORCH_PYTHON").unwrap_or_else(|_| "python3".to_string());
    let py = r#"
import time, torch
torch.set_num_threads(8)
R,C=4000,4000
x=(((torch.arange(R*C,dtype=torch.int64)*2654435761)%9973).double()-4986.0).reshape(R,C)
def t(fn,n=7):
    for _ in range(2):
        try: fn()
        except Exception: return float('nan')
    ts=[]
    for _ in range(n):
        s=time.perf_counter(); fn(); ts.append((time.perf_counter()-s)*1e3)
    return min(ts)
for name,fn in [("cat_anchor",lambda:torch.cat([x,x],1)),("cummin",lambda:torch.cummin(x.reshape(-1),0))]:
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
    report(
        "cat_anchor",
        timed(&mat, |s, x| {
            let _ = s.tensor_cat(&[x, x], 1);
        }),
    );
    report(
        "cummin",
        timed(&mat, |s, x| {
            let _ = s.tensor_cummin(x);
        }),
    );
    Ok(())
}

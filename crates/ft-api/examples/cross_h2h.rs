//! Batched cross-product FT vs PyTorch ([N,3] f64 no-grad). FT's batched cross composes
//! narrow/mul/sub/cat (multi-pass); the bilinear compute is trivial. `cat` is a landed-win ANCHOR.
//! Run: PYTORCH_PYTHON=/path/to/python cargo run --release -p ft-api --example cross_h2h

use std::io::Write;
use std::process::{Command, Stdio};
use std::time::Instant;

use ft_api::FrankenTorchSession;
use ft_core::ExecutionMode;

const N: usize = 5_000_000;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let a: Vec<f64> = (0..N * 3).map(|i| ((i % 17) as f64) - 8.0).collect();
    let b: Vec<f64> = (0..N * 3).map(|i| ((i % 13) as f64) - 6.0).collect();
    let anchor: Vec<f64> = (0..4_000_000).map(|i| (i % 7) as f64).collect();

    // FT cross
    let mut best = f64::INFINITY;
    for _ in 0..7 {
        let mut s = FrankenTorchSession::new(ExecutionMode::Strict);
        let x = s.tensor_variable(a.clone(), vec![N, 3], false).unwrap();
        let y = s.tensor_variable(b.clone(), vec![N, 3], false).unwrap();
        let t = Instant::now();
        let _ = s.tensor_cross(x, y);
        let el = t.elapsed().as_secs_f64() * 1e3;
        if el < best {
            best = el;
        }
    }
    // FT anchor (cat)
    let mut anchor_ms = f64::INFINITY;
    for _ in 0..7 {
        let mut s = FrankenTorchSession::new(ExecutionMode::Strict);
        let x = s
            .tensor_variable(anchor.clone(), vec![1_000_000, 4], false)
            .unwrap();
        let t = Instant::now();
        let _ = s.tensor_cat(&[x, x], 1);
        let el = t.elapsed().as_secs_f64() * 1e3;
        if el < anchor_ms {
            anchor_ms = el;
        }
    }

    let python = std::env::var("PYTORCH_PYTHON").unwrap_or_else(|_| "python3".to_string());
    let py = r#"
import time, torch
torch.set_num_threads(8)
N=5_000_000
idx=torch.arange(N*3,dtype=torch.int64)
a=((idx%17).double()-8.0).reshape(N,3)
b=((idx%13).double()-6.0).reshape(N,3)
ai=torch.arange(4_000_000,dtype=torch.int64)
anc=(ai%7).double().reshape(1_000_000,4)
def t(fn,n=7):
    for _ in range(2):
        try: fn()
        except Exception as e: return float('nan')
    ts=[]
    for _ in range(n):
        s=time.perf_counter(); fn(); ts.append((time.perf_counter()-s)*1e3)
    return min(ts)
print("PT cross %.4f"%t(lambda:torch.linalg.cross(a,b,dim=1)))
print("PT cat_anchor %.4f"%t(lambda:torch.cat([anc,anc],1)))
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
    let lk = |name: &str| -> Option<f64> {
        pt.lines().find_map(|l| {
            let mut it = l.strip_prefix("PT ")?.split_whitespace();
            if it.next()? == name {
                it.next()?.parse().ok()
            } else {
                None
            }
        })
    };
    println!("op            FT(ms)    PT(ms)   ratio(PT/FT, <1=FT slower)");
    for (name, ftv) in [("cross", best), ("cat_anchor", anchor_ms)] {
        if let Some(p) = lk(name) {
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

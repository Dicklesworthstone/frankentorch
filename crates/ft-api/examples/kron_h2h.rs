//! Kronecker product (no-grad) FT vs PyTorch. cat_anchor = landed-win sanity.
//! Run: PYTORCH_PYTHON=/path/to/python cargo run --release -p ft-api --example kron_h2h

use std::io::Write;
use std::process::{Command, Stdio};
use std::time::Instant;

use ft_api::FrankenTorchSession;
use ft_core::ExecutionMode;

const AR: usize = 200;
const AC: usize = 200;
const BR: usize = 20;
const BC: usize = 20;
const R: usize = 4000;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let a: Vec<f64> = (0..AR * AC).map(|i| (i % 17) as f64 - 8.0).collect();
    let b: Vec<f64> = (0..BR * BC).map(|i| (i % 13) as f64 - 6.0).collect();
    let mat: Vec<f64> = (0..R * R).map(|i| (i % 17) as f64).collect();

    let python = std::env::var("PYTORCH_PYTHON").unwrap_or_else(|_| "python3".to_string());
    let py = r#"
import time, torch
torch.set_num_threads(8)
AR,AC,BR,BC,R=200,200,20,20,4000
a=((torch.arange(AR*AC,dtype=torch.int64)%17).double()-8.0).reshape(AR,AC)
b=((torch.arange(BR*BC,dtype=torch.int64)%13).double()-6.0).reshape(BR,BC)
x=((torch.arange(R*R,dtype=torch.int64)%17).double()).reshape(R,R)
def t(fn,n=7):
    for _ in range(2):
        try: fn()
        except Exception: return float('nan')
    ts=[]
    for _ in range(n):
        s=time.perf_counter(); fn(); ts.append((time.perf_counter()-s)*1e3)
    return min(ts)
for name,fn in [("cat_anchor",lambda:torch.cat([x,x],1)),("kron",lambda:torch.kron(a,b))]:
    print("PT %s %.4f"%(name,t(fn)))
"#;
    let mut child = Command::new(&python).arg("-").stdin(Stdio::piped()).stdout(Stdio::piped()).spawn()?;
    child.stdin.as_mut().ok_or_else(|| std::io::Error::other("no stdin"))?.write_all(py.as_bytes())?;
    let out = child.wait_with_output();
    let pt = out.ok().filter(|o| o.status.success()).map(|o| String::from_utf8_lossy(&o.stdout).to_string()).unwrap_or_default();
    let getpt = |name: &str| pt.lines().find_map(|l| { let mut it=l.strip_prefix("PT ")?.split_whitespace(); if it.next()?==name {it.next()?.parse::<f64>().ok()} else {None} });
    let report = |name: &str, ftms: f64| {
        if let Some(p) = getpt(name) {
            let r = p / ftms;
            let tag = if r >= 1.0 { format!("FT {r:.2}x FASTER") } else { format!("FT {:.2}x SLOWER", 1.0 / r) };
            println!("  {name:<12} {ftms:8.3} {p:8.3}   {tag}");
        }
    };

    println!("op            FT(ms)    PT(ms)   verdict");
    let mut best = f64::INFINITY;
    for _ in 0..7 {
        let mut s = FrankenTorchSession::new(ExecutionMode::Strict);
        let x = s.tensor_variable(mat.clone(), vec![R, R], false).unwrap();
        let t = Instant::now(); let _ = s.tensor_cat(&[x, x], 1);
        let e = t.elapsed().as_secs_f64() * 1e3; if e < best { best = e; }
    }
    report("cat_anchor", best);

    let mut best = f64::INFINITY;
    for _ in 0..7 {
        let mut s = FrankenTorchSession::new(ExecutionMode::Strict);
        let av = s.tensor_variable(a.clone(), vec![AR, AC], false).unwrap();
        let bv = s.tensor_variable(b.clone(), vec![BR, BC], false).unwrap();
        let t = Instant::now(); let _ = s.tensor_kron(av, bv);
        let e = t.elapsed().as_secs_f64() * 1e3; if e < best { best = e; }
    }
    report("kron", best);
    Ok(())
}

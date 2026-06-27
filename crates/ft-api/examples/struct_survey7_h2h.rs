//! Survey #7 FT vs PyTorch: transformer-hot per-lane ops (softmax/log_softmax/layer_norm/rms_norm).
//! cat_anchor = landed-win sanity. Builds input OUTSIDE the timer.
//! Run: PYTORCH_PYTHON=/path/to/python cargo run --release -p ft-api --example struct_survey7_h2h

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
        if e < best { best = e; }
    }
    best
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mat: Vec<f64> = (0..R * C).map(|i| ((i.wrapping_mul(2654435761) % 9973) as f64) - 4986.0).collect();
    let python = std::env::var("PYTORCH_PYTHON").unwrap_or_else(|_| "python3".to_string());
    let py = r#"
import time, torch
import torch.nn.functional as Fn
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
for name,fn in [("cat_anchor",lambda:torch.cat([x,x],1)),
                ("softmax",lambda:torch.softmax(x,dim=1)),
                ("log_softmax",lambda:torch.log_softmax(x,dim=1)),
                ("layer_norm",lambda:Fn.layer_norm(x,[C])),
                ("rms_norm",lambda:Fn.rms_norm(x,[C]))]:
    print("PT %s %.4f"%(name,t(fn)))
"#;
    let mut child = Command::new(&python).arg("-").stdin(Stdio::piped()).stdout(Stdio::piped()).spawn()?;
    child.stdin.as_mut().ok_or_else(|| std::io::Error::other("no stdin"))?.write_all(py.as_bytes())?;
    let out = child.wait_with_output();
    let pt = out.ok().filter(|o| o.status.success()).map(|o| String::from_utf8_lossy(&o.stdout).to_string()).unwrap_or_default();
    let report = |name: &str, ftv: f64| {
        if let Some(p) = pt.lines().find_map(|l| { let mut it=l.strip_prefix("PT ")?.split_whitespace(); if it.next()?==name {it.next()?.parse::<f64>().ok()} else {None} }) {
            let r = p / ftv;
            let tag = if r >= 1.0 { format!("FT {r:.2}x FASTER") } else { format!("FT {:.2}x SLOWER", 1.0 / r) };
            println!("  {name:<12} {ftv:8.3} {p:8.3}   {tag}");
        } else { println!("  {name:<12} {ftv:8.3}      nan"); }
    };
    println!("op            FT(ms)    PT(ms)   verdict");
    report("cat_anchor", timed(&mat, |s, x| { let _ = s.tensor_cat(&[x, x], 1); }));
    report("softmax", timed(&mat, |s, x| { let _ = s.tensor_softmax(x, 1); }));
    report("log_softmax", timed(&mat, |s, x| { let _ = s.tensor_log_softmax(x, 1); }));
    report("layer_norm", timed(&mat, |s, x| { let _ = s.functional_layer_norm(x, vec![C], None, None, 1e-5); }));
    report("rms_norm", timed(&mat, |s, x| { let _ = s.functional_rms_norm(x, vec![C], None, 1e-5); }));
    Ok(())
}

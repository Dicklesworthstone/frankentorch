//! More special-function scan FT vs PyTorch ([4000,4000] f64 no-grad). Unary input in (0,1)
//! (safe for logit/ndtri/entr). `cat` is a landed-win ANCHOR.
//! Run: PYTORCH_PYTHON=/path/to/python cargo run --release -p ft-api --example special2_h2h

use std::io::Write;
use std::process::{Command, Stdio};
use std::time::Instant;

use ft_api::FrankenTorchSession;
use ft_core::ExecutionMode;

const R: usize = 4000;
const C: usize = 4000;

fn time_ft<F: Fn(&mut FrankenTorchSession, ft_autograd::TensorNodeId)>(a: &[f64], f: F) -> f64 {
    let mut best = f64::INFINITY;
    for _ in 0..7 {
        let mut s = FrankenTorchSession::new(ExecutionMode::Strict);
        let x = s.tensor_variable(a.to_vec(), vec![R, C], false).unwrap();
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
    let a: Vec<f64> = (0..R * C).map(|i| 0.05 + ((i % 9) as f64) * 0.1).collect();
    type Op = fn(&mut FrankenTorchSession, ft_autograd::TensorNodeId);
    let ops: Vec<(&str, Op)> = vec![
        ("cat_anchor", |s, x| {
            let _ = s.tensor_cat(&[x, x], 1);
        }),
        ("logit", |s, x| {
            let _ = s.tensor_logit(x, None);
        }),
        ("i1", |s, x| {
            let _ = s.tensor_i1(x);
        }),
        ("ndtr", |s, x| {
            let _ = s.tensor_ndtr(x);
        }),
        ("ndtri", |s, x| {
            let _ = s.tensor_ndtri(x);
        }),
        ("erfcx", |s, x| {
            let _ = s.tensor_erfcx(x);
        }),
        ("entr", |s, x| {
            let _ = s.tensor_entr(x);
        }),
    ];
    let python = std::env::var("PYTORCH_PYTHON").unwrap_or_else(|_| "python3".to_string());
    let py = r#"
import time, torch
torch.set_num_threads(8)
R,C=4000,4000
idx=torch.arange(R*C,dtype=torch.int64)
x=(0.05+(idx%9).double()*0.1).reshape(R,C)
def t(fn,n=7):
    for _ in range(2):
        try: fn()
        except Exception as e: return float('nan')
    ts=[]
    for _ in range(n):
        s=time.perf_counter(); fn(); ts.append((time.perf_counter()-s)*1e3)
    return min(ts)
for name,fn in [("cat_anchor",lambda:torch.cat([x,x],1)),
                ("logit",lambda:torch.special.logit(x)),
                ("i1",lambda:torch.special.i1(x)),
                ("ndtr",lambda:torch.special.ndtr(x)),
                ("ndtri",lambda:torch.special.ndtri(x)),
                ("erfcx",lambda:torch.special.erfcx(x)),
                ("entr",lambda:torch.special.entr(x))]:
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
        let ftv = time_ft(&a, *f);
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

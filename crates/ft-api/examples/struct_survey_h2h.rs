//! Structural-op scan FT vs PyTorch to find the biggest materialize-and-slow gap.
//! [4000,4000] f64 no-grad (diag builds from [4000]). cat_anchor = known landed win.
//! Run: PYTORCH_PYTHON=/path/to/python cargo run --release -p ft-api --example struct_survey_h2h

use std::io::Write;
use std::process::{Command, Stdio};
use std::time::Instant;

use ft_api::FrankenTorchSession;
use ft_autograd::TensorNodeId;
use ft_core::ExecutionMode;

const R: usize = 4000;
const C: usize = 4000;

fn time_ft<F: Fn(&mut FrankenTorchSession, TensorNodeId)>(
    mat: &[f64],
    vec1d: &[f64],
    use_vec: bool,
    f: F,
) -> f64 {
    let mut best = f64::INFINITY;
    for _ in 0..7 {
        let mut s = FrankenTorchSession::new(ExecutionMode::Strict);
        let x = if use_vec {
            s.tensor_variable(vec1d.to_vec(), vec![R], false).unwrap()
        } else {
            s.tensor_variable(mat.to_vec(), vec![R, C], false).unwrap()
        };
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
    let mat: Vec<f64> = (0..R * C).map(|i| ((i % 17) as f64) - 8.0).collect();
    let vec1d: Vec<f64> = (0..R).map(|i| (i % 13) as f64 - 6.0).collect();

    type Op = (
        &'static str,
        bool,
        fn(&mut FrankenTorchSession, TensorNodeId),
    );
    let ops: Vec<Op> = vec![
        ("cat_anchor", false, |s, x| {
            let _ = s.tensor_cat(&[x, x], 1);
        }),
        ("tril", false, |s, x| {
            let _ = s.tensor_tril(x, 0);
        }),
        ("triu", false, |s, x| {
            let _ = s.tensor_triu(x, 0);
        }),
        ("diagonal", false, |s, x| {
            let _ = s.tensor_diagonal(x, 0);
        }),
        ("trace", false, |s, x| {
            let _ = s.tensor_trace(x);
        }),
        ("flip0", false, |s, x| {
            let _ = s.tensor_flip(x, &[0]);
        }),
        ("diag_build", true, |s, x| {
            let _ = s.tensor_diag(x, 0);
        }),
    ];

    let python = std::env::var("PYTORCH_PYTHON").unwrap_or_else(|_| "python3".to_string());
    let py = r#"
import time, torch
torch.set_num_threads(8)
R,C=4000,4000
idx=torch.arange(R*C,dtype=torch.int64)
x=((idx%17).double()-8.0).reshape(R,C)
v=((torch.arange(R,dtype=torch.int64)%13).double()-6.0)
def t(fn,n=7):
    for _ in range(2):
        try: fn()
        except Exception: return float('nan')
    ts=[]
    for _ in range(n):
        s=time.perf_counter(); fn(); ts.append((time.perf_counter()-s)*1e3)
    return min(ts)
for name,fn in [("cat_anchor",lambda:torch.cat([x,x],1)),
                ("tril",lambda:torch.tril(x)),("triu",lambda:torch.triu(x)),
                ("diagonal",lambda:torch.diagonal(x).clone()),("trace",lambda:torch.trace(x)),
                ("flip0",lambda:torch.flip(x,[0])),("diag_build",lambda:torch.diag(v))]:
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

    println!("op            FT(ms)    PT(ms)   verdict");
    for (name, uv, f) in &ops {
        let ftv = time_ft(&mat, &vec1d, *uv, *f);
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
            println!("  {name:<12} {ftv:8.3} {p:8.3}   {tag}");
        } else {
            println!("  {name:<12} {ftv:8.3}      nan   (torch failed)");
        }
    }
    Ok(())
}

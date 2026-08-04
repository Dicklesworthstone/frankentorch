//! Broad f32 survey #2: var/std/prod/softmax/cumsum/sigmoid/logsumexp — find next gap.
use ft_api::FrankenTorchSession;
use ft_autograd::TensorNodeId;
use ft_core::ExecutionMode;
use std::io::Write;
use std::process::{Command, Stdio};
use std::time::Instant;
const R: usize = 4000;
const C: usize = 4000;

fn t1<F: Fn(&mut FrankenTorchSession, TensorNodeId)>(a: &[f32], f: F) -> f64 {
    let mut best = f64::INFINITY;
    for _ in 0..7 {
        let mut s = FrankenTorchSession::new(ExecutionMode::Strict);
        let x = s
            .tensor_variable_f32(a.to_vec(), vec![R, C], false)
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
    let a: Vec<f32> = (0..R * C)
        .map(|i| ((i % 2000) as f32 - 1000.0) * 0.01)
        .collect();
    let python = std::env::var("PYTORCH_PYTHON").unwrap_or_else(|_| "python3".to_string());
    let py = r#"
import time,torch
torch.set_num_threads(8)
import torch.nn.functional as F
R,C=4000,4000
a=(((torch.arange(R*C,dtype=torch.int64)%2000).float()-1000.0)*0.01).reshape(R,C)
def t(fn,n=7):
    for _ in range(2): fn()
    ts=[]
    for _ in range(n): s=time.perf_counter(); fn(); ts.append((time.perf_counter()-s)*1e3)
    return min(ts)
for name,fn in [("add_anchor",lambda:a+a),
                ("var_all",lambda:a.var()),
                ("std_all",lambda:a.std()),
                ("prod_all",lambda:a.prod()),
                ("var_dim1",lambda:a.var(1)),
                ("prod_dim1",lambda:a.prod(1)),
                ("softmax1",lambda:F.softmax(a,1)),
                ("cumsum1",lambda:a.cumsum(1)),
                ("sigmoid",lambda:torch.sigmoid(a)),
                ("logsumexp1",lambda:torch.logsumexp(a,1))]:
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
        t1(&a, |s, x| {
            let _ = s.tensor_add(x, x);
        }),
    );
    rep(
        "var_all",
        t1(&a, |s, x| {
            let _ = s.tensor_var(x, 1);
        }),
    );
    rep(
        "std_all",
        t1(&a, |s, x| {
            let _ = s.tensor_std(x, 1);
        }),
    );
    rep(
        "prod_all",
        t1(&a, |s, x| {
            let _ = s.tensor_prod(x);
        }),
    );
    rep(
        "var_dim1",
        t1(&a, |s, x| {
            let _ = s.tensor_var_dim(x, 1, 1);
        }),
    );
    rep(
        "prod_dim1",
        t1(&a, |s, x| {
            let _ = s.tensor_prod_dim(x, 1);
        }),
    );
    rep(
        "softmax1",
        t1(&a, |s, x| {
            let _ = s.tensor_softmax(x, 1);
        }),
    );
    rep(
        "cumsum1",
        t1(&a, |s, x| {
            let _ = s.tensor_cumsum(x, 1);
        }),
    );
    rep(
        "sigmoid",
        t1(&a, |s, x| {
            let _ = s.tensor_sigmoid(x);
        }),
    );
    rep(
        "logsumexp1",
        t1(&a, |s, x| {
            let _ = s.tensor_logsumexp(x, 1);
        }),
    );
    Ok(())
}

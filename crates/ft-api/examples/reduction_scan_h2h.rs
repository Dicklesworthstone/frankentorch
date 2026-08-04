//! Multi-op global-reduction scan: FT vs PyTorch on [4000,4000] f64 no-grad, to find any
//! op with a count_nonzero-class clone/serial pathology (a surprisingly large gap).
//!
//! Run: PYTORCH_PYTHON=/path/to/python cargo run --release -p ft-api --example reduction_scan_h2h

use std::io::Write;
use std::process::{Command, Stdio};
use std::time::Instant;

use ft_api::FrankenTorchSession;
use ft_core::ExecutionMode;

const R: usize = 4000;
const C: usize = 4000;

fn time_ft<F: Fn(&mut FrankenTorchSession, ft_autograd::TensorNodeId) -> f64>(
    data: &[f64],
    f: F,
) -> f64 {
    let mut best = f64::INFINITY;
    for _ in 0..6 {
        let mut s = FrankenTorchSession::new(ExecutionMode::Strict);
        let x = s.tensor_variable(data.to_vec(), vec![R, C], false).unwrap();
        let t = Instant::now();
        let _ = f(&mut s, x);
        let el = t.elapsed().as_secs_f64() * 1e3;
        if el < best {
            best = el;
        }
    }
    best
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let data: Vec<f64> = (0..R * C).map(|i| ((i % 17) as f64) - 8.0).collect();

    let ops: Vec<(
        &str,
        fn(&mut FrankenTorchSession, ft_autograd::TensorNodeId) -> f64,
    )> = vec![
        ("sum", |s, x| {
            let r = s.tensor_sum(x).unwrap();
            s.tensor_values(r).unwrap()[0]
        }),
        ("mean", |s, x| {
            let r = s.tensor_mean(x).unwrap();
            s.tensor_values(r).unwrap()[0]
        }),
        ("prod", |s, x| {
            let r = s.tensor_prod(x).unwrap();
            s.tensor_values(r).unwrap()[0]
        }),
        ("var", |s, x| {
            let r = s.tensor_var(x, 1).unwrap();
            s.tensor_values(r).unwrap()[0]
        }),
        ("std", |s, x| {
            let r = s.tensor_std(x, 1).unwrap();
            s.tensor_values(r).unwrap()[0]
        }),
        ("norm_p2", |s, x| {
            let r = s.tensor_norm(x, 2.0).unwrap();
            s.tensor_values(r).unwrap()[0]
        }),
        ("count_nonzero", |s, x| {
            let r = s.tensor_count_nonzero(x).unwrap();
            s.tensor_values(r).unwrap()[0]
        }),
    ];

    let mut ft_ms = String::new();
    println!("op             FT(ms)");
    for (name, f) in &ops {
        let ms = time_ft(&data, *f);
        println!("  {name:<13} {ms:8.3}");
        ft_ms.push_str(&format!("{name}={ms:.3} "));
    }

    let python = std::env::var("PYTORCH_PYTHON").unwrap_or_else(|_| "python3".to_string());
    let py = r#"
import time, torch
torch.set_num_threads(8)
R,C=4000,4000
idx=torch.arange(R*C,dtype=torch.int64)
x=((idx%17).double()-8.0).reshape(R,C)
def t(fn,n=6):
    for _ in range(2): fn()
    ts=[]
    for _ in range(n):
        s=time.perf_counter(); fn(); ts.append((time.perf_counter()-s)*1e3)
    return min(ts)
for name,fn in [("sum",lambda:torch.sum(x)),("mean",lambda:torch.mean(x)),("prod",lambda:torch.prod(x)),
                ("var",lambda:torch.var(x)),("std",lambda:torch.std(x)),("norm_p2",lambda:torch.norm(x,2)),
                ("count_nonzero",lambda:torch.count_nonzero(x))]:
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
    if let Ok(o) = out
        && o.status.success()
    {
        let s = String::from_utf8_lossy(&o.stdout);
        println!("\nop             FT(ms)   PT(ms)   ratio(PT/FT, <1 = FT slower)");
        for (name, f) in &ops {
            let ftv = time_ft(&data, *f);
            let pt = s.lines().find_map(|l| {
                let mut it = l.strip_prefix("PT ")?.split_whitespace();
                if it.next()? == *name {
                    it.next()?.parse::<f64>().ok()
                } else {
                    None
                }
            });
            if let Some(p) = pt {
                let r = p / ftv;
                let tag = if r >= 1.0 {
                    format!("FT {r:.2}x FASTER")
                } else {
                    format!("FT {:.2}x SLOWER", 1.0 / r)
                };
                println!("  {name:<13} {ftv:8.3} {p:8.3}   {tag}");
            }
        }
    }
    Ok(())
}

//! cosine_similarity f32 native fast path vs torch. Was: upcast f32->F64 (wrong dtype + 2x mem).
//! add = anchor. dim=last.
use ft_api::FrankenTorchSession;
use ft_core::ExecutionMode;
use std::io::Write;
use std::process::{Command, Stdio};
use std::time::Instant;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let python = std::env::var("PYTORCH_PYTHON").unwrap_or_else(|_| "python3".to_string());

    // parity [N=4, D=5] dim=1, within tolerance (cosine_similarity is a tolerance metric)
    let (nn, dd) = (4usize, 5usize);
    let x1: Vec<f32> = (0..nn * dd).map(|i| ((i % 7) as f32 - 3.0) * 0.5).collect();
    let x2: Vec<f32> = (0..nn * dd)
        .map(|i| ((i % 5) as f32 - 2.0) * 0.4 + 0.1)
        .collect();
    let py_s = format!(
        r#"
import torch
x1=torch.tensor({x1:?},dtype=torch.float32).reshape({nn},{dd})
x2=torch.tensor({x2:?},dtype=torch.float32).reshape({nn},{dd})
o=torch.nn.functional.cosine_similarity(x1,x2,dim=1,eps=1e-8)
print("VALS"," ".join("%.9g"%v for v in o.tolist()))
"#,
        x1 = x1,
        x2 = x2,
        nn = nn,
        dd = dd
    );
    let mut ch = Command::new(&python)
        .arg("-")
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .spawn()?;
    ch.stdin.as_mut().unwrap().write_all(py_s.as_bytes())?;
    let pt = String::from_utf8_lossy(&ch.wait_with_output()?.stdout).to_string();
    let pv: Vec<f64> = pt
        .lines()
        .find_map(|l| l.strip_prefix("VALS "))
        .map(|s| {
            s.split_whitespace()
                .filter_map(|t| t.parse().ok())
                .collect()
        })
        .unwrap_or_default();
    let mut s = FrankenTorchSession::new(ExecutionMode::Strict);
    let a = s.tensor_variable_f32(x1.clone(), vec![nn, dd], false)?;
    let b = s.tensor_variable_f32(x2.clone(), vec![nn, dd], false)?;
    let o = s.tensor_cosine_similarity(a, b, 1, 1e-8)?;
    let dt = s.tensor_dtype(o)?;
    let fv = s.tensor_values_lossy_f64(o)?;
    let maxrel = fv
        .iter()
        .zip(&pv)
        .map(|(a, b)| (a - b).abs() / b.abs().max(1e-30))
        .fold(0.0f64, f64::max);
    println!(
        "parity f32: dtype={dt:?} (was F64) max_rel_err={maxrel:.2e} (within tol: {})",
        maxrel < 1e-5 && fv.len() == pv.len()
    );

    // perf [N=200k, D=128] dim=1
    let (n, d) = (200_000usize, 128usize);
    let a: Vec<f32> = (0..n * d)
        .map(|i| ((i % 9973) as f32 - 5000.0) * 0.001)
        .collect();
    let bb: Vec<f32> = (0..n * d)
        .map(|i| ((i % 7919) as f32 - 4000.0) * 0.001)
        .collect();
    let tt = |which: u8| {
        let mut bst = f64::INFINITY;
        for _ in 0..9 {
            let mut s = FrankenTorchSession::new(ExecutionMode::Strict);
            let x = s.tensor_variable_f32(a.clone(), vec![n, d], false).unwrap();
            let y = s
                .tensor_variable_f32(bb.clone(), vec![n, d], false)
                .unwrap();
            let ti = Instant::now();
            let _ = if which == 0 {
                s.tensor_add(x, y)
            } else {
                s.tensor_cosine_similarity(x, y, 1, 1e-8)
            };
            let e = ti.elapsed().as_secs_f64() * 1e3;
            if e < bst {
                bst = e;
            }
        }
        bst
    };
    let (tr, tc) = (tt(0), tt(1));
    let py_b = format!(
        r#"
import time,torch
torch.set_num_threads(8)
n={n}; d={d}
x=(((torch.arange(n*d,dtype=torch.int64)%9973).float()-5000.0)*0.001).reshape(n,d)
y=(((torch.arange(n*d,dtype=torch.int64)%7919).float()-4000.0)*0.001).reshape(n,d)
def tm(fn,reps=9):
    for _ in range(2): fn()
    ts=[]
    for _ in range(reps): s=time.perf_counter(); fn(); ts.append((time.perf_counter()-s)*1e3)
    return min(ts)
print("PT add %.4f"%tm(lambda:x+y))
print("PT cossim %.4f"%tm(lambda:torch.nn.functional.cosine_similarity(x,y,dim=1,eps=1e-8)))
"#,
        n = n,
        d = d
    );
    let mut ch = Command::new(&python)
        .arg("-")
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .spawn()?;
    ch.stdin.as_mut().unwrap().write_all(py_b.as_bytes())?;
    let pt = String::from_utf8_lossy(&ch.wait_with_output()?.stdout).to_string();
    let g = |k: &str| {
        pt.lines()
            .find_map(|l| {
                let mut it = l.strip_prefix("PT ")?.split_whitespace();
                if it.next()? == k {
                    it.next()?.parse::<f64>().ok()
                } else {
                    None
                }
            })
            .unwrap_or(f64::NAN)
    };
    let v = |ft: f64, p: f64| {
        if p >= ft {
            format!("FT {:.2}x FASTER", p / ft)
        } else {
            format!("FT {:.2}x SLOWER", ft / p)
        }
    };
    println!(
        "  add_anchor   FT {tr:.3} PT {:.3}  => {}",
        g("add"),
        v(tr, g("add"))
    );
    println!(
        "  cosine_sim   FT {tc:.3} PT {:.3}  => {}",
        g("cossim"),
        v(tc, g("cossim"))
    );
    Ok(())
}

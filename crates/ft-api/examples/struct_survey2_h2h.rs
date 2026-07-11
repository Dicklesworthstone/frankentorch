//! Structural-op survey #2 FT vs PyTorch (movement/view ops that materialize in FT but are
//! views/cheap in torch). cat_anchor = landed-win sanity check (must show FASTER).
//! Builds session+input OUTSIDE the timer; only the op is timed.
//! Run: PYTORCH_PYTHON=/path/to/python cargo run --release -p ft-api --example struct_survey2_h2h

use std::io::Write;
use std::process::{Command, Stdio};
use std::time::Instant;

use ft_api::FrankenTorchSession;
use ft_autograd::TensorNodeId;
use ft_core::ExecutionMode;

const R: usize = 4000;
const C: usize = 4000;
const VN: usize = 4_000_000;

// Build (session, node) OUTSIDE the timer; time ONLY the op. Fresh session per iter.
fn timed<B, F>(build: B, op: F) -> f64
where
    B: Fn() -> (FrankenTorchSession, TensorNodeId),
    F: Fn(&mut FrankenTorchSession, TensorNodeId),
{
    let mut best = f64::INFINITY;
    for _ in 0..7 {
        let (mut s, x) = build();
        let t = Instant::now();
        op(&mut s, x);
        let e = t.elapsed().as_secs_f64() * 1e3;
        if e < best {
            best = e;
        }
    }
    best
}

fn report(name: &str, ftv: f64, pt: &str) {
    let p = pt.lines().find_map(|l| {
        let mut it = l.strip_prefix("PT ")?.split_whitespace();
        if it.next()? == name {
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
    } else {
        println!("  {name:<14} {ftv:8.3}      nan   (torch failed)");
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mat: Vec<f64> = (0..R * C).map(|i| ((i % 17) as f64) - 8.0).collect();
    let v1: Vec<f64> = (0..VN).map(|i| (i % 13) as f64 - 6.0).collect();
    let small: Vec<f64> = (0..R).map(|i| (i % 13) as f64).collect();

    let python = std::env::var("PYTORCH_PYTHON").unwrap_or_else(|_| "python3".to_string());
    let py = r#"
import time, torch
torch.set_num_threads(8)
R,C,VN=4000,4000,4000000
x=((torch.arange(R*C,dtype=torch.int64)%17).double()-8.0).reshape(R,C)
v=((torch.arange(VN,dtype=torch.int64)%13).double()-6.0)
s=((torch.arange(R,dtype=torch.int64)%13).double())
def t(fn,n=7):
    for _ in range(2):
        try: fn()
        except Exception: return float('nan')
    ts=[]
    for _ in range(n):
        a=time.perf_counter(); fn(); ts.append((time.perf_counter()-a)*1e3)
    return min(ts)
for name,fn in [("cat_anchor",lambda:torch.cat([x,x],1)),
                ("movedim",lambda:torch.movedim(x,0,1).clone()),
                ("select_d1",lambda:torch.select(x,1,100).clone()),
                ("repeat_int",lambda:torch.repeat_interleave(v,3)),
                ("meshgrid",lambda:[g.clone() for g in torch.meshgrid(s,s,indexing='ij')])]:
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

    println!("op              FT(ms)    PT(ms)   verdict");

    let build_mat = || {
        let mut s = FrankenTorchSession::new(ExecutionMode::Strict);
        let x = s.tensor_variable(mat.clone(), vec![R, C], false).unwrap();
        (s, x)
    };
    report(
        "cat_anchor",
        timed(build_mat, |s, x| {
            let _ = s.tensor_cat(&[x, x], 1);
        }),
        &pt,
    );
    report(
        "movedim",
        timed(build_mat, |s, x| {
            let _ = s.tensor_movedim(x, 0, 1);
        }),
        &pt,
    );
    report(
        "select_d1",
        timed(build_mat, |s, x| {
            let _ = s.tensor_select(x, 1, 100);
        }),
        &pt,
    );

    report(
        "repeat_int",
        timed(
            || {
                let mut s = FrankenTorchSession::new(ExecutionMode::Strict);
                let x = s.tensor_variable(v1.clone(), vec![VN], false).unwrap();
                (s, x)
            },
            |s, x| {
                let _ = s.tensor_repeat_interleave(x, 3);
            },
        ),
        &pt,
    );

    // meshgrid: 2 inputs — build both outside timer.
    let mesh = {
        let mut best = f64::INFINITY;
        for _ in 0..7 {
            let mut s = FrankenTorchSession::new(ExecutionMode::Strict);
            let a = s.tensor_variable(small.clone(), vec![R], false).unwrap();
            let b = s.tensor_variable(small.clone(), vec![R], false).unwrap();
            let t = Instant::now();
            let _ = s.tensor_meshgrid(&[a, b]);
            let e = t.elapsed().as_secs_f64() * 1e3;
            if e < best {
                best = e;
            }
        }
        best
    };
    report("meshgrid", mesh, &pt);

    Ok(())
}

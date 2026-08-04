//! Fused 3-input op scan FT vs PyTorch ([4000,4000] f64 no-grad). addcmul/addcdiv/clamp_tensor
//! are COMPOSED of 2-3 autograd-aware ops; torch fuses them. `cat` is a landed-win ANCHOR.
//! Run: PYTORCH_PYTHON=/path/to/python cargo run --release -p ft-api --example fused3_h2h

use std::io::Write;
use std::process::{Command, Stdio};
use std::time::Instant;

use ft_api::FrankenTorchSession;
use ft_core::ExecutionMode;

const R: usize = 4000;
const C: usize = 4000;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let a: Vec<f64> = (0..R * C).map(|i| ((i % 17) as f64) - 8.0).collect();
    let b: Vec<f64> = (0..R * C).map(|i| ((i % 13) as f64) - 6.0).collect();
    let c: Vec<f64> = (0..R * C).map(|i| ((i % 7) as f64) + 1.0).collect();

    let mk = |s: &mut FrankenTorchSession| {
        let x = s.tensor_variable(a.clone(), vec![R, C], false).unwrap();
        let y = s.tensor_variable(b.clone(), vec![R, C], false).unwrap();
        let z = s.tensor_variable(c.clone(), vec![R, C], false).unwrap();
        (x, y, z)
    };
    let bench = |name: &str,
                 f: &dyn Fn(
        &mut FrankenTorchSession,
        ft_autograd::TensorNodeId,
        ft_autograd::TensorNodeId,
        ft_autograd::TensorNodeId,
    )|
     -> f64 {
        let mut best = f64::INFINITY;
        for _ in 0..7 {
            let mut s = FrankenTorchSession::new(ExecutionMode::Strict);
            let (x, y, z) = mk(&mut s);
            let t = Instant::now();
            f(&mut s, x, y, z);
            let el = t.elapsed().as_secs_f64() * 1e3;
            if el < best {
                best = el;
            }
        }
        let _ = name;
        best
    };

    let names: Vec<(
        &str,
        Box<
            dyn Fn(
                &mut FrankenTorchSession,
                ft_autograd::TensorNodeId,
                ft_autograd::TensorNodeId,
                ft_autograd::TensorNodeId,
            ),
        >,
    )> = vec![
        (
            "cat_anchor",
            Box::new(|s, x, _y, _z| {
                let _ = s.tensor_cat(&[x, x], 1);
            }),
        ),
        (
            "addcmul",
            Box::new(|s, x, y, z| {
                let _ = s.tensor_addcmul(x, y, z, 0.7);
            }),
        ),
        (
            "addcdiv",
            Box::new(|s, x, y, z| {
                let _ = s.tensor_addcdiv(x, y, z, 0.7);
            }),
        ),
        (
            "clamp_tensor",
            Box::new(|s, x, y, z| {
                let _ = s.tensor_clamp_tensor(x, y, z);
            }),
        ),
        (
            "lerp",
            Box::new(|s, x, y, _z| {
                let _ = s.tensor_lerp(x, y, 0.3);
            }),
        ),
    ];
    let python = std::env::var("PYTORCH_PYTHON").unwrap_or_else(|_| "python3".to_string());
    let py = r#"
import time, torch
torch.set_num_threads(8)
R,C=4000,4000
idx=torch.arange(R*C,dtype=torch.int64)
x=((idx%17).double()-8.0).reshape(R,C)
y=((idx%13).double()-6.0).reshape(R,C)
z=((idx%7).double()+1.0).reshape(R,C)
def t(fn,n=7):
    for _ in range(2):
        try: fn()
        except Exception as e: return float('nan')
    ts=[]
    for _ in range(n):
        s=time.perf_counter(); fn(); ts.append((time.perf_counter()-s)*1e3)
    return min(ts)
for name,fn in [("cat_anchor",lambda:torch.cat([x,x],1)),
                ("addcmul",lambda:torch.addcmul(x,y,z,value=0.7)),
                ("addcdiv",lambda:torch.addcdiv(x,y,z,value=0.7)),
                ("clamp_tensor",lambda:torch.clamp(x,min=y,max=z)),
                ("lerp",lambda:torch.lerp(x,y,0.3))]:
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
    for (name, f) in &names {
        let ftv = bench(name, f.as_ref());
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

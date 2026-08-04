//! Structural op scan FT vs PyTorch (f64 no-grad). outer/vander/diag_embed all produce a
//! ~25M output — confirm bandwidth-walled (parity) or composed-slow. `cat` ANCHOR.
//! Run: PYTORCH_PYTHON=/path/to/python cargo run --release -p ft-api --example struct3_h2h

use std::io::Write;
use std::process::{Command, Stdio};
use std::time::Instant;

use ft_api::FrankenTorchSession;
use ft_core::ExecutionMode;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let v: Vec<f64> = (0..5000).map(|i| 0.5 + (i % 13) as f64 * 0.1).collect();
    let de: Vec<f64> = (0..10000 * 50).map(|i| (i % 17) as f64 - 8.0).collect();
    let anc: Vec<f64> = (0..4_000_000).map(|i| (i % 7) as f64).collect();

    let bench = |label: &str, mut f: Box<dyn FnMut()>| -> f64 {
        let mut best = f64::INFINITY;
        for _ in 0..7 {
            let t = Instant::now();
            f();
            let e = t.elapsed().as_secs_f64() * 1e3;
            if e < best {
                best = e;
            }
        }
        let _ = label;
        best
    };

    let outer_ms = bench("outer", {
        let v = v.clone();
        Box::new(move || {
            let mut s = FrankenTorchSession::new(ExecutionMode::Strict);
            let x = s.tensor_variable(v.clone(), vec![5000], false).unwrap();
            let y = s.tensor_variable(v.clone(), vec![5000], false).unwrap();
            let _ = s.tensor_outer(x, y);
        })
    });
    let vander_ms = bench("vander", {
        let v = v.clone();
        Box::new(move || {
            let mut s = FrankenTorchSession::new(ExecutionMode::Strict);
            let x = s.tensor_variable(v.clone(), vec![5000], false).unwrap();
            let _ = s.tensor_vander(x, Some(5000), false);
        })
    });
    let de_ms = bench("diag_embed", {
        let de = de.clone();
        Box::new(move || {
            let mut s = FrankenTorchSession::new(ExecutionMode::Strict);
            let x = s
                .tensor_variable(de.clone(), vec![10000, 50], false)
                .unwrap();
            let _ = s.tensor_diag_embed(x, 0);
        })
    });
    let anchor_ms = bench("cat", {
        let anc = anc.clone();
        Box::new(move || {
            let mut s = FrankenTorchSession::new(ExecutionMode::Strict);
            let x = s
                .tensor_variable(anc.clone(), vec![1_000_000, 4], false)
                .unwrap();
            let _ = s.tensor_cat(&[x, x], 1);
        })
    });

    let python = std::env::var("PYTORCH_PYTHON").unwrap_or_else(|_| "python3".to_string());
    let py = r#"
import time, torch
torch.set_num_threads(8)
v=(0.5+(torch.arange(5000)%13).double()*0.1)
de=((torch.arange(10000*50)%17).double()-8.0).reshape(10000,50)
anc=(torch.arange(4_000_000)%7).double().reshape(1_000_000,4)
def t(fn,n=7):
    for _ in range(2):
        try: fn()
        except Exception as e: return float('nan')
    ts=[]
    for _ in range(n):
        s=time.perf_counter(); fn(); ts.append((time.perf_counter()-s)*1e3)
    return min(ts)
print("PT outer %.4f"%t(lambda:torch.outer(v,v)))
print("PT vander %.4f"%t(lambda:torch.vander(v,5000,increasing=False)))
print("PT diag_embed %.4f"%t(lambda:torch.diag_embed(de)))
print("PT cat %.4f"%t(lambda:torch.cat([anc,anc],1)))
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
    for (name, ftv) in [
        ("outer", outer_ms),
        ("vander", vander_ms),
        ("diag_embed", de_ms),
        ("cat", anchor_ms),
    ] {
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

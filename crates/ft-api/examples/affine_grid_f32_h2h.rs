// affine_grid f32 vs torch: perf + dtype + value (grid generation, no-grad). cc.
// The no-grad path generated the whole N*H*W*2 grid SERIALLY; now parallel over rows.
use ft_api::FrankenTorchSession;
use ft_core::ExecutionMode;
use std::io::Write;
use std::process::{Command, Stdio};
use std::time::Instant;
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let python = std::env::var("PYTORCH_PYTHON").unwrap_or_else(|_| "python3".to_string());
    let (nn, cc, hh, ww) = (64usize, 3usize, 192usize, 192usize);
    // theta[n] = [[a, b, c],[d, e, f]] — a small rotation/scale/translation per batch.
    let theta: Vec<f32> = (0..nn)
        .flat_map(|n| {
            let t = (n as f32) * 0.01;
            vec![
                1.0 + 0.1 * t,
                0.05 * t,
                0.02 * t,
                -0.05 * t,
                1.0 - 0.1 * t,
                -0.03 * t,
            ]
        })
        .collect();
    let bench = || {
        let mut best = f64::INFINITY;
        for _ in 0..7 {
            let mut s = FrankenTorchSession::new(ExecutionMode::Strict);
            let th = s
                .tensor_variable_f32(theta.clone(), vec![nn, 2, 3], false)
                .unwrap();
            let t = Instant::now();
            let _ = s.tensor_affine_grid(th, vec![nn, cc, hh, ww], false);
            let e = t.elapsed().as_secs_f64() * 1e3;
            if e < best {
                best = e;
            }
        }
        best
    };
    let mut s = FrankenTorchSession::new(ExecutionMode::Strict);
    let th = s.tensor_variable_f32(theta.clone(), vec![nn, 2, 3], false)?;
    let o = s.tensor_affine_grid(th, vec![nn, cc, hh, ww], false)?;
    let dt = s.tensor_dtype(o)?;
    let fv: Vec<f32> = s
        .tensor_values_lossy_f64(o)?
        .iter()
        .take(8192)
        .map(|&v| v as f32)
        .collect();
    let py = format!(
        r#"
import time,torch
import torch.nn.functional as F
torch.set_num_threads(8)
nn,cc,hh,ww={nn},{cc},{hh},{ww}
rows=[]
for n in range(nn):
    t=n*0.01
    rows.append([[1.0+0.1*t,0.05*t,0.02*t],[-0.05*t,1.0-0.1*t,-0.03*t]])
theta=torch.tensor(rows,dtype=torch.float32)
def tm(fn,reps=7):
    for _ in range(2): fn()
    ts=[]
    for _ in range(reps): s=time.perf_counter(); fn(); ts.append((time.perf_counter()-s)*1e3)
    return min(ts)
print("PT ag %.3f"%tm(lambda:F.affine_grid(theta,[nn,cc,hh,ww],align_corners=False)))
o=F.affine_grid(theta,[nn,cc,hh,ww],align_corners=False); assert o.dtype==torch.float32
print("REF "+" ".join("%a"%float(v) for v in o.flatten()[:8192].tolist()))
"#
    );
    let mut ch = Command::new(&python)
        .arg("-")
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .spawn()?;
    ch.stdin.as_mut().unwrap().write_all(py.as_bytes())?;
    let out = String::from_utf8_lossy(&ch.wait_with_output()?.stdout).to_string();
    let g = |k: &str| {
        out.lines()
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
    let ft = bench();
    let pt = g("ag");
    let line = out.lines().find(|l| l.starts_with("REF ")).unwrap_or("");
    let tv: Vec<f32> = line
        .split_whitespace()
        .skip(1)
        .filter_map(|t| t.parse().ok())
        .collect();
    let exact = fv
        .iter()
        .zip(tv.iter())
        .filter(|(a, b)| a.to_bits() == b.to_bits())
        .count();
    let close = fv
        .iter()
        .zip(tv.iter())
        .filter(|(a, b)| (**a - **b).abs() <= 1e-5 * b.abs().max(1.0))
        .count();
    let vrb = if pt >= ft {
        format!("FT {:.2}x FASTER", pt / ft)
    } else {
        format!("FT {:.2}x SLOWER", ft / pt)
    };
    println!("affine_grid [{nn}x{cc}x{hh}x{ww}] f32:");
    println!("  perf:  FT {ft:8.3}ms  torch {pt:8.3}ms => {vrb}");
    println!(
        "  value: dtype={dt:?} bit_exact={exact}/{} close(1e-5)={close}/{}",
        fv.len().min(tv.len()),
        fv.len().min(tv.len())
    );
    Ok(())
}

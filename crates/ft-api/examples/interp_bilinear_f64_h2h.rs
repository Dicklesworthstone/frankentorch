// f64 bilinear interpolate vs torch: per-axis coord hoist. cc.
use ft_api::FrankenTorchSession;
use ft_core::ExecutionMode;
use std::io::Write;
use std::process::{Command, Stdio};
use std::time::Instant;
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let python = std::env::var("PYTORCH_PYTHON").unwrap_or_else(|_| "python3".to_string());
    let (nn, cc, ih, iw, oh, ow) = (16usize, 32usize, 64usize, 64usize, 160usize, 160usize);
    let numel = nn * cc * ih * iw;
    let x: Vec<f64> = (0..numel)
        .map(|i| ((i % 9973) as f64 - 5000.0) * 0.01)
        .collect();
    let bench = || {
        let mut best = f64::INFINITY;
        for _ in 0..7 {
            let mut s = FrankenTorchSession::new(ExecutionMode::Strict);
            let xi = s
                .tensor_variable(x.clone(), vec![nn, cc, ih, iw], false)
                .unwrap();
            let t = Instant::now();
            let _ = s.tensor_interpolate(xi, Some(vec![oh, ow]), None, "bilinear", Some(false));
            let e = t.elapsed().as_secs_f64() * 1e3;
            if e < best {
                best = e;
            }
        }
        best
    };
    let mut s = FrankenTorchSession::new(ExecutionMode::Strict);
    let xi = s.tensor_variable(x.clone(), vec![nn, cc, ih, iw], false)?;
    let o = s.tensor_interpolate(xi, Some(vec![oh, ow]), None, "bilinear", Some(false))?;
    let dt = s.tensor_dtype(o)?;
    let fv: Vec<f64> = s
        .tensor_values_lossy_f64(o)?
        .iter()
        .take(8192)
        .copied()
        .collect();
    let py = format!(
        r#"
import time,torch
import torch.nn.functional as F
torch.set_num_threads(8)
nn,cc,ih,iw,oh,ow={nn},{cc},{ih},{iw},{oh},{ow}
numel=nn*cc*ih*iw
x=(((torch.arange(numel,dtype=torch.int64)%9973).double()-5000.0)*0.01).reshape(nn,cc,ih,iw)
def tm(fn,reps=7):
    for _ in range(2): fn()
    ts=[]
    for _ in range(reps): s=time.perf_counter(); fn(); ts.append((time.perf_counter()-s)*1e3)
    return min(ts)
print("PT bi %.3f"%tm(lambda:F.interpolate(x,size=(oh,ow),mode='bilinear',align_corners=False)))
o=F.interpolate(x,size=(oh,ow),mode='bilinear',align_corners=False); assert o.dtype==torch.float64
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
    let pt = g("bi");
    let line = out.lines().find(|l| l.starts_with("REF ")).unwrap_or("");
    let tv: Vec<f64> = line
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
        .filter(|(a, b)| (**a - **b).abs() <= 1e-9 * b.abs().max(1.0))
        .count();
    let vrb = if pt >= ft {
        format!("FT {:.2}x FASTER", pt / ft)
    } else {
        format!("FT {:.2}x SLOWER", ft / pt)
    };
    println!("interp_bilinear_f64 [{nn}x{cc}x{ih}x{iw}]->[{oh}x{ow}]:");
    println!("  perf:  FT {ft:8.3}ms  torch {pt:8.3}ms => {vrb}");
    println!(
        "  value: dtype={dt:?} bit_exact={exact}/{} close(1e-9)={close}/{}",
        fv.len().min(tv.len()),
        fv.len().min(tv.len())
    );
    Ok(())
}

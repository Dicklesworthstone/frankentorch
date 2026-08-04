// prelu f32 vs torch: perf + dtype + value (per-channel, no-grad). cc.
// Before this fix, no-grad f32 prelu errored (UnsupportedDType(F32)) in the f64 path.
use ft_api::FrankenTorchSession;
use ft_core::ExecutionMode;
use std::io::Write;
use std::process::{Command, Stdio};
use std::time::Instant;
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let python = std::env::var("PYTORCH_PYTHON").unwrap_or_else(|_| "python3".to_string());
    let (n, c, h, w) = (8usize, 256usize, 56usize, 56usize);
    let numel = n * c * h * w;
    let x: Vec<f32> = (0..numel)
        .map(|i| ((i % 9973) as f32 - 5000.0) * 0.01)
        .collect();
    let wt: Vec<f32> = (0..c).map(|j| 0.05 + (j % 13) as f32 * 0.01).collect();
    let bench = || {
        let mut best = f64::INFINITY;
        for _ in 0..7 {
            let mut s = FrankenTorchSession::new(ExecutionMode::Strict);
            let xi = s
                .tensor_variable_f32(x.clone(), vec![n, c, h, w], false)
                .unwrap();
            let wi = s.tensor_variable_f32(wt.clone(), vec![c], false).unwrap();
            let t = Instant::now();
            let _ = s.tensor_prelu(xi, wi);
            let e = t.elapsed().as_secs_f64() * 1e3;
            if e < best {
                best = e;
            }
        }
        best
    };
    let mut s = FrankenTorchSession::new(ExecutionMode::Strict);
    let xi = s.tensor_variable_f32(x.clone(), vec![n, c, h, w], false)?;
    let wi = s.tensor_variable_f32(wt.clone(), vec![c], false)?;
    let o = s.tensor_prelu(xi, wi)?;
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
n,c,h,w={n},{c},{h},{w}
numel=n*c*h*w
x=(((torch.arange(numel,dtype=torch.int64)%9973).float()-5000.0)*0.01).reshape(n,c,h,w)
wt=(0.05+(torch.arange(c,dtype=torch.int64)%13).float()*0.01)
def tm(fn,reps=7):
    for _ in range(2): fn()
    ts=[]
    for _ in range(reps): s=time.perf_counter(); fn(); ts.append((time.perf_counter()-s)*1e3)
    return min(ts)
print("PT prelu %.3f"%tm(lambda:F.prelu(x,wt)))
o=F.prelu(x,wt); assert o.dtype==torch.float32
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
    let pt = g("prelu");
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
    let vrb = if pt >= ft {
        format!("FT {:.2}x FASTER", pt / ft)
    } else {
        format!("FT {:.2}x SLOWER", ft / pt)
    };
    println!("prelu [{n}x{c}x{h}x{w}] per-channel f32:");
    println!("  perf:  FT {ft:8.3}ms  torch {pt:8.3}ms => {vrb}");
    println!(
        "  value: dtype={dt:?} bit_exact={exact}/{}",
        fv.len().min(tv.len())
    );
    Ok(())
}

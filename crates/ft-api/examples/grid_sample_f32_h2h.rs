// grid_sample f32 vs torch (no-grad): native-f32 output (was f64+downcast). cc.
use ft_api::{FrankenTorchSession, GridSampleMode, GridSamplePaddingMode};
use ft_core::{DType, ExecutionMode};
use std::io::Write;
use std::process::{Command, Stdio};
use std::time::Instant;
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let python = std::env::var("PYTORCH_PYTHON").unwrap_or_else(|_| "python3".to_string());
    let (n, c, h, w, oh, ow) = (8usize, 32usize, 64usize, 64usize, 128usize, 128usize);
    let inp: Vec<f32> = (0..n * c * h * w)
        .map(|i| ((i % 9973) as f32 - 5000.0) * 0.001)
        .collect();
    // grid in [-1,1]: [n, oh, ow, 2]
    let grid: Vec<f32> = (0..n * oh * ow * 2)
        .map(|i| (((i * 7919) % 2001) as f32 / 1000.0) - 1.0)
        .collect();
    let bench = || {
        let mut best = f64::INFINITY;
        for _ in 0..7 {
            let mut s = FrankenTorchSession::new(ExecutionMode::Strict);
            let xi = s
                .tensor_variable_f32(inp.clone(), vec![n, c, h, w], false)
                .unwrap();
            let gi = s
                .tensor_variable_f32(grid.clone(), vec![n, oh, ow, 2], false)
                .unwrap();
            let t = Instant::now();
            let _ = s.functional_grid_sample(
                xi,
                gi,
                GridSampleMode::Bilinear,
                GridSamplePaddingMode::Zeros,
                false,
            );
            let e = t.elapsed().as_secs_f64() * 1e3;
            if e < best {
                best = e;
            }
        }
        best
    };
    let mut s = FrankenTorchSession::new(ExecutionMode::Strict);
    let xi = s.tensor_variable_f32(inp.clone(), vec![n, c, h, w], false)?;
    let gi = s.tensor_variable_f32(grid.clone(), vec![n, oh, ow, 2], false)?;
    let o = s.functional_grid_sample(
        xi,
        gi,
        GridSampleMode::Bilinear,
        GridSamplePaddingMode::Zeros,
        false,
    )?;
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
n,c,h,w,oh,ow={n},{c},{h},{w},{oh},{ow}
inp=(((torch.arange(n*c*h*w,dtype=torch.int64)%9973).float()-5000.0)*0.001).reshape(n,c,h,w)
grid=(((torch.arange(n*oh*ow*2,dtype=torch.int64)*7919)%2001).float()/1000.0-1.0).reshape(n,oh,ow,2)
def tm(fn,reps=7):
    for _ in range(2): fn()
    ts=[]
    for _ in range(reps): s=time.perf_counter(); fn(); ts.append((time.perf_counter()-s)*1e3)
    return min(ts)
print("PT gs %.3f"%tm(lambda:F.grid_sample(inp,grid,mode='bilinear',padding_mode='zeros',align_corners=False)))
o=F.grid_sample(inp,grid,mode='bilinear',padding_mode='zeros',align_corners=False); assert o.dtype==torch.float32
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
    let pt = out
        .lines()
        .find_map(|l| {
            let mut it = l.strip_prefix("PT ")?.split_whitespace();
            if it.next()? == "gs" {
                it.next()?.parse::<f64>().ok()
            } else {
                None
            }
        })
        .unwrap_or(f64::NAN);
    let ft = bench();
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
    let _ = DType::F32;
    println!(
        "grid_sample f32 [{n}x{c}x{h}x{w}]->[{oh}x{ow}] bilinear: FT {ft:8.3}ms torch {pt:8.3}ms => {vrb} | dtype={dt:?} exact={exact} close={close}/{}",
        fv.len().min(tv.len())
    );
    Ok(())
}

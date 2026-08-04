// grid_sample f32 vs torch (bilinear, zeros padding, align_corners=false).
use ft_api::{FrankenTorchSession, GridSampleMode, GridSamplePaddingMode};
use ft_core::ExecutionMode;
use std::io::Write;
use std::process::{Command, Stdio};
use std::time::Instant;
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let python = std::env::var("PYTORCH_PYTHON").unwrap_or_else(|_| "python3".to_string());
    let (nb, c, h, w) = (8usize, 64usize, 128usize, 128usize);
    let inp: Vec<f32> = (0..nb * c * h * w)
        .map(|i| ((i % 997) as f32 / 500.0) - 1.0)
        .collect();
    let grid: Vec<f32> = (0..nb * h * w * 2)
        .map(|i| ((i % 401) as f32 / 200.0) - 1.0)
        .collect();
    let bench = || {
        let mut best = f64::INFINITY;
        for _ in 0..7 {
            let mut s = FrankenTorchSession::new(ExecutionMode::Strict);
            let x = s
                .tensor_variable_f32(inp.clone(), vec![nb, c, h, w], false)
                .unwrap();
            let gr = s
                .tensor_variable_f32(grid.clone(), vec![nb, h, w, 2], false)
                .unwrap();
            let t = Instant::now();
            let _ = s.tensor_grid_sample(
                x,
                gr,
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
    // correctness
    let mut s = FrankenTorchSession::new(ExecutionMode::Strict);
    let x = s.tensor_variable_f32(inp.clone(), vec![nb, c, h, w], false)?;
    let gr = s.tensor_variable_f32(grid.clone(), vec![nb, h, w, 2], false)?;
    let y = s.tensor_grid_sample(
        x,
        gr,
        GridSampleMode::Bilinear,
        GridSamplePaddingMode::Zeros,
        false,
    )?;
    let dt = s.tensor_dtype(y)?;
    let fv: Vec<f32> = s
        .tensor_values_lossy_f64(y)?
        .iter()
        .take(4096)
        .map(|&v| v as f32)
        .collect();
    let py = format!(
        r#"
import time,torch
import torch.nn.functional as F
torch.set_num_threads(8)
nb,c,h,w={nb},{c},{h},{w}
inp=(((torch.arange(nb*c*h*w,dtype=torch.int64)%997).float()/500.0)-1.0).reshape(nb,c,h,w)
grid=(((torch.arange(nb*h*w*2,dtype=torch.int64)%401).float()/200.0)-1.0).reshape(nb,h,w,2)
def tm(fn,reps=7):
    for _ in range(2): fn()
    ts=[]
    for _ in range(reps): s=time.perf_counter(); fn(); ts.append((time.perf_counter()-s)*1e3)
    return min(ts)
print("PT grid %.3f"%tm(lambda:F.grid_sample(inp,grid,mode='bilinear',padding_mode='zeros',align_corners=False)))
y=F.grid_sample(inp,grid,mode='bilinear',padding_mode='zeros',align_corners=False)
print("REF "+" ".join("%a"%float(v) for v in y.flatten()[:4096].tolist()))
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
    let pt = g("grid");
    let line = out.lines().find(|l| l.starts_with("REF ")).unwrap_or("");
    let tv: Vec<f32> = line
        .split_whitespace()
        .skip(1)
        .filter_map(|t| t.parse().ok())
        .collect();
    let mut exact = 0usize;
    let mut maxabs = 0f32;
    for (&a, &b) in fv.iter().zip(tv.iter()) {
        if a.to_bits() == b.to_bits() {
            exact += 1;
        }
        let d = (a - b).abs();
        if d > maxabs {
            maxabs = d;
        }
    }
    let vrb = if pt >= ft {
        format!("FT {:.2}x FASTER", pt / ft)
    } else {
        format!("FT {:.2}x SLOWER", ft / pt)
    };
    println!("grid_sample [{nb}x{c}x{h}x{w}] f32 bilinear:");
    println!("  perf:  FT {ft:8.3}ms  torch {pt:8.3}ms => {vrb}");
    println!(
        "  value: dtype={dt:?} bit_exact={exact}/{} max_abs={maxabs:.3e}",
        fv.len().min(tv.len())
    );
    Ok(())
}

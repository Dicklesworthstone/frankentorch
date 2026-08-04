// pad reflect/replicate/circular f32 vs torch (no-grad): asymmetric-dtype f32 mirror. cc.
use ft_api::FrankenTorchSession;
use ft_core::ExecutionMode;
use std::io::Write;
use std::process::{Command, Stdio};
use std::time::Instant;
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let python = std::env::var("PYTORCH_PYTHON").unwrap_or_else(|_| "python3".to_string());
    let (n, c, h, w) = (16usize, 32usize, 128usize, 128usize);
    let pad = 8usize; // pad last two dims by 8 each side -> [.,.,144,144]
    let inp: Vec<f32> = (0..n * c * h * w)
        .map(|i| ((i % 9973) as f32 - 5000.0) * 0.001)
        .collect();
    let bench = || {
        let mut best = f64::INFINITY;
        for _ in 0..5 {
            let mut s = FrankenTorchSession::new(ExecutionMode::Strict);
            let xi = s
                .tensor_variable_f32(inp.clone(), vec![n, c, h, w], false)
                .unwrap();
            let t = Instant::now();
            let _ = s.tensor_pad_mode(xi, &[pad, pad, pad, pad], "reflect", 0.0);
            let e = t.elapsed().as_secs_f64() * 1e3;
            if e < best {
                best = e;
            }
        }
        best
    };
    let mut s = FrankenTorchSession::new(ExecutionMode::Strict);
    let xi = s.tensor_variable_f32(inp.clone(), vec![n, c, h, w], false)?;
    let ores = s.tensor_pad_mode(xi, &[pad, pad, pad, pad], "reflect", 0.0);
    match &ores {
        Ok(o) => {
            let dt = s.tensor_dtype(*o)?;
            let fv: Vec<f32> = s
                .tensor_values_lossy_f64(*o)?
                .iter()
                .take(8192)
                .map(|&v| v as f32)
                .collect();
            let py = format!(
                r#"
import time,torch
import torch.nn.functional as F
torch.set_num_threads(8)
n,c,h,w,pad={n},{c},{h},{w},{pad}
x=(((torch.arange(n*c*h*w,dtype=torch.int64)%9973).float()-5000.0)*0.001).reshape(n,c,h,w)
def tm(fn,reps=5):
    for _ in range(2): fn()
    ts=[]
    for _ in range(reps): s=time.perf_counter(); fn(); ts.append((time.perf_counter()-s)*1e3)
    return min(ts)
print("PT pd %.3f"%tm(lambda:F.pad(x,(pad,pad,pad,pad),mode='reflect')))
o=F.pad(x,(pad,pad,pad,pad),mode='reflect'); assert o.dtype==torch.float32
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
                    if it.next()? == "pd" {
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
            let vrb = if pt >= ft {
                format!("FT {:.2}x FASTER", pt / ft)
            } else {
                format!("FT {:.2}x SLOWER", ft / pt)
            };
            println!(
                "pad reflect f32 [{n}x{c}x{h}x{w}] pad={pad}: FT {ft:8.3}ms torch {pt:8.3}ms => {vrb} | dtype={dt:?} exact={exact}/{}",
                fv.len().min(tv.len())
            );
        }
        Err(e) => {
            println!(
                "pad reflect f32 [{n}x{c}x{h}x{w}] pad={pad}: FT *** ERROR (f32 crash): {e:?}"
            );
        }
    }
    Ok(())
}

// diagonal_scatter f32 vs torch (no-grad): native-f32 (was a crash). cc.
use ft_api::FrankenTorchSession;
use ft_core::ExecutionMode;
use std::io::Write;
use std::process::{Command, Stdio};
use std::time::Instant;
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let python = std::env::var("PYTORCH_PYTHON").unwrap_or_else(|_| "python3".to_string());
    let nn = 4096usize;
    let input: Vec<f32> = (0..nn * nn)
        .map(|i| ((i % 9973) as f32 - 5000.0) * 0.001)
        .collect();
    let src: Vec<f32> = (0..nn).map(|i| ((i % 7919) as f32) * 0.001 + 0.5).collect();
    let bench = || {
        let mut best = f64::INFINITY;
        for _ in 0..5 {
            let mut s = FrankenTorchSession::new(ExecutionMode::Strict);
            let xi = s
                .tensor_variable_f32(input.clone(), vec![nn, nn], false)
                .unwrap();
            let sr = s.tensor_variable_f32(src.clone(), vec![nn], false).unwrap();
            let t = Instant::now();
            let _ = s.tensor_diagonal_scatter(xi, sr, 0);
            let e = t.elapsed().as_secs_f64() * 1e3;
            if e < best {
                best = e;
            }
        }
        best
    };
    let mut s = FrankenTorchSession::new(ExecutionMode::Strict);
    let xi = s.tensor_variable_f32(input.clone(), vec![nn, nn], false)?;
    let sr = s.tensor_variable_f32(src.clone(), vec![nn], false)?;
    let o = match s.tensor_diagonal_scatter(xi, sr, 0) {
        Ok(o) => o,
        Err(e) => {
            println!("diagonal_scatter f32: FT ERROR {e:?}");
            return Ok(());
        }
    };
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
torch.set_num_threads(8)
nn={nn}
input=(((torch.arange(nn*nn,dtype=torch.int64)%9973).float()-5000.0)*0.001).reshape(nn,nn)
src=(((torch.arange(nn,dtype=torch.int64)%7919).float())*0.001+0.5)
def tm(fn,reps=5):
    for _ in range(2): fn()
    ts=[]
    for _ in range(reps): s=time.perf_counter(); fn(); ts.append((time.perf_counter()-s)*1e3)
    return min(ts)
print("PT ds %.3f"%tm(lambda:torch.diagonal_scatter(input,src,0)))
o=torch.diagonal_scatter(input,src,0)
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
            if it.next()? == "ds" {
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
        "diagonal_scatter f32 [{nn}x{nn}]: FT {ft:8.3}ms torch {pt:8.3}ms => {vrb} | dtype={dt:?} exact={exact}/{}",
        fv.len().min(tv.len())
    );
    Ok(())
}

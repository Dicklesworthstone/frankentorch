// scatter_reduce f32 vs torch across reduce modes (no-grad). cc.
// torch leaves rare CPU reduce modes (amax/amin/prod) slow (cf. embedding_bag max).
use ft_api::FrankenTorchSession;
use ft_core::ExecutionMode;
use std::io::Write;
use std::process::{Command, Stdio};
use std::time::Instant;
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let python = std::env::var("PYTORCH_PYTHON").unwrap_or_else(|_| "python3".to_string());
    let (rows, cols, k) = (4096usize, 1024usize, 1024usize);
    let input: Vec<f32> = (0..rows * cols)
        .map(|i| ((i % 9973) as f32 - 5000.0) * 0.001)
        .collect();
    let idx: Vec<f64> = (0..rows * k).map(|i| ((i * 7919) % cols) as f64).collect();
    let src: Vec<f32> = (0..rows * k)
        .map(|i| ((i % 7919) as f32) * 0.001 + 0.5)
        .collect();
    for mode in ["sum", "amax", "amin", "prod", "mean"] {
        let bench = || {
            let mut best = f64::INFINITY;
            for _ in 0..5 {
                let mut s = FrankenTorchSession::new(ExecutionMode::Strict);
                let xi = s
                    .tensor_variable_f32(input.clone(), vec![rows, cols], false)
                    .unwrap();
                let ix = s
                    .tensor_variable(idx.clone(), vec![rows, k], false)
                    .unwrap();
                let sr = s
                    .tensor_variable_f32(src.clone(), vec![rows, k], false)
                    .unwrap();
                let t = Instant::now();
                let _ = s.tensor_scatter_reduce(xi, 1, ix, sr, mode);
                let e = t.elapsed().as_secs_f64() * 1e3;
                if e < best {
                    best = e;
                }
            }
            best
        };
        let mut s = FrankenTorchSession::new(ExecutionMode::Strict);
        let xi = s.tensor_variable_f32(input.clone(), vec![rows, cols], false)?;
        let ix = s.tensor_variable(idx.clone(), vec![rows, k], false)?;
        let sr = s.tensor_variable_f32(src.clone(), vec![rows, k], false)?;
        let o = match s.tensor_scatter_reduce(xi, 1, ix, sr, mode) {
            Ok(o) => o,
            Err(e) => {
                println!("scatter_reduce mode={mode}: FT ERROR {e:?}");
                continue;
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
rows,cols,k={rows},{cols},{k}
input=(((torch.arange(rows*cols,dtype=torch.int64)%9973).float()-5000.0)*0.001).reshape(rows,cols)
idx=((torch.arange(rows*k,dtype=torch.int64)*7919)%cols).reshape(rows,k)
src=(((torch.arange(rows*k,dtype=torch.int64)%7919).float())*0.001+0.5).reshape(rows,k)
def tm(fn,reps=5):
    for _ in range(2): fn()
    ts=[]
    for _ in range(reps): s=time.perf_counter(); fn(); ts.append((time.perf_counter()-s)*1e3)
    return min(ts)
print("PT sr %.3f"%tm(lambda:input.scatter_reduce(1,idx,src,reduce='{mode}',include_self=True)))
o=input.scatter_reduce(1,idx,src,reduce='{mode}',include_self=True)
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
        let pt = g("sr");
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
        println!(
            "scatter_reduce [{rows}x{cols}] k={k} f32 mode={mode}: FT {ft:7.3}ms torch {pt:7.3}ms => {vrb} | dtype={dt:?} exact={exact} close={close}/{}",
            fv.len().min(tv.len())
        );
    }
    Ok(())
}

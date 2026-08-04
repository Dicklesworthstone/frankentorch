// embedding f32 vs torch: perf + dtype + value (gather, no-grad). cc.
use ft_api::FrankenTorchSession;
use ft_core::ExecutionMode;
use std::io::Write;
use std::process::{Command, Stdio};
use std::time::Instant;
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let python = std::env::var("PYTORCH_PYTHON").unwrap_or_else(|_| "python3".to_string());
    let (ne, dim, ni) = (50_000usize, 128usize, 100_000usize);
    let weight: Vec<f32> = (0..ne * dim)
        .map(|i| ((i % 997) as f32 / 500.0) - 1.0)
        .collect();
    let idx: Vec<f64> = (0..ni).map(|i| ((i * 7919) % ne) as f64).collect();
    let bench = || {
        let mut best = f64::INFINITY;
        for _ in 0..7 {
            let mut s = FrankenTorchSession::new(ExecutionMode::Strict);
            let w = s
                .tensor_variable_f32(weight.clone(), vec![ne, dim], false)
                .unwrap();
            let ix = s.tensor_variable(idx.clone(), vec![ni], false).unwrap();
            let t = Instant::now();
            let _ = s.tensor_embedding(ix, w, None);
            let e = t.elapsed().as_secs_f64() * 1e3;
            if e < best {
                best = e;
            }
        }
        best
    };
    let mut s = FrankenTorchSession::new(ExecutionMode::Strict);
    let w = s.tensor_variable_f32(weight.clone(), vec![ne, dim], false)?;
    let ix = s.tensor_variable(idx.clone(), vec![ni], false)?;
    let o = s.tensor_embedding(ix, w, None)?;
    let dt = s.tensor_dtype(o)?;
    let fv: Vec<f32> = s
        .tensor_values_lossy_f64(o)?
        .iter()
        .take(4096)
        .map(|&v| v as f32)
        .collect();
    let py = format!(
        r#"
import time,torch
import torch.nn.functional as F
torch.set_num_threads(8)
ne,dim,ni={ne},{dim},{ni}
weight=(((torch.arange(ne*dim,dtype=torch.int64)%997).float()/500.0)-1.0).reshape(ne,dim)
idx=((torch.arange(ni,dtype=torch.int64)*7919)%ne)
def tm(fn,reps=7):
    for _ in range(2): fn()
    ts=[]
    for _ in range(reps): s=time.perf_counter(); fn(); ts.append((time.perf_counter()-s)*1e3)
    return min(ts)
print("PT emb %.3f"%tm(lambda:F.embedding(idx,weight)))
o=F.embedding(idx,weight); assert o.dtype==torch.float32
print("REF "+" ".join("%a"%float(v) for v in o.flatten()[:4096].tolist()))
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
    let pt = g("emb");
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
    println!("embedding [{ne}x{dim}] gather {ni} idx f32:");
    println!("  perf:  FT {ft:8.3}ms  torch {pt:8.3}ms => {vrb}");
    println!(
        "  value: dtype={dt:?} bit_exact={exact}/{}",
        fv.len().min(tv.len())
    );
    Ok(())
}

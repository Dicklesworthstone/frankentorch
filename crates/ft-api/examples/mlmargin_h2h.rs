// multilabel_margin_loss: value vs torch + perf (no-grad fused vs serial).
use ft_api::FrankenTorchSession;
use ft_core::ExecutionMode;
use std::io::Write;
use std::process::{Command, Stdio};
use std::time::Instant;
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let python = std::env::var("PYTORCH_PYTHON").unwrap_or_else(|_| "python3".to_string());
    let nn = 50_000usize;
    let c = 64usize;
    let inp: Vec<f32> = (0..nn * c).map(|i| ((i % 211) as f32 / 100.0) - 1.0).collect();
    // target [N,C] as f64 label indices (FT reads target via f64): 2 positive labels/row, rest -1
    let tgt: Vec<f64> = (0..nn * c).map(|i| { let (r, j) = (i / c, i % c); if j == 0 { (r % c) as f64 } else if j == 1 { ((r + 7) % c) as f64 } else { -1.0 } }).collect();
    let bench = || {
        let mut best = f64::INFINITY;
        for _ in 0..7 {
            let mut s = FrankenTorchSession::new(ExecutionMode::Strict);
            let x = s.tensor_variable_f32(inp.clone(), vec![nn, c], false).unwrap();
            let t = s.tensor_variable(tgt.clone(), vec![nn, c], false).unwrap();
            let ti = Instant::now();
            let _ = s.tensor_multilabel_margin_loss(x, t, "mean");
            let e = ti.elapsed().as_secs_f64() * 1e3;
            if e < best { best = e; }
        }
        best
    };
    let mut s = FrankenTorchSession::new(ExecutionMode::Strict);
    let x = s.tensor_variable_f32(inp.clone(), vec![nn, c], false)?;
    let t = s.tensor_variable(tgt.clone(), vec![nn, c], false)?;
    let loss = s.tensor_multilabel_margin_loss(x, t, "mean")?;
    let ft_val = s.tensor_values_lossy_f64(loss)?[0];
    let ft_ms = bench();
    let py = format!(r#"
import time,torch
import torch.nn.functional as F
torch.set_num_threads(8)
nn={nn}; c={c}
inp=(((torch.arange(nn*c,dtype=torch.int64)%211).float()/100.0)-1.0).reshape(nn,c)
idx=torch.arange(nn*c,dtype=torch.int64); r=idx//c; j=idx%c
tgt=torch.full((nn*c,),-1,dtype=torch.int64)
tgt[j==0]=(r[j==0]%c); tgt[j==1]=((r[j==1]+7)%c)
tgt=tgt.reshape(nn,c)
def tm(fn,reps=7):
    for _ in range(2): fn()
    ts=[]
    for _ in range(reps): s=time.perf_counter(); fn(); ts.append((time.perf_counter()-s)*1e3)
    return min(ts)
v=F.multilabel_margin_loss(inp,tgt,reduction='mean')
print("REFVAL %.10e"%float(v))
print("PT %.3f"%tm(lambda:F.multilabel_margin_loss(inp,tgt,reduction='mean')))
"#);
    let mut ch = Command::new(&python).arg("-").stdin(Stdio::piped()).stdout(Stdio::piped()).spawn()?;
    ch.stdin.as_mut().unwrap().write_all(py.as_bytes())?;
    let out = String::from_utf8_lossy(&ch.wait_with_output()?.stdout).to_string();
    let g = |k: &str| out.lines().find_map(|l| l.strip_prefix(k)?.trim().parse::<f64>().ok());
    let tv = g("REFVAL ").unwrap_or(f64::NAN);
    let pt = g("PT ").unwrap_or(f64::NAN);
    let rel = (ft_val - tv).abs() / tv.abs().max(1e-30);
    println!("multilabel_margin [{nn}x{c}] f32 mean:");
    println!("  value: FT {ft_val:.10e}  torch {tv:.10e}  rel_err {rel:.3e}");
    let vrb = if pt >= ft_ms { format!("FT {:.2}x FASTER", pt / ft_ms) } else { format!("FT {:.2}x SLOWER", ft_ms / pt) };
    println!("  perf:  FT {ft_ms:8.3}ms  torch {pt:8.3}ms => {vrb}");
    Ok(())
}

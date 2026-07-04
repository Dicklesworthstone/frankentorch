// diagonal f32 head-to-head + BIT-EXACT vs torch (pure data-movement value op).
use ft_api::FrankenTorchSession;
use ft_core::ExecutionMode;
use std::io::Write;
use std::process::{Command, Stdio};
use std::time::Instant;
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let python = std::env::var("PYTORCH_PYTHON").unwrap_or_else(|_| "python3".to_string());
    let nn = 4000usize; // 4000x4000 = 16M f32
    let data: Vec<f32> = (0..nn * nn).map(|i| ((i % 9973) as f32 / 1000.0) - 5.0).collect();
    let bench = |w: u8| {
        let mut best = f64::INFINITY;
        for _ in 0..7 {
            let mut s = FrankenTorchSession::new(ExecutionMode::Strict);
            let x = s.tensor_variable_f32(data.clone(), vec![nn, nn], false).unwrap();
            let t = Instant::now();
            if w == 0 { let _ = s.tensor_add(x, x); } else { let _ = s.tensor_diagonal(x, 0); }
            let e = t.elapsed().as_secs_f64() * 1e3;
            if e < best { best = e; }
        }
        best
    };
    // correctness
    let mut s = FrankenTorchSession::new(ExecutionMode::Strict);
    let x = s.tensor_variable_f32(data.clone(), vec![nn, nn], false)?;
    let d = s.tensor_diagonal(x, 0)?;
    let dt = s.tensor_dtype(d)?;
    let fv: Vec<f32> = s.tensor_values_lossy_f64(d)?.iter().map(|&v| v as f32).collect();
    let m = 512usize.min(fv.len());
    let py = format!(r#"
import time,torch
torch.set_num_threads(8)
nn={nn}; m={m}
x=(((torch.arange(nn*nn,dtype=torch.int64)%9973).float()/1000.0)-5.0).reshape(nn,nn)
def tm(fn,reps=7):
    for _ in range(2): fn()
    ts=[]
    for _ in range(reps): s=time.perf_counter(); fn(); ts.append((time.perf_counter()-s)*1e3)
    return min(ts)
print("PT add %.3f"%tm(lambda:x+x))
print("PT diagonal %.3f"%tm(lambda:torch.diagonal(x,0).contiguous()))
d=torch.diagonal(x,0).contiguous(); assert d.dtype==torch.float32
print("REF "+" ".join("%a"%float(v) for v in d[:m].tolist()))
"#);
    let mut ch = Command::new(&python).arg("-").stdin(Stdio::piped()).stdout(Stdio::piped()).spawn()?;
    ch.stdin.as_mut().unwrap().write_all(py.as_bytes())?;
    let out = String::from_utf8_lossy(&ch.wait_with_output()?.stdout).to_string();
    let g = |k: &str| out.lines().find_map(|l| { let mut it = l.strip_prefix("PT ")?.split_whitespace(); if it.next()? == k { it.next()?.parse::<f64>().ok() } else { None } }).unwrap_or(f64::NAN);
    let vrb = |ft: f64, pp: f64| if pp >= ft { format!("FT {:.2}x FASTER", pp / ft) } else { format!("FT {:.2}x SLOWER", ft / pp) };
    println!("diagonal 4000x4000 f32 (torch 8t / FT default), min-of-7:");
    let (fa, fd) = (bench(0), bench(1));
    println!("  add       FT {fa:8.3}  PT {:8.3}  => {}", g("add"), vrb(fa, g("add")));
    println!("  diagonal  FT {fd:8.3}  PT {:8.3}  => {}", g("diagonal"), vrb(fd, g("diagonal")));
    let line = out.lines().find(|l| l.starts_with("REF ")).unwrap_or("");
    let tv: Vec<f32> = line.split_whitespace().skip(1).filter_map(|t| t.parse().ok()).collect();
    let mut exact = 0usize;
    for (&f, &t) in fv.iter().take(tv.len()).zip(tv.iter()) { if f.to_bits() == t.to_bits() { exact += 1; } }
    println!("correctness: diagonal dtype={dt:?} len={} bit_exact={exact}/{}", fv.len(), tv.len());
    Ok(())
}

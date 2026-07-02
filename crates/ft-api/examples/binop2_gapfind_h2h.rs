//! Data-driven gapfind over binary/composite no-grad ops vs torch, 16M f64. `add` = anchor (must read
//! ~parity). Inputs materialized OUTSIDE the timer (a 16M clone is ~ms and would swamp). Finds which
//! ops fall through to a slow tape/compose path (candidates for a no-grad fused fast path).
use ft_api::FrankenTorchSession;
use ft_core::ExecutionMode;
use std::io::Write;
use std::process::{Command, Stdio};
use std::time::Instant;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let python = std::env::var("PYTORCH_PYTHON").unwrap_or_else(|_| "python3".to_string());
    let n = 16_000_000usize;
    // y in a spread incl negatives; x avoids 0 mostly (atan2/copysign) and >0 for float_power base.
    let yd: Vec<f64> = (0..n).map(|i| ((i % 2000) as f64) * 0.01 - 10.0).collect();
    let xd: Vec<f64> = (0..n).map(|i| ((i % 1500) as f64) * 0.013 + 0.5).collect();

    let ops: &[&str] = &["add", "atan2", "hypot", "copysign", "clamp_tensor"];
    let run = |which: &str| -> f64 {
        let mut best = f64::INFINITY;
        for _ in 0..7 {
            let mut s = FrankenTorchSession::new(ExecutionMode::Strict);
            let y = s.tensor_variable(yd.clone(), vec![n], false).unwrap();
            let x = s.tensor_variable(xd.clone(), vec![n], false).unwrap();
            let lo = s.tensor_variable(vec![-3.0; n], vec![n], false).unwrap();
            let hi = s.tensor_variable(vec![3.0; n], vec![n], false).unwrap();
            let t = Instant::now();
            let _ = match which {
                "add" => s.tensor_add(y, x).unwrap(),
                "atan2" => s.tensor_atan2(y, x).unwrap(),
                "hypot" => s.tensor_hypot(y, x).unwrap(),
                "copysign" => s.tensor_copysign(y, x).unwrap(),
                "clamp_tensor" => s.tensor_clamp_tensor(y, lo, hi).unwrap(),
                _ => unreachable!(),
            };
            let e = t.elapsed().as_secs_f64() * 1e3;
            if e < best { best = e; }
        }
        best
    };
    let ft: Vec<(String, f64)> = ops.iter().map(|o| (o.to_string(), run(o))).collect();

    let py = format!(
        r#"
import time,torch
torch.set_num_threads(8)
n={n}
y=((torch.arange(n,dtype=torch.int64)%2000).double())*0.01-10.0
x=((torch.arange(n,dtype=torch.int64)%1500).double())*0.013+0.5
lo=torch.full((n,),-3.0,dtype=torch.float64); hi=torch.full((n,),3.0,dtype=torch.float64)
def t(fn,reps=7):
    for _ in range(2): fn()
    ts=[]
    for _ in range(reps): s=time.perf_counter(); fn(); ts.append((time.perf_counter()-s)*1e3)
    return min(ts)
print("PT add %.4f"%t(lambda:y+x))
print("PT atan2 %.4f"%t(lambda:torch.atan2(y,x)))
print("PT hypot %.4f"%t(lambda:torch.hypot(y,x)))
print("PT copysign %.4f"%t(lambda:torch.copysign(y,x)))
print("PT clamp_tensor %.4f"%t(lambda:torch.clamp(y,lo,hi)))
"#
    );
    let mut ch = Command::new(&python).arg("-").stdin(Stdio::piped()).stdout(Stdio::piped()).spawn()?;
    ch.stdin.as_mut().unwrap().write_all(py.as_bytes())?;
    let pt = String::from_utf8_lossy(&ch.wait_with_output()?.stdout).to_string();
    let g = |k: &str| pt.lines().find_map(|l| { let mut it = l.strip_prefix("PT ")?.split_whitespace(); if it.next()? == k { it.next()?.parse::<f64>().ok() } else { None } }).unwrap_or(f64::NAN);
    let v = |ft: f64, p: f64| if p >= ft { format!("FT {:.2}x FASTER", p / ft) } else { format!("FT {:.2}x SLOWER", ft / p) };
    for (name, ftms) in &ft {
        let pt = g(name);
        println!("  {name:14} FT {ftms:8.3}ms  PT {pt:8.3}ms  => {}", v(*ftms, pt));
    }
    Ok(())
}

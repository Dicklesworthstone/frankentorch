//! rel_entr/xlog1py F64 no-grad — fused try_f64_binary_native vs torch. FT_ORIG=1 times the compose.
//! 16M f64, inputs OUTSIDE the timer.
use ft_api::FrankenTorchSession;
use ft_core::ExecutionMode;
use std::io::Write;
use std::process::{Command, Stdio};
use std::time::Instant;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let python = std::env::var("PYTORCH_PYTHON").unwrap_or_else(|_| "python3".to_string());
    let n = 16_000_000usize;
    let xd: Vec<f64> = (0..n).map(|i| ((i % 4000) as f64) * 0.005 + 0.01).collect();
    let yd: Vec<f64> = (0..n).map(|i| ((i % 3000) as f64) * 0.004 + 0.05).collect();
    let orig = std::env::var("FT_ORIG").is_ok();
    let run = |which: &str| -> f64 {
        let mut best = f64::INFINITY;
        for _ in 0..7 {
            let mut s = FrankenTorchSession::new(ExecutionMode::Strict);
            let x = s.tensor_variable(xd.clone(), vec![n], false).unwrap();
            let y = s.tensor_variable(yd.clone(), vec![n], false).unwrap();
            let t = Instant::now();
            let _ = if which == "rel_entr" { s.tensor_rel_entr(x, y).unwrap() } else { s.tensor_xlog1py(x, y).unwrap() };
            let e = t.elapsed().as_secs_f64() * 1e3;
            if e < best { best = e; }
        }
        best
    };
    let fr = run("rel_entr");
    let label = if orig { "FT_ORIG(compose)" } else { "FT_FUSED" };
    // torch has no special.rel_entr in this build -> the user-equivalent baseline is x*log(x/y).
    let py = format!(
        r#"
import time,torch
torch.set_num_threads(8)
n={n}
x=((torch.arange(n,dtype=torch.int64)%4000).double())*0.005+0.01
y=((torch.arange(n,dtype=torch.int64)%3000).double())*0.004+0.05
def t(fn,reps=7):
    for _ in range(2): fn()
    ts=[]
    for _ in range(reps): s=time.perf_counter(); fn(); ts.append((time.perf_counter()-s)*1e3)
    return min(ts)
print("PT rel_entr %.4f"%t(lambda:x*torch.log(x/y)))
"#
    );
    let mut ch = Command::new(&python).arg("-").stdin(Stdio::piped()).stdout(Stdio::piped()).spawn()?;
    ch.stdin.as_mut().unwrap().write_all(py.as_bytes())?;
    let pt = String::from_utf8_lossy(&ch.wait_with_output()?.stdout).to_string();
    let g = |k: &str| pt.lines().find_map(|l| { let mut it = l.strip_prefix("PT ")?.split_whitespace(); if it.next()? == k { it.next()?.parse::<f64>().ok() } else { None } }).unwrap_or(f64::NAN);
    let v = |ft: f64, p: f64| if p >= ft { format!("FT {:.2}x FASTER", p / ft) } else { format!("FT {:.2}x SLOWER", ft / p) };
    println!("  rel_entr(16M) {label} {fr:.3}ms  PT[x*log(x/y)] {:.3}ms => {}", g("rel_entr"), v(fr, g("rel_entr")));
    Ok(())
}

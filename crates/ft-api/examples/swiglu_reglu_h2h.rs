//! swiglu (a*silu(b)) / reglu (a*relu(b)) F64 no-grad vs torch, transformer shape. FT_ORIG=1 times the
//! strided-split compose. Torch baseline = the user-equivalent chunk + act + mul. Inputs OUTSIDE timer.
use ft_api::FrankenTorchSession;
use ft_core::ExecutionMode;
use std::io::Write;
use std::process::{Command, Stdio};
use std::time::Instant;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let python = std::env::var("PYTORCH_PYTHON").unwrap_or_else(|_| "python3".to_string());
    let (b, s, d) = (64usize, 512usize, 512usize);
    let n = b * s * 2 * d;
    let xd: Vec<f64> = (0..n).map(|i| ((i % 991) as f64) * 0.01 - 5.0).collect();
    let run = |which: &str| -> f64 {
        let mut best = f64::INFINITY;
        for _ in 0..7 {
            let mut sess = FrankenTorchSession::new(ExecutionMode::Strict);
            let x = sess.tensor_variable(xd.clone(), vec![b, s, 2 * d], false).unwrap();
            let t = Instant::now();
            let _ = match which { "swiglu" => sess.tensor_swiglu(x, 2).unwrap(), "reglu" => sess.tensor_reglu(x, 2).unwrap(), _ => sess.tensor_geglu(x, 2).unwrap() };
            let e = t.elapsed().as_secs_f64() * 1e3;
            if e < best { best = e; }
        }
        best
    };
    let (fsg, frg, fgg) = (run("swiglu"), run("reglu"), run("geglu"));
    let label = if std::env::var("FT_ORIG").is_ok() { "FT_ORIG(compose)" } else { "FT_FUSED" };
    let py = format!(
        r#"
import time,torch,torch.nn.functional as F
torch.set_num_threads(8)
b,s,d={b},{s},{d}
x=((torch.arange(b*s*2*d,dtype=torch.int64)%991).double())*0.01-5.0; x=x.reshape(b,s,2*d)
def t(fn,reps=7):
    for _ in range(2): fn()
    ts=[]
    for _ in range(reps): st=time.perf_counter(); fn(); ts.append((time.perf_counter()-st)*1e3)
    return min(ts)
def swiglu(x):
    a,bb=x.chunk(2,dim=2); return a*F.silu(bb)
def reglu(x):
    a,bb=x.chunk(2,dim=2); return a*F.relu(bb)
def geglu(x):
    a,bb=x.chunk(2,dim=2); return a*F.gelu(bb)
print("PT swiglu %.4f"%t(lambda:swiglu(x)))
print("PT reglu %.4f"%t(lambda:reglu(x)))
print("PT geglu %.4f"%t(lambda:geglu(x)))
"#
    );
    let mut ch = Command::new(&python).arg("-").stdin(Stdio::piped()).stdout(Stdio::piped()).spawn()?;
    ch.stdin.as_mut().unwrap().write_all(py.as_bytes())?;
    let pt = String::from_utf8_lossy(&ch.wait_with_output()?.stdout).to_string();
    let g = |k: &str| pt.lines().find_map(|l| { let mut it = l.strip_prefix("PT ")?.split_whitespace(); if it.next()? == k { it.next()?.parse::<f64>().ok() } else { None } }).unwrap_or(f64::NAN);
    let v = |ft: f64, p: f64| if p >= ft { format!("FT {:.2}x FASTER", p / ft) } else { format!("FT {:.2}x SLOWER", ft / p) };
    println!("  swiglu {label} {fsg:.3}ms  PT {:.3}ms => {}", g("swiglu"), v(fsg, g("swiglu")));
    println!("  reglu  {label} {frg:.3}ms  PT {:.3}ms => {}", g("reglu"), v(frg, g("reglu")));
    println!("  geglu  {label} {fgg:.3}ms  PT {:.3}ms => {}", g("geglu"), v(fgg, g("geglu")));
    Ok(())
}

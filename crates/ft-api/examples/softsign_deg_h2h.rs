//! softsign/deg2rad/rad2deg F64 no-grad — fused try_f64_unary_native vs torch. FT_ORIG=1 times the
//! compose. 16M f64, inputs OUTSIDE the timer.
use ft_api::FrankenTorchSession;
use ft_core::ExecutionMode;
use std::io::Write;
use std::process::{Command, Stdio};
use std::time::Instant;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let python = std::env::var("PYTORCH_PYTHON").unwrap_or_else(|_| "python3".to_string());
    let n = 16_000_000usize;
    let xd: Vec<f64> = (0..n).map(|i| ((i % 4000) as f64) * 0.01 - 20.0).collect();
    let orig = std::env::var("FT_ORIG").is_ok();
    let run = |which: &str| -> f64 {
        let mut best = f64::INFINITY;
        for _ in 0..7 {
            let mut s = FrankenTorchSession::new(ExecutionMode::Strict);
            let x = s.tensor_variable(xd.clone(), vec![n], false).unwrap();
            let t = Instant::now();
            let _ = match which {
                "softsign" => s.tensor_softsign(x).unwrap(),
                "deg2rad" => s.tensor_deg2rad(x).unwrap(),
                _ => s.tensor_rad2deg(x).unwrap(),
            };
            let e = t.elapsed().as_secs_f64() * 1e3;
            if e < best {
                best = e;
            }
        }
        best
    };
    let vals: Vec<(&str, f64)> = ["softsign", "deg2rad", "rad2deg"]
        .iter()
        .map(|o| (*o, run(o)))
        .collect();
    let label = if orig { "FT_ORIG(compose)" } else { "FT_FUSED" };
    let py = format!(
        r#"
import time,torch,torch.nn.functional as F
torch.set_num_threads(8)
n={n}
x=((torch.arange(n,dtype=torch.int64)%4000).double())*0.01-20.0
def t(fn,reps=7):
    for _ in range(2): fn()
    ts=[]
    for _ in range(reps): s=time.perf_counter(); fn(); ts.append((time.perf_counter()-s)*1e3)
    return min(ts)
print("PT softsign %.4f"%t(lambda:F.softsign(x)))
print("PT deg2rad %.4f"%t(lambda:torch.deg2rad(x)))
print("PT rad2deg %.4f"%t(lambda:torch.rad2deg(x)))
"#
    );
    let mut ch = Command::new(&python)
        .arg("-")
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .spawn()?;
    ch.stdin.as_mut().unwrap().write_all(py.as_bytes())?;
    let pt = String::from_utf8_lossy(&ch.wait_with_output()?.stdout).to_string();
    let g = |k: &str| {
        pt.lines()
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
    let v = |ft: f64, p: f64| {
        if p >= ft {
            format!("FT {:.2}x FASTER", p / ft)
        } else {
            format!("FT {:.2}x SLOWER", ft / p)
        }
    };
    for (name, ftms) in &vals {
        println!(
            "  {name:9} {label} {ftms:.3}ms  PT {:.3}ms => {}",
            g(name),
            v(*ftms, g(name))
        );
    }
    Ok(())
}

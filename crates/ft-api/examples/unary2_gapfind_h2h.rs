//! Data-driven gapfind over unary/binary composite ops vs torch, 16M f64. `exp`/`add` = anchors (must
//! read ~parity-or-faster). Inputs materialized OUTSIDE the timer. Finds ops that fall through to a
//! slow compose/tape path (candidates for a no-grad fused fast path).
use ft_api::FrankenTorchSession;
use ft_core::ExecutionMode;
use std::io::Write;
use std::process::{Command, Stdio};
use std::time::Instant;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let python = std::env::var("PYTORCH_PYTHON").unwrap_or_else(|_| "python3".to_string());
    let n = 16_000_000usize;
    let xd: Vec<f64> = (0..n).map(|i| ((i % 4000) as f64) * 0.005 - 10.0).collect();
    let dd: Vec<f64> = (0..n).map(|i| ((i % 300) as f64) * 0.01 + 0.7).collect(); // divisor for remainder/fmod

    let ops: &[&str] = &[
        "exp",
        "add",
        "frac",
        "hardswish",
        "mish",
        "softplus",
        "logsigmoid",
        "hardsigmoid",
        "remainder",
        "fmod",
    ];
    let run = |which: &str| -> f64 {
        let mut best = f64::INFINITY;
        for _ in 0..7 {
            let mut s = FrankenTorchSession::new(ExecutionMode::Strict);
            let x = s.tensor_variable(xd.clone(), vec![n], false).unwrap();
            let d = s.tensor_variable(dd.clone(), vec![n], false).unwrap();
            let t = Instant::now();
            let _ = match which {
                "exp" => s.tensor_exp(x).unwrap(),
                "add" => s.tensor_add(x, d).unwrap(),
                "frac" => s.tensor_frac(x).unwrap(),
                "hardswish" => s.tensor_hardswish(x).unwrap(),
                "mish" => s.tensor_mish(x).unwrap(),
                "softplus" => s.tensor_softplus(x).unwrap(),
                "logsigmoid" => s.tensor_logsigmoid(x).unwrap(),
                "hardsigmoid" => s.tensor_hardsigmoid(x).unwrap(),
                "remainder" => s.tensor_remainder(x, d).unwrap(),
                "fmod" => s.tensor_fmod(x, d).unwrap(),
                _ => unreachable!(),
            };
            let e = t.elapsed().as_secs_f64() * 1e3;
            if e < best {
                best = e;
            }
        }
        best
    };
    let ft: Vec<(String, f64)> = ops.iter().map(|o| (o.to_string(), run(o))).collect();

    let py = format!(
        r#"
import time,torch,torch.nn.functional as F
torch.set_num_threads(8)
n={n}
x=((torch.arange(n,dtype=torch.int64)%4000).double())*0.005-10.0
d=((torch.arange(n,dtype=torch.int64)%300).double())*0.01+0.7
def t(fn,reps=7):
    for _ in range(2): fn()
    ts=[]
    for _ in range(reps): s=time.perf_counter(); fn(); ts.append((time.perf_counter()-s)*1e3)
    return min(ts)
print("PT exp %.4f"%t(lambda:torch.exp(x)))
print("PT add %.4f"%t(lambda:x+d))
print("PT frac %.4f"%t(lambda:torch.frac(x)))
print("PT hardswish %.4f"%t(lambda:F.hardswish(x)))
print("PT mish %.4f"%t(lambda:F.mish(x)))
print("PT softplus %.4f"%t(lambda:F.softplus(x)))
print("PT logsigmoid %.4f"%t(lambda:F.logsigmoid(x)))
print("PT hardsigmoid %.4f"%t(lambda:F.hardsigmoid(x)))
print("PT remainder %.4f"%t(lambda:torch.remainder(x,d)))
print("PT fmod %.4f"%t(lambda:torch.fmod(x,d)))
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
    for (name, ftms) in &ft {
        let pt = g(name);
        println!(
            "  {name:14} FT {ftms:8.3}ms  PT {pt:8.3}ms  => {}",
            v(*ftms, pt)
        );
    }
    Ok(())
}

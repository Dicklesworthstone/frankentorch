//! cumsum/cumprod/logcumsumexp F64 no-grad vs torch — scan ops. Along last dim (contiguous lanes) and
//! dim=0 (strided lanes). Finds whether the scan kernels have a real gap (parallel-scan lever) or are
//! already competitive. Inputs OUTSIDE the timer.
use ft_api::FrankenTorchSession;
use ft_core::ExecutionMode;
use std::io::Write;
use std::process::{Command, Stdio};
use std::time::Instant;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let python = std::env::var("PYTORCH_PYTHON").unwrap_or_else(|_| "python3".to_string());
    let (rows, cols) = (4096usize, 4096usize); // 16M, scan along cols (last dim) or rows (dim0)
    let n = rows * cols;
    let xd: Vec<f64> = (0..n).map(|i| ((i % 101) as f64) * 0.001 + 0.01).collect();
    let run = |which: &str, dim: usize| -> f64 {
        let mut best = f64::INFINITY;
        for _ in 0..7 {
            let mut s = FrankenTorchSession::new(ExecutionMode::Strict);
            let x = s
                .tensor_variable(xd.clone(), vec![rows, cols], false)
                .unwrap();
            let t = Instant::now();
            let _ = match which {
                "cumsum" => s.tensor_cumsum(x, dim).unwrap(),
                "cumprod" => s.tensor_cumprod(x, dim).unwrap(),
                _ => s.tensor_logcumsumexp(x, dim).unwrap(),
            };
            let e = t.elapsed().as_secs_f64() * 1e3;
            if e < best {
                best = e;
            }
        }
        best
    };
    let results: Vec<(String, f64)> = [
        ("cumsum", 1usize),
        ("cumsum", 0),
        ("cumprod", 1),
        ("logcumsumexp", 1),
    ]
    .iter()
    .map(|(o, d)| (format!("{o}_d{d}"), run(o, *d)))
    .collect();
    let py = format!(
        r#"
import time,torch
torch.set_num_threads(8)
r,c={rows},{cols}
x=((torch.arange(r*c,dtype=torch.int64)%101).double())*0.001+0.01; x=x.reshape(r,c)
def t(fn,reps=7):
    for _ in range(2): fn()
    ts=[]
    for _ in range(reps): s=time.perf_counter(); fn(); ts.append((time.perf_counter()-s)*1e3)
    return min(ts)
print("PT cumsum_d1 %.4f"%t(lambda:torch.cumsum(x,1)))
print("PT cumsum_d0 %.4f"%t(lambda:torch.cumsum(x,0)))
print("PT cumprod_d1 %.4f"%t(lambda:torch.cumprod(x,1)))
print("PT logcumsumexp_d1 %.4f"%t(lambda:torch.logcumsumexp(x,1)))
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
    for (name, ftms) in &results {
        println!(
            "  {name:18} FT {ftms:8.3}ms  PT {:8.3}ms => {}",
            g(name),
            v(*ftms, g(name))
        );
    }
    Ok(())
}

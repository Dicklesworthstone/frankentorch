//! Sweep elementwise activations for f32->f64 upcast gaps.
use ft_api::FrankenTorchSession;
use ft_core::ExecutionMode;
use std::io::Write;
use std::process::{Command, Stdio};
use std::time::Instant;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let python = std::env::var("PYTORCH_PYTHON").unwrap_or_else(|_| "python3".to_string());
    let n = 16_000_000usize;
    let a: Vec<f32> = (0..n)
        .map(|i| ((i % 9973) as f32 - 5000.0) * 0.01)
        .collect();
    let tt = |which: u8| {
        let mut best = f64::INFINITY;
        for _ in 0..7 {
            let mut s = FrankenTorchSession::new(ExecutionMode::Strict);
            let x = s.tensor_variable_f32(a.clone(), vec![n], false).unwrap();
            let ti = Instant::now();
            match which {
                0 => {
                    let _ = s.tensor_add(x, x);
                }
                1 => {
                    let _ = s.tensor_relu6(x);
                }
                2 => {
                    let _ = s.tensor_celu(x, 1.0);
                }
                3 => {
                    let _ = s.tensor_selu(x);
                }
                4 => {
                    let _ = s.tensor_elu(x);
                }
                5 => {
                    let _ = s.tensor_hardswish(x);
                }
                6 => {
                    let _ = s.tensor_mish(x);
                }
                7 => {
                    let _ = s.tensor_softplus(x);
                }
                8 => {
                    let _ = s.tensor_softsign(x);
                }
                _ => {
                    let _ = s.tensor_tanhshrink(x);
                }
            }
            let e = ti.elapsed().as_secs_f64() * 1e3;
            if e < best {
                best = e;
            }
        }
        best
    };
    let py = format!(
        r#"
import time, torch
import torch.nn.functional as F
torch.set_num_threads(8)
n={n}
a=(((torch.arange(n,dtype=torch.int64)%9973).float()-5000.0)*0.01)
def tm(fn,reps=7):
    for _ in range(2): fn()
    ts=[]
    for _ in range(reps): s=time.perf_counter(); fn(); ts.append((time.perf_counter()-s)*1e3)
    return min(ts)
print("PT add %.3f"%tm(lambda:a+a))
print("PT relu6 %.3f"%tm(lambda:F.relu6(a)))
print("PT celu %.3f"%tm(lambda:F.celu(a,1.0)))
print("PT selu %.3f"%tm(lambda:F.selu(a)))
print("PT elu %.3f"%tm(lambda:F.elu(a)))
print("PT hardswish %.3f"%tm(lambda:F.hardswish(a)))
print("PT mish %.3f"%tm(lambda:F.mish(a)))
print("PT softplus %.3f"%tm(lambda:F.softplus(a)))
print("PT softsign %.3f"%tm(lambda:F.softsign(a)))
print("PT tanhshrink %.3f"%tm(lambda:F.tanhshrink(a)))
"#,
        n = n
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
    println!("activation sweep ~16M f32 (torch 8t / FT default), min-of-7");
    for (lbl, w) in [
        ("add", 0u8),
        ("relu6", 1),
        ("celu", 2),
        ("selu", 3),
        ("elu", 4),
        ("hardswish", 5),
        ("mish", 6),
        ("softplus", 7),
        ("softsign", 8),
        ("tanhshrink", 9),
    ] {
        let ft = tt(w);
        println!(
            "  {lbl:<11} FT {ft:8.3}  PT {:8.3}  => {}",
            g(lbl),
            v(ft, g(lbl))
        );
    }
    Ok(())
}

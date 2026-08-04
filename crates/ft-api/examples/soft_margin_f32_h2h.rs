//! f32 soft_margin_loss vs PyTorch. ORIG returned F64 for f32 because the
//! composed path built F64 constants; this probe reports functional status and
//! no-grad mean timing. relu = f32 unary anchor.

use std::io::Write;
use std::process::{Command, Stdio};
use std::time::Instant;

use ft_api::FrankenTorchSession;
use ft_core::ExecutionMode;

const N: usize = 16_000_000;

fn time_relu(x_data: &[f32]) -> Result<f64, Box<dyn std::error::Error>> {
    let mut best = f64::INFINITY;
    for _ in 0..9 {
        let mut s = FrankenTorchSession::new(ExecutionMode::Strict);
        let x = s.tensor_variable_f32(x_data.to_vec(), vec![N], false)?;
        let start = Instant::now();
        let _ = s.tensor_relu(x)?;
        let elapsed = start.elapsed().as_secs_f64() * 1e3;
        if elapsed < best {
            best = elapsed;
        }
    }
    Ok(best)
}

fn time_soft_margin(x_data: &[f32], t_data: &[f32]) -> Result<f64, Box<dyn std::error::Error>> {
    let mut best_ms = f64::INFINITY;
    let mut last_err = None;
    for _ in 0..9 {
        let mut s = FrankenTorchSession::new(ExecutionMode::Strict);
        let x = s.tensor_variable_f32(x_data.to_vec(), vec![N], false)?;
        let t = s.tensor_variable_f32(t_data.to_vec(), vec![N], false)?;
        let start = Instant::now();
        match s.tensor_soft_margin_loss(x, t, "mean") {
            Ok(_) => {
                let elapsed = start.elapsed().as_secs_f64() * 1e3;
                if elapsed < best_ms {
                    best_ms = elapsed;
                }
            }
            Err(err) => {
                last_err = Some(format!("{err:?}"));
            }
        }
    }
    if best_ms.is_finite() {
        Ok(best_ms)
    } else {
        Err(last_err
            .unwrap_or_else(|| "soft_margin_loss failed".to_string())
            .into())
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let x_data: Vec<f32> = (0..N)
        .map(|i| -2.0 + (i % 257) as f32 * (4.0 / 257.0))
        .collect();
    let t_data: Vec<f32> = (0..N)
        .map(|i| if i % 2 == 0 { 1.0 } else { -1.0 })
        .collect();

    let mut parity = FrankenTorchSession::new(ExecutionMode::Strict);
    let px = parity.tensor_variable_f32(x_data[..8].to_vec(), vec![8], false)?;
    let pt = parity.tensor_variable_f32(t_data[..8].to_vec(), vec![8], false)?;
    match parity.tensor_soft_margin_loss(px, pt, "none") {
        Ok(out) => {
            let vals = parity.tensor_values_f32(out)?;
            println!(
                "FT parity OK dtype={:?} vals={vals:?}",
                parity.tensor_dtype(out)?
            );
        }
        Err(err) => println!("FT parity ERR {err:?}"),
    }

    let ft_relu = time_relu(&x_data)?;
    let ft_soft = time_soft_margin(&x_data, &t_data);

    let python = std::env::var("PYTORCH_PYTHON").unwrap_or_else(|_| "python3".to_string());
    let py = r#"
import time, torch
import torch.nn.functional as F
torch.set_num_threads(8)
N=16_000_000
idx=torch.arange(N,dtype=torch.int64)
x=(-2.0+(idx%257).float()*(4.0/257.0))
t=torch.where(idx%2==0, torch.tensor(1.0,dtype=torch.float32), torch.tensor(-1.0,dtype=torch.float32))
def tm(fn,n=9):
    for _ in range(2):
        fn()
    best=1e99
    for _ in range(n):
        s=time.perf_counter(); fn(); best=min(best,(time.perf_counter()-s)*1e3)
    return best
print("PT parity", F.soft_margin_loss(x[:8],t[:8],reduction='none').tolist())
print("PT relu %.4f"%tm(lambda:torch.relu(x)))
print("PT soft_margin %.4f"%tm(lambda:F.soft_margin_loss(x,t,reduction='mean')))
"#;
    let mut child = Command::new(&python)
        .arg("-")
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .spawn()?;
    child
        .stdin
        .as_mut()
        .ok_or_else(|| std::io::Error::other("no stdin"))?
        .write_all(py.as_bytes())?;
    let out = child.wait_with_output()?;
    let pt = if out.status.success() {
        String::from_utf8_lossy(&out.stdout).to_string()
    } else {
        String::new()
    };
    print!("{pt}");
    let lookup = |name: &str| -> Option<f64> {
        pt.lines().find_map(|line| {
            let mut parts = line.strip_prefix("PT ")?.split_whitespace();
            if parts.next()? == name {
                parts.next()?.parse().ok()
            } else {
                None
            }
        })
    };

    println!("FT relu {ft_relu:.4}");
    match ft_soft {
        Ok(ms) => {
            println!("FT soft_margin {ms:.4}");
            if let Some(pt_ms) = lookup("soft_margin") {
                let ratio = pt_ms / ms;
                if ratio >= 1.0 {
                    println!("RATIO soft_margin FT {ratio:.2}x FASTER");
                } else {
                    println!("RATIO soft_margin FT {:.2}x SLOWER", 1.0 / ratio);
                }
            }
        }
        Err(err) => println!("FT soft_margin ERR {err}"),
    }

    Ok(())
}

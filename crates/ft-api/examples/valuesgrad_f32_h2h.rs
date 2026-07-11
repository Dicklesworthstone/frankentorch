use std::error::Error;
use std::io::Write;
use std::process::{Command, Stdio};
use std::time::Instant;

use ft_api::FrankenTorchSession;
use ft_core::ExecutionMode;

fn boxed<E: std::fmt::Debug>(err: E) -> std::io::Error {
    std::io::Error::other(format!("{err:?}"))
}

fn fill(op: &str, batch: usize, m: usize, n: usize) -> Vec<f32> {
    let mut a = vec![0.0_f32; batch * m * n];
    for plane in 0..batch {
        for r in 0..m {
            for c in 0..n {
                let v = ((((plane + 1) * (r + 2) * (c + 3)) % 19) as f32 - 9.0) * 0.02;
                a[plane * m * n + r * n + c] = v + if r == c { 3.0 + c as f32 } else { 0.0 };
            }
        }
    }
    if op == "eigvalsh" {
        for plane in 0..batch {
            let b = plane * m * n;
            for r in 0..m {
                for c in (r + 1)..n {
                    let avg = 0.5 * (a[b + r * n + c] + a[b + c * n + r]);
                    a[b + r * n + c] = avg;
                    a[b + c * n + r] = avg;
                }
            }
        }
    }
    a
}

fn run_ft(op: &str, batch: usize, m: usize, n: usize) -> Result<f64, Box<dyn Error>> {
    let mut best = f64::INFINITY;
    for _ in 0..5 {
        let data = fill(op, batch, m, n);
        let mut s = FrankenTorchSession::new(ExecutionMode::Strict);
        let a = s
            .tensor_variable_f32(data, vec![batch, m, n], true)
            .map_err(boxed)?;
        let start = Instant::now();
        let out = if op == "eigvalsh" {
            s.tensor_linalg_eigvalsh(a).map_err(boxed)?
        } else {
            s.tensor_linalg_svdvals(a).map_err(boxed)?
        };
        let sq = s.tensor_mul(out, out).map_err(boxed)?;
        let loss = s.tensor_sum(sq).map_err(boxed)?;
        s.tensor_backward(loss).map_err(boxed)?;
        let elapsed_ms = start.elapsed().as_secs_f64() * 1e3;
        if elapsed_ms < best {
            best = elapsed_ms;
        }
    }
    Ok(best)
}

fn run_pytorch(op: &str, batch: usize, m: usize, n: usize) -> Option<f64> {
    let python = std::env::var("PYTORCH_PYTHON").unwrap_or_else(|_| "python3".to_string());
    let body = if op == "eigvalsh" {
        "torch.linalg.eigvalsh(Ar)"
    } else {
        "torch.linalg.svdvals(Ar)"
    };
    let script = format!(
        r#"
import time, torch
torch.set_num_threads(8); torch.set_num_interop_threads(8)
B,M,N,op = {batch},{m},{n},"{op}"
p=torch.arange(B,dtype=torch.int64).reshape(B,1,1)
rows=torch.arange(M,dtype=torch.int64).reshape(1,M,1)
cols=torch.arange(N,dtype=torch.int64).reshape(1,1,N)
A=((((p+1)*(rows+2)*(cols+3))%19).to(torch.float32)-9.0)*0.02
A=A+torch.eye(M,N,dtype=torch.float32).unsqueeze(0)*(3.0+cols.to(torch.float32))
if op=="eigvalsh": A=0.5*(A+A.transpose(-1,-2))
def step():
    Ar=A.clone().requires_grad_(True); ({body}**2).sum().backward(); return Ar.grad
for _ in range(2): step()
s=[]
for _ in range(5):
    t=time.perf_counter(); step(); s.append((time.perf_counter()-t)*1e3)
print("MS", min(s))
"#
    );
    let mut child = Command::new(&python)
        .arg("-")
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .ok()?;
    child.stdin.as_mut()?.write_all(script.as_bytes()).ok()?;
    let output = child.wait_with_output().ok()?;
    if !output.status.success() {
        eprintln!("{}", String::from_utf8_lossy(&output.stderr));
        return None;
    }
    let stdout = String::from_utf8_lossy(&output.stdout);
    stdout
        .lines()
        .find_map(|l| l.strip_prefix("MS "))
        .and_then(|v| v.trim().parse::<f64>().ok())
}

fn main() -> Result<(), Box<dyn Error>> {
    for (op, m, n) in [
        ("eigvalsh", 8, 8),
        ("eigvalsh", 16, 16),
        ("eigvalsh", 32, 32),
        ("svdvals", 8, 4),
        ("svdvals", 16, 8),
        ("svdvals", 32, 16),
    ] {
        let batch = match m {
            8 => 20_000usize,
            16 => 8_000,
            _ => 3_000,
        };
        let ft_ms = run_ft(op, batch, m, n)?;
        print!("op={op} B={batch} m={m} n={n}: FT {ft_ms:.3} ms");
        if let Some(tms) = run_pytorch(op, batch, m, n) {
            let ratio = tms / ft_ms;
            let tag = if ratio >= 1.0 { "FASTER" } else { "SLOWER" };
            println!(" | PyTorch {tms:.3} ms | FT {ratio:.2}x {tag}");
        } else {
            println!(" | PyTorch unavailable");
        }
    }
    Ok(())
}

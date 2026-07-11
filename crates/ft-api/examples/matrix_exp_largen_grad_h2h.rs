use std::error::Error;
use std::io::Write;
use std::process::{Command, Stdio};
use std::time::Instant;

use ft_api::FrankenTorchSession;
use ft_core::ExecutionMode;

fn boxed<E: std::fmt::Debug>(err: E) -> std::io::Error {
    std::io::Error::other(format!("{err:?}"))
}

fn fill(batch: usize, n: usize) -> Vec<f64> {
    let mut a = vec![0.0_f64; batch * n * n];
    for plane in 0..batch {
        for r in 0..n {
            for c in 0..n {
                let v = ((((plane + 1) * (r + 2) * (c + 3)) % 19) as f64 - 9.0) * 0.01;
                a[plane * n * n + r * n + c] = v + if r == c { 0.5 } else { 0.0 };
            }
        }
    }
    a
}

fn run_ft(batch: usize, n: usize) -> Result<(f64, f64), Box<dyn Error>> {
    let mut best = f64::INFINITY;
    let mut checksum = 0.0_f64;
    for _ in 0..3 {
        let data = fill(batch, n);
        let mut s = FrankenTorchSession::new(ExecutionMode::Strict);
        let a = s
            .tensor_variable(data, vec![batch, n, n], true)
            .map_err(boxed)?;
        let start = Instant::now();
        let y = s.tensor_matrix_exp(a).map_err(boxed)?;
        let loss = s.tensor_sum(y).map_err(boxed)?;
        s.tensor_backward(loss).map_err(boxed)?;
        let elapsed_ms = start.elapsed().as_secs_f64() * 1e3;
        if elapsed_ms < best {
            best = elapsed_ms;
            checksum = s
                .tensor_grad(a)
                .map_err(boxed)?
                .unwrap_or_default()
                .iter()
                .sum();
        }
    }
    Ok((best, checksum))
}

fn run_pytorch(batch: usize, n: usize) -> Option<(f64, f64)> {
    let python = std::env::var("PYTORCH_PYTHON").unwrap_or_else(|_| "python3".to_string());
    let script = format!(
        r#"
import time, torch
torch.set_num_threads(8); torch.set_num_interop_threads(8)
B, N = {batch},{n}
p=torch.arange(B,dtype=torch.int64).reshape(B,1,1)
rows=torch.arange(N,dtype=torch.int64).reshape(1,N,1)
cols=torch.arange(N,dtype=torch.int64).reshape(1,1,N)
A=((((p+1)*(rows+2)*(cols+3))%19).to(torch.float64)-9.0)*0.01
A=A+torch.eye(N,dtype=torch.float64).unsqueeze(0)*0.5
def step():
    Ar=A.clone().requires_grad_(True); torch.matrix_exp(Ar).sum().backward(); return Ar.grad
for _ in range(2): step()
s=[]
for _ in range(3):
    t=time.perf_counter(); step(); s.append((time.perf_counter()-t)*1e3)
g=step()
print("MS", min(s)); print("SUM", g.sum().item())
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
    let get = |pre: &str| {
        stdout
            .lines()
            .find_map(|l| l.strip_prefix(pre))
            .and_then(|v| v.trim().parse::<f64>().ok())
    };
    Some((get("MS ")?, get("SUM ")?))
}

fn main() -> Result<(), Box<dyn Error>> {
    for (batch, n) in [
        (2000usize, 64usize),
        (800usize, 96usize),
        (400usize, 128usize),
    ] {
        let (ft_ms, ft_sum) = run_ft(batch, n)?;
        print!("B={batch} n={n}: FT {ft_ms:.1} ms gradsum {ft_sum:.6e}");
        if let Some((tms, tsum)) = run_pytorch(batch, n) {
            let rel = (ft_sum - tsum).abs() / (tsum.abs() + 1e-12);
            let ratio = tms / ft_ms;
            let tag = if ratio >= 1.0 { "FASTER" } else { "SLOWER" };
            println!(
                " | PyTorch {tms:.1} ms gradsum {tsum:.6e} rel {rel:.3e} | FT {ratio:.2}x {tag}"
            );
        } else {
            println!(" | PyTorch unavailable");
        }
    }
    Ok(())
}

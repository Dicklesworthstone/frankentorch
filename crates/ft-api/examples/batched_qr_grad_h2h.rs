use std::error::Error;
use std::io::Write;
use std::process::{Command, Stdio};
use std::time::Instant;

use ft_api::FrankenTorchSession;
use ft_core::ExecutionMode;

fn boxed<E: std::fmt::Debug>(err: E) -> std::io::Error {
    std::io::Error::other(format!("{err:?}"))
}

// Tall/square (m>=n) planes, full-rank R.
fn fill_input(batch: usize, m: usize, n: usize) -> Vec<f64> {
    let mut a = vec![0.0_f64; batch * m * n];
    for plane in 0..batch {
        let base = plane * m * n;
        for r in 0..m {
            for c in 0..n {
                let v = ((((plane + 1) * (r + 2) * (c + 3)) % 19) as f64 - 9.0) * 0.02;
                a[base + r * n + c] = v + if r == c { 3.0 + c as f64 } else { 0.0 };
            }
        }
    }
    a
}

fn run_ft(batch: usize, m: usize, n: usize) -> Result<(f64, f64), Box<dyn Error>> {
    let mut best = f64::INFINITY;
    let mut checksum = 0.0_f64;
    for _ in 0..5 {
        let data = fill_input(batch, m, n);
        let mut s = FrankenTorchSession::new(ExecutionMode::Strict);
        let a = s
            .tensor_variable(data, vec![batch, m, n], true)
            .map_err(boxed)?;
        let start = Instant::now();
        let (q, r) = s.tensor_linalg_qr(a, true).map_err(boxed)?;
        let sq = s.tensor_sum(q).map_err(boxed)?;
        let sr = s.tensor_sum(r).map_err(boxed)?;
        let loss = s.tensor_add(sq, sr).map_err(boxed)?;
        s.tensor_backward(loss).map_err(boxed)?;
        let elapsed_ms = start.elapsed().as_secs_f64() * 1e3;
        if elapsed_ms < best {
            best = elapsed_ms;
            let g = s.tensor_grad(a).map_err(boxed)?.unwrap_or_default();
            checksum = g.iter().sum();
        }
    }
    Ok((best, checksum))
}

fn run_pytorch(batch: usize, m: usize, n: usize) -> Option<(f64, f64)> {
    let python = std::env::var("PYTORCH_PYTHON").unwrap_or_else(|_| "python3".to_string());
    let script = format!(
        r#"
import time, torch
torch.set_num_threads(8)
torch.set_num_interop_threads(8)
B, M, N = {batch}, {m}, {n}
plane = torch.arange(B, dtype=torch.int64).reshape(B,1,1)
rows = torch.arange(M, dtype=torch.int64).reshape(1,M,1)
cols = torch.arange(N, dtype=torch.int64).reshape(1,1,N)
v = ((((plane+1)*(rows+2)*(cols+3))%19).to(torch.float64)-9.0)*0.02
eye = torch.zeros(M, N, dtype=torch.float64)
for c in range(N):
    eye[c, c] = 3.0 + c
base = v + eye.unsqueeze(0)
def step():
    A = base.clone().requires_grad_(True)
    Q, R = torch.linalg.qr(A, mode='reduced')
    (Q.sum() + R.sum()).backward()
    return A.grad
for _ in range(2): step()
samples = []
for _ in range(5):
    t = time.perf_counter(); step(); samples.append((time.perf_counter()-t)*1e3)
g = step()
print("MS", min(samples))
print("SUM", g.sum().item())
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
    let get = |prefix: &str| {
        stdout
            .lines()
            .find_map(|line| line.strip_prefix(prefix))
            .and_then(|value| value.trim().parse::<f64>().ok())
    };
    Some((get("MS ")?, get("SUM ")?))
}

fn main() -> Result<(), Box<dyn Error>> {
    for (batch, m, n) in [
        (50_000usize, 4usize, 4usize),
        (20_000usize, 8usize, 8usize),
        (8_000usize, 16usize, 16usize),
        (3_000usize, 32usize, 32usize),
    ] {
        let (ft_ms, ft_sum) = run_ft(batch, m, n)?;
        print!("B={batch} m={m} n={n}: FT {ft_ms:.3} ms gradsum {ft_sum:.6e}");
        if let Some((torch_ms, torch_sum)) = run_pytorch(batch, m, n) {
            let rel = (ft_sum - torch_sum).abs() / (torch_sum.abs() + 1e-12);
            let ratio = torch_ms / ft_ms;
            let tag = if ratio >= 1.0 { "FASTER" } else { "SLOWER" };
            println!(
                " | PyTorch {torch_ms:.3} ms gradsum {torch_sum:.6e} rel {rel:.3e} | FT {ratio:.2}x {tag}"
            );
        } else {
            println!(" | PyTorch unavailable");
        }
    }
    Ok(())
}

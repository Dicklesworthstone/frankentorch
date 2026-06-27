use std::error::Error;
use std::io::Write;
use std::process::{Command, Stdio};
use std::time::Instant;

use ft_api::FrankenTorchSession;
use ft_core::ExecutionMode;

fn boxed<E: std::fmt::Debug>(err: E) -> std::io::Error {
    std::io::Error::other(format!("{err:?}"))
}

fn fill(batch: usize, m: usize, n: usize, nrhs: usize) -> (Vec<f64>, Vec<f64>) {
    let mut a = vec![0.0_f64; batch * m * n];
    let mut b = vec![0.0_f64; batch * m * nrhs];
    for plane in 0..batch {
        for r in 0..m {
            for c in 0..n {
                let noise = ((((plane + 1) * (r + 3) * (c + 5)) % 23) as f64 - 11.0) * 0.01;
                a[plane * m * n + r * n + c] = noise
                    + if r == c {
                        2.0 + (plane % 7) as f64 * 0.001
                    } else {
                        0.0
                    };
            }
            for c in 0..nrhs {
                b[plane * m * nrhs + r * nrhs + c] =
                    ((((plane + 7) * (r + 2) * (c + 11)) % 31) as f64 - 15.0) * 0.01;
            }
        }
    }
    (a, b)
}

fn run_ft(batch: usize, m: usize, n: usize, nrhs: usize) -> Result<(f64, f64), Box<dyn Error>> {
    let mut best = f64::INFINITY;
    let mut checksum = 0.0_f64;
    for _ in 0..5 {
        let (ad, bd) = fill(batch, m, n, nrhs);
        let mut s = FrankenTorchSession::new(ExecutionMode::Strict);
        let a = s
            .tensor_variable(ad, vec![batch, m, n], true)
            .map_err(boxed)?;
        let b = s
            .tensor_variable(bd, vec![batch, m, nrhs], true)
            .map_err(boxed)?;
        let start = Instant::now();
        let x = s.tensor_linalg_lstsq(a, b).map_err(boxed)?;
        let sq = s.tensor_mul(x, x).map_err(boxed)?;
        let loss = s.tensor_sum(sq).map_err(boxed)?;
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

fn run_pytorch(batch: usize, m: usize, n: usize, nrhs: usize) -> Option<(f64, f64)> {
    let python = std::env::var("PYTORCH_PYTHON").unwrap_or_else(|_| "python3".to_string());
    let script = format!(
        r#"
import time, torch
torch.set_num_threads(8)
torch.set_num_interop_threads(8)
B, M, N, R = {batch}, {m}, {n}, {nrhs}
p = torch.arange(B, dtype=torch.int64).reshape(B,1,1)
rows = torch.arange(M, dtype=torch.int64).reshape(1,M,1)
cols = torch.arange(N, dtype=torch.int64).reshape(1,1,N)
A = ((((p+1)*(rows+3)*(cols+5))%23).to(torch.float64)-11.0)*0.01
A = A + torch.eye(M,N,dtype=torch.float64).unsqueeze(0)*(2.0+(p%7).to(torch.float64)*0.001)
rc = torch.arange(R, dtype=torch.int64).reshape(1,1,R)
Bm = ((((p+7)*(rows+2)*(rc+11))%31).to(torch.float64)-15.0)*0.01
def step():
    Ar=A.clone().requires_grad_(True)
    X=torch.linalg.lstsq(Ar,Bm).solution
    (X*X).sum().backward()
    return Ar.grad
for _ in range(2): step()
s=[]
for _ in range(5):
    t=time.perf_counter(); step(); s.append((time.perf_counter()-t)*1e3)
g=step()
print("MS", min(s))
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
    for (batch, m, n, nrhs) in [
        (20_000usize, 8usize, 4usize, 2usize),
        (8_000usize, 16usize, 8usize, 2usize),
        (3_000usize, 32usize, 16usize, 2usize),
        // underdetermined m<n (min-norm via pinv wide branch)
        (20_000usize, 4usize, 8usize, 2usize),
        (8_000usize, 8usize, 16usize, 2usize),
        (3_000usize, 16usize, 32usize, 2usize),
    ] {
        let (ft_ms, ft_sum) = run_ft(batch, m, n, nrhs)?;
        print!("B={batch} m={m} n={n} rhs={nrhs}: FT {ft_ms:.3} ms gradsum {ft_sum:.6e}");
        if let Some((torch_ms, torch_sum)) = run_pytorch(batch, m, n, nrhs) {
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

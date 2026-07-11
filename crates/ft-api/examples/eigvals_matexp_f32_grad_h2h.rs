use std::error::Error;
use std::io::Write;
use std::process::{Command, Stdio};
use std::time::Instant;

use ft_api::FrankenTorchSession;
use ft_core::ExecutionMode;

fn boxed<E: std::fmt::Debug>(err: E) -> std::io::Error {
    std::io::Error::other(format!("{err:?}"))
}

fn fill_eigvals(batch: usize, k: usize) -> Vec<f32> {
    let mut a = vec![0.0_f32; batch * k * k];
    for plane in 0..batch {
        for r in 0..k {
            for c in 0..k {
                let off = ((((plane + 1) * (r + 2) * (c + 5)) % 11) as f32 - 5.0) * 0.003;
                a[plane * k * k + r * k + c] = off;
            }
            a[plane * k * k + r * k + r] = 5.0 + 3.0 * r as f32 + plane as f32 * 0.05;
        }
    }
    a
}

fn fill_matexp(batch: usize, n: usize) -> Vec<f32> {
    let mut a = vec![0.0_f32; batch * n * n];
    for plane in 0..batch {
        for r in 0..n {
            for c in 0..n {
                a[plane * n * n + r * n + c] =
                    ((((plane + 1) * (r + 2) * (c + 3)) % 13) as f32 - 6.0) * 0.05;
            }
        }
    }
    a
}

fn run_ft(op: &str, batch: usize, n: usize) -> Result<(f64, f64), Box<dyn Error>> {
    let mut best = f64::INFINITY;
    let mut checksum = 0.0_f64;
    for _ in 0..5 {
        let data = if op == "eigvals" {
            fill_eigvals(batch, n)
        } else {
            fill_matexp(batch, n)
        };
        let mut s = FrankenTorchSession::new(ExecutionMode::Strict);
        let a = s
            .tensor_variable_f32(data, vec![batch, n, n], true)
            .map_err(boxed)?;
        let start = Instant::now();
        let out = if op == "eigvals" {
            s.tensor_linalg_eigvals(a).map_err(boxed)?
        } else {
            s.tensor_matrix_exp(a).map_err(boxed)?
        };
        let sq = s.tensor_mul(out, out).map_err(boxed)?;
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

fn run_pytorch(op: &str, batch: usize, n: usize) -> Option<(f64, f64)> {
    let python = std::env::var("PYTORCH_PYTHON").unwrap_or_else(|_| "python3".to_string());
    let setup = if op == "eigvals" {
        "off=((((p+1)*(rows+2)*(cols+5))%11).to(torch.float32)-5.0)*0.003\nA=off-torch.diag_embed(torch.diagonal(off,dim1=-2,dim2=-1))+torch.diag_embed(5.0+3.0*torch.arange(N,dtype=torch.float32).reshape(1,N)+p.reshape(B,1).to(torch.float32)*0.05)\ndef op(Ar):\n  ev=torch.linalg.eigvals(Ar)\n  return (ev.real**2+ev.imag**2)"
    } else {
        "A=((((p+1)*(rows+2)*(cols+3))%13).to(torch.float32)-6.0)*0.05\ndef op(Ar):\n  Y=torch.linalg.matrix_exp(Ar)\n  return Y*Y"
    };
    let script = format!(
        r#"
import time, torch
torch.set_num_threads(8); torch.set_num_interop_threads(8)
B, N = {batch}, {n}
p=torch.arange(B,dtype=torch.int64).reshape(B,1,1)
rows=torch.arange(N,dtype=torch.int64).reshape(1,N,1)
cols=torch.arange(N,dtype=torch.int64).reshape(1,1,N)
{setup}
def step():
    Ar=A.clone().requires_grad_(True)
    op(Ar).sum().backward()
    return Ar.grad
for _ in range(2): step()
s=[]
for _ in range(5):
    t=time.perf_counter(); step(); s.append((time.perf_counter()-t)*1e3)
g=step()
print("MS", min(s)); print("SUM", float(g.double().sum().item()))
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
    for op in ["eigvals", "matrix_exp"] {
        for (batch, n) in [
            (20_000usize, 4usize),
            (8_000usize, 8usize),
            (3_000usize, 16usize),
            (1_000usize, 32usize),
        ] {
            let (ft_ms, ft_sum) = run_ft(op, batch, n)?;
            print!("op={op} B={batch} n={n}: FT {ft_ms:.3} ms gradsum {ft_sum:.6e}");
            if let Some((tms, tsum)) = run_pytorch(op, batch, n) {
                let rel = (ft_sum - tsum).abs() / (tsum.abs() + 1e-6);
                let ratio = tms / ft_ms;
                let tag = if ratio >= 1.0 { "FASTER" } else { "SLOWER" };
                println!(
                    " | PyTorch {tms:.3} ms gradsum {tsum:.6e} rel {rel:.3e} | FT {ratio:.2}x {tag}"
                );
            } else {
                println!(" | PyTorch unavailable");
            }
        }
    }
    Ok(())
}

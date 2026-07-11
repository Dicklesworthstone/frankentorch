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
                let v = ((((plane + 1) * (r + 2) * (c + 3)) % 19) as f64 - 9.0) * 0.02;
                a[plane * n * n + r * n + c] = v + if r == c { 3.0 } else { 0.0 };
            }
        }
    }
    a
}

fn run_ft(batch: usize, n: usize) -> Result<f64, Box<dyn Error>> {
    let mut best = f64::INFINITY;
    for _ in 0..3 {
        let data = fill(batch, n);
        let mut s = FrankenTorchSession::new(ExecutionMode::Strict);
        let a = s
            .tensor_variable(data, vec![batch, n, n], true)
            .map_err(boxed)?;
        let start = Instant::now();
        let (u, sg, vh) = s.tensor_linalg_svd(a, false).map_err(boxed)?;
        let su = s.tensor_sum(u).map_err(boxed)?;
        let ss = s.tensor_sum(sg).map_err(boxed)?;
        let sv = s.tensor_sum(vh).map_err(boxed)?;
        let t = s.tensor_add(su, ss).map_err(boxed)?;
        let loss = s.tensor_add(t, sv).map_err(boxed)?;
        s.tensor_backward(loss).map_err(boxed)?;
        let elapsed_ms = start.elapsed().as_secs_f64() * 1e3;
        if elapsed_ms < best {
            best = elapsed_ms;
        }
    }
    Ok(best)
}

fn run_pytorch(batch: usize, n: usize) -> Option<f64> {
    let python = std::env::var("PYTORCH_PYTHON").unwrap_or_else(|_| "python3".to_string());
    let script = format!(
        r#"
import time, torch
torch.set_num_threads(8); torch.set_num_interop_threads(8)
B, N = {batch},{n}
p=torch.arange(B,dtype=torch.int64).reshape(B,1,1)
rows=torch.arange(N,dtype=torch.int64).reshape(1,N,1)
cols=torch.arange(N,dtype=torch.int64).reshape(1,1,N)
A=((((p+1)*(rows+2)*(cols+3))%19).to(torch.float64)-9.0)*0.02+torch.eye(N,dtype=torch.float64).unsqueeze(0)*3.0
def step():
    Ar=A.clone().requires_grad_(True); U,S,Vh=torch.linalg.svd(Ar,full_matrices=False); (U.sum()+S.sum()+Vh.sum()).backward(); return Ar.grad
step()
s=[]
for _ in range(3):
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
    // Smaller batches so torch (whose batched svd backward is pathologically slow) completes.
    for (batch, n) in [
        (400usize, 64usize),
        (200usize, 96usize),
        (100usize, 128usize),
    ] {
        let ft_ms = run_ft(batch, n)?;
        print!("B={batch} n={n}: FT {ft_ms:.1} ms");
        if let Some(tms) = run_pytorch(batch, n) {
            let ratio = tms / ft_ms;
            let tag = if ratio >= 1.0 { "FASTER" } else { "SLOWER" };
            println!(" | PyTorch {tms:.1} ms | FT {ratio:.2}x {tag}");
        } else {
            println!(" | PyTorch unavailable");
        }
    }
    Ok(())
}

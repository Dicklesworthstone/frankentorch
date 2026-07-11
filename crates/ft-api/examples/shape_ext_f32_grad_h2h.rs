use std::error::Error;
use std::io::Write;
use std::process::{Command, Stdio};
use std::time::Instant;

use ft_api::FrankenTorchSession;
use ft_core::ExecutionMode;

fn boxed<E: std::fmt::Debug>(err: E) -> std::io::Error {
    std::io::Error::other(format!("{err:?}"))
}

// op: "svd_wide" (m<n svd, loss sum S), "pinv_wide" (m<n pinv, loss sum sq), "lstsq_under" (m<n lstsq)
fn fill(batch: usize, m: usize, n: usize) -> Vec<f32> {
    let mut a = vec![0.0_f32; batch * m * n];
    for plane in 0..batch {
        for r in 0..m {
            for c in 0..n {
                let v = ((((plane + 1) * (r + 2) * (c + 3)) % 17) as f32 - 8.0) * 0.02;
                a[plane * m * n + r * n + c] = v + if r == c { 3.0 + r as f32 } else { 0.0 };
            }
        }
    }
    a
}

fn run_ft(op: &str, batch: usize, m: usize, n: usize) -> Result<f64, Box<dyn Error>> {
    let mut best = f64::INFINITY;
    for _ in 0..5 {
        let data = fill(batch, m, n);
        let mut s = FrankenTorchSession::new(ExecutionMode::Strict);
        let a = s
            .tensor_variable_f32(data, vec![batch, m, n], true)
            .map_err(boxed)?;
        let start = Instant::now();
        let loss = match op {
            "svd_wide" => {
                let (_u, sg, _vh) = s.tensor_linalg_svd(a, false).map_err(boxed)?;
                s.tensor_sum(sg).map_err(boxed)?
            }
            "pinv_wide" => {
                let y = s.tensor_linalg_pinv(a).map_err(boxed)?;
                let sq = s.tensor_mul(y, y).map_err(boxed)?;
                s.tensor_sum(sq).map_err(boxed)?
            }
            _ => {
                let bdata: Vec<f32> = (0..batch * m * 2)
                    .map(|i| ((i % 13) as f32) * 0.01 - 0.06)
                    .collect();
                let b = s
                    .tensor_variable_f32(bdata, vec![batch, m, 2], true)
                    .map_err(boxed)?;
                let x = s.tensor_linalg_lstsq(a, b).map_err(boxed)?;
                let sq = s.tensor_mul(x, x).map_err(boxed)?;
                s.tensor_sum(sq).map_err(boxed)?
            }
        };
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
    let body = match op {
        "svd_wide" => "U,S,Vh=torch.linalg.svd(Ar,full_matrices=False); return S.sum()",
        "pinv_wide" => "Y=torch.linalg.pinv(Ar); return (Y*Y).sum()",
        _ => "X=torch.linalg.lstsq(Ar,Bm).solution; return (X*X).sum()",
    };
    let script = format!(
        r#"
import time, torch
torch.set_num_threads(8); torch.set_num_interop_threads(8)
B,M,N,op = {batch},{m},{n},"{op}"
p=torch.arange(B,dtype=torch.int64).reshape(B,1,1)
rows=torch.arange(M,dtype=torch.int64).reshape(1,M,1)
cols=torch.arange(N,dtype=torch.int64).reshape(1,1,N)
A=((((p+1)*(rows+2)*(cols+3))%17).to(torch.float32)-8.0)*0.02
A=A+torch.eye(M,N,dtype=torch.float32).unsqueeze(0)*(3.0+rows.to(torch.float32))
Bm=torch.tensor([((i%13)*0.01-0.06) for i in range(B*M*2)],dtype=torch.float32).reshape(B,M,2)
def fwd(Ar):
    {body}
def step():
    Ar=A.clone().requires_grad_(True); fwd(Ar).backward(); return Ar.grad
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
    for op in ["svd_wide", "pinv_wide", "lstsq_under"] {
        for (batch, m, n) in [
            (20_000usize, 4usize, 8usize),
            (8_000usize, 8usize, 16usize),
            (3_000usize, 16usize, 32usize),
        ] {
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
    }
    Ok(())
}

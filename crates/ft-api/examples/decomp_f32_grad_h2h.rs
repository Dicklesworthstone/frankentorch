use std::error::Error;
use std::io::Write;
use std::process::{Command, Stdio};
use std::time::Instant;

use ft_api::FrankenTorchSession;
use ft_core::ExecutionMode;

fn boxed<E: std::fmt::Debug>(err: E) -> std::io::Error {
    std::io::Error::other(format!("{err:?}"))
}

// op: eigh (square sym), svd/qr (tall m>=n), lstsq (overdetermined)
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
    if op == "eigh" {
        // symmetrize
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

fn run_ft(op: &str, batch: usize, m: usize, n: usize) -> Result<(f64, f64), Box<dyn Error>> {
    let mut best = f64::INFINITY;
    let mut checksum = 0.0_f64;
    for _ in 0..5 {
        let data = fill(op, batch, m, n);
        let mut s = FrankenTorchSession::new(ExecutionMode::Strict);
        let a = s
            .tensor_variable_f32(data, vec![batch, m, n], true)
            .map_err(boxed)?;
        let start = Instant::now();
        let loss = match op {
            "eigh" => {
                let (w, v) = s.tensor_linalg_eigh(a).map_err(boxed)?;
                let sw = s.tensor_sum(w).map_err(boxed)?;
                let sv = s.tensor_sum(v).map_err(boxed)?;
                s.tensor_add(sw, sv).map_err(boxed)?
            }
            "svd" => {
                let (u, sg, vh) = s.tensor_linalg_svd(a, false).map_err(boxed)?;
                let su = s.tensor_sum(u).map_err(boxed)?;
                let ss = s.tensor_sum(sg).map_err(boxed)?;
                let sv = s.tensor_sum(vh).map_err(boxed)?;
                let t = s.tensor_add(su, ss).map_err(boxed)?;
                s.tensor_add(t, sv).map_err(boxed)?
            }
            "qr" => {
                let (q, r) = s.tensor_linalg_qr(a, true).map_err(boxed)?;
                let sq = s.tensor_sum(q).map_err(boxed)?;
                let sr = s.tensor_sum(r).map_err(boxed)?;
                s.tensor_add(sq, sr).map_err(boxed)?
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

fn run_pytorch(op: &str, batch: usize, m: usize, n: usize) -> Option<f64> {
    let python = std::env::var("PYTORCH_PYTHON").unwrap_or_else(|_| "python3".to_string());
    let body = match op {
        "eigh" => "w,V=torch.linalg.eigh(Ar); return w.sum()+V.sum()",
        "svd" => "U,S,Vh=torch.linalg.svd(Ar,full_matrices=False); return U.sum()+S.sum()+Vh.sum()",
        "qr" => "Q,R=torch.linalg.qr(Ar,mode='reduced'); return Q.sum()+R.sum()",
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
A=((((p+1)*(rows+2)*(cols+3))%19).to(torch.float32)-9.0)*0.02
A=A+torch.eye(M,N,dtype=torch.float32).unsqueeze(0)*(3.0+cols.to(torch.float32))
if op=="eigh":
    A=0.5*(A+A.transpose(-1,-2))
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
    // (op, m, n): eigh/svd/qr/lstsq
    for (op, m, n) in [
        ("eigh", 8, 8),
        ("eigh", 16, 16),
        ("eigh", 32, 32),
        ("svd", 8, 4),
        ("svd", 16, 8),
        ("svd", 32, 16),
        ("qr", 8, 4),
        ("qr", 16, 8),
        ("qr", 32, 16),
        ("lstsq", 8, 4),
        ("lstsq", 16, 8),
        ("lstsq", 32, 16),
    ] {
        let batch = match m {
            8 => 20_000usize,
            16 => 8_000,
            _ => 3_000,
        };
        let (ft_ms, _ft_sum) = run_ft(op, batch, m, n)?;
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

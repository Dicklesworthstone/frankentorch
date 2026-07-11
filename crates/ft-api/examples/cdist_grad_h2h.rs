use std::error::Error;
use std::io::Write;
use std::process::{Command, Stdio};
use std::time::Instant;

use ft_api::FrankenTorchSession;
use ft_core::ExecutionMode;

fn boxed<E: std::fmt::Debug>(err: E) -> std::io::Error {
    std::io::Error::other(format!("{err:?}"))
}

fn fill(n: usize, d: usize, seed: usize) -> Vec<f64> {
    (0..n * d)
        .map(|i| (((i * 1103515245 + seed * 12345) % 1000) as f64 / 1000.0) - 0.5)
        .collect()
}

fn run_ft(p: usize, d: usize, p_dim: usize, r_dim: usize) -> Result<(f64, f64), Box<dyn Error>> {
    let mut best = f64::INFINITY;
    let mut checksum = 0.0_f64;
    for _ in 0..3 {
        let x1d = fill(p_dim, d, 1);
        let x2d = fill(r_dim, d, 2);
        let mut s = FrankenTorchSession::new(ExecutionMode::Strict);
        let x1 = s
            .tensor_variable(x1d, vec![p_dim, d], true)
            .map_err(boxed)?;
        let x2 = s
            .tensor_variable(x2d, vec![r_dim, d], false)
            .map_err(boxed)?;
        let start = Instant::now();
        let dist = s.tensor_cdist(x1, x2, p as f64).map_err(boxed)?;
        let loss = s.tensor_sum(dist).map_err(boxed)?;
        s.tensor_backward(loss).map_err(boxed)?;
        let elapsed_ms = start.elapsed().as_secs_f64() * 1e3;
        if elapsed_ms < best {
            best = elapsed_ms;
            let g = s.tensor_grad(x1).map_err(boxed)?.unwrap_or_default();
            checksum = g.iter().sum();
        }
    }
    Ok((best, checksum))
}

fn run_pytorch(p: usize, d: usize, p_dim: usize, r_dim: usize) -> Option<(f64, f64)> {
    let python = std::env::var("PYTORCH_PYTHON").unwrap_or_else(|_| "python3".to_string());
    let script = format!(
        r#"
import time, torch
torch.set_num_threads(8); torch.set_num_interop_threads(8)
P,R,D,p = {p_dim},{r_dim},{d},{p}
def fill(n,seed): return torch.tensor([(((i*1103515245+seed*12345)%1000)/1000.0)-0.5 for i in range(n*D)],dtype=torch.float64).reshape(n,D)
x1=fill(P,1); x2=fill(R,2)
def step():
    xr=x1.clone().requires_grad_(True)
    torch.cdist(xr,x2,p=float(p)).sum().backward()
    return xr.grad
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
    for p in [1usize, 2usize] {
        for (pd, rd, d) in [(1500usize, 1500usize, 8usize), (1000, 1000, 16usize)] {
            let (ft_ms, ft_sum) = run_ft(p, d, pd, rd)?;
            print!("p={p} P={pd} R={rd} D={d}: FT {ft_ms:.3} ms gradsum {ft_sum:.6e}");
            if let Some((tms, tsum)) = run_pytorch(p, d, pd, rd) {
                let rel = (ft_sum - tsum).abs() / (tsum.abs() + 1e-12);
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

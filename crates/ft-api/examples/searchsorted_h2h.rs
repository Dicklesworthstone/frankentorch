use std::error::Error;
use std::io::Write;
use std::process::{Command, Stdio};
use std::time::Instant;

use ft_api::FrankenTorchSession;
use ft_core::ExecutionMode;

fn boxed<E: std::fmt::Debug>(err: E) -> std::io::Error {
    std::io::Error::other(format!("{err:?}"))
}

fn run_ft(seq_len: usize, n_q: usize) -> Result<(f64, f64), Box<dyn Error>> {
    let seq: Vec<f64> = (0..seq_len).map(|i| i as f64 / seq_len as f64).collect();
    let q: Vec<f64> = (0..n_q)
        .map(|i| (((i * 2654435761usize) % 100000) as f64) / 100000.0)
        .collect();
    let mut best = f64::INFINITY;
    let mut checksum = 0.0_f64;
    for _ in 0..5 {
        let mut s = FrankenTorchSession::new(ExecutionMode::Strict);
        let seq_t = s
            .tensor_variable(seq.clone(), vec![seq_len], false)
            .map_err(boxed)?;
        let q_t = s
            .tensor_variable(q.clone(), vec![n_q], false)
            .map_err(boxed)?;
        let start = Instant::now();
        let out = s.tensor_searchsorted(seq_t, q_t, false).map_err(boxed)?;
        let elapsed_ms = start.elapsed().as_secs_f64() * 1e3;
        if elapsed_ms < best {
            best = elapsed_ms;
            checksum = s.tensor_values(out).map_err(boxed)?.iter().sum();
        }
    }
    Ok((best, checksum))
}

fn run_pytorch(seq_len: usize, n_q: usize) -> Option<(f64, f64)> {
    let python = std::env::var("PYTORCH_PYTHON").unwrap_or_else(|_| "python3".to_string());
    let script = format!(
        r#"
import time, torch
torch.set_num_threads(8); torch.set_num_interop_threads(8)
S,Q = {seq_len},{n_q}
seq=torch.arange(S,dtype=torch.float64)/S
q=torch.tensor([(((i*2654435761)%100000)/100000.0) for i in range(Q)],dtype=torch.float64)
def step(): return torch.searchsorted(seq,q,right=False)
for _ in range(2): step()
s=[]
for _ in range(5):
    t=time.perf_counter(); step(); s.append((time.perf_counter()-t)*1e3)
print("MS", min(s)); print("SUM", float(step().sum().item()))
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
    for (sl, nq) in [
        (10_000usize, 50_000_000usize),
        (1_000, 20_000_000),
        (100_000, 10_000_000),
    ] {
        let (ft_ms, ft_sum) = run_ft(sl, nq)?;
        print!("seq={sl} nq={nq}: FT {ft_ms:.3} ms sum {ft_sum:.6e}");
        if let Some((tms, tsum)) = run_pytorch(sl, nq) {
            let rel = (ft_sum - tsum).abs() / (tsum.abs() + 1e-12);
            let ratio = tms / ft_ms;
            let tag = if ratio >= 1.0 { "FASTER" } else { "SLOWER" };
            println!(" | PyTorch {tms:.3} ms sum {tsum:.6e} rel {rel:.3e} | FT {ratio:.2}x {tag}");
        } else {
            println!(" | PyTorch unavailable");
        }
    }
    Ok(())
}

use ft_api::FrankenTorchSession;
use ft_core::ExecutionMode;
use std::process::Command;
use std::time::Instant;
fn main() {
    for (bb, k) in [(100000usize, 4usize), (50000, 8), (20000, 16)] {
        let a: Vec<f64> = (0..bb * k * k)
            .map(|x| (((x * 2654435761usize) % 9973) as f64) * 0.001 - 5.0)
            .collect();
        let b: Vec<f64> = (0..bb * k * k)
            .map(|x| (((x * 40503usize) % 9973) as f64) * 0.001 - 5.0)
            .collect();
        let mut best = f64::INFINITY;
        let mut chk = 0.0;
        for _ in 0..6 {
            let mut s = FrankenTorchSession::new(ExecutionMode::Strict);
            let x = s.tensor_variable(a.clone(), vec![bb, k, k], false).unwrap();
            let y = s.tensor_variable(b.clone(), vec![bb, k, k], false).unwrap();
            let t = Instant::now();
            let z = s.tensor_matmul(x, y).unwrap();
            let e = t.elapsed().as_secs_f64() * 1e3;
            if e < best {
                best = e;
                chk = s
                    .tensor_values(z)
                    .unwrap()
                    .iter()
                    .step_by(997)
                    .map(|v| v.abs())
                    .sum();
            }
        }
        let py = format!(
            r#"
import time, torch
torch.set_num_threads(8); B,k={bb},{k}
ia=torch.arange(B*k*k,dtype=torch.float64); A=((ia*2654435761)%9973).double().mul(0.001).sub(5.0).reshape(B,k,k)
ib=torch.arange(B*k*k,dtype=torch.float64); Bm=((ib*40503)%9973).double().mul(0.001).sub(5.0).reshape(B,k,k)
for _ in range(2): torch.matmul(A,Bm)
ts=[]
for _ in range(6): t=time.perf_counter(); torch.matmul(A,Bm); ts.append((time.perf_counter()-t)*1e3)
r=torch.matmul(A,Bm); print("MS",sorted(ts)[0]); print("CHK", r.reshape(-1)[::997].abs().sum().item())
"#
        );
        let python = std::env::var("PYTORCH_PYTHON").unwrap_or("python3".into());
        print!("matmul k={k}: FT {best:.1}ms chk {chk:.4e}",);
        let Ok(o) = Command::new(&python).arg("-c").arg(&py).output() else {
            continue;
        };
        if !o.status.success() {
            continue;
        }
        let s = String::from_utf8_lossy(&o.stdout);
        let g = |p: &str| {
            s.lines()
                .find_map(|l| l.strip_prefix(p).and_then(|v| v.trim().parse::<f64>().ok()))
        };
        if let (Some(p), Some(pc)) = (g("MS "), g("CHK ")) {
            let rel = (chk - pc).abs() / (pc.abs() + 1e-9);
            let rr = p / best;
            println!(
                " | torch {p:.1}ms chk {pc:.4e} | {} | {}",
                if rel < 1e-9 { "MATCH" } else { "DIFF" },
                if rr >= 1.0 {
                    format!("FT {rr:.2}x FASTER")
                } else {
                    format!("FT {:.2}x slower", 1.0 / rr)
                }
            );
        }
    }
}

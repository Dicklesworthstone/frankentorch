use ft_api::FrankenTorchSession;
use ft_core::ExecutionMode;
use std::process::Command;
use std::time::Instant;
fn main() {
    for (bb, k) in [(100000usize, 4usize), (20000, 16), (4000, 32)] {
        let data: Vec<f64> = (0..bb * k * k)
            .map(|x| (((x * 2654435761usize) % 9973) as f64) * 0.001 - 5.0)
            .collect();
        let mut best = f64::INFINITY;
        let mut ssum = 0.0;
        for _ in 0..5 {
            let mut s = FrankenTorchSession::new(ExecutionMode::Strict);
            let x = s
                .tensor_variable(data.clone(), vec![bb, k, k], false)
                .unwrap();
            let t = Instant::now();
            let sv = s.tensor_linalg_svdvals(x).unwrap();
            let el = t.elapsed().as_secs_f64() * 1e3;
            if el < best {
                best = el;
                ssum = s.tensor_values(sv).unwrap().iter().sum();
            }
        }
        let pysrc = format!(
            r#"
import time, torch
torch.set_num_threads(8); B,k={bb},{k}
idx=torch.arange(B*k*k,dtype=torch.float64)
d=(((idx*2654435761)%9973))*0.001 - 5.0
A=d.reshape(B,k,k)
for _ in range(2): torch.linalg.svdvals(A)
ts=[]
for _ in range(5):
    t=time.perf_counter(); torch.linalg.svdvals(A); ts.append((time.perf_counter()-t)*1e3)
s=torch.linalg.svdvals(A); print("MS",sorted(ts)[0]); print("SSUM",s.sum().item())
"#
        );
        let python = std::env::var("PYTORCH_PYTHON").unwrap_or("python3".into());
        print!("B={bb} k={k}: FT {best:.1} ms  ssum {ssum:.5e}");
        let Ok(o) = Command::new(&python).arg("-c").arg(&pysrc).output() else {
            println!();
            continue;
        };
        if !o.status.success() {
            eprintln!("\n{}", String::from_utf8_lossy(&o.stderr));
            continue;
        }
        let s = String::from_utf8_lossy(&o.stdout);
        let g = |p: &str| {
            s.lines()
                .find_map(|l| l.strip_prefix(p).and_then(|v| v.trim().parse::<f64>().ok()))
        };
        if let (Some(p), Some(ps)) = (g("MS "), g("SSUM ")) {
            let rel = (ssum - ps).abs() / (ps.abs() + 1e-9);
            let rr = p / best;
            println!(
                "  | PyTorch {p:.1} ms  ssum {ps:.5e} | {} | {}",
                if rel < 1e-6 { "MATCH" } else { "DIFF" },
                if rr >= 1.0 {
                    format!("FT {rr:.2}x FASTER")
                } else {
                    format!("FT {:.2}x slower", 1.0 / rr)
                }
            );
        }
    }
}

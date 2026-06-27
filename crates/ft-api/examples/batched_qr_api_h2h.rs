use ft_api::FrankenTorchSession;
use ft_core::ExecutionMode;
use std::process::Command;
use std::time::Instant;
fn main() {
    for (bb, k) in [(100000usize, 4usize), (20000, 16), (4000, 32)] {
        let data: Vec<f64> = (0..bb * k * k)
            .map(|x| (((x * 2246822519usize) % 9941) as f64) * 0.002 - 9.0)
            .collect();
        let mut best = f64::INFINITY;
        let mut rsq = 0.0;
        for _ in 0..5 {
            let mut s = FrankenTorchSession::new(ExecutionMode::Strict);
            let x = s
                .tensor_variable(data.clone(), vec![bb, k, k], false)
                .unwrap();
            let t = Instant::now();
            let (_q, r) = s.tensor_linalg_qr(x, true).unwrap();
            let el = t.elapsed().as_secs_f64() * 1e3;
            if el < best {
                best = el;
                rsq = s.tensor_values(r).unwrap().iter().map(|v| v * v).sum();
            }
        }
        let pysrc = format!(
            r#"
import time, torch
torch.set_num_threads(8); B,k={bb},{k}
idx=torch.arange(B*k*k,dtype=torch.float64)
d=(((idx*2246822519)%9941))*0.002 - 9.0
A=d.reshape(B,k,k)
for _ in range(2): torch.linalg.qr(A)
ts=[]
for _ in range(5):
    t=time.perf_counter(); torch.linalg.qr(A); ts.append((time.perf_counter()-t)*1e3)
_,R=torch.linalg.qr(A); print("MS",sorted(ts)[0]); print("RSQ",(R*R).sum().item())
"#
        );
        let python = std::env::var("PYTORCH_PYTHON").unwrap_or("python3".into());
        print!("B={bb} k={k}: FT {best:.1} ms  rsq {rsq:.5e}");
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
        if let (Some(p), Some(pr)) = (g("MS "), g("RSQ ")) {
            let rel = (rsq - pr).abs() / (pr.abs() + 1e-9);
            let rr = p / best;
            println!(
                "  | PyTorch {p:.1} ms  rsq {pr:.5e} | {} | {}",
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

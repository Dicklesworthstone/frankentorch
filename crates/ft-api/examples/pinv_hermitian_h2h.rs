use ft_api::FrankenTorchSession;
use ft_core::ExecutionMode;
use std::process::Command;
use std::time::Instant;
fn main() {
    for (bb, k) in [(100000usize, 4usize), (20000, 16), (4000, 32)] {
        let mut data = vec![0.0f64; bb * k * k];
        for (x, v) in data.iter_mut().enumerate() {
            *v = (((x * 2654435761usize) % 9973) as f64) * 0.001 - 5.0;
        }
        for b in 0..bb {
            for i in 0..k {
                for j in (i + 1)..k {
                    let s = (data[b * k * k + i * k + j] + data[b * k * k + j * k + i]) * 0.5;
                    data[b * k * k + i * k + j] = s;
                    data[b * k * k + j * k + i] = s;
                }
            }
            for i in 0..k {
                data[b * k * k + i * k + i] += (k + 10) as f64;
            }
        }
        let mut best = f64::INFINITY;
        let mut iderr = 0.0;
        for _ in 0..5 {
            let mut s = FrankenTorchSession::new(ExecutionMode::Strict);
            let x = s
                .tensor_variable(data.clone(), vec![bb, k, k], false)
                .unwrap();
            let t = Instant::now();
            let pinv = s.tensor_linalg_pinv_hermitian(x).unwrap();
            let el = t.elapsed().as_secs_f64() * 1e3;
            if el < best {
                best = el;
                let xx = s
                    .tensor_variable(data.clone(), vec![bb, k, k], false)
                    .unwrap();
                let prod = s.tensor_matmul(xx, pinv).unwrap();
                let pv = s.tensor_values(prod).unwrap();
                let mut e = 0.0;
                for b in 0..bb.min(2000) {
                    for i in 0..k {
                        for j in 0..k {
                            let w = if i == j { 1.0 } else { 0.0 };
                            e += (pv[b * k * k + i * k + j] - w).abs();
                        }
                    }
                }
                iderr = e / (bb.min(2000) * k * k) as f64;
            }
        }
        let pysrc = format!(
            r#"
import time, torch
torch.set_num_threads(8); B,k={bb},{k}
idx=torch.arange(B*k*k,dtype=torch.float64)
A=((idx*2654435761)%9973).double().mul(0.001).sub(5.0).reshape(B,k,k); A=(A+A.transpose(-1,-2))*0.5+(k+10)*torch.eye(k,dtype=torch.float64)
for _ in range(2): torch.linalg.pinv(A,hermitian=True)
ts=[]
for _ in range(5): t=time.perf_counter(); torch.linalg.pinv(A,hermitian=True); ts.append((time.perf_counter()-t)*1e3)
print("MS",sorted(ts)[0])
"#
        );
        let python = std::env::var("PYTORCH_PYTHON").unwrap_or("python3".into());
        print!("k={k}: FT {best:.1}ms (A@pinv-I err {iderr:.1e})");
        if let Ok(o) = Command::new(&python).arg("-c").arg(&pysrc).output()
            && o.status.success()
            && let Some(p) = String::from_utf8_lossy(&o.stdout).lines().find_map(|l| {
                l.strip_prefix("MS ")
                    .and_then(|v| v.trim().parse::<f64>().ok())
            })
        {
            let rr = p / best;
            println!(
                " | torch {p:.1}ms | {}",
                if rr >= 1.0 {
                    format!("FT {rr:.2}x FASTER")
                } else {
                    format!("FT {:.2}x slower", 1.0 / rr)
                }
            );
        }
    }
}

use ft_api::FrankenTorchSession;
use ft_core::ExecutionMode;
use std::process::Command;
use std::time::Instant;
fn main() {
    for (bb, k) in [(100000usize, 4usize), (20000, 16)] {
        let mut a = vec![0.0f64; bb * k * k];
        for (x, value) in a.iter_mut().enumerate().take(bb * k * k) {
            *value = (((x * 2654435761usize) % 9973) as f64) * 0.001 - 5.0;
        }
        for b in 0..bb {
            for i in 0..k {
                for j in (i + 1)..k {
                    let s = (a[b * k * k + i * k + j] + a[b * k * k + j * k + i]) * 0.5;
                    a[b * k * k + i * k + j] = s;
                    a[b * k * k + j * k + i] = s;
                }
            }
            for i in 0..k {
                a[b * k * k + i * k + i] += (k + 10) as f64;
            }
        }
        let mut best = f64::INFINITY;
        let mut err = 0.0;
        for _ in 0..5 {
            let mut s = FrankenTorchSession::new(ExecutionMode::Strict);
            let x = s.tensor_variable(a.clone(), vec![bb, k, k], true).unwrap();
            let t = Instant::now();
            let y = s.tensor_linalg_eigvalsh(x).unwrap();
            let sq = s.tensor_mul(y, y).unwrap();
            let loss = s.tensor_sum(sq).unwrap();
            let rep = s.tensor_backward(loss).unwrap();
            let g = s.tensor_gradient(&rep, x).unwrap();
            let el = t.elapsed().as_secs_f64() * 1e3;
            if el < best {
                best = el;
                // grad should be 2A
                let mut e = 0.0;
                for i in 0..bb.min(2000) * k * k {
                    e += (g[i] - 2.0 * a[i]).abs();
                }
                err = e / (bb.min(2000) * k * k) as f64;
            }
        }
        let pysrc = format!(
            r#"
import time, torch
torch.set_num_threads(8); B,k={bb},{k}
idx=torch.arange(B*k*k,dtype=torch.float64)
A0=((idx*2654435761)%9973).double().mul(0.001).sub(5.0).reshape(B,k,k); A0=(A0+A0.transpose(-1,-2))*0.5+(k+10)*torch.eye(k,dtype=torch.float64)
def step():
    A=A0.clone().requires_grad_(True); torch.linalg.eigvalsh(A).pow(2).sum().backward(); return A.grad
for _ in range(2): step()
ts=[]
for _ in range(5): t=time.perf_counter(); step(); ts.append((time.perf_counter()-t)*1e3)
print("MS",sorted(ts)[0])
"#
        );
        let python = std::env::var("PYTORCH_PYTHON").unwrap_or("python3".into());
        print!("k={k}: FT {best:.1}ms (grad-2A err {err:.1e})");
        let Ok(o) = Command::new(&python).arg("-c").arg(&pysrc).output() else {
            continue;
        };
        if !o.status.success() {
            eprintln!("\n{}", String::from_utf8_lossy(&o.stderr));
            continue;
        }
        if let Some(p) = String::from_utf8_lossy(&o.stdout).lines().find_map(|l| {
            l.strip_prefix("MS ")
                .and_then(|v| v.trim().parse::<f64>().ok())
        }) {
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

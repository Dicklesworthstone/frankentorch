use ft_api::FrankenTorchSession;
use ft_core::ExecutionMode;
use std::process::Command;
use std::time::Instant;
fn main() {
    let nrhs = 4usize;
    for (bb, k) in [(100000usize, 4usize), (20000, 16), (4000, 32)] {
        let mut a = vec![0.0f64; bb * k * k];
        for (x, value) in a.iter_mut().enumerate().take(bb * k * k) {
            *value = (((x * 2654435761usize) % 9973) as f64) * 0.001 - 5.0;
        }
        for b in 0..bb {
            for d in 0..k {
                a[b * k * k + d * k + d] += (2 * k) as f64;
            }
        }
        let bm: Vec<f64> = (0..bb * k * nrhs)
            .map(|x| (((x * 40503usize) % 7919) as f64) * 0.01 - 3.0)
            .collect();
        let mut best = f64::INFINITY;
        let mut err = 0.0;
        for _ in 0..5 {
            let mut s = FrankenTorchSession::new(ExecutionMode::Strict);
            let av = s.tensor_variable(a.clone(), vec![bb, k, k], false).unwrap();
            let bv = s
                .tensor_variable(bm.clone(), vec![bb, k, nrhs], false)
                .unwrap();
            let t = Instant::now();
            let x = s.tensor_linalg_lstsq(av, bv).unwrap();
            let el = t.elapsed().as_secs_f64() * 1e3;
            if el < best {
                best = el;
                // A@X vs B
                let av2 = s.tensor_variable(a.clone(), vec![bb, k, k], false).unwrap();
                let ax = s.tensor_matmul(av2, x).unwrap();
                let axv = s.tensor_values(ax).unwrap();
                let mut e = 0.0;
                for i in 0..bb.min(2000) * k * nrhs {
                    e += (axv[i] - bm[i]).abs();
                }
                err = e / (bb.min(2000) * k * nrhs) as f64;
            }
        }
        let pysrc = format!(
            r#"
import time, torch
torch.set_num_threads(8); B,k,nrhs={bb},{k},{nrhs}
ia=torch.arange(B*k*k,dtype=torch.float64); A=((ia*2654435761)%9973).double().mul(0.001).sub(5.0).reshape(B,k,k); A=A+(2*k)*torch.eye(k,dtype=torch.float64)
ib=torch.arange(B*k*nrhs,dtype=torch.float64); Bm=((ib*40503)%7919).double().mul(0.01).sub(3.0).reshape(B,k,nrhs)
for _ in range(2): torch.linalg.lstsq(A,Bm)
ts=[]
for _ in range(5): t=time.perf_counter(); torch.linalg.lstsq(A,Bm); ts.append((time.perf_counter()-t)*1e3)
print("MS",sorted(ts)[0])
"#
        );
        let python = std::env::var("PYTORCH_PYTHON").unwrap_or("python3".into());
        print!("k={k}: FT {best:.1}ms (A@X-B err {err:.1e})");
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

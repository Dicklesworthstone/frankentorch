use ft_api::FrankenTorchSession;
use ft_core::{DType, ExecutionMode};
use std::process::Command;
use std::time::Instant;
fn main() {
    for (bb, k) in [(100000usize, 4usize), (20000, 16)] {
        let mut data = vec![0.0f64; bb * k * k];
        for (i, value) in data.iter_mut().enumerate().take(bb * k * k) {
            *value = (((i * 2654435761usize) % 9973) as f64) * 0.001 - 5.0;
        }
        for b in 0..bb {
            for i in 0..k {
                data[b * k * k + i * k + i] += (k + 10) as f64;
            }
        }
        let run = |op: &str| -> (f64, f64) {
            let mut best = f64::INFINITY;
            let mut chk = 0.0;
            for _ in 0..5 {
                let mut s = FrankenTorchSession::new(ExecutionMode::Strict);
                let xf = s
                    .tensor_variable(data.clone(), vec![bb, k, k], false)
                    .unwrap();
                let x = s.tensor_to_dtype(xf, DType::F32).unwrap();
                let t = Instant::now();
                let y = match op {
                    "nuc" => s.tensor_linalg_matrix_norm(x, "nuc").unwrap(),
                    "cond" => s.tensor_linalg_cond(x, 2.0).unwrap(),
                    _ => s.tensor_linalg_matrix_rank(x, None).unwrap(),
                };
                let e = t.elapsed().as_secs_f64() * 1e3;
                if e < best {
                    best = e;
                    chk = s.tensor_values_lossy_f64(y).unwrap().iter().sum();
                }
            }
            (best, chk)
        };
        for op in ["nuc", "cond", "rank"] {
            let (ms, chk) = run(op);
            let pysrc = format!(
                r#"
import time, torch
torch.set_num_threads(8); B,k={bb},{k}
idx=torch.arange(B*k*k,dtype=torch.float64)
A=((idx*2654435761)%9973).double().mul(0.001).sub(5.0).reshape(B,k,k); A=(A+(k+10)*torch.eye(k,dtype=torch.float64)).float()
op="{op}"
f={{"nuc":lambda:torch.linalg.matrix_norm(A,ord='nuc'),"cond":lambda:torch.linalg.cond(A,2),"rank":lambda:torch.linalg.matrix_rank(A)}}[op]
for _ in range(2): f()
import time as tm; ts=[]
for _ in range(5): s=tm.perf_counter(); f(); ts.append((tm.perf_counter()-s)*1e3)
r=f(); print("MS",sorted(ts)[0]); print("CHK", r.double().sum().item())
"#
            );
            let python = std::env::var("PYTORCH_PYTHON").unwrap_or("python3".into());
            print!("k={k} {op} f32: FT {ms:.1}ms chk {chk:.4e}");
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
            if let (Some(p), Some(pc)) = (g("MS "), g("CHK ")) {
                let rel = (chk - pc).abs() / (pc.abs() + 1e-6);
                let rr = p / ms;
                println!(
                    " | torch {p:.1}ms chk {pc:.4e} | {} | {}",
                    if rel < 1e-3 { "OK" } else { "DIFF" },
                    if rr >= 1.0 {
                        format!("FT {rr:.2}x FASTER")
                    } else {
                        format!("{:.2}x slower", 1.0 / rr)
                    }
                );
            }
        }
    }
}

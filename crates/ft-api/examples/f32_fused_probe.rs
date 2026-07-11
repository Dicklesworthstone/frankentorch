use ft_api::FrankenTorchSession;
use ft_core::{DType, ExecutionMode};
use std::process::Command;
use std::time::Instant;
fn main() {
    for (bb, k) in [(100000usize, 4usize), (20000, 16)] {
        let mut a = vec![0.0f64; bb * k * k];
        for (x, value) in a.iter_mut().enumerate().take(bb * k * k) {
            *value = (((x * 2654435761usize) % 9973) as f64) * 0.001 - 5.0;
        }
        for b in 0..bb {
            for d in 0..k {
                a[b * k * k + d * k + d] += (2 * k) as f64;
            }
        }
        let rhs: Vec<f64> = (0..bb * k * 2)
            .map(|x| (((x * 40503usize) % 7919) as f64) * 0.01 - 3.0)
            .collect();
        let run = |op: &str| -> f64 {
            let mut best = f64::INFINITY;
            for _ in 0..5 {
                let mut s = FrankenTorchSession::new(ExecutionMode::Strict);
                let af = s.tensor_variable(a.clone(), vec![bb, k, k], false).unwrap();
                let x = s.tensor_to_dtype(af, DType::F32).unwrap();
                let t = Instant::now();
                let r = match op {
                    "pinv" => s.tensor_linalg_pinv(x),
                    "hpinv" => s.tensor_linalg_pinv_hermitian(x),
                    _ => {
                        let bf = s
                            .tensor_variable(rhs.clone(), vec![bb, k, 2], false)
                            .unwrap();
                        let bx = s.tensor_to_dtype(bf, DType::F32).unwrap();
                        s.tensor_linalg_lstsq(x, bx)
                    }
                };
                match r {
                    Ok(_) => {}
                    Err(e) => {
                        eprintln!("{op} ERR: {e:?}");
                        return -1.0;
                    }
                }
                let e = t.elapsed().as_secs_f64() * 1e3;
                if e < best {
                    best = e;
                }
            }
            best
        };
        for op in ["pinv", "hpinv", "lstsq"] {
            let ft = run(op);
            if ft < 0.0 {
                println!("k={k} {op} f32: FT ERROR");
                continue;
            }
            let pysrc = format!(
                r#"
import time, torch
torch.set_num_threads(8); B,k={bb},{k}; op="{op}"
ia=torch.arange(B*k*k,dtype=torch.float64); A=(((ia*2654435761)%9973).double().mul(0.001).sub(5.0).reshape(B,k,k)+(2*k)*torch.eye(k,dtype=torch.float64)).float()
Bm=((torch.arange(B*k*2,dtype=torch.float64)*40503)%7919).double().mul(0.01).sub(3.0).reshape(B,k,2).float()
f={{"pinv":lambda:torch.linalg.pinv(A),"hpinv":lambda:torch.linalg.pinv(A+A.transpose(-1,-2),hermitian=True),"lstsq":lambda:torch.linalg.lstsq(A,Bm)}}[op]
for _ in range(2): f()
ts=[]
for _ in range(5): t=time.perf_counter(); f(); ts.append((time.perf_counter()-t)*1e3)
print("MS",sorted(ts)[0])
"#
            );
            let p = Command::new(std::env::var("PYTORCH_PYTHON").unwrap_or("python3".into()))
                .arg("-c")
                .arg(&pysrc)
                .output()
                .ok()
                .and_then(|o| {
                    String::from_utf8_lossy(&o.stdout).lines().find_map(|l| {
                        l.strip_prefix("MS ")
                            .and_then(|v| v.trim().parse::<f64>().ok())
                    })
                });
            print!("k={k} {op} f32: FT {ft:.1}ms");
            if let Some(p) = p {
                println!(
                    " | torch {p:.1}ms | {}",
                    if p / ft >= 1.0 {
                        format!("FT {:.2}x FASTER", p / ft)
                    } else {
                        format!("FT {:.2}x slower", ft / p)
                    }
                );
            } else {
                println!();
            }
        }
    }
}

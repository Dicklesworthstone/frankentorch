use ft_api::FrankenTorchSession;
use ft_core::ExecutionMode;
use std::process::Command;
use std::time::Instant;
fn main() {
    for (bb, k) in [(100000usize, 4usize), (20000, 16)] {
        let data: Vec<f64> = (0..bb * k * k)
            .map(|x| (((x * 2654435761usize) % 9973) as f64) * 0.001 - 5.0)
            .collect();
        for ord in ["nuc", "2"] {
            let mut best = f64::INFINITY;
            let mut chk = 0.0;
            let mut oshape = vec![];
            for _ in 0..5 {
                let mut s = FrankenTorchSession::new(ExecutionMode::Strict);
                let x = s
                    .tensor_variable(data.clone(), vec![bb, k, k], false)
                    .unwrap();
                let t = Instant::now();
                let y = s.tensor_linalg_matrix_norm(x, ord).unwrap();
                let el = t.elapsed().as_secs_f64() * 1e3;
                if el < best {
                    best = el;
                    let v = s.tensor_values(y).unwrap();
                    chk = v.iter().sum();
                    oshape = s.tensor_shape(y).unwrap();
                }
            }
            let pysrc = format!(
                r#"
import time, torch
torch.set_num_threads(8); B,k={bb},{k}
idx=torch.arange(B*k*k,dtype=torch.float64)
A=((idx*2654435761)%9973).double().mul(0.001).sub(5.0).reshape(B,k,k)
o = 'nuc' if "{ord}"=="nuc" else 2
for _ in range(2): torch.linalg.matrix_norm(A,ord=o)
ts=[]
for _ in range(5):
    t=time.perf_counter(); torch.linalg.matrix_norm(A,ord=o); ts.append((time.perf_counter()-t)*1e3)
y=torch.linalg.matrix_norm(A,ord=o); print("MS",sorted(ts)[0]); print("CHK",y.sum().item()); print("SHAPE",list(y.shape))
"#
            );
            let python = std::env::var("PYTORCH_PYTHON").unwrap_or("python3".into());
            print!("ord={ord} k={k}: FT {best:.1}ms chk {chk:.5e} shape {oshape:?}");
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
            let psh = s
                .lines()
                .find_map(|l| l.strip_prefix("SHAPE ").map(|v| v.to_string()))
                .unwrap_or_default();
            if let (Some(p), Some(pc)) = (g("MS "), g("CHK ")) {
                let rel = (chk - pc).abs() / (pc.abs() + 1e-9);
                let rr = p / best;
                println!(
                    " | torch {p:.1}ms chk {pc:.5e} shape {psh} | {} | {}",
                    if rel < 1e-6 {
                        "MATCH".into()
                    } else {
                        format!("rel{rel:.1e}")
                    },
                    if rr >= 1.0 {
                        format!("FT {rr:.2}x FASTER")
                    } else {
                        format!("FT {:.2}x slower", 1.0 / rr)
                    }
                );
            }
        }
    }
}

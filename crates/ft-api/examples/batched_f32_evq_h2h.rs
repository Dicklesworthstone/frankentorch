use ft_api::FrankenTorchSession;
use ft_core::{DType, ExecutionMode};
use std::process::Command;
use std::time::Instant;
fn main() {
    for (bb, k) in [(100000usize, 4usize), (20000, 16), (4000, 32)] {
        // symmetric for eigvalsh
        let mut sym = vec![0.0f64; bb * k * k];
        for b in 0..bb {
            for i in 0..k {
                for j in 0..k {
                    sym[b * k * k + i * k + j] = (((b * 7 + i * 13 + j * 5) % 97) as f64) * 0.01;
                }
            }
        }
        for b in 0..bb {
            for i in 0..k {
                for j in 0..k {
                    let s = (sym[b * k * k + i * k + j] + sym[b * k * k + j * k + i]) * 0.5;
                    sym[b * k * k + i * k + j] = s;
                }
            }
            for i in 0..k {
                sym[b * k * k + i * k + i] += k as f64;
            }
        }
        let genm: Vec<f64> = (0..bb * k * k)
            .map(|x| (((x * 2246822519usize) % 9941) as f64) * 0.002 - 9.0)
            .collect();
        let (mut eb, mut esum) = (f64::INFINITY, 0.0);
        let (mut qb, mut rsq) = (f64::INFINITY, 0.0);
        for _ in 0..5 {
            let mut s = FrankenTorchSession::new(ExecutionMode::Strict);
            let xf = s
                .tensor_variable(sym.clone(), vec![bb, k, k], false)
                .unwrap();
            let x = s.tensor_to_dtype(xf, DType::F32).unwrap();
            let t = Instant::now();
            let ev = s.tensor_linalg_eigvalsh(x).unwrap();
            let e = t.elapsed().as_secs_f64() * 1e3;
            if e < eb {
                eb = e;
                esum = s.tensor_values_lossy_f64(ev).unwrap().iter().sum();
            }
            let gf = s
                .tensor_variable(genm.clone(), vec![bb, k, k], false)
                .unwrap();
            let g = s.tensor_to_dtype(gf, DType::F32).unwrap();
            let t = Instant::now();
            let (_q, r) = s.tensor_linalg_qr(g, true).unwrap();
            let e = t.elapsed().as_secs_f64() * 1e3;
            if e < qb {
                qb = e;
                rsq = s
                    .tensor_values_lossy_f64(r)
                    .unwrap()
                    .iter()
                    .map(|v| v * v)
                    .sum();
            }
        }
        let py = format!(
            r#"
import time, torch
torch.set_num_threads(8); B,k={bb},{k}
idx=torch.arange(B*k*k,dtype=torch.float64); b=idx//(k*k); r=(idx//k)%k; c=idx%k
S=((((b*7+r*13+c*5)%97).double())*0.01).reshape(B,k,k); S=((S+S.transpose(-1,-2))*0.5+k*torch.eye(k,dtype=torch.float64)).float()
G=(((torch.arange(B*k*k,dtype=torch.float64)*2246822519)%9941)*0.002-9.0).reshape(B,k,k).float()
def bm(fn):
    for _ in range(2): fn()
    ts=[]
    for _ in range(5): t=time.perf_counter(); fn(); ts.append((time.perf_counter()-t)*1e3)
    return sorted(ts)[0]
em=bm(lambda: torch.linalg.eigvalsh(S)); qm=bm(lambda: torch.linalg.qr(G))
w=torch.linalg.eigvalsh(S); _,R=torch.linalg.qr(G)
print("EM",em); print("ESUM",w.double().sum().item()); print("QM",qm); print("RSQ",(R.double()**2).sum().item())
"#
        );
        let python = std::env::var("PYTORCH_PYTHON").unwrap_or("python3".into());
        let Ok(o) = Command::new(&python).arg("-c").arg(&py).output() else {
            continue;
        };
        if !o.status.success() {
            eprintln!("{}", String::from_utf8_lossy(&o.stderr));
            continue;
        }
        let s = String::from_utf8_lossy(&o.stdout);
        let g = |p: &str| {
            s.lines()
                .find_map(|l| l.strip_prefix(p).and_then(|v| v.trim().parse::<f64>().ok()))
        };
        let (em, es, qm, rs) = (
            g("EM ").unwrap(),
            g("ESUM ").unwrap(),
            g("QM ").unwrap(),
            g("RSQ ").unwrap(),
        );
        println!(
            "k={k} f32 eigvalsh: FT {eb:.1} vs {em:.1}ms = {:.2}x ({}) | qr: FT {qb:.1} vs {qm:.1}ms = {:.2}x ({})",
            em / eb,
            if (esum - es).abs() / (es.abs() + 1e-6) < 1e-3 {
                "OK"
            } else {
                "DIFF"
            },
            qm / qb,
            if (rsq - rs).abs() / (rs.abs() + 1e-6) < 1e-3 {
                "OK"
            } else {
                "DIFF"
            }
        );
    }
}

use ft_api::FrankenTorchSession;
use ft_core::ExecutionMode;
use std::process::Command;
use std::time::Instant;
fn main() {
    for (bb, k) in [(100000usize, 4usize), (20000, 16)] {
        // cond: full-rank random
        let mut cd: Vec<f64> = (0..bb * k * k)
            .map(|x| (((x * 2654435761usize) % 9973) as f64) * 0.001 - 5.0)
            .collect();
        for b in 0..bb {
            for i in 0..k {
                cd[b * k * k + i * k + i] += (k + 10) as f64;
            }
        }
        // rank: rank-deficient (row k-1 := row 0 per plane)
        let mut rd = cd.clone();
        for b in 0..bb {
            for j in 0..k {
                rd[b * k * k + (k - 1) * k + j] = rd[b * k * k + j];
            }
        }
        // --- cond ---
        let mut cb = f64::INFINITY;
        let mut csum = 0.0;
        for _ in 0..5 {
            let mut s = FrankenTorchSession::new(ExecutionMode::Strict);
            let x = s
                .tensor_variable(cd.clone(), vec![bb, k, k], false)
                .unwrap();
            let t = Instant::now();
            let y = s.tensor_linalg_cond(x, 2.0).unwrap();
            let e = t.elapsed().as_secs_f64() * 1e3;
            if e < cb {
                cb = e;
                csum = s.tensor_values(y).unwrap().iter().sum();
            }
        }
        // --- rank ---
        let mut rb = f64::INFINITY;
        let mut rsum = 0.0;
        let mut rshape = vec![];
        for _ in 0..5 {
            let mut s = FrankenTorchSession::new(ExecutionMode::Strict);
            let x = s
                .tensor_variable(rd.clone(), vec![bb, k, k], false)
                .unwrap();
            let t = Instant::now();
            let y = s.tensor_linalg_matrix_rank(x, None).unwrap();
            let e = t.elapsed().as_secs_f64() * 1e3;
            if e < rb {
                rb = e;
                let v = s.tensor_values(y).unwrap();
                rsum = v.iter().sum();
                rshape = s.tensor_shape(y).unwrap();
            }
        }
        let py = format!(
            r#"
import time, torch
torch.set_num_threads(8); B,k={bb},{k}
idx=torch.arange(B*k*k,dtype=torch.float64)
CD=((idx*2654435761)%9973).double().mul(0.001).sub(5.0).reshape(B,k,k); CD=CD+(k+10)*torch.eye(k,dtype=torch.float64)
RD=CD.clone(); RD[:,k-1,:]=RD[:,0,:]
def bm(fn):
    for _ in range(2): fn()
    import time as tm; xs=[]
    for _ in range(5): s=tm.perf_counter(); fn(); xs.append((tm.perf_counter()-s)*1e3)
    return sorted(xs)[0]
cm=bm(lambda: torch.linalg.cond(CD,2)); rm=bm(lambda: torch.linalg.matrix_rank(RD))
print("CM",cm); print("CSUM",torch.linalg.cond(CD,2).sum().item())
print("RM",rm); print("RSUM",torch.linalg.matrix_rank(RD).double().sum().item())
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
        let (cm, cs, rm, rs) = (
            g("CM ").unwrap(),
            g("CSUM ").unwrap(),
            g("RM ").unwrap(),
            g("RSUM ").unwrap(),
        );
        println!(
            "k={k} cond: FT {cb:.1} vs {cm:.1}ms = {:.2}x ({}) | rank: FT {rb:.1} vs {rm:.1}ms = {:.2}x ({} rsum {rsum} vs {rs} shape {rshape:?})",
            cm / cb,
            if (csum - cs).abs() / (cs.abs() + 1e-9) < 1e-3 {
                "OK".to_string()
            } else {
                format!("DIFF {csum:.4e} vs {cs:.4e}")
            },
            rm / rb,
            if (rsum - rs).abs() < 0.5 {
                "OK"
            } else {
                "DIFF"
            }
        );
    }
}

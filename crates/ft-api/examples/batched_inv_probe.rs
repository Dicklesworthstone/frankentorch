use ft_api::FrankenTorchSession;
use ft_core::ExecutionMode;
use std::process::Command;
use std::time::Instant;
fn main() {
    for (b, k) in [
        (20000usize, 16usize),
        (100000usize, 4usize),
        (5000usize, 32usize),
    ] {
        let mut a = vec![0.0f64; b * k * k];
        for m in 0..b {
            for i in 0..k {
                for j in 0..k {
                    a[m * k * k + i * k + j] = if i == j {
                        (k as f64) + 1.0
                    } else {
                        0.1 / ((i + j + 1) as f64)
                    };
                }
            }
        }
        let mut s = FrankenTorchSession::new(ExecutionMode::Strict);
        let av = s.tensor_variable(a.clone(), vec![b, k, k], false).unwrap();
        let op = |s: &mut FrankenTorchSession| -> f64 {
            let x = s.tensor_linalg_inv(av).unwrap();
            s.tensor_values(x).unwrap().iter().map(|v| v.abs()).sum()
        };
        for _ in 0..3 {
            let _ = op(&mut s);
        }
        let mut t = Vec::new();
        let mut c = 0.0;
        for _ in 0..20 {
            let s0 = Instant::now();
            c = op(&mut s);
            t.push(s0.elapsed().as_secs_f64() * 1e3);
        }
        t.sort_by(|x, y| x.partial_cmp(y).unwrap());
        let py = format!(
            "import time,torch\ntorch.set_num_threads(32)\nB,K={b},{k}\nv=torch.zeros(B,K,K,dtype=torch.float64)\nfor i in range(K):\n v[:,i,i]=K+1.0\n for j in range(K):\n  if i!=j: v[:,i,j]=0.1/(i+j+1)\nfor _ in range(3): torch.linalg.inv(v).abs().sum().item()\nimport time as tt;s=tt.perf_counter()\nfor _ in range(20): cc=torch.linalg.inv(v).abs().sum().item()\nprint((tt.perf_counter()-s)/20*1e3, cc)"
        );
        let out = Command::new(std::env::var("PYTORCH_PYTHON").unwrap_or("python3".into()))
            .arg("-c")
            .arg(&py)
            .output()
            .unwrap();
        let so = String::from_utf8_lossy(&out.stdout);
        let last = so.trim().lines().last().unwrap_or("");
        let mut it = last.split_whitespace();
        let pms: f64 = it.next().and_then(|x| x.parse().ok()).unwrap_or(-1.0);
        let pc: f64 = it.next().and_then(|x| x.parse().ok()).unwrap_or(0.0);
        let r = pms / t[0];
        let rel = (c - pc).abs() / (pc.abs() + 1e-9);
        println!(
            "inv [{b},{k},{k}]: FT {:.2}ms PyTorch {:.2}ms => {} (rel {:.1e})",
            t[0],
            pms,
            if r >= 1.0 {
                format!("FT {:.2}x FASTER", r)
            } else {
                format!("FT {:.2}x slower", 1.0 / r)
            },
            rel
        );
    }
}

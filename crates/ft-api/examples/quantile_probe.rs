use ft_api::FrankenTorchSession;
use ft_core::ExecutionMode;
use std::process::Command;
use std::time::Instant;
fn main() {
    for (rows, cols) in [(4000usize, 4000usize), (20000usize, 2000usize)] {
        let data: Vec<f64> = (0..rows * cols)
            .map(|i| (((i as u64).wrapping_mul(2654435761)) % 1000003) as f64 * 0.001 - 500.0)
            .collect();
        let mut s = FrankenTorchSession::new(ExecutionMode::Strict);
        let a = s
            .tensor_variable(data.clone(), vec![rows, cols], false)
            .unwrap();
        let op = |s: &mut FrankenTorchSession| -> f64 {
            let x = s.tensor_quantile_dim(a, 0.5, 1, false, "linear").unwrap();
            s.tensor_values(x).unwrap().iter().map(|v| v.abs()).sum()
        };
        for _ in 0..3 {
            op(&mut s);
        }
        let mut t = Vec::new();
        let mut c = 0.0;
        for _ in 0..15 {
            let s0 = Instant::now();
            c = op(&mut s);
            t.push(s0.elapsed().as_secs_f64() * 1e3);
        }
        t.sort_by(|x, y| x.partial_cmp(y).unwrap());
        let py = format!(
            "import time,torch,numpy as np\ntorch.set_num_threads(32)\nN={rows}*{cols}\nd=((np.arange(N,dtype=np.uint64)*np.uint64(2654435761))%np.uint64(1000003)).astype(np.float64)*0.001-500.0\na=torch.from_numpy(d.reshape({rows},{cols}))\nfor _ in range(3): torch.quantile(a,0.5,dim=1)\ns=time.perf_counter()\nfor _ in range(15): cc=torch.quantile(a,0.5,dim=1).abs().sum().item()\nprint((time.perf_counter()-s)/15*1e3, cc)"
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
            "quantile_dim [{rows},{cols}] q=.5: FT {:.2}ms PyTorch {:.2}ms => {} (rel {:.1e})",
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

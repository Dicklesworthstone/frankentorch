//! cumsum f64 head-to-head vs PyTorch (BlackThrush) — testing whether FT's parallel prefix-scan
//! beats PyTorch on its SLOW cumsum cases (PyTorch CPU cumsum is slow for strided dim=0 and
//! few-lane/long-scan shapes: measured [262144,64] dim0=189ms, [4M] 1-D=19.4ms, [64,262144]
//! dim1=23.9ms). cumsum has no transcendental (no Sleef wall), so this is a genuine candidate
//! for a real vs-PyTorch win. f64, no-grad. Verifies correctness (sum of result) + measures.
//!
//! Run: PYTORCH_PYTHON=/path/to/python cargo run --release -p ft-api --example cumsum_headtohead

use std::process::Command;
use std::time::Instant;

use ft_api::FrankenTorchSession;
use ft_core::ExecutionMode;

fn ft_cumsum(shape: &[usize], dim: usize, iters: usize) -> (f64, f64) {
    let n: usize = shape.iter().product();
    let data: Vec<f64> = (0..n).map(|i| (i as f64) * 1e-6).collect();
    let mut best = f64::INFINITY;
    let mut chk = 0.0;
    for _ in 0..iters + 3 {
        let mut s = FrankenTorchSession::new(ExecutionMode::Strict);
        let x = s
            .tensor_variable(data.clone(), shape.to_vec(), false)
            .unwrap();
        let t = Instant::now();
        let out = s.tensor_cumsum(x, dim).unwrap();
        let el = t.elapsed().as_secs_f64() * 1e3;
        chk = s.tensor_values(out).unwrap().iter().sum();
        if el < best {
            best = el;
        }
    }
    (chk, best)
}

const PY: &str = r#"
import os, time, sys
import torch
torch.set_num_threads(8)
shape = tuple(int(x) for x in os.environ["SHAPE"].split(","))
dim = int(os.environ["DIM"])
n=1
for d in shape: n*=d
x = torch.arange(n, dtype=torch.float64).mul_(1e-6).reshape(*shape)
for _ in range(3): torch.cumsum(x, dim=dim)
ts=[]; chk=0.0
for _ in range(15):
    t=time.perf_counter(); o=torch.cumsum(x, dim=dim); ts.append((time.perf_counter()-t)*1e3); chk=o.sum().item()
print("MS", sorted(ts)[0]); print("CHK", chk)
"#;

fn py(shape: &[usize], dim: usize) -> (Option<f64>, Option<f64>) {
    let python = std::env::var("PYTORCH_PYTHON").unwrap_or_else(|_| "python3".to_string());
    let shp = shape
        .iter()
        .map(|x| x.to_string())
        .collect::<Vec<_>>()
        .join(",");
    let o = Command::new(&python)
        .arg("-c")
        .arg(PY)
        .env("SHAPE", shp)
        .env("DIM", dim.to_string())
        .output();
    match o {
        Ok(o) if o.status.success() => {
            let s = String::from_utf8_lossy(&o.stdout);
            let g = |p: &str| {
                s.lines()
                    .find_map(|l| l.strip_prefix(p).and_then(|v| v.trim().parse::<f64>().ok()))
            };
            (g("MS "), g("CHK "))
        }
        _ => (None, None),
    }
}

fn main() {
    let cases: &[(&[usize], usize)] = &[
        (&[262144, 64], 0),
        (&[4194304], 0),
        (&[64, 262144], 1),
        (&[2048, 2048], 1),
    ];
    println!("cumsum f64 head-to-head (15 iters MIN):");
    for (shape, dim) in cases {
        let (ft_sum, ft_ms) = ft_cumsum(shape, *dim, 15);
        let (py_ms, py_sum) = py(shape, *dim);
        print!("  {shape:?} dim={dim}: FT {ft_ms:8.3} ms");
        match (py_ms, py_sum) {
            (Some(p), Some(ps)) => {
                let rel = (ft_sum - ps).abs() / (ps.abs() + 1e-12);
                let r = p / ft_ms;
                let verdict = if r >= 1.0 {
                    format!("FT {r:.2}x FASTER")
                } else {
                    format!("FT {:.2}x slower", 1.0 / r)
                };
                let corr = if rel < 1e-9 { "MATCH" } else { "MISMATCH!" };
                println!("  | PyTorch {p:8.3} ms  => {verdict}  ({corr} {rel:.1e})");
            }
            _ => println!("  | PyTorch (unavailable)"),
        }
    }
}

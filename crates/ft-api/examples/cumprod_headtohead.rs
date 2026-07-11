//! cumprod f64 head-to-head vs PyTorch (BlackThrush) — same cache-friendly loop-reorder lever as
//! cumsum (0ccf6167). PyTorch CPU cumprod along a strided (non-last) dim is cache-thrashing
//! (measured [262144,64] dim0=205ms vs dim1=24ms). FT's reordered kernel (cumprod_lane_block_f64)
//! walks contiguous inner runs. Values alternate 1.0001/0.9999 so the product stays bounded.
//! Verifies correctness (sum of result) + measures. f64, no-grad.
//!
//! Run: PYTORCH_PYTHON=/path/to/python cargo run --release -p ft-api --example cumprod_headtohead

use std::process::Command;
use std::time::Instant;

use ft_api::FrankenTorchSession;
use ft_core::ExecutionMode;

fn vdata(n: usize) -> Vec<f64> {
    (0..n)
        .map(|i| if i % 2 == 0 { 1.0001 } else { 0.9999 })
        .collect()
}

fn ft_cumprod(shape: &[usize], dim: usize, iters: usize) -> (f64, f64) {
    let n: usize = shape.iter().product();
    let d = vdata(n);
    let mut best = f64::INFINITY;
    let mut chk = 0.0;
    for _ in 0..iters + 3 {
        let mut s = FrankenTorchSession::new(ExecutionMode::Strict);
        let x = s.tensor_variable(d.clone(), shape.to_vec(), false).unwrap();
        let t = Instant::now();
        let out = s.tensor_cumprod(x, dim).unwrap();
        let el = t.elapsed().as_secs_f64() * 1e3;
        chk = s.tensor_values(out).unwrap().iter().sum();
        if el < best {
            best = el;
        }
    }
    (chk, best)
}

const PY: &str = r#"
import os, time, torch
torch.set_num_threads(8)
shape = tuple(int(x) for x in os.environ["SHAPE"].split(","))
dim = int(os.environ["DIM"])
n=1
for d in shape: n*=d
x = torch.where(torch.arange(n)%2==0, torch.tensor(1.0001,dtype=torch.float64), torch.tensor(0.9999,dtype=torch.float64)).reshape(*shape)
for _ in range(3): torch.cumprod(x, dim=dim)
ts=[]; chk=0.0
for _ in range(15):
    t=time.perf_counter(); o=torch.cumprod(x, dim=dim); ts.append((time.perf_counter()-t)*1e3); chk=o.sum().item()
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
    let cases: &[(&[usize], usize)] = &[(&[262144, 64], 0), (&[2048, 2048], 0), (&[64, 262144], 1)];
    println!("cumprod f64 head-to-head (15 iters MIN):");
    for (shape, dim) in cases {
        let (fs, fm) = ft_cumprod(shape, *dim, 15);
        let (pm, ps) = py(shape, *dim);
        print!("  {shape:?} dim={dim}: FT {fm:8.3} ms");
        match (pm, ps) {
            (Some(p), Some(psum)) => {
                let rel = (fs - psum).abs() / (psum.abs() + 1e-12);
                let r = p / fm;
                let v = if r >= 1.0 {
                    format!("FT {r:.2}x FASTER")
                } else {
                    format!("FT {:.2}x slower", 1.0 / r)
                };
                println!(
                    "  | PyTorch {p:8.3} ms => {v}  ({} {rel:.1e})",
                    if rel < 1e-9 { "MATCH" } else { "MISMATCH!" }
                );
            }
            _ => println!("  | PyTorch (unavailable)"),
        }
    }
}

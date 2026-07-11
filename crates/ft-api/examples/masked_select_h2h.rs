//! torch.masked_select(x, mask) head-to-head vs PyTorch. torch is serial (~30ms@4M). FT builds the
//! kept-index list with a SERIAL filter-collect, then index_select. f64 no-grad.
//!
//! Run: PYTORCH_PYTHON=/path/to/python cargo run --release -p ft-api --example masked_select_h2h

use std::process::Command;
use std::time::Instant;

use ft_api::FrankenTorchSession;
use ft_core::ExecutionMode;

const N: usize = 4_000_000;

fn main() {
    let data: Vec<f64> = (0..N).map(|i| ((i as f64) * 0.0007).sin()).collect();
    // ~50% kept.
    let mask: Vec<f64> = (0..N)
        .map(|i| if data[i] > 0.0 { 1.0 } else { 0.0 })
        .collect();
    let mut best = f64::INFINITY;
    let mut chk = 0.0;
    let mut klen = 0.0;
    for _ in 0..6 {
        let mut s = FrankenTorchSession::new(ExecutionMode::Strict);
        let x = s.tensor_variable(data.clone(), vec![N], false).unwrap();
        let m = s.tensor_variable(mask.clone(), vec![N], false).unwrap();
        let t = Instant::now();
        let out = s.tensor_masked_select(x, m).unwrap();
        let el = t.elapsed().as_secs_f64() * 1e3;
        if el < best {
            best = el;
            let v = s.tensor_values(out).unwrap();
            klen = v.len() as f64;
            chk = v.iter().sum();
        }
    }
    let python = std::env::var("PYTORCH_PYTHON").unwrap_or_else(|_| "python3".to_string());
    let py = format!(
        r#"
import time, torch
torch.set_num_threads(8)
N={N}
x = (torch.arange(N, dtype=torch.float64).mul_(0.0007).sin_())
m = x > 0
for _ in range(2): torch.masked_select(x, m)
ts=[]; chk=0.0; kl=0
for _ in range(6):
    s=time.perf_counter(); o=torch.masked_select(x, m); ts.append((time.perf_counter()-s)*1e3); chk=o.sum().item(); kl=o.numel()
print("MS", sorted(ts)[0]); print("CHK", chk); print("KL", kl)
"#
    );
    let out = Command::new(&python).arg("-c").arg(&py).output();
    println!("masked_select(x[{N}], mask) f64 no-grad, 6 iters MIN:");
    println!("  FrankenTorch : {best:9.2} ms   kept {klen:.0}  chk {chk:.6e}");
    if let Ok(o) = out
        && o.status.success()
    {
        let s = String::from_utf8_lossy(&o.stdout);
        let g = |p: &str| {
            s.lines()
                .find_map(|l| l.strip_prefix(p).and_then(|v| v.trim().parse::<f64>().ok()))
        };
        if let (Some(p), Some(pc), Some(kl)) = (g("MS "), g("CHK "), g("KL ")) {
            let ok = (klen - kl).abs() < 0.5 && (chk - pc).abs() / (pc.abs() + 1e-6) < 1e-9;
            let r = p / best;
            let verdict = if r >= 1.0 {
                format!("FT {r:.2}x FASTER")
            } else {
                format!("FT {:.2}x slower", 1.0 / r)
            };
            println!(
                "  PyTorch      : {p:9.2} ms   kept {kl:.0}  chk {pc:.6e}  => {verdict}  [{}]",
                if ok { "MATCH" } else { "MISMATCH!" }
            );
        }
    }
}

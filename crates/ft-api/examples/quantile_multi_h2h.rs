//! Multi-q quantile FT vs torch on a large 1-D array. FT tensor_quantile_multi
//! loops scalar quantile per q (N independent quickselects + clones); torch
//! sorts ONCE for all q. Probes whether the per-q loop is slower than torch.
use ft_api::FrankenTorchSession;
use ft_core::ExecutionMode;
use std::io::Write;
use std::process::{Command, Stdio};
use std::time::Instant;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let python = std::env::var("PYTORCH_PYTHON").unwrap_or_else(|_| "python3".to_string());
    let n: usize = std::env::var("N")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(8_000_000);
    let data: Vec<f64> = (0..n)
        .map(|i| {
            let z = (i as u64)
                .wrapping_mul(2862933555777941757)
                .wrapping_add(3037000493);
            ((z >> 11) as f64 / (1u64 << 53) as f64) * 2.0 - 1.0
        })
        .collect();
    let qs = [0.1, 0.25, 0.5, 0.75, 0.9];
    // 0=median anchor 1=quantile_q1 2=quantile_q5
    let tt = |which: u8| {
        let mut best = f64::INFINITY;
        for _ in 0..5 {
            let mut s = FrankenTorchSession::new(ExecutionMode::Strict);
            let x = s.tensor_variable(data.clone(), vec![n], false).unwrap();
            let ti = Instant::now();
            match which {
                0 => {
                    let _ = s.tensor_median(x);
                }
                1 => {
                    let _ = s.tensor_quantile_multi(x, &[0.5], "linear");
                }
                _ => {
                    let _ = s.tensor_quantile_multi(x, &qs, "linear");
                }
            }
            let e = ti.elapsed().as_secs_f64() * 1e3;
            if e < best {
                best = e;
            }
        }
        best
    };
    let py = format!(
        r#"
import time, torch
torch.set_num_threads(8)
n={n}
g=torch.Generator().manual_seed(0)
x=torch.rand(n,generator=g,dtype=torch.float64)*2-1
q5=torch.tensor([0.1,0.25,0.5,0.75,0.9],dtype=torch.float64)
q1=torch.tensor([0.5],dtype=torch.float64)
def tm(fn,reps=5):
    for _ in range(2): fn()
    ts=[]
    for _ in range(reps): s=time.perf_counter(); fn(); ts.append((time.perf_counter()-s)*1e3)
    return min(ts)
print("PT median %.3f"%tm(lambda:torch.median(x)))
print("PT q1 %.3f"%tm(lambda:torch.quantile(x,q1)))
print("PT q5 %.3f"%tm(lambda:torch.quantile(x,q5)))
"#,
        n = n
    );
    let mut ch = Command::new(&python)
        .arg("-")
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .spawn()?;
    ch.stdin.as_mut().unwrap().write_all(py.as_bytes())?;
    let pt = String::from_utf8_lossy(&ch.wait_with_output()?.stdout).to_string();
    let g = |k: &str| {
        pt.lines()
            .find_map(|l| {
                let mut it = l.strip_prefix("PT ")?.split_whitespace();
                if it.next()? == k {
                    it.next()?.parse::<f64>().ok()
                } else {
                    None
                }
            })
            .unwrap_or(f64::NAN)
    };
    let v = |ft: f64, p: f64| {
        if p >= ft {
            format!("FT {:.2}x FASTER", p / ft)
        } else {
            format!("FT {:.2}x SLOWER", ft / p)
        }
    };
    println!("multi-q quantile, N={n} f64 (torch 8t / FT default), min-of-5");
    for (lbl, key, w) in [
        ("median", "median", 0u8),
        ("quantile_q1", "q1", 1),
        ("quantile_q5", "q5", 2),
    ] {
        let ft = tt(w);
        println!(
            "  {lbl:<12} FT {ft:9.3}  PT {:9.3}  => {}",
            g(key),
            v(ft, g(key))
        );
    }
    Ok(())
}

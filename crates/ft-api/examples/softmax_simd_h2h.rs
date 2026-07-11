//! f32 softmax/log_softmax FT vs torch. FT's kernel parallelizes over rows but
//! does the within-row exp with SCALAR libm; torch vectorizes exp. Conformance
//! allows 8 ULP (libm != MKL), so a SIMD exp is shippable. Measures the gap.
use ft_api::FrankenTorchSession;
use ft_core::ExecutionMode;
use std::io::Write;
use std::process::{Command, Stdio};
use std::time::Instant;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let python = std::env::var("PYTORCH_PYTHON").unwrap_or_else(|_| "python3".to_string());
    let (r, c) = (8192usize, 4096usize);
    let data: Vec<f32> = (0..r * c)
        .map(|i| {
            let z = (i as u64)
                .wrapping_mul(2862933555777941757)
                .wrapping_add(3037000493);
            ((z >> 40) as f32 / (1u64 << 24) as f32) * 8.0 - 4.0
        })
        .collect();
    // 0=add anchor 1=softmax 2=log_softmax
    let tt = |which: u8| {
        let mut best = f64::INFINITY;
        for _ in 0..7 {
            let mut s = FrankenTorchSession::new(ExecutionMode::Strict);
            let x = s
                .tensor_variable_f32(data.clone(), vec![r, c], false)
                .unwrap();
            let ti = Instant::now();
            match which {
                0 => {
                    let _ = s.tensor_add(x, x);
                }
                1 => {
                    let _ = s.tensor_softmax(x, 1);
                }
                _ => {
                    let _ = s.tensor_log_softmax(x, 1);
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
import torch.nn.functional as F
torch.set_num_threads(8)
r,c={r},{c}
x=(((torch.arange(r*c,dtype=torch.int64)%99371).float()/99371.0)*8.0-4.0).reshape(r,c)
def tm(fn,n=7):
    for _ in range(2): fn()
    ts=[]
    for _ in range(n): s=time.perf_counter(); fn(); ts.append((time.perf_counter()-s)*1e3)
    return min(ts)
print("PT add %.3f"%tm(lambda:x+x))
print("PT softmax %.3f"%tm(lambda:F.softmax(x,dim=1)))
print("PT log_softmax %.3f"%tm(lambda:F.log_softmax(x,dim=1)))
"#,
        r = r,
        c = c
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
    println!("f32 softmax [{r},{c}] dim=1 (torch 8t / FT default), min-of-7");
    for (lbl, w) in [("add", 0u8), ("softmax", 1), ("log_softmax", 2)] {
        let ft = tt(w);
        println!(
            "  {lbl:<12} FT {ft:8.3}  PT {:8.3}  => {}",
            g(lbl),
            v(ft, g(lbl))
        );
    }
    Ok(())
}

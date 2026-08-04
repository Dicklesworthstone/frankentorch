//! Profile f64 median + f32/f64 kthvalue vs torch (selection follow-ups to f32 median).
use ft_api::FrankenTorchSession;
use ft_core::ExecutionMode;
use std::io::Write;
use std::process::{Command, Stdio};
use std::time::Instant;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let python = std::env::var("PYTORCH_PYTHON").unwrap_or_else(|_| "python3".to_string());
    let n = 16_000_000usize;
    let d64: Vec<f64> = (0..n)
        .map(|i| ((i * 7919) % 1_000_003) as f64 * 0.001)
        .collect();
    let d32: Vec<f32> = d64.iter().map(|&v| v as f32).collect();
    let kth = n / 2;
    let tt = |which: u8| {
        let mut best = f64::INFINITY;
        for _ in 0..7 {
            let mut s = FrankenTorchSession::new(ExecutionMode::Strict);
            let ti;
            match which {
                0 => {
                    let x = s.tensor_variable(d64.clone(), vec![n], false).unwrap();
                    ti = Instant::now();
                    let _ = s.tensor_add(x, x);
                }
                1 => {
                    let x = s.tensor_variable(d64.clone(), vec![n], false).unwrap();
                    ti = Instant::now();
                    let _ = s.tensor_median(x);
                }
                2 => {
                    let x = s.tensor_variable_f32(d32.clone(), vec![n], false).unwrap();
                    ti = Instant::now();
                    let _ = s.tensor_kthvalue(x, kth);
                }
                _ => {
                    let x = s.tensor_variable(d64.clone(), vec![n], false).unwrap();
                    ti = Instant::now();
                    let _ = s.tensor_kthvalue(x, kth);
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
n={n}; kth={kth}
d64=(((torch.arange(n,dtype=torch.int64)*7919)%1000003).double()*0.001)
d32=d64.float()
def tm(fn,reps=7):
    for _ in range(2): fn()
    ts=[]
    for _ in range(reps): s=time.perf_counter(); fn(); ts.append((time.perf_counter()-s)*1e3)
    return min(ts)
print("PT add %.3f"%tm(lambda:d64+d64))
print("PT f64median %.3f"%tm(lambda:torch.median(d64)))
print("PT f32kth %.3f"%tm(lambda:torch.kthvalue(d32,kth)))
print("PT f64kth %.3f"%tm(lambda:torch.kthvalue(d64,kth)))
"#,
        n = n,
        kth = kth
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
    println!("f64 median + kthvalue, n={n} (torch 8t / FT default), min-of-7");
    for (lbl, w) in [("add", 0u8), ("f64median", 1), ("f32kth", 2), ("f64kth", 3)] {
        let ft = tt(w);
        println!(
            "  {lbl:<10} FT {ft:8.3}  PT {:8.3}  => {}",
            g(lbl),
            v(ft, g(lbl))
        );
    }
    Ok(())
}

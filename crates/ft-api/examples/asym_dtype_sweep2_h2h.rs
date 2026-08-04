//! Sweep 2: more f32 ops likely upcasting through lossy_f64 / apply_function.
use ft_api::FrankenTorchSession;
use ft_core::ExecutionMode;
use std::io::Write;
use std::process::{Command, Stdio};
use std::time::Instant;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let python = std::env::var("PYTORCH_PYTHON").unwrap_or_else(|_| "python3".to_string());
    let n = 16_000_000usize;
    let a: Vec<f32> = (0..n)
        .map(|i| ((i % 9973) as f32 - 5000.0) * 0.01)
        .collect();
    let b: Vec<f32> = (0..n).map(|i| ((i % 7919) as f32) * 0.001 + 0.1).collect();
    let p: Vec<f32> = (0..n).map(|i| ((i % 999) as f32 + 1.0) * 0.001).collect(); // (0,1) for logit
    let tt = |which: u8| {
        let mut best = f64::INFINITY;
        for _ in 0..7 {
            let mut s = FrankenTorchSession::new(ExecutionMode::Strict);
            let ti;
            match which {
                0 => {
                    let x = s.tensor_variable_f32(a.clone(), vec![n], false).unwrap();
                    ti = Instant::now();
                    let _ = s.tensor_add(x, x);
                }
                1 => {
                    let x = s.tensor_variable_f32(a.clone(), vec![n], false).unwrap();
                    ti = Instant::now();
                    let _ = s.tensor_nanmedian(x);
                }
                2 => {
                    let x = s.tensor_variable_f32(a.clone(), vec![n], false).unwrap();
                    ti = Instant::now();
                    let _ = s.tensor_special_i1e(x);
                }
                3 => {
                    let x = s.tensor_variable_f32(p.clone(), vec![n], false).unwrap();
                    ti = Instant::now();
                    let _ = s.tensor_logit(x, None);
                }
                4 => {
                    let x = s.tensor_variable_f32(b.clone(), vec![n], false).unwrap();
                    let y = s.tensor_variable_f32(a.clone(), vec![n], false).unwrap();
                    ti = Instant::now();
                    let _ = s.tensor_xlogy(x, y);
                }
                5 => {
                    let x = s.tensor_variable_f32(a.clone(), vec![n], false).unwrap();
                    let y = s.tensor_variable_f32(b.clone(), vec![n], false).unwrap();
                    ti = Instant::now();
                    let _ = s.tensor_hypot(x, y);
                }
                _ => {
                    let x = s.tensor_variable_f32(a.clone(), vec![n], false).unwrap();
                    ti = Instant::now();
                    let _ = s.tensor_nan_to_num(x, 0.0, None, None);
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
a=(((torch.arange(n,dtype=torch.int64)%9973).float()-5000.0)*0.01)
b=(((torch.arange(n,dtype=torch.int64)%7919).float())*0.001+0.1)
p=(((torch.arange(n,dtype=torch.int64)%999).float()+1.0)*0.001)
def tm(fn,reps=7):
    for _ in range(2): fn()
    ts=[]
    for _ in range(reps): s=time.perf_counter(); fn(); ts.append((time.perf_counter()-s)*1e3)
    return min(ts)
print("PT add %.3f"%tm(lambda:a+a))
print("PT nanmedian %.3f"%tm(lambda:torch.nanmedian(a)))
print("PT i1e %.3f"%tm(lambda:torch.special.i1e(a)))
print("PT logit %.3f"%tm(lambda:torch.logit(p)))
print("PT xlogy %.3f"%tm(lambda:torch.xlogy(b,a)))
print("PT hypot %.3f"%tm(lambda:torch.hypot(a,b)))
print("PT nan_to_num %.3f"%tm(lambda:torch.nan_to_num(a))
)
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
    println!("asym-dtype sweep2 ~16M f32 (torch 8t / FT default), min-of-7");
    for (lbl, w) in [
        ("add", 0u8),
        ("nanmedian", 1),
        ("i1e", 2),
        ("logit", 3),
        ("xlogy", 4),
        ("hypot", 5),
        ("nan_to_num", 6),
    ] {
        let ft = tt(w);
        println!(
            "  {lbl:<11} FT {ft:8.3}  PT {:8.3}  => {}",
            g(lbl),
            v(ft, g(lbl))
        );
    }
    Ok(())
}

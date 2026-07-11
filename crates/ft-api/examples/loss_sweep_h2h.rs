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
    let b: Vec<f32> = (0..n)
        .map(|i| ((i % 7919) as f32 - 4000.0) * 0.02)
        .collect();
    let p: Vec<f32> = (0..n).map(|i| ((i % 999) as f32 + 1.0) * 0.001).collect();
    let tt = |w: u8| {
        let mut best = f64::INFINITY;
        for _ in 0..7 {
            let mut s = FrankenTorchSession::new(ExecutionMode::Strict);
            let ti;
            match w {
                0 => {
                    let x = s.tensor_variable_f32(a.clone(), vec![n], false).unwrap();
                    ti = Instant::now();
                    let _ = s.tensor_add(x, x);
                }
                1 => {
                    let x = s.tensor_variable_f32(a.clone(), vec![n], false).unwrap();
                    let y = s.tensor_variable_f32(b.clone(), vec![n], false).unwrap();
                    ti = Instant::now();
                    let _ = s.tensor_mse_loss(x, y, "mean");
                }
                2 => {
                    let x = s.tensor_variable_f32(a.clone(), vec![n], false).unwrap();
                    let y = s.tensor_variable_f32(b.clone(), vec![n], false).unwrap();
                    ti = Instant::now();
                    let _ = s.tensor_l1_loss(x, y, "mean");
                }
                3 => {
                    let x = s.tensor_variable_f32(a.clone(), vec![n], false).unwrap();
                    let y = s.tensor_variable_f32(b.clone(), vec![n], false).unwrap();
                    ti = Instant::now();
                    let _ = s.tensor_smooth_l1_loss(x, y, "mean", 1.0);
                }
                4 => {
                    let x = s.tensor_variable_f32(a.clone(), vec![n], false).unwrap();
                    let y = s.tensor_variable_f32(b.clone(), vec![n], false).unwrap();
                    ti = Instant::now();
                    let _ = s.tensor_huber_loss(x, y, "mean", 1.0);
                }
                5 => {
                    let x = s.tensor_variable_f32(p.clone(), vec![n], false).unwrap();
                    ti = Instant::now();
                    let _ = s.tensor_special_ndtri(x);
                }
                _ => {
                    let x = s.tensor_variable_f32(a.clone(), vec![n], false).unwrap();
                    ti = Instant::now();
                    let _ = s.tensor_angle(x);
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
import time,torch
import torch.nn.functional as F
torch.set_num_threads(8)
n={n}
a=(((torch.arange(n,dtype=torch.int64)%9973).float()-5000.0)*0.01)
b=(((torch.arange(n,dtype=torch.int64)%7919).float()-4000.0)*0.02)
p=(((torch.arange(n,dtype=torch.int64)%999).float()+1.0)*0.001)
def tm(fn,reps=7):
    for _ in range(2): fn()
    ts=[]
    for _ in range(reps): s=time.perf_counter(); fn(); ts.append((time.perf_counter()-s)*1e3)
    return min(ts)
print("PT add %.3f"%tm(lambda:a+a))
print("PT mse %.3f"%tm(lambda:F.mse_loss(a,b)))
print("PT l1 %.3f"%tm(lambda:F.l1_loss(a,b)))
print("PT smooth_l1 %.3f"%tm(lambda:F.smooth_l1_loss(a,b,beta=1.0)))
print("PT huber %.3f"%tm(lambda:F.huber_loss(a,b,delta=1.0)))
print("PT ndtri %.3f"%tm(lambda:torch.special.ndtri(p)))
print("PT angle %.3f"%tm(lambda:torch.angle(a)))
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
    println!("loss sweep ~16M f32 (torch 8t / FT default), min-of-7");
    for (lbl, w) in [
        ("add", 0u8),
        ("mse", 1),
        ("l1", 2),
        ("smooth_l1", 3),
        ("huber", 4),
        ("ndtri", 5),
        ("angle", 6),
    ] {
        let ft = tt(w);
        println!(
            "  {lbl:<10} FT {ft:8.3}  PT {:8.3}  => {}",
            g(lbl),
            v(ft, g(lbl))
        );
    }
    Ok(())
}

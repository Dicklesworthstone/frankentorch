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
    let b: Vec<f32> = (0..n).map(|i| ((i % 7919) as f32) * 0.001 + 0.5).collect();
    let c: Vec<f32> = (0..n).map(|i| ((i % 6131) as f32) * 0.002 + 0.3).collect();
    let tt = |w: u8| {
        let mut best = f64::INFINITY;
        for _ in 0..7 {
            let mut s = FrankenTorchSession::new(ExecutionMode::Strict);
            let x = s.tensor_variable_f32(a.clone(), vec![n], false).unwrap();
            let y = s.tensor_variable_f32(b.clone(), vec![n], false).unwrap();
            let z = s.tensor_variable_f32(c.clone(), vec![n], false).unwrap();
            let ti = Instant::now();
            match w {
                0 => {
                    let _ = s.tensor_add(x, x);
                }
                1 => {
                    let _ = s.tensor_addcdiv(x, y, z, 0.5);
                }
                2 => {
                    let _ = s.tensor_addcmul(x, y, z, 0.5);
                }
                3 => {
                    let _ = s.tensor_hardshrink(x, 0.5);
                }
                _ => {
                    let _ = s.tensor_normalize(x, 2.0, 0, 1e-12);
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
b=(((torch.arange(n,dtype=torch.int64)%7919).float())*0.001+0.5)
c=(((torch.arange(n,dtype=torch.int64)%6131).float())*0.002+0.3)
def tm(fn,reps=7):
    for _ in range(2): fn()
    ts=[]
    for _ in range(reps): s=time.perf_counter(); fn(); ts.append((time.perf_counter()-s)*1e3)
    return min(ts)
print("PT add %.3f"%tm(lambda:a+a))
print("PT addcdiv %.3f"%tm(lambda:torch.addcdiv(a,b,c,value=0.5)))
print("PT addcmul %.3f"%tm(lambda:torch.addcmul(a,b,c,value=0.5)))
print("PT hardshrink %.3f"%tm(lambda:F.hardshrink(a,0.5)))
print("PT normalize %.3f"%tm(lambda:F.normalize(a,p=2.0,dim=0,eps=1e-12)))
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
    println!("fused/elementwise ~16M f32 (torch 8t / FT default), min-of-7");
    for (lbl, w) in [
        ("add", 0u8),
        ("addcdiv", 1),
        ("addcmul", 2),
        ("hardshrink", 3),
        ("normalize", 4),
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

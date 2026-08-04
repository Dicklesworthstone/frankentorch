use ft_api::FrankenTorchSession;
use ft_core::ExecutionMode;
use std::io::Write;
use std::process::{Command, Stdio};
use std::time::Instant;
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let python = std::env::var("PYTORCH_PYTHON").unwrap_or_else(|_| "python3".to_string());
    let m = 4096usize;
    let mat: Vec<f32> = (0..m * m)
        .map(|i| ((i % 9973) as f32 - 5000.0) * 0.01)
        .collect();
    let big: Vec<f32> = (0..16_000_000usize)
        .map(|i| ((i % 9973) as f32 - 5000.0) * 0.01)
        .collect();
    let big2: Vec<f32> = (0..16_000_000usize)
        .map(|i| ((i % 7919) as f32) * 0.001 + 1.5)
        .collect();
    let tt = |w: u8| {
        let mut best = f64::INFINITY;
        for _ in 0..7 {
            let mut s = FrankenTorchSession::new(ExecutionMode::Strict);
            let ti;
            match w {
                0 => {
                    let x = s
                        .tensor_variable_f32(big.clone(), vec![16_000_000], false)
                        .unwrap();
                    ti = Instant::now();
                    let _ = s.tensor_add(x, x);
                }
                1 => {
                    let x = s
                        .tensor_variable_f32(mat.clone(), vec![m, m], false)
                        .unwrap();
                    ti = Instant::now();
                    let _ = s.tensor_std_mean(x, 1, 1);
                }
                2 => {
                    let x = s
                        .tensor_variable_f32(mat.clone(), vec![m, m], false)
                        .unwrap();
                    ti = Instant::now();
                    let _ = s.tensor_var_mean(x, 1, 1);
                }
                3 => {
                    let x = s
                        .tensor_variable_f32(mat.clone(), vec![m, m], false)
                        .unwrap();
                    ti = Instant::now();
                    let _ = s.tensor_aminmax(x, 1);
                }
                4 => {
                    let x = s
                        .tensor_variable_f32(big.clone(), vec![16_000_000], false)
                        .unwrap();
                    let y = s
                        .tensor_variable_f32(big2.clone(), vec![16_000_000], false)
                        .unwrap();
                    ti = Instant::now();
                    let _ = s.tensor_atan2(x, y);
                }
                5 => {
                    let x = s
                        .tensor_variable_f32(big.clone(), vec![16_000_000], false)
                        .unwrap();
                    ti = Instant::now();
                    let _ = s.tensor_asinh(x);
                }
                _ => {
                    let x = s
                        .tensor_variable_f32(big2.clone(), vec![16_000_000], false)
                        .unwrap();
                    ti = Instant::now();
                    let _ = s.tensor_acosh(x);
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
torch.set_num_threads(8)
m={m}
mat=(((torch.arange(m*m,dtype=torch.int64)%9973).float()-5000.0)*0.01).reshape(m,m)
big=(((torch.arange(16000000,dtype=torch.int64)%9973).float()-5000.0)*0.01)
big2=(((torch.arange(16000000,dtype=torch.int64)%7919).float())*0.001+1.5)
def tm(fn,reps=7):
    for _ in range(2): fn()
    ts=[]
    for _ in range(reps): s=time.perf_counter(); fn(); ts.append((time.perf_counter()-s)*1e3)
    return min(ts)
print("PT add %.3f"%tm(lambda:big+big))
print("PT std_mean %.3f"%tm(lambda:torch.std_mean(mat,1,correction=1)))
print("PT var_mean %.3f"%tm(lambda:torch.var_mean(mat,1,correction=1)))
print("PT aminmax %.3f"%tm(lambda:torch.aminmax(mat,dim=1)))
print("PT atan2 %.3f"%tm(lambda:torch.atan2(big,big2)))
print("PT asinh %.3f"%tm(lambda:torch.asinh(big)))
print("PT acosh %.3f"%tm(lambda:torch.acosh(big2)))
"#,
        m = m
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
    println!("reduction sweep (torch 8t / FT default), min-of-7");
    for (lbl, w) in [
        ("add", 0u8),
        ("std_mean", 1),
        ("var_mean", 2),
        ("aminmax", 3),
        ("atan2", 4),
        ("asinh", 5),
        ("acosh", 6),
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

use ft_api::FrankenTorchSession;
use ft_core::ExecutionMode;
use std::io::Write;
use std::process::{Command, Stdio};
use std::time::Instant;
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let python = std::env::var("PYTORCH_PYTHON").unwrap_or_else(|_| "python3".to_string());
    let n = 16_000_000usize;
    let a: Vec<f32> = (0..n).map(|i| ((i % 1999) as f32 / 2000.0) - 0.5).collect(); // (-0.5,0.5) for asin/acos
    let b: Vec<f32> = (0..n).map(|i| ((i % 9973) as f32) * 0.001 + 0.1).collect(); // >0 for log
    let tt = |w: u8| {
        let mut best = f64::INFINITY;
        for _ in 0..7 {
            let mut s = FrankenTorchSession::new(ExecutionMode::Strict);
            let x = s.tensor_variable_f32(a.clone(), vec![n], false).unwrap();
            let y = s.tensor_variable_f32(b.clone(), vec![n], false).unwrap();
            let ti = Instant::now();
            match w {
                0 => {
                    let _ = s.tensor_add(x, x);
                }
                1 => {
                    let _ = s.tensor_asin(x);
                }
                2 => {
                    let _ = s.tensor_acos(x);
                }
                3 => {
                    let _ = s.tensor_atan(x);
                }
                4 => {
                    let _ = s.tensor_expm1(x);
                }
                5 => {
                    let _ = s.tensor_log1p(y);
                }
                6 => {
                    let _ = s.tensor_rsqrt(y);
                }
                7 => {
                    let _ = s.tensor_digamma(y);
                }
                _ => {
                    let _ = s.tensor_lgamma(y);
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
n={n}
a=(((torch.arange(n,dtype=torch.int64)%1999).float()/2000.0)-0.5)
b=(((torch.arange(n,dtype=torch.int64)%9973).float())*0.001+0.1)
def tm(fn,reps=7):
    for _ in range(2): fn()
    ts=[]
    for _ in range(reps): s=time.perf_counter(); fn(); ts.append((time.perf_counter()-s)*1e3)
    return min(ts)
print("PT add %.3f"%tm(lambda:a+a))
print("PT asin %.3f"%tm(lambda:torch.asin(a)))
print("PT acos %.3f"%tm(lambda:torch.acos(a)))
print("PT atan %.3f"%tm(lambda:torch.atan(a)))
print("PT expm1 %.3f"%tm(lambda:torch.expm1(a)))
print("PT log1p %.3f"%tm(lambda:torch.log1p(b)))
print("PT rsqrt %.3f"%tm(lambda:torch.rsqrt(b)))
print("PT digamma %.3f"%tm(lambda:torch.digamma(b)))
print("PT lgamma %.3f"%tm(lambda:torch.lgamma(b)))
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
    println!("elemmath ~16M f32 (torch 8t / FT default), min-of-7");
    for (lbl, w) in [
        ("add", 0u8),
        ("asin", 1),
        ("acos", 2),
        ("atan", 3),
        ("expm1", 4),
        ("log1p", 5),
        ("rsqrt", 6),
        ("digamma", 7),
        ("lgamma", 8),
    ] {
        let ft = tt(w);
        println!(
            "  {lbl:<9} FT {ft:8.3}  PT {:8.3}  => {}",
            g(lbl),
            v(ft, g(lbl))
        );
    }
    Ok(())
}

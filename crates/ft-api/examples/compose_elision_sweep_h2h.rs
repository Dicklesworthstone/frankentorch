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
    let tt = |w: u8| {
        let mut best = f64::INFINITY;
        for _ in 0..7 {
            let mut s = FrankenTorchSession::new(ExecutionMode::Strict);
            let x = s.tensor_variable_f32(a.clone(), vec![n], false).unwrap();
            let ti = Instant::now();
            match w {
                0 => {
                    let _ = s.tensor_add(x, x);
                }
                1 => {
                    let _ = s.tensor_special_ndtr(x);
                }
                2 => {
                    let _ = s.tensor_rad2deg(x);
                }
                3 => {
                    let _ = s.tensor_deg2rad(x);
                }
                4 => {
                    let _ = s.tensor_sinc(x);
                }
                5 => {
                    let _ = s.tensor_erfc(x);
                }
                _ => {
                    let _ = s.tensor_frac(x);
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
a=(((torch.arange(n,dtype=torch.int64)%9973).float()-5000.0)*0.01)
def tm(fn,reps=7):
    for _ in range(2): fn()
    ts=[]
    for _ in range(reps): s=time.perf_counter(); fn(); ts.append((time.perf_counter()-s)*1e3)
    return min(ts)
print("PT add %.3f"%tm(lambda:a+a))
print("PT ndtr %.3f"%tm(lambda:torch.special.ndtr(a)))
print("PT rad2deg %.3f"%tm(lambda:torch.rad2deg(a)))
print("PT deg2rad %.3f"%tm(lambda:torch.deg2rad(a)))
print("PT sinc %.3f"%tm(lambda:torch.sinc(a)))
print("PT erfc %.3f"%tm(lambda:torch.erfc(a)))
print("PT frac %.3f"%tm(lambda:torch.frac(a)))
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
    println!("compose-elision sweep ~16M f32 (torch 8t / FT default), min-of-7");
    for (lbl, w) in [
        ("add", 0u8),
        ("ndtr", 1),
        ("rad2deg", 2),
        ("deg2rad", 3),
        ("sinc", 4),
        ("erfc", 5),
        ("frac", 6),
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

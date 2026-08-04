// Binary-composite gap-finder vs torch (8t / FT default), find the worst ratio.
use ft_api::FrankenTorchSession;
use ft_core::ExecutionMode;
use std::io::Write;
use std::process::{Command, Stdio};
use std::time::Instant;
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let python = std::env::var("PYTORCH_PYTHON").unwrap_or_else(|_| "python3".to_string());
    let n = 16_000_000usize;
    let a: Vec<f32> = (0..n).map(|i| ((i % 4001) as f32 / 500.0) - 4.0).collect(); // (-4,4)
    let b: Vec<f32> = (0..n).map(|i| 0.1 + (i % 3997) as f32 / 500.0).collect(); // (0.1,8.1)
    let ops: &[(&str, u8)] = &[
        ("add", 0),
        ("hypot", 1),
        ("copysign", 2),
        ("xlogy", 3),
        ("logaddexp", 4),
        ("nextafter", 5),
        ("heaviside", 6),
        ("float_power", 7),
    ];
    let bench = |w: u8| {
        let mut best = f64::INFINITY;
        for _ in 0..7 {
            let mut s = FrankenTorchSession::new(ExecutionMode::Strict);
            let x = s.tensor_variable_f32(a.clone(), vec![n], false).unwrap();
            let y = s.tensor_variable_f32(b.clone(), vec![n], false).unwrap();
            let t = Instant::now();
            match w {
                0 => {
                    let _ = s.tensor_add(x, y);
                }
                1 => {
                    let _ = s.tensor_hypot(x, y);
                }
                2 => {
                    let _ = s.tensor_copysign(x, y);
                }
                3 => {
                    let _ = s.tensor_xlogy(x, y);
                }
                4 => {
                    let _ = s.tensor_logaddexp(x, y);
                }
                5 => {
                    let _ = s.tensor_nextafter(x, y);
                }
                6 => {
                    let _ = s.tensor_heaviside(x, y);
                }
                _ => {
                    let _ = s.tensor_float_power(x, 2.5);
                }
            }
            let e = t.elapsed().as_secs_f64() * 1e3;
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
a=((torch.arange(n,dtype=torch.int64)%4001).float()/500.0-4.0)
b=(0.1+(torch.arange(n,dtype=torch.int64)%3997).float()/500.0)
def tm(fn,reps=7):
    for _ in range(2): fn()
    ts=[]
    for _ in range(reps): s=time.perf_counter(); fn(); ts.append((time.perf_counter()-s)*1e3)
    return min(ts)
print("PT add %.3f"%tm(lambda:a+b))
print("PT hypot %.3f"%tm(lambda:torch.hypot(a,b)))
print("PT copysign %.3f"%tm(lambda:torch.copysign(a,b)))
print("PT xlogy %.3f"%tm(lambda:torch.xlogy(a,b)))
print("PT logaddexp %.3f"%tm(lambda:torch.logaddexp(a,b)))
print("PT nextafter %.3f"%tm(lambda:torch.nextafter(a,b)))
print("PT heaviside %.3f"%tm(lambda:torch.heaviside(a,b)))
print("PT float_power %.3f"%tm(lambda:torch.float_power(a,2.5)))
"#
    );
    let mut ch = Command::new(&python)
        .arg("-")
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .spawn()?;
    ch.stdin.as_mut().unwrap().write_all(py.as_bytes())?;
    let out = String::from_utf8_lossy(&ch.wait_with_output()?.stdout).to_string();
    let g = |k: &str| {
        out.lines()
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
    let vrb = |ft: f64, pp: f64| {
        if pp >= ft {
            format!("FT {:.2}x FASTER", pp / ft)
        } else {
            format!("FT {:.2}x SLOWER", ft / pp)
        }
    };
    println!("binop_gapfind ~16M f32 (torch 8t / FT default), min-of-7:");
    for &(lbl, w) in ops {
        let ft = bench(w);
        println!(
            "  {lbl:<12} FT {ft:8.3}  PT {:8.3}  => {}",
            g(lbl),
            vrb(ft, g(lbl))
        );
    }
    Ok(())
}

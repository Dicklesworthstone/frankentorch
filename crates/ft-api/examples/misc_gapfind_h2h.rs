// Misc composite-op gap-finder vs torch (8t / FT default). Inputs built OUTSIDE the timed region.
use ft_api::FrankenTorchSession;
use ft_core::ExecutionMode;
use std::io::Write;
use std::process::{Command, Stdio};
use std::time::Instant;
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let python = std::env::var("PYTORCH_PYTHON").unwrap_or_else(|_| "python3".to_string());
    let d = 4096usize; // d*d = ~16M
    let big: Vec<f32> = (0..d * d)
        .map(|i| 0.1 + (i % 3997) as f32 / 500.0)
        .collect();
    let vecd: Vec<f32> = (0..d).map(|i| 0.1 + (i % 997) as f32 / 100.0).collect();
    let a128: Vec<f32> = (0..128 * 128)
        .map(|i| 0.1 + (i % 311) as f32 / 50.0)
        .collect();
    let b32: Vec<f32> = (0..32 * 32).map(|i| 0.1 + (i % 97) as f32 / 30.0).collect();
    let bench = |w: u8| {
        let mut best = f64::INFINITY;
        for _ in 0..7 {
            let mut s = FrankenTorchSession::new(ExecutionMode::Strict);
            let t = match w {
                0 => {
                    let x = s
                        .tensor_variable_f32(big.clone(), vec![d, d], false)
                        .unwrap();
                    let ti = Instant::now();
                    let _ = s.tensor_add(x, x);
                    ti
                }
                1 => {
                    let y = s
                        .tensor_variable_f32(big.clone(), vec![d, d], false)
                        .unwrap();
                    let ti = Instant::now();
                    let _ = s.tensor_trapezoid(y, None, 1);
                    ti
                }
                2 => {
                    let y = s
                        .tensor_variable_f32(big.clone(), vec![d, d], false)
                        .unwrap();
                    let ti = Instant::now();
                    let _ = s.tensor_cumulative_trapezoid(y, None, 1);
                    ti
                }
                3 => {
                    let x = s
                        .tensor_variable_f32(big.clone(), vec![d, d], false)
                        .unwrap();
                    let ti = Instant::now();
                    let _ = s.tensor_renorm(x, 2.0, 0, 1.0);
                    ti
                }
                4 => {
                    let a = s
                        .tensor_variable_f32(a128.clone(), vec![128, 128], false)
                        .unwrap();
                    let b = s
                        .tensor_variable_f32(b32.clone(), vec![32, 32], false)
                        .unwrap();
                    let ti = Instant::now();
                    let _ = s.tensor_kron(a, b);
                    ti
                }
                _ => {
                    let x = s.tensor_variable_f32(vecd.clone(), vec![d], false).unwrap();
                    let ti = Instant::now();
                    let _ = s.tensor_diagflat(x, 0);
                    ti
                }
            };
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
d={d}
big=(0.1+(torch.arange(d*d,dtype=torch.int64)%3997).float()/500.0).reshape(d,d)
vecd=(0.1+(torch.arange(d,dtype=torch.int64)%997).float()/100.0)
a=(0.1+(torch.arange(128*128,dtype=torch.int64)%311).float()/50.0).reshape(128,128)
b=(0.1+(torch.arange(32*32,dtype=torch.int64)%97).float()/30.0).reshape(32,32)
def tm(fn,reps=7):
    for _ in range(2): fn()
    ts=[]
    for _ in range(reps): s=time.perf_counter(); fn(); ts.append((time.perf_counter()-s)*1e3)
    return min(ts)
def safe(name,fn):
    try: print("PT %s %.3f"%(name,tm(fn)))
    except Exception as e: print("PT %s NaN  # %s"%(name,type(e).__name__))
safe("add", lambda:big+big)
safe("trapezoid", lambda:torch.trapezoid(big,dim=1))
safe("cumtrapz", lambda:torch.cumulative_trapezoid(big,dim=1))
safe("renorm", lambda:torch.renorm(big,2,0,1.0))
safe("kron", lambda:torch.kron(a,b))
safe("diagflat", lambda:torch.diagflat(vecd))
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
    println!("misc_gapfind f32 (torch 8t / FT default), min-of-7:");
    for (lbl, w, key) in [
        ("add", 0u8, "add"),
        ("trapezoid", 1, "trapezoid"),
        ("cumtrapz", 2, "cumtrapz"),
        ("renorm", 3, "renorm"),
        ("kron", 4, "kron"),
        ("diagflat", 5, "diagflat"),
    ] {
        let ft = bench(w);
        println!(
            "  {lbl:<11} FT {ft:8.3}  PT {:8.3}  => {}",
            g(key),
            vrb(ft, g(key))
        );
    }
    Ok(())
}

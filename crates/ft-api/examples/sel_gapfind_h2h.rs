// Selection/search op gap-finder vs torch. Inputs built OUTSIDE the timed region.
use ft_api::FrankenTorchSession;
use ft_core::ExecutionMode;
use std::io::Write;
use std::process::{Command, Stdio};
use std::time::Instant;
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let python = std::env::var("PYTORCH_PYTHON").unwrap_or_else(|_| "python3".to_string());
    let n = 16_000_000usize;
    // limited range so unique has real work (~2000 distinct)
    let x: Vec<f32> = (0..n).map(|i| (i % 2000) as f32 * 0.5 - 500.0).collect();
    let bnd: Vec<f32> = (0..256).map(|i| i as f32 * 4.0 - 500.0).collect();
    let bench = |w: u8| {
        let mut best = f64::INFINITY;
        for _ in 0..7 {
            let mut s = FrankenTorchSession::new(ExecutionMode::Strict);
            let xt = s.tensor_variable_f32(x.clone(), vec![n], false).unwrap();
            let bt = s
                .tensor_variable_f32(bnd.clone(), vec![bnd.len()], false)
                .unwrap();
            let t = Instant::now();
            match w {
                0 => {
                    let _ = s.tensor_add(xt, xt);
                }
                1 => {
                    let _ = s.tensor_unique(xt, true, false, false);
                }
                2 => {
                    let _ = s.tensor_quantile(xt, 0.5);
                }
                _ => {
                    let _ = s.tensor_bucketize(xt, bt, false);
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
x=((torch.arange(n,dtype=torch.int64)%2000).float()*0.5-500.0)
bnd=(torch.arange(256,dtype=torch.int64).float()*4.0-500.0)
def tm(fn,reps=7):
    for _ in range(2): fn()
    ts=[]
    for _ in range(reps): s=time.perf_counter(); fn(); ts.append((time.perf_counter()-s)*1e3)
    return min(ts)
def safe(name,fn):
    try: print("PT %s %.3f"%(name,tm(fn)))
    except Exception as e: print("PT %s NaN  # %s"%(name,type(e).__name__))
safe("add", lambda:x+x)
safe("unique", lambda:torch.unique(x))
safe("quantile", lambda:torch.quantile(x,0.5))
safe("bucketize", lambda:torch.bucketize(x,bnd))
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
    println!("sel_gapfind ~16M f32 (torch 8t / FT default), min-of-7:");
    for (lbl, w) in [
        ("add", 0u8),
        ("unique", 1),
        ("quantile", 2),
        ("bucketize", 3),
    ] {
        let ft = bench(w);
        println!(
            "  {lbl:<11} FT {ft:8.3}  PT {:8.3}  => {}",
            g(lbl),
            vrb(ft, g(lbl))
        );
    }
    Ok(())
}

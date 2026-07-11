// NaN-family gap-finder vs torch (8t / FT default). Inputs built OUTSIDE the timed region.
use ft_api::FrankenTorchSession;
use ft_core::ExecutionMode;
use std::io::Write;
use std::process::{Command, Stdio};
use std::time::Instant;
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let python = std::env::var("PYTORCH_PYTHON").unwrap_or_else(|_| "python3".to_string());
    let n = 16_000_000usize;
    // ~1% NaN, ~0.5% +inf; positive-ish so nanprod doesn't underflow to 0 instantly
    let a: Vec<f32> = (0..n)
        .map(|i| {
            if i % 101 == 0 {
                f32::NAN
            } else if i % 211 == 0 {
                f32::INFINITY
            } else {
                0.9 + ((i % 4001) as f32 / 4000.0) * 0.2
            } // (0.9,1.1)
        })
        .collect();
    let bench = |w: u8| {
        let mut best = f64::INFINITY;
        for _ in 0..7 {
            let mut s = FrankenTorchSession::new(ExecutionMode::Strict);
            let x = s.tensor_variable_f32(a.clone(), vec![n], false).unwrap();
            let t = Instant::now();
            match w {
                0 => {
                    let _ = s.tensor_add(x, x);
                }
                1 => {
                    let _ = s.tensor_nanprod(x);
                }
                2 => {
                    let _ = s.tensor_nanstd(x, 1);
                }
                3 => {
                    let _ = s.tensor_nanvar(x, 1);
                }
                _ => {
                    let _ = s.tensor_nan_to_num(x, 0.0, None, None);
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
idx=torch.arange(n,dtype=torch.int64)
a=(0.9+((idx%4001).float()/4000.0)*0.2)
a[idx%101==0]=float('nan')
a[idx%211==0]=float('inf')
def tm(fn,reps=7):
    for _ in range(2): fn()
    ts=[]
    for _ in range(reps): s=time.perf_counter(); fn(); ts.append((time.perf_counter()-s)*1e3)
    return min(ts)
def safe(name, fn):
    try: print("PT %s %.3f"%(name, tm(fn)))
    except Exception as e: print("PT %s NaN  # %s"%(name, type(e).__name__))
safe("add", lambda:a+a)
# torch lacks nanprod/nanvar/nanstd; the fair user-equivalent is mask + reduction
safe("nanprod", lambda: torch.prod(torch.nan_to_num(a, nan=1.0)))
m=~torch.isnan(a)
safe("nanstd", lambda: torch.std(a[m]))
safe("nanvar", lambda: torch.var(a[m]))
safe("nan_to_num", lambda: torch.nan_to_num(a))
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
    println!(
        "nan_gapfind ~16M f32 (torch 8t / FT default), min-of-7 [nanstd/nanvar PT=nansum proxy]:"
    );
    for (lbl, w) in [
        ("add", 0u8),
        ("nanprod", 1),
        ("nanstd", 2),
        ("nanvar", 3),
        ("nan_to_num", 4),
    ] {
        let ft = bench(w);
        let key = if lbl == "nan_to_num" {
            "nan_to_num"
        } else {
            lbl
        };
        println!(
            "  {lbl:<11} FT {ft:8.3}  PT {:8.3}  => {}",
            g(key),
            vrb(ft, g(key))
        );
    }
    Ok(())
}

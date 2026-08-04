// Statistical/composite op gap-finder vs torch (8t / FT default).
use ft_api::FrankenTorchSession;
use ft_core::ExecutionMode;
use std::io::Write;
use std::process::{Command, Stdio};
use std::time::Instant;
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let python = std::env::var("PYTORCH_PYTHON").unwrap_or_else(|_| "python3".to_string());
    let n = 16_000_000usize;
    // spread with ~1% NaN for nan-ops
    let a: Vec<f32> = (0..n)
        .map(|i| {
            if i % 101 == 0 {
                f32::NAN
            } else {
                ((i % 4001) as f32 / 500.0) - 4.0
            }
        })
        .collect();
    let b: Vec<f32> = (0..n).map(|i| 0.1 + (i % 3997) as f32 / 500.0).collect();
    let rows = 100usize;
    let cols = n / rows; // 2D for corrcoef
    let bench = |w: u8| {
        let mut best = f64::INFINITY;
        for _ in 0..7 {
            let mut s = FrankenTorchSession::new(ExecutionMode::Strict);
            // Create inputs OUTSIDE the timed region (mirror act_gapfind) — otherwise the 16M
            // clone + tensor_variable materialization (~34ms) dwarfs the op being measured.
            let shape = if w == 5 { vec![rows, cols] } else { vec![n] };
            let src = if matches!(w, 1 | 2) {
                a.clone()
            } else {
                b.clone()
            };
            let x = s.tensor_variable_f32(src, shape, false).unwrap();
            let y = if w == 4 {
                Some(s.tensor_variable_f32(b.clone(), vec![n], false).unwrap())
            } else {
                None
            };
            let t = Instant::now();
            match w {
                0 => {
                    let _ = s.tensor_add(x, x);
                }
                1 => {
                    let _ = s.tensor_nansum(x);
                }
                2 => {
                    let _ = s.tensor_nanmean(x);
                }
                3 => {
                    let _ = s.tensor_diff(x, 1);
                }
                4 => {
                    let _ = s.tensor_dist(x, y.unwrap(), 2.0);
                }
                _ => {
                    let _ = s.tensor_corrcoef(x);
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
n={n}; rows={rows}; cols={cols}
a=((torch.arange(n,dtype=torch.int64)%4001).float()/500.0-4.0)
a[torch.arange(0,n,101)]=float('nan')
b=(0.1+(torch.arange(n,dtype=torch.int64)%3997).float()/500.0)
bb=b.reshape(rows,cols)
def tm(fn,reps=7):
    for _ in range(2): fn()
    ts=[]
    for _ in range(reps): s=time.perf_counter(); fn(); ts.append((time.perf_counter()-s)*1e3)
    return min(ts)
print("PT add %.3f"%tm(lambda:b+b))
print("PT nansum %.3f"%tm(lambda:torch.nansum(a)))
print("PT nanmean %.3f"%tm(lambda:torch.nanmean(a)))
print("PT diff %.3f"%tm(lambda:torch.diff(b)))
print("PT dist %.3f"%tm(lambda:torch.dist(b,b,2)))
print("PT corrcoef %.3f"%tm(lambda:torch.corrcoef(bb)))
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
    println!("stat_gapfind ~16M f32 (torch 8t / FT default), min-of-7:");
    for (lbl, w) in [
        ("add", 0u8),
        ("nansum", 1),
        ("nanmean", 2),
        ("diff", 3),
        ("dist", 4),
        ("corrcoef", 5),
    ] {
        let ft = bench(w);
        println!(
            "  {lbl:<10} FT {ft:8.3}  PT {:8.3}  => {}",
            g(lbl),
            vrb(ft, g(lbl))
        );
    }
    Ok(())
}

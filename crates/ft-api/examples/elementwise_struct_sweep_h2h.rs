//! Profiling sweep: materializing non-GEMM ops likely on the tape/apply_function path.
//! Find the biggest gap vs torch. All ~16M f32, torch 8t / FT default, min-of-7.
use ft_api::FrankenTorchSession;
use ft_core::ExecutionMode;
use std::io::Write;
use std::process::{Command, Stdio};
use std::time::Instant;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let python = std::env::var("PYTORCH_PYTHON").unwrap_or_else(|_| "python3".to_string());
    let n = 4000usize;
    let mat: Vec<f32> = (0..n * n)
        .map(|i| ((i % 9973) as f32 - 5000.0) * 0.01)
        .collect();
    let mat2: Vec<f32> = (0..n * n)
        .map(|i| ((i % 7919) as f32 - 4000.0) * 0.02)
        .collect();
    let cond: Vec<f32> = (0..n * n).map(|i| (i % 2) as f32).collect();
    let sorted: Vec<f32> = (0..16_000_000usize).map(|i| i as f32 * 0.5).collect();
    let queries: Vec<f32> = (0..16_000_000usize)
        .map(|i| ((i * 7919) % 16_000_000) as f32 * 0.5)
        .collect();
    let tt = |which: u8| {
        let mut best = f64::INFINITY;
        for _ in 0..7 {
            let mut s = FrankenTorchSession::new(ExecutionMode::Strict);
            let ti;
            match which {
                0 => {
                    let x = s
                        .tensor_variable_f32(mat.clone(), vec![n, n], false)
                        .unwrap();
                    ti = Instant::now();
                    let _ = s.tensor_add(x, x);
                }
                1 => {
                    let c = s
                        .tensor_variable_f32(cond.clone(), vec![n, n], false)
                        .unwrap();
                    let x = s
                        .tensor_variable_f32(mat.clone(), vec![n, n], false)
                        .unwrap();
                    let y = s
                        .tensor_variable_f32(mat2.clone(), vec![n, n], false)
                        .unwrap();
                    ti = Instant::now();
                    let _ = s.tensor_where(c, x, y);
                }
                2 => {
                    let x = s
                        .tensor_variable_f32(mat.clone(), vec![n, n], false)
                        .unwrap();
                    let m = s
                        .tensor_variable_f32(cond.clone(), vec![n, n], false)
                        .unwrap();
                    ti = Instant::now();
                    let _ = s.tensor_masked_fill(x, m, 0.0);
                }
                3 => {
                    let x = s
                        .tensor_variable_f32(mat.clone(), vec![n, n], false)
                        .unwrap();
                    ti = Instant::now();
                    let _ = s.tensor_diff(x, 1);
                }
                _ => {
                    let ss = s
                        .tensor_variable_f32(sorted.clone(), vec![16_000_000], false)
                        .unwrap();
                    let q = s
                        .tensor_variable_f32(queries.clone(), vec![16_000_000], false)
                        .unwrap();
                    ti = Instant::now();
                    let _ = s.tensor_searchsorted(ss, q, false);
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
import time, torch
torch.set_num_threads(8)
n={n}
mat=(((torch.arange(n*n,dtype=torch.int64)%9973).float()-5000.0)*0.01).reshape(n,n)
mat2=(((torch.arange(n*n,dtype=torch.int64)%7919).float()-4000.0)*0.02).reshape(n,n)
cond=(torch.arange(n*n,dtype=torch.int64)%2).reshape(n,n).bool()
sorted=(torch.arange(16000000,dtype=torch.int64).float()*0.5)
queries=(((torch.arange(16000000,dtype=torch.int64)*7919)%16000000).float()*0.5)
def tm(fn,reps=7):
    for _ in range(2): fn()
    ts=[]
    for _ in range(reps): s=time.perf_counter(); fn(); ts.append((time.perf_counter()-s)*1e3)
    return min(ts)
print("PT add %.3f"%tm(lambda:mat+mat))
print("PT where %.3f"%tm(lambda:torch.where(cond,mat,mat2)))
print("PT maskfill %.3f"%tm(lambda:mat.masked_fill(cond,0.0)))
print("PT diff %.3f"%tm(lambda:torch.diff(mat,1)))
print("PT searchsort %.3f"%tm(lambda:torch.searchsorted(sorted,queries)))
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
    println!("elementwise/struct sweep ~16M f32 (torch 8t / FT default), min-of-7");
    for (lbl, w) in [
        ("add", 0u8),
        ("where", 1),
        ("maskfill", 2),
        ("diff", 3),
        ("searchsort", 4),
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

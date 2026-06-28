//! 1-D sort-surface FT vs torch profile: sort / argsort / unique / median on a
//! large 1-D array. The single-lane case runs FT's SERIAL radix (no per-column
//! transpose-trick parallelism); this measures whether FT is slower than torch
//! (torch's 1-D sort is serial O(n log n)). add = anchor.
use ft_api::FrankenTorchSession;
use ft_core::ExecutionMode;
use std::io::Write;
use std::process::{Command, Stdio};
use std::time::Instant;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let python = std::env::var("PYTORCH_PYTHON").unwrap_or_else(|_| "python3".to_string());
    let n: usize = std::env::var("N").ok().and_then(|s| s.parse().ok()).unwrap_or(16_000_000);
    // deterministic pseudo-random NaN-free f64 data
    let data: Vec<f64> = (0..n)
        .map(|i| {
            let z = (i as u64).wrapping_mul(2862933555777941757).wrapping_add(3037000493);
            ((z >> 11) as f64 / (1u64 << 53) as f64) * 2.0 - 1.0
        })
        .collect();

    let data32: Vec<f32> = data.iter().map(|&v| v as f32).collect();
    // which: 0=add anchor 1=sort 2=argsort 3=unique 4=median (f64)
    let tt = |which: u8| {
        let mut best = f64::INFINITY;
        for _ in 0..5 {
            let mut s = FrankenTorchSession::new(ExecutionMode::Strict);
            let x = s.tensor_variable(data.clone(), vec![n], false).unwrap();
            let ti = Instant::now();
            match which {
                0 => { let _ = s.tensor_add(x, x); }
                1 => { let _ = s.tensor_sort(x, 0, false); }
                2 => { let _ = s.tensor_argsort(x, 0, false); }
                3 => { let _ = s.tensor_unique(x, true, false, false); }
                _ => { let _ = s.tensor_median(x); }
            }
            let e = ti.elapsed().as_secs_f64() * 1e3;
            if e < best { best = e; }
        }
        best
    };
    // f32: 1=sort 2=argsort
    let tt32 = |which: u8| {
        let mut best = f64::INFINITY;
        for _ in 0..5 {
            let mut s = FrankenTorchSession::new(ExecutionMode::Strict);
            let x = s.tensor_variable_f32(data32.clone(), vec![n], false).unwrap();
            let ti = Instant::now();
            match which {
                1 => { let _ = s.tensor_sort(x, 0, false); }
                _ => { let _ = s.tensor_argsort(x, 0, false); }
            }
            let e = ti.elapsed().as_secs_f64() * 1e3;
            if e < best { best = e; }
        }
        best
    };

    let py = format!(
        r#"
import time, torch
torch.set_num_threads(8)
n={n}
g=torch.Generator().manual_seed(0)
x=torch.rand(n,generator=g,dtype=torch.float64)*2-1
def tm(fn,reps=5):
    for _ in range(2): fn()
    ts=[]
    for _ in range(reps): s=time.perf_counter(); fn(); ts.append((time.perf_counter()-s)*1e3)
    return min(ts)
x32=x.float()
print("PT add %.3f"%tm(lambda:x+x))
print("PT sort %.3f"%tm(lambda:torch.sort(x)[0]))
print("PT argsort %.3f"%tm(lambda:torch.argsort(x)))
print("PT unique %.3f"%tm(lambda:torch.unique(x)))
print("PT median %.3f"%tm(lambda:torch.median(x)))
print("PT sort32 %.3f"%tm(lambda:torch.sort(x32)[0]))
print("PT argsort32 %.3f"%tm(lambda:torch.argsort(x32)))
"#,
        n = n
    );
    let mut ch = Command::new(&python).arg("-").stdin(Stdio::piped()).stdout(Stdio::piped()).spawn()?;
    ch.stdin.as_mut().unwrap().write_all(py.as_bytes())?;
    let pt = String::from_utf8_lossy(&ch.wait_with_output()?.stdout).to_string();
    let g = |k: &str| pt.lines().find_map(|l| { let mut it = l.strip_prefix("PT ")?.split_whitespace(); if it.next()? == k { it.next()?.parse::<f64>().ok() } else { None } }).unwrap_or(f64::NAN);
    let v = |ft: f64, p: f64| if p >= ft { format!("FT {:.2}x FASTER", p / ft) } else { format!("FT {:.2}x SLOWER", ft / p) };
    println!("1-D sort surface, N={n} f64 (torch 8t / FT default cores), min-of-5");
    for (lbl, w) in [("add", 0u8), ("sort", 1), ("argsort", 2), ("unique", 3), ("median", 4)] {
        let ft = tt(w);
        println!("  {lbl:<9} FT {ft:9.3}  PT {:9.3}  => {}", g(lbl), v(ft, g(lbl)));
    }
    for (lbl, key, w) in [("sort_f32", "sort32", 1u8), ("argsort_f32", "argsort32", 2)] {
        let ft = tt32(w);
        println!("  {lbl:<11} FT {ft:9.3}  PT {:9.3}  => {}", g(key), v(ft, g(key)));
    }
    Ok(())
}

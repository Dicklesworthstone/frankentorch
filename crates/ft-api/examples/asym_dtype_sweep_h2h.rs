//! Sweep ops that route f32 through the slow tensor_values_lossy_f64 generic path.
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
    let b: Vec<f32> = (0..n)
        .map(|i| ((i % 7919) as f32 - 4000.0) * 0.02)
        .collect();
    let mask: Vec<f32> = (0..n).map(|i| (i % 2) as f32).collect();
    let test: Vec<f32> = (0..1000).map(|i| (i as f32 - 500.0) * 0.01).collect();
    let mat: Vec<f32> = (0..4096 * 4096)
        .map(|i| ((i * 7919) % 1_000_003) as f32 * 0.001)
        .collect();
    let tt = |which: u8| {
        let mut best = f64::INFINITY;
        for _ in 0..7 {
            let mut s = FrankenTorchSession::new(ExecutionMode::Strict);
            let ti;
            match which {
                0 => {
                    let x = s.tensor_variable_f32(a.clone(), vec![n], false).unwrap();
                    ti = Instant::now();
                    let _ = s.tensor_add(x, x);
                }
                1 => {
                    let x = s.tensor_variable_f32(a.clone(), vec![n], false).unwrap();
                    ti = Instant::now();
                    let _ = s.tensor_nanmedian(x);
                }
                2 => {
                    let m = s.tensor_variable_f32(a.clone(), vec![n], false).unwrap();
                    let sg = s.tensor_variable_f32(b.clone(), vec![n], false).unwrap();
                    ti = Instant::now();
                    let _ = s.tensor_copysign(m, sg);
                }
                3 => {
                    let x = s.tensor_variable_f32(a.clone(), vec![n], false).unwrap();
                    let y = s.tensor_variable_f32(b.clone(), vec![n], false).unwrap();
                    ti = Instant::now();
                    let _ = s.tensor_logaddexp(x, y);
                }
                4 => {
                    let e = s.tensor_variable_f32(a.clone(), vec![n], false).unwrap();
                    let t = s
                        .tensor_variable_f32(test.clone(), vec![1000], false)
                        .unwrap();
                    ti = Instant::now();
                    let _ = s.tensor_isin(e, t);
                }
                5 => {
                    let x = s
                        .tensor_variable_f32(mat.clone(), vec![4096, 4096], false)
                        .unwrap();
                    ti = Instant::now();
                    let _ = s.tensor_cummax_dim(x, 1);
                }
                _ => {
                    let x = s.tensor_variable_f32(a.clone(), vec![n], false).unwrap();
                    let m = s.tensor_variable_f32(mask.clone(), vec![n], false).unwrap();
                    ti = Instant::now();
                    let _ = s.tensor_masked_select(x, m);
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
a=(((torch.arange(n,dtype=torch.int64)%9973).float()-5000.0)*0.01)
b=(((torch.arange(n,dtype=torch.int64)%7919).float()-4000.0)*0.02)
mask=(torch.arange(n,dtype=torch.int64)%2).bool()
test=((torch.arange(1000,dtype=torch.int64).float()-500.0)*0.01)
mat=(((torch.arange(4096*4096,dtype=torch.int64)*7919)%1000003).float()*0.001).reshape(4096,4096)
def tm(fn,reps=7):
    for _ in range(2): fn()
    ts=[]
    for _ in range(reps): s=time.perf_counter(); fn(); ts.append((time.perf_counter()-s)*1e3)
    return min(ts)
print("PT add %.3f"%tm(lambda:a+a))
print("PT nanmedian %.3f"%tm(lambda:torch.nanmedian(a)))
print("PT copysign %.3f"%tm(lambda:torch.copysign(a,b)))
print("PT logaddexp %.3f"%tm(lambda:torch.logaddexp(a,b)))
print("PT isin %.3f"%tm(lambda:torch.isin(a,test)))
print("PT cummaxdim %.3f"%tm(lambda:torch.cummax(mat,1)))
print("PT maskselect %.3f"%tm(lambda:torch.masked_select(a,mask)))
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
    println!("asym-dtype sweep ~16M f32 (torch 8t / FT default), min-of-7");
    for (lbl, w) in [
        ("add", 0u8),
        ("nanmedian", 1),
        ("copysign", 2),
        ("logaddexp", 3),
        ("isin", 4),
        ("cummaxdim", 5),
        ("maskselect", 6),
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

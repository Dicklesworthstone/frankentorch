//! margin_ranking_loss fused fast path vs torch (f32 was ERRORING; f64 had no fast path).
//! per-element: max(0, margin - t*(x1-x2)). margin=0.5. relu = anchor.
use ft_api::FrankenTorchSession;
use ft_core::ExecutionMode;
use std::io::Write;
use std::process::{Command, Stdio};
use std::time::Instant;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let python = std::env::var("PYTORCH_PYTHON").unwrap_or_else(|_| "python3".to_string());
    let margin = 0.5_f64;

    // parity reduction='none' (bit-exact per-element), both dtypes
    let x1: Vec<f64> = vec![1.0, 2.0, -1.0, 0.5, 3.0, -2.0, 0.7, 1.2];
    let x2: Vec<f64> = vec![0.5, 2.5, -0.5, 1.0, 1.0, -1.0, 1.0, 0.2];
    let tg: Vec<f64> = vec![1.0, -1.0, 1.0, -1.0, 1.0, 1.0, -1.0, 1.0];
    let py_s = format!(
        r#"
import torch
def run(dt):
    x1=torch.tensor({x1:?},dtype=dt); x2=torch.tensor({x2:?},dtype=dt); t=torch.tensor({tg:?},dtype=dt)
    o=torch.nn.functional.margin_ranking_loss(x1,x2,t,margin={margin},reduction='none')
    return o
o64=run(torch.float64); o32=run(torch.float32)
print("V64"," ".join("%.17g"%v for v in o64.tolist()))
print("V32"," ".join("%.9g"%v for v in o32.tolist()))
"#,
        x1 = x1,
        x2 = x2,
        tg = tg,
        margin = margin
    );
    let mut ch = Command::new(&python)
        .arg("-")
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .spawn()?;
    ch.stdin.as_mut().unwrap().write_all(py_s.as_bytes())?;
    let pt = String::from_utf8_lossy(&ch.wait_with_output()?.stdout).to_string();
    let parse = |k: &str| -> Vec<f64> {
        pt.lines()
            .find_map(|l| l.strip_prefix(k))
            .map(|s| {
                s.split_whitespace()
                    .filter_map(|t| t.parse().ok())
                    .collect()
            })
            .unwrap_or_default()
    };
    let (p64, p32) = (parse("V64 "), parse("V32 "));
    let mut s = FrankenTorchSession::new(ExecutionMode::Strict);
    // f64
    let a = s.tensor_variable(x1.clone(), vec![8], false)?;
    let b = s.tensor_variable(x2.clone(), vec![8], false)?;
    let t = s.tensor_variable(tg.clone(), vec![8], false)?;
    let o = s.tensor_margin_ranking_loss(a, b, t, margin, "none")?;
    let f64v = s.tensor_values(o)?;
    let mm64 = f64v
        .iter()
        .zip(&p64)
        .filter(|(a, b)| a.to_bits() != b.to_bits())
        .count()
        + f64v.len().abs_diff(p64.len());
    // f32 (was erroring)
    let a2 = s.tensor_variable_f32(x1.iter().map(|&v| v as f32).collect(), vec![8], false)?;
    let b2 = s.tensor_variable_f32(x2.iter().map(|&v| v as f32).collect(), vec![8], false)?;
    let t2 = s.tensor_variable_f32(tg.iter().map(|&v| v as f32).collect(), vec![8], false)?;
    let o2 = s.tensor_margin_ranking_loss(a2, b2, t2, margin, "none")?;
    let dt2 = s.tensor_dtype(o2)?;
    let f32v = s.tensor_values_lossy_f64(o2)?;
    let mm32 = f32v
        .iter()
        .zip(&p32)
        .filter(|(a, b)| (**a as f32).to_bits() != (**b as f32).to_bits())
        .count()
        + f32v.len().abs_diff(p32.len());
    println!(
        "parity: f64 {mm64}/{} | f32 dtype={dt2:?} (was ERROR) {mm32}/{}",
        p64.len(),
        p32.len()
    );

    // perf 16M f32, reduction='mean'
    let n = 16_000_000usize;
    let a: Vec<f32> = (0..n)
        .map(|i| ((i % 9973) as f32 - 5000.0) * 0.001)
        .collect();
    let bb: Vec<f32> = (0..n)
        .map(|i| ((i % 7919) as f32 - 4000.0) * 0.001)
        .collect();
    let tb: Vec<f32> = (0..n)
        .map(|i| if i % 2 == 0 { 1.0 } else { -1.0 })
        .collect();
    let tt = |which: u8| {
        let mut bst = f64::INFINITY;
        for _ in 0..9 {
            let mut s = FrankenTorchSession::new(ExecutionMode::Strict);
            let x = s.tensor_variable_f32(a.clone(), vec![n], false).unwrap();
            let y = s.tensor_variable_f32(bb.clone(), vec![n], false).unwrap();
            let t = s.tensor_variable_f32(tb.clone(), vec![n], false).unwrap();
            let ti = Instant::now();
            let _ = if which == 0 {
                s.tensor_relu(x)
            } else {
                s.tensor_margin_ranking_loss(x, y, t, margin, "mean")
            };
            let e = ti.elapsed().as_secs_f64() * 1e3;
            if e < bst {
                bst = e;
            }
        }
        bst
    };
    let (tr, tm) = (tt(0), tt(1));
    let py_b = format!(
        r#"
import time,torch
torch.set_num_threads(8)
n={n}
x=(((torch.arange(n,dtype=torch.int64)%9973).float()-5000.0)*0.001)
y=(((torch.arange(n,dtype=torch.int64)%7919).float()-4000.0)*0.001)
t=torch.where((torch.arange(n)%2)==0,1.0,-1.0).float()
def tm(fn,reps=9):
    for _ in range(2): fn()
    ts=[]
    for _ in range(reps): s=time.perf_counter(); fn(); ts.append((time.perf_counter()-s)*1e3)
    return min(ts)
print("PT relu %.4f"%tm(lambda:torch.relu(x)))
print("PT mr %.4f"%tm(lambda:torch.nn.functional.margin_ranking_loss(x,y,t,margin={margin},reduction='mean')))
"#,
        n = n,
        margin = margin
    );
    let mut ch = Command::new(&python)
        .arg("-")
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .spawn()?;
    ch.stdin.as_mut().unwrap().write_all(py_b.as_bytes())?;
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
    println!(
        "  relu_anchor  FT {tr:.3} PT {:.3}  => {}",
        g("relu"),
        v(tr, g("relu"))
    );
    println!(
        "  margin_rank  FT {tm:.3} PT {:.3}  => {}",
        g("mr"),
        v(tm, g("mr"))
    );
    Ok(())
}

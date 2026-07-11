//! Probe the f32->f64->f32 typed-dispatch round-trip (the softmax bug, 56765cdd) on
//! OTHER large-output ops: reduction along a tiny dim (large output + fast kernel),
//! elementwise lerp, and cat. add = anchor.
use ft_api::FrankenTorchSession;
use ft_core::ExecutionMode;
use std::io::Write;
use std::process::{Command, Stdio};
use std::time::Instant;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let python = std::env::var("PYTORCH_PYTHON").unwrap_or_else(|_| "python3".to_string());
    let (r, c) = (4096usize, 2048usize);
    let n3 = r * c * 2;
    let d3: Vec<f32> = (0..n3)
        .map(|i| ((i % 9973) as f32 - 5000.0) * 0.001)
        .collect();
    let n1 = 16_000_000usize;
    let a1: Vec<f32> = (0..n1)
        .map(|i| ((i % 9973) as f32 - 5000.0) * 0.001)
        .collect();
    let b1: Vec<f32> = (0..n1)
        .map(|i| ((i % 7919) as f32 - 4000.0) * 0.001)
        .collect();

    let tt = |which: u8| {
        let mut best = f64::INFINITY;
        for _ in 0..7 {
            let mut s = FrankenTorchSession::new(ExecutionMode::Strict);
            let ti;
            match which {
                0 => {
                    let x = s.tensor_variable_f32(a1.clone(), vec![n1], false).unwrap();
                    let y = s.tensor_variable_f32(b1.clone(), vec![n1], false).unwrap();
                    ti = Instant::now();
                    let _ = s.tensor_add(x, y);
                }
                1 => {
                    let x = s
                        .tensor_variable_f32(d3.clone(), vec![r, c, 2], false)
                        .unwrap();
                    ti = Instant::now();
                    let _ = s.tensor_var_dim(x, 2, 1);
                }
                2 => {
                    let x = s
                        .tensor_variable_f32(d3.clone(), vec![r, c, 2], false)
                        .unwrap();
                    ti = Instant::now();
                    let _ = s.tensor_mean_dim(x, 2);
                }
                3 => {
                    let x = s.tensor_variable_f32(a1.clone(), vec![n1], false).unwrap();
                    let y = s.tensor_variable_f32(b1.clone(), vec![n1], false).unwrap();
                    ti = Instant::now();
                    let _ = s.tensor_lerp(x, y, 0.3);
                }
                _ => {
                    let x = s
                        .tensor_variable_f32(a1[..8_000_000].to_vec(), vec![8_000_000], false)
                        .unwrap();
                    let y = s
                        .tensor_variable_f32(b1[..8_000_000].to_vec(), vec![8_000_000], false)
                        .unwrap();
                    ti = Instant::now();
                    let _ = s.tensor_cat(&[x, y], 0);
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
r,c={r},{c}
x3=(((torch.arange(r*c*2,dtype=torch.int64)%9973).float()-5000.0)*0.001).reshape(r,c,2)
n1={n1}
a=(((torch.arange(n1,dtype=torch.int64)%9973).float()-5000.0)*0.001)
b=(((torch.arange(n1,dtype=torch.int64)%7919).float()-4000.0)*0.001)
def tm(fn,reps=7):
    for _ in range(2): fn()
    ts=[]
    for _ in range(reps): s=time.perf_counter(); fn(); ts.append((time.perf_counter()-s)*1e3)
    return min(ts)
print("PT add %.3f"%tm(lambda:a+b))
print("PT var_dim2 %.3f"%tm(lambda:torch.var(x3,dim=2,keepdim=True,unbiased=True)))
print("PT mean_dim2 %.3f"%tm(lambda:torch.mean(x3,dim=2)))
print("PT lerp %.3f"%tm(lambda:torch.lerp(a,b,0.3)))
print("PT cat %.3f"%tm(lambda:torch.cat([a[:8000000],b[:8000000]],0)))
"#,
        r = r,
        c = c,
        n1 = n1
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
    println!("f32 round-trip probe (torch 8t / FT default), min-of-7");
    for (lbl, w) in [
        ("add", 0u8),
        ("var_dim2", 1),
        ("mean_dim2", 2),
        ("lerp", 3),
        ("cat", 4),
    ] {
        let ft = tt(w);
        println!(
            "  {lbl:<10} FT {ft:8.3}  PT {:8.3}  => {}",
            g(lbl),
            v(ft, g(lbl))
        );
    }
    Ok(())
}

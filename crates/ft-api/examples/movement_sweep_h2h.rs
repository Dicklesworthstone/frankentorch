//! Broad movement-op H2H sweep vs torch to find the biggest remaining gap.
//! All f32, ~16M outputs, min-of-7, torch 8t / FT default cores.
use ft_api::FrankenTorchSession;
use ft_core::ExecutionMode;
use std::io::Write;
use std::process::{Command, Stdio};
use std::time::Instant;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let python = std::env::var("PYTORCH_PYTHON").unwrap_or_else(|_| "python3".to_string());
    let n = 4000usize; // 4000x4000 = 16M
    let mat: Vec<f32> = (0..n * n)
        .map(|i| ((i % 9973) as f32 - 5000.0) * 0.01)
        .collect();
    let vec1: Vec<f32> = (0..16_000_000usize)
        .map(|i| (i % 9973) as f32 * 0.1)
        .collect();
    let vec4: Vec<f32> = (0..4_000_000usize)
        .map(|i| (i % 9973) as f32 * 0.1)
        .collect();
    let idx0: Vec<f64> = (0..n).map(|i| ((i * 7919) % n) as f64).collect();
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
                    let x = s
                        .tensor_variable_f32(mat.clone(), vec![n, n], false)
                        .unwrap();
                    ti = Instant::now();
                    let _ = s.tensor_tril(x, 0);
                }
                2 => {
                    let x = s
                        .tensor_variable_f32(mat.clone(), vec![n, n], false)
                        .unwrap();
                    ti = Instant::now();
                    let _ = s.tensor_triu(x, 0);
                }
                3 => {
                    let x = s
                        .tensor_variable_f32(vec1.clone(), vec![16_000_000], false)
                        .unwrap();
                    ti = Instant::now();
                    let _ = s.tensor_roll(x, 12345, 0);
                }
                4 => {
                    let x = s
                        .tensor_variable_f32(vec1.clone(), vec![16_000_000], false)
                        .unwrap();
                    ti = Instant::now();
                    let _ = s.tensor_flip(x, &[0]);
                }
                5 => {
                    let x = s
                        .tensor_variable_f32(vec4.clone(), vec![4_000_000], false)
                        .unwrap();
                    ti = Instant::now();
                    let _ = s.tensor_repeat_interleave(x, 4);
                }
                _ => {
                    let x = s
                        .tensor_variable_f32(mat.clone(), vec![n, n], false)
                        .unwrap();
                    let ix = s.tensor_variable(idx0.clone(), vec![n], false).unwrap();
                    ti = Instant::now();
                    let _ = s.tensor_index_select(x, 0, ix);
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
vec1=((torch.arange(16000000,dtype=torch.int64)%9973).float()*0.1)
vec4=((torch.arange(4000000,dtype=torch.int64)%9973).float()*0.1)
idx0=((torch.arange(n,dtype=torch.int64)*7919)%n)
def tm(fn,reps=7):
    for _ in range(2): fn()
    ts=[]
    for _ in range(reps): s=time.perf_counter(); fn(); ts.append((time.perf_counter()-s)*1e3)
    return min(ts)
print("PT add %.3f"%tm(lambda:mat+mat))
print("PT tril %.3f"%tm(lambda:torch.tril(mat,0)))
print("PT triu %.3f"%tm(lambda:torch.triu(mat,0)))
print("PT roll %.3f"%tm(lambda:torch.roll(vec1,12345,0)))
print("PT flip %.3f"%tm(lambda:torch.flip(vec1,[0])))
print("PT rinterleave %.3f"%tm(lambda:vec4.repeat_interleave(4)))
print("PT idxsel %.3f"%tm(lambda:torch.index_select(mat,0,idx0)))
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
    println!("movement sweep ~16M f32 (torch 8t / FT default), min-of-7");
    for (lbl, w) in [
        ("add", 0u8),
        ("tril", 1),
        ("triu", 2),
        ("roll", 3),
        ("flip", 4),
        ("rinterleave", 5),
        ("idxsel", 6),
    ] {
        let ft = tt(w);
        println!(
            "  {lbl:<12} FT {ft:8.3}  PT {:8.3}  => {}",
            g(lbl),
            v(ft, g(lbl))
        );
    }
    Ok(())
}

//! cross product f32 fused fast path vs torch. f64 had a fused per-row path; f32 fell to the
//! composed broadcast+narrow/mul/sub/cat path (~17x slower per the f64 ledger). add = anchor.
use ft_api::FrankenTorchSession;
use ft_core::ExecutionMode;
use std::io::Write;
use std::process::{Command, Stdio};
use std::time::Instant;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let python = std::env::var("PYTORCH_PYTHON").unwrap_or_else(|_| "python3".to_string());

    // parity [N=4, 3] cross along last dim
    let nn = 4usize;
    let a: Vec<f32> = (0..nn * 3).map(|i| ((i % 7) as f32 - 3.0) * 0.5).collect();
    let b: Vec<f32> = (0..nn * 3)
        .map(|i| ((i % 5) as f32 - 2.0) * 0.4 + 0.1)
        .collect();
    let py_s = format!(
        r#"
import torch
a=torch.tensor({a:?},dtype=torch.float32).reshape({nn},3)
b=torch.tensor({b:?},dtype=torch.float32).reshape({nn},3)
o32=torch.linalg.cross(a,b,dim=-1)
o64=torch.linalg.cross(a.double(),b.double(),dim=-1)
print("V32"," ".join("%.9g"%v for v in o32.flatten().tolist()))
print("V64"," ".join("%.17g"%v for v in o64.flatten().tolist()))
"#,
        a = a,
        b = b,
        nn = nn
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
    let (p32, p64) = (parse("V32 "), parse("V64 "));
    let mut s = FrankenTorchSession::new(ExecutionMode::Strict);
    let av = s.tensor_variable_f32(a.clone(), vec![nn, 3], false)?;
    let bv = s.tensor_variable_f32(b.clone(), vec![nn, 3], false)?;
    let o = s.tensor_cross(av, bv)?;
    let dt = s.tensor_dtype(o)?;
    let fv = s.tensor_values_lossy_f64(o)?;
    let mm32 = fv
        .iter()
        .zip(&p32)
        .filter(|(x, y)| (**x as f32).to_bits() != (**y as f32).to_bits())
        .count();
    let a64: Vec<f64> = a.iter().map(|&v| v as f64).collect();
    let b64: Vec<f64> = b.iter().map(|&v| v as f64).collect();
    let av6 = s.tensor_variable(a64, vec![nn, 3], false)?;
    let bv6 = s.tensor_variable(b64, vec![nn, 3], false)?;
    let o6 = s.tensor_cross(av6, bv6)?;
    let fv6 = s.tensor_values(o6)?;
    let mm64 = fv6
        .iter()
        .zip(&p64)
        .filter(|(x, y)| x.to_bits() != y.to_bits())
        .count();
    println!(
        "parity: f32 dtype={dt:?} bitexact_mismatch={mm32}/{} | f64 mismatch={mm64}/{}",
        p32.len(),
        p64.len()
    );

    // perf [N=2M, 3], f32 + f64
    let n = 2_000_000usize;
    let af: Vec<f32> = (0..n * 3)
        .map(|i| ((i % 9973) as f32 - 5000.0) * 0.001)
        .collect();
    let bf: Vec<f32> = (0..n * 3)
        .map(|i| ((i % 7919) as f32 - 4000.0) * 0.001)
        .collect();
    let a6: Vec<f64> = af.iter().map(|&v| v as f64).collect();
    let b6: Vec<f64> = bf.iter().map(|&v| v as f64).collect();
    let tt = |which: u8| {
        let mut bst = f64::INFINITY;
        for _ in 0..9 {
            let mut s = FrankenTorchSession::new(ExecutionMode::Strict);
            let e = match which {
                0 => {
                    let x = s
                        .tensor_variable_f32(af.clone(), vec![n, 3], false)
                        .unwrap();
                    let y = s
                        .tensor_variable_f32(bf.clone(), vec![n, 3], false)
                        .unwrap();
                    let ti = Instant::now();
                    let _ = s.tensor_add(x, y);
                    ti.elapsed().as_secs_f64() * 1e3
                }
                1 => {
                    let x = s
                        .tensor_variable_f32(af.clone(), vec![n, 3], false)
                        .unwrap();
                    let y = s
                        .tensor_variable_f32(bf.clone(), vec![n, 3], false)
                        .unwrap();
                    let ti = Instant::now();
                    let _ = s.tensor_cross(x, y);
                    ti.elapsed().as_secs_f64() * 1e3
                }
                _ => {
                    let x = s.tensor_variable(a6.clone(), vec![n, 3], false).unwrap();
                    let y = s.tensor_variable(b6.clone(), vec![n, 3], false).unwrap();
                    let ti = Instant::now();
                    let _ = s.tensor_cross(x, y);
                    ti.elapsed().as_secs_f64() * 1e3
                }
            };
            if e < bst {
                bst = e;
            }
        }
        bst
    };
    let (tadd, tf32, tf64) = (tt(0), tt(1), tt(2));
    let py_b = format!(
        r#"
import time,torch
torch.set_num_threads(8)
n={n}
a=(((torch.arange(n*3,dtype=torch.int64)%9973).float()-5000.0)*0.001).reshape(n,3)
b=(((torch.arange(n*3,dtype=torch.int64)%7919).float()-4000.0)*0.001).reshape(n,3)
ad=a.double(); bd=b.double()
def tm(fn,reps=9):
    for _ in range(2): fn()
    ts=[]
    for _ in range(reps): s=time.perf_counter(); fn(); ts.append((time.perf_counter()-s)*1e3)
    return min(ts)
print("PT add %.4f"%tm(lambda:a+b))
print("PT cr32 %.4f"%tm(lambda:torch.linalg.cross(a,b,dim=-1)))
print("PT cr64 %.4f"%tm(lambda:torch.linalg.cross(ad,bd,dim=-1)))
"#,
        n = n
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
        "  add_anchor FT {tadd:.3} PT {:.3}  => {}",
        g("add"),
        v(tadd, g("add"))
    );
    println!(
        "  cross_f32  FT {tf32:.3} PT {:.3}  => {}",
        g("cr32"),
        v(tf32, g("cr32"))
    );
    println!(
        "  cross_f64  FT {tf64:.3} PT {:.3}  => {}",
        g("cr64"),
        v(tf64, g("cr64"))
    );
    Ok(())
}

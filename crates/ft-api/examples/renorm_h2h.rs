//! renorm f32 dim==0 fused fast path vs torch. f64 had a fused dim0 path; f32 fell to the
//! composed permute+powf path (~2.4x slower than torch per the f64 ledger). add = anchor.
use ft_api::FrankenTorchSession;
use ft_core::ExecutionMode;
use std::io::Write;
use std::process::{Command, Stdio};
use std::time::Instant;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let python = std::env::var("PYTORCH_PYTHON").unwrap_or_else(|_| "python3".to_string());
    let (p, maxnorm) = (2.0_f64, 3.0_f64);

    // parity [N=6, M=5] dim=0 (each row renormed if ||row||_p > maxnorm)
    let (nn, mm) = (6usize, 5usize);
    let x: Vec<f32> = (0..nn * mm)
        .map(|i| ((i % 11) as f32 - 4.0) * 0.7)
        .collect();
    let py_s = format!(
        r#"
import torch
x=torch.tensor({x:?},dtype=torch.float32).reshape({nn},{mm})
o32=torch.renorm(x,{p},0,{maxnorm})
o64=torch.renorm(x.double(),{p},0,{maxnorm})
print("V32"," ".join("%.9g"%v for v in o32.flatten().tolist()))
print("V64"," ".join("%.17g"%v for v in o64.flatten().tolist()))
"#,
        x = x,
        nn = nn,
        mm = mm,
        p = p,
        maxnorm = maxnorm
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
    let a = s.tensor_variable_f32(x.clone(), vec![nn, mm], false)?;
    let o = s.tensor_renorm(a, p, 0, maxnorm)?;
    let dt = s.tensor_dtype(o)?;
    let fv = s.tensor_values_lossy_f64(o)?;
    let mr32 = fv
        .iter()
        .zip(&p32)
        .map(|(u, w)| (u - w).abs() / w.abs().max(1e-6))
        .fold(0.0f64, f64::max);
    let x64: Vec<f64> = x.iter().map(|&v| v as f64).collect();
    let a6 = s.tensor_variable(x64, vec![nn, mm], false)?;
    let o6 = s.tensor_renorm(a6, p, 0, maxnorm)?;
    let fv6 = s.tensor_values(o6)?;
    let mr64 = fv6
        .iter()
        .zip(&p64)
        .map(|(u, w)| (u - w).abs() / w.abs().max(1e-6))
        .fold(0.0f64, f64::max);
    println!(
        "parity: f32 dtype={dt:?} max_rel={mr32:.2e} | f64 max_rel={mr64:.2e} (tol<1e-5: {})",
        mr32 < 1e-5 && mr64 < 1e-5
    );

    // perf [N=200k, M=128] dim=0, f32
    let (n, m) = (200_000usize, 128usize);
    let xf: Vec<f32> = (0..n * m)
        .map(|i| ((i % 9973) as f32 - 5000.0) * 0.01)
        .collect();
    let tt = |which: u8| {
        let mut bst = f64::INFINITY;
        for _ in 0..9 {
            let mut s = FrankenTorchSession::new(ExecutionMode::Strict);
            let x = s
                .tensor_variable_f32(xf.clone(), vec![n, m], false)
                .unwrap();
            let e = if which == 0 {
                let y = s
                    .tensor_variable_f32(xf.clone(), vec![n, m], false)
                    .unwrap();
                let ti = Instant::now();
                let _ = s.tensor_add(x, y);
                ti.elapsed().as_secs_f64() * 1e3
            } else {
                let ti = Instant::now();
                let _ = s.tensor_renorm(x, p, 0, maxnorm);
                ti.elapsed().as_secs_f64() * 1e3
            };
            if e < bst {
                bst = e;
            }
        }
        bst
    };
    let (tadd, tr) = (tt(0), tt(1));
    let py_b = format!(
        r#"
import time,torch
torch.set_num_threads(8)
n={n}; m={m}; p={p}; maxnorm={maxnorm}
x=(((torch.arange(n*m,dtype=torch.int64)%9973).float()-5000.0)*0.01).reshape(n,m)
def tm(fn,reps=9):
    for _ in range(2): fn()
    ts=[]
    for _ in range(reps): s=time.perf_counter(); fn(); ts.append((time.perf_counter()-s)*1e3)
    return min(ts)
print("PT add %.4f"%tm(lambda:x+x))
print("PT rn %.4f"%tm(lambda:torch.renorm(x,p,0,maxnorm)))
"#,
        n = n,
        m = m,
        p = p,
        maxnorm = maxnorm
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
        "  add_anchor   FT {tadd:.3} PT {:.3}  => {}",
        g("add"),
        v(tadd, g("add"))
    );
    println!(
        "  renorm_f32   FT {tr:.3} PT {:.3}  => {}",
        g("rn"),
        v(tr, g("rn"))
    );
    Ok(())
}

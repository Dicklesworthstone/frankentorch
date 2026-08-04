//! cosine_embedding_loss f32 fused fast path vs torch. Was: f32 ERRORED (tensor_eq target[F32]
//! vs ones[F64] dtype mismatch). add = anchor. dim=last. y in {+1,-1}, margin.
use ft_api::FrankenTorchSession;
use ft_core::ExecutionMode;
use std::io::Write;
use std::process::{Command, Stdio};
use std::time::Instant;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let python = std::env::var("PYTORCH_PYTHON").unwrap_or_else(|_| "python3".to_string());
    let margin = 0.3_f64;

    // parity [N=6,D=5] reduction='none', f32 + f64
    let (nn, dd) = (6usize, 5usize);
    let x1: Vec<f32> = (0..nn * dd).map(|i| ((i % 7) as f32 - 3.0) * 0.5).collect();
    let x2: Vec<f32> = (0..nn * dd)
        .map(|i| ((i % 5) as f32 - 2.0) * 0.4 + 0.1)
        .collect();
    let yv: Vec<f32> = (0..nn)
        .map(|i| if i % 2 == 0 { 1.0 } else { -1.0 })
        .collect();
    let py_s = format!(
        r#"
import torch
x1=torch.tensor({x1:?},dtype=torch.float32).reshape({nn},{dd})
x2=torch.tensor({x2:?},dtype=torch.float32).reshape({nn},{dd})
y=torch.tensor({yv:?},dtype=torch.float32)
o32=torch.nn.functional.cosine_embedding_loss(x1,x2,y,margin={margin},reduction='none')
o64=torch.nn.functional.cosine_embedding_loss(x1.double(),x2.double(),y.double(),margin={margin},reduction='none')
print("V32"," ".join("%.9g"%v for v in o32.tolist()))
print("V64"," ".join("%.17g"%v for v in o64.tolist()))
"#,
        x1 = x1,
        x2 = x2,
        yv = yv,
        nn = nn,
        dd = dd,
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
    let (p32, p64) = (parse("V32 "), parse("V64 "));
    let mut s = FrankenTorchSession::new(ExecutionMode::Strict);
    let a = s.tensor_variable_f32(x1.clone(), vec![nn, dd], false)?;
    let b = s.tensor_variable_f32(x2.clone(), vec![nn, dd], false)?;
    let y = s.tensor_variable_f32(yv.clone(), vec![nn], false)?;
    let o = s.tensor_cosine_embedding_loss(a, b, y, margin, "none")?;
    let dt = s.tensor_dtype(o)?;
    let fv = s.tensor_values_lossy_f64(o)?;
    let mr32 = fv
        .iter()
        .zip(&p32)
        .map(|(x, y)| (x - y).abs() / y.abs().max(1e-6))
        .fold(0.0f64, f64::max);
    let x1_64: Vec<f64> = x1.iter().map(|&v| v as f64).collect();
    let x2_64: Vec<f64> = x2.iter().map(|&v| v as f64).collect();
    let y_64: Vec<f64> = yv.iter().map(|&v| v as f64).collect();
    let a6 = s.tensor_variable(x1_64, vec![nn, dd], false)?;
    let b6 = s.tensor_variable(x2_64, vec![nn, dd], false)?;
    let y6 = s.tensor_variable(y_64, vec![nn], false)?;
    let o6 = s.tensor_cosine_embedding_loss(a6, b6, y6, margin, "none")?;
    let fv6 = s.tensor_values(o6)?;
    let mr64 = fv6
        .iter()
        .zip(&p64)
        .map(|(x, y)| (x - y).abs() / y.abs().max(1e-6))
        .fold(0.0f64, f64::max);
    println!(
        "parity: f32 dtype={dt:?}(was ERROR) max_rel={mr32:.2e} | f64 max_rel={mr64:.2e} (tol<1e-5: {})",
        mr32 < 1e-5 && mr64 < 1e-5
    );

    // perf [N=200k,D=128] reduction='mean', f32
    let (n, d) = (200_000usize, 128usize);
    let af: Vec<f32> = (0..n * d)
        .map(|i| ((i % 9973) as f32 - 5000.0) * 0.001)
        .collect();
    let bf: Vec<f32> = (0..n * d)
        .map(|i| ((i % 7919) as f32 - 4000.0) * 0.001)
        .collect();
    let yf: Vec<f32> = (0..n)
        .map(|i| if i % 2 == 0 { 1.0 } else { -1.0 })
        .collect();
    let tt = |which: u8| {
        let mut bst = f64::INFINITY;
        for _ in 0..9 {
            let mut s = FrankenTorchSession::new(ExecutionMode::Strict);
            let e = if which == 0 {
                let x = s
                    .tensor_variable_f32(af.clone(), vec![n, d], false)
                    .unwrap();
                let y = s
                    .tensor_variable_f32(bf.clone(), vec![n, d], false)
                    .unwrap();
                let ti = Instant::now();
                let _ = s.tensor_add(x, y);
                ti.elapsed().as_secs_f64() * 1e3
            } else {
                let x = s
                    .tensor_variable_f32(af.clone(), vec![n, d], false)
                    .unwrap();
                let y2 = s
                    .tensor_variable_f32(bf.clone(), vec![n, d], false)
                    .unwrap();
                let yt = s.tensor_variable_f32(yf.clone(), vec![n], false).unwrap();
                let ti = Instant::now();
                let _ = s.tensor_cosine_embedding_loss(x, y2, yt, margin, "mean");
                ti.elapsed().as_secs_f64() * 1e3
            };
            if e < bst {
                bst = e;
            }
        }
        bst
    };
    let (tadd, tce) = (tt(0), tt(1));
    let py_b = format!(
        r#"
import time,torch
torch.set_num_threads(8)
n={n}; d={d}; margin={margin}
x=(((torch.arange(n*d,dtype=torch.int64)%9973).float()-5000.0)*0.001).reshape(n,d)
y=(((torch.arange(n*d,dtype=torch.int64)%7919).float()-4000.0)*0.001).reshape(n,d)
t=torch.where((torch.arange(n)%2)==0,1.0,-1.0).float()
def tm(fn,reps=9):
    for _ in range(2): fn()
    ts=[]
    for _ in range(reps): s=time.perf_counter(); fn(); ts.append((time.perf_counter()-s)*1e3)
    return min(ts)
print("PT add %.4f"%tm(lambda:x+y))
print("PT ce %.4f"%tm(lambda:torch.nn.functional.cosine_embedding_loss(x,y,t,margin=margin,reduction='mean')))
"#,
        n = n,
        d = d,
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
        "  add_anchor   FT {tadd:.3} PT {:.3}  => {}",
        g("add"),
        v(tadd, g("add"))
    );
    println!(
        "  cos_emb_f32  FT {tce:.3} PT {:.3}  => {}",
        g("ce"),
        v(tce, g("ce"))
    );
    Ok(())
}

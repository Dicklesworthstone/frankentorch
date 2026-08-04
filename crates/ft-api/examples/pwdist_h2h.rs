//! pairwise_distance fused fast path vs torch. Was: eps-AFTER-norm (divergence vs torch
//! eps-INSIDE), powf even for p=2, full(eps)[F64] (f32 returned F64). add = anchor. dim=last.
use ft_api::FrankenTorchSession;
use ft_core::ExecutionMode;
use std::io::Write;
use std::process::{Command, Stdio};
use std::time::Instant;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let python = std::env::var("PYTORCH_PYTHON").unwrap_or_else(|_| "python3".to_string());

    // parity [N=4, D=5] dim=last, p=2/1/3, eps=1e-6, within tolerance (distance metric)
    let (nn, dd) = (4usize, 5usize);
    let x1: Vec<f32> = (0..nn * dd).map(|i| ((i % 7) as f32 - 3.0) * 0.5).collect();
    let x2: Vec<f32> = (0..nn * dd)
        .map(|i| ((i % 5) as f32 - 2.0) * 0.4 + 0.1)
        .collect();
    let py_s = format!(
        r#"
import torch
x1=torch.tensor({x1:?},dtype=torch.float32).reshape({nn},{dd})
x2=torch.tensor({x2:?},dtype=torch.float32).reshape({nn},{dd})
for p in (2.0,1.0,3.0):
    o32=torch.nn.functional.pairwise_distance(x1,x2,p=p,eps=1e-6)
    o64=torch.nn.functional.pairwise_distance(x1.double(),x2.double(),p=p,eps=1e-6)
    print("V32 %g"%p," ".join("%.9g"%v for v in o32.tolist()))
    print("V64 %g"%p," ".join("%.17g"%v for v in o64.tolist()))
"#,
        x1 = x1,
        x2 = x2,
        nn = nn,
        dd = dd
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
    let x1_64: Vec<f64> = x1.iter().map(|&v| v as f64).collect();
    let x2_64: Vec<f64> = x2.iter().map(|&v| v as f64).collect();
    for p in [2.0_f64, 1.0, 3.0] {
        let p32 = parse(&format!("V32 {p} "));
        let p64 = parse(&format!("V64 {p} "));
        let mut s = FrankenTorchSession::new(ExecutionMode::Strict);
        let a = s.tensor_variable_f32(x1.clone(), vec![nn, dd], false)?;
        let b = s.tensor_variable_f32(x2.clone(), vec![nn, dd], false)?;
        let o = s.tensor_pairwise_distance(a, b, p, 1e-6)?;
        let dt = s.tensor_dtype(o)?;
        let fv = s.tensor_values_lossy_f64(o)?;
        let mr32 = fv
            .iter()
            .zip(&p32)
            .map(|(a, b)| (a - b).abs() / b.abs().max(1e-30))
            .fold(0.0f64, f64::max);
        let a6 = s.tensor_variable(x1_64.clone(), vec![nn, dd], false)?;
        let b6 = s.tensor_variable(x2_64.clone(), vec![nn, dd], false)?;
        let o6 = s.tensor_pairwise_distance(a6, b6, p, 1e-6)?;
        let dt6 = s.tensor_dtype(o6)?;
        let fv6 = s.tensor_values(o6)?;
        let mr64 = fv6
            .iter()
            .zip(&p64)
            .map(|(a, b)| (a - b).abs() / b.abs().max(1e-30))
            .fold(0.0f64, f64::max);
        println!(
            "parity p={p}: f32 dtype={dt:?}(was F64) max_rel={mr32:.2e} | f64 dtype={dt6:?} max_rel={mr64:.2e} (tol<1e-5: {})",
            mr32 < 1e-5 && mr64 < 1e-5
        );
    }

    // grad-path (requires_grad=true) forces the composed fallback; verify it ALSO uses
    // eps-inside now (the old fallback added eps AFTER the norm -> divergence vs torch).
    {
        let p64 = parse("V64 2 ");
        let mut s = FrankenTorchSession::new(ExecutionMode::Strict);
        let a = s.tensor_variable(x1_64.clone(), vec![nn, dd], true)?;
        let b = s.tensor_variable(x2_64.clone(), vec![nn, dd], true)?;
        let o = s.tensor_pairwise_distance(a, b, 2.0, 1e-6)?;
        let fv = s.tensor_values(o)?;
        let mr = fv
            .iter()
            .zip(&p64)
            .map(|(a, b)| (a - b).abs() / b.abs().max(1e-30))
            .fold(0.0f64, f64::max);
        println!(
            "grad-path fallback p=2 f64: max_rel={mr:.2e} (eps-inside, tol<1e-9: {})",
            mr < 1e-9
        );
    }

    // perf [N=200k, D=128] dim=last, p=2, f32 + f64
    let (n, d) = (200_000usize, 128usize);
    let af: Vec<f32> = (0..n * d)
        .map(|i| ((i % 9973) as f32 - 5000.0) * 0.001)
        .collect();
    let bf: Vec<f32> = (0..n * d)
        .map(|i| ((i % 7919) as f32 - 4000.0) * 0.001)
        .collect();
    let a6: Vec<f64> = af.iter().map(|&v| v as f64).collect();
    let b6: Vec<f64> = bf.iter().map(|&v| v as f64).collect();
    let tt = |which: u8| {
        let mut bst = f64::INFINITY;
        for _ in 0..9 {
            let mut s = FrankenTorchSession::new(ExecutionMode::Strict);
            let (e, _g) = match which {
                0 => {
                    let x = s
                        .tensor_variable_f32(af.clone(), vec![n, d], false)
                        .unwrap();
                    let y = s
                        .tensor_variable_f32(bf.clone(), vec![n, d], false)
                        .unwrap();
                    let ti = Instant::now();
                    let _ = s.tensor_add(x, y);
                    (ti.elapsed().as_secs_f64() * 1e3, 0)
                }
                1 => {
                    let x = s
                        .tensor_variable_f32(af.clone(), vec![n, d], false)
                        .unwrap();
                    let y = s
                        .tensor_variable_f32(bf.clone(), vec![n, d], false)
                        .unwrap();
                    let ti = Instant::now();
                    let _ = s.tensor_pairwise_distance(x, y, 2.0, 1e-6);
                    (ti.elapsed().as_secs_f64() * 1e3, 0)
                }
                _ => {
                    let x = s.tensor_variable(a6.clone(), vec![n, d], false).unwrap();
                    let y = s.tensor_variable(b6.clone(), vec![n, d], false).unwrap();
                    let ti = Instant::now();
                    let _ = s.tensor_pairwise_distance(x, y, 2.0, 1e-6);
                    (ti.elapsed().as_secs_f64() * 1e3, 0)
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
n={n}; d={d}
x=(((torch.arange(n*d,dtype=torch.int64)%9973).float()-5000.0)*0.001).reshape(n,d)
y=(((torch.arange(n*d,dtype=torch.int64)%7919).float()-4000.0)*0.001).reshape(n,d)
xd=x.double(); yd=y.double()
def tm(fn,reps=9):
    for _ in range(2): fn()
    ts=[]
    for _ in range(reps): s=time.perf_counter(); fn(); ts.append((time.perf_counter()-s)*1e3)
    return min(ts)
print("PT add %.4f"%tm(lambda:x+y))
print("PT pw32 %.4f"%tm(lambda:torch.nn.functional.pairwise_distance(x,y,p=2.0,eps=1e-6)))
print("PT pw64 %.4f"%tm(lambda:torch.nn.functional.pairwise_distance(xd,yd,p=2.0,eps=1e-6)))
"#,
        n = n,
        d = d
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
        "  pwdist_f32   FT {tf32:.3} PT {:.3}  => {}",
        g("pw32"),
        v(tf32, g("pw32"))
    );
    println!(
        "  pwdist_f64   FT {tf64:.3} PT {:.3}  => {}",
        g("pw64"),
        v(tf64, g("pw64"))
    );
    Ok(())
}

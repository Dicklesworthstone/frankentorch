//! triplet_margin_loss perf/parity vs torch. FT composes sub*2 + norm_dim*2 (powf) +
//! full[F64]*2 + maximum + reduce; eps=0 (golden eps-tolerant). add = anchor. dim=last.
use ft_api::FrankenTorchSession;
use ft_core::ExecutionMode;
use std::io::Write;
use std::process::{Command, Stdio};
use std::time::Instant;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let python = std::env::var("PYTORCH_PYTHON").unwrap_or_else(|_| "python3".to_string());

    // parity [N=4,D=5] reduction='none', p=2, margin=1.0 (FT eps=0 vs torch eps=1e-6 -> tol)
    let (nn, dd) = (4usize, 5usize);
    let a: Vec<f32> = (0..nn * dd).map(|i| ((i % 7) as f32 - 3.0) * 0.5).collect();
    let p: Vec<f32> = (0..nn * dd)
        .map(|i| ((i % 5) as f32 - 2.0) * 0.4 + 0.1)
        .collect();
    let ng: Vec<f32> = (0..nn * dd)
        .map(|i| ((i % 3) as f32 - 1.0) * 0.6 - 0.2)
        .collect();
    let py_s = format!(
        r#"
import torch
a=torch.tensor({a:?},dtype=torch.float32).reshape({nn},{dd})
p=torch.tensor({p:?},dtype=torch.float32).reshape({nn},{dd})
n=torch.tensor({ng:?},dtype=torch.float32).reshape({nn},{dd})
o32=torch.nn.functional.triplet_margin_loss(a,p,n,margin=1.0,p=2.0,reduction='none')
o64=torch.nn.functional.triplet_margin_loss(a.double(),p.double(),n.double(),margin=1.0,p=2.0,reduction='none')
print("V32"," ".join("%.9g"%v for v in o32.tolist()))
print("V64"," ".join("%.17g"%v for v in o64.tolist()))
"#,
        a = a,
        p = p,
        ng = ng,
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
    let (p32, p64) = (parse("V32 "), parse("V64 "));
    let mut s = FrankenTorchSession::new(ExecutionMode::Strict);
    let av = s.tensor_variable_f32(a.clone(), vec![nn, dd], false)?;
    let pv = s.tensor_variable_f32(p.clone(), vec![nn, dd], false)?;
    let nv = s.tensor_variable_f32(ng.clone(), vec![nn, dd], false)?;
    let o = s.tensor_triplet_margin_loss(av, pv, nv, 1.0, 2.0, "none")?;
    let dt = s.tensor_dtype(o)?;
    let fv = s.tensor_values_lossy_f64(o)?;
    let mr32 = fv
        .iter()
        .zip(&p32)
        .map(|(x, y)| (x - y).abs() / y.abs().max(1e-6))
        .fold(0.0f64, f64::max);
    let a6: Vec<f64> = a.iter().map(|&v| v as f64).collect();
    let p6: Vec<f64> = p.iter().map(|&v| v as f64).collect();
    let n6: Vec<f64> = ng.iter().map(|&v| v as f64).collect();
    let av6 = s.tensor_variable(a6.clone(), vec![nn, dd], false)?;
    let pv6 = s.tensor_variable(p6.clone(), vec![nn, dd], false)?;
    let nv6 = s.tensor_variable(n6.clone(), vec![nn, dd], false)?;
    let o6 = s.tensor_triplet_margin_loss(av6, pv6, nv6, 1.0, 2.0, "none")?;
    let fv6 = s.tensor_values(o6)?;
    let mr64 = fv6
        .iter()
        .zip(&p64)
        .map(|(x, y)| (x - y).abs() / y.abs().max(1e-6))
        .fold(0.0f64, f64::max);
    println!(
        "parity: f32 dtype={dt:?} max_rel={mr32:.2e} | f64 max_rel={mr64:.2e} (eps0-vs-torch-eps1e-6, tol<1e-4: {})",
        mr32 < 1e-4 && mr64 < 1e-4
    );

    // perf [N=200k,D=128] reduction='mean', f32 + f64
    let (n, d) = (200_000usize, 128usize);
    let af: Vec<f32> = (0..n * d)
        .map(|i| ((i % 9973) as f32 - 5000.0) * 0.001)
        .collect();
    let pf: Vec<f32> = (0..n * d)
        .map(|i| ((i % 7919) as f32 - 4000.0) * 0.001)
        .collect();
    let nf: Vec<f32> = (0..n * d)
        .map(|i| ((i % 6113) as f32 - 3000.0) * 0.001)
        .collect();
    let a6: Vec<f64> = af.iter().map(|&v| v as f64).collect();
    let p6: Vec<f64> = pf.iter().map(|&v| v as f64).collect();
    let n6: Vec<f64> = nf.iter().map(|&v| v as f64).collect();
    let tt = |which: u8| {
        let mut bst = f64::INFINITY;
        for _ in 0..9 {
            let mut s = FrankenTorchSession::new(ExecutionMode::Strict);
            let e = match which {
                0 => {
                    let x = s
                        .tensor_variable_f32(af.clone(), vec![n, d], false)
                        .unwrap();
                    let y = s
                        .tensor_variable_f32(pf.clone(), vec![n, d], false)
                        .unwrap();
                    let ti = Instant::now();
                    let _ = s.tensor_add(x, y);
                    ti.elapsed().as_secs_f64() * 1e3
                }
                1 => {
                    let x = s
                        .tensor_variable_f32(af.clone(), vec![n, d], false)
                        .unwrap();
                    let y = s
                        .tensor_variable_f32(pf.clone(), vec![n, d], false)
                        .unwrap();
                    let z = s
                        .tensor_variable_f32(nf.clone(), vec![n, d], false)
                        .unwrap();
                    let ti = Instant::now();
                    let _ = s.tensor_triplet_margin_loss(x, y, z, 1.0, 2.0, "mean");
                    ti.elapsed().as_secs_f64() * 1e3
                }
                _ => {
                    let x = s.tensor_variable(a6.clone(), vec![n, d], false).unwrap();
                    let y = s.tensor_variable(p6.clone(), vec![n, d], false).unwrap();
                    let z = s.tensor_variable(n6.clone(), vec![n, d], false).unwrap();
                    let ti = Instant::now();
                    let _ = s.tensor_triplet_margin_loss(x, y, z, 1.0, 2.0, "mean");
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
n={n}; d={d}
a=(((torch.arange(n*d,dtype=torch.int64)%9973).float()-5000.0)*0.001).reshape(n,d)
p=(((torch.arange(n*d,dtype=torch.int64)%7919).float()-4000.0)*0.001).reshape(n,d)
nn=(((torch.arange(n*d,dtype=torch.int64)%6113).float()-3000.0)*0.001).reshape(n,d)
ad=a.double(); pd=p.double(); nd=nn.double()
def tm(fn,reps=9):
    for _ in range(2): fn()
    ts=[]
    for _ in range(reps): s=time.perf_counter(); fn(); ts.append((time.perf_counter()-s)*1e3)
    return min(ts)
print("PT add %.4f"%tm(lambda:a+p))
print("PT t32 %.4f"%tm(lambda:torch.nn.functional.triplet_margin_loss(a,p,nn,margin=1.0,p=2.0,reduction='mean')))
print("PT t64 %.4f"%tm(lambda:torch.nn.functional.triplet_margin_loss(ad,pd,nd,margin=1.0,p=2.0,reduction='mean')))
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
        "  triplet_f32  FT {tf32:.3} PT {:.3}  => {}",
        g("t32"),
        v(tf32, g("t32"))
    );
    println!(
        "  triplet_f64  FT {tf64:.3} PT {:.3}  => {}",
        g("t64"),
        v(tf64, g("t64"))
    );
    Ok(())
}

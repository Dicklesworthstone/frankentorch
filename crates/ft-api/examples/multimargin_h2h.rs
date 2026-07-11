//! multi_margin_loss f32 vs torch. Currently routes input through apply_function (f64 upcast
//! + clone) -> likely F64 output (dtype bug; torch->f32) + slow. add = anchor.
use ft_api::FrankenTorchSession;
use ft_core::ExecutionMode;
use std::io::Write;
use std::process::{Command, Stdio};
use std::time::Instant;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let python = std::env::var("PYTORCH_PYTHON").unwrap_or_else(|_| "python3".to_string());
    let (p, margin) = (1usize, 1.0_f64);

    // parity [N=6, C=5] reduction='none'
    let (nn, cc) = (6usize, 5usize);
    let x: Vec<f32> = (0..nn * cc).map(|i| ((i % 7) as f32 - 3.0) * 0.4).collect();
    let tg: Vec<i64> = (0..nn).map(|i| (i % cc) as i64).collect();
    let py_s = format!(
        r#"
import torch
x=torch.tensor({x:?},dtype=torch.float32).reshape({nn},{cc})
t=torch.tensor({tg:?},dtype=torch.int64)
o32=torch.nn.functional.multi_margin_loss(x,t,p={p},margin={margin},reduction='none')
o64=torch.nn.functional.multi_margin_loss(x.double(),t,p={p},margin={margin},reduction='none')
print("V32"," ".join("%.9g"%v for v in o32.tolist()))
print("V64"," ".join("%.17g"%v for v in o64.tolist()))
"#,
        x = x,
        tg = tg,
        nn = nn,
        cc = cc,
        p = p,
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
    let a = s.tensor_variable_f32(x.clone(), vec![nn, cc], false)?;
    let t = s.tensor_variable(tg.iter().map(|&v| v as f64).collect(), vec![nn], false)?;
    match s.tensor_multi_margin_loss(a, t, p, margin, None, "none") {
        Ok(o) => {
            let dt = s.tensor_dtype(o)?;
            let fv = s.tensor_values_lossy_f64(o)?;
            let mr = fv
                .iter()
                .zip(&p32)
                .map(|(u, w)| (u - w).abs() / w.abs().max(1e-6))
                .fold(0.0f64, f64::max);
            let _ = &p64;
            println!(
                "parity f32: dtype={dt:?} max_rel={mr:.2e} (tol<1e-5: {})",
                mr < 1e-5
            );
        }
        Err(e) => println!("f32 ERROR -> {e:?}"),
    }

    // perf [N=200k, C=128] reduction='mean', f32
    let (n, c) = (200_000usize, 128usize);
    let xf: Vec<f32> = (0..n * c)
        .map(|i| ((i % 9973) as f32 - 5000.0) * 0.001)
        .collect();
    let tf: Vec<f64> = (0..n).map(|i| (i % c) as f64).collect();
    let tt = |which: u8| {
        let mut bst = f64::INFINITY;
        for _ in 0..7 {
            let mut s = FrankenTorchSession::new(ExecutionMode::Strict);
            let x = s
                .tensor_variable_f32(xf.clone(), vec![n, c], false)
                .unwrap();
            let e = if which == 0 {
                let y = s
                    .tensor_variable_f32(xf.clone(), vec![n, c], false)
                    .unwrap();
                let ti = Instant::now();
                let _ = s.tensor_add(x, y);
                ti.elapsed().as_secs_f64() * 1e3
            } else {
                let t = s.tensor_variable(tf.clone(), vec![n], false).unwrap();
                let ti = Instant::now();
                let _ = s.tensor_multi_margin_loss(x, t, p, margin, None, "mean");
                ti.elapsed().as_secs_f64() * 1e3
            };
            if e < bst {
                bst = e;
            }
        }
        bst
    };
    let (tadd, tm) = (tt(0), tt(1));
    let py_b = format!(
        r#"
import time,torch
torch.set_num_threads(8)
n={n}; c={c}; p={p}; margin={margin}
x=(((torch.arange(n*c,dtype=torch.int64)%9973).float()-5000.0)*0.001).reshape(n,c)
t=(torch.arange(n,dtype=torch.int64)%c)
def tm(fn,reps=7):
    for _ in range(2): fn()
    ts=[]
    for _ in range(reps): s=time.perf_counter(); fn(); ts.append((time.perf_counter()-s)*1e3)
    return min(ts)
print("PT add %.4f"%tm(lambda:x+x))
print("PT mm %.4f"%tm(lambda:torch.nn.functional.multi_margin_loss(x,t,p=p,margin=margin,reduction='mean')))
"#,
        n = n,
        c = c,
        p = p,
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
        "  multimargin  FT {tm:.3} PT {:.3}  => {}",
        g("mm"),
        v(tm, g("mm"))
    );
    Ok(())
}

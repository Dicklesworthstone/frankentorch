//! cartesian_prod f32 native fast path vs torch. Was: serial Vec<Vec<usize>> mapping precompute
//! over all rows + apply_function f64-gather (~14x slower). add = anchor.
use ft_api::FrankenTorchSession;
use ft_core::ExecutionMode;
use std::io::Write;
use std::process::{Command, Stdio};
use std::time::Instant;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let python = std::env::var("PYTORCH_PYTHON").unwrap_or_else(|_| "python3".to_string());

    // parity: cartesian_prod of [3],[2],[2]
    let a: Vec<f32> = vec![1.0, 2.0, 3.0];
    let b: Vec<f32> = vec![4.0, 5.0];
    let c: Vec<f32> = vec![6.0, 7.0];
    let py_s = format!(
        r#"
import torch
a=torch.tensor({a:?},dtype=torch.float32); b=torch.tensor({b:?},dtype=torch.float32); c=torch.tensor({c:?},dtype=torch.float32)
o=torch.cartesian_prod(a,b,c)
print("SHAPE",list(o.shape))
print("V"," ".join("%.9g"%v for v in o.flatten().tolist()))
"#,
        a = a,
        b = b,
        c = c
    );
    let mut ch = Command::new(&python)
        .arg("-")
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .spawn()?;
    ch.stdin.as_mut().unwrap().write_all(py_s.as_bytes())?;
    let pt = String::from_utf8_lossy(&ch.wait_with_output()?.stdout).to_string();
    let pv: Vec<f64> = pt
        .lines()
        .find_map(|l| l.strip_prefix("V "))
        .map(|s| {
            s.split_whitespace()
                .filter_map(|t| t.parse().ok())
                .collect()
        })
        .unwrap_or_default();
    let mut s = FrankenTorchSession::new(ExecutionMode::Strict);
    let ia = s.tensor_variable_f32(a.clone(), vec![3], false)?;
    let ib = s.tensor_variable_f32(b.clone(), vec![2], false)?;
    let ic = s.tensor_variable_f32(c.clone(), vec![2], false)?;
    let o = s.tensor_cartesian_prod(&[ia, ib, ic])?;
    let dt = s.tensor_dtype(o)?;
    let fv = s.tensor_values_lossy_f64(o)?;
    let mm = fv
        .iter()
        .zip(&pv)
        .filter(|(x, y)| (**x as f32).to_bits() != (**y as f32).to_bits())
        .count()
        + fv.len().abs_diff(pv.len());
    println!(
        "parity f32: dtype={dt:?} mismatch={mm}/{} (ft_len={} torch_len={})",
        pv.len(),
        fv.len(),
        pv.len()
    );

    // perf: 2x [4096] -> [16.7M, 2]
    let n = 4096usize;
    let af: Vec<f32> = (0..n).map(|i| i as f32).collect();
    let bf: Vec<f32> = (0..n).map(|i| i as f32 * 0.5).collect();
    let big: Vec<f32> = vec![1.0f32; 16_777_216];
    let tadd = || {
        let mut bst = f64::INFINITY;
        for _ in 0..7 {
            let mut s = FrankenTorchSession::new(ExecutionMode::Strict);
            let x = s
                .tensor_variable_f32(big.clone(), vec![16_777_216], false)
                .unwrap();
            let y = s
                .tensor_variable_f32(big.clone(), vec![16_777_216], false)
                .unwrap();
            let ti = Instant::now();
            let _ = s.tensor_add(x, y);
            let e = ti.elapsed().as_secs_f64() * 1e3;
            if e < bst {
                bst = e;
            }
        }
        bst
    };
    let tcp = || {
        let mut bst = f64::INFINITY;
        for _ in 0..7 {
            let mut s = FrankenTorchSession::new(ExecutionMode::Strict);
            let x = s.tensor_variable_f32(af.clone(), vec![n], false).unwrap();
            let y = s.tensor_variable_f32(bf.clone(), vec![n], false).unwrap();
            let ti = Instant::now();
            let _ = s.tensor_cartesian_prod(&[x, y]);
            let e = ti.elapsed().as_secs_f64() * 1e3;
            if e < bst {
                bst = e;
            }
        }
        bst
    };
    let (ta, tc) = (tadd(), tcp());
    let py_b = format!(
        r#"
import time,torch
torch.set_num_threads(8)
n={n}
a=torch.arange(n,dtype=torch.float32); b=torch.arange(n,dtype=torch.float32)*0.5
big=torch.ones(16777216)
def tm(fn,reps=7):
    for _ in range(2): fn()
    ts=[]
    for _ in range(reps): s=time.perf_counter(); fn(); ts.append((time.perf_counter()-s)*1e3)
    return min(ts)
print("PT add %.4f"%tm(lambda:big+big))
print("PT cp %.4f"%tm(lambda:torch.cartesian_prod(a,b)))
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
        "  add_anchor   FT {ta:.3} PT {:.3}  => {}",
        g("add"),
        v(ta, g("add"))
    );
    println!(
        "  cartprod_f32 FT {tc:.3} PT {:.3}  => {}",
        g("cp"),
        v(tc, g("cp"))
    );
    Ok(())
}

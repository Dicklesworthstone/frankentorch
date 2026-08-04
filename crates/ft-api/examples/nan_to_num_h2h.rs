//! nan_to_num no-grad fast path (f32+f64) vs torch. Was: ~9-pass composed (5 full_like + 2 eq
//! + 2 where + isnan). add = anchor. nan_to_num(x): NaN->0, +inf->dtype_max, -inf->dtype_min.
use ft_api::FrankenTorchSession;
use ft_core::ExecutionMode;
use std::io::Write;
use std::process::{Command, Stdio};
use std::time::Instant;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let python = std::env::var("PYTORCH_PYTHON").unwrap_or_else(|_| "python3".to_string());

    // parity (f64): explicit specials built on both sides (no NaN/inf in a list literal).
    let py_s = r#"
import torch
a=torch.tensor([1.0,float('nan'),float('inf'),float('-inf'),-2.5,0.0,float('nan'),3.0],dtype=torch.float64)
o=torch.nan_to_num(a)
print("VALS"," ".join("%.17g"%v for v in o.tolist()))
af=a.float(); of=torch.nan_to_num(af)
print("VALSF"," ".join("%.9g"%v for v in of.tolist()))
"#;
    let mut ch = Command::new(&python)
        .arg("-")
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .spawn()?;
    ch.stdin.as_mut().unwrap().write_all(py_s.as_bytes())?;
    let pt = String::from_utf8_lossy(&ch.wait_with_output()?.stdout).to_string();
    let parse = |key: &str| -> Vec<f64> {
        pt.lines()
            .find_map(|l| l.strip_prefix(key))
            .map(|s| {
                s.split_whitespace()
                    .filter_map(|t| t.parse().ok())
                    .collect()
            })
            .unwrap_or_default()
    };
    let pv = parse("VALS ");
    let pvf = parse("VALSF ");
    let specials = vec![
        1.0_f64,
        f64::NAN,
        f64::INFINITY,
        f64::NEG_INFINITY,
        -2.5,
        0.0,
        f64::NAN,
        3.0,
    ];
    // f64 parity
    let mut s = FrankenTorchSession::new(ExecutionMode::Strict);
    let x = s.tensor_variable(specials.clone(), vec![specials.len()], false)?;
    let o = s.tensor_nan_to_num(x, 0.0, None, None)?;
    let fv = s.tensor_values(o)?;
    let mm = fv
        .iter()
        .zip(&pv)
        .filter(|(a, b)| a.to_bits() != b.to_bits())
        .count()
        + fv.len().abs_diff(pv.len());
    // f32 parity
    let specials_f: Vec<f32> = specials.iter().map(|&v| v as f32).collect();
    let xf = s.tensor_variable_f32(specials_f.clone(), vec![specials_f.len()], false)?;
    let of = s.tensor_nan_to_num(xf, 0.0, None, None)?;
    let dtf = s.tensor_dtype(of)?;
    let fvf = s.tensor_values_lossy_f64(of)?;
    let mmf = fvf
        .iter()
        .zip(&pvf)
        .filter(|(a, b)| (**a as f32).to_bits() != (**b as f32).to_bits())
        .count()
        + fvf.len().abs_diff(pvf.len());
    println!(
        "parity f64: {mm}/{} mismatches | f32: dtype={dtf:?} {mmf}/{} mismatches",
        pv.len(),
        pvf.len()
    );

    // perf 16M, time only the op, both dtypes
    let n = 16_000_000usize;
    let a64: Vec<f64> = (0..n)
        .map(|i| match i % 7 {
            0 => f64::NAN,
            1 => f64::INFINITY,
            2 => f64::NEG_INFINITY,
            k => (k as f64 - 3.0) * 0.5,
        })
        .collect();
    let a32: Vec<f32> = a64.iter().map(|&v| v as f32).collect();
    let t64 = {
        let mut b = f64::INFINITY;
        for _ in 0..9 {
            let mut s = FrankenTorchSession::new(ExecutionMode::Strict);
            let x = s.tensor_variable(a64.clone(), vec![n], false).unwrap();
            let t = Instant::now();
            let _ = s.tensor_nan_to_num(x, 0.0, None, None);
            let e = t.elapsed().as_secs_f64() * 1e3;
            if e < b {
                b = e;
            }
        }
        b
    };
    let t32 = {
        let mut b = f64::INFINITY;
        for _ in 0..9 {
            let mut s = FrankenTorchSession::new(ExecutionMode::Strict);
            let x = s.tensor_variable_f32(a32.clone(), vec![n], false).unwrap();
            let t = Instant::now();
            let _ = s.tensor_nan_to_num(x, 0.0, None, None);
            let e = t.elapsed().as_secs_f64() * 1e3;
            if e < b {
                b = e;
            }
        }
        b
    };
    let tadd = {
        let mut b = f64::INFINITY;
        for _ in 0..9 {
            let mut s = FrankenTorchSession::new(ExecutionMode::Strict);
            let x = s.tensor_variable(a64.clone(), vec![n], false).unwrap();
            let t = Instant::now();
            let _ = s.tensor_relu(x);
            let e = t.elapsed().as_secs_f64() * 1e3;
            if e < b {
                b = e;
            }
        }
        b
    };
    let py_b = format!(
        r#"
import time,torch
torch.set_num_threads(8)
n={n}
base=(((torch.arange(n,dtype=torch.int64)%7)).double()-3.0)*0.5
a=base.clone(); a[0::7]=float('nan'); a[1::7]=float('inf'); a[2::7]=float('-inf')
af=a.float()
def t(fn,reps=9):
    for _ in range(2): fn()
    ts=[]
    for _ in range(reps): s=time.perf_counter(); fn(); ts.append((time.perf_counter()-s)*1e3)
    return min(ts)
print("PT relu %.4f"%t(lambda:torch.relu(a)))
print("PT n2n64 %.4f"%t(lambda:torch.nan_to_num(a)))
print("PT n2n32 %.4f"%t(lambda:torch.nan_to_num(af)))
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
    let v = |ft: f64, pt: f64| {
        if pt >= ft {
            format!("FT {:.2}x FASTER", pt / ft)
        } else {
            format!("FT {:.2}x SLOWER", ft / pt)
        }
    };
    println!(
        "  relu_anchor  FT {tadd:.3} PT {:.3}  => {}",
        g("relu"),
        v(tadd, g("relu"))
    );
    println!(
        "  n2n f64      FT {t64:.3} PT {:.3}  => {}",
        g("n2n64"),
        v(t64, g("n2n64"))
    );
    println!(
        "  n2n f32      FT {t32:.3} PT {:.3}  => {}",
        g("n2n32"),
        v(t32, g("n2n32"))
    );
    Ok(())
}

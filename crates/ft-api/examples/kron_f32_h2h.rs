//! kron f32 no-grad vs torch. Was: ERROR (UnsupportedDType(F32) — F64-only tensor_values).
//! Now: native rank-2 f32 fast path. add = anchor.
use ft_api::FrankenTorchSession;
use ft_core::ExecutionMode;
use std::io::Write;
use std::process::{Command, Stdio};
use std::time::Instant;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let python = std::env::var("PYTORCH_PYTHON").unwrap_or_else(|_| "python3".to_string());

    // parity [2,2] kron [2,3]
    let a: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0];
    let b: Vec<f32> = vec![0.0, 5.0, 6.0, 7.0, 8.0, 9.0];
    let py_s = format!(
        r#"
import torch
a=torch.tensor({a:?},dtype=torch.float32).reshape(2,2)
b=torch.tensor({b:?},dtype=torch.float32).reshape(2,3)
o32=torch.kron(a,b); o64=torch.kron(a.double(),b.double())
print("SHAPE",list(o32.shape))
print("V32"," ".join("%.9g"%v for v in o32.flatten().tolist()))
print("V64"," ".join("%.17g"%v for v in o64.flatten().tolist()))
"#,
        a = a,
        b = b
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
    let av = s.tensor_variable_f32(a.clone(), vec![2, 2], false)?;
    let bv = s.tensor_variable_f32(b.clone(), vec![2, 3], false)?;
    match s.tensor_kron(av, bv) {
        Ok(o) => {
            let dt = s.tensor_dtype(o)?;
            let fv = s.tensor_values_lossy_f64(o)?;
            let mm = fv
                .iter()
                .zip(&p32)
                .filter(|(x, y)| (**x as f32).to_bits() != (**y as f32).to_bits())
                .count()
                + fv.len().abs_diff(p32.len());
            let _ = &p64;
            println!(
                "parity f32 NO-GRAD (was ERROR): dtype={dt:?} bitexact_mismatch={mm}/{}",
                p32.len()
            );
        }
        Err(e) => println!("f32 no-grad STILL ERRORS -> {e:?}"),
    }
    // grad path still works
    let av2 = s.tensor_variable_f32(a.clone(), vec![2, 2], true)?;
    let bv2 = s.tensor_variable_f32(b.clone(), vec![2, 3], false)?;
    let og = s.tensor_kron(av2, bv2)?;
    println!(
        "f32 GRAD path: dtype={:?} (still works)",
        s.tensor_dtype(og)?
    );

    // perf [256,256] kron [16,16] -> [4096,4096] (16.7M dense), f32
    let (am, bm) = (256usize, 16usize);
    let af: Vec<f32> = (0..am * am).map(|i| (i % 97) as f32 * 0.01).collect();
    let bf: Vec<f32> = (0..bm * bm).map(|i| (i % 13) as f32 * 0.1).collect();
    let big: Vec<f32> = vec![1.0f32; 4096 * 4096];
    let tadd = || {
        let mut bst = f64::INFINITY;
        for _ in 0..7 {
            let mut s = FrankenTorchSession::new(ExecutionMode::Strict);
            let x = s
                .tensor_variable_f32(big.clone(), vec![4096, 4096], false)
                .unwrap();
            let y = s
                .tensor_variable_f32(big.clone(), vec![4096, 4096], false)
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
    let tk = || {
        let mut bst = f64::INFINITY;
        for _ in 0..7 {
            let mut s = FrankenTorchSession::new(ExecutionMode::Strict);
            let x = s
                .tensor_variable_f32(af.clone(), vec![am, am], false)
                .unwrap();
            let y = s
                .tensor_variable_f32(bf.clone(), vec![bm, bm], false)
                .unwrap();
            let ti = Instant::now();
            let _ = s.tensor_kron(x, y);
            let e = ti.elapsed().as_secs_f64() * 1e3;
            if e < bst {
                bst = e;
            }
        }
        bst
    };
    let (ta, tkr) = (tadd(), tk());
    let py_b = format!(
        r#"
import time,torch
torch.set_num_threads(8)
am={am}; bm={bm}
a=((torch.arange(am*am,dtype=torch.int64)%97).float()*0.01).reshape(am,am)
b=((torch.arange(bm*bm,dtype=torch.int64)%13).float()*0.1).reshape(bm,bm)
big=torch.ones(4096,4096)
def tm(fn,reps=7):
    for _ in range(2): fn()
    ts=[]
    for _ in range(reps): s=time.perf_counter(); fn(); ts.append((time.perf_counter()-s)*1e3)
    return min(ts)
print("PT add %.4f"%tm(lambda:big+big))
print("PT kr %.4f"%tm(lambda:torch.kron(a,b)))
"#,
        am = am,
        bm = bm
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
        "  add_anchor FT {ta:.3} PT {:.3}  => {}",
        g("add"),
        v(ta, g("add"))
    );
    println!(
        "  kron_f32   FT {tkr:.3} PT {:.3}  => {}",
        g("kr"),
        v(tkr, g("kr"))
    );
    Ok(())
}

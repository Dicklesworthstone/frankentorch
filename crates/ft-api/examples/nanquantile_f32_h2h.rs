//! nanquantile f32 native fast path vs torch. Was: ERROR (UnsupportedDType(F32), F64-only
//! tensor_values). torch.nanquantile is a slow full-sort (~704ms@16M). add = anchor.
use ft_api::FrankenTorchSession;
use ft_core::ExecutionMode;
use std::io::Write;
use std::process::{Command, Stdio};
use std::time::Instant;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let python = std::env::var("PYTORCH_PYTHON").unwrap_or_else(|_| "python3".to_string());

    // parity: [N=11] with NaNs, q in {0.5, 0.25, 0.9}, default linear interp
    let xs: Vec<f32> = vec![3.0, 1.0, f32::NAN, 4.0, 1.5, 5.0, 9.0, 2.0, 6.0, f32::NAN, 0.5];
    for q in [0.5_f64, 0.25, 0.9] {
        let py_s = format!(r#"
import torch
x=torch.tensor([3.0,1.0,float('nan'),4.0,1.5,5.0,9.0,2.0,6.0,float('nan'),0.5],dtype=torch.float32)
o32=torch.nanquantile(x,{q})
o64=torch.nanquantile(x.double(),{q})
print("V32 %.9g"%o32.item())
print("V64 %.17g"%o64.item())
"#, q = q);
        let mut ch = Command::new(&python).arg("-").stdin(Stdio::piped()).stdout(Stdio::piped()).spawn()?;
        ch.stdin.as_mut().unwrap().write_all(py_s.as_bytes())?;
        let pt = String::from_utf8_lossy(&ch.wait_with_output()?.stdout).to_string();
        let g = |k: &str| pt.lines().find_map(|l| l.strip_prefix(k)).and_then(|s| s.trim().parse::<f64>().ok()).unwrap_or(f64::NAN);
        let (p32, _p64) = (g("V32 "), g("V64 "));
        let mut s = FrankenTorchSession::new(ExecutionMode::Strict);
        let a = s.tensor_variable_f32(xs.clone(), vec![xs.len()], false)?;
        let o = s.tensor_nanquantile(a, q)?;
        let dt = s.tensor_dtype(o)?;
        let fv = s.tensor_values_lossy_f64(o)?[0];
        let rel = (fv - p32).abs() / p32.abs().max(1e-6);
        println!("parity q={q}: dtype={dt:?}(was ERROR) ft={fv:.7} torch={p32:.7} rel={rel:.2e} (tol<1e-6: {})", rel < 1e-6);
    }

    // perf: 16M f32 with ~1% NaN, q=0.5
    let n = 16_000_000usize;
    let af: Vec<f32> = (0..n).map(|i| if i % 97 == 0 { f32::NAN } else { ((i % 9973) as f32 - 5000.0) * 0.01 }).collect();
    let big: Vec<f32> = vec![1.0f32; n];
    let tadd = || { let mut bst = f64::INFINITY; for _ in 0..7 { let mut s = FrankenTorchSession::new(ExecutionMode::Strict); let x = s.tensor_variable_f32(big.clone(), vec![n], false).unwrap(); let y = s.tensor_variable_f32(big.clone(), vec![n], false).unwrap(); let ti = Instant::now(); let _ = s.tensor_add(x, y); let e = ti.elapsed().as_secs_f64()*1e3; if e<bst{bst=e;} } bst };
    let tnq = || { let mut bst = f64::INFINITY; for _ in 0..7 { let mut s = FrankenTorchSession::new(ExecutionMode::Strict); let x = s.tensor_variable_f32(af.clone(), vec![n], false).unwrap(); let ti = Instant::now(); let _ = s.tensor_nanquantile(x, 0.5); let e = ti.elapsed().as_secs_f64()*1e3; if e<bst{bst=e;} } bst };
    let (ta, tn) = (tadd(), tnq());
    let py_b = format!(r#"
import time,torch
torch.set_num_threads(8)
n={n}
i=torch.arange(n)
a=torch.where(i%97==0, torch.tensor(float('nan')), ((i%9973).float()-5000.0)*0.01)
big=torch.ones(n)
def tm(fn,reps=7):
    for _ in range(2): fn()
    ts=[]
    for _ in range(reps): s=time.perf_counter(); fn(); ts.append((time.perf_counter()-s)*1e3)
    return min(ts)
print("PT add %.4f"%tm(lambda:big+big))
print("PT nq %.4f"%tm(lambda:torch.nanquantile(a,0.5)))
"#, n = n);
    let mut ch = Command::new(&python).arg("-").stdin(Stdio::piped()).stdout(Stdio::piped()).spawn()?;
    ch.stdin.as_mut().unwrap().write_all(py_b.as_bytes())?;
    let pt = String::from_utf8_lossy(&ch.wait_with_output()?.stdout).to_string();
    let g = |k: &str| pt.lines().find_map(|l| { let mut it = l.strip_prefix("PT ")?.split_whitespace(); if it.next()? == k { it.next()?.parse::<f64>().ok() } else { None } }).unwrap_or(f64::NAN);
    let v = |ft: f64, p: f64| if p >= ft { format!("FT {:.2}x FASTER", p / ft) } else { format!("FT {:.2}x SLOWER", ft / p) };
    println!("  add_anchor   FT {ta:.3} PT {:.3}  => {}", g("add"), v(ta, g("add")));
    println!("  nanquantile  FT {tn:.3} PT {:.3}  => {}", g("nq"), v(tn, g("nq")));
    Ok(())
}

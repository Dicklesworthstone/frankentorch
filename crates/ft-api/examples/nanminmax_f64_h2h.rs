//! f64 nanmin/nanmax fused no-grad fast path vs torch (was: full(±inf)+isnan+where+amin/amax).
//! relu = anchor. nanmin/nanmax = min/max ignoring NaN (all-NaN -> ±inf).
use ft_api::FrankenTorchSession;
use ft_core::ExecutionMode;
use std::io::Write;
use std::process::{Command, Stdio};
use std::time::Instant;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let python = std::env::var("PYTORCH_PYTHON").unwrap_or_else(|_| "python3".to_string());

    // parity: explicit specials (NaN + finite)
    let specials = vec![3.0_f64, f64::NAN, -2.0, 5.5, f64::NAN, -10.0, 0.0, 7.0];
    // This torch build lacks torch.nanmin/nanmax -> use the idiomatic equivalent a[~isnan].min().
    let py_s = r#"
import torch
a=torch.tensor([3.0,float('nan'),-2.0,5.5,float('nan'),-10.0,0.0,7.0],dtype=torch.float64)
def nmin(t): return t[~torch.isnan(t)].min()
def nmax(t): return t[~torch.isnan(t)].max()
print("MIN %.17g"%float(nmin(a)))
print("MAX %.17g"%float(nmax(a)))
"#;
    let mut ch = Command::new(&python)
        .arg("-")
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .spawn()?;
    ch.stdin.as_mut().unwrap().write_all(py_s.as_bytes())?;
    let pt = String::from_utf8_lossy(&ch.wait_with_output()?.stdout).to_string();
    let g1 = |k: &str| {
        pt.lines()
            .find_map(|l| l.strip_prefix(k).and_then(|s| s.trim().parse::<f64>().ok()))
            .unwrap_or(f64::NAN)
    };
    let (pmin, pmax) = (g1("MIN "), g1("MAX "));
    let mut s = FrankenTorchSession::new(ExecutionMode::Strict);
    let x = s.tensor_variable(specials.clone(), vec![specials.len()], false)?;
    let omin = s.tensor_nanmin(x)?;
    let x2 = s.tensor_variable(specials.clone(), vec![specials.len()], false)?;
    let omax = s.tensor_nanmax(x2)?;
    let (fmin, fmax) = (s.tensor_values(omin)?[0], s.tensor_values(omax)?[0]);
    println!(
        "parity f64: nanmin FT={fmin} PT={pmin} match={} | nanmax FT={fmax} PT={pmax} match={}",
        fmin.to_bits() == pmin.to_bits(),
        fmax.to_bits() == pmax.to_bits()
    );
    // f32 parity (f32 nanmin/nanmax were BROKEN before this fix)
    let sf: Vec<f32> = specials.iter().map(|&v| v as f32).collect();
    let xf = s.tensor_variable_f32(sf.clone(), vec![sf.len()], false)?;
    let ofmin = s.tensor_nanmin(xf)?;
    let xf2 = s.tensor_variable_f32(sf.clone(), vec![sf.len()], false)?;
    let ofmax = s.tensor_nanmax(xf2)?;
    let dtmin = s.tensor_dtype(ofmin)?;
    let (ffmin, ffmax) = (
        s.tensor_values_lossy_f64(ofmin)?[0] as f32,
        s.tensor_values_lossy_f64(ofmax)?[0] as f32,
    );
    println!(
        "parity f32: dtype={dtmin:?} nanmin FT={ffmin} PT={} match={} | nanmax FT={ffmax} PT={} match={}",
        pmin as f32,
        ffmin.to_bits() == (pmin as f32).to_bits(),
        pmax as f32,
        ffmax.to_bits() == (pmax as f32).to_bits()
    );

    // perf 16M f64, time only the op
    let n = 16_000_000usize;
    let a: Vec<f64> = (0..n)
        .map(|i| {
            if i % 5 == 0 {
                f64::NAN
            } else {
                ((i % 9973) as f64 - 5000.0) * 0.01
            }
        })
        .collect();
    let timed = |which: u8| {
        let mut b = f64::INFINITY;
        for _ in 0..9 {
            let mut s = FrankenTorchSession::new(ExecutionMode::Strict);
            let x = s.tensor_variable(a.clone(), vec![n], false).unwrap();
            let t = Instant::now();
            let _ = match which {
                0 => s.tensor_relu(x),
                1 => s.tensor_nanmin(x),
                _ => s.tensor_nanmax(x),
            };
            let e = t.elapsed().as_secs_f64() * 1e3;
            if e < b {
                b = e;
            }
        }
        b
    };
    let (tr, tmin, tmax) = (timed(0), timed(1), timed(2));
    let py_b = format!(
        r#"
import time,torch
torch.set_num_threads(8)
n={n}
a=(((torch.arange(n,dtype=torch.int64)%9973).double()-5000.0)*0.01)
a[0::5]=float('nan')
def t(fn,reps=9):
    for _ in range(2): fn()
    ts=[]
    for _ in range(reps): s=time.perf_counter(); fn(); ts.append((time.perf_counter()-s)*1e3)
    return min(ts)
print("PT relu %.4f"%t(lambda:torch.relu(a)))
print("PT nanmin %.4f"%t(lambda:a[~torch.isnan(a)].min()))
print("PT nanmax %.4f"%t(lambda:a[~torch.isnan(a)].max()))
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
        "  relu_anchor  FT {tr:.3} PT {:.3}  => {}",
        g("relu"),
        v(tr, g("relu"))
    );
    println!(
        "  nanmin       FT {tmin:.3} PT {:.3}  => {}",
        g("nanmin"),
        v(tmin, g("nanmin"))
    );
    println!(
        "  nanmax       FT {tmax:.3} PT {:.3}  => {}",
        g("nanmax"),
        v(tmax, g("nanmax"))
    );
    Ok(())
}

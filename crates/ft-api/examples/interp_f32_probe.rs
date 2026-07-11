//! interpolate bilinear/bicubic f32 dtype+perf probe vs torch.
//! Checks: (1) does f32 interpolate return F32 (torch) or upcast to F64 (bug)?
//! (2) parity vs torch f32/f64. (3) FT-vs-torch timing both dtypes (add anchor).
use ft_api::FrankenTorchSession;
use ft_core::ExecutionMode;
use std::io::Write;
use std::process::{Command, Stdio};
use std::time::Instant;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let python = std::env::var("PYTORCH_PYTHON").unwrap_or_else(|_| "python3".to_string());

    // ---- parity: small [1,2,4,5] -> [1,2,8,10] both modes, f32 + f64 ----
    let (n, c, h, w) = (1usize, 2usize, 4usize, 5usize);
    let (oh, ow) = (8usize, 10usize);
    let dataf: Vec<f32> = (0..n * c * h * w)
        .map(|i| ((i % 13) as f32 - 6.0) * 0.5)
        .collect();
    let datad: Vec<f64> = dataf.iter().map(|&v| v as f64).collect();

    for mode in ["bilinear", "bicubic"] {
        let py = format!(
            r#"
import torch
d={dataf:?}
for dt,name in [(torch.float32,'f32'),(torch.float64,'f64')]:
    x=torch.tensor(d,dtype=dt).reshape({n},{c},{h},{w})
    o=torch.nn.functional.interpolate(x,size=({oh},{ow}),mode='{mode}',align_corners=False)
    print("VALS",name," ".join("%.9g"%v for v in o.flatten().tolist()))
"#,
            dataf = dataf,
            n = n,
            c = c,
            h = h,
            w = w,
            oh = oh,
            ow = ow,
            mode = mode
        );
        let mut ch = Command::new(&python)
            .arg("-")
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .spawn()?;
        ch.stdin.as_mut().unwrap().write_all(py.as_bytes())?;
        let pt = String::from_utf8_lossy(&ch.wait_with_output()?.stdout).to_string();
        let pv = |name: &str| -> Vec<f64> {
            pt.lines()
                .find_map(|l| l.strip_prefix(&format!("VALS {name} ")))
                .map(|s| {
                    s.split_whitespace()
                        .filter_map(|t| t.parse().ok())
                        .collect()
                })
                .unwrap_or_default()
        };
        // f32
        {
            let mut s = FrankenTorchSession::new(ExecutionMode::Strict);
            let x = s.tensor_variable_f32(dataf.clone(), vec![n, c, h, w], false)?;
            let o = s.tensor_interpolate(x, Some(vec![oh, ow]), None, mode, Some(false))?;
            let dt = s.tensor_dtype(o)?;
            let fv = s.tensor_values_lossy_f64(o)?;
            let p = pv("f32");
            let maxabs = fv
                .iter()
                .zip(&p)
                .map(|(a, b)| (a - b).abs())
                .fold(0.0f64, f64::max);
            let bitexact = fv.len() == p.len() && fv.iter().zip(&p).all(|(a, b)| a == b);
            println!(
                "{mode:>9} f32: dtype={dt:?} (torch=F32) max_abs={maxabs:.3e} bit_exact={bitexact}"
            );
        }
        // f64
        {
            let mut s = FrankenTorchSession::new(ExecutionMode::Strict);
            let x = s.tensor_variable(datad.clone(), vec![n, c, h, w], false)?;
            let o = s.tensor_interpolate(x, Some(vec![oh, ow]), None, mode, Some(false))?;
            let dt = s.tensor_dtype(o)?;
            let fv = s.tensor_values_lossy_f64(o)?;
            let p = pv("f64");
            let maxabs = fv
                .iter()
                .zip(&p)
                .map(|(a, b)| (a - b).abs())
                .fold(0.0f64, f64::max);
            let bitexact = fv.len() == p.len() && fv.iter().zip(&p).all(|(a, b)| a == b);
            println!("{mode:>9} f64: dtype={dt:?} max_abs={maxabs:.3e} bit_exact={bitexact}");
        }
    }

    // ---- perf: [8,16,128,128] -> x2 = [8,16,256,256] ----
    let (pn, pc, ph, pw) = (8usize, 16usize, 128usize, 128usize);
    let (poh, pow) = (256usize, 256usize);
    let pf: Vec<f32> = (0..pn * pc * ph * pw)
        .map(|i| ((i % 9973) as f32 - 5000.0) * 0.001)
        .collect();
    let pd: Vec<f64> = pf.iter().map(|&v| v as f64).collect();
    // which: 0=add anchor(f32) 1=bilinear 2=bicubic ; d=dtype 32/64
    let tt = |which: u8, d: u8| {
        let mut best = f64::INFINITY;
        for _ in 0..7 {
            let mut s = FrankenTorchSession::new(ExecutionMode::Strict);
            let x = if d == 32 {
                s.tensor_variable_f32(pf.clone(), vec![pn, pc, ph, pw], false)
                    .unwrap()
            } else {
                s.tensor_variable(pd.clone(), vec![pn, pc, ph, pw], false)
                    .unwrap()
            };
            let ti = Instant::now();
            let _ = match which {
                0 => s.tensor_add(x, x),
                1 => s.tensor_interpolate(x, Some(vec![poh, pow]), None, "bilinear", Some(false)),
                _ => s.tensor_interpolate(x, Some(vec![poh, pow]), None, "bicubic", Some(false)),
            };
            let e = ti.elapsed().as_secs_f64() * 1e3;
            if e < best {
                best = e;
            }
        }
        best
    };
    let py_b = format!(
        r#"
import time,torch
torch.set_num_threads(8)
def tm(fn,reps=7):
    for _ in range(2): fn()
    ts=[]
    for _ in range(reps): s=time.perf_counter(); fn(); ts.append((time.perf_counter()-s)*1e3)
    return min(ts)
for dt,name in [(torch.float32,'32'),(torch.float64,'64')]:
    x=(((torch.arange({pn}*{pc}*{ph}*{pw},dtype=torch.int64)%9973).to(dt)-5000.0)*0.001).reshape({pn},{pc},{ph},{pw})
    print("PT add%s %.4f"%(name,tm(lambda:x+x)))
    print("PT bil%s %.4f"%(name,tm(lambda:torch.nn.functional.interpolate(x,size=({poh},{pow}),mode='bilinear',align_corners=False))))
    print("PT bic%s %.4f"%(name,tm(lambda:torch.nn.functional.interpolate(x,size=({poh},{pow}),mode='bicubic',align_corners=False))))
"#,
        pn = pn,
        pc = pc,
        ph = ph,
        pw = pw,
        poh = poh,
        pow = pow
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
    println!("\nperf [8,16,128,128]->x2  (FT ms / PT ms)");
    for (lbl, w) in [("add", 0u8), ("bilinear", 1), ("bicubic", 2)] {
        for d in [32u8, 64] {
            let ft = tt(w, d);
            let key = format!("{}{}", &lbl[..3], d);
            println!(
                "  {lbl:<9} f{d}  FT {ft:8.3}  PT {:8.3}  => {}",
                g(&key),
                v(ft, g(&key))
            );
        }
    }
    Ok(())
}

// f32 head-to-head + correctness: log_ndtr, mvlgamma(p=3) vs torch (8t / FT default).
use ft_api::FrankenTorchSession;
use ft_core::{DType, ExecutionMode};
use std::io::Write;
use std::process::{Command, Stdio};
use std::time::Instant;
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let python = std::env::var("PYTORCH_PYTHON").unwrap_or_else(|_| "python3".to_string());
    let n = 16_000_000usize;
    // log_ndtr domain: all reals. mvlgamma(p=3) domain: x > (p-1)/2 = 1.0.
    let lx: Vec<f32> = (0..n).map(|i| ((i % 4001) as f32 / 1000.0) - 2.0).collect(); // (-2,2)
    let mx: Vec<f32> = (0..n).map(|i| 1.25 + (i % 4001) as f32 / 1000.0).collect(); // (1.25,5.25)
    let bench = |w: u8| {
        let mut best = f64::INFINITY;
        for _ in 0..7 {
            let mut s = FrankenTorchSession::new(ExecutionMode::Strict);
            let data = if w == 0 { lx.clone() } else { mx.clone() };
            let x = s.tensor_variable_f32(data, vec![n], false).unwrap();
            let t = Instant::now();
            match w {
                0 => {
                    let _ = s.tensor_special_log_ndtr(x);
                }
                _ => {
                    let _ = s.tensor_mvlgamma(x, 3);
                }
            }
            let e = t.elapsed().as_secs_f64() * 1e3;
            if e < best {
                best = e;
            }
        }
        best
    };
    // correctness on first 4096
    let m = 4096usize;
    let mut s = FrankenTorchSession::new(ExecutionMode::Strict);
    let xl = s.tensor_variable_f32(lx[..m].to_vec(), vec![m], false)?;
    let yl = s.tensor_special_log_ndtr(xl)?;
    let dl = s.tensor_dtype(yl)?;
    let vl: Vec<f32> = s
        .tensor_values_lossy_f64(yl)?
        .iter()
        .map(|&v| v as f32)
        .collect();
    let xm = s.tensor_variable_f32(mx[..m].to_vec(), vec![m], false)?;
    let ym = s.tensor_mvlgamma(xm, 3)?;
    let dm = s.tensor_dtype(ym)?;
    let vm: Vec<f32> = s
        .tensor_values_lossy_f64(ym)?
        .iter()
        .map(|&v| v as f32)
        .collect();

    let py = format!(
        r#"
import time,torch
torch.set_num_threads(8)
n={n}; m={m}
lx=(((torch.arange(n,dtype=torch.int64)%4001).float()/1000.0)-2.0)
mx=(1.25+(torch.arange(n,dtype=torch.int64)%4001).float()/1000.0)
def tm(fn,reps=7):
    for _ in range(2): fn()
    ts=[]
    for _ in range(reps): s=time.perf_counter(); fn(); ts.append((time.perf_counter()-s)*1e3)
    return min(ts)
print("PT log_ndtr %.3f"%tm(lambda:torch.special.log_ndtr(lx)))
print("PT mvlgamma %.3f"%tm(lambda:torch.mvlgamma(mx,3)))
yl=torch.special.log_ndtr(lx[:m]); ym=torch.mvlgamma(mx[:m],3)
assert yl.dtype==torch.float32 and ym.dtype==torch.float32
print("REF log_ndtr "+" ".join("%a"%float(v) for v in yl.tolist()))
print("REF mvlgamma "+" ".join("%a"%float(v) for v in ym.tolist()))
"#
    );
    let mut ch = Command::new(&python)
        .arg("-")
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .spawn()?;
    ch.stdin.as_mut().unwrap().write_all(py.as_bytes())?;
    let out = String::from_utf8_lossy(&ch.wait_with_output()?.stdout).to_string();
    let g = |k: &str| {
        out.lines()
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
    let vrb = |ft: f64, pp: f64| {
        if pp >= ft {
            format!("FT {:.2}x FASTER", pp / ft)
        } else {
            format!("FT {:.2}x SLOWER", ft / pp)
        }
    };
    let check = |lbl: &str, dt: DType, fv: &[f32]| {
        let line = out
            .lines()
            .find(|l| l.starts_with(&format!("REF {lbl} ")))
            .unwrap_or("");
        let tv: Vec<f32> = line
            .split_whitespace()
            .skip(2)
            .filter_map(|t| t.parse().ok())
            .collect();
        let mut max_abs = 0f32;
        let mut max_rel = 0f32;
        let mut exact = 0usize;
        for (&f, &t) in fv.iter().zip(tv.iter()) {
            if f.to_bits() == t.to_bits() {
                exact += 1;
            }
            let a = (f - t).abs();
            if a > max_abs {
                max_abs = a;
            }
            let r = if t.abs() > 0.0 { a / t.abs() } else { a };
            if r > max_rel {
                max_rel = r;
            }
        }
        println!(
            "  {lbl:<9} dtype={dt:?} bit_exact={exact}/{} max_abs={max_abs:.3e} max_rel={max_rel:.3e}",
            fv.len()
        );
    };
    println!("specfn3 ~16M f32 (torch 8t / FT default), min-of-7:");
    let (fl, fm) = (bench(0), bench(1));
    println!(
        "  log_ndtr  FT {fl:8.3}  PT {:8.3}  => {}",
        g("log_ndtr"),
        vrb(fl, g("log_ndtr"))
    );
    println!(
        "  mvlgamma  FT {fm:8.3}  PT {:8.3}  => {}",
        g("mvlgamma"),
        vrb(fm, g("mvlgamma"))
    );
    println!("correctness vs torch f32:");
    check("log_ndtr", dl, &vl);
    check("mvlgamma", dm, &vm);
    Ok(())
}

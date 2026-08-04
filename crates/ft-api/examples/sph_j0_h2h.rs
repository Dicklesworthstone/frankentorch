// spherical_bessel_j0 f32 head-to-head + correctness vs torch (8t / FT default).
use ft_api::FrankenTorchSession;
use ft_core::ExecutionMode;
use std::io::Write;
use std::process::{Command, Stdio};
use std::time::Instant;
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let python = std::env::var("PYTORCH_PYTHON").unwrap_or_else(|_| "python3".to_string());
    let n = 16_000_000usize;
    // include exact 0.0 occasionally (i%4001==2000 -> -... ; ensure a zero appears)
    let x: Vec<f32> = (0..n).map(|i| ((i % 4001) as f32 / 400.0) - 5.0).collect(); // (-5,5), hits 0 at i%4001==2000
    let bench = |w: u8| {
        let mut best = f64::INFINITY;
        for _ in 0..7 {
            let mut s = FrankenTorchSession::new(ExecutionMode::Strict);
            let t = s.tensor_variable_f32(x.clone(), vec![n], false).unwrap();
            let ti = Instant::now();
            if w == 0 {
                let _ = s.tensor_add(t, t);
            } else {
                let _ = s.tensor_special_spherical_bessel_j0(t);
            }
            let e = ti.elapsed().as_secs_f64() * 1e3;
            if e < best {
                best = e;
            }
        }
        best
    };
    let m = 4096usize;
    let mut s = FrankenTorchSession::new(ExecutionMode::Strict);
    let xt = s.tensor_variable_f32(x[..m].to_vec(), vec![m], false)?;
    let yt = s.tensor_special_spherical_bessel_j0(xt)?;
    let dt = s.tensor_dtype(yt)?;
    let fv: Vec<f32> = s
        .tensor_values_lossy_f64(yt)?
        .iter()
        .map(|&v| v as f32)
        .collect();

    let py = format!(
        r#"
import time,torch
torch.set_num_threads(8)
n={n}; m={m}
x=((torch.arange(n,dtype=torch.int64)%4001).float()/400.0-5.0)
def tm(fn,reps=7):
    for _ in range(2): fn()
    ts=[]
    for _ in range(reps): s=time.perf_counter(); fn(); ts.append((time.perf_counter()-s)*1e3)
    return min(ts)
print("PT add %.3f"%tm(lambda:x+x))
print("PT sph_j0 %.3f"%tm(lambda:torch.special.spherical_bessel_j0(x)))
y=torch.special.spherical_bessel_j0(x[:m]); assert y.dtype==torch.float32
print("REF "+" ".join("%a"%float(v) for v in y.tolist()))
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
    println!("sph_j0 ~16M f32 (torch 8t / FT default), min-of-7:");
    let (fa, fj) = (bench(0), bench(1));
    println!(
        "  add     FT {fa:8.3}  PT {:8.3}  => {}",
        g("add"),
        vrb(fa, g("add"))
    );
    println!(
        "  sph_j0  FT {fj:8.3}  PT {:8.3}  => {}",
        g("sph_j0"),
        vrb(fj, g("sph_j0"))
    );
    let line = out.lines().find(|l| l.starts_with("REF ")).unwrap_or("");
    let tv: Vec<f32> = line
        .split_whitespace()
        .skip(1)
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
        "correctness: sph_j0 dtype={dt:?} bit_exact={exact}/{} max_abs={max_abs:.3e} max_rel={max_rel:.3e}",
        fv.len()
    );
    Ok(())
}

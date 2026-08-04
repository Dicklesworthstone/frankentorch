// mul_scalar f32 vs torch (tensor * scalar) + value check.
use ft_api::FrankenTorchSession;
use ft_core::ExecutionMode;
use std::io::Write;
use std::process::{Command, Stdio};
use std::time::Instant;
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let python = std::env::var("PYTORCH_PYTHON").unwrap_or_else(|_| "python3".to_string());
    let n = 16_000_000usize;
    let a: Vec<f32> = (0..n).map(|i| ((i % 4001) as f32 / 500.0) - 4.0).collect();
    let bench = |w: u8| {
        let mut best = f64::INFINITY;
        for _ in 0..7 {
            let mut s = FrankenTorchSession::new(ExecutionMode::Strict);
            let x = s.tensor_variable_f32(a.clone(), vec![n], false).unwrap();
            let t = Instant::now();
            if w == 0 {
                let _ = s.tensor_add(x, x);
            } else {
                let _ = s.tensor_mul_scalar(x, 2.5);
            }
            let e = t.elapsed().as_secs_f64() * 1e3;
            if e < best {
                best = e;
            }
        }
        best
    };
    let m = 4096usize;
    let mut s = FrankenTorchSession::new(ExecutionMode::Strict);
    let xt = s.tensor_variable_f32(a[..m].to_vec(), vec![m], false)?;
    let yt = s.tensor_mul_scalar(xt, 2.5)?;
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
a=((torch.arange(n,dtype=torch.int64)%4001).float()/500.0-4.0)
def tm(fn,reps=7):
    for _ in range(2): fn()
    ts=[]
    for _ in range(reps): s=time.perf_counter(); fn(); ts.append((time.perf_counter()-s)*1e3)
    return min(ts)
print("PT add %.3f"%tm(lambda:a+a))
print("PT mul %.3f"%tm(lambda:a*2.5))
y=a[:m]*2.5; assert y.dtype==torch.float32
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
    let line = out.lines().find(|l| l.starts_with("REF ")).unwrap_or("");
    let tv: Vec<f32> = line
        .split_whitespace()
        .skip(1)
        .filter_map(|t| t.parse().ok())
        .collect();
    let exact = fv
        .iter()
        .zip(tv.iter())
        .filter(|(a, b)| a.to_bits() == b.to_bits())
        .count();
    println!("mul_scalar ~16M f32 (torch 8t / FT default), min-of-7:");
    let (fa, fm) = (bench(0), bench(1));
    println!(
        "  add        FT {fa:8.3}  PT {:8.3}  => {}",
        g("add"),
        vrb(fa, g("add"))
    );
    println!(
        "  mul_scalar FT {fm:8.3}  PT {:8.3}  => {}",
        g("mul"),
        vrb(fm, g("mul"))
    );
    println!("  value: dtype={dt:?} bit_exact={exact}/{}", fv.len());
    Ok(())
}

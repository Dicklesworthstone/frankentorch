// nextafter f32 head-to-head + BIT-EXACT vs torch (value op, parity absolute).
use ft_api::FrankenTorchSession;
use ft_core::ExecutionMode;
use std::io::Write;
use std::process::{Command, Stdio};
use std::time::Instant;
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let python = std::env::var("PYTORCH_PYTHON").unwrap_or_else(|_| "python3".to_string());
    let n = 16_000_000usize;
    let a: Vec<f32> = (0..n).map(|i| ((i % 4001) as f32 / 500.0) - 4.0).collect();
    let b: Vec<f32> = (0..n).map(|i| 0.1 + (i % 3997) as f32 / 500.0).collect();
    let bench = |w: u8| {
        let mut best = f64::INFINITY;
        for _ in 0..7 {
            let mut s = FrankenTorchSession::new(ExecutionMode::Strict);
            let x = s.tensor_variable_f32(a.clone(), vec![n], false).unwrap();
            let y = s.tensor_variable_f32(b.clone(), vec![n], false).unwrap();
            let t = Instant::now();
            if w == 0 {
                let _ = s.tensor_add(x, y);
            } else {
                let _ = s.tensor_nextafter(x, y);
            }
            let e = t.elapsed().as_secs_f64() * 1e3;
            if e < best {
                best = e;
            }
        }
        best
    };
    // correctness incl edge values: 0, equal, x>y, x<y, negatives
    let m = 4096usize;
    let xa: Vec<f32> = (0..m).map(|i| ((i as i32 - 2048) as f32) / 64.0).collect();
    let xb: Vec<f32> = (0..m)
        .map(|i| {
            if i % 7 == 0 {
                xa[i]
            } else {
                ((i as i32 - 1024) as f32) / 32.0
            }
        })
        .collect();
    let mut s = FrankenTorchSession::new(ExecutionMode::Strict);
    let xt = s.tensor_variable_f32(xa.clone(), vec![m], false)?;
    let yt = s.tensor_variable_f32(xb.clone(), vec![m], false)?;
    let zt = s.tensor_nextafter(xt, yt)?;
    let dt = s.tensor_dtype(zt)?;
    let fv: Vec<f32> = s
        .tensor_values_lossy_f64(zt)?
        .iter()
        .map(|&v| v as f32)
        .collect();
    let xa_s = xa
        .iter()
        .map(|v| v.to_bits().to_string())
        .collect::<Vec<_>>()
        .join(" ");
    let xb_s = xb
        .iter()
        .map(|v| v.to_bits().to_string())
        .collect::<Vec<_>>()
        .join(" ");
    let py = format!(
        r#"
import time,torch
torch.set_num_threads(8)
n={n}; m={m}
a=((torch.arange(n,dtype=torch.int64)%4001).float()/500.0-4.0)
b=(0.1+(torch.arange(n,dtype=torch.int64)%3997).float()/500.0)
def tm(fn,reps=7):
    for _ in range(2): fn()
    ts=[]
    for _ in range(reps): s=time.perf_counter(); fn(); ts.append((time.perf_counter()-s)*1e3)
    return min(ts)
print("PT add %.3f"%tm(lambda:a+b))
print("PT nextafter %.3f"%tm(lambda:torch.nextafter(a,b)))
import struct
def fb(t): return struct.unpack('<f', struct.pack('<I', int(t)))[0]
xa=torch.tensor([fb(t) for t in "{xa}".split()],dtype=torch.float32)
xb=torch.tensor([fb(t) for t in "{xb}".split()],dtype=torch.float32)
z=torch.nextafter(xa,xb); assert z.dtype==torch.float32
print("REF "+" ".join("%a"%float(v) for v in z.tolist()))
"#,
        n = n,
        m = m,
        xa = xa_s,
        xb = xb_s
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
    println!("nextafter ~16M f32 (torch 8t / FT default), min-of-7:");
    let (fa, fj) = (bench(0), bench(1));
    println!(
        "  add        FT {fa:8.3}  PT {:8.3}  => {}",
        g("add"),
        vrb(fa, g("add"))
    );
    println!(
        "  nextafter  FT {fj:8.3}  PT {:8.3}  => {}",
        g("nextafter"),
        vrb(fj, g("nextafter"))
    );
    let line = out.lines().find(|l| l.starts_with("REF ")).unwrap_or("");
    let tv: Vec<f32> = line
        .split_whitespace()
        .skip(1)
        .filter_map(|t| t.parse().ok())
        .collect();
    let mut exact = 0usize;
    let mut mism = 0usize;
    for (&f, &t) in fv.iter().zip(tv.iter()) {
        if f.to_bits() == t.to_bits() {
            exact += 1;
        } else {
            mism += 1;
        }
    }
    println!(
        "correctness: nextafter dtype={dt:?} bit_exact={exact}/{} mism={mism} (torch_ref_len={})",
        fv.len(),
        tv.len()
    );
    Ok(())
}

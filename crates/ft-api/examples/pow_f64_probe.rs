//! f64 pow: measure pow2 vs torch + verify trivial-exponent bit-exactness vs torch f64.
use ft_api::FrankenTorchSession;
use ft_autograd::TensorNodeId;
use ft_core::ExecutionMode;
use std::io::Write;
use std::process::{Command, Stdio};
use std::time::Instant;
const R: usize = 4000;
const C: usize = 4000;

fn t1<F: Fn(&mut FrankenTorchSession, TensorNodeId)>(a: &[f64], f: F) -> f64 {
    let mut best = f64::INFINITY;
    for _ in 0..7 {
        let mut s = FrankenTorchSession::new(ExecutionMode::Strict);
        let x = s.tensor_variable(a.to_vec(), vec![R, C], false).unwrap();
        let t = Instant::now();
        f(&mut s, x);
        let e = t.elapsed().as_secs_f64() * 1e3;
        if e < best {
            best = e;
        }
    }
    best
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let python = std::env::var("PYTORCH_PYTHON").unwrap_or_else(|_| "python3".to_string());
    // ---- parity sweep (small) ----
    let mut pa: Vec<f64> = Vec::new();
    for i in 0..20000u32 {
        pa.push((i as f64 - 10000.0) * 0.0007);
    }
    pa.extend_from_slice(&[
        0.0,
        -0.0,
        1.0,
        -1.0,
        2.0,
        0.5,
        1e-300,
        1e300,
        f64::INFINITY,
        f64::NEG_INFINITY,
        f64::NAN,
    ]);
    let pn = pa.len();
    let mut s = FrankenTorchSession::new(ExecutionMode::Strict);
    let x = s.tensor_variable(pa.clone(), vec![pn], false)?;
    let np = s.tensor_pow(x, 2.0)?;
    let ftpow = s.tensor_values(np)?;
    let xx: Vec<f64> = pa.iter().map(|&v| v * v).collect();
    let cube: Vec<f64> = pa.iter().map(|&v| v * v * v).collect();
    let recip: Vec<f64> = pa.iter().map(|&v| 1.0 / v).collect();
    let sq: Vec<f64> = pa.iter().map(|&v| v.sqrt()).collect();

    let bits: Vec<u64> = pa.iter().map(|x| x.to_bits()).collect();
    let py = format!(
        r#"
import struct,torch
bits={:?}
a=torch.tensor([struct.unpack('<d',struct.pack('<Q',b))[0] for b in bits],dtype=torch.float64)
def emit(t): print(' '.join(str(struct.unpack('<Q',struct.pack('<d',v))[0]) for v in t.tolist()))
emit(a.pow(2.0)); emit(a.pow(3.0)); emit(a.pow(-1.0)); emit(a.pow(0.5))
"#,
        bits
    );
    let mut ch = Command::new(&python)
        .arg("-")
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .spawn()?;
    ch.stdin.as_mut().unwrap().write_all(py.as_bytes())?;
    let o = ch.wait_with_output()?;
    let out = String::from_utf8_lossy(&o.stdout);
    let l: Vec<Vec<u64>> = out
        .lines()
        .map(|x| {
            x.split_whitespace()
                .filter_map(|t| t.parse().ok())
                .collect()
        })
        .collect();
    let cmp = |name: &str, v: &[f64], pt: &[u64]| {
        let mut mm = 0;
        let mut first = None;
        for i in 0..v.len() {
            let fb = v[i].to_bits();
            let eq = fb == pt[i] || (f64::from_bits(fb).is_nan() && f64::from_bits(pt[i]).is_nan());
            if !eq {
                mm += 1;
                if first.is_none() {
                    first = Some((i, fb, pt[i]));
                }
            }
        }
        println!("{name:<16}: {mm}/{} mismatches  {:?}", v.len(), first);
    };
    cmp("ft_pow(2)", &ftpow, &l[0]);
    cmp("x*x vs pow2", &xx, &l[0]);
    cmp("x*x*x vs pow3", &cube, &l[1]);
    cmp("1/x vs pow-1", &recip, &l[2]);
    cmp("sqrt vs pow.5", &sq, &l[3]);

    // ---- perf ----
    let big: Vec<f64> = (0..R * C)
        .map(|i| ((i * 31 % 9973) as f64 - 4000.0) * 0.01)
        .collect();
    let pyp = r#"
import time,torch
torch.set_num_threads(8)
R,C=4000,4000
a=(((torch.arange(R*C,dtype=torch.int64)*31%9973).double()-4000.0)*0.01).reshape(R,C)
def t(fn,n=7):
    for _ in range(2): fn()
    ts=[]
    for _ in range(n): s=time.perf_counter(); fn(); ts.append((time.perf_counter()-s)*1e3)
    return min(ts)
print("PT pow2 %.4f"%t(lambda:a.pow(2.0)))
"#;
    let mut ch2 = Command::new(&python)
        .arg("-")
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .spawn()?;
    ch2.stdin.as_mut().unwrap().write_all(pyp.as_bytes())?;
    let o2 = ch2.wait_with_output()?;
    let pt2 = String::from_utf8_lossy(&o2.stdout);
    let ptv: f64 = pt2
        .lines()
        .find_map(|l| l.strip_prefix("PT pow2 ")?.trim().parse().ok())
        .unwrap_or(0.0);
    let ftv = t1(&big, |s, x| {
        let _ = s.tensor_pow(x, 2.0);
    });
    let r = ptv / ftv;
    let tag = if r >= 1.0 {
        format!("FT {r:.2}x FASTER")
    } else {
        format!("FT {:.2}x SLOWER", 1.0 / r)
    };
    println!("pow2_f64: FT {ftv:.3}ms  PT {ptv:.3}ms  {tag}");
    Ok(())
}

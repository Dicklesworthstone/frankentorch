//! gcd/lcm FT vs torch on large integer tensors. tensor_gcd/tensor_lcm build the
//! result with a SERIAL .iter().zip().map() over a compute-bound Euclidean loop.
//! Integer-exact + elementwise -> parallelizing is trivially bit-exact.
use ft_api::FrankenTorchSession;
use ft_core::ExecutionMode;
use std::io::Write;
use std::process::{Command, Stdio};
use std::time::Instant;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let python = std::env::var("PYTORCH_PYTHON").unwrap_or_else(|_| "python3".to_string());
    let n: usize = std::env::var("N")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(8_000_000);
    // integer-valued f64 data (gcd reads as i64)
    let a: Vec<f64> = (0..n)
        .map(|i| ((i as u64 * 2654435761 % 1_000_000) + 1) as f64)
        .collect();
    let b: Vec<f64> = (0..n)
        .map(|i| ((i as u64 * 40503 % 700_000) + 1) as f64)
        .collect();
    // 0=add anchor 1=gcd 2=lcm
    let tt = |which: u8| {
        let mut best = f64::INFINITY;
        for _ in 0..7 {
            let mut s = FrankenTorchSession::new(ExecutionMode::Strict);
            let x = s.tensor_variable(a.clone(), vec![n], false).unwrap();
            let y = s.tensor_variable(b.clone(), vec![n], false).unwrap();
            let ti = Instant::now();
            match which {
                0 => {
                    let _ = s.tensor_add(x, y);
                }
                1 => {
                    let _ = s.tensor_gcd(x, y);
                }
                _ => {
                    let _ = s.tensor_lcm(x, y);
                }
            }
            let e = ti.elapsed().as_secs_f64() * 1e3;
            if e < best {
                best = e;
            }
        }
        best
    };
    // parity check vs torch (small)
    let mut s = FrankenTorchSession::new(ExecutionMode::Strict);
    let xa = s.tensor_variable(a[..1000].to_vec(), vec![1000], false)?;
    let xb = s.tensor_variable(b[..1000].to_vec(), vec![1000], false)?;
    let gnode = s.tensor_gcd(xa, xb)?;
    let g = s.tensor_values(gnode)?;
    let lnode = s.tensor_lcm(xa, xb)?;
    let l = s.tensor_values(lnode)?;
    let py = format!(
        r#"
import time, torch
torch.set_num_threads(8)
n={n}
a=((torch.arange(n,dtype=torch.int64)*2654435761 % 1000000)+1)
b=((torch.arange(n,dtype=torch.int64)*40503 % 700000)+1)
def tm(fn,reps=7):
    for _ in range(2): fn()
    ts=[]
    for _ in range(reps): s=time.perf_counter(); fn(); ts.append((time.perf_counter()-s)*1e3)
    return min(ts)
print("PT add %.3f"%tm(lambda:a+b))
print("PT gcd %.3f"%tm(lambda:torch.gcd(a,b)))
print("PT lcm %.3f"%tm(lambda:torch.lcm(a,b)))
g=torch.gcd(a[:1000],b[:1000]).tolist(); l=torch.lcm(a[:1000],b[:1000]).tolist()
print("GCDVALS"," ".join(str(v) for v in g))
print("LCMVALS"," ".join(str(v) for v in l))
"#,
        n = n
    );
    let mut ch = Command::new(&python)
        .arg("-")
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .spawn()?;
    ch.stdin.as_mut().unwrap().write_all(py.as_bytes())?;
    let pt = String::from_utf8_lossy(&ch.wait_with_output()?.stdout).to_string();
    let pv = |tag: &str| -> Vec<f64> {
        pt.lines()
            .find_map(|l| l.strip_prefix(tag))
            .map(|s| {
                s.split_whitespace()
                    .filter_map(|t| t.parse().ok())
                    .collect()
            })
            .unwrap_or_default()
    };
    let pg = pv("GCDVALS ");
    let pl = pv("LCMVALS ");
    let gok = pg.len() == g.len() && g.iter().zip(&pg).all(|(x, y)| x == y);
    let lok = pl.len() == l.len() && l.iter().zip(&pl).all(|(x, y)| x == y);
    println!("parity (1000): gcd bit-exact={gok}  lcm bit-exact={lok}");
    let getp = |k: &str| {
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
    println!("gcd/lcm, N={n} (torch 8t / FT default), min-of-7");
    for (lbl, w) in [("add", 0u8), ("gcd", 1), ("lcm", 2)] {
        let ft = tt(w);
        println!(
            "  {lbl:<5} FT {ft:9.3}  PT {:9.3}  => {}",
            getp(lbl),
            v(ft, getp(lbl))
        );
    }
    Ok(())
}

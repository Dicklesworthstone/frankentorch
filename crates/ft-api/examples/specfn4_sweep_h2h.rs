// f32 head-to-head + correctness: igamma/igammac/zeta (binary, equal-shape) vs torch (8t / FT default).
use ft_api::FrankenTorchSession;
use ft_core::{DType, ExecutionMode};
use std::io::Write;
use std::process::{Command, Stdio};
use std::time::Instant;
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let python = std::env::var("PYTORCH_PYTHON").unwrap_or_else(|_| "python3".to_string());
    let n = 16_000_000usize;
    // a>0, x>=0 for igamma/igammac; s>1, q>0 for zeta.
    let av: Vec<f32> = (0..n).map(|i| 0.5 + (i % 4001) as f32 / 800.0).collect(); // (0.5,5.5)
    let xv: Vec<f32> = (0..n).map(|i| (i % 4001) as f32 / 400.0).collect(); // (0,10)
    let sv: Vec<f32> = (0..n).map(|i| 1.5 + (i % 4001) as f32 / 1000.0).collect(); // (1.5,5.5)
    let qv: Vec<f32> = (0..n).map(|i| 0.5 + (i % 4001) as f32 / 1000.0).collect(); // (0.5,4.5)
    let bench = |w: u8| {
        let mut best = f64::INFINITY;
        for _ in 0..7 {
            let mut s = FrankenTorchSession::new(ExecutionMode::Strict);
            let (p, q) = match w {
                2 => (sv.clone(), qv.clone()),
                _ => (av.clone(), xv.clone()),
            };
            let pa = s.tensor_variable_f32(p, vec![n], false).unwrap();
            let pb = s.tensor_variable_f32(q, vec![n], false).unwrap();
            let t = Instant::now();
            match w {
                0 => {
                    let _ = s.tensor_igamma(pa, pb);
                }
                1 => {
                    let _ = s.tensor_igammac(pa, pb);
                }
                _ => {
                    let _ = s.tensor_zeta(pa, pb);
                }
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
    let mk = |s: &mut FrankenTorchSession, d: &[f32]| {
        s.tensor_variable_f32(d[..m].to_vec(), vec![m], false)
            .unwrap()
    };
    let ai = mk(&mut s, &av);
    let xi = mk(&mut s, &xv);
    let yg = s.tensor_igamma(ai, xi)?;
    let dg = s.tensor_dtype(yg)?;
    let vg: Vec<f32> = s
        .tensor_values_lossy_f64(yg)?
        .iter()
        .map(|&v| v as f32)
        .collect();
    let ai2 = mk(&mut s, &av);
    let xi2 = mk(&mut s, &xv);
    let yc = s.tensor_igammac(ai2, xi2)?;
    let dc = s.tensor_dtype(yc)?;
    let vc: Vec<f32> = s
        .tensor_values_lossy_f64(yc)?
        .iter()
        .map(|&v| v as f32)
        .collect();
    let si = mk(&mut s, &sv);
    let qi = mk(&mut s, &qv);
    let yz = s.tensor_zeta(si, qi)?;
    let dz = s.tensor_dtype(yz)?;
    let vz: Vec<f32> = s
        .tensor_values_lossy_f64(yz)?
        .iter()
        .map(|&v| v as f32)
        .collect();

    let py = format!(
        r#"
import time,torch
torch.set_num_threads(8)
n={n}; m={m}
av=(0.5+(torch.arange(n,dtype=torch.int64)%4001).float()/800.0)
xv=((torch.arange(n,dtype=torch.int64)%4001).float()/400.0)
sv=(1.5+(torch.arange(n,dtype=torch.int64)%4001).float()/1000.0)
qv=(0.5+(torch.arange(n,dtype=torch.int64)%4001).float()/1000.0)
def tm(fn,reps=7):
    for _ in range(2): fn()
    ts=[]
    for _ in range(reps): s=time.perf_counter(); fn(); ts.append((time.perf_counter()-s)*1e3)
    return min(ts)
print("PT igamma %.3f"%tm(lambda:torch.special.gammainc(av,xv)))
print("PT igammac %.3f"%tm(lambda:torch.special.gammaincc(av,xv)))
print("PT zeta %.3f"%tm(lambda:torch.special.zeta(sv,qv)))
yg=torch.special.gammainc(av[:m],xv[:m]); yc=torch.special.gammaincc(av[:m],xv[:m]); yz=torch.special.zeta(sv[:m],qv[:m])
for nm,y in [("igamma",yg),("igammac",yc),("zeta",yz)]:
    assert y.dtype==torch.float32, (nm,y.dtype)
    print("REF %s "%nm+" ".join("%a"%float(v) for v in y.tolist()))
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
        let mut cmp = 0usize;
        for (&f, &t) in fv.iter().zip(tv.iter()) {
            if !f.is_finite() || !t.is_finite() {
                continue;
            }
            cmp += 1;
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
            "  {lbl:<8} dtype={dt:?} bit_exact={exact}/{cmp} max_abs={max_abs:.3e} max_rel={max_rel:.3e}"
        );
    };
    println!("specfn4 ~16M f32 (torch 8t / FT default), min-of-7:");
    for (lbl, w) in [("igamma", 0u8), ("igammac", 1), ("zeta", 2)] {
        let ft = bench(w);
        println!(
            "  {lbl:<8} FT {ft:8.3}  PT {:8.3}  => {}",
            g(lbl),
            vrb(ft, g(lbl))
        );
    }
    println!("correctness vs torch f32 (finite-only):");
    check("igamma", dg, &vg);
    check("igammac", dc, &vc);
    check("zeta", dz, &vz);
    Ok(())
}

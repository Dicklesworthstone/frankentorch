// Broad gap-finder: measure a handful of unmeasured/compose-heavy ops vs torch (8t / FT default).
use ft_api::FrankenTorchSession;
use ft_core::ExecutionMode;
use std::io::Write;
use std::process::{Command, Stdio};
use std::time::Instant;
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let python = std::env::var("PYTORCH_PYTHON").unwrap_or_else(|_| "python3".to_string());
    let n = 16_000_000usize;
    let pos: Vec<f32> = (0..n).map(|i| 0.5 + (i % 4001) as f32 / 500.0).collect(); // (0.5,8.5)
    let spread: Vec<f32> = (0..n).map(|i| ((i % 4001) as f32 / 400.0) - 5.0).collect(); // (-5,5)
    let bench = |w: u8| {
        let mut best = f64::INFINITY;
        for _ in 0..7 {
            let mut s = FrankenTorchSession::new(ExecutionMode::Strict);
            let data = if matches!(w, 1 | 2) {
                pos.clone()
            } else {
                spread.clone()
            };
            let x = s.tensor_variable_f32(data, vec![n], false).unwrap();
            let t = Instant::now();
            match w {
                0 => {
                    let _ = s.tensor_add(x, x);
                } // anchor
                1 => {
                    let _ = s.tensor_special_spherical_bessel_j0(x);
                }
                2 => {
                    let _ = s.tensor_polygamma(5, x);
                }
                3 => {
                    let _ = s.tensor_special_airy_ai(x);
                }
                4 => {
                    let _ = s.tensor_special_scaled_modified_bessel_k0(x);
                } // shipped-fast anchor
                _ => {
                    let _ = s.tensor_digamma(x);
                } // shipped-fast anchor
            }
            let e = t.elapsed().as_secs_f64() * 1e3;
            if e < best {
                best = e;
            }
        }
        best
    };
    let py = format!(
        r#"
import time,torch
torch.set_num_threads(8)
n={n}
pos=(0.5+(torch.arange(n,dtype=torch.int64)%4001).float()/500.0)
spread=((torch.arange(n,dtype=torch.int64)%4001).float()/400.0-5.0)
def tm(fn,reps=7):
    for _ in range(2): fn()
    ts=[]
    for _ in range(reps): s=time.perf_counter(); fn(); ts.append((time.perf_counter()-s)*1e3)
    return min(ts)
print("PT add %.3f"%tm(lambda:spread+spread))
print("PT sph_j0 %.3f"%tm(lambda:torch.special.spherical_bessel_j0(spread)))
print("PT polyg5 %.3f"%tm(lambda:torch.special.polygamma(5,pos)))
print("PT airy %.3f"%tm(lambda:torch.special.airy_ai(spread)))
print("PT sk0 %.3f"%tm(lambda:torch.special.scaled_modified_bessel_k0(pos)))
print("PT digamma %.3f"%tm(lambda:torch.special.digamma(spread)))
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
    println!("gapfind ~16M f32 (torch 8t / FT default), min-of-7:");
    for (lbl, w) in [
        ("add", 0u8),
        ("sph_j0", 1),
        ("polyg5", 2),
        ("airy", 3),
        ("sk0", 4),
        ("digamma", 5),
    ] {
        let ft = bench(w);
        println!(
            "  {lbl:<9} FT {ft:8.3}  PT {:8.3}  => {}",
            g(lbl),
            vrb(ft, g(lbl))
        );
    }
    Ok(())
}

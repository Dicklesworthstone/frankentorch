use ft_api::FrankenTorchSession;
use ft_core::ExecutionMode;
use std::io::Write;
use std::process::{Command, Stdio};
use std::time::Instant;
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let python = std::env::var("PYTORCH_PYTHON").unwrap_or_else(|_| "python3".to_string());
    let n = 16_000_000usize;
    let a: Vec<f32> = (0..n).map(|i| ((i % 9973) as f32) * 0.001 + 0.01).collect();
    let b: Vec<f32> = (0..n).map(|i| ((i % 7919) as f32) * 0.001 + 0.01).collect();
    let tt = |w: u8| {
        let mut best = f64::INFINITY;
        for _ in 0..7 {
            let mut s = FrankenTorchSession::new(ExecutionMode::Strict);
            let x = s.tensor_variable_f32(a.clone(), vec![n], false).unwrap();
            let y = s.tensor_variable_f32(b.clone(), vec![n], false).unwrap();
            let ti = Instant::now();
            match w {
                0 => {
                    let _ = s.tensor_entr(x);
                }
                1 => {
                    let _ = s.tensor_xlog1py(x, y);
                }
                _ => {
                    let _ = s.tensor_rel_entr(x, y);
                }
            }
            let e = ti.elapsed().as_secs_f64() * 1e3;
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
a=(((torch.arange(n,dtype=torch.int64)%9973).float())*0.001+0.01)
b=(((torch.arange(n,dtype=torch.int64)%7919).float())*0.001+0.01)
def tm(fn,reps=7):
    for _ in range(2): fn()
    ts=[]
    for _ in range(reps): s=time.perf_counter(); fn(); ts.append((time.perf_counter()-s)*1e3)
    return min(ts)
print("PT entr %.3f"%tm(lambda:torch.special.entr(a)))
print("PT xlog1py %.3f"%tm(lambda:torch.special.xlog1py(a,b)))
print("PT rel_entr %.3f"%tm(lambda:torch.special.rel_entr(a,b)))
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
    println!("entr/xlog1py/rel_entr ~16M f32 (torch 8t / FT default), min-of-7");
    for (lbl, w) in [("entr", 0u8), ("xlog1py", 1), ("rel_entr", 2)] {
        let ft = tt(w);
        println!(
            "  {lbl:<9} FT {ft:8.3}  PT {:8.3}  => {}",
            g(lbl),
            v(ft, g(lbl))
        );
    }
    Ok(())
}

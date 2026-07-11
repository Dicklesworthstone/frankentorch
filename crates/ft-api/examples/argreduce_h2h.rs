use ft_api::FrankenTorchSession;
use ft_core::ExecutionMode;
use std::io::Write;
use std::process::{Command, Stdio};
use std::time::Instant;
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let python = std::env::var("PYTORCH_PYTHON").unwrap_or_else(|_| "python3".to_string());
    let m = 4096usize;
    let mat: Vec<f32> = (0..m * m)
        .map(|i| ((i * 7919) % 1_000_003) as f32 * 0.001)
        .collect();
    let tt = |w: u8| {
        let mut best = f64::INFINITY;
        for _ in 0..7 {
            let mut s = FrankenTorchSession::new(ExecutionMode::Strict);
            let x = s
                .tensor_variable_f32(mat.clone(), vec![m, m], false)
                .unwrap();
            let ti = Instant::now();
            match w {
                0 => {
                    let _ = s.tensor_add(x, x);
                }
                1 => {
                    let _ = s.tensor_argmax(x, 1);
                }
                2 => {
                    let _ = s.tensor_argmin(x, 1);
                }
                3 => {
                    let _ = s.tensor_argmax(x, 0);
                }
                _ => {
                    let _ = s.tensor_logsumexp(x, 1);
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
m={m}
mat=(((torch.arange(m*m,dtype=torch.int64)*7919)%1000003).float()*0.001).reshape(m,m)
def tm(fn,reps=7):
    for _ in range(2): fn()
    ts=[]
    for _ in range(reps): s=time.perf_counter(); fn(); ts.append((time.perf_counter()-s)*1e3)
    return min(ts)
print("PT add %.3f"%tm(lambda:mat+mat))
print("PT argmax1 %.3f"%tm(lambda:torch.argmax(mat,1)))
print("PT argmin1 %.3f"%tm(lambda:torch.argmin(mat,1)))
print("PT argmax0 %.3f"%tm(lambda:torch.argmax(mat,0)))
print("PT logsumexp %.3f"%tm(lambda:torch.logsumexp(mat,1)))
"#,
        m = m
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
    println!("arg/reduce [4096,4096] (torch 8t / FT default), min-of-7");
    for (lbl, w) in [
        ("add", 0u8),
        ("argmax1", 1),
        ("argmin1", 2),
        ("argmax0", 3),
        ("logsumexp1", 4),
    ] {
        let ft = tt(w);
        println!(
            "  {lbl:<11} FT {ft:8.3}  PT {:8.3}  => {}",
            g(lbl),
            v(ft, g(lbl))
        );
    }
    Ok(())
}

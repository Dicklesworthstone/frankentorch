//! f32 norm-dim round-trip probe (softmax-class). norm over a small dim -> large
//! output; compute-bound (sum x^2 + sqrt). add = anchor.
use ft_api::FrankenTorchSession;
use ft_core::ExecutionMode;
use std::io::Write;
use std::process::{Command, Stdio};
use std::time::Instant;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let python = std::env::var("PYTORCH_PYTHON").unwrap_or_else(|_| "python3".to_string());
    let (r, c) = (4096usize, 2048usize);
    let d3: Vec<f32> = (0..r * c * 2)
        .map(|i| ((i % 9973) as f32 - 5000.0) * 0.001)
        .collect();
    let tt = |which: u8| {
        let mut best = f64::INFINITY;
        for _ in 0..7 {
            let mut s = FrankenTorchSession::new(ExecutionMode::Strict);
            let x = s
                .tensor_variable_f32(d3.clone(), vec![r, c, 2], false)
                .unwrap();
            let ti = Instant::now();
            match which {
                0 => {
                    let _ = s.tensor_add(x, x);
                }
                1 => {
                    let _ = s.tensor_norm_dim(x, 2.0, 2);
                }
                _ => {
                    let _ = s.tensor_norm_dim(x, 1.0, 2);
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
import time, torch
torch.set_num_threads(8)
r,c={r},{c}
x3=(((torch.arange(r*c*2,dtype=torch.int64)%9973).float()-5000.0)*0.001).reshape(r,c,2)
def tm(fn,reps=7):
    for _ in range(2): fn()
    ts=[]
    for _ in range(reps): s=time.perf_counter(); fn(); ts.append((time.perf_counter()-s)*1e3)
    return min(ts)
print("PT add %.3f"%tm(lambda:x3+x3))
print("PT norm2 %.3f"%tm(lambda:torch.linalg.vector_norm(x3,ord=2,dim=2)))
print("PT norm1 %.3f"%tm(lambda:torch.linalg.vector_norm(x3,ord=1,dim=2)))
"#,
        r = r,
        c = c
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
    println!("f32 norm_dim [{r},{c},2] dim=2 (torch 8t / FT default), min-of-7");
    for (lbl, w) in [("add", 0u8), ("norm2", 1), ("norm1", 2)] {
        let ft = tt(w);
        println!(
            "  {lbl:<7} FT {ft:8.3}  PT {:8.3}  => {}",
            g(lbl),
            v(ft, g(lbl))
        );
    }
    Ok(())
}

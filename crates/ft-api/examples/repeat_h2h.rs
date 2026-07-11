//! repeat (tile) H2H vs torch. repeat materializes a large replicated output =
//! sequential-write, page-fault-bound (the cat-win class). Current ft-api fast path
//! is F64 + LAST-DIM-only; f32 + general patterns fall to the slow tape unravel.
use ft_api::FrankenTorchSession;
use ft_core::ExecutionMode;
use std::io::Write;
use std::process::{Command, Stdio};
use std::time::Instant;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let python = std::env::var("PYTORCH_PYTHON").unwrap_or_else(|_| "python3".to_string());
    let (r, c) = (2000usize, 2000usize);
    let d64: Vec<f64> = (0..r * c)
        .map(|i| ((i % 9973) as f64 - 5000.0) * 0.01)
        .collect();
    let d32: Vec<f32> = d64.iter().map(|&v| v as f32).collect();
    // which: 0 add anchor(f32), 1 f32 last-dim [1,4], 2 f32 dim0 [4,1],
    //        3 f64 last-dim [1,4], 4 f64 dim0 [4,1]
    let tt = |which: u8| {
        let mut best = f64::INFINITY;
        for _ in 0..7 {
            let mut s = FrankenTorchSession::new(ExecutionMode::Strict);
            let ti;
            match which {
                0 => {
                    let x = s
                        .tensor_variable_f32(d32.clone(), vec![r, c], false)
                        .unwrap();
                    ti = Instant::now();
                    let _ = s.tensor_add(x, x);
                }
                1 => {
                    let x = s
                        .tensor_variable_f32(d32.clone(), vec![r, c], false)
                        .unwrap();
                    ti = Instant::now();
                    let _ = s.tensor_repeat(x, &[1, 4]);
                }
                2 => {
                    let x = s
                        .tensor_variable_f32(d32.clone(), vec![r, c], false)
                        .unwrap();
                    ti = Instant::now();
                    let _ = s.tensor_repeat(x, &[4, 1]);
                }
                3 => {
                    let x = s.tensor_variable(d64.clone(), vec![r, c], false).unwrap();
                    ti = Instant::now();
                    let _ = s.tensor_repeat(x, &[1, 4]);
                }
                _ => {
                    let x = s.tensor_variable(d64.clone(), vec![r, c], false).unwrap();
                    ti = Instant::now();
                    let _ = s.tensor_repeat(x, &[4, 1]);
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
x32=(((torch.arange(r*c,dtype=torch.int64)%9973).float()-5000.0)*0.01).reshape(r,c)
x64=x32.double()
def tm(fn,reps=7):
    for _ in range(2): fn()
    ts=[]
    for _ in range(reps): s=time.perf_counter(); fn(); ts.append((time.perf_counter()-s)*1e3)
    return min(ts)
print("PT add %.3f"%tm(lambda:x32+x32))
print("PT f32last %.3f"%tm(lambda:x32.repeat(1,4)))
print("PT f32dim0 %.3f"%tm(lambda:x32.repeat(4,1)))
print("PT f64last %.3f"%tm(lambda:x64.repeat(1,4)))
print("PT f64dim0 %.3f"%tm(lambda:x64.repeat(4,1)))
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
    println!("repeat [{r},{c}] -> 16M out (torch 8t / FT default), min-of-7");
    for (lbl, w) in [
        ("add", 0u8),
        ("f32last", 1),
        ("f32dim0", 2),
        ("f64last", 3),
        ("f64dim0", 4),
    ] {
        let ft = tt(w);
        println!(
            "  {lbl:<8} FT {ft:8.3}  PT {:8.3}  => {}",
            g(lbl),
            v(ft, g(lbl))
        );
    }
    Ok(())
}

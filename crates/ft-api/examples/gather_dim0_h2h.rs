//! gather along dim=0 (outer_size==1 -> currently serial single block). Probe vs torch.
//! out[i,j] = src[idx[i,j], j]. Output is a fresh 64MB buffer (page-fault-bound, like cat).
use ft_api::FrankenTorchSession;
use ft_core::ExecutionMode;
use std::io::Write;
use std::process::{Command, Stdio};
use std::time::Instant;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let python = std::env::var("PYTORCH_PYTHON").unwrap_or_else(|_| "python3".to_string());
    let (r, c, i) = (4000usize, 4000usize, 4000usize);
    let src: Vec<f32> = (0..r * c)
        .map(|k| ((k % 9973) as f32 - 5000.0) * 0.01)
        .collect();
    // index[i,j] in [0,r): a deterministic spread of rows.
    let idx: Vec<f64> = (0..i * c)
        .map(|k| ((k * 2654435761usize) % r) as f64)
        .collect();
    let tt = |which: u8| {
        let mut best = f64::INFINITY;
        for _ in 0..7 {
            let mut s = FrankenTorchSession::new(ExecutionMode::Strict);
            let x = s
                .tensor_variable_f32(src.clone(), vec![r, c], false)
                .unwrap();
            let ix = s.tensor_variable(idx.clone(), vec![i, c], false).unwrap();
            let ti = Instant::now();
            match which {
                0 => {
                    let _ = s.tensor_add(x, x);
                }
                _ => {
                    let _ = s.tensor_gather(x, 0, ix);
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
r,c,i={r},{c},{i}
src=(((torch.arange(r*c,dtype=torch.int64)%9973).float()-5000.0)*0.01).reshape(r,c)
k=torch.arange(i*c,dtype=torch.int64)
idx=((k*2654435761)%r).reshape(i,c)
def tm(fn,reps=7):
    for _ in range(2): fn()
    ts=[]
    for _ in range(reps): s=time.perf_counter(); fn(); ts.append((time.perf_counter()-s)*1e3)
    return min(ts)
print("PT add %.3f"%tm(lambda:src+src))
print("PT gather %.3f"%tm(lambda:torch.gather(src,0,idx)))
"#,
        r = r,
        c = c,
        i = i
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
    println!(
        "f32 gather dim=0 [{r},{c}] idx[{i},{c}] -> {}M out (torch 8t / FT default), min-of-7",
        (i * c) / 1_000_000
    );
    for (lbl, w) in [("add", 0u8), ("gather", 1)] {
        let ft = tt(w);
        println!(
            "  {lbl:<7} FT {ft:8.3}  PT {:8.3}  => {}",
            g(lbl),
            v(ft, g(lbl))
        );
    }
    Ok(())
}

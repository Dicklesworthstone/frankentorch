//! Profiling sweep: histogram/cumulative/embedding/encoding ops. Find any remaining gap.
use ft_api::FrankenTorchSession;
use ft_core::ExecutionMode;
use std::io::Write;
use std::process::{Command, Stdio};
use std::time::Instant;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let python = std::env::var("PYTORCH_PYTHON").unwrap_or_else(|_| "python3".to_string());
    let n = 16_000_000usize;
    let idxbig: Vec<f64> = (0..n).map(|i| ((i * 7919) % 1000) as f64).collect();
    let cumdata: Vec<f32> = (0..n)
        .map(|i| ((i * 2654435761usize) % 1_000_003) as f32 * 0.01)
        .collect();
    let emb_idx: Vec<f64> = (0..1_000_000usize)
        .map(|i| ((i * 7919) % 100_000) as f64)
        .collect();
    let weight: Vec<f32> = (0..100_000 * 128)
        .map(|i| (i % 9973) as f32 * 0.01)
        .collect();
    let oh_idx: Vec<f64> = (0..4_000_000usize)
        .map(|i| ((i * 7919) % 256) as f64)
        .collect();
    let tt = |which: u8| {
        let mut best = f64::INFINITY;
        for _ in 0..7 {
            let mut s = FrankenTorchSession::new(ExecutionMode::Strict);
            let ti;
            match which {
                0 => {
                    let x = s.tensor_variable(idxbig.clone(), vec![n], false).unwrap();
                    ti = Instant::now();
                    let _ = s.tensor_add(x, x);
                }
                1 => {
                    let x = s.tensor_variable(idxbig.clone(), vec![n], false).unwrap();
                    ti = Instant::now();
                    let _ = s.tensor_bincount(x, None, 0);
                }
                2 => {
                    let x = s
                        .tensor_variable_f32(cumdata.clone(), vec![n], false)
                        .unwrap();
                    ti = Instant::now();
                    let _ = s.tensor_cummax(x);
                }
                3 => {
                    let w = s
                        .tensor_variable_f32(weight.clone(), vec![100_000, 128], false)
                        .unwrap();
                    let ix = s
                        .tensor_variable(emb_idx.clone(), vec![1_000_000], false)
                        .unwrap();
                    ti = Instant::now();
                    let _ = s.tensor_embedding(ix, w, None);
                }
                4 => {
                    let ix = s
                        .tensor_variable(oh_idx.clone(), vec![4_000_000], false)
                        .unwrap();
                    ti = Instant::now();
                    let _ = s.tensor_one_hot(ix, 256);
                }
                _ => {
                    let x = s.tensor_variable(idxbig.clone(), vec![n], false).unwrap();
                    ti = Instant::now();
                    let _ = s.tensor_count_nonzero(x);
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
n={n}
idxbig=((torch.arange(n,dtype=torch.int64)*7919)%1000)
cumdata=(((torch.arange(n,dtype=torch.int64)*2654435761)%1000003).float()*0.01)
emb_idx=((torch.arange(1000000,dtype=torch.int64)*7919)%100000)
weight=((torch.arange(100000*128,dtype=torch.int64)%9973).float()*0.01).reshape(100000,128)
oh_idx=((torch.arange(4000000,dtype=torch.int64)*7919)%256)
def tm(fn,reps=7):
    for _ in range(2): fn()
    ts=[]
    for _ in range(reps): s=time.perf_counter(); fn(); ts.append((time.perf_counter()-s)*1e3)
    return min(ts)
print("PT add %.3f"%tm(lambda:idxbig+idxbig))
print("PT bincount %.3f"%tm(lambda:torch.bincount(idxbig)))
print("PT cummax %.3f"%tm(lambda:torch.cummax(cumdata,0)))
print("PT embedding %.3f"%tm(lambda:torch.nn.functional.embedding(emb_idx,weight)))
print("PT onehot %.3f"%tm(lambda:torch.nn.functional.one_hot(oh_idx,256)))
print("PT countnz %.3f"%tm(lambda:torch.count_nonzero(idxbig)))
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
    println!("hist/embed sweep (torch 8t / FT default), min-of-7");
    for (lbl, w) in [
        ("add", 0u8),
        ("bincount", 1),
        ("cummax", 2),
        ("embedding", 3),
        ("onehot", 4),
        ("countnz", 5),
    ] {
        let ft = tt(w);
        println!(
            "  {lbl:<10} FT {ft:8.3}  PT {:8.3}  => {}",
            g(lbl),
            v(ft, g(lbl))
        );
    }
    Ok(())
}

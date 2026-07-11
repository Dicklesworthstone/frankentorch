//! 1-D selection-surface FT vs torch: msort / topk / kthvalue on a large 1-D
//! array. msort routes through tensor_sort (now parallel radix). Probes whether
//! topk/kthvalue single-large-lane paths are slower than torch.
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
        .unwrap_or(16_000_000);
    let data: Vec<f64> = (0..n)
        .map(|i| {
            let z = (i as u64)
                .wrapping_mul(2862933555777941757)
                .wrapping_add(3037000493);
            ((z >> 11) as f64 / (1u64 << 53) as f64) * 2.0 - 1.0
        })
        .collect();
    let kbig = n / 4;
    // 0=sort 1=msort 2=topk100 3=topkN/4 4=kthvalue 5=unique_inv 6=unique_counts 7=unique_both
    let tt = |which: u8| {
        let mut best = f64::INFINITY;
        for _ in 0..5 {
            let mut s = FrankenTorchSession::new(ExecutionMode::Strict);
            let x = s.tensor_variable(data.clone(), vec![n], false).unwrap();
            let ti = Instant::now();
            match which {
                0 => {
                    let _ = s.tensor_sort(x, 0, false);
                }
                1 => {
                    let _ = s.tensor_msort(x);
                }
                2 => {
                    let _ = s.tensor_topk(x, 100, 0, true, true);
                }
                3 => {
                    let _ = s.tensor_topk(x, kbig, 0, true, true);
                }
                4 => {
                    let _ = s.tensor_kthvalue(x, n / 2);
                }
                5 => {
                    let _ = s.tensor_unique(x, true, true, false);
                }
                6 => {
                    let _ = s.tensor_unique(x, true, false, true);
                }
                _ => {
                    let _ = s.tensor_unique(x, true, true, true);
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
n={n}; kbig={kbig}
g=torch.Generator().manual_seed(0)
x=torch.rand(n,generator=g,dtype=torch.float64)*2-1
def tm(fn,reps=5):
    for _ in range(2): fn()
    ts=[]
    for _ in range(reps): s=time.perf_counter(); fn(); ts.append((time.perf_counter()-s)*1e3)
    return min(ts)
print("PT sort %.3f"%tm(lambda:torch.sort(x)[0]))
print("PT msort %.3f"%tm(lambda:torch.msort(x)))
print("PT topk100 %.3f"%tm(lambda:torch.topk(x,100)))
print("PT topkbig %.3f"%tm(lambda:torch.topk(x,kbig)))
print("PT kthvalue %.3f"%tm(lambda:torch.kthvalue(x,n//2)))
print("PT unique_inv %.3f"%tm(lambda:torch.unique(x,return_inverse=True)))
print("PT unique_counts %.3f"%tm(lambda:torch.unique(x,return_counts=True)))
print("PT unique_both %.3f"%tm(lambda:torch.unique(x,return_inverse=True,return_counts=True)))
"#,
        n = n,
        kbig = kbig
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
    println!("1-D selection surface, N={n} f64 (torch 8t / FT default), min-of-5");
    for (lbl, key, w) in [
        ("sort", "sort", 0u8),
        ("msort", "msort", 1),
        ("topk100", "topk100", 2),
        ("topkN/4", "topkbig", 3),
        ("kthvalue", "kthvalue", 4),
        ("unique_inv", "unique_inv", 5),
        ("unique_cnt", "unique_counts", 6),
        ("unique_both", "unique_both", 7),
    ] {
        let ft = tt(w);
        println!(
            "  {lbl:<9} FT {ft:9.3}  PT {:9.3}  => {}",
            g(key),
            v(ft, g(key))
        );
    }
    Ok(())
}

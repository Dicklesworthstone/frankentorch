//! Profiling sweep: normalization-forward + selection ops. Find the biggest gap vs torch.
use ft_api::FrankenTorchSession;
use ft_core::ExecutionMode;
use std::io::Write;
use std::process::{Command, Stdio};
use std::time::Instant;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let python = std::env::var("PYTORCH_PYTHON").unwrap_or_else(|_| "python3".to_string());
    let big: Vec<f32> = (0..8192 * 2048)
        .map(|i| ((i % 9973) as f32 - 5000.0) * 0.01)
        .collect();
    let mat: Vec<f32> = (0..4096 * 4096)
        .map(|i| ((i * 7919) % 1_000_003) as f32 * 0.001)
        .collect();
    let gn: Vec<f32> = (0..64 * 256 * 16 * 16)
        .map(|i| ((i % 9973) as f32 - 5000.0) * 0.01)
        .collect();
    let tt = |which: u8| {
        let mut best = f64::INFINITY;
        for _ in 0..7 {
            let mut s = FrankenTorchSession::new(ExecutionMode::Strict);
            let ti;
            match which {
                0 => {
                    let x = s
                        .tensor_variable_f32(big.clone(), vec![8192, 2048], false)
                        .unwrap();
                    ti = Instant::now();
                    let _ = s.tensor_add(x, x);
                }
                1 => {
                    let x = s
                        .tensor_variable_f32(big.clone(), vec![8192, 2048], false)
                        .unwrap();
                    ti = Instant::now();
                    let _ = s.tensor_layer_norm(x, vec![2048], None, None, 1e-5);
                }
                2 => {
                    let x = s
                        .tensor_variable_f32(gn.clone(), vec![64, 256, 16, 16], false)
                        .unwrap();
                    ti = Instant::now();
                    let _ = s.tensor_group_norm(x, 32, None, None, 1e-5);
                }
                3 => {
                    let x = s
                        .tensor_variable_f32(mat.clone(), vec![4096, 4096], false)
                        .unwrap();
                    ti = Instant::now();
                    let _ = s.tensor_topk(x, 100, 1, true, true);
                }
                4 => {
                    let x = s
                        .tensor_variable_f32(mat.clone(), vec![16_777_216], false)
                        .unwrap();
                    ti = Instant::now();
                    let _ = s.tensor_median(x);
                }
                _ => {
                    let x = s
                        .tensor_variable_f32(mat.clone(), vec![16_777_216], false)
                        .unwrap();
                    ti = Instant::now();
                    let _ = s.tensor_sort(x, 0, false);
                }
            }
            let e = ti.elapsed().as_secs_f64() * 1e3;
            if e < best {
                best = e;
            }
        }
        best
    };
    let py = r#"
import time, torch
torch.set_num_threads(8)
big=(((torch.arange(8192*2048,dtype=torch.int64)%9973).float()-5000.0)*0.01).reshape(8192,2048)
mat=(((torch.arange(4096*4096,dtype=torch.int64)*7919)%1000003).float()*0.001).reshape(4096,4096)
matf=mat.flatten()
gn=(((torch.arange(64*256*16*16,dtype=torch.int64)%9973).float()-5000.0)*0.01).reshape(64,256,16,16)
def tm(fn,reps=7):
    for _ in range(2): fn()
    ts=[]
    for _ in range(reps): s=time.perf_counter(); fn(); ts.append((time.perf_counter()-s)*1e3)
    return min(ts)
print("PT add %.3f"%tm(lambda:big+big))
print("PT layernorm %.3f"%tm(lambda:torch.nn.functional.layer_norm(big,[2048])))
print("PT groupnorm %.3f"%tm(lambda:torch.nn.functional.group_norm(gn,32)))
print("PT topk %.3f"%tm(lambda:torch.topk(mat,100,1)))
print("PT median %.3f"%tm(lambda:torch.median(matf)))
print("PT sort %.3f"%tm(lambda:torch.sort(matf)))
"#;
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
    println!("norm/select sweep (torch 8t / FT default), min-of-7");
    for (lbl, w) in [
        ("add", 0u8),
        ("layernorm", 1),
        ("groupnorm", 2),
        ("topk", 3),
        ("median", 4),
        ("sort", 5),
    ] {
        let ft = tt(w);
        println!(
            "  {lbl:<10} FT {ft:9.3}  PT {:9.3}  => {}",
            g(lbl),
            v(ft, g(lbl))
        );
    }
    Ok(())
}

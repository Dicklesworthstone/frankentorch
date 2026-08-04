// index_reduce probe: f32 crash check + f64 perf vs torch (no-grad). cc.
use ft_api::FrankenTorchSession;
use ft_core::ExecutionMode;
use std::io::Write;
use std::process::{Command, Stdio};
use std::time::Instant;
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let python = std::env::var("PYTORCH_PYTHON").unwrap_or_else(|_| "python3".to_string());
    let (rows, cols, nsrc) = (4096usize, 1024usize, 16384usize);
    // f32 crash check
    {
        let mut s = FrankenTorchSession::new(ExecutionMode::Strict);
        let inp = s.tensor_variable_f32(vec![1.0f32; rows * cols], vec![rows, cols], false)?;
        let idx = s.tensor_variable(
            (0..nsrc).map(|i| (i % rows) as f64).collect(),
            vec![nsrc],
            false,
        )?;
        let src = s.tensor_variable_f32(vec![2.0f32; nsrc * cols], vec![nsrc, cols], false)?;
        match s.tensor_index_reduce(inp, 0, idx, src, "amax", true) {
            Ok(o) => println!("f32 index_reduce: OK dtype={:?}", s.tensor_dtype(o)?),
            Err(e) => println!("f32 index_reduce: ERROR {e:?}"),
        }
    }
    // f64 perf
    let input: Vec<f64> = (0..rows * cols)
        .map(|i| ((i % 9973) as f64 - 5000.0) * 0.001)
        .collect();
    let idxv: Vec<f64> = (0..nsrc).map(|i| ((i * 7919) % rows) as f64).collect();
    let srcv: Vec<f64> = (0..nsrc * cols)
        .map(|i| ((i % 7919) as f64) * 0.001 + 0.5)
        .collect();
    for mode in ["amax", "prod", "mean"] {
        let bench = || {
            let mut best = f64::INFINITY;
            for _ in 0..5 {
                let mut s = FrankenTorchSession::new(ExecutionMode::Strict);
                let xi = s
                    .tensor_variable(input.clone(), vec![rows, cols], false)
                    .unwrap();
                let ix = s.tensor_variable(idxv.clone(), vec![nsrc], false).unwrap();
                let sr = s
                    .tensor_variable(srcv.clone(), vec![nsrc, cols], false)
                    .unwrap();
                let t = Instant::now();
                let _ = s.tensor_index_reduce(xi, 0, ix, sr, mode, true);
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
rows,cols,nsrc={rows},{cols},{nsrc}
input=(((torch.arange(rows*cols,dtype=torch.int64)%9973).double()-5000.0)*0.001).reshape(rows,cols)
idx=((torch.arange(nsrc,dtype=torch.int64)*7919)%rows)
src=(((torch.arange(nsrc*cols,dtype=torch.int64)%7919).double())*0.001+0.5).reshape(nsrc,cols)
def tm(fn,reps=5):
    for _ in range(2): fn()
    ts=[]
    for _ in range(reps): s=time.perf_counter(); fn(); ts.append((time.perf_counter()-s)*1e3)
    return min(ts)
print("PT ir %.3f"%tm(lambda:input.index_reduce(0,idx,src,'{mode}',include_self=True)))
"#
        );
        let mut ch = Command::new(&python)
            .arg("-")
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .spawn()?;
        ch.stdin.as_mut().unwrap().write_all(py.as_bytes())?;
        let out = String::from_utf8_lossy(&ch.wait_with_output()?.stdout).to_string();
        let pt = out
            .lines()
            .find_map(|l| {
                let mut it = l.strip_prefix("PT ")?.split_whitespace();
                if it.next()? == "ir" {
                    it.next()?.parse::<f64>().ok()
                } else {
                    None
                }
            })
            .unwrap_or(f64::NAN);
        let ft = bench();
        let vrb = if pt >= ft {
            format!("FT {:.2}x FASTER", pt / ft)
        } else {
            format!("FT {:.2}x SLOWER", ft / pt)
        };
        println!(
            "index_reduce f64 dim=0 [{rows}x{cols}] nsrc={nsrc} mode={mode}: FT {ft:8.3}ms torch {pt:8.3}ms => {vrb}"
        );
    }
    Ok(())
}

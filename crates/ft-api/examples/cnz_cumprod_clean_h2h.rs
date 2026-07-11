// Clean re-measure of count_nonzero + cumprod f32 (inputs OUTSIDE timed region). cc.
use ft_api::FrankenTorchSession;
use ft_core::ExecutionMode;
use std::io::Write;
use std::process::{Command, Stdio};
use std::time::Instant;
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let python = std::env::var("PYTORCH_PYTHON").unwrap_or_else(|_| "python3".to_string());
    // count_nonzero [4M]
    let cnz_in: Vec<f32> = (0..4_000_000)
        .map(|i| ((i % 9973) as f32 - 5000.0) * 0.001)
        .collect();
    let t_cnz = {
        let mut best = f64::INFINITY;
        for _ in 0..7 {
            let mut s = FrankenTorchSession::new(ExecutionMode::Strict);
            let x = s
                .tensor_variable_f32(cnz_in.clone(), vec![4_000_000], false)
                .unwrap();
            let t = Instant::now();
            let _ = s.tensor_count_nonzero(x);
            let e = t.elapsed().as_secs_f64() * 1e3;
            if e < best {
                best = e;
            }
        }
        best
    };
    // cumprod [2048,2048] dim=1
    let cp_in: Vec<f32> = (0..2048 * 2048)
        .map(|i| 0.999 + ((i % 7) as f32) * 0.0003)
        .collect();
    let t_cp = {
        let mut best = f64::INFINITY;
        for _ in 0..7 {
            let mut s = FrankenTorchSession::new(ExecutionMode::Strict);
            let x = s
                .tensor_variable_f32(cp_in.clone(), vec![2048, 2048], false)
                .unwrap();
            let t = Instant::now();
            let _ = s.tensor_cumprod(x, 1);
            let e = t.elapsed().as_secs_f64() * 1e3;
            if e < best {
                best = e;
            }
        }
        best
    };
    let py = r#"
import time,torch
torch.set_num_threads(8)
def tm(fn,reps=7):
    for _ in range(2): fn()
    ts=[]
    for _ in range(reps): s=time.perf_counter(); fn(); ts.append((time.perf_counter()-s)*1e3)
    return min(ts)
xn=(((torch.arange(4000000,dtype=torch.int64)%9973).float()-5000.0)*0.001)
print("PT cnz %.4f"%tm(lambda:torch.count_nonzero(xn)))
xc=(0.999+(torch.arange(2048*2048,dtype=torch.int64)%7).float()*0.0003).reshape(2048,2048)
print("PT cumprod %.4f"%tm(lambda:torch.cumprod(xc,1)))
"#;
    let mut ch = Command::new(&python)
        .arg("-")
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .spawn()?;
    ch.stdin.as_mut().unwrap().write_all(py.as_bytes())?;
    let out = String::from_utf8_lossy(&ch.wait_with_output()?.stdout).to_string();
    let pt = |name: &str| -> f64 {
        out.lines()
            .find_map(|l| {
                let mut it = l.strip_prefix("PT ")?.split_whitespace();
                if it.next()? == name {
                    it.next()?.parse::<f64>().ok()
                } else {
                    None
                }
            })
            .unwrap_or(f64::NAN)
    };
    for (name, ft) in [("cnz", t_cnz), ("cumprod", t_cp)] {
        let p = pt(name);
        let vrb = if p >= ft {
            format!("FT {:.2}x FASTER", p / ft)
        } else {
            format!("FT {:.2}x SLOWER", ft / p)
        };
        println!("{name:<10} FT {ft:8.4}ms torch {p:8.4}ms => {vrb}");
    }
    Ok(())
}

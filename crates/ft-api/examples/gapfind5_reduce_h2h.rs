// RELEASE gapfind for full-tensor reductions f32 vs torch (inputs OUTSIDE timer). cc.
use ft_api::FrankenTorchSession;
use ft_core::ExecutionMode;
use std::io::Write;
use std::process::{Command, Stdio};
use std::time::Instant;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let python = std::env::var("PYTORCH_PYTHON").unwrap_or_else(|_| "python3".to_string());
    let big = 16 * 1024 * 1024;
    let med = 4 * 1024 * 1024;
    let vbig: Vec<f32> = (0..big)
        .map(|i| ((i % 9973) as f32 - 5000.0) * 0.001)
        .collect();
    // prod: values near 1.0 to keep the product finite
    let vprod: Vec<f32> = (0..med)
        .map(|i| 1.0 + ((i % 7) as f32 - 3.0) * 1e-7)
        .collect();
    let vmed: Vec<f32> = (0..med)
        .map(|i| ((i % 9973) as f32 - 5000.0) * 0.001)
        .collect();
    macro_rules! bench {
        ($v:expr, $shape:expr, $op:expr) => {{
            let mut best = f64::INFINITY;
            for _ in 0..7 {
                let mut s = FrankenTorchSession::new(ExecutionMode::Strict);
                let x = s.tensor_variable_f32($v.clone(), $shape, false).unwrap();
                let t = Instant::now();
                let _ = $op(&mut s, x);
                best = best.min(t.elapsed().as_secs_f64() * 1e3);
            }
            best
        }};
    }
    let t_sum = bench!(vbig, vec![big], |s: &mut FrankenTorchSession, x| s
        .tensor_sum(x));
    let t_mean = bench!(vbig, vec![big], |s: &mut FrankenTorchSession, x| s
        .tensor_mean(x));
    let t_prod = bench!(vprod, vec![med], |s: &mut FrankenTorchSession, x| s
        .tensor_prod(x));
    let t_var = bench!(vbig, vec![big], |s: &mut FrankenTorchSession, x| s
        .tensor_var(x, 1));
    let t_std = bench!(vbig, vec![big], |s: &mut FrankenTorchSession, x| s
        .tensor_std(x, 1));
    let t_median = bench!(vmed, vec![med], |s: &mut FrankenTorchSession, x| s
        .tensor_median(x));

    let py = r#"
import time,torch
torch.set_num_threads(8)
def tm(fn,reps=7):
    for _ in range(2): fn()
    ts=[]
    for _ in range(reps): s=time.perf_counter(); fn(); ts.append((time.perf_counter()-s)*1e3)
    return min(ts)
big=16*1024*1024; med=4*1024*1024
vb=(((torch.arange(big,dtype=torch.int64)%9973).float()-5000.0)*0.001)
vp=1.0+((torch.arange(med,dtype=torch.int64)%7).float()-3.0)*1e-7
vm=(((torch.arange(med,dtype=torch.int64)%9973).float()-5000.0)*0.001)
print("PT sum %.4f"%tm(lambda:torch.sum(vb)))
print("PT mean %.4f"%tm(lambda:torch.mean(vb)))
print("PT prod %.4f"%tm(lambda:torch.prod(vp)))
print("PT var %.4f"%tm(lambda:torch.var(vb,correction=1)))
print("PT std %.4f"%tm(lambda:torch.std(vb,correction=1)))
print("PT median %.4f"%tm(lambda:torch.median(vm)))
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
    for (name, ft) in [
        ("sum", t_sum),
        ("mean", t_mean),
        ("prod", t_prod),
        ("var", t_var),
        ("std", t_std),
        ("median", t_median),
    ] {
        let p = pt(name);
        let vrb = if p >= ft {
            format!("FT {:.2}x FASTER", p / ft)
        } else {
            format!("FT {:.2}x SLOWER", ft / p)
        };
        println!("{name:<8} FT {ft:9.4}ms torch {p:9.4}ms => {vrb}");
    }
    Ok(())
}

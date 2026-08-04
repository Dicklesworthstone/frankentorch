// Full-tensor vector norm f32 vs torch (no-grad): f32 reduction is SERIAL (f64 is parallel). cc.
use ft_api::FrankenTorchSession;
use ft_core::ExecutionMode;
use std::io::Write;
use std::process::{Command, Stdio};
use std::time::Instant;
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let python = std::env::var("PYTORCH_PYTHON").unwrap_or_else(|_| "python3".to_string());
    let n = 16 * 1024 * 1024;
    let inp: Vec<f32> = (0..n)
        .map(|i| ((i % 9973) as f32 - 5000.0) * 0.001)
        .collect();
    let bench = |p: f64| {
        let mut best = f64::INFINITY;
        let mut val = 0.0f64;
        for _ in 0..7 {
            let mut s = FrankenTorchSession::new(ExecutionMode::Strict);
            let x = s.tensor_variable_f32(inp.clone(), vec![n], false).unwrap();
            let t = Instant::now();
            let o = s.tensor_norm(x, p).unwrap();
            let e = t.elapsed().as_secs_f64() * 1e3;
            if e < best {
                best = e;
                val = s.tensor_values_lossy_f64(o).unwrap()[0];
            }
        }
        (best, val)
    };
    let (l1, v1) = bench(1.0);
    let (l2, v2) = bench(2.0);
    let (l3, v3) = bench(3.0);
    let (linf, vinf) = bench(f64::INFINITY);
    let (l0, v0) = bench(0.0);
    let py = format!(
        r#"
import time,torch
torch.set_num_threads(8)
n={n}
x=(((torch.arange(n,dtype=torch.int64)%9973).float()-5000.0)*0.001)
def tm(fn,reps=7):
    for _ in range(2): fn()
    ts=[]
    for _ in range(reps): s=time.perf_counter(); fn(); ts.append((time.perf_counter()-s)*1e3)
    return min(ts)
for p in (1.0,2.0,3.0):
    print("PT norm%d %.4f %.6f"%(int(p),tm(lambda:torch.linalg.vector_norm(x,p)),float(torch.linalg.vector_norm(x,p))))
import math
print("PT norminf %.4f %.6f"%(tm(lambda:torch.linalg.vector_norm(x,math.inf)),float(torch.linalg.vector_norm(x,math.inf))))
print("PT norm0 %.4f %.6f"%(tm(lambda:torch.linalg.vector_norm(x,0.0)),float(torch.linalg.vector_norm(x,0.0))))
"#
    );
    let mut ch = Command::new(&python)
        .arg("-")
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .spawn()?;
    ch.stdin.as_mut().unwrap().write_all(py.as_bytes())?;
    let out = String::from_utf8_lossy(&ch.wait_with_output()?.stdout).to_string();
    let get = |tag: &str| -> (f64, f64) {
        out.lines()
            .find_map(|l| {
                let mut it = l.strip_prefix("PT ")?.split_whitespace();
                if it.next()? == tag {
                    Some((it.next()?.parse().ok()?, it.next()?.parse().ok()?))
                } else {
                    None
                }
            })
            .unwrap_or((f64::NAN, f64::NAN))
    };
    for (tag, ft, fv) in [
        ("norm1", l1, v1),
        ("norm2", l2, v2),
        ("norm3", l3, v3),
        ("norminf", linf, vinf),
        ("norm0", l0, v0),
    ] {
        let (pt, pv) = get(tag);
        let vrb = if pt >= ft {
            format!("FT {:.2}x FASTER", pt / ft)
        } else {
            format!("FT {:.2}x SLOWER", ft / pt)
        };
        let rel = ((fv - pv) / pv.abs().max(1e-30)).abs();
        println!(
            "{tag:<7} FT {ft:8.4}ms torch {pt:8.4}ms => {vrb} | ftval={fv:.5} ptval={pv:.5} rel={rel:.2e}"
        );
    }
    Ok(())
}

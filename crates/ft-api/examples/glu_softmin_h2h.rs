//! glu/softmin F64 no-grad vs torch, transformer-ish shape. Measures whether there's a real gap before
//! investing in a fused fast path (xlog1py lesson: check the compose time first). Inputs OUTSIDE timer.
use ft_api::FrankenTorchSession;
use ft_core::ExecutionMode;
use std::io::Write;
use std::process::{Command, Stdio};
use std::time::Instant;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let python = std::env::var("PYTORCH_PYTHON").unwrap_or_else(|_| "python3".to_string());
    let (b, s, d) = (64usize, 512usize, 512usize); // glu input last dim = 2d
    let n_glu = b * s * 2 * d;
    let n_sm = b * s * d;
    let glu_x: Vec<f64> = (0..n_glu)
        .map(|i| ((i % 991) as f64) * 0.01 - 5.0)
        .collect();
    let sm_x: Vec<f64> = (0..n_sm).map(|i| ((i % 997) as f64) * 0.01 - 5.0).collect();
    let run = |which: &str| -> f64 {
        let mut best = f64::INFINITY;
        for _ in 0..7 {
            let mut sess = FrankenTorchSession::new(ExecutionMode::Strict);
            let (v, shp) = if which == "glu" {
                (&glu_x, vec![b, s, 2 * d])
            } else {
                (&sm_x, vec![b, s, d])
            };
            let x = sess.tensor_variable(v.clone(), shp, false).unwrap();
            let t = Instant::now();
            let _ = if which == "glu" {
                sess.tensor_glu(x, 2).unwrap()
            } else {
                sess.tensor_softmin(x, 2).unwrap()
            };
            let e = t.elapsed().as_secs_f64() * 1e3;
            if e < best {
                best = e;
            }
        }
        best
    };
    let (fg, fs) = (run("glu"), run("softmin"));
    let py = format!(
        r#"
import time,torch,torch.nn.functional as F
torch.set_num_threads(8)
b,s,d={b},{s},{d}
xg=((torch.arange(b*s*2*d,dtype=torch.int64)%991).double())*0.01-5.0; xg=xg.reshape(b,s,2*d)
xs=((torch.arange(b*s*d,dtype=torch.int64)%997).double())*0.01-5.0; xs=xs.reshape(b,s,d)
def t(fn,reps=7):
    for _ in range(2): fn()
    ts=[]
    for _ in range(reps): st=time.perf_counter(); fn(); ts.append((time.perf_counter()-st)*1e3)
    return min(ts)
print("PT glu %.4f"%t(lambda:F.glu(xg,dim=2)))
print("PT softmin %.4f"%t(lambda:F.softmin(xs,dim=2)))
"#
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
        "  glu     FT {fg:.3}ms  PT {:.3}ms => {}",
        g("glu"),
        v(fg, g("glu"))
    );
    println!(
        "  softmin FT {fs:.3}ms  PT {:.3}ms => {}",
        g("softmin"),
        v(fs, g("softmin"))
    );
    Ok(())
}

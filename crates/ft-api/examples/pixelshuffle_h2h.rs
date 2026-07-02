//! pixel_shuffle / pixel_unshuffle F64 no-grad vs torch, vision shape. Measures whether the
//! reshape+permute+reshape (strided-view materialize) has a real gap before fusing. Inputs OUTSIDE timer.
use ft_api::FrankenTorchSession;
use ft_core::ExecutionMode;
use std::io::Write;
use std::process::{Command, Stdio};
use std::time::Instant;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let python = std::env::var("PYTORCH_PYTHON").unwrap_or_else(|_| "python3".to_string());
    let (n, c, h, w, r) = (16usize, 256usize, 64usize, 64usize, 2usize); // shuffle: [n,c,h,w]->[n,c/4,2h,2w]
    let ns = n * c * h * w;
    let xd: Vec<f64> = (0..ns).map(|i| ((i % 4093) as f64) * 0.01 - 20.0).collect();
    // unshuffle input: [n, c/4, 2h, 2w] (same numel), rearranged back
    let run = |which: &str| -> f64 {
        let mut best = f64::INFINITY;
        for _ in 0..7 {
            let mut s = FrankenTorchSession::new(ExecutionMode::Strict);
            let (shp, ) = if which == "shuffle" { (vec![n, c, h, w],) } else { (vec![n, c / (r * r), h * r, w * r],) };
            let x = s.tensor_variable(xd.clone(), shp, false).unwrap();
            let t = Instant::now();
            let _ = if which == "shuffle" { s.tensor_pixel_shuffle(x, r).unwrap() } else { s.tensor_pixel_unshuffle(x, r).unwrap() };
            let e = t.elapsed().as_secs_f64() * 1e3;
            if e < best { best = e; }
        }
        best
    };
    let (fsh, fun) = (run("shuffle"), run("unshuffle"));
    let py = format!(
        r#"
import time,torch,torch.nn.functional as F
torch.set_num_threads(8)
n,c,h,w,r={n},{c},{h},{w},{r}
xs=((torch.arange(n*c*h*w,dtype=torch.int64)%4093).double())*0.01-20.0; xs=xs.reshape(n,c,h,w)
xu=xs.reshape(n,c//(r*r),h*r,w*r)
def t(fn,reps=7):
    for _ in range(2): fn()
    ts=[]
    for _ in range(reps): st=time.perf_counter(); fn(); ts.append((time.perf_counter()-st)*1e3)
    return min(ts)
print("PT shuffle %.4f"%t(lambda:F.pixel_shuffle(xs,r)))
print("PT unshuffle %.4f"%t(lambda:F.pixel_unshuffle(xu,r)))
"#
    );
    let mut ch = Command::new(&python).arg("-").stdin(Stdio::piped()).stdout(Stdio::piped()).spawn()?;
    ch.stdin.as_mut().unwrap().write_all(py.as_bytes())?;
    let pt = String::from_utf8_lossy(&ch.wait_with_output()?.stdout).to_string();
    let g = |k: &str| pt.lines().find_map(|l| { let mut it = l.strip_prefix("PT ")?.split_whitespace(); if it.next()? == k { it.next()?.parse::<f64>().ok() } else { None } }).unwrap_or(f64::NAN);
    let v = |ft: f64, p: f64| if p >= ft { format!("FT {:.2}x FASTER", p / ft) } else { format!("FT {:.2}x SLOWER", ft / p) };
    println!("  pixel_shuffle   FT {fsh:.3}ms  PT {:.3}ms => {}", g("shuffle"), v(fsh, g("shuffle")));
    println!("  pixel_unshuffle FT {fun:.3}ms  PT {:.3}ms => {}", g("unshuffle"), v(fun, g("unshuffle")));
    Ok(())
}

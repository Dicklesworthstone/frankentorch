use ft_api::FrankenTorchSession;
use ft_core::ExecutionMode;
use std::io::Write;
use std::process::{Command, Stdio};
use std::time::Instant;
const N: usize = 4_000_000;
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let x: Vec<f32> = (0..N).map(|i| (i % 5000) as f32).collect(); // ~5000 uniques, high dup
    let mut b = f64::INFINITY;
    for _ in 0..7 {
        let mut s = FrankenTorchSession::new(ExecutionMode::Strict);
        let xn = s.tensor_variable_f32(x.clone(), vec![N], false).unwrap();
        let t = Instant::now();
        let _ = s.tensor_unique(xn, true, false, false);
        let e = t.elapsed().as_secs_f64() * 1e3;
        if e < b {
            b = e;
        }
    }
    let py = r#"
import time,torch
torch.set_num_threads(8)
x=(torch.arange(4000000,dtype=torch.int64)%5000).float()
def t(fn,n=7):
    for _ in range(2): fn()
    return min((lambda s: (fn(), (time.perf_counter()-s)*1e3)[1])(time.perf_counter()) for _ in range(n))
print("PT %.4f"%t(lambda:torch.unique(x)))
"#;
    let python = std::env::var("PYTORCH_PYTHON").unwrap_or_else(|_| "python3".into());
    let mut ch = Command::new(&python)
        .arg("-")
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .spawn()?;
    ch.stdin.as_mut().unwrap().write_all(py.as_bytes())?;
    let o = ch.wait_with_output()?;
    let pt = String::from_utf8_lossy(&o.stdout);
    let p: f64 = pt.lines().find(|l| l.starts_with("PT ")).unwrap()[3..]
        .trim()
        .parse()
        .unwrap();
    let r = p / b;
    let tag = if r >= 1.0 {
        format!("FT {r:.2}x FASTER")
    } else {
        format!("FT {:.2}x SLOWER", 1.0 / r)
    };
    println!("unique f32 [4M]: FT {b:.3}ms  PT {p:.3}ms  {tag}");
    Ok(())
}

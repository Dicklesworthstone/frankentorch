//! f64 hardswish/hardsigmoid/hardtanh SIMD vs torch. relu_f64 = SIMD anchor.
use ft_api::FrankenTorchSession;
use ft_core::ExecutionMode;
use std::io::Write;
use std::process::{Command, Stdio};
use std::time::Instant;

fn t1<F: Fn(&mut FrankenTorchSession, ft_autograd::TensorNodeId)>(a: &[f64], f: F) -> f64 {
    let mut best = f64::INFINITY;
    for _ in 0..9 {
        let mut s = FrankenTorchSession::new(ExecutionMode::Strict);
        let x = s.tensor_variable(a.to_vec(), vec![a.len()], false).unwrap();
        let t = Instant::now();
        f(&mut s, x);
        let e = t.elapsed().as_secs_f64() * 1e3;
        if e < best { best = e; }
    }
    best
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let n = 16_000_000usize;
    let a: Vec<f64> = (0..n).map(|i| ((i % 9973) as f64 - 5000.0) * 0.0017).collect();
    let python = std::env::var("PYTORCH_PYTHON").unwrap_or_else(|_| "python3".to_string());
    let py = r#"
import time,torch
torch.set_num_threads(8)
n=16_000_000
a=(((torch.arange(n,dtype=torch.int64)%9973).double()-5000.0)*0.0017)
def t(fn,reps=9):
    for _ in range(2): fn()
    ts=[]
    for _ in range(reps): s=time.perf_counter(); fn(); ts.append((time.perf_counter()-s)*1e3)
    return min(ts)
F=torch.nn.functional
print("PT relu_anchor %.4f"%t(lambda:torch.relu(a)))
print("PT hardswish %.4f"%t(lambda:F.hardswish(a)))
print("PT hardsigmoid %.4f"%t(lambda:F.hardsigmoid(a)))
print("PT hardtanh %.4f"%t(lambda:F.hardtanh(a)))
"#;
    let mut ch = Command::new(&python).arg("-").stdin(Stdio::piped()).stdout(Stdio::piped()).spawn()?;
    ch.stdin.as_mut().unwrap().write_all(py.as_bytes())?;
    let pt = String::from_utf8_lossy(&ch.wait_with_output()?.stdout).to_string();
    let rep = |name: &str, ft: f64| {
        if let Some(p) = pt.lines().find_map(|l| {
            let mut it = l.strip_prefix("PT ")?.split_whitespace();
            if it.next()? == name { it.next()?.parse::<f64>().ok() } else { None }
        }) {
            let r = p / ft;
            let v = if r >= 1.0 { format!("FT {r:.2}x FASTER") } else { format!("FT {:.2}x SLOWER", 1.0 / r) };
            println!("  {name:<12} {ft:8.3} {p:8.3}   {v}");
        }
    };
    println!("op            FT(ms)    PT(ms)   verdict");
    rep("relu_anchor", t1(&a, |s, x| { let _ = s.tensor_relu(x); }));
    rep("hardswish", t1(&a, |s, x| { let _ = s.tensor_hardswish(x); }));
    rep("hardsigmoid", t1(&a, |s, x| { let _ = s.tensor_hardsigmoid(x); }));
    rep("hardtanh", t1(&a, |s, x| { let _ = s.tensor_hardtanh(x); }));
    Ok(())
}

//! Broadcast binary survey: [N,N]+[N,1] (col) and [N,N]+[1,N] (row), f64 & f32, vs torch.
use ft_api::FrankenTorchSession;
use ft_autograd::TensorNodeId;
use ft_core::ExecutionMode;
use std::io::Write;
use std::process::{Command, Stdio};
use std::time::Instant;
const N: usize = 4000;

fn run64<F: Fn(&mut FrankenTorchSession, TensorNodeId, TensorNodeId)>(
    a: &[f64],
    bshape: Vec<usize>,
    b: &[f64],
    f: F,
) -> f64 {
    let mut best = f64::INFINITY;
    for _ in 0..7 {
        let mut s = FrankenTorchSession::new(ExecutionMode::Strict);
        let x = s.tensor_variable(a.to_vec(), vec![N, N], false).unwrap();
        let y = s
            .tensor_variable(b.to_vec(), bshape.clone(), false)
            .unwrap();
        let t = Instant::now();
        f(&mut s, x, y);
        let e = t.elapsed().as_secs_f64() * 1e3;
        if e < best {
            best = e;
        }
    }
    best
}
fn run32<F: Fn(&mut FrankenTorchSession, TensorNodeId, TensorNodeId)>(
    a: &[f32],
    bshape: Vec<usize>,
    b: &[f32],
    f: F,
) -> f64 {
    let mut best = f64::INFINITY;
    for _ in 0..7 {
        let mut s = FrankenTorchSession::new(ExecutionMode::Strict);
        let x = s
            .tensor_variable_f32(a.to_vec(), vec![N, N], false)
            .unwrap();
        let y = s
            .tensor_variable_f32(b.to_vec(), bshape.clone(), false)
            .unwrap();
        let t = Instant::now();
        f(&mut s, x, y);
        let e = t.elapsed().as_secs_f64() * 1e3;
        if e < best {
            best = e;
        }
    }
    best
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let a64: Vec<f64> = (0..N * N).map(|i| (i % 100) as f64 * 0.01).collect();
    let col64: Vec<f64> = (0..N).map(|i| i as f64 * 0.1).collect(); // [N,1]
    let row64: Vec<f64> = (0..N).map(|i| i as f64 * 0.1).collect(); // [1,N]
    let a32: Vec<f32> = a64.iter().map(|&v| v as f32).collect();
    let col32: Vec<f32> = col64.iter().map(|&v| v as f32).collect();
    let row32: Vec<f32> = row64.iter().map(|&v| v as f32).collect();
    let python = std::env::var("PYTORCH_PYTHON").unwrap_or_else(|_| "python3".to_string());
    let py = r#"
import time,torch
torch.set_num_threads(8)
N=4000
def mk(dt):
    a=((torch.arange(N*N,dtype=torch.int64)%100).to(dt)*0.01).reshape(N,N)
    col=(torch.arange(N,dtype=torch.int64).to(dt)*0.1).reshape(N,1)
    row=(torch.arange(N,dtype=torch.int64).to(dt)*0.1).reshape(1,N)
    return a,col,row
def t(fn,n=7):
    for _ in range(2): fn()
    ts=[]
    for _ in range(n): s=time.perf_counter(); fn(); ts.append((time.perf_counter()-s)*1e3)
    return min(ts)
for tag,dt in [("f64",torch.float64),("f32",torch.float32)]:
    a,col,row=mk(dt)
    print("PT %s_anchor %.4f"%(tag,t(lambda:a+a)))
    print("PT %s_col %.4f"%(tag,t(lambda:a+col)))
    print("PT %s_row %.4f"%(tag,t(lambda:a+row)))
"#;
    let mut ch = Command::new(&python)
        .arg("-")
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .spawn()?;
    ch.stdin.as_mut().unwrap().write_all(py.as_bytes())?;
    let o = ch.wait_with_output();
    let pt = String::from_utf8_lossy(&o.unwrap().stdout).to_string();
    let rep = |n: &str, ft: f64| {
        if let Some(p) = pt.lines().find_map(|l| {
            let mut it = l.strip_prefix("PT ")?.split_whitespace();
            if it.next()? == n {
                it.next()?.parse::<f64>().ok()
            } else {
                None
            }
        }) {
            let r = p / ft;
            let tag = if r >= 1.0 {
                format!("FT {r:.2}x FASTER")
            } else {
                format!("FT {:.2}x SLOWER", 1.0 / r)
            };
            println!("  {n:<12} {ft:8.3} {p:8.3}   {tag}");
        }
    };
    println!("op            FT(ms)    PT(ms)   verdict");
    rep(
        "f64_anchor",
        run64(&a64, vec![N, N], &a64, |s, x, y| {
            let _ = s.tensor_add(x, y);
        }),
    );
    rep(
        "f64_col",
        run64(&a64, vec![N, 1], &col64, |s, x, y| {
            let _ = s.tensor_add(x, y);
        }),
    );
    rep(
        "f64_row",
        run64(&a64, vec![1, N], &row64, |s, x, y| {
            let _ = s.tensor_add(x, y);
        }),
    );
    rep(
        "f32_anchor",
        run32(&a32, vec![N, N], &a32, |s, x, y| {
            let _ = s.tensor_add(x, y);
        }),
    );
    rep(
        "f32_col",
        run32(&a32, vec![N, 1], &col32, |s, x, y| {
            let _ = s.tensor_add(x, y);
        }),
    );
    rep(
        "f32_row",
        run32(&a32, vec![1, N], &row32, |s, x, y| {
            let _ = s.tensor_add(x, y);
        }),
    );
    Ok(())
}

//! diag_embed (torch.diag of a 1-D vector) f32 native fast path: parity + dtype + perf.
//!
//! The generic apply_function path upcasts the f32 input to f64, builds the n*n
//! output in f64, and returns an F64 node (dtype divergence + 2x bandwidth). The
//! native-f32 fast path builds the n*n matrix directly in f32. Pure positional
//! copy + zero fill -> bit-exact.
use ft_api::FrankenTorchSession;
use ft_core::{DType, ExecutionMode};
use std::io::Write;
use std::process::{Command, Stdio};
use std::time::Instant;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let python = std::env::var("PYTORCH_PYTHON").unwrap_or_else(|_| "python3".to_string());

    // ── parity + dtype on a small vector ────────────────────────────────────
    let small: Vec<f32> = vec![1.5, -2.25, 0.0, 7.0, -0.5, 3.125, 100.0, -42.0];
    let py_small = format!(
        r#"
import torch
a=torch.tensor({small:?},dtype=torch.float32)
m=torch.diag(a)
print("DT",m.dtype)
print("VALS"," ".join("%.9g"%v for v in m.flatten().tolist()))
"#,
        small = small
    );
    let mut ch = Command::new(&python).arg("-").stdin(Stdio::piped()).stdout(Stdio::piped()).spawn()?;
    ch.stdin.as_mut().unwrap().write_all(py_small.as_bytes())?;
    let out = ch.wait_with_output()?;
    let pt = String::from_utf8_lossy(&out.stdout).to_string();
    let pt_vals: Vec<f64> = pt
        .lines()
        .find_map(|l| l.strip_prefix("VALS "))
        .map(|s| s.split_whitespace().filter_map(|t| t.parse::<f64>().ok()).collect())
        .unwrap_or_default();

    let mut s = FrankenTorchSession::new(ExecutionMode::Strict);
    let x = s.tensor_variable_f32(small.clone(), vec![small.len()], false)?;
    let m = s.tensor_diag(x, 0)?;
    let dt = s.tensor_dtype(m)?;
    let ft_vals = s.tensor_values_lossy_f64(m)?;
    let dtype_ok = dt == DType::F32;
    let mismatch = ft_vals
        .iter()
        .zip(pt_vals.iter())
        .filter(|(a, b)| a.to_bits() != b.to_bits())
        .count()
        + ft_vals.len().abs_diff(pt_vals.len());
    println!(
        "parity: dtype={dt:?} (torch f32, ours_is_f32={dtype_ok})  value-bit mismatches: {mismatch} / {}",
        pt_vals.len()
    );

    // ── perf: large diagonal matrix construction ─────────────────────────────
    let k = 4096usize; // -> 4096x4096 = 16.8M output
    let v: Vec<f32> = (0..k).map(|i| (i as f32) * 0.5 - 1000.0).collect();
    let mut best = f64::INFINITY;
    let mut sink = 0.0f64;
    for _ in 0..9 {
        let mut s = FrankenTorchSession::new(ExecutionMode::Strict);
        let x = s.tensor_variable_f32(v.clone(), vec![k], false)?;
        let t = Instant::now();
        let m = s.tensor_diag(x, 0)?;
        let e = t.elapsed().as_secs_f64() * 1e3;
        sink += s.tensor_values_lossy_f64(m)?[0];
        if e < best {
            best = e;
        }
    }
    let py_big = format!(
        r#"
import time,torch
torch.set_num_threads(8)
k={k}
v=(torch.arange(k,dtype=torch.float32)*0.5-1000.0)
def t(fn,reps=9):
    for _ in range(2): fn()
    ts=[]
    for _ in range(reps):
        s=time.perf_counter(); fn(); ts.append((time.perf_counter()-s)*1e3)
    return min(ts)
print("PT %.4f"%t(lambda:torch.diag(v)))
"#,
        k = k
    );
    let mut ch = Command::new(&python).arg("-").stdin(Stdio::piped()).stdout(Stdio::piped()).spawn()?;
    ch.stdin.as_mut().unwrap().write_all(py_big.as_bytes())?;
    let out = ch.wait_with_output()?;
    let pt = String::from_utf8_lossy(&out.stdout).to_string();
    let ptms: f64 = pt
        .lines()
        .find_map(|l| l.strip_prefix("PT ")?.trim().parse::<f64>().ok())
        .unwrap_or(f64::NAN);
    let ratio = ptms / best;
    let verdict = if ratio >= 1.0 {
        format!("FT {ratio:.2}x FASTER")
    } else {
        format!("FT {:.2}x SLOWER", 1.0 / ratio)
    };
    println!("diag_embed f32 [{k}x{k}]: FT {best:.3} ms  PT {ptms:.3} ms  => {verdict}  (sink {sink:.3})");
    Ok(())
}

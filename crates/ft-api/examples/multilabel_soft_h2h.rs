//! f32 multilabel_soft_margin_loss vs PyTorch.

use ft_api::FrankenTorchSession;
use ft_core::ExecutionMode;
use std::io::Write;
use std::process::{Command, Stdio};
use std::time::Instant;

fn parse_metric(output: &str, name: &str) -> Option<f64> {
    output.lines().find_map(|line| {
        let mut it = line.strip_prefix("PT ")?.split_whitespace();
        if it.next()? == name {
            it.next()?.parse::<f64>().ok()
        } else {
            None
        }
    })
}

fn time_ft(
    x_data: &[f32],
    t_data: &[f32],
    rows: usize,
    cols: usize,
    reps: usize,
) -> Result<(f64, String), Box<dyn std::error::Error>> {
    let mut best = f64::INFINITY;
    let mut dtype = String::new();
    for _ in 0..reps {
        let mut s = FrankenTorchSession::new(ExecutionMode::Strict);
        let x = s
            .tensor_variable_f32(x_data.to_vec(), vec![rows, cols], false)
            .unwrap();
        let t = s
            .tensor_variable_f32(t_data.to_vec(), vec![rows, cols], false)
            .unwrap();
        let start = Instant::now();
        let out = s.tensor_multilabel_soft_margin_loss(x, t, None, "mean")?;
        let elapsed = start.elapsed().as_secs_f64() * 1e3;
        if elapsed < best {
            best = elapsed;
            dtype = format!("{:?}", s.tensor_dtype(out)?);
        }
    }
    Ok((best, dtype))
}

fn time_ft_f64(
    x_data: &[f64],
    t_data: &[f64],
    rows: usize,
    cols: usize,
    reps: usize,
) -> Result<(f64, String), Box<dyn std::error::Error>> {
    let mut best = f64::INFINITY;
    let mut dtype = String::new();
    for _ in 0..reps {
        let mut s = FrankenTorchSession::new(ExecutionMode::Strict);
        let x = s
            .tensor_variable(x_data.to_vec(), vec![rows, cols], false)
            .unwrap();
        let t = s
            .tensor_variable(t_data.to_vec(), vec![rows, cols], false)
            .unwrap();
        let start = Instant::now();
        let out = s.tensor_multilabel_soft_margin_loss(x, t, None, "mean")?;
        let elapsed = start.elapsed().as_secs_f64() * 1e3;
        if elapsed < best {
            best = elapsed;
            dtype = format!("{:?}", s.tensor_dtype(out)?);
        }
    }
    Ok((best, dtype))
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let python = std::env::var("PYTORCH_PYTHON").unwrap_or_else(|_| "python3".to_string());

    let small_x = vec![-1.0f32, 0.0, 1.0, 2.0, -2.0, 0.5];
    let small_t = vec![0.0f32, 1.0, 1.0, 0.0, 0.0, 1.0];
    let py_parity = format!(
        r#"
import torch
import torch.nn.functional as F
x=torch.tensor({small_x:?},dtype=torch.float32).reshape(2,3)
t=torch.tensor({small_t:?},dtype=torch.float32).reshape(2,3)
print("PT parity", " ".join("%.9g"%v for v in F.multilabel_soft_margin_loss(x,t,reduction='none').tolist()))
"#
    );
    let mut child = Command::new(&python)
        .arg("-")
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .spawn()?;
    child
        .stdin
        .as_mut()
        .ok_or_else(|| std::io::Error::other("missing python stdin"))?
        .write_all(py_parity.as_bytes())?;
    let parity_out = child.wait_with_output()?;
    print!("{}", String::from_utf8_lossy(&parity_out.stdout));

    let mut s = FrankenTorchSession::new(ExecutionMode::Strict);
    let sx = s.tensor_variable_f32(small_x, vec![2, 3], false)?;
    let st = s.tensor_variable_f32(small_t, vec![2, 3], false)?;
    match s.tensor_multilabel_soft_margin_loss(sx, st, None, "none") {
        Ok(out) => println!(
            "FT parity dtype={:?} values={:?}",
            s.tensor_dtype(out)?,
            s.tensor_values_lossy_f64(out)?
        ),
        Err(err) => println!("FT parity ERR {err:?}"),
    }

    let (rows, cols) = (65_536usize, 128usize);
    let n = rows * cols;
    let x_data: Vec<f32> = (0..n)
        .map(|i| -2.0 + (i % 257) as f32 * (4.0 / 257.0))
        .collect();
    let t_data: Vec<f32> = (0..n)
        .map(|i| if (i + i / cols) % 3 == 0 { 1.0 } else { 0.0 })
        .collect();

    let ft_f32 = time_ft(&x_data, &t_data, rows, cols, 7);
    let x64: Vec<f64> = x_data.iter().map(|&v| f64::from(v)).collect();
    let t64: Vec<f64> = t_data.iter().map(|&v| f64::from(v)).collect();
    let (ft64_ms, ft64_dtype) = time_ft_f64(&x64, &t64, rows, cols, 7)?;
    let py_bench = format!(
        r#"
import time, torch
import torch.nn.functional as F
torch.set_num_threads(8)
rows={rows}; cols={cols}
i=torch.arange(rows*cols,dtype=torch.int64)
x=(-2.0 + (i % 257).float() * (4.0 / 257.0)).reshape(rows,cols)
t=(((i + (i // cols)) % 3) == 0).float().reshape(rows,cols)
def tm(fn, reps=7):
    for _ in range(2): fn()
    vals=[]
    for _ in range(reps):
        s=time.perf_counter(); fn(); vals.append((time.perf_counter()-s)*1e3)
    return min(vals)
print("PT mlsoft_f32 %.4f"%tm(lambda:F.multilabel_soft_margin_loss(x,t,reduction='mean')))
print("PT mlsoft_f64 %.4f"%tm(lambda:F.multilabel_soft_margin_loss(x.double(),t.double(),reduction='mean')))
print("PT add %.4f"%tm(lambda:x+x))
"#
    );
    let mut child = Command::new(&python)
        .arg("-")
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .spawn()?;
    child
        .stdin
        .as_mut()
        .ok_or_else(|| std::io::Error::other("missing python stdin"))?
        .write_all(py_bench.as_bytes())?;
    let out = child.wait_with_output()?;
    let py_out = String::from_utf8_lossy(&out.stdout).to_string();
    print!("{py_out}");
    match ft_f32 {
        Ok((ft_ms, ft_dtype)) => {
            if let Some(pt_ms) = parse_metric(&py_out, "mlsoft_f32") {
                let ratio = pt_ms / ft_ms;
                let tag = if ratio >= 1.0 {
                    format!("FT {ratio:.2}x FASTER")
                } else {
                    format!("FT {:.2}x SLOWER", 1.0 / ratio)
                };
                println!("FT mlsoft_f32 {ft_ms:.4} dtype={ft_dtype} => {tag}");
            }
        }
        Err(err) => println!("FT mlsoft_f32 ERR {err}"),
    }
    if let Some(pt_ms) = parse_metric(&py_out, "mlsoft_f64") {
        let ratio = pt_ms / ft64_ms;
        let tag = if ratio >= 1.0 {
            format!("FT {ratio:.2}x FASTER")
        } else {
            format!("FT {:.2}x SLOWER", 1.0 / ratio)
        };
        println!("FT mlsoft_f64 {ft64_ms:.4} dtype={ft64_dtype} => {tag}");
    }
    Ok(())
}

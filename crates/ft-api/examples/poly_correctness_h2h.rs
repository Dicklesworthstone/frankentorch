// Correctness probe: FT f32 polynomial special-fns vs torch f32 (dtype + values).
use ft_api::FrankenTorchSession;
use ft_core::{DType, ExecutionMode};
use std::io::Write;
use std::process::{Command, Stdio};
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let python = std::env::var("PYTORCH_PYTHON").unwrap_or_else(|_| "python3".to_string());
    let n = 4096usize;
    let a: Vec<f32> = (0..n).map(|i| ((i % 1999) as f32 / 2000.0) - 0.5).collect();
    let mut s = FrankenTorchSession::new(ExecutionMode::Strict);
    let ops: &[(&str, u8)] = &[("cheb_t", 1), ("herm_h", 2), ("herm_he", 3), ("lag_l", 4)];
    let mut ft_out: Vec<(String, DType, Vec<f32>)> = vec![];
    for &(lbl, w) in ops {
        let x = s.tensor_variable_f32(a.clone(), vec![n], false).unwrap();
        let y = match w {
            1 => s.tensor_special_chebyshev_polynomial_t(x, 5),
            2 => s.tensor_special_hermite_polynomial_h(x, 5),
            3 => s.tensor_special_hermite_polynomial_he(x, 5),
            _ => s.tensor_special_laguerre_polynomial_l(x, 5),
        }?;
        let dt = s.tensor_dtype(y)?;
        let vals: Vec<f32> = s
            .tensor_values_lossy_f64(y)?
            .iter()
            .map(|&v| v as f32)
            .collect();
        ft_out.push((lbl.to_string(), dt, vals));
    }
    // torch reference (f32), emit as space-separated per op
    let py = format!(
        r#"
import torch
torch.set_num_threads(1)
n={n}
a=(((torch.arange(n,dtype=torch.int64)%1999).float()/2000.0)-0.5)
def emit(name,y):
    assert y.dtype==torch.float32, (name,y.dtype)
    print(name, " ".join("%a"%float(v) for v in y.tolist()))
emit("cheb_t", torch.special.chebyshev_polynomial_t(a,5))
emit("herm_h", torch.special.hermite_polynomial_h(a,5))
emit("herm_he", torch.special.hermite_polynomial_he(a,5))
emit("lag_l", torch.special.laguerre_polynomial_l(a,5))
"#
    );
    let mut ch = Command::new(&python)
        .arg("-")
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .spawn()?;
    ch.stdin.as_mut().unwrap().write_all(py.as_bytes())?;
    let out = String::from_utf8_lossy(&ch.wait_with_output()?.stdout).to_string();
    for (lbl, dt, fv) in &ft_out {
        let line = out
            .lines()
            .find(|l| l.starts_with(&format!("{lbl} ")))
            .unwrap_or("");
        let tv: Vec<f32> = line
            .split_whitespace()
            .skip(1)
            .filter_map(|t| t.parse().ok())
            .collect();
        assert_eq!(
            tv.len(),
            fv.len(),
            "{lbl}: torch len {} != ft {}",
            tv.len(),
            fv.len()
        );
        let mut max_abs = 0f32;
        let mut max_rel = 0f32;
        let mut exact = 0usize;
        for (&f, &t) in fv.iter().zip(tv.iter()) {
            if f.to_bits() == t.to_bits() {
                exact += 1;
            }
            let a = (f - t).abs();
            if a > max_abs {
                max_abs = a;
            }
            let r = if t.abs() > 0.0 { a / t.abs() } else { a };
            if r > max_rel {
                max_rel = r;
            }
        }
        println!(
            "{lbl:<8} dtype={dt:?} bit_exact={exact}/{} max_abs={max_abs:.3e} max_rel={max_rel:.3e}",
            fv.len()
        );
    }
    Ok(())
}

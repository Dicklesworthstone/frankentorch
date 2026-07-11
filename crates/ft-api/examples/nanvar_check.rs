// Correctness: FT nanvar/nanstd value vs torch masked var/std on 16M f32 (tolerance reduction).
use ft_api::FrankenTorchSession;
use ft_core::ExecutionMode;
use std::io::Write;
use std::process::{Command, Stdio};
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let python = std::env::var("PYTORCH_PYTHON").unwrap_or_else(|_| "python3".to_string());
    let n = 16_000_000usize;
    let a: Vec<f32> = (0..n)
        .map(|i| {
            if i % 101 == 0 {
                f32::NAN
            } else if i % 211 == 0 {
                f32::INFINITY
            } else {
                0.9 + ((i % 4001) as f32 / 4000.0) * 0.2
            }
        })
        .collect();
    // FT: nanvar/nanstd correction=1 (unbiased, matches torch default). NOTE: inf present -> both
    // FT and torch include inf in the non-NaN set, so the variance is inf; use a finite variant too.
    let finite: Vec<f32> = a
        .iter()
        .map(|&x| if x.is_infinite() { 1.0 } else { x })
        .collect();
    let mut s = FrankenTorchSession::new(ExecutionMode::Strict);
    let xf = s.tensor_variable_f32(finite.clone(), vec![n], false)?;
    let v = s.tensor_nanvar(xf, 1)?;
    let ft_var = s.tensor_values_lossy_f64(v)?[0];
    let xf2 = s.tensor_variable_f32(finite.clone(), vec![n], false)?;
    let sd = s.tensor_nanstd(xf2, 1)?;
    let ft_std = s.tensor_values_lossy_f64(sd)?[0];
    let py = format!(
        r#"
import torch
n={n}
idx=torch.arange(n,dtype=torch.int64)
a=(0.9+((idx%4001).float()/4000.0)*0.2)
a[idx%101==0]=float('nan')
a[idx%211==0]=1.0  # finite variant (match FT input)
m=~torch.isnan(a)
v=torch.var(a[m]); sd=torch.std(a[m])
print("REFVAR %.10e"%float(v)); print("REFSTD %.10e"%float(sd))
"#
    );
    let mut ch = Command::new(&python)
        .arg("-")
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .spawn()?;
    ch.stdin.as_mut().unwrap().write_all(py.as_bytes())?;
    let out = String::from_utf8_lossy(&ch.wait_with_output()?.stdout).to_string();
    let g = |k: &str| {
        out.lines()
            .find_map(|l| l.strip_prefix(k)?.trim().parse::<f64>().ok())
            .unwrap_or(f64::NAN)
    };
    let (tv, td) = (g("REFVAR"), g("REFSTD"));
    let rel = |a: f64, b: f64| (a - b).abs() / b.abs().max(1e-30);
    println!(
        "nanvar  FT {ft_var:.10e}  torch {tv:.10e}  rel_err {:.3e}",
        rel(ft_var, tv)
    );
    println!(
        "nanstd  FT {ft_std:.10e}  torch {td:.10e}  rel_err {:.3e}",
        rel(ft_std, td)
    );
    Ok(())
}

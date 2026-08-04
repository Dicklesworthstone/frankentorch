//! Bit-exact parity for f32 tril/triu fast paths vs torch (diagonals, NaN/inf/-0 in zeroed positions).
use ft_api::FrankenTorchSession;
use ft_core::ExecutionMode;
use std::io::Write;
use std::process::{Command, Stdio};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let python = std::env::var("PYTORCH_PYTHON").unwrap_or_else(|_| "python3".to_string());
    let (m, n) = (37usize, 41usize);
    let mut a: Vec<f32> = (0..m * n)
        .map(|i| ((i * 37 % 257) as f32 - 128.0) * 0.1)
        .collect();
    // specials that must survive in kept positions / be zeroed in masked positions
    a[0] = f32::NAN;
    a[5] = f32::INFINITY;
    a[10] = f32::NEG_INFINITY;
    a[15] = -0.0;
    a[m * n - 1] = f32::NAN;
    a[m * n - 3] = -0.0;
    let diags: Vec<i64> = vec![-40, -5, -1, 0, 1, 5, 40];

    let mut ftvals: Vec<Vec<u32>> = Vec::new();
    for &d in &diags {
        for is_tril in [true, false] {
            let mut s = FrankenTorchSession::new(ExecutionMode::Strict);
            let x = s.tensor_variable_f32(a.clone(), vec![m, n], false)?;
            let o = if is_tril {
                s.tensor_tril(x, d)?
            } else {
                s.tensor_triu(x, d)?
            };
            ftvals.push(
                s.tensor_values_f32(o)?
                    .iter()
                    .map(|v| v.to_bits())
                    .collect(),
            );
        }
    }
    let bits: Vec<u32> = a.iter().map(|v| v.to_bits()).collect();
    let py = format!(
        r#"
import struct,torch
bits={:?}
M,N={m},{n}
a=torch.tensor([struct.unpack('<f',struct.pack('<I',b))[0] for b in bits],dtype=torch.float32).reshape(M,N)
diags={:?}
def emit(t): print(' '.join(str(struct.unpack('<I',struct.pack('<f',v))[0]) for v in t.flatten().tolist()))
for d in diags:
    emit(torch.tril(a,d)); emit(torch.triu(a,d))
"#,
        bits, diags
    );
    let mut ch = Command::new(&python)
        .arg("-")
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .spawn()?;
    ch.stdin.as_mut().unwrap().write_all(py.as_bytes())?;
    let o = ch.wait_with_output()?;
    let out = String::from_utf8_lossy(&o.stdout);
    let pt: Vec<Vec<u32>> = out
        .lines()
        .map(|l| {
            l.split_whitespace()
                .filter_map(|t| t.parse().ok())
                .collect()
        })
        .collect();
    let mut mm = 0;
    for k in 0..ftvals.len() {
        for i in 0..ftvals[k].len() {
            let fb = ftvals[k][i];
            let pb = pt[k][i];
            let eq = fb == pb || (f32::from_bits(fb).is_nan() && f32::from_bits(pb).is_nan());
            if !eq {
                mm += 1;
                if mm <= 3 {
                    println!(
                        "case {k} idx {i}: ft={} pt={}",
                        f32::from_bits(fb),
                        f32::from_bits(pb)
                    );
                }
            }
        }
    }
    println!("=> {mm} mismatches across {} (diag,op) cases", ftvals.len());
    Ok(())
}

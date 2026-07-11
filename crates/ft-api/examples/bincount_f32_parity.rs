//! Bit-exact parity for f32 bincount fast path vs torch.
use ft_api::FrankenTorchSession;
use ft_core::ExecutionMode;
use std::io::Write;
use std::process::{Command, Stdio};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let python = std::env::var("PYTORCH_PYTHON").unwrap_or_else(|_| "python3".to_string());
    // mixed counts, some gaps, zeros, a large-ish max
    let n = 300_007usize;
    let a: Vec<f32> = (0..n)
        .map(|i| ((i * 2654435761usize) % 257) as f32)
        .collect();
    let mut s = FrankenTorchSession::new(ExecutionMode::Strict);
    let x = s.tensor_variable_f32(a.clone(), vec![n], false)?;
    let b = s.tensor_bincount(x, None, 0)?;
    let ft: Vec<f64> = s.tensor_values(b)?;

    let bits: Vec<u32> = a.iter().map(|v| v.to_bits()).collect();
    let py = format!(
        r#"
import struct,torch
bits={:?}
a=torch.tensor([struct.unpack('<f',struct.pack('<I',b))[0] for b in bits],dtype=torch.float32).long()
out=torch.bincount(a)
print(len(out))
print(' '.join(str(int(v)) for v in out.tolist()))
"#,
        bits
    );
    let mut ch = Command::new(&python)
        .arg("-")
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .spawn()?;
    ch.stdin.as_mut().unwrap().write_all(py.as_bytes())?;
    let o = ch.wait_with_output()?;
    let out = String::from_utf8_lossy(&o.stdout);
    let mut lines = out.lines();
    let pt_len: usize = lines.next().unwrap_or("0").trim().parse().unwrap_or(0);
    let pt: Vec<i64> = lines
        .next()
        .unwrap_or("")
        .split_whitespace()
        .filter_map(|t| t.parse().ok())
        .collect();
    println!("ft len={} pt len={}", ft.len(), pt_len);
    let mut mm = if ft.len() != pt_len { 1 } else { 0 };
    for i in 0..ft.len().min(pt.len()) {
        if ft[i] as i64 != pt[i] {
            mm += 1;
            println!("bin {i}: ft={} pt={}", ft[i] as i64, pt[i]);
        }
    }
    println!("=> {mm} mismatches (len+bins)");
    Ok(())
}

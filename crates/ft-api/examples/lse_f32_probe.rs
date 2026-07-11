//! Check f32 logsumexp: output dtype + bit-parity vs torch (decides the fast-path approach).
use ft_api::FrankenTorchSession;
use ft_core::{DType, ExecutionMode};
use std::io::Write;
use std::process::{Command, Stdio};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let python = std::env::var("PYTORCH_PYTHON").unwrap_or_else(|_| "python3".to_string());
    let (r, c) = (337usize, 251usize);
    let a: Vec<f32> = (0..r * c)
        .map(|i| (((i * 37) % 613) as f32 - 306.0) * 0.01)
        .collect();

    let mut s = FrankenTorchSession::new(ExecutionMode::Strict);
    let x = s.tensor_variable_f32(a.clone(), vec![r, c], false)?;
    let n = s.tensor_logsumexp(x, 1)?;
    let dt = s.tensor_dtype(n)?;
    println!("ft logsumexp output dtype = {:?}", dt);
    let ft: Vec<f32> = s.tensor_values_f32(n).unwrap_or_else(|_| {
        // if it returned f64, read as f64 then cast
        vec![]
    });
    let ftbits: Vec<u32> = ft.iter().map(|v| v.to_bits()).collect();

    let bits: Vec<u32> = a.iter().map(|x| x.to_bits()).collect();
    let py = format!(
        r#"
import struct,torch
bits={:?}
R,C={r},{c}
a=torch.tensor([struct.unpack('<f',struct.pack('<I',b))[0] for b in bits],dtype=torch.float32).reshape(R,C)
out=torch.logsumexp(a,1)
print('DTYPE',out.dtype)
print(' '.join(str(struct.unpack('<I',struct.pack('<f',v))[0]) for v in out.tolist()))
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
    println!("torch {}", lines.next().unwrap_or(""));
    let pt: Vec<u32> = lines
        .next()
        .unwrap_or("")
        .split_whitespace()
        .filter_map(|t| t.parse().ok())
        .collect();
    if dt == DType::F32 && !ftbits.is_empty() {
        let mut mm = 0;
        let mut maxulp = 0i64;
        let mut first = None;
        for i in 0..ftbits.len().min(pt.len()) {
            if ftbits[i] != pt[i] {
                mm += 1;
                let d = (ftbits[i] as i64 - pt[i] as i64).abs();
                if d > maxulp {
                    maxulp = d;
                }
                if first.is_none() {
                    first = Some((i, ftbits[i], pt[i]));
                }
            }
        }
        println!(
            "ft_f32 vs torch: {mm}/{} mismatches maxulpbits={maxulp} {:?}",
            ftbits.len(),
            first
        );
    } else {
        println!(
            "ft did not return f32 (dtype={:?}) — cannot bit-compare as f32",
            dt
        );
    }
    Ok(())
}

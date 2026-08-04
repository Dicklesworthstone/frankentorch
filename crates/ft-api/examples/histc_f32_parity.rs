//! Bit-exact parity for f32 histc fast path vs torch (boundary values, out-of-range).
use ft_api::FrankenTorchSession;
use ft_core::ExecutionMode;
use std::io::Write;
use std::process::{Command, Stdio};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let python = std::env::var("PYTORCH_PYTHON").unwrap_or_else(|_| "python3".to_string());
    // values spanning [-3,3] incl exact bin boundaries + some out-of-range
    let n = 500_003usize;
    let mut a: Vec<f32> = (0..n)
        .map(|i| (((i * 2654435761usize) % 800_003) as f32) * 0.00001 - 4.0)
        .collect();
    // sprinkle exact boundary values for bins over [-2,2], 8 bins (width 0.5)
    for (k, slot) in a.iter_mut().take(16).enumerate() {
        *slot = -2.0 + k as f32 * 0.5;
    }
    a[20] = 2.0;
    a[21] = -2.0;
    a[22] = -10.0;
    a[23] = 10.0; // boundaries + out of range
    let bins = 8usize;
    let (lo, hi) = (-2.0_f64, 2.0_f64);

    let mut s = FrankenTorchSession::new(ExecutionMode::Strict);
    let x = s.tensor_variable_f32(a.clone(), vec![n], false)?;
    let h = s.tensor_histc(x, bins, lo, hi)?;
    let ft: Vec<f64> = s.tensor_values(h)?;

    let bits: Vec<u32> = a.iter().map(|v| v.to_bits()).collect();
    let py = format!(
        r#"
import struct,torch
bits={:?}
a=torch.tensor([struct.unpack('<f',struct.pack('<I',b))[0] for b in bits],dtype=torch.float32)
out=torch.histc(a,bins={bins},min={lo},max={hi})
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
    let pt: Vec<i64> = out
        .lines()
        .next()
        .unwrap_or("")
        .split_whitespace()
        .filter_map(|t| t.parse().ok())
        .collect();
    let mut mm = 0;
    for i in 0..ft.len() {
        let fi = ft[i] as i64;
        if fi != pt[i] {
            mm += 1;
            println!("bin {i}: ft={fi} pt={}", pt[i]);
        }
    }
    println!(
        "ft counts: {:?}",
        ft.iter().map(|&v| v as i64).collect::<Vec<_>>()
    );
    println!("=> {mm}/{} bin mismatches", ft.len());
    Ok(())
}

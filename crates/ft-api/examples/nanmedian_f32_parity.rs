//! Bit-exact parity for f32 nanmedian fast path vs torch.
use ft_api::FrankenTorchSession;
use ft_core::ExecutionMode;
use std::io::Write;
use std::process::{Command, Stdio};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let python = std::env::var("PYTORCH_PYTHON").unwrap_or_else(|_| "python3".to_string());
    let cases: Vec<Vec<f32>> = vec![
        (0..101)
            .map(|i| ((i * 37 % 101) as f32 - 50.0) * 0.1)
            .collect(), // odd
        (0..100)
            .map(|i| ((i * 37 % 100) as f32 - 50.0) * 0.1)
            .collect(), // even
        vec![3.0, f32::NAN, 1.0, 2.0, f32::NAN, 5.0, 4.0], // NaNs ignored, odd non-nan
        vec![1.0, f32::NAN, 2.0, 3.0],                     // even non-nan
        vec![0.0, -0.0, 0.0, -0.0, 1.0],                   // signed zeros
        vec![f32::NAN, f32::NAN, f32::NAN],                // all NaN
        vec![f32::INFINITY, 1.0, f32::NAN, 2.0, f32::NEG_INFINITY], // infs + nan
        (0..50_001)
            .map(|i| {
                if i % 3 == 0 {
                    f32::NAN
                } else {
                    ((i * 7919 % 5000) as f32 - 2500.0) * 0.01
                }
            })
            .collect(),
    ];
    let mut ftb: Vec<u32> = Vec::new();
    for c in &cases {
        let mut s = FrankenTorchSession::new(ExecutionMode::Strict);
        let x = s.tensor_variable_f32(c.clone(), vec![c.len()], false)?;
        let m = s.tensor_nanmedian(x)?;
        ftb.push(s.tensor_values_f32(m)?[0].to_bits());
    }
    let payload: Vec<Vec<u32>> = cases
        .iter()
        .map(|c| c.iter().map(|v| v.to_bits()).collect())
        .collect();
    let py = format!(
        r#"
import struct,torch
cases={:?}
for bits in cases:
    a=torch.tensor([struct.unpack('<f',struct.pack('<I',b))[0] for b in bits],dtype=torch.float32)
    m=torch.nanmedian(a)
    print(struct.unpack('<I',struct.pack('<f',float(m)))[0])
"#,
        payload
    );
    let mut ch = Command::new(&python)
        .arg("-")
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .spawn()?;
    ch.stdin.as_mut().unwrap().write_all(py.as_bytes())?;
    let o = ch.wait_with_output()?;
    let out = String::from_utf8_lossy(&o.stdout);
    let pt: Vec<u32> = out.lines().filter_map(|l| l.trim().parse().ok()).collect();
    let mut mm = 0;
    for i in 0..ftb.len() {
        let eq =
            ftb[i] == pt[i] || (f32::from_bits(ftb[i]).is_nan() && f32::from_bits(pt[i]).is_nan());
        let mark = if eq {
            "ok"
        } else {
            mm += 1;
            "MISMATCH"
        };
        println!(
            "case {i}: ft={} pt={}  {mark}",
            f32::from_bits(ftb[i]),
            f32::from_bits(pt[i])
        );
    }
    println!("=> {mm}/{} mismatches", ftb.len());
    Ok(())
}

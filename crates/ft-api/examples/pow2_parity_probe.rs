//! Verify f32 pow(x,2.0): is x*x bit-exact vs torch? Is ft's current powf bit-exact vs torch?
use ft_api::FrankenTorchSession;
use ft_core::ExecutionMode;
use std::io::Write;
use std::process::{Command, Stdio};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let python = std::env::var("PYTORCH_PYTHON").unwrap_or_else(|_| "python3".to_string());
    // wide value sweep + edges
    let mut a: Vec<f32> = Vec::new();
    for i in 0..20000u32 {
        a.push((i as f32 - 10000.0) * 0.0007);
    }
    a.extend_from_slice(&[
        0.0,
        -0.0,
        1.0,
        -1.0,
        2.0,
        -2.0,
        0.5,
        -0.5,
        1e-20,
        -1e-20,
        1e20,
        -1e20,
        3.4e38,
        -3.4e38,
        1e-38,
        f32::INFINITY,
        f32::NEG_INFINITY,
        f32::NAN,
        std::f32::consts::PI,
    ]);
    let n = a.len();

    let mut s = FrankenTorchSession::new(ExecutionMode::Strict);
    let x = s.tensor_variable_f32(a.clone(), vec![n], false)?;
    let np = s.tensor_pow(x, 2.0)?;
    let ft_pow = s.tensor_values_f32(np)?;
    let xx: Vec<f32> = a.iter().map(|&v| v * v).collect();

    let bits: Vec<u32> = a.iter().map(|x| x.to_bits()).collect();
    let py = format!(
        r#"
import struct,torch
bits={:?}
a=torch.tensor([struct.unpack('<f',struct.pack('<I',b))[0] for b in bits],dtype=torch.float32)
out=a.pow(2.0)
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
    let pt: Vec<u32> = out
        .lines()
        .next()
        .unwrap_or("")
        .split_whitespace()
        .filter_map(|t| t.parse().ok())
        .collect();
    let cmp = |name: &str, v: &[f32]| {
        let mut mm = 0;
        let mut first = None;
        let mut maxulp = 0i64;
        for i in 0..v.len() {
            let fb = v[i].to_bits();
            let eq = fb == pt[i] || (f32::from_bits(fb).is_nan() && f32::from_bits(pt[i]).is_nan());
            if !eq {
                mm += 1;
                let d = (fb as i64 - pt[i] as i64).abs();
                if d > maxulp {
                    maxulp = d;
                }
                if first.is_none() {
                    first = Some((i, fb, pt[i]));
                }
            }
        }
        println!(
            "{name:<12}: {mm}/{} mismatches maxulpbits={maxulp}  {:?}",
            v.len(),
            first
        );
    };
    cmp("ft_pow", &ft_pow);
    cmp("x*x", &xx);
    Ok(())
}

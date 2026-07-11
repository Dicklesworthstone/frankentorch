//! Bit-exact parity + dtype for f32 masked_fill fast path vs torch.
use ft_api::FrankenTorchSession;
use ft_core::{DType, ExecutionMode};
use std::io::Write;
use std::process::{Command, Stdio};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let python = std::env::var("PYTORCH_PYTHON").unwrap_or_else(|_| "python3".to_string());
    let n = 100_003usize;
    let mut a: Vec<f32> = (0..n)
        .map(|i| ((i * 37 % 257) as f32 - 128.0) * 0.1)
        .collect();
    let mask: Vec<f32> = (0..n).map(|i| (i % 3 == 0) as i32 as f32).collect();
    a[0] = f32::NAN;
    a[1] = f32::INFINITY;
    a[2] = f32::NEG_INFINITY;
    a[3] = -0.0;
    let value = -1.5_f64;

    let mut s = FrankenTorchSession::new(ExecutionMode::Strict);
    let x = s.tensor_variable_f32(a.clone(), vec![n], false)?;
    let mk = s.tensor_variable_f32(mask.clone(), vec![n], false)?;
    let o = s.tensor_masked_fill(x, mk, value)?;
    let dt = s.tensor_dtype(o)?;
    println!("ft masked_fill output dtype = {:?} (torch -> f32)", dt);
    let ft = s.tensor_values_f32(o)?;

    let bits: Vec<u32> = a.iter().map(|v| v.to_bits()).collect();
    let mbits: Vec<u32> = mask.iter().map(|v| v.to_bits()).collect();
    let py = format!(
        r#"
import struct,torch
ba={:?}
bm={:?}
a=torch.tensor([struct.unpack('<f',struct.pack('<I',b))[0] for b in ba],dtype=torch.float32)
m=torch.tensor([struct.unpack('<f',struct.pack('<I',b))[0] for b in bm],dtype=torch.float32)!=0
out=a.masked_fill(m,{value})
print(' '.join(str(struct.unpack('<I',struct.pack('<f',v))[0]) for v in out.tolist()))
"#,
        bits, mbits
    );
    let mut ch = Command::new(&python)
        .arg("-")
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .spawn()?;
    ch.stdin.as_mut().unwrap().write_all(py.as_bytes())?;
    let o2 = ch.wait_with_output()?;
    let out = String::from_utf8_lossy(&o2.stdout);
    let pt: Vec<u32> = out
        .lines()
        .next()
        .unwrap_or("")
        .split_whitespace()
        .filter_map(|t| t.parse().ok())
        .collect();
    let mut mm = if dt != DType::F32 {
        println!("DTYPE MISMATCH");
        1
    } else {
        0
    };
    for i in 0..ft.len() {
        let fb = ft[i].to_bits();
        let pb = pt[i];
        let eq = fb == pb || (f32::from_bits(fb).is_nan() && f32::from_bits(pb).is_nan());
        if !eq {
            mm += 1;
            if mm <= 3 {
                println!(
                    "idx {i}: ft={} pt={}",
                    f32::from_bits(fb),
                    f32::from_bits(pb)
                );
            }
        }
    }
    println!("=> {mm}/{} mismatches", ft.len());
    Ok(())
}

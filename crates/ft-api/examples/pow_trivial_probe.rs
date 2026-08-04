//! Check which trivial exponents' cheap-op result is bit-exact vs torch f32 pow.
use std::io::Write;
use std::process::{Command, Stdio};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let python = std::env::var("PYTORCH_PYTHON").unwrap_or_else(|_| "python3".to_string());
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
        4.0,
        0.25,
        1e-20,
        1e20,
        3.4e38,
        1e-38,
        f32::INFINITY,
        f32::NEG_INFINITY,
        f32::NAN,
    ]);
    let n = a.len();
    // candidate cheap ops
    let id: Vec<f32> = a.clone();
    let sq: Vec<f32> = a.iter().map(|&v| v.sqrt()).collect();
    let cube: Vec<f32> = a.iter().map(|&v| v * v * v).collect();
    let cube2: Vec<f32> = a.iter().map(|&v| v * (v * v)).collect();
    let recip: Vec<f32> = a.iter().map(|&v| 1.0f32 / v).collect();

    let bits: Vec<u32> = a.iter().map(|x| x.to_bits()).collect();
    let py = format!(
        r#"
import struct,torch
bits={:?}
a=torch.tensor([struct.unpack('<f',struct.pack('<I',b))[0] for b in bits],dtype=torch.float32)
def emit(t): print(' '.join(str(struct.unpack('<I',struct.pack('<f',v))[0]) for v in t.tolist()))
emit(a.pow(1.0)); emit(a.pow(0.5)); emit(a.pow(3.0)); emit(a.pow(-1.0))
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
    let l: Vec<Vec<u32>> = out
        .lines()
        .map(|x| {
            x.split_whitespace()
                .filter_map(|t| t.parse().ok())
                .collect()
        })
        .collect();
    let cmp = |name: &str, v: &[f32], pt: &[u32]| {
        let mut mm = 0;
        let mut first = None;
        for i in 0..v.len() {
            let fb = v[i].to_bits();
            let eq = fb == pt[i] || (f32::from_bits(fb).is_nan() && f32::from_bits(pt[i]).is_nan());
            if !eq {
                mm += 1;
                if first.is_none() {
                    first = Some((i, a_val(i), fb, pt[i]));
                }
            }
        }
        println!("{name:<16}: {mm}/{} mismatches  {:?}", v.len(), first);
    };
    fn a_val(_i: usize) -> u8 {
        0
    }
    cmp("id vs pow(1)", &id, &l[0]);
    cmp("sqrt vs pow(.5)", &sq, &l[1]);
    cmp("x*x*x vs pow(3)", &cube, &l[2]);
    cmp("x*(x*x) vs pow3", &cube2, &l[2]);
    cmp("1/x vs pow(-1)", &recip, &l[3]);
    let _ = n;
    Ok(())
}

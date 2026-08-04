//! Bit-exact parity for f32 amax/amin (dim0 strided + dim1 SIMD) vs torch.
use ft_api::FrankenTorchSession;
use ft_core::ExecutionMode;
use std::io::Write;
use std::process::{Command, Stdio};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let python = std::env::var("PYTORCH_PYTHON").unwrap_or_else(|_| "python3".to_string());
    let (r, c) = (337usize, 251usize); // neither multiple of 8
    let mut a: Vec<f32> = (0..r * c)
        .map(|i| (((i * 37) % 613) as f32 - 306.0) * 0.013)
        .collect();
    // seed specials: a NaN in one column + one row, ±0 ties
    a[5 * c + 7] = f32::NAN;
    a[10 * c + 11] = f32::from_bits(0x7fc0_abcd);
    a[0] = -0.0;
    a[1] = 0.0;
    a[c] = 0.0;
    a[c + 1] = -0.0;
    a[20 * c + 20] = f32::INFINITY;
    a[21 * c + 21] = f32::NEG_INFINITY;

    let mut s = FrankenTorchSession::new(ExecutionMode::Strict);
    let x = s.tensor_variable_f32(a.clone(), vec![r, c], false)?;
    let n0 = s.tensor_amax(x, 0)?;
    let amax0 = s.tensor_values_f32(n0)?;
    let n1 = s.tensor_amax(x, 1)?;
    let amax1 = s.tensor_values_f32(n1)?;
    let x2 = s.tensor_variable_f32(a.clone(), vec![r, c], false)?;
    let m0 = s.tensor_amin(x2, 0)?;
    let amin0 = s.tensor_values_f32(m0)?;
    let m1 = s.tensor_amin(x2, 1)?;
    let amin1 = s.tensor_values_f32(m1)?;

    let bits: Vec<u32> = a.iter().map(|x| x.to_bits()).collect();
    let py = format!(
        r#"
import struct,torch
bits={:?}
R,C={r},{c}
a=torch.tensor([struct.unpack('<f',struct.pack('<I',b))[0] for b in bits],dtype=torch.float32).reshape(R,C)
def emit(t):
    print(' '.join(str(struct.unpack('<I',struct.pack('<f',v))[0]) for v in t.flatten().tolist()))
emit(a.amax(0)); emit(a.amax(1)); emit(a.amin(0)); emit(a.amin(1))
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
    let lines: Vec<&str> = out.lines().collect();
    let parse = |l: &str| -> Vec<u32> {
        l.split_whitespace()
            .filter_map(|t| t.parse().ok())
            .collect()
    };
    let cmp = |name: &str, ft: &[f32], pt: &[u32]| {
        let mut mm = 0;
        let mut first = None;
        for i in 0..ft.len() {
            let fb = ft[i].to_bits();
            let eq = fb == pt[i] || (f32::from_bits(fb).is_nan() && f32::from_bits(pt[i]).is_nan());
            if !eq {
                mm += 1;
                if first.is_none() {
                    first = Some((i, fb, pt[i]));
                }
            }
        }
        println!("{name:<8}: {mm}/{} mismatches  {:?}", ft.len(), first);
    };
    cmp("amax0", &amax0, &parse(lines[0]));
    cmp("amax1", &amax1, &parse(lines[1]));
    cmp("amin0", &amin0, &parse(lines[2]));
    cmp("amin1", &amin1, &parse(lines[3]));
    Ok(())
}

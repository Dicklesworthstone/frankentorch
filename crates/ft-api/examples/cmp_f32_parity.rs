//! Bit-exact parity for f32 comparison borrow fast path (eq/ne/lt/gt/le/ge) vs torch.
use ft_api::FrankenTorchSession;
use ft_core::ExecutionMode;
use std::io::Write;
use std::process::{Command, Stdio};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let python = std::env::var("PYTORCH_PYTHON").unwrap_or_else(|_| "python3".to_string());
    let n: usize = 100_003;
    // include equal values (ties), NaN, ±inf, ±0 so every comparison branch is hit
    let mut a: Vec<f32> = (0..n).map(|i| (i % 7) as f32 - 3.0).collect();
    let mut b: Vec<f32> = (0..n).map(|i| (i % 5) as f32 - 2.0).collect();
    a[10] = f32::NAN;
    b[10] = f32::NAN;
    a[11] = f32::INFINITY;
    b[11] = f32::INFINITY;
    a[12] = -0.0;
    b[12] = 0.0;
    a[13] = f32::NAN;
    b[13] = 1.0;

    let mut s = FrankenTorchSession::new(ExecutionMode::Strict);
    let x = s.tensor_variable_f32(a.clone(), vec![n], false)?;
    let y = s.tensor_variable_f32(b.clone(), vec![n], false)?;
    let neq = s.tensor_eq(x, y)?;
    let eq = s.tensor_values_f32(neq)?;
    let nne = s.tensor_ne(x, y)?;
    let ne = s.tensor_values_f32(nne)?;
    let nlt = s.tensor_lt(x, y)?;
    let lt = s.tensor_values_f32(nlt)?;
    let ngt = s.tensor_gt(x, y)?;
    let gt = s.tensor_values_f32(ngt)?;
    let nle = s.tensor_le(x, y)?;
    let le = s.tensor_values_f32(nle)?;
    let nge = s.tensor_ge(x, y)?;
    let ge = s.tensor_values_f32(nge)?;

    let ba: Vec<u32> = a.iter().map(|x| x.to_bits()).collect();
    let bb: Vec<u32> = b.iter().map(|x| x.to_bits()).collect();
    let py = format!(
        r#"
import struct,torch
ba={:?}
bb={:?}
a=torch.tensor([struct.unpack('<f',struct.pack('<I',x))[0] for x in ba],dtype=torch.float32)
b=torch.tensor([struct.unpack('<f',struct.pack('<I',x))[0] for x in bb],dtype=torch.float32)
def emit(t): print(' '.join('1' if v else '0' for v in t.tolist()))
emit(a.eq(b)); emit(a.ne(b)); emit(a.lt(b)); emit(a.gt(b)); emit(a.le(b)); emit(a.ge(b))
"#,
        ba, bb
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
    let parse = |l: &str| -> Vec<f32> {
        l.split_whitespace()
            .map(|t| if t == "1" { 1.0 } else { 0.0 })
            .collect()
    };
    let cmp = |name: &str, ft: &[f32], pt: Vec<f32>| {
        let mut mm = 0;
        let mut first = None;
        for i in 0..ft.len() {
            if ft[i] != pt[i] {
                mm += 1;
                if first.is_none() {
                    first = Some((i, ft[i], pt[i]));
                }
            }
        }
        println!("{name:<4}: {mm}/{} mismatches  {:?}", ft.len(), first);
    };
    cmp("eq", &eq, parse(lines[0]));
    cmp("ne", &ne, parse(lines[1]));
    cmp("lt", &lt, parse(lines[2]));
    cmp("gt", &gt, parse(lines[3]));
    cmp("le", &le, parse(lines[4]));
    cmp("ge", &ge, parse(lines[5]));
    Ok(())
}

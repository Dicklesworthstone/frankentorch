//! Bit-exact parity for native-f32 clamp vs torch (incl NaN/±inf/±0/boundaries).
use std::io::Write; use std::process::{Command, Stdio};
use ft_api::FrankenTorchSession; use ft_core::ExecutionMode;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let python = std::env::var("PYTORCH_PYTHON").unwrap_or_else(|_| "python3".to_string());
    let n: usize = 100_003;
    let mut a: Vec<f32> = (0..n).map(|i| (((i*31)%2003) as f32 - 1001.0) * 0.0017).collect();
    a[5] = f32::NAN; a[6] = f32::INFINITY; a[7] = f32::NEG_INFINITY;
    a[8] = -0.0; a[9] = 0.0;
    a[10] = -1.0; a[11] = 1.0; a[12] = -1.0000001; a[13] = 1.0000001;
    let (lo, hi) = (-1.0_f64, 1.0_f64);

    let mut s = FrankenTorchSession::new(ExecutionMode::Strict);
    let x = s.tensor_variable_f32(a.clone(), vec![n], false)?;
    let nc = s.tensor_clamp(x, lo, hi)?;
    let clamp = s.tensor_values_f32(nc)?;

    let bits: Vec<u32> = a.iter().map(|x| x.to_bits()).collect();
    let py = format!(r#"
import struct,torch
bits={:?}
a=torch.tensor([struct.unpack('<f',struct.pack('<I',b))[0] for b in bits],dtype=torch.float32)
out=torch.clamp(a,{lo},{hi})
print(' '.join(str(struct.unpack('<I',struct.pack('<f',v))[0]) for v in out.tolist()))
"#, bits);
    let mut ch=Command::new(&python).arg("-").stdin(Stdio::piped()).stdout(Stdio::piped()).spawn()?;
    ch.stdin.as_mut().unwrap().write_all(py.as_bytes())?;
    let o=ch.wait_with_output()?; let out=String::from_utf8_lossy(&o.stdout);
    let pt: Vec<u32> = out.lines().next().unwrap_or("").split_whitespace().filter_map(|t|t.parse().ok()).collect();
    let mut mm=0; let mut first=None;
    for i in 0..clamp.len(){ let fb=clamp[i].to_bits();
        let eq = fb==pt[i] || (f32::from_bits(fb).is_nan() && f32::from_bits(pt[i]).is_nan());
        if !eq { mm+=1; if first.is_none(){first=Some((i,fb,pt[i]));}} }
    println!("clamp: {mm}/{} mismatches  {:?}", clamp.len(), first);
    Ok(())
}

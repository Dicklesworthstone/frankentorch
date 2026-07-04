//! Bit-exact parity for parallel-SIMD f32 add/sub/mul/div vs torch.
//! Awkward size (not multiple of 8 nor CHUNK) + specials in the tail.
use std::io::Write; use std::process::{Command, Stdio};
use ft_api::FrankenTorchSession; use ft_core::ExecutionMode;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let python = std::env::var("PYTORCH_PYTHON").unwrap_or_else(|_| "python3".to_string());
    let n: usize = 1_000_003;
    let mut a: Vec<f32> = (0..n).map(|i| ((i % 4099) as f32 - 2049.5) * 0.013).collect();
    let mut b: Vec<f32> = (0..n).map(|i| ((i % 3001) as f32 - 1500.5) * 0.017 + 1.0).collect();
    let t = n - 5;
    a[t]=f32::INFINITY; a[t+1]=f32::NEG_INFINITY; a[t+2]=f32::NAN; a[t+3] = -0.0; a[t+4]=0.0;
    b[t]=2.0; b[t+1] = -3.0; b[t+2]=0.0; b[t+3]=f32::INFINITY; b[t+4]=f32::NAN;

    let mut s = FrankenTorchSession::new(ExecutionMode::Strict);
    let xa = s.tensor_variable_f32(a.clone(), vec![n], false)?;
    let xb = s.tensor_variable_f32(b.clone(), vec![n], false)?;
    let na = s.tensor_add(xa, xb)?; let add = s.tensor_values_f32(na)?;
    let ns = s.tensor_sub(xa, xb)?; let sub = s.tensor_values_f32(ns)?;
    let nm = s.tensor_mul(xa, xb)?; let mul = s.tensor_values_f32(nm)?;
    let nd = s.tensor_div(xa, xb)?; let div = s.tensor_values_f32(nd)?;

    let ba: Vec<u32> = a.iter().map(|x| x.to_bits()).collect();
    let bb: Vec<u32> = b.iter().map(|x| x.to_bits()).collect();
    let py = format!(r#"
import struct,torch
ba={:?}
bb={:?}
a=torch.tensor([struct.unpack('<f',struct.pack('<I',x))[0] for x in ba],dtype=torch.float32)
b=torch.tensor([struct.unpack('<f',struct.pack('<I',x))[0] for x in bb],dtype=torch.float32)
def emit(t):
    print(' '.join(str(struct.unpack('<I',struct.pack('<f',v))[0]) for v in t.tolist()))
emit(a+b); emit(a-b); emit(a*b); emit(a/b)
"#, ba, bb);
    let mut ch=Command::new(&python).arg("-").stdin(Stdio::piped()).stdout(Stdio::piped()).spawn()?;
    ch.stdin.as_mut().unwrap().write_all(py.as_bytes())?;
    let o=ch.wait_with_output()?; let out=String::from_utf8_lossy(&o.stdout);
    let lines: Vec<&str> = out.lines().collect();
    let parse=|l:&str|->Vec<u32>{l.split_whitespace().filter_map(|t|t.parse().ok()).collect()};
    let cmp=|name:&str, ft:&[f32], pt:&[u32]|{
        let mut mm=0; let mut first=None;
        for i in 0..ft.len(){ let fb=ft[i].to_bits();
            let eq = fb==pt[i] || (f32::from_bits(fb).is_nan() && f32::from_bits(pt[i]).is_nan());
            if !eq { mm+=1; if first.is_none(){first=Some((i,fb,pt[i]));}} }
        println!("{name:<4}: {mm}/{} mismatches  {:?}", ft.len(), first);
    };
    cmp("add", &add, &parse(lines[0]));
    cmp("sub", &sub, &parse(lines[1]));
    cmp("mul", &mul, &parse(lines[2]));
    cmp("div", &div, &parse(lines[3]));
    Ok(())
}

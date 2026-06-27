//! Bit-exact parity + timing for f32 relu6 / hardshrink vs torch.
use std::io::Write; use std::process::{Command, Stdio}; use std::time::Instant;
use ft_api::FrankenTorchSession; use ft_core::ExecutionMode;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let python = std::env::var("PYTORCH_PYTHON").unwrap_or_else(|_| "python3".to_string());
    // Parity set: span normal, boundary (0,6,-6,0.5,-0.5), inf, nan, tiny, subnormal.
    let mut a: Vec<f32> = Vec::new();
    for i in 0..4000u32 { a.push((i as f32 - 2000.0) * 0.003); }
    a.extend_from_slice(&[0.0, 6.0, -6.0, 0.5, -0.5, 6.000001, 5.999999, -0.0,
        f32::INFINITY, f32::NEG_INFINITY, f32::NAN, 1e-30, -1e-30, 1e30, -1e30]);
    let n = a.len();
    // FT compute
    let mut s = FrankenTorchSession::new(ExecutionMode::Strict);
    let x = s.tensor_variable_f32(a.clone(), vec![n], false)?;
    let r6 = s.tensor_relu6(x)?;
    let r6v = s.tensor_values_f32(r6)?;
    let x2 = s.tensor_variable_f32(a.clone(), vec![n], false)?;
    let hs = s.tensor_hardshrink(x2, 0.5)?;
    let hsv = s.tensor_values_f32(hs)?;
    // Send bits to torch, get back bits
    let bits_a: Vec<u32> = a.iter().map(|x| x.to_bits()).collect();
    let py = format!(r#"
import sys,struct,torch
import torch.nn.functional as F
bits={:?}
a=torch.tensor([struct.unpack('<f',struct.pack('<I',b))[0] for b in bits],dtype=torch.float32)
r6=F.relu6(a); hs=F.hardshrink(a,0.5)
def emit(t):
    print(' '.join(str(struct.unpack('<I',struct.pack('<f',v))[0]) for v in t.tolist()))
emit(r6); emit(hs)
"#, bits_a);
    let mut ch=Command::new(&python).arg("-").stdin(Stdio::piped()).stdout(Stdio::piped()).spawn()?;
    ch.stdin.as_mut().unwrap().write_all(py.as_bytes())?;
    let o=ch.wait_with_output()?; let out=String::from_utf8_lossy(&o.stdout);
    let lines: Vec<&str> = out.lines().collect();
    let parse=|l:&str|->Vec<u32>{l.split_whitespace().filter_map(|t|t.parse().ok()).collect()};
    let pr6=parse(lines[0]); let phs=parse(lines[1]);
    let cmp=|name:&str, ft:&[f32], pt:&[u32]|{
        let mut mm=0; let mut first=None;
        for i in 0..ft.len(){ let fb=ft[i].to_bits();
            // NaN: any NaN bit pattern counts as equal
            let eq = fb==pt[i] || (f32::from_bits(fb).is_nan() && f32::from_bits(pt[i]).is_nan());
            if !eq { mm+=1; if first.is_none(){first=Some((i,fb,pt[i]));}} }
        println!("{name}: {mm}/{} mismatches  {:?}", ft.len(), first);
    };
    cmp("relu6", &r6v, &pr6);
    cmp("hardshrink", &hsv, &phs);
    // timing 4000x4000
    const R:usize=4000; const C:usize=4000;
    let big: Vec<f32> = (0..R*C).map(|i| ((i%2000) as f32 - 1000.0)*0.01).collect();
    let tim=|name:&str, f:&dyn Fn(&mut FrankenTorchSession, ft_autograd::TensorNodeId)|{
        let mut b=f64::INFINITY;
        for _ in 0..7 { let mut s=FrankenTorchSession::new(ExecutionMode::Strict);
            let x=s.tensor_variable_f32(big.clone(),vec![R,C],false).unwrap();
            let t=Instant::now(); f(&mut s,x); let e=t.elapsed().as_secs_f64()*1e3; if e<b{b=e;} }
        println!("{name}: {b:.3} ms"); };
    tim("relu6_4k", &|s,x|{let _=s.tensor_relu6(x);});
    tim("hardshrink_4k", &|s,x|{let _=s.tensor_hardshrink(x,0.5);});
    Ok(())
}

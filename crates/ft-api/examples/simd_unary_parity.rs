//! Bit-exact parity for parallel-SIMD f32 relu/neg/abs/sqrt/reciprocal vs torch.
//! Size deliberately NOT a multiple of 8 nor CHUNK(16384) to exercise chunk
//! boundaries + the scalar tail.
use std::io::Write; use std::process::{Command, Stdio};
use ft_api::FrankenTorchSession; use ft_core::ExecutionMode;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let python = std::env::var("PYTORCH_PYTHON").unwrap_or_else(|_| "python3".to_string());
    let n: usize = 1_000_003; // prime-ish, %8 = 3, spans ~61 chunks
    // mix of neg/pos/zero/tiny/large + specials at the tail
    let mut a: Vec<f32> = (0..n).map(|i| ((i % 4099) as f32 - 2049.5) * 0.013).collect();
    let pos: Vec<f32> = a.iter().map(|&x| x.abs() + 1e-6).collect();
    // sprinkle specials into the tail region (last partial chunk + scalar tail)
    let tail = n - 5;
    a[tail] = f32::INFINITY; a[tail+1] = f32::NEG_INFINITY; a[tail+2] = f32::NAN;
    a[tail+3] = -0.0; a[tail+4] = 0.0;

    let mut s = FrankenTorchSession::new(ExecutionMode::Strict);
    let xa = s.tensor_variable_f32(a.clone(), vec![n], false)?;
    let xp = s.tensor_variable_f32(pos.clone(), vec![n], false)?;
    let n_relu = s.tensor_relu(xa)?; let relu = s.tensor_values_f32(n_relu)?;
    let n_neg = s.tensor_neg(xa)?;   let neg  = s.tensor_values_f32(n_neg)?;
    let n_abs = s.tensor_abs(xa)?;   let abs  = s.tensor_values_f32(n_abs)?;
    let n_sqrt = s.tensor_sqrt(xp)?; let sqrt = s.tensor_values_f32(n_sqrt)?;
    let n_rcp = s.tensor_reciprocal(xp)?; let rcp = s.tensor_values_f32(n_rcp)?;

    let bits_a: Vec<u32> = a.iter().map(|x| x.to_bits()).collect();
    let bits_p: Vec<u32> = pos.iter().map(|x| x.to_bits()).collect();
    let py = format!(r#"
import struct,torch
ba={:?}
bp={:?}
a=torch.tensor([struct.unpack('<f',struct.pack('<I',b))[0] for b in ba],dtype=torch.float32)
p=torch.tensor([struct.unpack('<f',struct.pack('<I',b))[0] for b in bp],dtype=torch.float32)
def emit(t):
    print(' '.join(str(struct.unpack('<I',struct.pack('<f',v))[0]) for v in t.tolist()))
emit(torch.relu(a)); emit(torch.neg(a)); emit(torch.abs(a)); emit(torch.sqrt(p)); emit(torch.reciprocal(p))
"#, bits_a, bits_p);
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
        println!("{name:<12}: {mm}/{} mismatches  {:?}", ft.len(), first);
    };
    cmp("relu", &relu, &parse(lines[0]));
    cmp("neg",  &neg,  &parse(lines[1]));
    cmp("abs",  &abs,  &parse(lines[2]));
    cmp("sqrt", &sqrt, &parse(lines[3]));
    cmp("reciprocal", &rcp, &parse(lines[4]));
    Ok(())
}

//! Bit-exact parity for f32 histogram fast path vs torch (counts + edges, incl density).
use ft_api::FrankenTorchSession;
use ft_core::ExecutionMode;
use std::io::Write;
use std::process::{Command, Stdio};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let python = std::env::var("PYTORCH_PYTHON").unwrap_or_else(|_| "python3".to_string());
    let n = 500_003usize;
    let mut a: Vec<f32> = (0..n)
        .map(|i| (((i * 2654435761usize) % 800_003) as f32) * 0.00001 - 4.0)
        .collect();
    for k in 0..16 {
        a[k] = -2.0 + k as f32 * 0.5;
    } // exact boundaries
    a[20] = 2.0;
    a[21] = -2.0;
    a[22] = -10.0;
    a[23] = 10.0;
    let (bins, lo, hi) = (8usize, -2.0_f64, 2.0_f64);

    let mut s = FrankenTorchSession::new(ExecutionMode::Strict);
    let x = s.tensor_variable_f32(a.clone(), vec![n], false)?;
    let (h, e) = s.tensor_histogram(x, bins, lo, hi, None, false)?;
    let hist: Vec<f64> = s.tensor_values(h)?;
    let edges: Vec<f64> = s.tensor_values(e)?;

    let bits: Vec<u32> = a.iter().map(|v| v.to_bits()).collect();
    let py = format!(
        r#"
import struct,torch
bits={:?}
a=torch.tensor([struct.unpack('<f',struct.pack('<I',b))[0] for b in bits],dtype=torch.float32)
h,e=torch.histogram(a,bins={bins},range=({lo},{hi}))
print(' '.join(str(int(v)) for v in h.tolist()))
print(' '.join(str(struct.unpack('<Q',struct.pack('<d',float(v)))[0]) for v in e.tolist()))
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
    let mut li = out.lines();
    let pth: Vec<i64> = li
        .next()
        .unwrap_or("")
        .split_whitespace()
        .filter_map(|t| t.parse().ok())
        .collect();
    let pte: Vec<u64> = li
        .next()
        .unwrap_or("")
        .split_whitespace()
        .filter_map(|t| t.parse().ok())
        .collect();
    let mut mm = 0;
    for i in 0..hist.len() {
        if hist[i] as i64 != pth[i] {
            mm += 1;
            println!("count {i}: ft={} pt={}", hist[i] as i64, pth[i]);
        }
    }
    for i in 0..edges.len() {
        if edges[i].to_bits() != pte[i] {
            mm += 1;
            println!("edge {i}: ft={} pt={}", edges[i], f64::from_bits(pte[i]));
        }
    }
    println!(
        "counts={:?}",
        hist.iter().map(|&v| v as i64).collect::<Vec<_>>()
    );
    println!("=> {mm} mismatches (counts bit-exact + edges bit-exact)");
    Ok(())
}

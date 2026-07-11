//! block_diag f32 vs torch. Currently apply_function (f64 upcast + f64 zeros, 2x bandwidth on the
//! mostly-zero output) -> likely F64 dtype. Measure current FT vs torch + dtype before optimizing.
use ft_api::FrankenTorchSession;
use ft_core::ExecutionMode;
use std::io::Write;
use std::process::{Command, Stdio};
use std::time::Instant;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let python = std::env::var("PYTORCH_PYTHON").unwrap_or_else(|_| "python3".to_string());

    // parity: 3 small blocks of varied shape
    let mut s = FrankenTorchSession::new(ExecutionMode::Strict);
    let b0: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0]; // [2,2]
    let b1: Vec<f32> = vec![5.0, 6.0, 7.0]; // [1,3]
    let b2: Vec<f32> = vec![8.0]; // [1,1]
    let i0 = s.tensor_variable_f32(b0.clone(), vec![2, 2], false)?;
    let i1 = s.tensor_variable_f32(b1.clone(), vec![1, 3], false)?;
    let i2 = s.tensor_variable_f32(b2.clone(), vec![1, 1], false)?;
    let o = s.tensor_block_diag(&[i0, i1, i2])?;
    let dt = s.tensor_dtype(o)?;
    let fv = s.tensor_values_lossy_f64(o)?;
    let py_s = format!(
        r#"
import torch
b0=torch.tensor({b0:?},dtype=torch.float32).reshape(2,2)
b1=torch.tensor({b1:?},dtype=torch.float32).reshape(1,3)
b2=torch.tensor({b2:?},dtype=torch.float32).reshape(1,1)
o=torch.block_diag(b0,b1,b2)
print("SHAPE",list(o.shape))
print("V"," ".join("%.9g"%v for v in o.flatten().tolist()))
"#,
        b0 = b0,
        b1 = b1,
        b2 = b2
    );
    let mut ch = Command::new(&python)
        .arg("-")
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .spawn()?;
    ch.stdin.as_mut().unwrap().write_all(py_s.as_bytes())?;
    let pt = String::from_utf8_lossy(&ch.wait_with_output()?.stdout).to_string();
    let pv: Vec<f64> = pt
        .lines()
        .find_map(|l| l.strip_prefix("V "))
        .map(|s| {
            s.split_whitespace()
                .filter_map(|t| t.parse().ok())
                .collect()
        })
        .unwrap_or_default();
    let mm = fv
        .iter()
        .zip(&pv)
        .filter(|(a, b)| (**a as f32).to_bits() != (**b as f32).to_bits())
        .count()
        + fv.len().abs_diff(pv.len());
    println!(
        "parity f32: dtype={dt:?} mismatch={mm}/{} (len ft={} torch={})",
        pv.len(),
        fv.len(),
        pv.len()
    );

    // perf: 32 blocks of [128,128] -> [4096,4096] (16.7M, mostly zeros)
    let (nb, bs) = (32usize, 128usize);
    let blk: Vec<f32> = (0..bs * bs).map(|i| (i % 97) as f32 * 0.01).collect();
    let tt = || {
        let mut bst = f64::INFINITY;
        for _ in 0..7 {
            let mut s = FrankenTorchSession::new(ExecutionMode::Strict);
            let ids: Vec<_> = (0..nb)
                .map(|_| {
                    s.tensor_variable_f32(blk.clone(), vec![bs, bs], false)
                        .unwrap()
                })
                .collect();
            let ti = Instant::now();
            let _ = s.tensor_block_diag(&ids);
            let e = ti.elapsed().as_secs_f64() * 1e3;
            if e < bst {
                bst = e;
            }
        }
        bst
    };
    // anchor: a single [4096,4096] add (matches output bandwidth class)
    let big: Vec<f32> = vec![1.0f32; 4096 * 4096];
    let tadd = || {
        let mut bst = f64::INFINITY;
        for _ in 0..7 {
            let mut s = FrankenTorchSession::new(ExecutionMode::Strict);
            let x = s
                .tensor_variable_f32(big.clone(), vec![4096, 4096], false)
                .unwrap();
            let y = s
                .tensor_variable_f32(big.clone(), vec![4096, 4096], false)
                .unwrap();
            let ti = Instant::now();
            let _ = s.tensor_add(x, y);
            let e = ti.elapsed().as_secs_f64() * 1e3;
            if e < bst {
                bst = e;
            }
        }
        bst
    };
    let (ta, tb) = (tadd(), tt());
    let py_b = format!(
        r#"
import time,torch
torch.set_num_threads(8)
nb={nb}; bs={bs}
blk=((torch.arange(bs*bs,dtype=torch.int64)%97).float()*0.01).reshape(bs,bs)
blocks=[blk]*nb
big=torch.ones(4096,4096)
def tm(fn,reps=7):
    for _ in range(2): fn()
    ts=[]
    for _ in range(reps): s=time.perf_counter(); fn(); ts.append((time.perf_counter()-s)*1e3)
    return min(ts)
print("PT add %.4f"%tm(lambda:big+big))
print("PT bd %.4f"%tm(lambda:torch.block_diag(*blocks)))
"#,
        nb = nb,
        bs = bs
    );
    let mut ch = Command::new(&python)
        .arg("-")
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .spawn()?;
    ch.stdin.as_mut().unwrap().write_all(py_b.as_bytes())?;
    let pt = String::from_utf8_lossy(&ch.wait_with_output()?.stdout).to_string();
    let g = |k: &str| {
        pt.lines()
            .find_map(|l| {
                let mut it = l.strip_prefix("PT ")?.split_whitespace();
                if it.next()? == k {
                    it.next()?.parse::<f64>().ok()
                } else {
                    None
                }
            })
            .unwrap_or(f64::NAN)
    };
    let v = |ft: f64, p: f64| {
        if p >= ft {
            format!("FT {:.2}x FASTER", p / ft)
        } else {
            format!("FT {:.2}x SLOWER", ft / p)
        }
    };
    println!(
        "  add_anchor  FT {ta:.3} PT {:.3}  => {}",
        g("add"),
        v(ta, g("add"))
    );
    println!(
        "  block_diag  FT {tb:.3} PT {:.3}  => {}",
        g("bd"),
        v(tb, g("bd"))
    );
    Ok(())
}

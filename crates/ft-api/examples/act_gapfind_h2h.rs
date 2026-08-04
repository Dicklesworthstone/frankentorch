// Activation gap-finder: measure many activations vs torch (8t / FT default), find the worst ratio.
use ft_api::FrankenTorchSession;
use ft_core::ExecutionMode;
use std::io::Write;
use std::process::{Command, Stdio};
use std::time::Instant;
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let python = std::env::var("PYTORCH_PYTHON").unwrap_or_else(|_| "python3".to_string());
    let n = 16_000_000usize;
    let sp: Vec<f32> = (0..n).map(|i| ((i % 4001) as f32 / 500.0) - 4.0).collect(); // (-4,4)
    let unit: Vec<f32> = (0..n).map(|i| 0.001 + (i % 4001) as f32 / 4002.0).collect(); // (0,1)
    let ops: &[(&str, u8)] = &[
        ("add", 0),
        ("gelu", 1),
        ("silu", 2),
        ("mish", 3),
        ("softplus", 4),
        ("elu", 5),
        ("selu", 6),
        ("celu", 7),
        ("hardswish", 8),
        ("softsign", 9),
        ("tanhshrink", 10),
        ("logsigmoid", 11),
        ("logit", 12),
    ];
    let bench = |w: u8| {
        let mut best = f64::INFINITY;
        for _ in 0..7 {
            let mut s = FrankenTorchSession::new(ExecutionMode::Strict);
            let data = if w == 12 { unit.clone() } else { sp.clone() };
            let x = s.tensor_variable_f32(data, vec![n], false).unwrap();
            let t = Instant::now();
            match w {
                0 => {
                    let _ = s.tensor_add(x, x);
                }
                1 => {
                    let _ = s.tensor_gelu(x);
                }
                2 => {
                    let _ = s.tensor_silu(x);
                }
                3 => {
                    let _ = s.tensor_mish(x);
                }
                4 => {
                    let _ = s.tensor_softplus(x);
                }
                5 => {
                    let _ = s.tensor_elu(x);
                }
                6 => {
                    let _ = s.tensor_selu(x);
                }
                7 => {
                    let _ = s.tensor_celu(x, 1.0);
                }
                8 => {
                    let _ = s.tensor_hardswish(x);
                }
                9 => {
                    let _ = s.tensor_softsign(x);
                }
                10 => {
                    let _ = s.tensor_tanhshrink(x);
                }
                11 => {
                    let _ = s.tensor_logsigmoid(x);
                }
                _ => {
                    let _ = s.tensor_logit(x, None);
                }
            }
            let e = t.elapsed().as_secs_f64() * 1e3;
            if e < best {
                best = e;
            }
        }
        best
    };
    let py = format!(
        r#"
import time,torch
import torch.nn.functional as F
torch.set_num_threads(8)
n={n}
sp=((torch.arange(n,dtype=torch.int64)%4001).float()/500.0-4.0)
unit=(0.001+(torch.arange(n,dtype=torch.int64)%4001).float()/4002.0)
def tm(fn,reps=7):
    for _ in range(2): fn()
    ts=[]
    for _ in range(reps): s=time.perf_counter(); fn(); ts.append((time.perf_counter()-s)*1e3)
    return min(ts)
print("PT add %.3f"%tm(lambda:sp+sp))
print("PT gelu %.3f"%tm(lambda:F.gelu(sp)))
print("PT silu %.3f"%tm(lambda:F.silu(sp)))
print("PT mish %.3f"%tm(lambda:F.mish(sp)))
print("PT softplus %.3f"%tm(lambda:F.softplus(sp)))
print("PT elu %.3f"%tm(lambda:F.elu(sp)))
print("PT selu %.3f"%tm(lambda:F.selu(sp)))
print("PT celu %.3f"%tm(lambda:F.celu(sp)))
print("PT hardswish %.3f"%tm(lambda:F.hardswish(sp)))
print("PT softsign %.3f"%tm(lambda:F.softsign(sp)))
print("PT tanhshrink %.3f"%tm(lambda:F.tanhshrink(sp)))
print("PT logsigmoid %.3f"%tm(lambda:F.logsigmoid(sp)))
print("PT logit %.3f"%tm(lambda:torch.logit(unit)))
"#
    );
    let mut ch = Command::new(&python)
        .arg("-")
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .spawn()?;
    ch.stdin.as_mut().unwrap().write_all(py.as_bytes())?;
    let out = String::from_utf8_lossy(&ch.wait_with_output()?.stdout).to_string();
    let g = |k: &str| {
        out.lines()
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
    let vrb = |ft: f64, pp: f64| {
        if pp >= ft {
            format!("FT {:.2}x FASTER", pp / ft)
        } else {
            format!("FT {:.2}x SLOWER", ft / pp)
        }
    };
    println!("act_gapfind ~16M f32 (torch 8t / FT default), min-of-7:");
    for &(lbl, w) in ops {
        let ft = bench(w);
        println!(
            "  {lbl:<11} FT {ft:8.3}  PT {:8.3}  => {}",
            g(lbl),
            vrb(ft, g(lbl))
        );
    }
    Ok(())
}

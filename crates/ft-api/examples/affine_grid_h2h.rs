//! affine_grid f32 no-grad: serial nested loop -> parallel row fill. vs torch (matmul+transpose).
//! add = anchor. parity must match torch's grid coordinate convention.
use ft_api::FrankenTorchSession;
use ft_core::ExecutionMode;
use std::io::Write;
use std::process::{Command, Stdio};
use std::time::Instant;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let python = std::env::var("PYTORCH_PYTHON").unwrap_or_else(|_| "python3".to_string());

    // parity: theta[2,2,3] -> grid[2,1,3,4,...] use [2,3,4,5] (N=2,H=4,W=5), align_corners both
    let (nb, hh, ww) = (2usize, 4usize, 5usize);
    let th: Vec<f32> = vec![
        1.2, -0.3, 0.1, 0.4, 0.9, -0.2, 0.8, 0.0, 0.5, -0.6, 1.1, 0.3,
    ];
    for ac in [false, true] {
        let py_s = format!(
            r#"
import torch
th=torch.tensor({th:?},dtype=torch.float32).reshape({nb},2,3)
o=torch.nn.functional.affine_grid(th,[{nb},3,{hh},{ww}],align_corners={ac})
print("V"," ".join("%.9g"%v for v in o.flatten().tolist()))
"#,
            th = th,
            nb = nb,
            hh = hh,
            ww = ww,
            ac = if ac { "True" } else { "False" }
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
        let mut s = FrankenTorchSession::new(ExecutionMode::Strict);
        let t = s.tensor_variable_f32(th.clone(), vec![nb, 2, 3], false)?;
        let o = s.tensor_affine_grid(t, vec![nb, 3, hh, ww], ac)?;
        let dt = s.tensor_dtype(o)?;
        let fv = s.tensor_values_lossy_f64(o)?;
        let mr = fv
            .iter()
            .zip(&pv)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f64, f64::max);
        println!(
            "parity ac={ac}: dtype={dt:?} max_abs_err={mr:.2e} (len ft={} torch={}) tol<1e-6: {}",
            fv.len(),
            pv.len(),
            mr < 1e-6 && fv.len() == pv.len()
        );
    }

    // perf: theta[16,2,3] -> grid[16,256,256,2] = 2.1M f32
    let (nb, hh, ww) = (16usize, 256usize, 256usize);
    let thp: Vec<f32> = (0..nb * 6).map(|i| ((i % 6) as f32) * 0.1 + 0.5).collect();
    let big: Vec<f32> = vec![1.0f32; nb * hh * ww * 2];
    let tadd = || {
        let mut bst = f64::INFINITY;
        for _ in 0..9 {
            let mut s = FrankenTorchSession::new(ExecutionMode::Strict);
            let x = s
                .tensor_variable_f32(big.clone(), vec![nb * hh * ww * 2], false)
                .unwrap();
            let y = s
                .tensor_variable_f32(big.clone(), vec![nb * hh * ww * 2], false)
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
    let tag = || {
        let mut bst = f64::INFINITY;
        for _ in 0..9 {
            let mut s = FrankenTorchSession::new(ExecutionMode::Strict);
            let t = s
                .tensor_variable_f32(thp.clone(), vec![nb, 2, 3], false)
                .unwrap();
            let ti = Instant::now();
            let _ = s.tensor_affine_grid(t, vec![nb, 3, hh, ww], false);
            let e = ti.elapsed().as_secs_f64() * 1e3;
            if e < bst {
                bst = e;
            }
        }
        bst
    };
    let (ta, tg) = (tadd(), tag());
    let py_b = format!(
        r#"
import time,torch
torch.set_num_threads(8)
th=(((torch.arange(16*6,dtype=torch.int64)%6).float())*0.1+0.5).reshape(16,2,3)
big=torch.ones(16*256*256*2)
def tm(fn,reps=9):
    for _ in range(2): fn()
    ts=[]
    for _ in range(reps): s=time.perf_counter(); fn(); ts.append((time.perf_counter()-s)*1e3)
    return min(ts)
print("PT add %.4f"%tm(lambda:big+big))
print("PT ag %.4f"%tm(lambda:torch.nn.functional.affine_grid(th,[16,3,256,256],align_corners=False)))
"#
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
        "  add_anchor   FT {ta:.3} PT {:.3}  => {}",
        g("add"),
        v(ta, g("add"))
    );
    println!(
        "  affine_grid  FT {tg:.3} PT {:.3}  => {}",
        g("ag"),
        v(tg, g("ag"))
    );
    Ok(())
}

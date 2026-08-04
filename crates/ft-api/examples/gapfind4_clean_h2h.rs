// Clean gapfind (inputs OUTSIDE timer) for elementwise composite ops f32 vs torch. cc.
use ft_api::FrankenTorchSession;
use ft_core::ExecutionMode;
use std::io::Write;
use std::process::{Command, Stdio};
use std::time::Instant;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let python = std::env::var("PYTORCH_PYTHON").unwrap_or_else(|_| "python3".to_string());
    let n = 4096 * 4096;
    let va: Vec<f32> = (0..n)
        .map(|i| ((i % 9973) as f32 - 5000.0) * 0.001)
        .collect();
    let vb: Vec<f32> = (0..n)
        .map(|i| ((i % 7919) as f32 - 4000.0) * 0.001 + 0.01)
        .collect();
    let vc: Vec<f32> = (0..n)
        .map(|i| ((i % 6151) as f32 - 3000.0) * 0.001 + 1.5)
        .collect();
    macro_rules! bench {
        ($setup:expr, $op:expr) => {{
            let mut best = f64::INFINITY;
            for _ in 0..5 {
                let mut s = FrankenTorchSession::new(ExecutionMode::Strict);
                let inp = $setup(&mut s);
                let t = Instant::now();
                let _ = $op(&mut s, inp);
                best = best.min(t.elapsed().as_secs_f64() * 1e3);
            }
            best
        }};
    }
    let mk = |s: &mut FrankenTorchSession, v: &Vec<f32>| {
        s.tensor_variable_f32(v.clone(), vec![4096, 4096], false)
            .unwrap()
    };

    let t_lerp = bench!(
        |s: &mut FrankenTorchSession| (mk(s, &va), mk(s, &vb)),
        |s: &mut FrankenTorchSession, (a, b)| s.tensor_lerp(a, b, 0.3)
    );
    let t_addcmul = bench!(
        |s: &mut FrankenTorchSession| (mk(s, &va), mk(s, &vb), mk(s, &vc)),
        |s: &mut FrankenTorchSession, (t, a, b)| s.tensor_addcmul(t, a, b, 0.5)
    );
    let t_addcdiv = bench!(
        |s: &mut FrankenTorchSession| (mk(s, &va), mk(s, &vb), mk(s, &vc)),
        |s: &mut FrankenTorchSession, (t, a, b)| s.tensor_addcdiv(t, a, b, 0.5)
    );
    let t_hypot = bench!(
        |s: &mut FrankenTorchSession| (mk(s, &va), mk(s, &vb)),
        |s: &mut FrankenTorchSession, (a, b)| s.tensor_hypot(a, b)
    );
    let t_xlogy = bench!(
        |s: &mut FrankenTorchSession| (mk(s, &va), mk(s, &vc)),
        |s: &mut FrankenTorchSession, (a, b)| s.tensor_xlogy(a, b)
    );

    let py = r#"
import time,torch
torch.set_num_threads(8)
def tm(fn,reps=5):
    for _ in range(2): fn()
    ts=[]
    for _ in range(reps): s=time.perf_counter(); fn(); ts.append((time.perf_counter()-s)*1e3)
    return min(ts)
n=4096*4096
a=(((torch.arange(n,dtype=torch.int64)%9973).float()-5000.0)*0.001).reshape(4096,4096)
b=(((torch.arange(n,dtype=torch.int64)%7919).float()-4000.0)*0.001+0.01).reshape(4096,4096)
c=(((torch.arange(n,dtype=torch.int64)%6151).float()-3000.0)*0.001+1.5).reshape(4096,4096)
print("PT lerp %.4f"%tm(lambda:torch.lerp(a,b,0.3)))
print("PT addcmul %.4f"%tm(lambda:torch.addcmul(a,b,c,value=0.5)))
print("PT addcdiv %.4f"%tm(lambda:torch.addcdiv(a,b,c,value=0.5)))
print("PT hypot %.4f"%tm(lambda:torch.hypot(a,b)))
print("PT xlogy %.4f"%tm(lambda:torch.xlogy(a,c)))
"#;
    let mut ch = Command::new(&python)
        .arg("-")
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .spawn()?;
    ch.stdin.as_mut().unwrap().write_all(py.as_bytes())?;
    let out = String::from_utf8_lossy(&ch.wait_with_output()?.stdout).to_string();
    let pt = |name: &str| -> f64 {
        out.lines()
            .find_map(|l| {
                let mut it = l.strip_prefix("PT ")?.split_whitespace();
                if it.next()? == name {
                    it.next()?.parse::<f64>().ok()
                } else {
                    None
                }
            })
            .unwrap_or(f64::NAN)
    };
    for (name, ft) in [
        ("lerp", t_lerp),
        ("addcmul", t_addcmul),
        ("addcdiv", t_addcdiv),
        ("hypot", t_hypot),
        ("xlogy", t_xlogy),
    ] {
        let p = pt(name);
        let vrb = if p >= ft {
            format!("FT {:.2}x FASTER", p / ft)
        } else {
            format!("FT {:.2}x SLOWER", ft / p)
        };
        println!("{name:<10} FT {ft:9.4}ms torch {p:9.4}ms => {vrb}");
    }
    Ok(())
}

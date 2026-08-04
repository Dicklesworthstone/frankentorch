// Clean gapfind (inputs OUTSIDE timer) for movement/reduction ops f32 vs torch. cc.
use ft_api::FrankenTorchSession;
use ft_core::ExecutionMode;
use std::io::Write;
use std::process::{Command, Stdio};
use std::time::Instant;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let python = std::env::var("PYTORCH_PYTHON").unwrap_or_else(|_| "python3".to_string());
    let v = |n: usize| -> Vec<f32> {
        (0..n)
            .map(|i| ((i % 9973) as f32 - 5000.0) * 0.001)
            .collect()
    };

    // kron [128,128] x [16,16] -> [2048,2048]
    let ka = v(128 * 128);
    let kb = v(16 * 16);
    let t_kron = {
        let mut best = f64::INFINITY;
        for _ in 0..5 {
            let mut s = FrankenTorchSession::new(ExecutionMode::Strict);
            let a = s
                .tensor_variable_f32(ka.clone(), vec![128, 128], false)
                .unwrap();
            let b = s
                .tensor_variable_f32(kb.clone(), vec![16, 16], false)
                .unwrap();
            let t = Instant::now();
            let _ = s.tensor_kron(a, b);
            best = best.min(t.elapsed().as_secs_f64() * 1e3);
        }
        best
    };
    // vdot [4M]
    let va = v(4_000_000);
    let t_vdot = {
        let mut best = f64::INFINITY;
        for _ in 0..5 {
            let mut s = FrankenTorchSession::new(ExecutionMode::Strict);
            let a = s
                .tensor_variable_f32(va.clone(), vec![4_000_000], false)
                .unwrap();
            let b = s
                .tensor_variable_f32(va.clone(), vec![4_000_000], false)
                .unwrap();
            let t = Instant::now();
            let _ = s.tensor_vdot(a, b);
            best = best.min(t.elapsed().as_secs_f64() * 1e3);
        }
        best
    };
    // roll_dims [2048,2048] shifts[3,5] dims[0,1]
    let ri = v(2048 * 2048);
    let t_roll = {
        let mut best = f64::INFINITY;
        for _ in 0..5 {
            let mut s = FrankenTorchSession::new(ExecutionMode::Strict);
            let x = s
                .tensor_variable_f32(ri.clone(), vec![2048, 2048], false)
                .unwrap();
            let t = Instant::now();
            let _ = s.tensor_roll_dims(x, &[3, 5], &[0, 1]);
            best = best.min(t.elapsed().as_secs_f64() * 1e3);
        }
        best
    };
    // fliplr [4096,4096]
    let fi = v(4096 * 4096);
    let t_flr = {
        let mut best = f64::INFINITY;
        for _ in 0..5 {
            let mut s = FrankenTorchSession::new(ExecutionMode::Strict);
            let x = s
                .tensor_variable_f32(fi.clone(), vec![4096, 4096], false)
                .unwrap();
            let t = Instant::now();
            let _ = s.tensor_fliplr(x);
            best = best.min(t.elapsed().as_secs_f64() * 1e3);
        }
        best
    };
    // flipud [4096,4096]
    let t_fud = {
        let mut best = f64::INFINITY;
        for _ in 0..5 {
            let mut s = FrankenTorchSession::new(ExecutionMode::Strict);
            let x = s
                .tensor_variable_f32(fi.clone(), vec![4096, 4096], false)
                .unwrap();
            let t = Instant::now();
            let _ = s.tensor_flipud(x);
            best = best.min(t.elapsed().as_secs_f64() * 1e3);
        }
        best
    };

    let py = r#"
import time,torch
torch.set_num_threads(8)
def tm(fn,reps=5):
    for _ in range(2): fn()
    ts=[]
    for _ in range(reps): s=time.perf_counter(); fn(); ts.append((time.perf_counter()-s)*1e3)
    return min(ts)
def mk(n,shape): return (((torch.arange(n,dtype=torch.int64)%9973).float()-5000.0)*0.001).reshape(shape)
ka=mk(128*128,(128,128)); kb=mk(16*16,(16,16))
print("PT kron %.4f"%tm(lambda:torch.kron(ka,kb)))
va=mk(4000000,(4000000,))
print("PT vdot %.4f"%tm(lambda:torch.vdot(va,va)))
ri=mk(2048*2048,(2048,2048))
print("PT roll %.4f"%tm(lambda:torch.roll(ri,[3,5],[0,1])))
fi=mk(4096*4096,(4096,4096))
print("PT fliplr %.4f"%tm(lambda:torch.fliplr(fi)))
print("PT flipud %.4f"%tm(lambda:torch.flipud(fi)))
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
        ("kron", t_kron),
        ("vdot", t_vdot),
        ("roll", t_roll),
        ("fliplr", t_flr),
        ("flipud", t_fud),
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

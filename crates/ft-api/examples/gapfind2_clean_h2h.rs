// Clean gapfind (inputs OUTSIDE timed region) for fresh compose/movement ops f32 vs torch. cc.
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

    // rot90 [2048,2048] k=1 dims[0,1]
    let rot_in = v(2048 * 2048);
    let t_rot = {
        let mut best = f64::INFINITY;
        for _ in 0..5 {
            let mut s = FrankenTorchSession::new(ExecutionMode::Strict);
            let x = s
                .tensor_variable_f32(rot_in.clone(), vec![2048, 2048], false)
                .unwrap();
            let t = Instant::now();
            let _ = s.tensor_rot90(x, 1, [0, 1]);
            best = best.min(t.elapsed().as_secs_f64() * 1e3);
        }
        best
    };
    // diag_embed [512,128] -> [512,128,128]
    let de_in = v(512 * 128);
    let t_de = {
        let mut best = f64::INFINITY;
        for _ in 0..5 {
            let mut s = FrankenTorchSession::new(ExecutionMode::Strict);
            let x = s
                .tensor_variable_f32(de_in.clone(), vec![512, 128], false)
                .unwrap();
            let t = Instant::now();
            let _ = s.tensor_diag_embed(x, 0);
            best = best.min(t.elapsed().as_secs_f64() * 1e3);
        }
        best
    };
    // column_stack 4 x [500000] -> [500000,4]
    let cs_in = v(500_000);
    let t_cs = {
        let mut best = f64::INFINITY;
        for _ in 0..5 {
            let mut s = FrankenTorchSession::new(ExecutionMode::Strict);
            let cols: Vec<_> = (0..4)
                .map(|_| {
                    s.tensor_variable_f32(cs_in.clone(), vec![500_000], false)
                        .unwrap()
                })
                .collect();
            let t = Instant::now();
            let _ = s.tensor_column_stack(&cols);
            best = best.min(t.elapsed().as_secs_f64() * 1e3);
        }
        best
    };
    // dstack 4 x [512,512] -> [512,512,4]
    let ds_in = v(512 * 512);
    let t_ds = {
        let mut best = f64::INFINITY;
        for _ in 0..5 {
            let mut s = FrankenTorchSession::new(ExecutionMode::Strict);
            let parts: Vec<_> = (0..4)
                .map(|_| {
                    s.tensor_variable_f32(ds_in.clone(), vec![512, 512], false)
                        .unwrap()
                })
                .collect();
            let t = Instant::now();
            let _ = s.tensor_dstack(&parts);
            best = best.min(t.elapsed().as_secs_f64() * 1e3);
        }
        best
    };
    // tensordot [256,256]x[256,256] dims=1
    let td_in = v(256 * 256);
    let t_td = {
        let mut best = f64::INFINITY;
        for _ in 0..5 {
            let mut s = FrankenTorchSession::new(ExecutionMode::Strict);
            let a = s
                .tensor_variable_f32(td_in.clone(), vec![256, 256], false)
                .unwrap();
            let b = s
                .tensor_variable_f32(td_in.clone(), vec![256, 256], false)
                .unwrap();
            let t = Instant::now();
            let _ = s.tensor_tensordot(a, b, 1);
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
xr=mk(2048*2048,(2048,2048))
print("PT rot90 %.4f"%tm(lambda:torch.rot90(xr,1,[0,1])))
xd=mk(512*128,(512,128))
print("PT diag_embed %.4f"%tm(lambda:torch.diag_embed(xd)))
c=mk(500000,(500000,))
print("PT column_stack %.4f"%tm(lambda:torch.column_stack([c,c,c,c])))
d=mk(512*512,(512,512))
print("PT dstack %.4f"%tm(lambda:torch.dstack([d,d,d,d])))
ta=mk(256*256,(256,256))
print("PT tensordot %.4f"%tm(lambda:torch.tensordot(ta,ta,dims=1)))
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
        ("rot90", t_rot),
        ("diag_embed", t_de),
        ("column_stack", t_cs),
        ("dstack", t_ds),
        ("tensordot", t_td),
    ] {
        let p = pt(name);
        let vrb = if p >= ft {
            format!("FT {:.2}x FASTER", p / ft)
        } else {
            format!("FT {:.2}x SLOWER", ft / p)
        };
        println!("{name:<14} FT {ft:9.4}ms torch {p:9.4}ms => {vrb}");
    }
    Ok(())
}

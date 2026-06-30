// Gap-finder: batch of diverse ops f32 vs torch (no-grad). Finds biggest SLOWER ratio. cc.
use ft_api::FrankenTorchSession;
use ft_core::ExecutionMode;
use std::io::Write;
use std::process::{Command, Stdio};
use std::time::Instant;

fn bench<F: FnMut(&mut FrankenTorchSession)>(mut f: F) -> f64 {
    let mut best = f64::INFINITY;
    for _ in 0..5 {
        let mut s = FrankenTorchSession::new(ExecutionMode::Strict);
        let t = Instant::now();
        f(&mut s);
        let e = t.elapsed().as_secs_f64() * 1e3;
        if e < best { best = e; }
    }
    best
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let python = std::env::var("PYTORCH_PYTHON").unwrap_or_else(|_| "python3".to_string());
    let mkf32 = |s: &mut FrankenTorchSession, n: usize, shape: Vec<usize>| {
        let v: Vec<f32> = (0..n).map(|i| ((i % 9973) as f32 - 5000.0) * 0.001).collect();
        s.tensor_variable_f32(v, shape, false).unwrap()
    };

    // vander: [8192] -> N=48 powers
    let t_vander = bench(|s| {
        let x = mkf32(s, 8192, vec![8192]);
        let _ = s.tensor_vander(x, Some(48), false);
    });
    // histc: [4M] bins=256
    let t_histc = bench(|s| {
        let x = mkf32(s, 4_000_000, vec![4_000_000]);
        let _ = s.tensor_histc(x, 256, -6.0, 6.0);
    });
    // meshgrid: two [2048] -> two [2048,2048]
    let t_meshgrid = bench(|s| {
        let a = mkf32(s, 2048, vec![2048]);
        let b = mkf32(s, 2048, vec![2048]);
        let _ = s.tensor_meshgrid(&[a, b]);
    });
    // cross: [1M,3] x [1M,3]
    let t_cross = bench(|s| {
        let a = mkf32(s, 3_000_000, vec![1_000_000, 3]);
        let b = mkf32(s, 3_000_000, vec![1_000_000, 3]);
        let _ = s.tensor_cross(a, b);
    });
    // outer: [4096] x [4096] -> [4096,4096]
    let t_outer = bench(|s| {
        let a = mkf32(s, 4096, vec![4096]);
        let b = mkf32(s, 4096, vec![4096]);
        let _ = s.tensor_outer(a, b);
    });
    // cumprod: [2048,2048] dim=1
    let t_cumprod = bench(|s| {
        let x = mkf32(s, 2048 * 2048, vec![2048, 2048]);
        let _ = s.tensor_cumprod(x, 1);
    });
    // count_nonzero: [4M]
    let t_cnz = bench(|s| {
        let x = mkf32(s, 4_000_000, vec![4_000_000]);
        let _ = s.tensor_count_nonzero(x);
    });

    let py = r#"
import time,torch
torch.set_num_threads(8)
def tm(fn,reps=5):
    for _ in range(2): fn()
    ts=[]
    for _ in range(reps): s=time.perf_counter(); fn(); ts.append((time.perf_counter()-s)*1e3)
    return min(ts)
def mk(n,shape):
    return (((torch.arange(n,dtype=torch.int64)%9973).float()-5000.0)*0.001).reshape(shape)
xv=mk(8192,(8192,))
print("PT vander %.3f"%tm(lambda:torch.vander(xv,48)))
xh=mk(4000000,(4000000,))
print("PT histc %.3f"%tm(lambda:torch.histc(xh,256,-6.0,6.0)))
ma=mk(2048,(2048,)); mb=mk(2048,(2048,))
print("PT meshgrid %.3f"%tm(lambda:torch.meshgrid(ma,mb,indexing='ij')))
ca=mk(3000000,(1000000,3)); cb=mk(3000000,(1000000,3))
print("PT cross %.3f"%tm(lambda:torch.linalg.cross(ca,cb,dim=-1)))
oa=mk(4096,(4096,)); ob=mk(4096,(4096,))
print("PT outer %.3f"%tm(lambda:torch.outer(oa,ob)))
xc=mk(2048*2048,(2048,2048))
print("PT cumprod %.3f"%tm(lambda:torch.cumprod(xc,1)))
xn=mk(4000000,(4000000,))
print("PT count_nonzero %.3f"%tm(lambda:torch.count_nonzero(xn)))
"#;
    let mut ch = Command::new(&python).arg("-").stdin(Stdio::piped()).stdout(Stdio::piped()).spawn()?;
    ch.stdin.as_mut().unwrap().write_all(py.as_bytes())?;
    let out = String::from_utf8_lossy(&ch.wait_with_output()?.stdout).to_string();
    let pt = |name: &str| -> f64 {
        out.lines().find_map(|l| {
            let mut it = l.strip_prefix("PT ")?.split_whitespace();
            if it.next()? == name { it.next()?.parse::<f64>().ok() } else { None }
        }).unwrap_or(f64::NAN)
    };
    let report = |name: &str, ft: f64| {
        let p = pt(name);
        let vrb = if p >= ft { format!("FT {:.2}x FASTER", p / ft) } else { format!("FT {:.2}x SLOWER", ft / p) };
        println!("{name:<16} FT {ft:9.3}ms torch {p:9.3}ms => {vrb}");
    };
    report("vander", t_vander);
    report("histc", t_histc);
    report("meshgrid", t_meshgrid);
    report("cross", t_cross);
    report("outer", t_outer);
    report("cumprod", t_cumprod);
    report("count_nonzero", t_cnz);
    Ok(())
}

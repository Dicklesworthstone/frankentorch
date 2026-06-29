// Loss-fn gap-finder vs torch (8t / FT default). Inputs built OUTSIDE the timed region.
use ft_api::FrankenTorchSession;
use ft_core::ExecutionMode;
use std::io::Write;
use std::process::{Command, Stdio};
use std::time::Instant;
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let python = std::env::var("PYTORCH_PYTHON").unwrap_or_else(|_| "python3".to_string());
    let n = 16_000_000usize;
    let inp: Vec<f32> = (0..n).map(|i| ((i % 4001) as f32 / 500.0) - 4.0).collect(); // (-4,4)
    let tgt01: Vec<f32> = (0..n).map(|i| (i % 1000) as f32 / 1000.0).collect(); // (0,1)
    let tgtpos: Vec<f32> = (0..n).map(|i| (i % 50) as f32).collect(); // counts >=0
    let varv: Vec<f32> = (0..n).map(|i| 0.1 + (i % 997) as f32 / 200.0).collect(); // var >0
    let bench = |w: u8| {
        let mut best = f64::INFINITY;
        for _ in 0..7 {
            let mut s = FrankenTorchSession::new(ExecutionMode::Strict);
            let x = s.tensor_variable_f32(inp.clone(), vec![n], false).unwrap();
            let t = s.tensor_variable_f32(if w == 2 { tgtpos.clone() } else { tgt01.clone() }, vec![n], false).unwrap();
            let ti = Instant::now();
            match w {
                0 => { let _ = s.tensor_add(x, t); }
                1 => { let _ = s.tensor_bce_with_logits_loss(x, t, "mean"); }
                2 => { let _ = s.tensor_poisson_nll_loss(x, t, true, false, 1e-8, "mean"); }
                3 => { let _ = s.tensor_bce_with_logits_pos_weight(x, t, 2.5, "mean"); }
                _ => { let v = s.tensor_variable_f32(varv.clone(), vec![n], false).unwrap(); let _ = s.tensor_gaussian_nll_loss(x, t, v, "mean", false); }
            }
            let e = ti.elapsed().as_secs_f64() * 1e3;
            if e < best { best = e; }
        }
        best
    };
    let py = format!(r#"
import time,torch
import torch.nn.functional as F
torch.set_num_threads(8)
n={n}
inp=((torch.arange(n,dtype=torch.int64)%4001).float()/500.0-4.0)
t01=((torch.arange(n,dtype=torch.int64)%1000).float()/1000.0)
tpos=((torch.arange(n,dtype=torch.int64)%50).float())
def tm(fn,reps=7):
    for _ in range(2): fn()
    ts=[]
    for _ in range(reps): s=time.perf_counter(); fn(); ts.append((time.perf_counter()-s)*1e3)
    return min(ts)
def safe(name,fn):
    try: print("PT %s %.3f"%(name,tm(fn)))
    except Exception as e: print("PT %s NaN  # %s"%(name,type(e).__name__))
safe("add", lambda:inp+t01)
safe("bce_logits", lambda:F.binary_cross_entropy_with_logits(inp,t01,reduction='mean'))
safe("poisson", lambda:F.poisson_nll_loss(inp,tpos,log_input=True,full=False,eps=1e-8,reduction='mean'))
safe("bce_pw", lambda:F.binary_cross_entropy_with_logits(inp,t01,pos_weight=torch.tensor(2.5),reduction='mean'))
varv=(0.1+(torch.arange(n,dtype=torch.int64)%997).float()/200.0)
safe("gauss_nll", lambda:F.gaussian_nll_loss(inp,t01,varv,full=False,reduction='mean'))
"#);
    let mut ch = Command::new(&python).arg("-").stdin(Stdio::piped()).stdout(Stdio::piped()).spawn()?;
    ch.stdin.as_mut().unwrap().write_all(py.as_bytes())?;
    let out = String::from_utf8_lossy(&ch.wait_with_output()?.stdout).to_string();
    let g = |k: &str| out.lines().find_map(|l| { let mut it = l.strip_prefix("PT ")?.split_whitespace(); if it.next()? == k { it.next()?.parse::<f64>().ok() } else { None } }).unwrap_or(f64::NAN);
    let vrb = |ft: f64, pp: f64| if pp >= ft { format!("FT {:.2}x FASTER", pp / ft) } else { format!("FT {:.2}x SLOWER", ft / pp) };
    println!("loss_gapfind ~16M f32 (torch 8t / FT default), min-of-7:");
    for (lbl, w, key) in [("add", 0u8, "add"), ("bce_logits", 1, "bce_logits"), ("poisson", 2, "poisson"), ("bce_pw", 3, "bce_pw"), ("gauss_nll", 4, "gauss_nll")] {
        let ft = bench(w);
        println!("  {lbl:<11} FT {ft:8.3}  PT {:8.3}  => {}", g(key), vrb(ft, g(key)));
    }
    Ok(())
}

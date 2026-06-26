//! More loss-fn scan FT vs PyTorch ([4000,4000] f64 no-grad, reduction='none'). `cat` ANCHOR.
//! Run: PYTORCH_PYTHON=/path/to/python cargo run --release -p ft-api --example loss2_h2h

use std::io::Write;
use std::process::{Command, Stdio};
use std::time::Instant;

use ft_api::FrankenTorchSession;
use ft_core::ExecutionMode;

const R: usize = 4000;
const C: usize = 4000;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let a: Vec<f64> = (0..R * C).map(|i| 0.1 + ((i % 17) as f64) * 0.04).collect();
    let b: Vec<f64> = (0..R * C).map(|i| 0.1 + ((i % 13) as f64) * 0.05).collect();
    let v: Vec<f64> = (0..R * C).map(|i| 0.5 + ((i % 7) as f64) * 0.2).collect();
    let tgt: Vec<f64> = (0..R * C).map(|i| if i % 2 == 0 { 1.0 } else { -1.0 }).collect();
    let anc: Vec<f64> = (0..R * C).map(|i| (i % 7) as f64).collect();

    let bench = |mut f: Box<dyn FnMut() -> ()>| -> f64 {
        let mut best = f64::INFINITY;
        for _ in 0..7 { let t = Instant::now(); f(); let e = t.elapsed().as_secs_f64()*1e3; if e<best {best=e;} }
        best
    };

    let gnll = bench({ let a=a.clone(); let b=b.clone(); let v=v.clone(); Box::new(move || {
        let mut s = FrankenTorchSession::new(ExecutionMode::Strict);
        let x=s.tensor_variable(a.clone(),vec![R,C],false).unwrap();
        let y=s.tensor_variable(b.clone(),vec![R,C],false).unwrap();
        let vr=s.tensor_variable(v.clone(),vec![R,C],false).unwrap();
        let _=s.tensor_gaussian_nll_loss(x,y,vr,"none",false);
    })});
    let kld = bench({ let a=a.clone(); let b=b.clone(); Box::new(move || {
        let mut s = FrankenTorchSession::new(ExecutionMode::Strict);
        let x=s.tensor_variable(a.clone(),vec![R,C],false).unwrap();
        let y=s.tensor_variable(b.clone(),vec![R,C],false).unwrap();
        let _=s.tensor_kl_div(x,y,"none",false);
    })});
    let hinge = bench({ let a=a.clone(); let tgt=tgt.clone(); Box::new(move || {
        let mut s = FrankenTorchSession::new(ExecutionMode::Strict);
        let x=s.tensor_variable(a.clone(),vec![R,C],false).unwrap();
        let y=s.tensor_variable(tgt.clone(),vec![R,C],false).unwrap();
        let _=s.tensor_hinge_embedding_loss(x,y,1.0,"none");
    })});
    let anchor = bench({ let anc=anc.clone(); Box::new(move || {
        let mut s = FrankenTorchSession::new(ExecutionMode::Strict);
        let x=s.tensor_variable(anc.clone(),vec![R,C],false).unwrap();
        let _=s.tensor_cat(&[x,x],1);
    })});

    let python = std::env::var("PYTORCH_PYTHON").unwrap_or_else(|_| "python3".to_string());
    let py = r#"
import time, torch
import torch.nn.functional as F
torch.set_num_threads(8)
R,C=4000,4000
idx=torch.arange(R*C,dtype=torch.int64)
x=(0.1+(idx%17).double()*0.04).reshape(R,C)
y=(0.1+(idx%13).double()*0.05).reshape(R,C)
v=(0.5+(idx%7).double()*0.2).reshape(R,C)
tgt=torch.where(idx%2==0, torch.tensor(1.0,dtype=torch.float64), torch.tensor(-1.0,dtype=torch.float64)).reshape(R,C)
anc=(idx%7).double().reshape(R,C)
def t(fn,n=7):
    for _ in range(2):
        try: fn()
        except Exception as e: return float('nan')
    ts=[]
    for _ in range(n):
        s=time.perf_counter(); fn(); ts.append((time.perf_counter()-s)*1e3)
    return min(ts)
print("PT gaussian_nll %.4f"%t(lambda:F.gaussian_nll_loss(x,y,v,reduction='none',full=False)))
print("PT kl_div %.4f"%t(lambda:F.kl_div(x,y,reduction='none',log_target=False)))
print("PT hinge %.4f"%t(lambda:F.hinge_embedding_loss(x,tgt,margin=1.0,reduction='none')))
print("PT cat_anchor %.4f"%t(lambda:torch.cat([anc,anc],1)))
"#;
    let mut child = Command::new(&python).arg("-").stdin(Stdio::piped()).stdout(Stdio::piped()).spawn()?;
    child.stdin.as_mut().ok_or_else(|| std::io::Error::other("no stdin"))?.write_all(py.as_bytes())?;
    let out = child.wait_with_output();
    let pt = out.ok().filter(|o| o.status.success()).map(|o| String::from_utf8_lossy(&o.stdout).to_string()).unwrap_or_default();
    let lk = |name: &str| -> Option<f64> { pt.lines().find_map(|l| { let mut it=l.strip_prefix("PT ")?.split_whitespace(); if it.next()?==name {it.next()?.parse().ok()} else {None} }) };
    println!("op            FT(ms)    PT(ms)   ratio(PT/FT, <1=FT slower)");
    for (name, ftv) in [("gaussian_nll", gnll), ("kl_div", kld), ("hinge", hinge), ("cat_anchor", anchor)] {
        if let Some(p) = lk(name) {
            let r = p/ftv;
            let tag = if r>=1.0 {format!("FT {r:.2}x FASTER")} else {format!("FT {:.2}x SLOWER",1.0/r)};
            println!("  {name:<14} {ftv:8.3} {p:8.3}   {tag}");
        }
    }
    Ok(())
}

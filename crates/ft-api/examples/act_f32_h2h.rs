//! F32 activation survey: find activations whose f32 path is slow vs torch.
//! Run: PYTORCH_PYTHON=/path/to/python cargo run --release -p ft-api --example act_f32_h2h
use std::io::Write; use std::process::{Command, Stdio}; use std::time::Instant;
use ft_api::FrankenTorchSession; use ft_autograd::TensorNodeId; use ft_core::ExecutionMode;
const R: usize = 4000; const C: usize = 4000;

fn t1<F: Fn(&mut FrankenTorchSession, TensorNodeId)>(a: &[f32], f: F) -> f64 {
    let mut b = f64::INFINITY;
    for _ in 0..7 { let mut s = FrankenTorchSession::new(ExecutionMode::Strict);
        let x = s.tensor_variable_f32(a.to_vec(), vec![R, C], false).unwrap();
        let t = Instant::now(); f(&mut s, x); let e = t.elapsed().as_secs_f64()*1e3; if e<b {b=e;} }
    b
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let a: Vec<f32> = (0..R*C).map(|i| ((i%2000) as f32 - 1000.0) * 0.01).collect();
    let python = std::env::var("PYTORCH_PYTHON").unwrap_or_else(|_| "python3".to_string());
    let py = r#"
import time,torch
torch.set_num_threads(8)
import torch.nn.functional as F
R,C=4000,4000
a=(((torch.arange(R*C,dtype=torch.int64)%2000).float()-1000.0)*0.01).reshape(R,C)
def t(fn,n=7):
    for _ in range(2): fn()
    ts=[]
    for _ in range(n): s=time.perf_counter(); fn(); ts.append((time.perf_counter()-s)*1e3)
    return min(ts)
for name,fn in [("relu_anchor",lambda:torch.relu(a)),
                ("hardtanh",lambda:F.hardtanh(a)),
                ("hardsigmoid",lambda:F.hardsigmoid(a)),
                ("hardswish",lambda:F.hardswish(a)),
                ("relu6",lambda:F.relu6(a)),
                ("leaky_relu",lambda:F.leaky_relu(a,0.01)),
                ("elu",lambda:F.elu(a)),
                ("selu",lambda:F.selu(a)),
                ("celu",lambda:F.celu(a)),
                ("softplus",lambda:F.softplus(a)),
                ("mish",lambda:F.mish(a)),
                ("silu",lambda:F.silu(a)),
                ("tanhshrink",lambda:F.tanhshrink(a)),
                ("hardshrink",lambda:F.hardshrink(a))]:
    print("PT %s %.4f"%(name,t(fn)))
"#;
    let mut ch=Command::new(&python).arg("-").stdin(Stdio::piped()).stdout(Stdio::piped()).spawn()?;
    ch.stdin.as_mut().unwrap().write_all(py.as_bytes())?;
    let o=ch.wait_with_output(); let pt=String::from_utf8_lossy(&o.unwrap().stdout).to_string();
    let rep=|n:&str,ft:f64|{ if let Some(p)=pt.lines().find_map(|l|{let mut it=l.strip_prefix("PT ")?.split_whitespace(); if it.next()?==n {it.next()?.parse::<f64>().ok()} else {None}}){let r=p/ft; let tag=if r>=1.0{format!("FT {r:.2}x FASTER")}else{format!("FT {:.2}x SLOWER",1.0/r)}; println!("  {n:<12} {ft:8.3} {p:8.3}   {tag}");}};
    println!("op            FT(ms)    PT(ms)   verdict");
    rep("relu_anchor", t1(&a, |s,x| { let _=s.tensor_relu(x); }));
    rep("hardtanh", t1(&a, |s,x| { let _=s.tensor_hardtanh(x); }));
    rep("hardsigmoid", t1(&a, |s,x| { let _=s.tensor_hardsigmoid(x); }));
    rep("hardswish", t1(&a, |s,x| { let _=s.tensor_hardswish(x); }));
    rep("relu6", t1(&a, |s,x| { let _=s.tensor_relu6(x); }));
    rep("leaky_relu", t1(&a, |s,x| { let _=s.tensor_leaky_relu(x); }));
    rep("elu", t1(&a, |s,x| { let _=s.tensor_elu(x); }));
    rep("selu", t1(&a, |s,x| { let _=s.tensor_selu(x); }));
    rep("celu", t1(&a, |s,x| { let _=s.tensor_celu(x,1.0); }));
    rep("softplus", t1(&a, |s,x| { let _=s.tensor_softplus(x); }));
    rep("mish", t1(&a, |s,x| { let _=s.tensor_mish(x); }));
    rep("silu", t1(&a, |s,x| { let _=s.tensor_silu(x); }));
    rep("tanhshrink", t1(&a, |s,x| { let _=s.tensor_tanhshrink(x); }));
    rep("hardshrink", t1(&a, |s,x| { let _=s.tensor_hardshrink(x,0.5); }));
    Ok(())
}

//! F32 SIMD-unary survey: relu/neg/abs/sqrt/reciprocal — serial-SIMD vs torch.
use std::io::Write; use std::process::{Command, Stdio}; use std::time::Instant;
use ft_api::FrankenTorchSession; use ft_autograd::TensorNodeId; use ft_core::ExecutionMode;
const R: usize = 4000; const C: usize = 4000;

fn t1<F: Fn(&mut FrankenTorchSession, TensorNodeId)>(a: &[f32], f: F) -> f64 {
    let mut b = f64::INFINITY;
    for _ in 0..9 { let mut s = FrankenTorchSession::new(ExecutionMode::Strict);
        let x = s.tensor_variable_f32(a.to_vec(), vec![R, C], false).unwrap();
        let t = Instant::now(); f(&mut s, x); let e = t.elapsed().as_secs_f64()*1e3; if e<b {b=e;} }
    b
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let a: Vec<f32> = (0..R*C).map(|i| ((i%2000) as f32 - 1000.0) * 0.01).collect();
    let pos: Vec<f32> = (0..R*C).map(|i| ((i%2000) as f32) * 0.01 + 0.001).collect();
    let python = std::env::var("PYTORCH_PYTHON").unwrap_or_else(|_| "python3".to_string());
    let py = r#"
import time,torch
torch.set_num_threads(8)
R,C=4000,4000
a=(((torch.arange(R*C,dtype=torch.int64)%2000).float()-1000.0)*0.01).reshape(R,C)
pos=(((torch.arange(R*C,dtype=torch.int64)%2000).float())*0.01+0.001).reshape(R,C)
def t(fn,n=9):
    for _ in range(2): fn()
    ts=[]
    for _ in range(n): s=time.perf_counter(); fn(); ts.append((time.perf_counter()-s)*1e3)
    return min(ts)
for name,fn in [("relu",lambda:torch.relu(a)),
                ("neg",lambda:torch.neg(a)),
                ("abs",lambda:torch.abs(a)),
                ("sqrt",lambda:torch.sqrt(pos)),
                ("reciprocal",lambda:torch.reciprocal(pos))]:
    print("PT %s %.4f"%(name,t(fn)))
"#;
    let mut ch=Command::new(&python).arg("-").stdin(Stdio::piped()).stdout(Stdio::piped()).spawn()?;
    ch.stdin.as_mut().unwrap().write_all(py.as_bytes())?;
    let o=ch.wait_with_output(); let pt=String::from_utf8_lossy(&o.unwrap().stdout).to_string();
    let rep=|n:&str,ft:f64|{ if let Some(p)=pt.lines().find_map(|l|{let mut it=l.strip_prefix("PT ")?.split_whitespace(); if it.next()?==n {it.next()?.parse::<f64>().ok()} else {None}}){let r=p/ft; let tag=if r>=1.0{format!("FT {r:.2}x FASTER")}else{format!("FT {:.2}x SLOWER",1.0/r)}; println!("  {n:<12} {ft:8.3} {p:8.3}   {tag}");}};
    println!("op            FT(ms)    PT(ms)   verdict");
    rep("relu", t1(&a, |s,x| { let _=s.tensor_relu(x); }));
    rep("neg", t1(&a, |s,x| { let _=s.tensor_neg(x); }));
    rep("abs", t1(&a, |s,x| { let _=s.tensor_abs(x); }));
    rep("sqrt", t1(&pos, |s,x| { let _=s.tensor_sqrt(x); }));
    rep("reciprocal", t1(&pos, |s,x| { let _=s.tensor_reciprocal(x); }));
    Ok(())
}

use std::io::Write; use std::process::{Command, Stdio}; use std::time::Instant;
use ft_api::FrankenTorchSession; use ft_autograd::TensorNodeId; use ft_core::ExecutionMode;
const R: usize = 4000; const C: usize = 4000;
fn t2<F: Fn(&mut FrankenTorchSession, TensorNodeId, TensorNodeId)>(a: &[f32], b: &[f32], f: F) -> f64 {
    let mut bst=f64::INFINITY;
    for _ in 0..7 { let mut s=FrankenTorchSession::new(ExecutionMode::Strict);
        let x=s.tensor_variable_f32(a.to_vec(),vec![R,C],false).unwrap();
        let y=s.tensor_variable_f32(b.to_vec(),vec![R,C],false).unwrap();
        let t=Instant::now(); f(&mut s,x,y); let e=t.elapsed().as_secs_f64()*1e3; if e<bst{bst=e;} } bst
}
fn t1<F: Fn(&mut FrankenTorchSession, TensorNodeId)>(a: &[f32], f: F) -> f64 {
    let mut bst=f64::INFINITY;
    for _ in 0..7 { let mut s=FrankenTorchSession::new(ExecutionMode::Strict);
        let x=s.tensor_variable_f32(a.to_vec(),vec![R,C],false).unwrap();
        let t=Instant::now(); f(&mut s,x); let e=t.elapsed().as_secs_f64()*1e3; if e<bst{bst=e;} } bst
}
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let a: Vec<f32>=(0..R*C).map(|i|(i%17) as f32 - 8.0).collect();
    let b: Vec<f32>=(0..R*C).map(|i|(i%13) as f32 - 6.0).collect();
    let python=std::env::var("PYTORCH_PYTHON").unwrap_or_else(|_|"python3".into());
    let py=r#"
import time,torch
import torch.nn.functional as Fn
torch.set_num_threads(8)
R,C=4000,4000
a=((torch.arange(R*C,dtype=torch.int64)%17).float()-8.0).reshape(R,C)
b=((torch.arange(R*C,dtype=torch.int64)%13).float()-6.0).reshape(R,C)
def t(fn,n=7):
    for _ in range(2): fn()
    ts=[]
    for _ in range(n): s=time.perf_counter(); fn(); ts.append((time.perf_counter()-s)*1e3)
    return min(ts)
for name,fn in [("cat_anchor",lambda:torch.cat([a,a],1)),
                ("copysign",lambda:torch.copysign(a,b)),
                ("heaviside",lambda:torch.heaviside(a,b)),
                ("softshrink",lambda:Fn.softshrink(a,0.5)),
                ("smoothl1_n",lambda:Fn.smooth_l1_loss(a,b,reduction='none',beta=1.0))]:
    print("PT %s %.4f"%(name,t(fn)))
"#;
    let mut ch=Command::new(&python).arg("-").stdin(Stdio::piped()).stdout(Stdio::piped()).spawn()?;
    ch.stdin.as_mut().unwrap().write_all(py.as_bytes())?;
    let o=ch.wait_with_output(); let pt=String::from_utf8_lossy(&o.unwrap().stdout).to_string();
    let rep=|n:&str,ft:f64|{ if let Some(p)=pt.lines().find_map(|l|{let mut it=l.strip_prefix("PT ")?.split_whitespace(); if it.next()?==n {it.next()?.parse::<f64>().ok()} else {None}}){let r=p/ft; let tag=if r>=1.0{format!("FT {r:.2}x FASTER")}else{format!("FT {:.2}x SLOWER",1.0/r)}; println!("  {n:<11} {ft:8.3} {p:8.3}   {tag}");}};
    println!("op           FT(ms)    PT(ms)   verdict");
    rep("cat_anchor", t2(&a,&b,|s,x,_|{let _=s.tensor_cat(&[x,x],1);}));
    rep("copysign", t2(&a,&b,|s,x,y|{let _=s.tensor_copysign(x,y);}));
    rep("heaviside", t2(&a,&b,|s,x,y|{let _=s.tensor_heaviside(x,y);}));
    rep("softshrink", t1(&a,|s,x|{let _=s.tensor_softshrink(x,0.5);}));
    rep("smoothl1_n", t2(&a,&b,|s,x,y|{let _=s.tensor_smooth_l1_loss(x,y,"none",1.0);}));
    Ok(())
}

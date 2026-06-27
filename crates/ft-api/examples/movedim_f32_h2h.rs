//! movedim (transpose) on f32 [4000,4000] FT vs PyTorch. cat_anchor = sanity.
use std::io::Write; use std::process::{Command,Stdio}; use std::time::Instant;
use ft_api::FrankenTorchSession; use ft_core::ExecutionMode;
const R: usize = 4000;
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let x: Vec<f32> = (0..R*R).map(|i| (i%9973) as f32 - 4986.0).collect();
    let python = std::env::var("PYTORCH_PYTHON").unwrap_or_else(|_| "python3".to_string());
    let py = r#"
import time,torch
torch.set_num_threads(8)
R=4000
x=((torch.arange(R*R,dtype=torch.int64)%9973).float()-4986.0).reshape(R,R)
def t(fn,n=7):
    for _ in range(2): fn()
    ts=[]
    for _ in range(n): s=time.perf_counter(); fn(); ts.append((time.perf_counter()-s)*1e3)
    return min(ts)
print("PT cat_anchor %.4f"%t(lambda:torch.cat([x,x],1)))
print("PT movedim %.4f"%t(lambda:torch.movedim(x,0,1).clone()))
"#;
    let mut ch=Command::new(&python).arg("-").stdin(Stdio::piped()).stdout(Stdio::piped()).spawn()?;
    ch.stdin.as_mut().unwrap().write_all(py.as_bytes())?;
    let o=ch.wait_with_output(); let pt=String::from_utf8_lossy(&o.unwrap().stdout).to_string();
    let g=|n:&str| pt.lines().find_map(|l|{let mut it=l.strip_prefix("PT ")?.split_whitespace(); if it.next()?==n {it.next()?.parse::<f64>().ok()} else {None}});
    let rep=|n:&str,ft:f64|{ if let Some(p)=g(n){let r=p/ft; let tag=if r>=1.0{format!("FT {r:.2}x FASTER")}else{format!("FT {:.2}x SLOWER",1.0/r)}; println!("  {n:<12} {ft:8.3} {p:8.3}   {tag}");}};
    println!("op            FT(ms)    PT(ms)   verdict");
    let xc=x.clone();
    let mut a=f64::INFINITY;
    for _ in 0..7 { let mut s=FrankenTorchSession::new(ExecutionMode::Strict); let xn=s.tensor_variable_f32(xc.clone(),vec![R,R],false).unwrap(); let t=Instant::now(); let _=s.tensor_cat(&[xn,xn],1); let e=t.elapsed().as_secs_f64()*1e3; if e<a{a=e;} }
    rep("cat_anchor",a);
    let mut m=f64::INFINITY;
    for _ in 0..7 { let mut s=FrankenTorchSession::new(ExecutionMode::Strict); let xn=s.tensor_variable_f32(x.clone(),vec![R,R],false).unwrap(); let t=Instant::now(); let _=s.tensor_movedim(xn,0,1); let e=t.elapsed().as_secs_f64()*1e3; if e<m{m=e;} }
    rep("movedim",m);
    Ok(())
}

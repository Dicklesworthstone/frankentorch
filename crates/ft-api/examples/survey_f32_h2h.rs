//! F32 survey: hot ops whose no-grad fast path may be F64-only (silently slow on f32).
//! Run: PYTORCH_PYTHON=/path/to/python cargo run --release -p ft-api --example survey_f32_h2h
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
fn t2<F: Fn(&mut FrankenTorchSession, TensorNodeId, TensorNodeId)>(a: &[f32], b2: &[f32], f: F) -> f64 {
    let mut b = f64::INFINITY;
    for _ in 0..7 { let mut s = FrankenTorchSession::new(ExecutionMode::Strict);
        let x = s.tensor_variable_f32(a.to_vec(), vec![R, C], false).unwrap();
        let y = s.tensor_variable_f32(b2.to_vec(), vec![R, C], false).unwrap();
        let t = Instant::now(); f(&mut s, x, y); let e = t.elapsed().as_secs_f64()*1e3; if e<b {b=e;} }
    b
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let a: Vec<f32> = (0..R*C).map(|i| (i%17) as f32 - 8.0).collect();
    let b: Vec<f32> = (0..R*C).map(|i| (i%13) as f32 - 6.0).collect();
    let cond: Vec<f32> = (0..R*C).map(|i| (i%2) as f32).collect();
    let python = std::env::var("PYTORCH_PYTHON").unwrap_or_else(|_| "python3".to_string());
    let py = r#"
import time,torch
torch.set_num_threads(8)
R,C=4000,4000
a=((torch.arange(R*C,dtype=torch.int64)%17).float()-8.0).reshape(R,C)
b=((torch.arange(R*C,dtype=torch.int64)%13).float()-6.0).reshape(R,C)
cond=((torch.arange(R*C,dtype=torch.int64)%2)==0).reshape(R,C)
def t(fn,n=7):
    for _ in range(2): fn()
    ts=[]
    for _ in range(n): s=time.perf_counter(); fn(); ts.append((time.perf_counter()-s)*1e3)
    return min(ts)
for name,fn in [("cat_anchor",lambda:torch.cat([a,a],1)),
                ("where",lambda:torch.where(cond,a,b)),
                ("maximum",lambda:torch.maximum(a,b)),
                ("lerp",lambda:torch.lerp(a,b,0.3)),
                ("addcmul",lambda:torch.addcmul(a,b,a,value=0.5)),
                ("mul_scalar",lambda:a*2.5)]:
    print("PT %s %.4f"%(name,t(fn)))
"#;
    let mut ch=Command::new(&python).arg("-").stdin(Stdio::piped()).stdout(Stdio::piped()).spawn()?;
    ch.stdin.as_mut().unwrap().write_all(py.as_bytes())?;
    let o=ch.wait_with_output(); let pt=String::from_utf8_lossy(&o.unwrap().stdout).to_string();
    let rep=|n:&str,ft:f64|{ if let Some(p)=pt.lines().find_map(|l|{let mut it=l.strip_prefix("PT ")?.split_whitespace(); if it.next()?==n {it.next()?.parse::<f64>().ok()} else {None}}){let r=p/ft; let tag=if r>=1.0{format!("FT {r:.2}x FASTER")}else{format!("FT {:.2}x SLOWER",1.0/r)}; println!("  {n:<12} {ft:8.3} {p:8.3}   {tag}");}};
    println!("op            FT(ms)    PT(ms)   verdict");
    rep("cat_anchor", t1(&a, |s,x| { let _ = s.tensor_cat(&[x,x],1); }));
    rep("where", { let cc=cond.clone(); let bb=b.clone(); let mut bst=f64::INFINITY;
        for _ in 0..7 { let mut s=FrankenTorchSession::new(ExecutionMode::Strict);
            let cn=s.tensor_variable_f32(cc.clone(),vec![R,C],false).unwrap();
            let xn=s.tensor_variable_f32(a.clone(),vec![R,C],false).unwrap();
            let yn=s.tensor_variable_f32(bb.clone(),vec![R,C],false).unwrap();
            let t=Instant::now(); let _=s.tensor_where(cn,xn,yn); let e=t.elapsed().as_secs_f64()*1e3; if e<bst{bst=e;} } bst });
    rep("maximum", t2(&a,&b, |s,x,y| { let _=s.tensor_maximum(x,y); }));
    rep("lerp", t2(&a,&b, |s,x,y| { let _=s.tensor_lerp(x,y,0.3); }));
    rep("addcmul", { let bb=b.clone(); let mut bst=f64::INFINITY;
        for _ in 0..7 { let mut s=FrankenTorchSession::new(ExecutionMode::Strict);
            let xn=s.tensor_variable_f32(a.clone(),vec![R,C],false).unwrap();
            let yn=s.tensor_variable_f32(bb.clone(),vec![R,C],false).unwrap();
            let t=Instant::now(); let _=s.tensor_addcmul(xn,yn,xn,0.5); let e=t.elapsed().as_secs_f64()*1e3; if e<bst{bst=e;} } bst });
    rep("mul_scalar", t1(&a, |s,x| { let _=s.mul_scalar(x,2.5); }));
    Ok(())
}

use ft_api::FrankenTorchSession;
use ft_core::ExecutionMode;
use std::io::Write;
use std::process::{Command, Stdio};
use std::time::Instant;
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let python = std::env::var("PYTORCH_PYTHON").unwrap_or_else(|_| "python3".to_string());
    let n = 16_000_000usize;
    let a: Vec<f32> = (0..n).map(|i| ((i % 9973) as f32 - 5000.0) * 0.002).collect(); // ~(-10,10)
    let e: Vec<f32> = (0..n).map(|i| ((i % 1999) as f32 / 2000.0) - 0.5).collect(); // (-0.5,0.5)
    let p: Vec<f32> = (0..n).map(|i| ((i % 9973) as f32) * 0.001 + 1.5).collect(); // >1
    let tt = |w: u8| { let mut best=f64::INFINITY; for _ in 0..7 {
        let mut s=FrankenTorchSession::new(ExecutionMode::Strict);
        let xa=s.tensor_variable_f32(a.clone(),vec![n],false).unwrap();
        let xe=s.tensor_variable_f32(e.clone(),vec![n],false).unwrap();
        let xp=s.tensor_variable_f32(p.clone(),vec![n],false).unwrap();
        let ti=Instant::now();
        match w {0=>{let _=s.tensor_add(xa,xa);}1=>{let _=s.tensor_erfinv(xe);}2=>{let _=s.tensor_erfcx(xa);}3=>{let _=s.tensor_special_bessel_j0(xa);}4=>{let _=s.tensor_special_bessel_j1(xa);}5=>{let _=s.tensor_special_airy_ai(xa);}6=>{let _=s.tensor_special_modified_bessel_k0(xp);}7=>{let _=s.tensor_polygamma(1,xp);}_=>{let _=s.tensor_multigammaln(xp,2);}}
        let e2=ti.elapsed().as_secs_f64()*1e3; if e2<best{best=e2;} } best };
    let py = format!(r#"
import time,torch
torch.set_num_threads(8)
n={n}
a=(((torch.arange(n,dtype=torch.int64)%9973).float()-5000.0)*0.002)
e=(((torch.arange(n,dtype=torch.int64)%1999).float()/2000.0)-0.5)
p=(((torch.arange(n,dtype=torch.int64)%9973).float())*0.001+1.5)
def tm(fn,reps=7):
    for _ in range(2): fn()
    ts=[]
    for _ in range(reps): s=time.perf_counter(); fn(); ts.append((time.perf_counter()-s)*1e3)
    return min(ts)
print("PT add %.3f"%tm(lambda:a+a))
print("PT erfinv %.3f"%tm(lambda:torch.erfinv(e)))
print("PT erfcx %.3f"%tm(lambda:torch.special.erfcx(a)))
print("PT bj0 %.3f"%tm(lambda:torch.special.bessel_j0(a)))
print("PT bj1 %.3f"%tm(lambda:torch.special.bessel_j1(a)))
print("PT airy %.3f"%tm(lambda:torch.special.airy_ai(a)))
print("PT k0 %.3f"%tm(lambda:torch.special.modified_bessel_k0(p)))
print("PT polyg %.3f"%tm(lambda:torch.polygamma(1,p)))
print("PT mgln %.3f"%tm(lambda:torch.special.multigammaln(p,2)))
"#, n=n);
    let mut ch=Command::new(&python).arg("-").stdin(Stdio::piped()).stdout(Stdio::piped()).spawn()?;
    ch.stdin.as_mut().unwrap().write_all(py.as_bytes())?;
    let pt=String::from_utf8_lossy(&ch.wait_with_output()?.stdout).to_string();
    let g=|k:&str| pt.lines().find_map(|l|{let mut it=l.strip_prefix("PT ")?.split_whitespace(); if it.next()?==k {it.next()?.parse::<f64>().ok()} else {None}}).unwrap_or(f64::NAN);
    let v=|ft:f64,pp:f64| if pp>=ft {format!("FT {:.2}x FASTER",pp/ft)} else {format!("FT {:.2}x SLOWER",ft/pp)};
    println!("specfn ~16M f32 (torch 8t / FT default), min-of-7");
    for (lbl,w) in [("add",0u8),("erfinv",1),("erfcx",2),("bj0",3),("bj1",4),("airy",5),("k0",6),("polyg",7),("mgln",8)] { let ft=tt(w); println!("  {lbl:<8} FT {ft:8.3}  PT {:8.3}  => {}",g(lbl),v(ft,g(lbl))); }
    Ok(())
}

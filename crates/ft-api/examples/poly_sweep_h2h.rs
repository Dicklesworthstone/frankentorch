use ft_api::FrankenTorchSession;
use ft_core::ExecutionMode;
use std::io::Write;
use std::process::{Command, Stdio};
use std::time::Instant;
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let python = std::env::var("PYTORCH_PYTHON").unwrap_or_else(|_| "python3".to_string());
    let n = 16_000_000usize;
    let a: Vec<f32> = (0..n).map(|i| ((i % 1999) as f32 / 2000.0) - 0.5).collect(); // (-0.5,0.5)
    let tt = |w: u8| { let mut best=f64::INFINITY; for _ in 0..7 {
        let mut s=FrankenTorchSession::new(ExecutionMode::Strict);
        let x=s.tensor_variable_f32(a.clone(),vec![n],false).unwrap();
        let ti=Instant::now();
        match w {0=>{let _=s.tensor_add(x,x);}1=>{let _=s.tensor_special_chebyshev_polynomial_t(x,5);}2=>{let _=s.tensor_special_hermite_polynomial_h(x,5);}3=>{let _=s.tensor_special_hermite_polynomial_he(x,5);}_=>{let _=s.tensor_special_laguerre_polynomial_l(x,5);}}
        let e2=ti.elapsed().as_secs_f64()*1e3; if e2<best{best=e2;} } best };
    let py = format!(r#"
import time,torch
torch.set_num_threads(8)
n={n}
a=(((torch.arange(n,dtype=torch.int64)%1999).float()/2000.0)-0.5)
def tm(fn,reps=7):
    for _ in range(2): fn()
    ts=[]
    for _ in range(reps): s=time.perf_counter(); fn(); ts.append((time.perf_counter()-s)*1e3)
    return min(ts)
print("PT add %.3f"%tm(lambda:a+a))
print("PT cheb_t %.3f"%tm(lambda:torch.special.chebyshev_polynomial_t(a,5)))
print("PT herm_h %.3f"%tm(lambda:torch.special.hermite_polynomial_h(a,5)))
print("PT herm_he %.3f"%tm(lambda:torch.special.hermite_polynomial_he(a,5)))
print("PT lag_l %.3f"%tm(lambda:torch.special.laguerre_polynomial_l(a,5)))
"#, n=n);
    let mut ch=Command::new(&python).arg("-").stdin(Stdio::piped()).stdout(Stdio::piped()).spawn()?;
    ch.stdin.as_mut().unwrap().write_all(py.as_bytes())?;
    let pt=String::from_utf8_lossy(&ch.wait_with_output()?.stdout).to_string();
    let g=|k:&str| pt.lines().find_map(|l|{let mut it=l.strip_prefix("PT ")?.split_whitespace(); if it.next()?==k {it.next()?.parse::<f64>().ok()} else {None}}).unwrap_or(f64::NAN);
    let v=|ft:f64,pp:f64| if pp>=ft {format!("FT {:.2}x FASTER",pp/ft)} else {format!("FT {:.2}x SLOWER",ft/pp)};
    println!("poly ~16M f32 (torch 8t / FT default), min-of-7");
    for (lbl,w) in [("add",0u8),("cheb_t",1),("herm_h",2),("herm_he",3),("lag_l",4)] { let ft=tt(w); println!("  {lbl:<8} FT {ft:8.3}  PT {:8.3}  => {}",g(lbl),v(ft,g(lbl))); }
    Ok(())
}

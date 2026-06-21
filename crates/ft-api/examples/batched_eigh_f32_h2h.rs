use std::process::Command; use std::time::Instant;
use ft_api::FrankenTorchSession; use ft_core::{ExecutionMode, DType};
fn main() {
    for (bb,k) in [(100000usize,4usize),(20000,16),(4000,32)] {
        let mut data=vec![0.0f64; bb*k*k];
        for b in 0..bb { for i in 0..k { for j in 0..k { data[b*k*k+i*k+j]=(((b*7+i*13+j*5)%97) as f64)*0.01; }}}
        for b in 0..bb { for i in 0..k { for j in 0..k { let s=(data[b*k*k+i*k+j]+data[b*k*k+j*k+i])*0.5; data[b*k*k+i*k+j]=s; }} for i in 0..k { data[b*k*k+i*k+i]+=k as f64; }}
        let mut best=f64::INFINITY; let mut esum=0.0;
        for _ in 0..5 {
            let mut s=FrankenTorchSession::new(ExecutionMode::Strict);
            let xf=s.tensor_variable(data.clone(),vec![bb,k,k],false).unwrap();
            let x=s.tensor_to_dtype(xf, DType::F32).unwrap();
            let t=Instant::now();
            let (ev,_q)=s.tensor_linalg_eigh(x).unwrap();
            let el=t.elapsed().as_secs_f64()*1e3;
            if el<best { best=el; esum=s.tensor_values_lossy_f64(ev).unwrap().iter().sum(); }
        }
        let pysrc=format!(r#"
import time, torch
torch.set_num_threads(8); B,k={bb},{k}
idx=torch.arange(B*k*k,dtype=torch.float64); b=idx//(k*k); r=(idx//k)%k; c=idx%k
d=(((b*7+r*13+c*5)%97).double())*0.01
A=(d.reshape(B,k,k)); A=(A+A.transpose(-1,-2))*0.5; A=A+k*torch.eye(k,dtype=torch.float64)
A=A.float()
for _ in range(2): torch.linalg.eigh(A)
ts=[]
for _ in range(5):
    t=time.perf_counter(); torch.linalg.eigh(A); ts.append((time.perf_counter()-t)*1e3)
w,_=torch.linalg.eigh(A); print("MS",sorted(ts)[0]); print("ESUM",w.double().sum().item())
"#);
        let python=std::env::var("PYTORCH_PYTHON").unwrap_or("python3".into());
        print!("B={bb} k={k} f32: FT {best:.1} ms  esum {esum:.4e}");
        if let Ok(o)=Command::new(&python).arg("-c").arg(&pysrc).output() { if o.status.success() {
            let s=String::from_utf8_lossy(&o.stdout);
            let g=|p:&str| s.lines().find_map(|l| l.strip_prefix(p).and_then(|v| v.trim().parse::<f64>().ok()));
            if let (Some(p),Some(pe))=(g("MS "),g("ESUM ")) {
                let rel=(esum-pe).abs()/(pe.abs()+1e-6); let rr=p/best;
                println!("  | PyTorch {p:.1} ms  esum {pe:.4e} | {} | {}", if rel<1e-4 {"MATCH"} else {"DIFF"}, if rr>=1.0 {format!("FT {rr:.2}x FASTER")} else {format!("FT {:.2}x slower",1.0/rr)});
            }
        } else { eprintln!("\n{}", String::from_utf8_lossy(&o.stderr)); }} else { println!(); }
    }
}

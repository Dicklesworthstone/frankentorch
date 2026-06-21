use std::process::Command; use std::time::Instant;
use ft_api::FrankenTorchSession; use ft_core::{ExecutionMode, DType};
fn main() {
    for (bb,k) in [(100000usize,4usize),(20000,16)] {
        let mut data=vec![0.0f64; bb*k*k];
        for i in 0..bb*k*k { data[i]=(((i*2654435761usize)%9973) as f64)*0.001 - 5.0; }
        for b in 0..bb { for i in 0..k { data[b*k*k+i*k+i]+=(k+10) as f64; } }
        let run=|op:&str| -> (f64,f64) {
            let mut best=f64::INFINITY; let mut chk=0.0;
            for _ in 0..5 {
                let mut s=FrankenTorchSession::new(ExecutionMode::Strict);
                let xf=s.tensor_variable(data.clone(),vec![bb,k,k],false).unwrap();
                let x=s.tensor_to_dtype(xf,DType::F32).unwrap();
                let t=Instant::now();
                let c = match op {
                    "svd" => { let (_u,sv,_v)=s.tensor_linalg_svd(x,true).unwrap(); let e=t.elapsed().as_secs_f64()*1e3; let c=s.tensor_values_lossy_f64(sv).unwrap().iter().sum(); (e,c) }
                    "svdvals" => { let sv=s.tensor_linalg_svdvals(x).unwrap(); let e=t.elapsed().as_secs_f64()*1e3; let c=s.tensor_values_lossy_f64(sv).unwrap().iter().sum(); (e,c) }
                    "eigvals" => { let w=s.tensor_linalg_eigvals(x).unwrap(); let e=t.elapsed().as_secs_f64()*1e3; let c=s.tensor_values_lossy_f64(w).unwrap().iter().step_by(2).sum(); (e,c) }
                    _ => { let (w,_v)=s.tensor_linalg_eig(x).unwrap(); let e=t.elapsed().as_secs_f64()*1e3; let c=s.tensor_values_lossy_f64(w).unwrap().iter().step_by(2).sum(); (e,c) }
                };
                if c.0<best { best=c.0; chk=c.1; }
            }
            (best,chk)
        };
        for op in ["svd","svdvals","eigvals","eig"] {
            let (ft_ms, chk)=run(op);
            let pysrc=format!(r#"
import time, torch
torch.set_num_threads(8); B,k={bb},{k}
idx=torch.arange(B*k*k,dtype=torch.float64)
A=((idx*2654435761)%9973).double().mul(0.001).sub(5.0).reshape(B,k,k); A=(A+(k+10)*torch.eye(k,dtype=torch.float64)).float()
op="{op}"
f={{"svd":lambda:torch.linalg.svd(A),"svdvals":lambda:torch.linalg.svdvals(A),"eigvals":lambda:torch.linalg.eigvals(A),"eig":lambda:torch.linalg.eig(A)}}[op]
for _ in range(2): f()
ts=[]
import time as tm
for _ in range(5): s=tm.perf_counter(); f(); ts.append((tm.perf_counter()-s)*1e3)
r=f()
if op in ("svd",): chk=r.S.double().sum().item()
elif op=="svdvals": chk=r.double().sum().item()
elif op=="eigvals": chk=r.real.double().sum().item()
else: chk=r[0].real.double().sum().item()
print("MS",sorted(ts)[0]); print("CHK",chk)
"#);
            let python=std::env::var("PYTORCH_PYTHON").unwrap_or("python3".into());
            print!("k={k} {op} f32: FT {ft_ms:.1}ms chk {chk:.4e}");
            if let Ok(o)=Command::new(&python).arg("-c").arg(&pysrc).output() { if o.status.success() {
                let s=String::from_utf8_lossy(&o.stdout);
                let g=|p:&str| s.lines().find_map(|l| l.strip_prefix(p).and_then(|v| v.trim().parse::<f64>().ok()));
                if let (Some(p),Some(pc))=(g("MS "),g("CHK ")) {
                    let rel=(chk-pc).abs()/(pc.abs()+1e-6); let rr=p/ft_ms;
                    println!(" | torch {p:.1}ms chk {pc:.4e} | {} | {}", if rel<1e-3 {"OK"} else {"DIFF"}, if rr>=1.0 {format!("FT {rr:.2}x FASTER")} else {format!("{:.2}x slower",1.0/rr)});
                }
            } else { eprintln!("\n{}", String::from_utf8_lossy(&o.stderr)); }} else { println!(); }
        }
    }
}

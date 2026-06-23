use std::error::Error;
use std::time::Instant;
use ft_api::FrankenTorchSession;
use ft_core::ExecutionMode;
fn boxed<E: std::fmt::Debug>(e: E) -> std::io::Error { std::io::Error::other(format!("{e:?}")) }
fn fill(n: usize) -> Vec<f64> { (0..n).map(|i| (((i*2654435761usize)%100003) as f64/100003.0)-0.5).collect() }
fn main() -> Result<(), Box<dyn Error>> {
    let (r,c)=(4000usize,20000usize);
    for op in ["cummax","cummin"] {
        let mut best=f64::INFINITY;
        for _ in 0..3 {
            let ad=fill(r*c);
            let mut s=FrankenTorchSession::new(ExecutionMode::Strict);
            let a=s.tensor_variable(ad,vec![r,c],false).map_err(boxed)?;
            let st=Instant::now();
            if op=="cummax" { s.tensor_cummax_dim(a,1).map_err(boxed)?; } else { s.tensor_cummin_dim(a,1).map_err(boxed)?; }
            let ms=st.elapsed().as_secs_f64()*1e3; if ms<best{best=ms;}
        }
        println!("[{r},{c}] dim=1: FT {op} {best:.1} ms");
    }
    Ok(())
}

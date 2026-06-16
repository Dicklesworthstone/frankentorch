//! conv2d f32 double-backward probe vs torch f32 (frankentorch-lboou).
//!   cargo run -q --release -p ft-api --example conv2d_f32_hessian_probe
use ft_api::FrankenTorchSession;
use ft_core::{DType, ExecutionMode};
fn main() {
    let mut s = FrankenTorchSession::new(ExecutionMode::Strict);
    let xv: Vec<f64> = (0..16).map(|i| (i%7) as f64*0.1-0.3).collect();
    let wv: Vec<f64> = (0..9).map(|i| (i%5) as f64*0.2-0.4).collect();
    let x = s.tensor_variable(xv, vec![1,1,4,4], true).unwrap();
    let x = s.tensor_to_dtype(x, DType::F32).unwrap();
    let w = s.tensor_variable(wv, vec![1,1,3,3], false).unwrap();
    let w = s.tensor_to_dtype(w, DType::F32).unwrap();
    match (|| { let y=s.functional_conv2d(x,w,None,(1,1),(0,0))?; let sq=s.tensor_mul(y,y)?; let loss=s.tensor_sum(sq)?; s.tensor_functional_hessian(loss,x) })() {
        Ok(h)=>{ let d:Vec<f64>=(0..16).map(|i|(h[i*16+i]*1e4).round()/1e4).collect(); println!("f32 diag: {d:?}"); println!("f32 H[0,5]={:.4} H[0,10]={:.4}", h[5], h[10]); },
        Err(e)=>println!("f32 ERR {e:?}"),
    }
}

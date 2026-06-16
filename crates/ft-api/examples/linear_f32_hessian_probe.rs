//! linear f32 double-backward status probe vs torch f32 (frankentorch-lboou):
//! linear f32 falls through to composed addmm/matmul (cg-correct) -> works free.
//!   cargo run -q --release -p ft-api --example linear_f32_hessian_probe
use ft_api::FrankenTorchSession;
use ft_core::{DType, ExecutionMode};
fn main() {
    let mut s = FrankenTorchSession::new(ExecutionMode::Strict);
    let x = s.tensor_variable((0..12).map(|i|(i%7) as f64*0.1-0.3).collect(), vec![3,4], true).unwrap();
    let x = s.tensor_to_dtype(x, DType::F32).unwrap();
    let w = s.tensor_variable((0..8).map(|i|(i%5) as f64*0.2-0.4).collect(), vec![2,4], false).unwrap();
    let w = s.tensor_to_dtype(w, DType::F32).unwrap();
    match (|| { let y=s.functional_linear(x,w,None)?; let sq=s.tensor_mul(y,y)?; let loss=s.tensor_sum(sq)?; s.tensor_functional_hessian(loss,x) })() {
        Ok(h)=>{ let d:Vec<f64>=(0..12).map(|i|(h[i*12+i]*1e4).round()/1e4).collect(); println!("lin f32 diag: {d:?}"); },
        Err(e)=>println!("lin f32 ERR {e:?}"),
    }
}

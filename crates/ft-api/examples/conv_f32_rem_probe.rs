//! f32 double-backward status probe for grouped/conv3d/conv_transpose2d vs torch.
//!   cargo run -q --release -p ft-api --example conv_f32_rem_probe
use ft_api::FrankenTorchSession; use ft_autograd::{AutogradError,TensorNodeId}; use ft_core::{DType,ExecutionMode};
fn run(name:&str,xv:Vec<f64>,xs:Vec<usize>,wv:Vec<f64>,ws:Vec<usize>,f:impl Fn(&mut FrankenTorchSession,TensorNodeId,TensorNodeId)->Result<TensorNodeId,AutogradError>,k:usize){
    let mut s=FrankenTorchSession::new(ExecutionMode::Strict); let n:usize=xs.iter().product();
    let x=s.tensor_variable(xv,xs,true).unwrap(); let x=s.tensor_to_dtype(x,DType::F32).unwrap();
    let w=s.tensor_variable(wv,ws,false).unwrap(); let w=s.tensor_to_dtype(w,DType::F32).unwrap();
    match (||{let y=f(&mut s,x,w)?; let sq=s.tensor_mul(y,y)?; let l=s.tensor_sum(sq)?; s.tensor_functional_hessian(l,x)})(){
        Ok(h)=>{let d:Vec<f64>=(0..k.min(n)).map(|i|(h[i*n+i]*1e4).round()/1e4).collect(); println!("{name}: {d:?}");}, Err(e)=>println!("{name}: ERR {e:?}"),}
}
fn main(){
    run("grouped_f32",(0..32).map(|i|(i%7) as f64*0.1-0.3).collect(),vec![1,2,4,4],(0..18).map(|i|(i%5) as f64*0.2-0.4).collect(),vec![2,1,3,3],|s,x,w|s.functional_conv2d_grouped(x,w,None,(1,1),(0,0),2),4);
    run("conv3d_f32",(0..64).map(|i|(i%7) as f64*0.1-0.3).collect(),vec![1,1,4,4,4],(0..8).map(|i|(i%5) as f64*0.2-0.4).collect(),vec![1,1,2,2,2],|s,x,w|s.functional_conv3d(x,w,None,(1,1,1),(0,0,0)),4);
    run("ct2d_f32",vec![0.1,0.2,-0.1,0.3],vec![1,1,2,2],vec![0.2,-0.4,0.1,0.3,0.0,0.2,-0.3,0.1,0.4],vec![1,1,3,3],|s,x,w|s.functional_conv_transpose2d(x,w,None,(1,1),(0,0),(0,0)),4);
}

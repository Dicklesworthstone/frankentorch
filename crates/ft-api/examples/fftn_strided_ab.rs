//! fftn 3-D strided-dim parallelization A/B (cc). tensor_fft_along_dim parallelized the
//! contiguous (last) dim but ran strided non-last dims serially; fftn's middle-dim pass
//! (stride_outer>=2) is now parallel over disjoint outer blocks. Times no-grad fftn on a
//! 3-D tensor (inputs once, op+read per iter). Run: cargo run --release -p ft-api --example fftn_strided_ab
use std::time::Instant;
use ft_api::FrankenTorchSession;
use ft_core::ExecutionMode;
const D0: usize = 256;
const D1: usize = 100;
const D2: usize = 16;
fn main() {
    let iters: usize = std::env::var("ITERS").ok().and_then(|s| s.parse().ok()).unwrap_or(30);
    let mut s = FrankenTorchSession::new(ExecutionMode::Strict);
    let re: Vec<f64> = (0..D0*D1*D2).map(|i| ((i as f64)*0.0007).sin()).collect();
    let im: Vec<f64> = (0..D0*D1*D2).map(|i| ((i as f64)*0.0009).cos()).collect();
    let rn = s.tensor_variable(re, vec![D0,D1,D2], false).unwrap();
    let imn = s.tensor_variable(im, vec![D0,D1,D2], false).unwrap();
    let x = s.tensor_complex(rn, imn).unwrap();
    let op = |s: &mut FrankenTorchSession| -> f64 {
        let o = s.tensor_fftn(x, None, None).unwrap();
        let r = s.tensor_real(o).unwrap();
        s.tensor_values(r).unwrap().iter().map(|v| v.abs()).sum()
    };
    for _ in 0..2 { let _ = op(&mut s); }
    let mut t = Vec::with_capacity(iters); let mut chk=0.0;
    for _ in 0..iters { let s0=Instant::now(); chk=op(&mut s); t.push(s0.elapsed().as_secs_f64()*1e3); }
    t.sort_by(|a,b| a.partial_cmp(b).unwrap());
    let tot: f64 = t.iter().sum();
    println!("fftn 3-D [{D0},{D1},{D2}] {iters} iters: min {:.3} ms  total {tot:.1} ms  checksum {chk:.6e}", t[0]);
}

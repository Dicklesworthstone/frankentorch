//! Decompose conv2d_backward_f64 at the training bench config (b4,ic64,oc64,k3,
//! s1,p1, hw=64) to size the streamable dpanel+col2im half.
use ft_kernel_cpu::{conv2d_backward_f64, conv2d_col2im_f64, conv2d_im2col_f64};
use std::time::Instant;
fn t<F: FnMut()>(mut f: F, it: usize) -> f64 {
    f();
    let s = Instant::now();
    for _ in 0..it {
        f();
    }
    s.elapsed().as_secs_f64() * 1e3 / it as f64
}
fn main() {
    println!("threads={}", rayon::current_num_threads());
    for &hw in &[32usize, 64] {
        let (batch, ic, oc, kh, kw) = (4usize, 64usize, 64usize, 3usize, 3usize);
        let (ph, pw) = (hw + 2, hw + 2);
        let (oh, ow) = (hw, hw);
        let (sh, sw) = (1usize, 1usize);
        let pad: Vec<f64> = (0..batch * ic * ph * pw)
            .map(|i| (i % 251) as f64 * 0.001 - 0.12)
            .collect();
        let wt: Vec<f64> = (0..oc * ic * kh * kw)
            .map(|i| (i % 97) as f64 * 0.002 - 0.1)
            .collect();
        let dout: Vec<f64> = (0..batch * oc * oh * ow)
            .map(|i| (i % 131) as f64 * 0.003 - 0.2)
            .collect();
        let pwid = ic * kh * kw;
        let pc = oh * ow;
        let flat = batch * pc;
        let dpanel = vec![0.5f64; flat * pwid];
        let it = if hw <= 32 { 15 } else { 8 };
        let full = t(
            || {
                let _ = conv2d_backward_f64(
                    &dout, &pad, &wt, batch, ic, ph, pw, kh, kw, oh, ow, sh, sw, oc, true,
                );
            },
            it,
        );
        let im = t(
            || {
                let _ = conv2d_im2col_f64(&pad, batch, ic, ph, pw, kh, kw, oh, ow, sh, sw);
            },
            it,
        );
        let c2i = t(
            || {
                let _ = conv2d_col2im_f64(&dpanel, batch, ic, ph, pw, kh, kw, oh, ow, sh, sw);
            },
            it,
        );
        println!(
            "hw={hw:<4} full_bwd={full:7.2}ms  im2col(panel)={im:6.2}ms  col2im={c2i:6.2}ms  (each panel={:.0}MB)",
            (flat * pwid * 8) as f64 / 1e6
        );
    }
}

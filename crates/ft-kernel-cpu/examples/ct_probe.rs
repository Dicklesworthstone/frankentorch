//! Measure the output-transpose fraction of the (streamed) conv2d_forward_f64.
use ft_kernel_cpu::conv2d_forward_f64;
use rayon::prelude::*;
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
    for &hw in &[64usize, 128] {
        let (batch, ic, oc, kh, kw) = (4usize, 64usize, 64usize, 3usize, 3usize);
        let (ph, pw) = (hw + 2, hw + 2);
        let (oh, ow) = (hw, hw);
        let pd: Vec<f64> = (0..batch * ic * ph * pw)
            .map(|i| (i % 251) as f64 * 0.001 - 0.12)
            .collect();
        let wt: Vec<f64> = (0..oc * ic * kh * kw)
            .map(|i| (i % 97) as f64 * 0.002 - 0.1)
            .collect();
        let pc = oh * ow;
        let flat = batch * pc;
        let it = if hw <= 64 { 12 } else { 6 };
        let full = t(
            || {
                let _ =
                    conv2d_forward_f64(&pd, &wt, None, batch, ic, ph, pw, kh, kw, oh, ow, 1, 1, oc);
            },
            it,
        );
        // isolate the transpose: pre-make out_flat[flat,oc], time the strided gather.
        let of: Vec<f64> = (0..flat * oc).map(|i| i as f64 * 0.001).collect();
        let tr = t(
            || {
                let mut out = vec![0.0f64; batch * oc * pc];
                out.par_chunks_mut(pc).enumerate().for_each(|(idx, orow)| {
                    let n = idx / oc;
                    let o = idx % oc;
                    for p in 0..pc {
                        orow[p] = of[(n * pc + p) * oc + o];
                    }
                });
                std::hint::black_box(out);
            },
            it,
        );
        println!(
            "hw={hw:<4} full={full:.2}ms transpose_alone={tr:.2}ms ({:.0}% of full)",
            tr / full * 100.0
        );
    }
}

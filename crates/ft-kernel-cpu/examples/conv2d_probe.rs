//! Same-process A/B: OLD full-panel im2col+GEMM vs NEW streaming conv2d_forward_f64,
//! ONE worker. Proves the streaming path is bit-identical (digest) AND faster.
//! frankentorch-conv2d-stream.
use ft_kernel_cpu::{
    conv2d_forward_f64, conv2d_im2col_f64, matmul_rhs_transposed_contiguous_f64_into,
};
use rayon::prelude::*;
use std::time::Instant;
fn fnv(v: &[f64]) -> u64 {
    let mut h = 0xcbf29ce484222325u64;
    for x in v {
        for b in x.to_bits().to_le_bytes() {
            h ^= b as u64;
            h = h.wrapping_mul(0x100000001b3);
        }
    }
    h
}
fn t<F: FnMut()>(mut f: F, it: usize) -> f64 {
    f();
    let s = Instant::now();
    for _ in 0..it {
        f();
    }
    s.elapsed().as_secs_f64() * 1e3 / it as f64
}
// OLD path: full panel + matmul + transpose.
#[allow(clippy::too_many_arguments)]
fn old(
    padded: &[f64],
    wt: &[f64],
    bias: &[f64],
    batch: usize,
    ic: usize,
    ph: usize,
    pw: usize,
    kh: usize,
    kw: usize,
    oh: usize,
    ow: usize,
    sh: usize,
    sw: usize,
    oc: usize,
) -> Vec<f64> {
    let pwid = ic * kh * kw;
    let pc = oh * ow;
    let flat = batch * pc;
    let panel = conv2d_im2col_f64(padded, batch, ic, ph, pw, kh, kw, oh, ow, sh, sw);
    let mut of = vec![0.0f64; flat * oc];
    matmul_rhs_transposed_contiguous_f64_into(&mut of, flat, pwid, oc, &panel, wt).unwrap();
    let mut out = vec![0.0f64; batch * oc * pc];
    out.par_chunks_mut(pc).enumerate().for_each(|(idx, orow)| {
        let n = idx / oc;
        let o = idx % oc;
        let bo = bias[o];
        for p in 0..pc {
            orow[p] = of[(n * pc + p) * oc + o] + bo;
        }
    });
    out
}
fn main() {
    println!("threads={}", rayon::current_num_threads());
    for &hw in &[32usize, 64, 128] {
        let (batch, ic, oc, kh, kw) = (4usize, 64usize, 64usize, 3usize, 3usize);
        let (ph, pw) = (hw + 2, hw + 2);
        let (oh, ow) = (hw, hw);
        let (sh, sw) = (1usize, 1usize);
        let pd: Vec<f64> = (0..batch * ic * ph * pw)
            .map(|i| (i % 251) as f64 * 0.001 - 0.12)
            .collect();
        let wt: Vec<f64> = (0..oc * ic * kh * kw)
            .map(|i| (i % 97) as f64 * 0.002 - 0.1)
            .collect();
        let bias: Vec<f64> = (0..oc).map(|i| i as f64 * 0.01 - 0.3).collect();
        let it = if hw <= 64 { 15 } else { 8 };
        let do_ = fnv(&old(
            &pd, &wt, &bias, batch, ic, ph, pw, kh, kw, oh, ow, sh, sw, oc,
        ));
        let dn = fnv(&conv2d_forward_f64(
            &pd,
            &wt,
            Some(&bias),
            batch,
            ic,
            ph,
            pw,
            kh,
            kw,
            oh,
            ow,
            sh,
            sw,
            oc,
        ));
        let mo = t(
            || {
                let _ = old(
                    &pd, &wt, &bias, batch, ic, ph, pw, kh, kw, oh, ow, sh, sw, oc,
                );
            },
            it,
        );
        let mn = t(
            || {
                let _ = conv2d_forward_f64(
                    &pd,
                    &wt,
                    Some(&bias),
                    batch,
                    ic,
                    ph,
                    pw,
                    kh,
                    kw,
                    oh,
                    ow,
                    sh,
                    sw,
                    oc,
                );
            },
            it,
        );
        println!(
            "hw={hw:<4} OLD={mo:7.2}ms NEW={mn:7.2}ms  speedup={:.2}x  digest_ok={}",
            mo / mn,
            do_ == dn
        );
    }
}

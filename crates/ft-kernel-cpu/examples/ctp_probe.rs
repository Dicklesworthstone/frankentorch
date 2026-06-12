//! conv_transpose2d channels-last A/B: inline OLD strided form vs NEW, same process.
use ft_kernel_cpu::conv_transpose2d_forward_f64;
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
#[allow(clippy::manual_is_multiple_of, clippy::too_many_arguments)]
fn old(
    input: &[f64],
    weight: &[f64],
    bias: Option<&[f64]>,
    batch: usize,
    in_ch: usize,
    ih: usize,
    iw: usize,
    out_ch: usize,
    kh: usize,
    kw: usize,
    oh: usize,
    ow: usize,
    sh: usize,
    sw: usize,
    ph: usize,
    pw: usize,
) -> Vec<f64> {
    let mut out = vec![0.0f64; batch * out_ch * oh * ow];
    out.par_chunks_mut(oh * ow)
        .enumerate()
        .for_each(|(idx, orow)| {
            let n = idx / out_ch;
            let oc = idx % out_ch;
            let b0 = bias.map_or(0.0, |b| b[oc]);
            for oy in 0..oh {
                for ox in 0..ow {
                    let mut acc = b0;
                    for kr in 0..kh {
                        let yn = oy + ph;
                        if yn < kr {
                            continue;
                        }
                        let yd = yn - kr;
                        if yd % sh != 0 {
                            continue;
                        }
                        let iy = yd / sh;
                        if iy >= ih {
                            continue;
                        }
                        for kc in 0..kw {
                            let xn = ox + pw;
                            if xn < kc {
                                continue;
                            }
                            let xd = xn - kc;
                            if xd % sw != 0 {
                                continue;
                            }
                            let ix = xd / sw;
                            if ix >= iw {
                                continue;
                            }
                            let mut s = 0.0;
                            for ic in 0..in_ch {
                                s += input[((n * in_ch + ic) * ih + iy) * iw + ix]
                                    * weight[((ic * out_ch + oc) * kh + kr) * kw + kc];
                            }
                            acc += s;
                        }
                    }
                    orow[oy * ow + ox] = acc;
                }
            }
        });
    out
}
fn main() {
    println!("threads={}", rayon::current_num_threads());
    for &(batch, ic, oc, ih, iw, k, s) in &[
        (2usize, 16usize, 16usize, 16usize, 16usize, 3usize, 2usize),
        (4, 64, 64, 32, 32, 3, 1),
        (2, 128, 128, 16, 16, 4, 2),
    ] {
        let (kh, kw) = (k, k);
        let (sh, sw) = (s, s);
        let (ph, pw) = (1usize, 1usize);
        let oh = (ih - 1) * sh + kh - 2 * ph;
        let ow = (iw - 1) * sw + kw - 2 * pw;
        let input: Vec<f64> = (0..batch * ic * ih * iw)
            .map(|i| (i % 251) as f64 * 0.001 - 0.12)
            .collect();
        let wt: Vec<f64> = (0..ic * oc * kh * kw)
            .map(|i| (i % 97) as f64 * 0.002 - 0.1)
            .collect();
        let bias: Vec<f64> = (0..oc).map(|i| i as f64 * 0.01 - 0.3).collect();
        let it = 8;
        let dgo = fnv(&old(
            &input,
            &wt,
            Some(&bias),
            batch,
            ic,
            ih,
            iw,
            oc,
            kh,
            kw,
            oh,
            ow,
            sh,
            sw,
            ph,
            pw,
        ));
        let dgn = fnv(&conv_transpose2d_forward_f64(
            &input,
            &wt,
            Some(&bias),
            batch,
            ic,
            ih,
            iw,
            oc,
            kh,
            kw,
            oh,
            ow,
            sh,
            sw,
            ph,
            pw,
        ));
        let mo = t(
            || {
                let _ = old(
                    &input,
                    &wt,
                    Some(&bias),
                    batch,
                    ic,
                    ih,
                    iw,
                    oc,
                    kh,
                    kw,
                    oh,
                    ow,
                    sh,
                    sw,
                    ph,
                    pw,
                );
            },
            it,
        );
        let mn = t(
            || {
                let _ = conv_transpose2d_forward_f64(
                    &input,
                    &wt,
                    Some(&bias),
                    batch,
                    ic,
                    ih,
                    iw,
                    oc,
                    kh,
                    kw,
                    oh,
                    ow,
                    sh,
                    sw,
                    ph,
                    pw,
                );
            },
            it,
        );
        println!(
            "ic={ic:<4} {ih}x{iw}k{k}s{s}: OLD={mo:.2}ms NEW={mn:.2}ms speedup={:.2}x digest_ok={}",
            mo / mn,
            dgo == dgn
        );
    }
}

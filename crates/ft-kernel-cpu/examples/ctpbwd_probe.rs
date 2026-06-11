//! Isolated dinput A/B: OLD strided Σ_oc vs NEW channels-last, same process.
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
#[allow(clippy::too_many_arguments)]
fn old_di(
    dout: &[f64],
    weight: &[f64],
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
    let mut di = vec![0.0f64; batch * in_ch * ih * iw];
    di.par_chunks_mut(ih * iw)
        .enumerate()
        .for_each(|(idx, drow)| {
            let n = idx / in_ch;
            let ic = idx % in_ch;
            for iy in 0..ih {
                for ix in 0..iw {
                    let mut acc = 0.0;
                    for kr in 0..kh {
                        let oys = iy * sh + kr;
                        if oys < ph {
                            continue;
                        }
                        let oy = oys - ph;
                        if oy >= oh {
                            continue;
                        }
                        for kc in 0..kw {
                            let oxs = ix * sw + kc;
                            if oxs < pw {
                                continue;
                            }
                            let ox = oxs - pw;
                            if ox >= ow {
                                continue;
                            }
                            for oc in 0..out_ch {
                                acc += dout[((n * out_ch + oc) * oh + oy) * ow + ox]
                                    * weight[((ic * out_ch + oc) * kh + kr) * kw + kc];
                            }
                        }
                    }
                    drow[iy * iw + ix] = acc;
                }
            }
        });
    di
}
#[allow(clippy::too_many_arguments)]
fn new_di(
    dout: &[f64],
    weight: &[f64],
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
    let mut dcl = vec![0.0f64; batch * oh * ow * out_ch];
    dcl.par_chunks_mut(oh * ow * out_ch)
        .enumerate()
        .for_each(|(n, dst)| {
            for oc in 0..out_ch {
                let sb = (n * out_ch + oc) * oh * ow;
                for s in 0..oh * ow {
                    dst[s * out_ch + oc] = dout[sb + s];
                }
            }
        });
    let mut wcl = vec![0.0f64; in_ch * kh * kw * out_ch];
    for ic in 0..in_ch {
        for oc in 0..out_ch {
            for kr in 0..kh {
                for kc in 0..kw {
                    wcl[((ic * kh + kr) * kw + kc) * out_ch + oc] =
                        weight[((ic * out_ch + oc) * kh + kr) * kw + kc];
                }
            }
        }
    }
    let mut di = vec![0.0f64; batch * in_ch * ih * iw];
    di.par_chunks_mut(ih * iw)
        .enumerate()
        .for_each(|(idx, drow)| {
            let n = idx / in_ch;
            let ic = idx % in_ch;
            let nb = n * oh * ow * out_ch;
            let wb = ic * kh * kw * out_ch;
            for iy in 0..ih {
                for ix in 0..iw {
                    let mut acc = 0.0;
                    for kr in 0..kh {
                        let oys = iy * sh + kr;
                        if oys < ph {
                            continue;
                        }
                        let oy = oys - ph;
                        if oy >= oh {
                            continue;
                        }
                        for kc in 0..kw {
                            let oxs = ix * sw + kc;
                            if oxs < pw {
                                continue;
                            }
                            let ox = oxs - pw;
                            if ox >= ow {
                                continue;
                            }
                            let dv = &dcl[nb + (oy * ow + ox) * out_ch..][..out_ch];
                            let wv = &wcl[wb + (kr * kw + kc) * out_ch..][..out_ch];
                            for oc in 0..out_ch {
                                acc += dv[oc] * wv[oc];
                            }
                        }
                    }
                    drow[iy * iw + ix] = acc;
                }
            }
        });
    di
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
        let dout: Vec<f64> = (0..batch * oc * oh * ow)
            .map(|i| (i % 131) as f64 * 0.003 - 0.2)
            .collect();
        let wt: Vec<f64> = (0..ic * oc * kh * kw)
            .map(|i| (i % 97) as f64 * 0.002 - 0.1)
            .collect();
        let it = 8;
        let dgo = fnv(&old_di(
            &dout, &wt, batch, ic, ih, iw, oc, kh, kw, oh, ow, sh, sw, ph, pw,
        ));
        let dgn = fnv(&new_di(
            &dout, &wt, batch, ic, ih, iw, oc, kh, kw, oh, ow, sh, sw, ph, pw,
        ));
        let mo = t(
            || {
                let _ = old_di(
                    &dout, &wt, batch, ic, ih, iw, oc, kh, kw, oh, ow, sh, sw, ph, pw,
                );
            },
            it,
        );
        let mn = t(
            || {
                let _ = new_di(
                    &dout, &wt, batch, ic, ih, iw, oc, kh, kw, oh, ow, sh, sw, ph, pw,
                );
            },
            it,
        );
        println!(
            "ic={ic:<4}oc={oc:<4}{ih}x{iw}k{k}s{s}: OLD={mo:.2}ms NEW={mn:.2}ms speedup={:.2}x digest_ok={}",
            mo / mn,
            dgo == dgn
        );
    }
}

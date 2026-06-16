//! A/B for depthwise_conv2d_backward_f32: direct (1 call, parallel over N*C
//! planes) vs per-channel (C calls, channels=1 — a CONSERVATIVE proxy for the
//! per-group dispatch; the real old f32 grad path was the even-slower composed
//! unfold+matmul, so the realized win is larger). frankentorch-48w0b.
//!   cargo run -q --release -p ft-kernel-cpu --example depthwise_conv_backward_f32_ab
use ft_kernel_cpu::depthwise_conv2d_backward_f32;
use std::time::Instant;

fn main() {
    for &(n, c, ph, pw) in &[(16usize, 64usize, 58, 58), (8, 256, 30, 30)] {
        let (kh, kw, sh, sw) = (3usize, 3, 1, 1);
        let oh = (ph - kh) / sh + 1;
        let ow = (pw - kw) / sw + 1;
        let padded: Vec<f32> = (0..n * c * ph * pw).map(|i| (i % 877) as f32 * 0.01).collect();
        let weight: Vec<f32> = (0..c * kh * kw).map(|i| (i % 47) as f32 * 0.1 - 2.0).collect();
        let dout: Vec<f32> = (0..n * c * oh * ow).map(|i| (i % 53) as f32 * 0.05 - 1.3).collect();
        let inplane = ph * pw;
        let outplane = oh * ow;
        let wsz = kh * kw;

        let per_channel = || {
            let mut dp = vec![0.0f32; n * c * inplane];
            let mut dw = vec![0.0f32; c * wsz];
            for ci in 0..c {
                let mut dch = vec![0.0f32; n * outplane];
                let mut pch = vec![0.0f32; n * inplane];
                for ni in 0..n {
                    dch[ni * outplane..(ni + 1) * outplane].copy_from_slice(
                        &dout[(ni * c + ci) * outplane..(ni * c + ci) * outplane + outplane],
                    );
                    pch[ni * inplane..(ni + 1) * inplane].copy_from_slice(
                        &padded[(ni * c + ci) * inplane..(ni * c + ci) * inplane + inplane],
                    );
                }
                let w = &weight[ci * wsz..(ci + 1) * wsz];
                let (dpc, dwc, _) = depthwise_conv2d_backward_f32(
                    &dch, &pch, w, n, 1, ph, pw, kh, kw, oh, ow, sh, sw, false,
                );
                for ni in 0..n {
                    dp[(ni * c + ci) * inplane..(ni * c + ci) * inplane + inplane]
                        .copy_from_slice(&dpc[ni * inplane..(ni + 1) * inplane]);
                }
                dw[ci * wsz..(ci + 1) * wsz].copy_from_slice(&dwc);
            }
            (dp, dw)
        };
        per_channel();
        let mut bo = f64::INFINITY;
        for _ in 0..30 {
            let t = Instant::now();
            std::hint::black_box(per_channel());
            bo = bo.min(t.elapsed().as_secs_f64() * 1e3);
        }
        let mut bn = f64::INFINITY;
        for _ in 0..30 {
            let t = Instant::now();
            std::hint::black_box(depthwise_conv2d_backward_f32(
                &dout, &padded, &weight, n, c, ph, pw, kh, kw, oh, ow, sh, sw, false,
            ));
            bn = bn.min(t.elapsed().as_secs_f64() * 1e3);
        }
        eprintln!("[dw-bwd-f32 {n}x{c}x{ph}x{pw}] per-channel {bo:.2} ms / direct {bn:.2} ms / speedup {:.2}x", bo / bn);
    }
}

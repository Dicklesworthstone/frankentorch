//! A/B for depthwise-conv no-grad clone-before-kernel fixes (functional_conv2d_grouped/conv3d). The
//! no-grad path does `let iv = self.tensor_values(input)?` then (optionally pads and) calls the parallel
//! depthwise kernel on it. This measures the clone-vs-borrow delta at the kernel boundary for the
//! no-padding case (pv = iv): OLD = to_vec() clone + kernel, NEW = borrow + kernel.
//! Run PLAIN: cargo run --release -p ft-api --example conv_borrow_ab

use ft_kernel_cpu as k;
use std::time::Instant;

fn bench<F: FnMut() -> usize>(mut f: F) -> f64 {
    let mut best = f64::INFINITY;
    for _ in 0..9 {
        let t = Instant::now();
        let s = f();
        let el = t.elapsed().as_secs_f64() * 1e3;
        std::hint::black_box(s);
        if el < best {
            best = el;
        }
    }
    best
}

fn main() {
    println!("depthwise conv no-grad clone-vs-borrow (f64), min-9:  OLD=to_vec+kernel  NEW=borrow+kernel");
    bench_conv2d();
    bench_conv3d();
}

fn bench_conv2d() {
    println!("depthwise_conv2d:");
    // [B, C, H, W], 3x3 depthwise, stride 1, no padding -> out (H-2)x(W-2).
    let (kh, kw, sh, sw) = (3usize, 3, 1, 1);
    let cases: [(&str, usize, usize, usize, usize); 3] = [
        ("16x64x128x128", 16, 64, 128, 128),
        ("8x64x256x256", 8, 64, 256, 256),
        ("32x32x160x160", 32, 32, 160, 160),
    ];
    for (label, b, c, ph, pw) in cases {
        let oh = (ph - kh) / sh + 1;
        let ow = (pw - kw) / sw + 1;
        let input: Vec<f64> = (0..b * c * ph * pw).map(|i| ((i % 1021) as f64 - 510.0) * 0.01).collect();
        let weight: Vec<f64> = (0..c * kh * kw).map(|i| ((i % 17) as f64 - 8.0) * 0.05).collect();

        let new_out = k::depthwise_conv2d_forward_f64(&input, &weight, None, b, c, ph, pw, kh, kw, oh, ow, sh, sw);
        let old_out = {
            let iv = input.to_vec();
            k::depthwise_conv2d_forward_f64(&iv, &weight, None, b, c, ph, pw, kh, kw, oh, ow, sh, sw)
        };
        let bitmatch = new_out == old_out;

        let old_ms = bench(|| {
            let iv = input.to_vec();
            k::depthwise_conv2d_forward_f64(&iv, &weight, None, b, c, ph, pw, kh, kw, oh, ow, sh, sw).len()
        });
        let new_ms = bench(|| {
            k::depthwise_conv2d_forward_f64(&input, &weight, None, b, c, ph, pw, kh, kw, oh, ow, sh, sw).len()
        });
        println!(
            "  {label:<16} ({:>4}MB in)  OLD {:8.3}  NEW {:8.3}  = {:.2}x  bitmatch={}",
            b * c * ph * pw * 8 / (1 << 20),
            old_ms,
            new_ms,
            old_ms / new_ms,
            bitmatch
        );
    }
}

fn bench_conv3d() {
    println!("depthwise_conv3d:");
    // [B, C, D, H, W], 1x1x1 depthwise, stride 1, no padding. This isolates the
    // full-input clone in functional_conv3d_dilated's no-grad F64 depthwise path.
    let (kd, kh, kw, sd, sh, sw) = (1usize, 1, 1, 1, 1, 1);
    let cases: [(&str, usize, usize, usize, usize, usize); 3] = [
        ("2x32x32x64x64", 2, 32, 32, 64, 64),
        ("4x32x24x64x64", 4, 32, 24, 64, 64),
        ("2x64x16x96x64", 2, 64, 16, 96, 64),
    ];
    for (label, b, c, pd, ph, pw) in cases {
        let od = (pd - kd) / sd + 1;
        let oh = (ph - kh) / sh + 1;
        let ow = (pw - kw) / sw + 1;
        let input: Vec<f64> = (0..b * c * pd * ph * pw).map(|i| ((i % 2053) as f64 - 1026.0) * 0.005).collect();
        let weight: Vec<f64> = (0..c * kd * kh * kw).map(|i| ((i % 23) as f64 - 11.0) * 0.04).collect();

        let new_out = k::depthwise_conv3d_forward_f64(&input, &weight, None, b, c, pd, ph, pw, kd, kh, kw, od, oh, ow, sd, sh, sw);
        let old_out = {
            let iv = input.to_vec();
            k::depthwise_conv3d_forward_f64(&iv, &weight, None, b, c, pd, ph, pw, kd, kh, kw, od, oh, ow, sd, sh, sw)
        };
        let bitmatch = new_out == old_out;

        let old_ms = bench(|| {
            let iv = input.to_vec();
            k::depthwise_conv3d_forward_f64(&iv, &weight, None, b, c, pd, ph, pw, kd, kh, kw, od, oh, ow, sd, sh, sw).len()
        });
        let new_ms = bench(|| {
            k::depthwise_conv3d_forward_f64(&input, &weight, None, b, c, pd, ph, pw, kd, kh, kw, od, oh, ow, sd, sh, sw).len()
        });
        println!(
            "  {label:<16} ({:>4}MB in)  OLD {:8.3}  NEW {:8.3}  = {:.2}x  bitmatch={}",
            b * c * pd * ph * pw * 8 / (1 << 20),
            old_ms,
            new_ms,
            old_ms / new_ms,
            bitmatch
        );
    }
}

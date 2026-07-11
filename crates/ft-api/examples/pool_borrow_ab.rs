//! A/B for the pool no-grad clone-before-kernel fix. The no-grad max_pool2d fast path did
//! `let iv = self.tensor_values(input)?` (a serial zero-faulted numel*8B clone of the FULL input)
//! then called the parallel windowed-max kernel on &iv. The kernel only READS the input, so the fix
//! borrows the contiguous storage (contiguous_values()), eliminating the clone. This example measures
//! the exact delta at the kernel boundary: OLD = to_vec() clone + kernel (models tensor_values), NEW =
//! borrow + kernel. Same lever as layer_norm/rms_norm (2.6-5.4x). Run plain:
//! cargo run --release -p ft-api --example pool_borrow_ab

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
    println!(
        "max_pool2d no-grad clone-vs-borrow (f64), min-9:  OLD=to_vec clone+kernel  NEW=borrow+kernel"
    );
    // [B, C, H, W] large inputs; 2x2 stride-2 pooling -> quarter output.
    let cases: [(&str, usize, usize, usize, usize); 3] = [
        ("16x64x128x128", 16, 64, 128, 128),
        ("8x64x256x256", 8, 64, 256, 256),
        ("32x32x128x128", 32, 32, 128, 128),
    ];
    let (kh, kw, sh, sw) = (2usize, 2, 2, 2);
    for (label, b, c, ih, iw) in cases {
        let oh = (ih - kh) / sh + 1;
        let ow = (iw - kw) / sw + 1;
        let input: Vec<f64> = (0..b * c * ih * iw)
            .map(|i| ((i % 1021) as f64 - 510.0) * 0.01)
            .collect();

        let new_out = k::max_pool2d_forward_f64(&input, b, c, ih, iw, kh, kw, oh, ow, sh, sw);
        let old_out = {
            let iv = input.to_vec();
            k::max_pool2d_forward_f64(&iv, b, c, ih, iw, kh, kw, oh, ow, sh, sw)
        };
        let bitmatch = new_out == old_out;

        let old_ms = bench(|| {
            let iv = input.to_vec();
            k::max_pool2d_forward_f64(&iv, b, c, ih, iw, kh, kw, oh, ow, sh, sw).len()
        });
        let new_ms =
            bench(|| k::max_pool2d_forward_f64(&input, b, c, ih, iw, kh, kw, oh, ow, sh, sw).len());
        println!(
            "  {label:<16} ({:>4}MB in)  OLD {:8.3}  NEW {:8.3}  = {:.2}x  bitmatch={}",
            b * c * ih * iw * 8 / (1 << 20),
            old_ms,
            new_ms,
            old_ms / new_ms,
            bitmatch
        );
    }
}

//! Non-power-of-two FFT: same-binary A/B of the OLD naive O(N^2) DFT vs the
//! NEW Bluestein chirp-z O(N log N) path (`tensor_fft`). Both run in one bench
//! binary, so the ratio is immune to cross-worker speed variance:
//!   cargo bench -p ft-api --bench fft_bench
//! `fft_naive/N` reproduces the exact pre-change inner loop; `fft_bluestein/N`
//! drives the production `tensor_fft` (which now routes non-power-of-two N
//! through Bluestein). Score = naive_time / bluestein_time.

use criterion::{Criterion, black_box, criterion_group, criterion_main};
use ft_api::FrankenTorchSession;
use ft_core::ExecutionMode;

/// The pre-change naive O(N^2) DFT, verbatim, as the A/B baseline.
fn naive_dft(re: &mut [f64], im: &mut [f64]) {
    let n = re.len();
    let mut out_re = vec![0.0_f64; n];
    let mut out_im = vec![0.0_f64; n];
    let two_pi = 2.0 * std::f64::consts::PI;
    for k in 0..n {
        let mut acc_re = 0.0_f64;
        let mut acc_im = 0.0_f64;
        for nn in 0..n {
            let angle = -two_pi * k as f64 * nn as f64 / n as f64;
            let c = angle.cos();
            let s = angle.sin();
            acc_re += re[nn] * c - im[nn] * s;
            acc_im += re[nn] * s + im[nn] * c;
        }
        out_re[k] = acc_re;
        out_im[k] = acc_im;
    }
    re.copy_from_slice(&out_re);
    im.copy_from_slice(&out_im);
}

fn bench_fft(c: &mut Criterion) {
    // Non-power-of-two sizes spanning audio/DSP ranges; 1009 is prime.
    for &n in &[1000usize, 1009, 3000] {
        let signal: Vec<f64> = (0..n).map(|i| (i as f64 * 0.013).sin() + 0.5).collect();

        c.bench_function(&format!("fft_naive/{n}"), |b| {
            b.iter(|| {
                let mut re = signal.clone();
                let mut im = vec![0.0_f64; n];
                naive_dft(black_box(&mut re), black_box(&mut im));
                black_box((re, im));
            });
        });

        c.bench_function(&format!("fft_bluestein/{n}"), |b| {
            let mut session = FrankenTorchSession::new(ExecutionMode::Strict);
            let x = session.tensor_variable(signal.clone(), vec![n], false).unwrap();
            b.iter(|| black_box(session.tensor_fft(black_box(x), None).unwrap()));
        });
    }
}

criterion_group!(benches, bench_fft);
criterion_main!(benches);

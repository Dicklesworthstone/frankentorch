use criterion::{BenchmarkId, Criterion, Throughput, black_box, criterion_group, criterion_main};
use ft_api::FrankenTorchSession;
use ft_core::ExecutionMode;

fn bench_matmul(c: &mut Criterion) {
    let mut group = c.benchmark_group("matmul");

    for size in [64, 128, 256, 512, 1024].iter() {
        let n = *size;
        group.throughput(Throughput::Elements((n * n * n) as u64));
        group.bench_with_input(BenchmarkId::new("square", n), &n, |b, &n| {
            let mut session = FrankenTorchSession::new(ExecutionMode::Strict);
            let a = session.tensor_randn(vec![n, n], false).unwrap();
            let bt = session.tensor_randn(vec![n, n], false).unwrap();
            b.iter(|| black_box(session.tensor_matmul(a, bt).unwrap()));
        });
    }
    group.finish();
}

fn bench_bmm(c: &mut Criterion) {
    let mut group = c.benchmark_group("bmm");

    for batch in [8, 16, 32].iter() {
        let b = *batch;
        let n = 128;
        group.throughput(Throughput::Elements((b * n * n * n) as u64));
        group.bench_with_input(BenchmarkId::new("batch", b), &b, |bencher, &b| {
            let mut session = FrankenTorchSession::new(ExecutionMode::Strict);
            let a = session.tensor_randn(vec![b, n, n], false).unwrap();
            let bt = session.tensor_randn(vec![b, n, n], false).unwrap();
            bencher.iter(|| black_box(session.tensor_bmm(a, bt).unwrap()));
        });
    }
    group.finish();
}

fn bench_conv2d(c: &mut Criterion) {
    let mut group = c.benchmark_group("conv2d");

    for hw in [32, 64, 128].iter() {
        let h = *hw;
        let batch = 4;
        let in_ch = 64;
        let out_ch = 64;
        let kh = 3;
        let kw = 3;

        group.throughput(Throughput::Elements((batch * out_ch * h * h) as u64));
        group.bench_with_input(BenchmarkId::new("hw", h), &h, |b, &h| {
            let mut session = FrankenTorchSession::new(ExecutionMode::Strict);
            let input = session
                .tensor_randn(vec![batch, in_ch, h, h], false)
                .unwrap();
            let weight = session
                .tensor_randn(vec![out_ch, in_ch, kh, kw], false)
                .unwrap();
            b.iter(|| {
                black_box(
                    session
                        .tensor_conv2d(input, weight, None, (1, 1), (1, 1))
                        .unwrap(),
                )
            });
        });
    }
    group.finish();
}

fn bench_sum(c: &mut Criterion) {
    let mut group = c.benchmark_group("sum");

    for size in [1000, 10000, 100000, 1000000].iter() {
        let n = *size;
        group.throughput(Throughput::Elements(n as u64));
        group.bench_with_input(BenchmarkId::new("elements", n), &n, |b, &n| {
            let mut session = FrankenTorchSession::new(ExecutionMode::Strict);
            let x = session.tensor_randn(vec![n], false).unwrap();
            b.iter(|| black_box(session.tensor_sum(x).unwrap()));
        });
    }
    group.finish();
}

fn bench_softmax(c: &mut Criterion) {
    let mut group = c.benchmark_group("softmax");

    for size in [128, 512, 2048, 8192].iter() {
        let n = *size;
        let batch = 32;
        group.throughput(Throughput::Elements((batch * n) as u64));
        group.bench_with_input(BenchmarkId::new("vocab", n), &n, |b, &n| {
            let mut session = FrankenTorchSession::new(ExecutionMode::Strict);
            let x = session.tensor_randn(vec![batch, n], false).unwrap();
            b.iter(|| black_box(session.tensor_softmax(x, 1).unwrap()));
        });
    }
    group.finish();
}

fn bench_relu(c: &mut Criterion) {
    let mut group = c.benchmark_group("relu");

    for size in [10000, 100000, 1000000].iter() {
        let n = *size;
        group.throughput(Throughput::Elements(n as u64));
        group.bench_with_input(BenchmarkId::new("elements", n), &n, |b, &n| {
            let mut session = FrankenTorchSession::new(ExecutionMode::Strict);
            let x = session.tensor_randn(vec![n], false).unwrap();
            b.iter(|| black_box(session.tensor_relu(x).unwrap()));
        });
    }
    group.finish();
}

fn bench_exp(c: &mut Criterion) {
    let mut group = c.benchmark_group("exp");

    for size in [10000, 100000, 1000000].iter() {
        let n = *size;
        group.throughput(Throughput::Elements(n as u64));
        group.bench_with_input(BenchmarkId::new("elements", n), &n, |b, &n| {
            let mut session = FrankenTorchSession::new(ExecutionMode::Strict);
            let x = session.tensor_randn(vec![n], false).unwrap();
            b.iter(|| black_box(session.tensor_exp(x).unwrap()));
        });
    }
    group.finish();
}

fn bench_add(c: &mut Criterion) {
    let mut group = c.benchmark_group("add");

    for size in [10000, 100000, 1000000].iter() {
        let n = *size;
        group.throughput(Throughput::Elements(n as u64));
        group.bench_with_input(BenchmarkId::new("elements", n), &n, |b, &n| {
            let mut session = FrankenTorchSession::new(ExecutionMode::Strict);
            let x = session.tensor_randn(vec![n], false).unwrap();
            let y = session.tensor_randn(vec![n], false).unwrap();
            b.iter(|| black_box(session.tensor_add(x, y).unwrap()));
        });
    }
    group.finish();
}

fn bench_backward_matmul(c: &mut Criterion) {
    let mut group = c.benchmark_group("backward_matmul");

    for size in [64, 128, 256].iter() {
        let n = *size;
        group.throughput(Throughput::Elements((n * n) as u64));
        group.bench_with_input(BenchmarkId::new("size", n), &n, |b, &n| {
            b.iter(|| {
                let mut session = FrankenTorchSession::new(ExecutionMode::Strict);
                let a = session.tensor_randn(vec![n, n], true).unwrap();
                let bt = session.tensor_randn(vec![n, n], true).unwrap();
                let c = session.tensor_matmul(a, bt).unwrap();
                let loss = session.tensor_sum(c).unwrap();
                black_box(session.tensor_backward(loss).unwrap())
            });
        });
    }
    group.finish();
}

fn bench_linear_forward(c: &mut Criterion) {
    let mut group = c.benchmark_group("linear_forward");

    for hidden in [256, 512, 1024, 2048].iter() {
        let h = *hidden;
        let batch = 32;
        let in_features = 512;
        group.throughput(Throughput::Elements((batch * h) as u64));
        group.bench_with_input(BenchmarkId::new("hidden", h), &h, |b, &h| {
            let mut session = FrankenTorchSession::new(ExecutionMode::Strict);
            let x = session
                .tensor_randn(vec![batch, in_features], false)
                .unwrap();
            let w = session.tensor_randn(vec![h, in_features], false).unwrap();
            let bias = session.tensor_randn(vec![h], false).unwrap();
            b.iter(|| black_box(session.tensor_linear(x, w, Some(bias)).unwrap()));
        });
    }
    group.finish();
}

fn bench_grid_sample(c: &mut Criterion) {
    use ft_api::{GridSampleMode, GridSamplePaddingMode};
    let mut group = c.benchmark_group("grid_sample");
    // [N, C, H, W] input + [N, H, W, 2] grid -> bilinear sample. The 4 scattered
    // bilinear gathers make this memory-bandwidth-bound (row-parallelizing it only
    // reached ~1.3x, rejected under frankentorch-kgs4.10); kept to track the hotspot.
    let (n, ch, h, w) = (8usize, 32usize, 64usize, 64usize);
    group.throughput(Throughput::Elements((n * ch * h * w) as u64));
    group.bench_function("8x32x64x64_bilinear", |b| {
        let mut session = FrankenTorchSession::new(ExecutionMode::Strict);
        let input = session.tensor_randn(vec![n, ch, h, w], false).unwrap();
        // Grid values in roughly [-1, 1] via tanh-free scaling of randn.
        let grid = session.tensor_rand(vec![n, h, w, 2], false).unwrap();
        b.iter(|| {
            black_box(
                session
                    .tensor_grid_sample(
                        black_box(input),
                        black_box(grid),
                        GridSampleMode::Bilinear,
                        GridSamplePaddingMode::Zeros,
                        false,
                    )
                    .unwrap(),
            )
        });
    });
    group.finish();
}

fn bench_fft2(c: &mut Criterion) {
    let mut group = c.benchmark_group("fft2");
    // Batched 2-D FFT [batch, rows, cols]: the row pass (per-row 1-D FFT) and the
    // column pass (per-column 1-D FFT) are both compute-bound trig and parallel
    // over rows / batch planes respectively.
    let (batch, rows, cols) = (32usize, 128usize, 128usize);
    group.throughput(Throughput::Elements((batch * rows * cols) as u64));
    group.bench_function("32x128x128", |b| {
        let mut session = FrankenTorchSession::new(ExecutionMode::Strict);
        let x = session.tensor_randn(vec![batch, rows, cols], false).unwrap();
        b.iter(|| black_box(session.tensor_fft2(black_box(x)).unwrap()));
    });
    group.finish();
}

fn bench_lrn(c: &mut Criterion) {
    let mut group = c.benchmark_group("local_response_norm");
    // [N, C, H, W] LRN: a powf per output element over a local channel window ->
    // compute-bound, parallel over (batch, channel) rows.
    let (n, ch, h, w) = (8usize, 64usize, 56usize, 56usize);
    group.throughput(Throughput::Elements((n * ch * h * w) as u64));
    group.bench_function("8x64x56x56", |b| {
        let mut session = FrankenTorchSession::new(ExecutionMode::Strict);
        let x = session.tensor_randn(vec![n, ch, h, w], false).unwrap();
        b.iter(|| {
            black_box(
                session
                    .tensor_local_response_norm(black_box(x), 5, 1e-4, 0.75, 2.0)
                    .unwrap(),
            )
        });
    });
    group.finish();
}

fn bench_rope_freqs(c: &mut Criterion) {
    let mut group = c.benchmark_group("rope_freqs");
    // RoPE cos/sin tables [max_seq_len, head_dim/2]: cos + sin per element ->
    // compute-bound, parallel over positions.
    let (max_seq_len, head_dim) = (32768usize, 128usize);
    group.throughput(Throughput::Elements((max_seq_len * head_dim / 2) as u64));
    group.bench_function("32768x128", |b| {
        let mut session = FrankenTorchSession::new(ExecutionMode::Strict);
        b.iter(|| {
            black_box(
                session
                    .precompute_rope_freqs(black_box(max_seq_len), black_box(head_dim), 10000.0)
                    .unwrap(),
            )
        });
    });
    group.finish();
}

fn bench_sinusoidal_pe(c: &mut Criterion) {
    let mut group = c.benchmark_group("sinusoidal_pe");
    // [seq_len, d_model] sinusoidal positional encoding: powf + sin/cos per
    // element -> compute-bound, parallel over positions.
    let (seq_len, d_model) = (4096usize, 1024usize);
    group.throughput(Throughput::Elements((seq_len * d_model) as u64));
    group.bench_function("4096x1024", |b| {
        let mut session = FrankenTorchSession::new(ExecutionMode::Strict);
        b.iter(|| {
            black_box(
                session
                    .sinusoidal_position_encoding(black_box(seq_len), black_box(d_model))
                    .unwrap(),
            )
        });
    });
    group.finish();
}

fn bench_istft(c: &mut Criterion) {
    use ft_api::IstftOptions;
    let mut group = c.benchmark_group("istft");
    // Complex spectrogram [n_fft/2+1, frames] -> inverse STFT. The per-frame
    // inverse pass is a naive O(n_fft^2) DFT (cos/sin per term) = compute-bound,
    // parallel over frames; the overlap-add is a cheap serial reduction.
    let (n_fft, frames) = (512usize, 256usize);
    let freq_bins = n_fft / 2 + 1;
    group.throughput(Throughput::Elements((freq_bins * frames) as u64));
    group.bench_function("nfft512_frames256", |b| {
        let mut session = FrankenTorchSession::new(ExecutionMode::Strict);
        let re = session.tensor_randn(vec![freq_bins, frames], false).unwrap();
        let im = session.tensor_randn(vec![freq_bins, frames], false).unwrap();
        let spec = session.tensor_complex(re, im).unwrap();
        b.iter(|| {
            black_box(
                session
                    .tensor_istft(
                        black_box(spec),
                        512,
                        IstftOptions {
                            hop_length: Some(128),
                            ..IstftOptions::default()
                        },
                    )
                    .unwrap(),
            )
        });
    });
    group.finish();
}

fn bench_stft(c: &mut Criterion) {
    use ft_api::StftOptions;
    let mut group = c.benchmark_group("stft");
    // 1-D signal -> STFT with a naive per-frame DFT (cos/sin per term) =
    // compute-bound, parallel over freq-bin rows. n_fft=512, hop=128.
    let len = 32768usize;
    group.throughput(Throughput::Elements(len as u64));
    group.bench_function("len32768_nfft512", |b| {
        let mut session = FrankenTorchSession::new(ExecutionMode::Strict);
        let x = session.tensor_randn(vec![len], false).unwrap();
        b.iter(|| {
            black_box(
                session
                    .tensor_stft(
                        black_box(x),
                        512,
                        StftOptions {
                            hop_length: Some(128),
                            ..StftOptions::default()
                        },
                    )
                    .unwrap(),
            )
        });
    });
    group.finish();
}

fn bench_irfft2(c: &mut Criterion) {
    let mut group = c.benchmark_group("irfft2");
    // Inverse 2-D real FFT from a complex half-spectrum [batch, rows, cols/2+1]
    // -> real [batch, rows, cols]. Same row+col compute-bound passes.
    let (batch, rows, in_cols) = (32usize, 128usize, 65usize); // out_cols = 128
    group.throughput(Throughput::Elements((batch * rows * (in_cols - 1) * 2) as u64));
    group.bench_function("32x128x128", |b| {
        let mut session = FrankenTorchSession::new(ExecutionMode::Strict);
        let re = session.tensor_randn(vec![batch, rows, in_cols], false).unwrap();
        let im = session.tensor_randn(vec![batch, rows, in_cols], false).unwrap();
        let cplx = session.tensor_complex(re, im).unwrap();
        b.iter(|| black_box(session.tensor_irfft2(black_box(cplx), None).unwrap()));
    });
    group.finish();
}

fn bench_rfft2(c: &mut Criterion) {
    let mut group = c.benchmark_group("rfft2");
    // Batched 2-D real FFT: same row+col compute-bound passes as fft2.
    let (batch, rows, cols) = (32usize, 128usize, 128usize);
    group.throughput(Throughput::Elements((batch * rows * cols) as u64));
    group.bench_function("32x128x128", |b| {
        let mut session = FrankenTorchSession::new(ExecutionMode::Strict);
        let x = session.tensor_randn(vec![batch, rows, cols], false).unwrap();
        b.iter(|| black_box(session.tensor_rfft2(black_box(x)).unwrap()));
    });
    group.finish();
}

fn bench_fft_1d(c: &mut Criterion) {
    let mut group = c.benchmark_group("fft_1d");
    // Single large power-of-two 1-D FFT: isolates dft_inplace_1d (one lane, no
    // lane parallelism) so the twiddle-precompute speedup is measured directly.
    let n = 1usize << 18; // 262144-pt
    group.throughput(Throughput::Elements(n as u64));
    group.bench_function("262144pt", |b| {
        let mut session = FrankenTorchSession::new(ExecutionMode::Strict);
        let x = session.tensor_randn(vec![n], false).unwrap();
        b.iter(|| black_box(session.tensor_fft(black_box(x), None).unwrap()));
    });
    group.finish();
}

fn bench_fftn(c: &mut Criterion) {
    let mut group = c.benchmark_group("fftn");
    // [lanes, N] real input, FFT along the last dim: `lanes` independent
    // O(N log N) Cooley-Tukey transforms (trig butterflies) -> compute-bound,
    // parallel over lanes. 512 lanes x 2048-pt.
    let (lanes, n) = (512usize, 2048usize);
    group.throughput(Throughput::Elements((lanes * n) as u64));
    group.bench_function("512x2048_dim1", |b| {
        let mut session = FrankenTorchSession::new(ExecutionMode::Strict);
        let x = session.tensor_randn(vec![lanes, n], false).unwrap();
        b.iter(|| black_box(session.tensor_fftn(black_box(x), None, Some(&[1])).unwrap()));
    });
    group.finish();
}

fn bench_interpolate_trilinear(c: &mut Criterion) {
    let mut group = c.benchmark_group("interpolate_trilinear");
    // [N, C, D, H, W] -> 2x upsample. Trilinear blends 8 local corner taps with
    // ~24 mults/output: compute-bound with cache-friendly local access, parallel
    // over output rows.
    let (n, ch, d, h, w) = (2usize, 8usize, 16usize, 16usize, 16usize);
    group.throughput(Throughput::Elements((n * ch * d * 2 * h * 2 * w * 2) as u64));
    group.bench_function("2x8x16x16x16_2x", |b| {
        let mut session = FrankenTorchSession::new(ExecutionMode::Strict);
        let x = session.tensor_randn(vec![n, ch, d, h, w], false).unwrap();
        b.iter(|| {
            black_box(
                session
                    .tensor_interpolate(
                        black_box(x),
                        Some(vec![d * 2, h * 2, w * 2]),
                        None,
                        "trilinear",
                        Some(false),
                    )
                    .unwrap(),
            )
        });
    });
    group.finish();
}

fn bench_interpolate_bicubic(c: &mut Criterion) {
    let mut group = c.benchmark_group("interpolate_bicubic");
    // [N, C, H, W] -> 2x upsample. Bicubic does 16 cubic-weight taps per output
    // element, so it is compute-bound and parallelizes over output rows.
    let (n, ch, h, w) = (8usize, 32usize, 64usize, 64usize);
    group.throughput(Throughput::Elements((n * ch * h * 2 * w * 2) as u64));
    group.bench_function("8x32x64x64_2x", |b| {
        let mut session = FrankenTorchSession::new(ExecutionMode::Strict);
        let x = session.tensor_randn(vec![n, ch, h, w], false).unwrap();
        b.iter(|| {
            black_box(
                session
                    .tensor_interpolate(
                        black_box(x),
                        Some(vec![h * 2, w * 2]),
                        None,
                        "bicubic",
                        Some(false),
                    )
                    .unwrap(),
            )
        });
    });
    group.finish();
}

criterion_group!(
    benches,
    bench_matmul,
    bench_bmm,
    bench_conv2d,
    bench_sum,
    bench_softmax,
    bench_relu,
    bench_exp,
    bench_add,
    bench_backward_matmul,
    bench_linear_forward,
    bench_interpolate_bicubic,
    bench_interpolate_trilinear,
    bench_grid_sample,
    bench_fft2,
    bench_lrn,
    bench_rope_freqs,
    bench_sinusoidal_pe,
    bench_istft,
    bench_stft,
    bench_irfft2,
    bench_rfft2,
    bench_fft_1d,
    bench_fftn,
);
criterion_main!(benches);

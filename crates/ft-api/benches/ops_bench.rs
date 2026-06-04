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
    // Small-M / large-N (a linear/projection layer's GEMM): M too small to
    // row-split, so the column-parallel dgemm path is the lever.
    for &nn in &[2048usize, 4096usize] {
        group.bench_with_input(BenchmarkId::new("wide_64x512", nn), &nn, |b, &nn| {
            let mut session = FrankenTorchSession::new(ExecutionMode::Strict);
            let a = session.tensor_randn(vec![64, 512], false).unwrap();
            let bt = session.tensor_randn(vec![512, nn], false).unwrap();
            b.iter(|| black_box(session.tensor_matmul(a, bt).unwrap()));
        });
    }
    // Tall-M / small-N (an attention/linear projection [batch*S, embed] @
    // [embed, embed] at large S): below the total-flops gate but splits into many
    // row blocks, so the per-row-block parallel gate is the lever.
    for &mm in &[4096usize, 8192usize] {
        group.bench_with_input(BenchmarkId::new("tall_x128x128", mm), &mm, |b, &mm| {
            let mut session = FrankenTorchSession::new(ExecutionMode::Strict);
            let a = session.tensor_randn(vec![mm, 128], false).unwrap();
            let bt = session.tensor_randn(vec![128, 128], false).unwrap();
            b.iter(|| black_box(session.tensor_matmul(a, bt).unwrap()));
        });
    }
    // f32 wide (the common ML dtype): exercises the sgemm column-parallel path.
    for &nn in &[2048usize, 4096usize] {
        group.bench_with_input(BenchmarkId::new("wide_f32_64x512", nn), &nn, |b, &nn| {
            let mut session = FrankenTorchSession::new(ExecutionMode::Strict);
            let a = session.randn_f32(vec![64, 512], false).unwrap();
            let bt = session.randn_f32(vec![512, nn], false).unwrap();
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

fn bench_sigmoid(c: &mut Criterion) {
    let mut group = c.benchmark_group("sigmoid");

    // Logistic sigmoid `1/(1+exp(-x))` — exercises the vectorised `exp_f64x4`
    // transcendental path (serial-SIMD below the parallel threshold, parallel-SIMD
    // above) versus the prior scalar-libm `unary_f64` map.
    for size in [10000, 100000, 1000000].iter() {
        let n = *size;
        group.throughput(Throughput::Elements(n as u64));
        group.bench_with_input(BenchmarkId::new("elements", n), &n, |b, &n| {
            let mut session = FrankenTorchSession::new(ExecutionMode::Strict);
            let x = session.tensor_randn(vec![n], false).unwrap();
            b.iter(|| black_box(session.tensor_sigmoid(x).unwrap()));
        });
    }
    group.finish();
}

fn bench_pow(c: &mut Criterion) {
    let mut group = c.benchmark_group("pow");

    // Non-integer exponent forces the per-element `powf` path (~exp+log each),
    // the transcendental scalar map gated by the parallel threshold.
    for size in [10000, 100000, 1000000].iter() {
        let n = *size;
        group.throughput(Throughput::Elements(n as u64));
        group.bench_with_input(BenchmarkId::new("elements", n), &n, |b, &n| {
            let mut session = FrankenTorchSession::new(ExecutionMode::Strict);
            let x = session.tensor_randn(vec![n], false).unwrap();
            b.iter(|| black_box(session.tensor_pow(x, 2.5).unwrap()));
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
    // f32 (the common ML dtype): exercises the sgemm_bt fused-linear path.
    for hidden in [1024, 2048].iter() {
        let h = *hidden;
        let batch = 32;
        let in_features = 512;
        group.bench_with_input(BenchmarkId::new("f32_hidden", h), &h, |b, &h| {
            let mut session = FrankenTorchSession::new(ExecutionMode::Strict);
            let x = session.randn_f32(vec![batch, in_features], false).unwrap();
            let w = session.randn_f32(vec![h, in_features], false).unwrap();
            let bias = session.randn_f32(vec![h], false).unwrap();
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

fn bench_vander(c: &mut Criterion) {
    let mut group = c.benchmark_group("vander");
    // [rows] -> [rows, cols] Vandermonde: x[i]^exp_j (powi) per element. powi is
    // cheap and the output is memory-bound, so row-parallelizing it REGRESSED
    // (3.6->7.1ms, rejected under frankentorch-kgs4.30); kept to track the spot.
    let (rows, cols) = (2048usize, 256usize);
    group.throughput(Throughput::Elements((rows * cols) as u64));
    group.bench_function("2048x256", |b| {
        let mut session = FrankenTorchSession::new(ExecutionMode::Strict);
        let x = session.tensor_rand(vec![rows], false).unwrap();
        b.iter(|| black_box(session.tensor_vander(black_box(x), Some(cols), true).unwrap()));
    });
    group.finish();
}

fn bench_matrix_nms(c: &mut Criterion) {
    let mut group = c.benchmark_group("matrix_nms");
    // [K] scores + [K, H, W] masks -> K x K IoU matrix via a naive
    // O(K^2 * H*W) pairwise loop (compute-bound), parallel over rows.
    let (num, h, w) = (256usize, 48usize, 48usize);
    group.throughput(Throughput::Elements((num * num * h * w) as u64));
    group.bench_function("256x48x48", |b| {
        let mut session = FrankenTorchSession::new(ExecutionMode::Strict);
        let scores = session.tensor_rand(vec![num], false).unwrap();
        let masks = session.tensor_rand(vec![num, h, w], false).unwrap();
        b.iter(|| {
            black_box(
                session
                    .matrix_nms(black_box(scores), black_box(masks), 2.0, num, 100)
                    .unwrap(),
            )
        });
    });
    group.finish();
}

fn bench_knn_search(c: &mut Criterion) {
    let mut group = c.benchmark_group("knn_search");
    // [N,3] points + [M,3] queries, k=8. The current implementation computes
    // every squared distance and fully sorts all N distances per query even
    // though only the first k entries are returned.
    let (n_points, n_queries, k) = (8192usize, 512usize, 8usize);
    group.throughput(Throughput::Elements((n_points * n_queries) as u64));
    group.bench_function("8192x512_k8", |b| {
        let mut session = FrankenTorchSession::new(ExecutionMode::Strict);
        let points = (0..n_points * 3)
            .map(|idx| ((idx * 17 + 13) % 1021) as f64 * 0.001)
            .collect::<Vec<_>>();
        let queries = (0..n_queries * 3)
            .map(|idx| ((idx * 31 + 7) % 997) as f64 * 0.001)
            .collect::<Vec<_>>();
        let points = session
            .tensor_variable(points, vec![n_points, 3], false)
            .unwrap();
        let queries = session
            .tensor_variable(queries, vec![n_queries, 3], false)
            .unwrap();
        b.iter(|| {
            let (indices, distances) = session
                .knn_search(black_box(points), black_box(queries), black_box(k))
                .unwrap();
            black_box((indices, distances))
        });
    });
    group.finish();
}

fn bench_supcon_loss(c: &mut Criterion) {
    let mut group = c.benchmark_group("supcon_loss");
    // [N, D] embeddings -> N x N similarity matrix via a naive O(N^2 * D) dot
    // product loop (compute-bound), parallel over rows.
    let (n, d) = (512usize, 512usize);
    group.throughput(Throughput::Elements((n * n * d) as u64));
    group.bench_function("512x512", |b| {
        let mut session = FrankenTorchSession::new(ExecutionMode::Strict);
        let emb = session.tensor_randn(vec![n, d], false).unwrap();
        let labels_v: Vec<f64> = (0..n).map(|i| (i % 8) as f64).collect();
        let labels = session.tensor_variable(labels_v, vec![n], false).unwrap();
        b.iter(|| black_box(session.supcon_loss(black_box(emb), black_box(labels), 0.07).unwrap()));
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
    bench_sigmoid,
    bench_pow,
    bench_add,
    bench_backward_matmul,
    bench_linear_forward,
    bench_interpolate_bicubic,
    bench_interpolate_trilinear,
    bench_grid_sample,
    bench_fft2,
    bench_vander,
    bench_matrix_nms,
    bench_knn_search,
    bench_supcon_loss,
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

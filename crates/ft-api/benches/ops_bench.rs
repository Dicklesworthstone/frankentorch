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

fn bench_max_pool3d(c: &mut Criterion) {
    // max_pool3d (video/volumetric): [N,C,D,H,W]=[2,32,16,32,32], 2x2x2 stride 2.
    let mut group = c.benchmark_group("max_pool3d");
    let (n, ch, d, h, w) = (2usize, 32usize, 16usize, 32usize, 32usize);
    group.bench_function("nograd", |b| {
        let mut s = FrankenTorchSession::new(ExecutionMode::Strict);
        let x = s.tensor_randn(vec![n, ch, d, h, w], false).unwrap();
        b.iter(|| black_box(s.functional_max_pool3d(x, (2, 2, 2), (2, 2, 2)).unwrap()));
    });
    // F32 no-grad (dominant ML dtype): fused max_pool3d_forward_f32 vs the f32
    // op-graph (narrow/amax/cat) it fell through to.
    group.bench_function("nograd_f32", |b| {
        let mut s = FrankenTorchSession::new(ExecutionMode::Strict);
        let xv: Vec<f32> = (0..n * ch * d * h * w)
            .map(|i| (i % 251) as f32 * 0.001 - 0.12)
            .collect();
        let x = s.tensor_variable_f32(xv, vec![n, ch, d, h, w], false).unwrap();
        b.iter(|| black_box(s.functional_max_pool3d(x, (2, 2, 2), (2, 2, 2)).unwrap()));
    });
    group.bench_function("grad", |b| {
        b.iter(|| {
            let mut s = FrankenTorchSession::new(ExecutionMode::Strict);
            let x = s.tensor_randn(vec![n, ch, d, h, w], true).unwrap();
            let out = s.functional_max_pool3d(x, (2, 2, 2), (2, 2, 2)).unwrap();
            let loss = s.tensor_sum(out).unwrap();
            black_box(s.tensor_backward(loss).unwrap())
        });
    });
    group.finish();
}

fn bench_pool1d_ct1d(c: &mut Criterion) {
    // 1D sliding-window ops now routed through their fused 2D kernels (H=1).
    let mut group = c.benchmark_group("conv1d_family");
    // conv_transpose1d (audio upsampling): [N,C,L]=[4,64,256], k=4 stride 2.
    group.bench_function("conv_transpose1d_grad", |b| {
        b.iter(|| {
            let mut s = FrankenTorchSession::new(ExecutionMode::Strict);
            let x = s.tensor_randn(vec![4, 64, 256], true).unwrap();
            let w = s.tensor_randn(vec![64, 64, 4], true).unwrap();
            let out = s.tensor_conv_transpose1d(x, w, None, 2, 1, 0).unwrap();
            let loss = s.tensor_sum(out).unwrap();
            black_box(s.tensor_backward(loss).unwrap())
        });
    });
    // max/avg_pool1d: [N,C,L]=[8,64,8192], k=2 stride 2.
    group.bench_function("max_pool1d_grad", |b| {
        b.iter(|| {
            let mut s = FrankenTorchSession::new(ExecutionMode::Strict);
            let x = s.tensor_randn(vec![8, 64, 8192], true).unwrap();
            let out = s.functional_max_pool1d(x, 2, 2).unwrap();
            let loss = s.tensor_sum(out).unwrap();
            black_box(s.tensor_backward(loss).unwrap())
        });
    });
    group.bench_function("avg_pool1d_grad", |b| {
        b.iter(|| {
            let mut s = FrankenTorchSession::new(ExecutionMode::Strict);
            let x = s.tensor_randn(vec![8, 64, 8192], true).unwrap();
            let out = s.functional_avg_pool1d(x, 2, 2).unwrap();
            let loss = s.tensor_sum(out).unwrap();
            black_box(s.tensor_backward(loss).unwrap())
        });
    });
    group.finish();
}

fn bench_avg_pool2d(c: &mut Criterion) {
    // avg_pool2d (every CNN): [N,C,H,W]=[8,64,64,64], 2x2 stride 2. no-grad + grad.
    let mut group = c.benchmark_group("avg_pool2d");
    let (n, ch, h, w) = (8usize, 64usize, 64usize, 64usize);
    group.bench_function("nograd", |b| {
        let mut s = FrankenTorchSession::new(ExecutionMode::Strict);
        let x = s.tensor_randn(vec![n, ch, h, w], false).unwrap();
        b.iter(|| black_box(s.functional_avg_pool2d(x, (2, 2), (2, 2), (0, 0), false, true).unwrap()));
    });
    // F32 no-grad (dominant ML dtype): fused avg_pool2d_forward_f32 vs the f32
    // op-graph (narrow/sum/div/cat, also upcast to f64) it fell through to.
    group.bench_function("nograd_f32", |b| {
        let mut s = FrankenTorchSession::new(ExecutionMode::Strict);
        let xv: Vec<f32> = (0..n * ch * h * w)
            .map(|i| (i % 251) as f32 * 0.001 - 0.12)
            .collect();
        let x = s.tensor_variable_f32(xv, vec![n, ch, h, w], false).unwrap();
        b.iter(|| black_box(s.functional_avg_pool2d(x, (2, 2), (2, 2), (0, 0), false, true).unwrap()));
    });
    group.bench_function("grad", |b| {
        b.iter(|| {
            let mut s = FrankenTorchSession::new(ExecutionMode::Strict);
            let x = s.tensor_randn(vec![n, ch, h, w], true).unwrap();
            let out = s.functional_avg_pool2d(x, (2, 2), (2, 2), (0, 0), false, true).unwrap();
            let loss = s.tensor_sum(out).unwrap();
            black_box(s.tensor_backward(loss).unwrap())
        });
    });
    group.finish();
}

fn bench_max_pool2d(c: &mut Criterion) {
    // max_pool2d (every CNN): [N,C,H,W]=[8,64,64,64], 2x2 stride 2. no-grad + grad.
    let mut group = c.benchmark_group("max_pool2d");
    let (n, ch, h, w) = (8usize, 64usize, 64usize, 64usize);
    group.bench_function("nograd", |b| {
        let mut s = FrankenTorchSession::new(ExecutionMode::Strict);
        let x = s.tensor_randn(vec![n, ch, h, w], false).unwrap();
        b.iter(|| black_box(s.functional_max_pool2d(x, (2, 2), (2, 2)).unwrap()));
    });
    // F32 no-grad (dominant ML dtype): fused max_pool2d_forward_f32 vs the f32
    // op-graph (narrow/amax/cat, also upcast to f64) it fell through to.
    group.bench_function("nograd_f32", |b| {
        let mut s = FrankenTorchSession::new(ExecutionMode::Strict);
        let xv: Vec<f32> = (0..n * ch * h * w)
            .map(|i| (i % 251) as f32 * 0.001 - 0.12)
            .collect();
        let x = s.tensor_variable_f32(xv, vec![n, ch, h, w], false).unwrap();
        b.iter(|| black_box(s.functional_max_pool2d(x, (2, 2), (2, 2)).unwrap()));
    });
    group.bench_function("grad", |b| {
        b.iter(|| {
            let mut s = FrankenTorchSession::new(ExecutionMode::Strict);
            let x = s.tensor_randn(vec![n, ch, h, w], true).unwrap();
            let out = s.functional_max_pool2d(x, (2, 2), (2, 2)).unwrap();
            let loss = s.tensor_sum(out).unwrap();
            black_box(s.tensor_backward(loss).unwrap())
        });
    });
    group.finish();
}

fn bench_conv_transpose2d(c: &mut Criterion) {
    // conv_transpose2d (GAN/segmentation upsampling): [N,C,H,W]=[2,64,16,16],
    // weight [C_in,C_out,3,3], stride 2 -> 2x upsample. no-grad + grad.
    let mut group = c.benchmark_group("conv_transpose2d");
    let (n, ic, oc, h, w, k) = (2usize, 16usize, 16usize, 16usize, 16usize, 3usize);
    group.bench_function("nograd", |b| {
        let mut s = FrankenTorchSession::new(ExecutionMode::Strict);
        let x = s.tensor_randn(vec![n, ic, h, w], false).unwrap();
        let wt = s.tensor_randn(vec![ic, oc, k, k], false).unwrap();
        b.iter(|| {
            black_box(
                s.tensor_conv_transpose2d(x, wt, None, (2, 2), (1, 1), (1, 1))
                    .unwrap(),
            )
        });
    });
    // F32 no-grad (dominant ML dtype): fused conv_transpose2d_forward_f32 vs the
    // O(kh*kw*ih) op-graph scatter (narrow/matmul/pad/add) it fell through to.
    group.bench_function("nograd_f32", |b| {
        let mut s = FrankenTorchSession::new(ExecutionMode::Strict);
        let xv: Vec<f32> = (0..n * ic * h * w)
            .map(|i| (i % 251) as f32 * 0.001 - 0.12)
            .collect();
        let wv: Vec<f32> = (0..ic * oc * k * k)
            .map(|i| (i % 97) as f32 * 0.002 - 0.1)
            .collect();
        let x = s.tensor_variable_f32(xv, vec![n, ic, h, w], false).unwrap();
        let wt = s.tensor_variable_f32(wv, vec![ic, oc, k, k], false).unwrap();
        b.iter(|| {
            black_box(
                s.tensor_conv_transpose2d(x, wt, None, (2, 2), (1, 1), (1, 1))
                    .unwrap(),
            )
        });
    });
    group.bench_function("grad", |b| {
        b.iter(|| {
            let mut s = FrankenTorchSession::new(ExecutionMode::Strict);
            let x = s.tensor_randn(vec![n, ic, h, w], true).unwrap();
            let wt = s.tensor_randn(vec![ic, oc, k, k], true).unwrap();
            let out = s
                .tensor_conv_transpose2d(x, wt, None, (2, 2), (1, 1), (1, 1))
                .unwrap();
            let loss = s.tensor_sum(out).unwrap();
            black_box(s.tensor_backward(loss).unwrap())
        });
    });
    group.finish();
}

fn bench_conv3d(c: &mut Criterion) {
    // conv3d (video/volumetric): [N,C,D,H,W]=[2,32,8,16,16], k=3^3 stride1 pad1.
    let mut group = c.benchmark_group("conv3d");
    let (n, ic, oc, d, h, w, k) = (2usize, 32usize, 32usize, 8usize, 16usize, 16usize, 3usize);
    group.bench_function("nograd", |b| {
        let mut s = FrankenTorchSession::new(ExecutionMode::Strict);
        let x = s.tensor_randn(vec![n, ic, d, h, w], false).unwrap();
        let wt = s.tensor_randn(vec![oc, ic, k, k, k], false).unwrap();
        b.iter(|| black_box(s.tensor_conv3d(x, wt, None, (1, 1, 1), (1, 1, 1)).unwrap()));
    });
    // F32 no-grad (dominant ML dtype): fused conv3d_forward_f32 (im2col + sgemm_bt)
    // vs the O(od*oh*ow) narrow/cat/bmm op-graph it fell through to.
    group.bench_function("nograd_f32", |b| {
        let mut s = FrankenTorchSession::new(ExecutionMode::Strict);
        let xv: Vec<f32> = (0..n * ic * d * h * w)
            .map(|i| (i % 251) as f32 * 0.001 - 0.12)
            .collect();
        let wv: Vec<f32> = (0..oc * ic * k * k * k)
            .map(|i| (i % 97) as f32 * 0.002 - 0.1)
            .collect();
        let x = s.tensor_variable_f32(xv, vec![n, ic, d, h, w], false).unwrap();
        let wt = s.tensor_variable_f32(wv, vec![oc, ic, k, k, k], false).unwrap();
        b.iter(|| black_box(s.tensor_conv3d(x, wt, None, (1, 1, 1), (1, 1, 1)).unwrap()));
    });
    group.bench_function("grad", |b| {
        b.iter(|| {
            let mut s = FrankenTorchSession::new(ExecutionMode::Strict);
            let x = s.tensor_randn(vec![n, ic, d, h, w], true).unwrap();
            let wt = s.tensor_randn(vec![oc, ic, k, k, k], true).unwrap();
            let out = s.tensor_conv3d(x, wt, None, (1, 1, 1), (1, 1, 1)).unwrap();
            let loss = s.tensor_sum(out).unwrap();
            black_box(s.tensor_backward(loss).unwrap())
        });
    });
    group.finish();
}

fn bench_conv1d(c: &mut Criterion) {
    // conv1d (audio/sequence): [N,C,L]=[8,64,L], k=3 stride1 pad1, no-grad + grad.
    let mut group = c.benchmark_group("conv1d");
    let (n, ic, oc, k) = (8usize, 64usize, 64usize, 3usize);
    for l in [1024usize, 4096].iter() {
        let l = *l;
        group.bench_with_input(BenchmarkId::new("nograd_L", l), &l, |b, &l| {
            let mut s = FrankenTorchSession::new(ExecutionMode::Strict);
            let x = s.tensor_randn(vec![n, ic, l], false).unwrap();
            let w = s.tensor_randn(vec![oc, ic, k], false).unwrap();
            b.iter(|| black_box(s.tensor_conv1d(x, w, None, 1, 1).unwrap()));
        });
        group.bench_with_input(BenchmarkId::new("grad_L", l), &l, |b, &l| {
            b.iter(|| {
                let mut s = FrankenTorchSession::new(ExecutionMode::Strict);
                let x = s.tensor_randn(vec![n, ic, l], true).unwrap();
                let w = s.tensor_randn(vec![oc, ic, k], true).unwrap();
                let out = s.tensor_conv1d(x, w, None, 1, 1).unwrap();
                let loss = s.tensor_sum(out).unwrap();
                black_box(s.tensor_backward(loss).unwrap())
            });
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
    // F32 no-grad (the dominant ML dtype): exercises the fused conv2d_forward_f32,
    // replacing the serial im2col gather + tensor_matmul the f32 path fell to.
    for hw in [32, 64, 128].iter() {
        let h = *hw;
        let (batch, in_ch, out_ch, kh, kw) = (4usize, 64usize, 64usize, 3usize, 3usize);
        group.bench_with_input(BenchmarkId::new("f32_hw", h), &h, |b, &h| {
            let mut session = FrankenTorchSession::new(ExecutionMode::Strict);
            let iv: Vec<f32> = (0..batch * in_ch * h * h)
                .map(|i| (i % 251) as f32 * 0.001 - 0.12)
                .collect();
            let wv: Vec<f32> = (0..out_ch * in_ch * kh * kw)
                .map(|i| (i % 97) as f32 * 0.002 - 0.1)
                .collect();
            let input = session
                .tensor_variable_f32(iv, vec![batch, in_ch, h, h], false)
                .unwrap();
            let weight = session
                .tensor_variable_f32(wv, vec![out_ch, in_ch, kh, kw], false)
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
    // Training (forward + backward): exercises the fused conv2d grad custom op.
    for hw in [32, 64].iter() {
        let h = *hw;
        let (batch, in_ch, out_ch, kh, kw) = (4, 64, 64, 3, 3);
        group.bench_with_input(BenchmarkId::new("grad_hw", h), &h, |b, &h| {
            b.iter(|| {
                let mut session = FrankenTorchSession::new(ExecutionMode::Strict);
                let input = session.tensor_randn(vec![batch, in_ch, h, h], true).unwrap();
                let weight = session
                    .tensor_randn(vec![out_ch, in_ch, kh, kw], true)
                    .unwrap();
                let out = session
                    .tensor_conv2d(input, weight, None, (1, 1), (1, 1))
                    .unwrap();
                let loss = session.tensor_sum(out).unwrap();
                black_box(session.tensor_backward(loss).unwrap())
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

fn bench_norm(c: &mut Criterion) {
    let mut group = c.benchmark_group("norm");
    for size in [100000, 1000000].iter() {
        let n = *size;
        group.throughput(Throughput::Elements(n as u64));
        group.bench_with_input(BenchmarkId::new("l2", n), &n, |b, &n| {
            let mut session = FrankenTorchSession::new(ExecutionMode::Strict);
            let x = session.tensor_randn(vec![n], false).unwrap();
            b.iter(|| black_box(session.tensor_norm(x, 2.0).unwrap()));
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

fn bench_layer_norm(c: &mut Criterion) {
    // LayerNorm over the last dim, no-grad f64 — every transformer layer. The
    // op-graph path allocates ~14 full [rows, hidden] intermediates + tape nodes.
    let mut group = c.benchmark_group("layer_norm");
    let (rows, hidden) = (2048usize, 1024usize);
    group.bench_function("nograd_2048x1024", |b| {
        let mut session = FrankenTorchSession::new(ExecutionMode::Strict);
        let x = session.tensor_randn(vec![rows, hidden], false).unwrap();
        let w = session.tensor_randn(vec![hidden], false).unwrap();
        let bias = session.tensor_randn(vec![hidden], false).unwrap();
        b.iter(|| {
            black_box(
                session
                    .functional_layer_norm(x, vec![hidden], Some(w), Some(bias), 1e-5)
                    .unwrap(),
            )
        });
    });
    // F32 no-grad (dominant ML dtype): fused layer_norm_forward_f32 vs the f32
    // op-graph (which also upcast to f64) it fell through to.
    group.bench_function("nograd_f32_2048x1024", |b| {
        let mut session = FrankenTorchSession::new(ExecutionMode::Strict);
        let xv: Vec<f32> = (0..rows * hidden)
            .map(|i| (i % 251) as f32 * 0.001 - 0.12)
            .collect();
        let wv: Vec<f32> = (0..hidden).map(|j| 1.0 + (j % 17) as f32 * 0.01).collect();
        let bv: Vec<f32> = (0..hidden).map(|j| (j % 13) as f32 * 0.005 - 0.03).collect();
        let x = session.tensor_variable_f32(xv, vec![rows, hidden], false).unwrap();
        let w = session.tensor_variable_f32(wv, vec![hidden], false).unwrap();
        let bias = session.tensor_variable_f32(bv, vec![hidden], false).unwrap();
        b.iter(|| {
            black_box(
                session
                    .functional_layer_norm(x, vec![hidden], Some(w), Some(bias), 1e-5)
                    .unwrap(),
            )
        });
    });
    // Forward + backward (training): exercises the fused grad LayerNorm op.
    group.bench_function("grad_2048x1024", |b| {
        b.iter(|| {
            let mut session = FrankenTorchSession::new(ExecutionMode::Strict);
            let x = session.tensor_randn(vec![rows, hidden], true).unwrap();
            let w = session.tensor_randn(vec![hidden], true).unwrap();
            let bias = session.tensor_randn(vec![hidden], true).unwrap();
            let out = session
                .functional_layer_norm(x, vec![hidden], Some(w), Some(bias), 1e-5)
                .unwrap();
            let loss = session.tensor_sum(out).unwrap();
            black_box(session.tensor_backward(loss).unwrap())
        });
    });
    group.finish();
}

fn bench_batch_norm(c: &mut Criterion) {
    // BatchNorm2d over [N,C,H,W] = [32,256,28,28], no-grad f64, training + eval.
    // The op-graph does two full-tensor permutes + ~10 expand/full intermediates.
    let mut group = c.benchmark_group("batch_norm");
    let (n, ch, h, w) = (32usize, 256usize, 28usize, 28usize);
    for (label, training) in [("train", true), ("eval", false)] {
        group.bench_function(format!("nograd_{label}_32x256x28x28"), |b| {
            let mut session = FrankenTorchSession::new(ExecutionMode::Strict);
            let x = session.tensor_randn(vec![n, ch, h, w], false).unwrap();
            let rm = session.tensor_randn(vec![ch], false).unwrap();
            let rv = session
                .tensor_variable(vec![1.0; ch], vec![ch], false)
                .unwrap();
            let wt = session.tensor_randn(vec![ch], false).unwrap();
            let bias = session.tensor_randn(vec![ch], false).unwrap();
            b.iter(|| {
                black_box(
                    session
                        .functional_batch_norm2d(
                            x, Some(rm), Some(rv), Some(wt), Some(bias), training, 0.1, 1e-5,
                        )
                        .unwrap(),
                )
            });
        });
    }
    // F32 no-grad (dominant ML dtype): fused batch_norm_stats_f32 + apply_f32 vs
    // the f32 op-graph (two permutes + ~10 nodes, also upcast to f64).
    for (label, training) in [("train", true), ("eval", false)] {
        group.bench_function(format!("nograd_f32_{label}_32x256x28x28"), |b| {
            let mut session = FrankenTorchSession::new(ExecutionMode::Strict);
            let xv: Vec<f32> = (0..n * ch * h * w)
                .map(|i| (i % 251) as f32 * 0.001 - 0.12)
                .collect();
            let x = session.tensor_variable_f32(xv, vec![n, ch, h, w], false).unwrap();
            let rm = session
                .tensor_variable_f32(vec![0.0f32; ch], vec![ch], false)
                .unwrap();
            let rv = session
                .tensor_variable_f32(vec![1.0f32; ch], vec![ch], false)
                .unwrap();
            let wt = session
                .tensor_variable_f32((0..ch).map(|j| 0.8 + (j % 17) as f32 * 0.01).collect(), vec![ch], false)
                .unwrap();
            let bias = session
                .tensor_variable_f32(vec![0.0f32; ch], vec![ch], false)
                .unwrap();
            b.iter(|| {
                black_box(
                    session
                        .functional_batch_norm2d(
                            x, Some(rm), Some(rv), Some(wt), Some(bias), training, 0.1, 1e-5,
                        )
                        .unwrap(),
                )
            });
        });
    }
    // BatchNorm1d [N,C] (MLP, spatial=1) no-grad + grad.
    {
        let (bn, bc) = (8192usize, 1024usize);
        group.bench_function("nograd_1d_8192x1024", |b| {
            let mut s = FrankenTorchSession::new(ExecutionMode::Strict);
            let x = s.tensor_randn(vec![bn, bc], false).unwrap();
            let rm = s.tensor_randn(vec![bc], false).unwrap();
            let rv = s.tensor_variable(vec![1.0; bc], vec![bc], false).unwrap();
            let wt = s.tensor_randn(vec![bc], false).unwrap();
            let bias = s.tensor_randn(vec![bc], false).unwrap();
            b.iter(|| {
                black_box(
                    s.functional_batch_norm1d(x, Some(rm), Some(rv), Some(wt), Some(bias), true, 0.1, 1e-5)
                        .unwrap(),
                )
            });
        });
        group.bench_function("grad_1d_8192x1024", |b| {
            b.iter(|| {
                let mut s = FrankenTorchSession::new(ExecutionMode::Strict);
                let x = s.tensor_randn(vec![bn, bc], true).unwrap();
                let rm = s.tensor_randn(vec![bc], false).unwrap();
                let rv = s.tensor_variable(vec![1.0; bc], vec![bc], false).unwrap();
                let wt = s.tensor_randn(vec![bc], true).unwrap();
                let bias = s.tensor_randn(vec![bc], true).unwrap();
                let (out, _, _) = s
                    .functional_batch_norm1d(x, Some(rm), Some(rv), Some(wt), Some(bias), true, 0.1, 1e-5)
                    .unwrap();
                let loss = s.tensor_sum(out).unwrap();
                black_box(s.tensor_backward(loss).unwrap())
            });
        });
    }
    // BatchNorm3d [N,C,D,H,W] (3D CNN) no-grad train.
    group.bench_function("nograd_3d_8x64x8x16x16", |b| {
        let mut s = FrankenTorchSession::new(ExecutionMode::Strict);
        let x = s.tensor_randn(vec![8, 64, 8, 16, 16], false).unwrap();
        let rm = s.tensor_randn(vec![64], false).unwrap();
        let rv = s.tensor_variable(vec![1.0; 64], vec![64], false).unwrap();
        let wt = s.tensor_randn(vec![64], false).unwrap();
        let bias = s.tensor_randn(vec![64], false).unwrap();
        b.iter(|| {
            black_box(
                s.functional_batch_norm3d(x, Some(rm), Some(rv), Some(wt), Some(bias), true, 0.1, 1e-5)
                    .unwrap(),
            )
        });
    });
    // Training forward + backward.
    group.bench_function("grad_train_32x256x28x28", |b| {
        b.iter(|| {
            let mut session = FrankenTorchSession::new(ExecutionMode::Strict);
            let x = session.tensor_randn(vec![n, ch, h, w], true).unwrap();
            let rm = session.tensor_randn(vec![ch], false).unwrap();
            let rv = session
                .tensor_variable(vec![1.0; ch], vec![ch], false)
                .unwrap();
            let wt = session.tensor_randn(vec![ch], true).unwrap();
            let bias = session.tensor_randn(vec![ch], true).unwrap();
            let (out, _, _) = session
                .functional_batch_norm2d(x, Some(rm), Some(rv), Some(wt), Some(bias), true, 0.1, 1e-5)
                .unwrap();
            let loss = session.tensor_sum(out).unwrap();
            black_box(session.tensor_backward(loss).unwrap())
        });
    });
    group.finish();
}

fn bench_group_norm(c: &mut Criterion) {
    // GroupNorm over [N, C, H, W] = [32, 256, 28, 28], num_groups=32 (ResNet-ish),
    // no-grad and grad f64. The op-graph allocates ~15 full-size intermediates.
    let mut group = c.benchmark_group("group_norm");
    let (n, ch, h, w, groups) = (32usize, 256usize, 28usize, 28usize, 32usize);
    group.bench_function("nograd_32x256x28x28", |b| {
        let mut session = FrankenTorchSession::new(ExecutionMode::Strict);
        let x = session.tensor_randn(vec![n, ch, h, w], false).unwrap();
        let wt = session.tensor_randn(vec![ch], false).unwrap();
        let bias = session.tensor_randn(vec![ch], false).unwrap();
        b.iter(|| {
            black_box(
                session
                    .functional_group_norm(x, groups, Some(wt), Some(bias), 1e-5)
                    .unwrap(),
            )
        });
    });
    // F32 no-grad (dominant ML dtype): fused group_norm_forward_f32 vs the f32
    // op-graph (which also upcast to f64) it fell through to.
    group.bench_function("nograd_f32_32x256x28x28", |b| {
        let mut session = FrankenTorchSession::new(ExecutionMode::Strict);
        let xv: Vec<f32> = (0..n * ch * h * w)
            .map(|i| (i % 251) as f32 * 0.001 - 0.12)
            .collect();
        let wv: Vec<f32> = (0..ch).map(|j| 0.8 + (j % 17) as f32 * 0.01).collect();
        let bv: Vec<f32> = (0..ch).map(|j| (j % 13) as f32 * 0.005 - 0.03).collect();
        let x = session.tensor_variable_f32(xv, vec![n, ch, h, w], false).unwrap();
        let wt = session.tensor_variable_f32(wv, vec![ch], false).unwrap();
        let bias = session.tensor_variable_f32(bv, vec![ch], false).unwrap();
        b.iter(|| {
            black_box(
                session
                    .functional_group_norm(x, groups, Some(wt), Some(bias), 1e-5)
                    .unwrap(),
            )
        });
    });
    group.bench_function("grad_32x256x28x28", |b| {
        b.iter(|| {
            let mut session = FrankenTorchSession::new(ExecutionMode::Strict);
            let x = session.tensor_randn(vec![n, ch, h, w], true).unwrap();
            let wt = session.tensor_randn(vec![ch], true).unwrap();
            let bias = session.tensor_randn(vec![ch], true).unwrap();
            let out = session
                .functional_group_norm(x, groups, Some(wt), Some(bias), 1e-5)
                .unwrap();
            let loss = session.tensor_sum(out).unwrap();
            black_box(session.tensor_backward(loss).unwrap())
        });
    });
    group.finish();
}

fn bench_smooth_l1(c: &mut Criterion) {
    // smooth-L1 / Huber loss (detection, RL), no-grad + grad f64. The op-graph
    // builds ~13 full [rows*cols] intermediates incl. 4 constant full() tensors.
    let mut group = c.benchmark_group("smooth_l1");
    let n = 4096 * 2048usize;
    group.bench_function("nograd_8m", |b| {
        let mut s = FrankenTorchSession::new(ExecutionMode::Strict);
        let x = s.tensor_randn(vec![n], false).unwrap();
        let t = s.tensor_randn(vec![n], false).unwrap();
        b.iter(|| black_box(s.tensor_smooth_l1_loss(x, t, "mean", 1.0).unwrap()));
    });
    group.bench_function("grad_8m", |b| {
        b.iter(|| {
            let mut s = FrankenTorchSession::new(ExecutionMode::Strict);
            let x = s.tensor_randn(vec![n], true).unwrap();
            let t = s.tensor_randn(vec![n], false).unwrap();
            let loss = s.tensor_smooth_l1_loss(x, t, "mean", 1.0).unwrap();
            black_box(s.tensor_backward(loss).unwrap())
        });
    });
    group.finish();
}

fn bench_gaussian_nll(c: &mut Criterion) {
    // Gaussian NLL (uncertainty regression), no-grad + grad f64. The op-graph
    // builds ~7 full [n] intermediates.
    let mut group = c.benchmark_group("gaussian_nll");
    let n = 4096 * 2048usize;
    group.bench_function("nograd_8m", |b| {
        let mut s = FrankenTorchSession::new(ExecutionMode::Strict);
        let x = s.tensor_randn(vec![n], false).unwrap();
        let t = s.tensor_randn(vec![n], false).unwrap();
        let v = s.tensor_variable(vec![1.5; n], vec![n], false).unwrap();
        b.iter(|| black_box(s.tensor_gaussian_nll_loss(x, t, v, "mean", false).unwrap()));
    });
    group.bench_function("grad_8m", |b| {
        b.iter(|| {
            let mut s = FrankenTorchSession::new(ExecutionMode::Strict);
            let x = s.tensor_randn(vec![n], true).unwrap();
            let t = s.tensor_randn(vec![n], false).unwrap();
            let v = s.tensor_variable(vec![1.5; n], vec![n], true).unwrap();
            let loss = s.tensor_gaussian_nll_loss(x, t, v, "mean", false).unwrap();
            black_box(s.tensor_backward(loss).unwrap())
        });
    });
    group.finish();
}

fn bench_cross_entropy(c: &mut Criterion) {
    // Softmax-cross-entropy, no-grad and grad f64. [batch, classes] = [4096, 8192]
    // (LLM-vocab scale) — the op-graph materialises a 256MB log-softmax tensor.
    let mut group = c.benchmark_group("cross_entropy");
    let (batch, classes) = (2048usize, 4096usize);
    let targets: Vec<f64> = (0..batch).map(|i| (i % classes) as f64).collect();
    group.bench_function("nograd_4096x8192", |b| {
        let mut session = FrankenTorchSession::new(ExecutionMode::Strict);
        let x = session.tensor_randn(vec![batch, classes], false).unwrap();
        let t = session.tensor_variable(targets.clone(), vec![batch], false).unwrap();
        b.iter(|| black_box(session.functional_cross_entropy(x, t, "mean").unwrap()));
    });
    group.bench_function("grad_4096x8192", |b| {
        b.iter(|| {
            let mut session = FrankenTorchSession::new(ExecutionMode::Strict);
            let x = session.tensor_randn(vec![batch, classes], true).unwrap();
            let t = session.tensor_variable(targets.clone(), vec![batch], false).unwrap();
            let loss = session.functional_cross_entropy(x, t, "mean").unwrap();
            black_box(session.tensor_backward(loss).unwrap())
        });
    });
    group.finish();
}

fn bench_rms_norm(c: &mut Criterion) {
    // RMSNorm over the last dim (LLaMA-style), no-grad and grad f64.
    let mut group = c.benchmark_group("rms_norm");
    let (rows, hidden) = (2048usize, 1024usize);
    group.bench_function("nograd_2048x1024", |b| {
        let mut session = FrankenTorchSession::new(ExecutionMode::Strict);
        let x = session.tensor_randn(vec![rows, hidden], false).unwrap();
        let w = session.tensor_randn(vec![hidden], false).unwrap();
        b.iter(|| {
            black_box(
                session
                    .functional_rms_norm(x, vec![hidden], Some(w), 1e-6)
                    .unwrap(),
            )
        });
    });
    // F32 no-grad (dominant ML dtype): fused rms_norm_forward_f32 vs the f32
    // op-graph (which also upcast to f64) it fell through to.
    group.bench_function("nograd_f32_2048x1024", |b| {
        let mut session = FrankenTorchSession::new(ExecutionMode::Strict);
        let xv: Vec<f32> = (0..rows * hidden)
            .map(|i| (i % 251) as f32 * 0.001 - 0.12)
            .collect();
        let wv: Vec<f32> = (0..hidden).map(|j| 0.8 + (j % 17) as f32 * 0.01).collect();
        let x = session.tensor_variable_f32(xv, vec![rows, hidden], false).unwrap();
        let w = session.tensor_variable_f32(wv, vec![hidden], false).unwrap();
        b.iter(|| {
            black_box(
                session
                    .functional_rms_norm(x, vec![hidden], Some(w), 1e-6)
                    .unwrap(),
            )
        });
    });
    group.bench_function("grad_2048x1024", |b| {
        b.iter(|| {
            let mut session = FrankenTorchSession::new(ExecutionMode::Strict);
            let x = session.tensor_randn(vec![rows, hidden], true).unwrap();
            let w = session.tensor_randn(vec![hidden], true).unwrap();
            let out = session
                .functional_rms_norm(x, vec![hidden], Some(w), 1e-6)
                .unwrap();
            let loss = session.tensor_sum(out).unwrap();
            black_box(session.tensor_backward(loss).unwrap())
        });
    });
    group.finish();
}

fn bench_sdpa(c: &mut Criterion) {
    // Scaled-dot-product attention, no-grad f64. num_bh=16, seq=512, d=64 -> the
    // score matrix is 16*512*512*8 = 33.5MB; the fused kernel never materialises
    // it (nor the scale tensor / softmax intermediate the op-graph path streams).
    let mut group = c.benchmark_group("sdpa");
    let (bh, seq, d) = (16usize, 512usize, 64usize);
    group.bench_function("nograd_16x512x64", |b| {
        let mut session = FrankenTorchSession::new(ExecutionMode::Strict);
        let q = session.tensor_randn(vec![bh, seq, d], false).unwrap();
        let k = session.tensor_randn(vec![bh, seq, d], false).unwrap();
        let v = session.tensor_randn(vec![bh, seq, d], false).unwrap();
        b.iter(|| {
            black_box(
                session
                    .scaled_dot_product_attention(q, k, v, None, 0.0, false)
                    .unwrap(),
            )
        });
    });
    // f32 forward (the common transformer inference dtype).
    group.bench_function("nograd_f32_16x512x64", |b| {
        let mut session = FrankenTorchSession::new(ExecutionMode::Strict);
        let q = session.randn_f32(vec![bh, seq, d], false).unwrap();
        let k = session.randn_f32(vec![bh, seq, d], false).unwrap();
        let v = session.randn_f32(vec![bh, seq, d], false).unwrap();
        b.iter(|| {
            black_box(
                session
                    .scaled_dot_product_attention(q, k, v, None, 0.0, false)
                    .unwrap(),
            )
        });
    });
    // Forward + backward (training): exercises the fused grad SDPA custom op.
    group.bench_function("grad_16x512x64", |b| {
        b.iter(|| {
            let mut session = FrankenTorchSession::new(ExecutionMode::Strict);
            let q = session.tensor_randn(vec![bh, seq, d], true).unwrap();
            let k = session.tensor_randn(vec![bh, seq, d], true).unwrap();
            let v = session.tensor_randn(vec![bh, seq, d], true).unwrap();
            let out = session
                .scaled_dot_product_attention(q, k, v, None, 0.0, false)
                .unwrap();
            let loss = session.tensor_sum(out).unwrap();
            black_box(session.tensor_backward(loss).unwrap())
        });
    });
    group.finish();
}

fn bench_linear_train(c: &mut Criterion) {
    // Forward + backward of a Linear layer (f64, requires_grad) — the training
    // hot path. Exercises the fused grad Linear op (transpose-free dgemm_bt fwd +
    // analytic linear_backward_f64) vs the materialise-transpose addmm path.
    let mut group = c.benchmark_group("linear_train");
    let batch = 32;
    let in_features = 512;
    for hidden in [512, 1024, 2048].iter() {
        let h = *hidden;
        group.bench_with_input(BenchmarkId::new("hidden", h), &h, |b, &h| {
            b.iter(|| {
                let mut session = FrankenTorchSession::new(ExecutionMode::Strict);
                let x = session.tensor_randn(vec![batch, in_features], true).unwrap();
                let w = session.tensor_randn(vec![h, in_features], true).unwrap();
                let bias = session.tensor_randn(vec![h], true).unwrap();
                let y = session.tensor_linear(x, w, Some(bias)).unwrap();
                let loss = session.tensor_sum(y).unwrap();
                black_box(session.tensor_backward(loss).unwrap())
            });
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
    // Same-binary A/B baseline: the pre-fusion op-graph (L2-normalize + gram
    // matmul + two [N,N] mask leaves + masked log-sum-exp + masked positive-mean
    // + reduction), reproduced inline with requires_grad=false inputs so the only
    // difference from "512x512" is the fused no-grad kernel.
    group.bench_function("512x512_opgraph", |b| {
        let mut session = FrankenTorchSession::new(ExecutionMode::Strict);
        let emb = session.tensor_randn(vec![n, d], false).unwrap();
        let labels_v: Vec<f64> = (0..n).map(|i| (i % 8) as f64).collect();
        let temperature = 0.07_f64;
        b.iter(|| {
            let labels_i: Vec<i64> = labels_v.iter().map(|&x| x as i64).collect();
            let mut pos_mask = vec![0.0f64; n * n];
            let mut diag_mask = vec![0.0f64; n * n];
            let mut pos_count = vec![0.0f64; n];
            let mut valid = vec![0.0f64; n];
            let mut count = 0usize;
            for i in 0..n {
                diag_mask[i * n + i] = -1e30;
                let mut cc = 0.0;
                for j in 0..n {
                    if i != j && labels_i[i] == labels_i[j] {
                        pos_mask[i * n + j] = 1.0;
                        cc += 1.0;
                    }
                }
                pos_count[i] = cc;
                if cc > 0.0 {
                    valid[i] = 1.0;
                    count += 1;
                }
            }
            let pos_count_safe: Vec<f64> = pos_count.iter().map(|&c| c.max(1.0)).collect();
            let s = &mut session;
            let sq = s.tensor_mul(emb, emb).unwrap();
            let sumsq = s.tensor_sum_dim(sq, 1).unwrap();
            let norm = s.tensor_sqrt(sumsq).unwrap();
            let norm_safe = s.tensor_clamp_min(norm, 1e-12).unwrap();
            let norm_col = s.tensor_unsqueeze(norm_safe, 1).unwrap();
            let norm_exp = s.tensor_expand(norm_col, vec![n, d]).unwrap();
            let normalized = s.tensor_div(emb, norm_exp).unwrap();
            let normalized_t = s.tensor_transpose(normalized, 0, 1).unwrap();
            let gram = s.tensor_matmul(normalized, normalized_t).unwrap();
            let sim = s.tensor_mul_scalar(gram, 1.0 / temperature).unwrap();
            let diag_t = s.tensor_variable(diag_mask, vec![n, n], false).unwrap();
            let sim_masked = s.tensor_add(sim, diag_t).unwrap();
            let log_denom = s.tensor_logsumexp(sim_masked, 1).unwrap();
            let pos_t = s.tensor_variable(pos_mask, vec![n, n], false).unwrap();
            let masked_sim = s.tensor_mul(sim, pos_t).unwrap();
            let num = s.tensor_sum_dim(masked_sim, 1).unwrap();
            let pos_count_t = s.tensor_variable(pos_count_safe, vec![n], false).unwrap();
            let mean_pos = s.tensor_div(num, pos_count_t).unwrap();
            let term = s.tensor_sub(mean_pos, log_denom).unwrap();
            let valid_t = s.tensor_variable(valid, vec![n], false).unwrap();
            let masked_term = s.tensor_mul(term, valid_t).unwrap();
            let summed = s.tensor_sum(masked_term).unwrap();
            let scaled = s.tensor_mul_scalar(summed, -1.0 / count as f64).unwrap();
            black_box(s.tensor_reshape(scaled, vec![1]).unwrap());
        });
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
    bench_conv1d,
    bench_conv2d,
    bench_conv3d,
    bench_conv_transpose2d,
    bench_max_pool2d,
    bench_avg_pool2d,
    bench_max_pool3d,
    bench_pool1d_ct1d,
    bench_sum,
    bench_norm,
    bench_softmax,
    bench_relu,
    bench_exp,
    bench_sigmoid,
    bench_pow,
    bench_add,
    bench_backward_matmul,
    bench_linear_train,
    bench_sdpa,
    bench_layer_norm,
    bench_rms_norm,
    bench_cross_entropy,
    bench_group_norm,
    bench_batch_norm,
    bench_smooth_l1,
    bench_gaussian_nll,
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

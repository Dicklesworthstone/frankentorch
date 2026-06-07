use criterion::{BatchSize, Criterion, black_box, criterion_group, criterion_main};
use ft_api::FrankenTorchSession;
use ft_core::ExecutionMode;
use ft_nn::{GRU, LSTM, Module, MultiheadAttention, RNN, RNNConfig};

fn make_mha_case(
    batch_size: usize,
    seq_len: usize,
    embed_dim: usize,
    num_heads: usize,
) -> (
    FrankenTorchSession,
    MultiheadAttention,
    ft_autograd::TensorNodeId,
) {
    let mut session = FrankenTorchSession::new(ExecutionMode::Strict);
    let attention = MultiheadAttention::new(&mut session, embed_dim, num_heads).expect("mha");
    let values = (0..(batch_size * seq_len * embed_dim))
        .map(|idx| (idx % 251) as f64 * 0.001)
        .collect::<Vec<_>>();
    let input = session
        .tensor_variable(values, vec![batch_size, seq_len, embed_dim], true)
        .expect("input");
    (session, attention, input)
}

fn bench_multihead_attention(c: &mut Criterion) {
    let mut group = c.benchmark_group("multihead_attention");
    group.bench_function("forward_8x64x128_h8", |b| {
        b.iter_batched(
            || make_mha_case(8, 64, 128, 8),
            |(mut session, attention, input)| {
                let output = attention.forward(&mut session, input).expect("forward");
                let first = session
                    .tensor_values(output)
                    .expect("values")
                    .first()
                    .copied()
                    .expect("nonempty output");
                black_box(first)
            },
            BatchSize::SmallInput,
        );
    });
    group.bench_function("forward_1x512x128_h8", |b| {
        b.iter_batched(
            || make_mha_case(1, 512, 128, 8),
            |(mut session, attention, input)| {
                let output = attention.forward(&mut session, input).expect("forward");
                let first = session
                    .tensor_values(output)
                    .expect("values")
                    .first()
                    .copied()
                    .expect("nonempty output");
                black_box(first)
            },
            BatchSize::SmallInput,
        );
    });
    group.finish();
}

fn bench_multihead_attention_nograd(c: &mut Criterion) {
    let mut group = c.benchmark_group("multihead_attention");
    group.bench_function("forward_8x64x128_h8_nograd", |b| {
        b.iter_batched(
            || make_mha_case(8, 64, 128, 8),
            |(mut session, attention, input)| {
                session.no_grad_enter();
                let output = attention.forward(&mut session, input).expect("forward");
                let first = session
                    .tensor_values(output)
                    .expect("values")
                    .first()
                    .copied()
                    .expect("nonempty output");
                session.no_grad_exit();
                black_box(first)
            },
            BatchSize::SmallInput,
        );
    });
    group.bench_function("forward_1x512x128_h8_nograd", |b| {
        b.iter_batched(
            || make_mha_case(1, 512, 128, 8),
            |(mut session, attention, input)| {
                session.no_grad_enter();
                let output = attention.forward(&mut session, input).expect("forward");
                let first = session
                    .tensor_values(output)
                    .expect("values")
                    .first()
                    .copied()
                    .expect("nonempty output");
                session.no_grad_exit();
                black_box(first)
            },
            BatchSize::SmallInput,
        );
    });
    group.bench_function("forward_1x1024x128_h8_nograd", |b| {
        b.iter_batched(
            || make_mha_case(1, 1024, 128, 8),
            |(mut session, attention, input)| {
                session.no_grad_enter();
                let output = attention.forward(&mut session, input).expect("forward");
                let first = session
                    .tensor_values(output)
                    .expect("values")
                    .first()
                    .copied()
                    .expect("nonempty output");
                session.no_grad_exit();
                black_box(first)
            },
            BatchSize::SmallInput,
        );
    });
    group.finish();
}

fn make_lstm_case(
    seq_len: usize,
    batch_size: usize,
    input_size: usize,
    hidden_size: usize,
) -> (FrankenTorchSession, LSTM, ft_autograd::TensorNodeId) {
    let mut session = FrankenTorchSession::new(ExecutionMode::Strict);
    let lstm = LSTM::new(
        &mut session,
        input_size,
        hidden_size,
        1,
        false,
        0.0,
        false,
    )
    .expect("lstm");
    let values = (0..(seq_len * batch_size * input_size))
        .map(|idx| (idx % 251) as f64 * 0.001)
        .collect::<Vec<_>>();
    let input = session
        .tensor_variable(values, vec![seq_len, batch_size, input_size], true)
        .expect("input");
    (session, lstm, input)
}

fn bench_lstm_forward(c: &mut Criterion) {
    let mut group = c.benchmark_group("lstm");
    group.bench_function("forward_seq128_b16_in128_h256", |b| {
        b.iter_batched(
            || make_lstm_case(128, 16, 128, 256),
            |(mut session, lstm, input)| {
                let output = lstm.forward(&mut session, input).expect("forward");
                let first = session
                    .tensor_values(output)
                    .expect("values")
                    .first()
                    .copied()
                    .expect("nonempty output");
                black_box(first)
            },
            BatchSize::SmallInput,
        );
    });
    group.bench_function("forward_seq128_b16_in128_h256_nograd", |b| {
        b.iter_batched(
            || make_lstm_case(128, 16, 128, 256),
            |(mut session, lstm, input)| {
                session.no_grad_enter();
                let output = lstm.forward(&mut session, input).expect("forward");
                let first = session
                    .tensor_values(output)
                    .expect("values")
                    .first()
                    .copied()
                    .expect("nonempty output");
                session.no_grad_exit();
                black_box(first)
            },
            BatchSize::SmallInput,
        );
    });
    group.finish();
}

fn make_gru_case(
    seq_len: usize,
    batch_size: usize,
    input_size: usize,
    hidden_size: usize,
) -> (FrankenTorchSession, GRU, ft_autograd::TensorNodeId) {
    let mut session = FrankenTorchSession::new(ExecutionMode::Strict);
    let gru = GRU::new(
        &mut session,
        input_size,
        hidden_size,
        1,
        false,
        0.0,
        false,
    )
    .expect("gru");
    let values = (0..(seq_len * batch_size * input_size))
        .map(|idx| (idx % 251) as f64 * 0.001)
        .collect::<Vec<_>>();
    let input = session
        .tensor_variable(values, vec![seq_len, batch_size, input_size], true)
        .expect("input");
    (session, gru, input)
}

fn bench_gru_forward(c: &mut Criterion) {
    let mut group = c.benchmark_group("gru");
    group.bench_function("forward_seq128_b16_in128_h256", |b| {
        b.iter_batched(
            || make_gru_case(128, 16, 128, 256),
            |(mut session, gru, input)| {
                let output = gru.forward(&mut session, input).expect("forward");
                let first = session
                    .tensor_values(output)
                    .expect("values")
                    .first()
                    .copied()
                    .expect("nonempty output");
                black_box(first)
            },
            BatchSize::SmallInput,
        );
    });
    group.bench_function("forward_seq128_b16_in128_h256_nograd", |b| {
        b.iter_batched(
            || make_gru_case(128, 16, 128, 256),
            |(mut session, gru, input)| {
                session.no_grad_enter();
                let output = gru.forward(&mut session, input).expect("forward");
                let first = session
                    .tensor_values(output)
                    .expect("values")
                    .first()
                    .copied()
                    .expect("nonempty output");
                session.no_grad_exit();
                black_box(first)
            },
            BatchSize::SmallInput,
        );
    });
    group.finish();
}

fn make_rnn_case(
    seq_len: usize,
    batch_size: usize,
    input_size: usize,
    hidden_size: usize,
) -> (FrankenTorchSession, RNN, ft_autograd::TensorNodeId) {
    let mut session = FrankenTorchSession::new(ExecutionMode::Strict);
    let rnn = RNN::new(
        &mut session,
        input_size,
        hidden_size,
        RNNConfig::default(),
    )
    .expect("rnn");
    let values = (0..(seq_len * batch_size * input_size))
        .map(|idx| (idx % 251) as f64 * 0.001)
        .collect::<Vec<_>>();
    let input = session
        .tensor_variable(values, vec![seq_len, batch_size, input_size], true)
        .expect("input");
    (session, rnn, input)
}

fn bench_rnn_forward(c: &mut Criterion) {
    let mut group = c.benchmark_group("rnn");
    group.bench_function("forward_seq128_b16_in128_h256_nograd", |b| {
        b.iter_batched(
            || make_rnn_case(128, 16, 128, 256),
            |(mut session, rnn, input)| {
                session.no_grad_enter();
                let output = rnn.forward(&mut session, input).expect("forward");
                let first = session
                    .tensor_values(output)
                    .expect("values")
                    .first()
                    .copied()
                    .expect("nonempty output");
                session.no_grad_exit();
                black_box(first)
            },
            BatchSize::SmallInput,
        );
    });
    group.finish();
}

/// Same-binary A/B for the MHA no-grad attention core (the lever changed in the
/// flash swap): the OLD `materialized` path (scale Q, transpose K, bmm scores,
/// row softmax, bmm scores@V — building the full [bh, S, S] score matrix) vs the
/// NEW `flash` path (ft_kernel_cpu::sdpa_forward_f64, tiled, no materialised
/// scores). Both run in one binary so the ratio is immune to worker variance.
fn bench_attention_core(c: &mut Criterion) {
    use ft_core::{DType, Device, TensorMeta};
    let mut group = c.benchmark_group("attention_core");
    for &(bh, s, hd) in &[(8usize, 512usize, 16usize), (8, 1024, 16)] {
        let scale = 1.0 / (hd as f64).sqrt();
        let q: Vec<f64> = (0..bh * s * hd).map(|i| (i % 251) as f64 * 0.001 - 0.12).collect();
        let k: Vec<f64> = (0..bh * s * hd).map(|i| (i % 241) as f64 * 0.0011 - 0.13).collect();
        let v: Vec<f64> = (0..bh * s * hd).map(|i| (i % 233) as f64 * 0.0009 - 0.10).collect();
        group.bench_function(&format!("materialized_{bh}x{s}x{hd}"), |b| {
            b.iter(|| {
                let qs: Vec<f64> = q.iter().map(|&x| x * scale).collect();
                let mut k_t = vec![0.0f64; bh * hd * s];
                for bhi in 0..bh {
                    for si in 0..s {
                        for di in 0..hd {
                            k_t[bhi * hd * s + di * s + si] = k[bhi * s * hd + si * hd + di];
                        }
                    }
                }
                let q_meta = TensorMeta::from_shape(vec![bh, s, hd], DType::F64, Device::Cpu);
                let kt_meta = TensorMeta::from_shape(vec![bh, hd, s], DType::F64, Device::Cpu);
                let mut scores =
                    ft_kernel_cpu::bmm_tensor_contiguous_f64(&qs, &k_t, &q_meta, &kt_meta).unwrap();
                // Parallel row softmax, mirroring the old softmax_attention_rows_in_place.
                use rayon::prelude::*;
                scores.par_chunks_mut(s).for_each(|row| {
                    let m = row.iter().copied().fold(f64::NEG_INFINITY, f64::max);
                    let mut sum = 0.0;
                    for x in row.iter_mut() {
                        *x = (*x - m).exp();
                        sum += *x;
                    }
                    for x in row.iter_mut() {
                        *x /= sum;
                    }
                });
                let sc_meta = TensorMeta::from_shape(vec![bh, s, s], DType::F64, Device::Cpu);
                let v_meta = TensorMeta::from_shape(vec![bh, s, hd], DType::F64, Device::Cpu);
                black_box(
                    ft_kernel_cpu::bmm_tensor_contiguous_f64(&scores, &v, &sc_meta, &v_meta)
                        .unwrap(),
                );
            });
        });
        group.bench_function(&format!("flash_{bh}x{s}x{hd}"), |b| {
            b.iter(|| {
                black_box(ft_kernel_cpu::sdpa_forward_f64(
                    &q, &k, &v, bh, s, s, hd, hd, scale, false,
                ))
            });
        });
    }
    group.finish();
}

fn bench_multihead_attention_train(c: &mut Criterion) {
    let mut group = c.benchmark_group("multihead_attention");
    for &(b_, s_, e_, h_) in &[(8usize, 64usize, 128usize, 8usize), (1, 512, 128, 8)] {
        group.bench_function(&format!("train_{b_}x{s_}x{e_}_h{h_}"), |bch| {
            bch.iter_batched(
                || make_mha_case(b_, s_, e_, h_),
                |(mut session, attention, input)| {
                    let out = attention.forward(&mut session, input).expect("forward");
                    let loss = session.tensor_sum(out).expect("sum");
                    black_box(session.tensor_backward(loss).expect("backward"))
                },
                BatchSize::SmallInput,
            );
        });
    }
    group.finish();
}

criterion_group!(
    benches,
    bench_multihead_attention,
    bench_multihead_attention_nograd,
    bench_multihead_attention_train,
    bench_attention_core,
    bench_lstm_forward,
    bench_gru_forward,
    bench_rnn_forward
);
criterion_main!(benches);

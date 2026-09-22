//! Independent public-API regressions for GitHub issue #1.
//!
//! The oracle must not call production quantization or dot-product helpers.
//! Run with FT_TEST_REQUIRE_AVX512_VNNI=1 to reject a silently skipped ISA path.
#![allow(clippy::cast_possible_truncation, clippy::cast_precision_loss)]

use ft_kernel_cpu::{linear_int8_dynamic_f32, quantize_rows_i8};

fn reference_quantize(x: &[f32], m: usize, k: usize) -> (Vec<i8>, Vec<f32>) {
    assert!(k > 0);
    assert_eq!(x.len(), m * k);
    let mut bytes = Vec::with_capacity(x.len());
    let mut scales = Vec::with_capacity(m);
    for row in x.chunks_exact(k) {
        let amax = row.iter().fold(0.0_f32, |a, &x| a.max(x.abs()));
        let scale = if amax > 0.0 { amax / 127.0 } else { 1.0 };
        scales.push(scale);
        bytes.extend(row.iter().map(|&x| {
            // Division, not multiplication by a precomputed reciprocal.
            (x / scale).round_ties_even().clamp(-127.0, 127.0) as i8
        }));
    }
    (bytes, scales)
}

fn assert_bits(actual: &[f32], expected: &[f32], context: &str) {
    assert_eq!(actual.len(), expected.len(), "{context}");
    for (index, (&actual, &expected)) in actual.iter().zip(expected).enumerate() {
        assert_eq!(
            actual.to_bits(),
            expected.to_bits(),
            "{context}: index={index}, actual={actual:?}, expected={expected:?}"
        );
    }
}

fn check_quantization(x: &[f32], m: usize, k: usize) {
    let (expected_bytes, expected_scales) = reference_quantize(x, m, k);
    let (actual_bytes, actual_scales) = quantize_rows_i8(x, m, k);
    assert_eq!(actual_bytes, expected_bytes, "m={m}, k={k}");
    assert_bits(&actual_scales, &expected_scales, "activation scales");
}

#[test]
fn requested_vnni_coverage_must_be_available() {
    if std::env::var_os("FT_TEST_REQUIRE_AVX512_VNNI").is_none() {
        return;
    }
    #[cfg(target_arch = "x86_64")]
    assert!(
        std::arch::is_x86_feature_detected!("avx512f")
            && std::arch::is_x86_feature_detected!("avx512bw")
            && std::arch::is_x86_feature_detected!("avx512vnni"),
        "AVX-512F/BW/VNNI coverage was requested but is unavailable"
    );
    #[cfg(not(target_arch = "x86_64"))]
    panic!("AVX-512/VNNI coverage requires an x86_64 test host");
}

#[test]
fn quantization_matches_independent_oracle_at_every_half_tie() {
    let mut boundaries = vec![-127.0_f32, -0.0, 0.0, 127.0];
    for integer in -127_i16..127 {
        let half = f32::from(integer) + 0.5;
        // Unlike integer bit +/- 1 helpers, these have the right direction
        // for negative values as well as positive ones.
        boundaries.extend([half.next_down(), half, half.next_up()]);
    }
    for factor in [0.003_906_25_f32, 0.1, 1.0, 3.7, 65_536.0] {
        let row: Vec<f32> = boundaries.iter().map(|&x| x * factor).collect();
        check_quantization(&row, 1, row.len());
        for k in [1, 7, 15, 16, 17, 31, 32, 63, 64, 65, 383, 384, 385, 1536] {
            for offset in [0, 11, 379, 751] {
                let mut row: Vec<f32> = (0..k)
                    .map(|i| boundaries[(i + offset) % boundaries.len()] * factor)
                    .collect();
                row[k - 1] = 127.0 * factor;
                check_quantization(&row, 1, k);
            }
        }
    }
}

#[test]
fn quantization_preserves_nonfinite_subnormal_and_zero_rows() {
    for k in [1, 15, 16, 17, 63, 64, 65, 384] {
        let mut x = Vec::new();
        for special in [
            0.0,
            -0.0,
            f32::from_bits(1),
            -f32::from_bits(1),
            f32::MIN_POSITIVE,
            f32::MAX,
            f32::NAN,
            f32::INFINITY,
            f32::NEG_INFINITY,
        ] {
            let mut row = vec![special; k];
            if k > 1 {
                row[0] = 0.0;
            }
            x.extend(row);
        }
        check_quantization(&x, 9, k);
    }
    check_quantization(&[], 0, 17);
}

fn activations(m: usize, k: usize) -> Vec<f32> {
    (0..m * k)
        .map(|i| match (i / k) % 6 {
            1 => 0.0,
            2 => 127.0,
            3 => -127.0,
            _ => {
                let value = i16::try_from((i * 73 + 19) % 4093).unwrap() - 2046;
                f32::from(value) * 0.001_937_5
            }
        })
        .collect()
}

fn weights(n: usize, k: usize) -> Vec<i8> {
    (0..n * k)
        .map(|i| {
            // Include -128, not just the symmetric quantizer's [-127,127].
            match (i / k) % 5 {
                0 => -128,
                1 => 127,
                2 => 0,
                _ => i8::try_from(i16::try_from((i * 29 + 7) % 256).unwrap() - 128).unwrap(),
            }
        })
        .collect()
}

#[allow(clippy::too_many_arguments)]
fn reference_linear(
    x: &[f32],
    m: usize,
    k: usize,
    w: &[i8],
    w_scales: &[f32],
    n: usize,
    bias: Option<&[f32]>,
) -> Vec<f32> {
    let (q, scales) = reference_quantize(x, m, k);
    let mut out = Vec::with_capacity(m * n);
    for row in 0..m {
        for column in 0..n {
            let acc: i64 = (0..k)
                .map(|j| i64::from(q[row * k + j]) * i64::from(w[column * k + j]))
                .sum();
            let acc = i32::try_from(acc).expect("fixture dot must fit i32");
            // Preserve BOTH multiplication roundings, then the separate add.
            let mut value = acc as f32 * scales[row] * w_scales[column];
            if let Some(bias) = bias {
                value += bias[column];
            }
            out.push(value);
        }
    }
    out
}

fn check_linear(m: usize, k: usize, n: usize) {
    let x = activations(m, k);
    let w = weights(n, k);
    let scales: Vec<f32> = (0..n)
        .map(|i| [0.000_37, 0.031_25, 0.123_456_7, 1.7][i % 4])
        .collect();
    let biases: Vec<f32> = (0..n)
        .map(|i| [0.0, -0.0, f32::from_bits(1), 0.001_937_5, -0.003_71][i % 5])
        .collect();
    for bias in [None, Some(biases.as_slice())] {
        let expected = reference_linear(&x, m, k, &w, &scales, n, bias);
        let actual = linear_int8_dynamic_f32(&x, m, k, &w, &scales, n, bias);
        assert_bits(
            &actual,
            &expected,
            &format!("m={m}, k={k}, n={n}, bias={}", bias.is_some()),
        );
    }
}

#[test]
fn linear_matches_independent_oracle_across_tiles_and_minilm_shapes() {
    for threads in [1, 4] {
        rayon::ThreadPoolBuilder::new()
            .num_threads(threads)
            .build()
            .unwrap()
            .install(|| {
                for (m, k, n) in [
                    (1, 1, 1),
                    (1, 63, 1),
                    (1, 64, 5),
                    (2, 65, 3),
                    (3, 127, 7),
                    (4, 64, 4),
                    (5, 65, 5),
                    (6, 129, 8),
                    (7, 127, 9),
                    (8, 384, 384),
                    (5, 1536, 384),
                    (4, 384, 1536),
                    (17, 383, 5),
                    (18, 385, 7),
                    (19, 1537, 9),
                    (24, 384, 9),
                    (8, 1023, 4),
                    (8, 1024, 4),
                    (0, 64, 5),
                ] {
                    check_linear(m, k, n);
                }
            });
    }
}

#[test]
#[should_panic(expected = "bias length must equal n")]
fn linear_rejects_malformed_bias_before_dispatch() {
    let _ = linear_int8_dynamic_f32(&[1.0; 64], 1, 64, &[1; 256], &[1.0; 4], 4, Some(&[]));
}

/// Compare release executables in alternating baseline/candidate order. This
/// measures the public LINEAR call, not whole-model or CASS wall-clock speed.
#[test]
#[ignore = "manual release-mode microbenchmark, not a correctness timing gate"]
fn linear_microbenchmark() {
    use std::hint::black_box;
    use std::time::Instant;

    requested_vnni_coverage_must_be_available();
    let threads: usize = std::env::var("FT_INT8_BENCH_THREADS")
        .map_or(Ok(12), |value| value.parse())
        .expect("FT_INT8_BENCH_THREADS must be a positive integer");
    assert!(threads > 0);
    rayon::ThreadPoolBuilder::new()
        .num_threads(threads)
        .build()
        .unwrap()
        .install(|| {
            for (m, k, n) in [
                (1, 384, 384),
                (3, 384, 384),
                (8, 384, 384),
                (32, 384, 384),
                (128, 384, 384),
                (128, 384, 1536),
                (128, 1536, 384),
            ] {
                let x = activations(m, k);
                let w = weights(n, k);
                let scales = vec![0.001_937_5; n];
                let bias = vec![0.000_37; n];
                let expected = reference_linear(&x, m, k, &w, &scales, n, Some(&bias));
                let warm = linear_int8_dynamic_f32(&x, m, k, &w, &scales, n, Some(&bias));
                assert_bits(&warm, &expected, "microbenchmark fixture");
                let digest = warm.iter().fold(0_u64, |hash, value| {
                    hash.rotate_left(7) ^ u64::from(value.to_bits())
                });
                for _ in 0..4 {
                    black_box(linear_int8_dynamic_f32(&x, m, k, &w, &scales, n, Some(&bias)));
                }
                let calls = 20;
                let start = Instant::now();
                for _ in 0..calls {
                    black_box(linear_int8_dynamic_f32(&x, m, k, &w, &scales, n, Some(&bias)));
                }
                let elapsed_ns = start.elapsed().as_nanos();
                println!(
                    "INT8_TIMING m={m} k={k} n={n} threads={threads} calls={calls} elapsed_ns={elapsed_ns} digest={digest:016x}"
                );
            }
        });
}

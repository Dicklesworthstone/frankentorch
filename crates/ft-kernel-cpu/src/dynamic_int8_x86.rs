//! Exact AVX-512 activation rounding and register-blocked VNNI linear kernels.
//!
//! Based on DanielsLoud's A/B/C proposal in FrankenTorch issue #1. ISA detection
//! stays local; no global native-codegen or production measurement switches.
#![allow(unsafe_code)]
#![allow(clippy::cast_possible_truncation, clippy::cast_precision_loss)]

use rayon::prelude::*;
use std::arch::x86_64::*;

// Bound the UNSIGNED intermediate, not just the corrected signed dot:
// |sum((a+128)*w)| <= k * 255 * 128. This also bounds every SIMD lane,
// scalar tail partial, weight sum, correction, and final signed result.
// Wider inputs retain the incumbent dispatch rather than entering this kernel.
const MAX_EXACT_K: usize = (i32::MAX as usize) / (255 * 128);
const NR: usize = 4;

#[derive(Clone, Copy)]
pub(super) struct RowQuantizer {
    _private: (),
}

impl RowQuantizer {
    pub(super) fn detect() -> Option<Self> {
        std::arch::is_x86_feature_detected!("avx512f").then_some(Self { _private: () })
    }

    pub(super) fn quantize(self, dst: &mut [i8], row: &[f32], scale: f32) {
        assert_eq!(dst.len(), row.len());
        // SAFETY: only detect() constructs this capability, after AVX-512F
        // detection. Equal lengths and full-vector loop bounds protect stores.
        unsafe { quantize_scaled_row(dst, row, scale) };
    }
}

#[target_feature(enable = "avx512f")]
unsafe fn quantize_scaled_row(dst: &mut [i8], row: &[f32], scale: f32) {
    let scale_vector = _mm512_set1_ps(scale);
    let mut rounded = [0.0_f32; 16];
    let vector_end = row.len() / 16 * 16;
    for index in (0..vector_end).step_by(16) {
        // SAFETY: each iteration reads exactly 16 in-bounds f32 values and
        // stores exactly 16 lanes to the local array, without alignment demands.
        unsafe {
            let values = _mm512_loadu_ps(row.as_ptr().add(index));
            let scaled = _mm512_div_ps(values, scale_vector);
            let integral = _mm512_roundscale_ps::<
                { _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC },
            >(scaled);
            _mm512_storeu_ps(rounded.as_mut_ptr(), integral);
        }
        for (dst, value) in dst[index..index + 16].iter_mut().zip(rounded) {
            // Preserve Rust's NaN/saturating float-to-i8 cast and clamp behavior.
            *dst = value.clamp(-127.0, 127.0) as i8;
        }
    }
    for (dst, &value) in dst[vector_end..].iter_mut().zip(&row[vector_end..]) {
        *dst = super::round_ties_even_f32(value / scale).clamp(-127.0, 127.0) as i8;
    }
}

fn vnni_available() -> bool {
    std::arch::is_x86_feature_detected!("avx512f")
        && std::arch::is_x86_feature_detected!("avx512bw")
        && std::arch::is_x86_feature_detected!("avx512vnni")
}

#[allow(clippy::too_many_arguments)]
pub(super) fn try_linear(
    x: &[i8],
    a_scales: &[f32],
    k: usize,
    weights: &[i8],
    w_scales: &[f32],
    n: usize,
    bias: Option<&[f32]>,
) -> Option<Vec<f32>> {
    let m = a_scales.len();
    if m == 0 || n == 0 || !(64..=MAX_EXACT_K).contains(&k) || !vnni_available() {
        return None;
    }
    // These checks make the safe internal API independently sound, even if a
    // future caller bypasses the public linear entry point's shape checks.
    assert_eq!(m.checked_mul(k), Some(x.len()));
    assert_eq!(n.checked_mul(k), Some(weights.len()));
    assert_eq!(w_scales.len(), n);
    if let Some(bias) = bias {
        assert_eq!(bias.len(), n);
    }
    let out_len = m.checked_mul(n).expect("int8 output shape overflow");
    let sums: Vec<i32> = weights
        .par_chunks(k)
        .map(|row| row.iter().map(|&value| i32::from(value)).sum())
        .collect();
    let mut out = vec![0.0_f32; out_len];
    // Do not collapse an already-small batch from m parallel rows into m/4
    // tasks. The 1x4 specialization still reuses each activation across outputs.
    let block_rows = if m / 4 >= rayon::current_num_threads() {
        4
    } else {
        1
    };
    // block_rows <= m, and m*n was checked above.
    out.par_chunks_mut(block_rows * n)
        .enumerate()
        .for_each(|(block, output)| {
            let start = block * block_rows;
            let rows = output.len() / n;
            let input = &x[start * k..(start + rows) * k];
            let scales = &a_scales[start..start + rows];
            // SAFETY: all ISA features and complete slice shapes were checked
            // above; k is within the proved i32 bound. Each task exclusively
            // owns rows*n outputs. Const row counts keep accumulators indexable
            // at compile time, including the final 1/2/3-row tail.
            unsafe {
                match rows {
                    1 => block_vnni::<1>(input, scales, k, weights, w_scales, &sums, n, bias, output),
                    2 => block_vnni::<2>(input, scales, k, weights, w_scales, &sums, n, bias, output),
                    3 => block_vnni::<3>(input, scales, k, weights, w_scales, &sums, n, bias, output),
                    4 => block_vnni::<4>(input, scales, k, weights, w_scales, &sums, n, bias, output),
                    _ => unreachable!("invalid int8 row tile"),
                }
            }
        });
    Some(out)
}

#[target_feature(enable = "avx512vnni,avx512bw,avx512f")]
unsafe fn dot_with_sum(a: &[i8], b: &[i8], weight_sum: i32) -> i32 {
    debug_assert_eq!(a.len(), b.len());
    debug_assert!(a.len() <= MAX_EXACT_K);
    let vector_end = a.len() / 64 * 64;
    let sign = _mm512_set1_epi8(i8::MIN);
    let mut acc = _mm512_setzero_si512();
    for index in (0..vector_end).step_by(64) {
        // SAFETY: equal lengths and vector_end guarantee two full 64-byte loads.
        unsafe {
            let va = _mm512_loadu_si512(a.as_ptr().add(index).cast());
            let vb = _mm512_loadu_si512(b.as_ptr().add(index).cast());
            acc = _mm512_dpbusd_epi32(acc, _mm512_xor_si512(va, sign), vb);
        }
    }
    let mut total = _mm512_reduce_add_epi32(acc);
    for index in vector_end..a.len() {
        // The correction covers all k values, so the tail must be unsigned too.
        total += (i32::from(a[index]) + 128) * i32::from(b[index]);
    }
    total - 128 * weight_sum
}

#[target_feature(enable = "avx512vnni,avx512bw,avx512f")]
#[allow(clippy::too_many_arguments, clippy::needless_range_loop)]
unsafe fn block_vnni<const ROWS: usize>(
    x: &[i8],
    a_scales: &[f32],
    k: usize,
    weights: &[i8],
    w_scales: &[f32],
    sums: &[i32],
    n: usize,
    bias: Option<&[f32]>,
    output: &mut [f32],
) {
    debug_assert!((1..=4).contains(&ROWS));
    debug_assert_eq!(x.len(), ROWS * k);
    debug_assert_eq!(a_scales.len(), ROWS);
    debug_assert_eq!(weights.len(), n * k);
    debug_assert_eq!(w_scales.len(), n);
    debug_assert_eq!(sums.len(), n);
    debug_assert_eq!(output.len(), ROWS * n);
    debug_assert!(k <= MAX_EXACT_K);
    let vector_end = k / 64 * 64;
    let output_end = n / NR * NR;
    let sign = _mm512_set1_epi8(i8::MIN);
    let zero = _mm512_setzero_si512();
    for first in (0..output_end).step_by(NR) {
        let mut accumulators = [[zero; NR]; ROWS];
        for inner in (0..vector_end).step_by(64) {
            // SAFETY: first+NR<=n, inner+64<=k and every input row is complete.
            // All loads are unaligned; every raw pointer stays within its slice.
            unsafe {
                let w = [
                    _mm512_loadu_si512(weights.as_ptr().add(first * k + inner).cast()),
                    _mm512_loadu_si512(weights.as_ptr().add((first + 1) * k + inner).cast()),
                    _mm512_loadu_si512(weights.as_ptr().add((first + 2) * k + inner).cast()),
                    _mm512_loadu_si512(weights.as_ptr().add((first + 3) * k + inner).cast()),
                ];
                for row in 0..ROWS {
                    let a = _mm512_loadu_si512(x.as_ptr().add(row * k + inner).cast());
                    let unsigned = _mm512_xor_si512(a, sign);
                    for column in 0..NR {
                        accumulators[row][column] =
                            _mm512_dpbusd_epi32(accumulators[row][column], unsigned, w[column]);
                    }
                }
            }
        }
        for row in 0..ROWS {
            for column in 0..NR {
                let out_column = first + column;
                let mut total = _mm512_reduce_add_epi32(accumulators[row][column]);
                for inner in vector_end..k {
                    total += (i32::from(x[row * k + inner]) + 128)
                        * i32::from(weights[out_column * k + inner]);
                }
                total -= 128 * sums[out_column];
                // Do not reassociate the scales, use a reciprocal, or fuse bias.
                let mut value = total as f32 * a_scales[row] * w_scales[out_column];
                if let Some(bias) = bias {
                    value += bias[out_column];
                }
                output[row * n + out_column] = value;
            }
        }
    }
    for column in output_end..n {
        let weight = &weights[column * k..(column + 1) * k];
        for row in 0..ROWS {
            let activation = &x[row * k..(row + 1) * k];
            // SAFETY: same ISA and length/bound invariants as this block.
            let total = unsafe { dot_with_sum(activation, weight, sums[column]) };
            let mut value = total as f32 * a_scales[row] * w_scales[column];
            if let Some(bias) = bias {
                value += bias[column];
            }
            output[row * n + column] = value;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn require_vnni() -> bool {
        if std::env::var_os("FT_TEST_REQUIRE_AVX512_VNNI").is_some() {
            assert!(vnni_available(), "requested VNNI coverage is unavailable");
        }
        vnni_available()
    }

    #[test]
    fn corrected_dot_covers_full_i8_domain_and_bound() {
        if !require_vnni() {
            return;
        }
        for k in [1, 63, 64, 65, 127, 384, 1536, MAX_EXACT_K] {
            for a_value in [i8::MIN, -127, 0, 127] {
                for b_value in [i8::MIN, -127, 0, 127] {
                    let a = vec![a_value; k];
                    let b = vec![b_value; k];
                    let sum = i32::from(b_value) * i32::try_from(k).unwrap();
                    let expected = i64::from(a_value) * i64::from(b_value)
                        * i64::try_from(k).unwrap();
                    // SAFETY: ISA checked, lengths equal, k is within bound.
                    let actual = unsafe { dot_with_sum(&a, &b, sum) };
                    assert_eq!(i64::from(actual), expected, "k={k}, a={a_value}, b={b_value}");
                }
            }
        }
    }

    #[test]
    fn detected_dispatch_is_exercised_and_wider_k_is_declined() {
        if !require_vnni() {
            return;
        }
        assert!(RowQuantizer::detect().is_some());
        let result = try_linear(&[1; 64], &[1.0], 64, &[1; 256], &[1.0; 4], 4, None);
        assert_eq!(result, Some(vec![64.0; 4]));
        // The bound is checked before allocation/shape validation or raw loads.
        assert!(try_linear(&[], &[1.0], MAX_EXACT_K + 1, &[], &[1.0], 1, None).is_none());
    }
}

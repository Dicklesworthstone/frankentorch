//! Conv2d backward with an unmaterialized elementwise upstream mask.

use rayon::prelude::*;

/// Run the generic f64 Conv2d backward after multiplying the incoming gradient
/// and an elementwise loss mask directly into the existing GEMM input layout.
///
/// `incoming` and `mask` both use Conv2d's `[N, C_out, H_out, W_out]` layout.
/// The product is written in the same `N, patch, C_out` gather order consumed by
/// the incumbent dweight GEMM and dinput path. The GEMM reduction and col2im
/// scatter are consequently unchanged.
#[allow(clippy::too_many_arguments)]
#[must_use]
pub fn conv2d_backward_mask_fused_f64(
    incoming: &[f64],
    mask: &[f64],
    padded: &[f64],
    weight_flat: &[f64],
    batch: usize,
    in_ch: usize,
    ph: usize,
    pw: usize,
    kh: usize,
    kw: usize,
    oh: usize,
    ow: usize,
    sh: usize,
    sw: usize,
    out_ch: usize,
    output_mask: [bool; 3],
) -> (Option<Vec<f64>>, Option<Vec<f64>>, Option<Vec<f64>>) {
    let patch_width = in_ch * kh * kw;
    let patch_count = oh * ow;
    let flat = batch * patch_count;
    debug_assert_eq!(incoming.len(), batch * out_ch * patch_count);
    debug_assert_eq!(mask.len(), incoming.len());

    let dout_flat = if output_mask[0] || output_mask[1] {
        super::build_uninit(flat * out_ch, |flat_grad: &mut [f64]| {
            flat_grad
                .par_chunks_mut(out_ch)
                .enumerate()
                .for_each(|(row, destination)| {
                    let n = row / patch_count;
                    let patch = row % patch_count;
                    for (out_channel, slot) in destination.iter_mut().enumerate() {
                        let source = (n * out_ch + out_channel) * patch_count + patch;
                        *slot = incoming[source] * mask[source];
                    }
                });
        })
    } else {
        Vec::new()
    };

    let dweight = output_mask[1].then(|| {
        let panel = super::conv2d_im2col_f64(padded, batch, in_ch, ph, pw, kh, kw, oh, ow, sh, sw);
        super::build_uninit(out_ch * patch_width, |dw: &mut [f64]| {
            if out_ch == 0 || patch_width == 0 || flat == 0 {
                dw.fill(0.0);
                return;
            }
            super::CONV2D_DWEIGHT_GEMMS.with(|count| count.set(count.get() + 1));
            super::gemm::dgemm_tb(out_ch, flat, patch_width, &dout_flat, &panel, dw);
        })
    });

    let dpadded = output_mask[0].then(|| {
        if super::conv2d_dinput_blocked_any(batch, in_ch) {
            return super::conv2d_backward_dinput_direct_f64(
                &dout_flat,
                weight_flat,
                batch,
                in_ch,
                ph,
                pw,
                kh,
                kw,
                oh,
                ow,
                sh,
                sw,
                out_ch,
            );
        }
        let dpanel = super::build_pool_output(flat * patch_width, |dp: &mut [f64]| {
            super::CONV2D_DPANEL_GEMMS.with(|count| count.set(count.get() + 1));
            super::gemm::dgemm(flat, out_ch, patch_width, &dout_flat, weight_flat, dp);
        });
        super::conv2d_col2im_f64(&dpanel, batch, in_ch, ph, pw, kh, kw, oh, ow, sh, sw)
    });

    let dbias = output_mask[2].then(|| {
        let mut dbias = vec![0.0f64; out_ch];
        dbias
            .par_iter_mut()
            .enumerate()
            .for_each(|(out_channel, slot)| {
                let mut sum = 0.0;
                for n in 0..batch {
                    let base = (n * out_ch + out_channel) * patch_count;
                    for patch in 0..patch_count {
                        sum += incoming[base + patch] * mask[base + patch];
                    }
                }
                *slot = sum;
            });
        dbias
    });

    (dpadded, dweight, dbias)
}

#[cfg(test)]
mod tests {
    use super::conv2d_backward_mask_fused_f64;

    fn assert_option_bits(actual: Option<Vec<f64>>, expected: Option<Vec<f64>>) {
        assert_eq!(actual.is_some(), expected.is_some());
        if let (Some(actual), Some(expected)) = (actual, expected) {
            assert!(
                actual
                    .iter()
                    .zip(expected)
                    .all(|(&a, b)| a.to_bits() == b.to_bits())
            );
        }
    }

    fn fixture() -> (Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>) {
        let incoming: Vec<f64> = (0..24).map(|i| (i as f64 - 11.0) * 0.125).collect();
        let padded: Vec<f64> = (0..50).map(|i| (i as f64 - 19.0) * -0.0625).collect();
        let weight: Vec<f64> = (0..18).map(|i| (i as f64 - 7.0) * 0.03125).collect();
        let mask = vec![
            -1.0, 0.0, 0.5, -0.25, 1.0, -0.75, 0.125, 0.0, -1.5, 0.25, -0.5, 1.0, 0.75, -0.125,
            0.0, -1.0, 0.5, -0.25, 1.0, 0.0, -0.5, 0.25, -0.75, 0.125,
        ];
        (incoming, mask, padded, weight)
    }

    #[test]
    fn mixed_signed_mask_matches_materialized_conv2d_backward_bitwise() {
        let (incoming, mask, padded, weight) = fixture();
        let materialized: Vec<f64> = incoming.iter().zip(&mask).map(|(&g, &m)| g * m).collect();
        let expected = super::super::conv2d_backward_masked_f64(
            &materialized,
            &padded,
            &weight,
            2,
            1,
            5,
            5,
            3,
            3,
            3,
            2,
            1,
            1,
            2,
            [true, true, true],
        );
        let actual = conv2d_backward_mask_fused_f64(
            &incoming,
            &mask,
            &padded,
            &weight,
            2,
            1,
            5,
            5,
            3,
            3,
            3,
            2,
            1,
            1,
            2,
            [true, true, true],
        );
        assert_option_bits(actual.0, expected.0);
        assert_option_bits(actual.1, expected.1);
        assert_option_bits(actual.2, expected.2);
    }

    #[test]
    fn all_masked_zero_gradient_matches_materialized_path_with_negative_inputs() {
        let (incoming, _, padded, weight) = fixture();
        let mask = vec![0.0; incoming.len()];
        let expected = super::super::conv2d_backward_masked_f64(
            &mask,
            &padded,
            &weight,
            2,
            1,
            5,
            5,
            3,
            3,
            3,
            2,
            1,
            1,
            2,
            [true, true, true],
        );
        let actual = conv2d_backward_mask_fused_f64(
            &incoming,
            &mask,
            &padded,
            &weight,
            2,
            1,
            5,
            5,
            3,
            3,
            3,
            2,
            1,
            1,
            2,
            [true, true, true],
        );
        assert_option_bits(actual.0, expected.0);
        assert_option_bits(actual.1, expected.1);
        assert_option_bits(actual.2, expected.2);
    }
}

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
            // TILED TRANSPOSE — `frankentorch-t1gph`. This is a transpose: the source is laid out
            // `[n][out_ch][patch]` and the destination `[n][patch][out_ch]`.
            //
            // The previous form handed out one `out_ch`-wide row per task and gathered its
            // sources from `(n*out_ch + oc)*patch_count + patch` — addresses `patch_count * 4`
            // bytes apart, 4 KiB at the f32 lane's shape. Every one of the 5.24M reads therefore
            // landed on its own cache line, and the task granularity was 128 bytes. Same shape as
            // the strided column gather item 274 removed from `lu_solve`.
            //
            // Here `oc` moves OUTSIDE and `patch` INSIDE over a block of patches: the source read
            // runs contiguously along `patch`, and the strided write stays inside a block-sized
            // tile that fits L1. One task per batch plane instead of one per 32 floats.
            //
            // BIT-EXACT by construction: identical products of identical values, only the
            // traversal order changes. MEASURED at the f32 lane's shape (hz4, rayon=16, min of 9):
            // 4.017 ms -> 2.784 ms, 1.4429x, bitwise match confirmed in the probe.
            //
            // Note the frame is SMALLER than a subtraction suggested. The budget-sweep arm timed
            // the dinput kernel alone at 8.4-8.9 ms against a 27.0 ms fused dinput-only arm, and
            // attributing that ~18 ms gap here would have been wrong by 4x — the build is 4.0 ms.
            // Item 141: a residual is not a measurement of whatever you name it.
            const PATCH_BLOCK: usize = 64;
            flat_grad
                .par_chunks_mut(patch_count * out_ch)
                .enumerate()
                .for_each(|(n, plane)| {
                    let mut p0 = 0;
                    while p0 < patch_count {
                        let p1 = (p0 + PATCH_BLOCK).min(patch_count);
                        for out_channel in 0..out_ch {
                            let base = (n * out_ch + out_channel) * patch_count;
                            for patch in p0..p1 {
                                plane[patch * out_ch + out_channel] =
                                    incoming[base + patch] * mask[base + patch];
                            }
                        }
                        p0 = p1;
                    }
                });
        })
    } else {
        Vec::new()
    };

    let dweight = output_mask[1].then(|| {
        // STREAMED dweight — `frankentorch-hi9r6`. This block is a SECOND COPY of the generic
        // backward's panel GEMM: the fused masked route does not delegate to
        // `conv2d_backward_f64`, so a toggle wired only there reaches this lane's `dweight`
        // never, and the paired lane measures the shipped path against itself. That is exactly
        // what the first attempt did — a 1.017x that was noise on a branch that never ran.
        if let Some(streamed) = super::conv2d_dweight_streamed_f64_if_enabled(
            &dout_flat, padded, batch, in_ch, ph, pw, kh, kw, oh, ow, sh, sw, out_ch,
        ) {
            return streamed;
        }
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

/// f32 mirror of [`conv2d_backward_mask_fused_f64`] —
/// `frankentorch-conv2d-mask-fusion-f64-only-iq1j1`.
///
/// WHY IT EXISTS, measured rather than assumed. `try_fuse_conv2d_loss_mask` in `ft-api` collapses
/// `conv2d(x, w) * mask` into ONE tape node whose backward is the fused kernel above — and its
/// gate reads `dtype != DType::F64 -> return Ok(None)`. f32 therefore falls off the fused route
/// onto the generic one: a separate `mul` tape node, a materialised `conv_output * mask`
/// intermediate, and an f64->f32 downcast of the incoming gradient. On the board that costs
///
/// ```text
/// conv2d_f32          26.260 ms vs PyTorch 25.704 ms   1.02x SLOWER   (summed, no mask)
/// conv2d_f32_masked   70.410 ms vs PyTorch 25.371 ms   2.78x SLOWER   (masked)
/// ```
///
/// — the same op at the same shape and dtype, differing only by the mask multiply
/// (commit `ffe22c15`, 64 rounds, live PyTorch 2.12.1+cpu co-process, our arm's A/A null 1.001).
///
/// `conv2d_backward_masked_f32` is NOT this function's predecessor despite the name: it takes a
/// PRE-MULTIPLIED `dout` and has no mask parameter, its "masked" meaning `output_mask =
/// [d_input, d_weight, d_bias]` — item 178's `needs_input_grad` gating, a different concept.
///
/// BIT-EXACTNESS. Identical in structure to the f64 original, and bit-identical to computing
/// `incoming * mask` into a materialised buffer and calling the generic f32 backward on it: the
/// product is formed elementwise in the same `N, patch, C_out` gather order the incumbent dweight
/// GEMM and dinput path already consume, so no reduction order changes and no value is
/// reassociated. Fusing removes a buffer, not an operation.
///
/// GATE CHOICE, deliberate and different from `conv2d_backward_masked_f32`'s. That function uses
/// `conv2d_dinput_blocked_selected` (images only); the f64 fused path uses
/// `conv2d_dinput_blocked_any` (images OR channel groups). This mirrors the f64 one, because the
/// image-only gate is what left `conv2d_masked_train` on the old panel round trip at batch 8 while
/// its batch-16 twin ran faster. At this lane's batch 160 the two agree, so the choice is
/// invisible here and matters only for small batches.
#[allow(clippy::too_many_arguments)]
#[must_use]
pub fn conv2d_backward_mask_fused_f32(
    incoming: &[f32],
    mask: &[f32],
    padded: &[f32],
    weight_flat: &[f32],
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
) -> (Option<Vec<f32>>, Option<Vec<f32>>, Option<Vec<f32>>) {
    let patch_width = in_ch * kh * kw;
    let patch_count = oh * ow;
    let flat = batch * patch_count;
    debug_assert_eq!(incoming.len(), batch * out_ch * patch_count);
    debug_assert_eq!(mask.len(), incoming.len());

    let dout_flat = if output_mask[0] || output_mask[1] {
        super::build_uninit(flat * out_ch, |flat_grad: &mut [f32]| {
            // TILED TRANSPOSE — `frankentorch-t1gph`. This is a transpose: the source is laid out
            // `[n][out_ch][patch]` and the destination `[n][patch][out_ch]`.
            //
            // The previous form handed out one `out_ch`-wide row per task and gathered its
            // sources from `(n*out_ch + oc)*patch_count + patch` — addresses `patch_count * 4`
            // bytes apart, 4 KiB at the f32 lane's shape. Every one of the 5.24M reads therefore
            // landed on its own cache line, and the task granularity was 128 bytes. Same shape as
            // the strided column gather item 274 removed from `lu_solve`.
            //
            // Here `oc` moves OUTSIDE and `patch` INSIDE over a block of patches: the source read
            // runs contiguously along `patch`, and the strided write stays inside a block-sized
            // tile that fits L1. One task per batch plane instead of one per 32 floats.
            //
            // BIT-EXACT by construction: identical products of identical values, only the
            // traversal order changes. MEASURED at the f32 lane's shape (hz4, rayon=16, min of 9):
            // 4.017 ms -> 2.784 ms, 1.4429x, bitwise match confirmed in the probe.
            //
            // Note the frame is SMALLER than a subtraction suggested. The budget-sweep arm timed
            // the dinput kernel alone at 8.4-8.9 ms against a 27.0 ms fused dinput-only arm, and
            // attributing that ~18 ms gap here would have been wrong by 4x — the build is 4.0 ms.
            // Item 141: a residual is not a measurement of whatever you name it.
            const PATCH_BLOCK: usize = 64;
            flat_grad
                .par_chunks_mut(patch_count * out_ch)
                .enumerate()
                .for_each(|(n, plane)| {
                    let mut p0 = 0;
                    while p0 < patch_count {
                        let p1 = (p0 + PATCH_BLOCK).min(patch_count);
                        for out_channel in 0..out_ch {
                            let base = (n * out_ch + out_channel) * patch_count;
                            for patch in p0..p1 {
                                plane[patch * out_ch + out_channel] =
                                    incoming[base + patch] * mask[base + patch];
                            }
                        }
                        p0 = p1;
                    }
                });
        })
    } else {
        Vec::new()
    };

    let dweight = output_mask[1].then(|| {
        // STREAMED dweight, f32 — `frankentorch-hi9r6`. Same second-copy hazard the f64 half
        // had: this fused entry does NOT delegate to `conv2d_backward_f32`, it keeps its own
        // panel GEMM, so a toggle wired only there would reach this lane never.
        if let Some(streamed) = super::conv2d_dweight_streamed_f32_if_enabled(
            &dout_flat, padded, batch, in_ch, ph, pw, kh, kw, oh, ow, sh, sw, out_ch,
        ) {
            return streamed;
        }
        let panel = super::conv2d_im2col_f32(padded, batch, in_ch, ph, pw, kh, kw, oh, ow, sh, sw);
        super::build_uninit(out_ch * patch_width, |dw: &mut [f32]| {
            if out_ch == 0 || patch_width == 0 || flat == 0 {
                dw.fill(0.0);
                return;
            }
            super::CONV2D_DWEIGHT_GEMMS.with(|count| count.set(count.get() + 1));
            super::gemm::sgemm_tb(out_ch, flat, patch_width, &dout_flat, &panel, dw);
        })
    });

    let dpadded = output_mask[0].then(|| {
        if super::conv2d_dinput_blocked_any(batch, in_ch) {
            return super::conv2d_backward_dinput_direct_f32(
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
        let dpanel = super::build_pool_output(flat * patch_width, |dp: &mut [f32]| {
            super::CONV2D_DPANEL_GEMMS.with(|count| count.set(count.get() + 1));
            super::gemm::sgemm(flat, out_ch, patch_width, &dout_flat, weight_flat, dp);
        });
        super::conv2d_col2im_f32(&dpanel, batch, in_ch, ph, pw, kh, kw, oh, ow, sh, sw)
    });

    let dbias = output_mask[2].then(|| {
        let mut dbias = vec![0.0f32; out_ch];
        dbias
            .par_iter_mut()
            .enumerate()
            .for_each(|(out_channel, slot)| {
                let mut sum = 0.0f32;
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
    use super::{conv2d_backward_mask_fused_f32, conv2d_backward_mask_fused_f64};

    fn assert_option_bits_f32(actual: Option<Vec<f32>>, expected: Option<Vec<f32>>) {
        assert_eq!(actual.is_some(), expected.is_some());
        if let (Some(actual), Some(expected)) = (actual, expected) {
            assert_eq!(actual.len(), expected.len());
            assert!(
                actual
                    .iter()
                    .zip(expected)
                    .all(|(&a, b)| a.to_bits() == b.to_bits())
            );
        }
    }

    /// The f32 fusion must be BIT-IDENTICAL to materialising `incoming * mask` and calling the
    /// generic f32 backward on it — `frankentorch-conv2d-mask-fusion-f64-only-iq1j1`.
    ///
    /// This is the whole licence for relaxing `try_fuse_conv2d_loss_mask`'s F64-only gate. Fusing
    /// removes a BUFFER, not an operation: the product is formed elementwise in the same
    /// `N, patch, C_out` gather order the dweight GEMM and dinput path already consume, so no
    /// reduction is reassociated. A tolerance comparison here would hide exactly the class of bug
    /// this has to exclude, so the assertion is on RAW BITS.
    ///
    /// The mask straddles every branch that matters: negative, positive, and EXACT ZERO entries,
    /// the last because a zeroed row is where a fused path is most tempted to skip work the
    /// materialised one still does.
    #[test]
    fn f32_mixed_signed_mask_matches_materialized_conv2d_backward_bitwise() {
        let incoming: Vec<f32> = (0..24).map(|i| (i as f32 - 11.0) * 0.125).collect();
        let padded: Vec<f32> = (0..50).map(|i| (i as f32 - 19.0) * -0.0625).collect();
        let weight: Vec<f32> = (0..18).map(|i| (i as f32 - 7.0) * 0.03125).collect();
        let mask: Vec<f32> = vec![
            -1.0, 0.0, 0.5, -0.25, 1.0, -0.75, 0.125, 0.0, -1.5, 0.25, -0.5, 1.0, 0.75, -0.125,
            0.0, -1.0, 0.5, -0.25, 1.0, 0.0, -0.5, 0.25, -0.75, 0.125,
        ];
        let materialized: Vec<f32> = incoming.iter().zip(&mask).map(|(&g, &m)| g * m).collect();
        let expected = super::super::conv2d_backward_masked_f32(
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
        let actual = conv2d_backward_mask_fused_f32(
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
        assert_option_bits_f32(actual.0, expected.0);
        assert_option_bits_f32(actual.1, expected.1);
        assert_option_bits_f32(actual.2, expected.2);
    }

    /// Every `output_mask` combination, because the fused path SKIPS building `dout_flat` when
    /// neither dinput nor dweight is wanted, and a skip is where an off-by-one lands silently.
    #[test]
    fn f32_every_output_mask_combination_matches_materialized_bitwise() {
        let incoming: Vec<f32> = (0..24).map(|i| (i as f32 - 11.0) * 0.125).collect();
        let padded: Vec<f32> = (0..50).map(|i| (i as f32 - 19.0) * -0.0625).collect();
        let weight: Vec<f32> = (0..18).map(|i| (i as f32 - 7.0) * 0.03125).collect();
        let mask: Vec<f32> = (0..24).map(|i| ((i % 5) as f32 - 2.0) * 0.5).collect();
        let materialized: Vec<f32> = incoming.iter().zip(&mask).map(|(&g, &m)| g * m).collect();
        for combo in 0..8u8 {
            let om = [combo & 1 != 0, combo & 2 != 0, combo & 4 != 0];
            let expected = super::super::conv2d_backward_masked_f32(
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
                om,
            );
            let actual = conv2d_backward_mask_fused_f32(
                &incoming, &mask, &padded, &weight, 2, 1, 5, 5, 3, 3, 3, 2, 1, 1, 2, om,
            );
            assert_option_bits_f32(actual.0, expected.0);
            assert_option_bits_f32(actual.1, expected.1);
            assert_option_bits_f32(actual.2, expected.2);
        }
    }

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

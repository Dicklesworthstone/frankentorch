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

/// f32 sibling of [`conv2d_backward_mask_fused_f64`]: run the generic f32 Conv2d backward
/// against `incoming * mask` without ever materialising that product on the tape.
///
/// `frankentorch-hi9r6`. Measured on the board, `conv2d_f32_masked` is **114.667 ms against
/// PyTorch's 25.178 ms — 4.55x SLOWER (5.05x under the min estimator)**, the worst conv2d
/// standing there, while `conv2d_f32` (the summed route, same shape) sits at parity. That is the
/// same summed-fine / masked-off-a-cliff asymmetry the f64 lane had until the f64 fusion above
/// closed it, and the cause is the same: an f32 `mul(conv2d(x, w), mask)` builds the product
/// tensor, and its backward writes a full-numel f64 gradient that the conv2d backward then
/// narrows straight back to f32.
///
/// This removes all three: the product tensor, its elementwise backward, and the separate narrow.
///
/// # Why `incoming` is f64 and the multiply happens there
///
/// The tape's gradient space is f64 whatever the tensor dtype, so the unfused route computes
/// `incoming * mask` in **f64** (the mask widened losslessly) and only then narrows the result to
/// f32 for the kernel. Folding the mask in as `(incoming as f32) * mask` would round in the other
/// order and change the bits. The pass below therefore multiplies in f64 exactly where the tape
/// did — `mask[i] as f64` is exact, f32 to f64 being lossless — and narrows exactly once, at the
/// same point. That is what makes this a pure movement of work rather than a tolerance change,
/// and `conv2d_mask_fused_f32_matches_the_unfused_narrow_then_backward` asserts it on the bits.
///
/// The reduction, the GEMM and the col2im scatter are untouched: this hands
/// `conv2d_backward_masked_f32` exactly the `dout` it would have received.
#[allow(clippy::too_many_arguments)]
#[must_use]
pub fn conv2d_backward_mask_fused_f32(
    incoming: &[f64],
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
    debug_assert_eq!(incoming.len(), batch * out_ch * oh * ow);
    debug_assert_eq!(mask.len(), incoming.len());

    // One parallel pass doing what the tape's mul-backward plus the kernel's narrow did in two
    // full-numel passes, one of them f64. `build_uninit` because every element is written.
    let dout = super::build_uninit(incoming.len(), |scored: &mut [f32]| {
        scored.par_iter_mut().enumerate().for_each(|(index, slot)| {
            *slot = (incoming[index] * f64::from(mask[index])) as f32;
        });
    });

    super::conv2d_backward_masked_f32(
        &dout,
        padded,
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
        output_mask,
    )
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

    /// The fused f32 entry must equal what the tape did unfused: multiply in f64, narrow once,
    /// then run the ordinary masked f32 backward. `frankentorch-hi9r6`.
    ///
    /// The rounding order is the whole risk here, so it is the whole test. `(a * b) as f32` and
    /// `(a as f32) * b` are different numbers, and the fused kernel is only a movement of work if
    /// it picks the first. Values are chosen so that difference is REACHABLE rather than
    /// theoretical: the mask carries thirds and sevenths, which are not representable in binary,
    /// and the incoming gradient is scaled so products land off the f32 grid.
    #[test]
    fn conv2d_mask_fused_f32_matches_the_unfused_narrow_then_backward() {
        //          batch in_ch out_ch ph pw kh kw sh sw
        let cases: [[usize; 9]; 5] = [
            [2, 3, 4, 7, 8, 3, 3, 1, 1],
            [1, 1, 1, 5, 5, 3, 3, 1, 1],  // degenerate channels
            [2, 2, 4, 9, 9, 3, 3, 2, 2],  // strided
            [3, 4, 8, 6, 7, 2, 3, 1, 1],  // asymmetric kernel
            [1, 5, 4, 10, 4, 3, 2, 1, 1], // non-square, mixed
        ];
        for (case, &[batch, in_ch, out_ch, ph, pw, kh, kw, sh, sw]) in cases.iter().enumerate() {
            let oh = (ph - kh) / sh + 1;
            let ow = (pw - kw) / sw + 1;
            let n_out = batch * out_ch * oh * ow;
            // Deliberately awkward: thirds and sevenths are inexact in binary, so the product
            // lands between f32 values and the two rounding orders disagree.
            let incoming: Vec<f64> = (0..n_out)
                .map(|i| ((i % 23) as f64 + 1.0) / 3.0 - 3.7)
                .collect();
            let mask: Vec<f32> = (0..n_out)
                .map(|i| (((i % 17) as f32) + 1.0) / 7.0 - 1.3)
                .collect();
            let padded: Vec<f32> = (0..batch * in_ch * ph * pw)
                .map(|i| ((i % 29) as f32) * 0.031 - 0.4)
                .collect();
            let weight: Vec<f32> = (0..out_ch * in_ch * kh * kw)
                .map(|i| ((i % 19) as f32) * 0.047 - 0.35)
                .collect();

            for mask_flags in [
                [true, true, true],
                [true, false, false],
                [false, true, false],
            ] {
                // The unfused route, verbatim: f64 multiply on the tape, then the kernel's narrow.
                let narrowed: Vec<f32> = incoming
                    .iter()
                    .zip(mask.iter())
                    .map(|(&g, &m)| (g * f64::from(m)) as f32)
                    .collect();
                let expected = super::super::conv2d_backward_masked_f32(
                    &narrowed, &padded, &weight, batch, in_ch, ph, pw, kh, kw, oh, ow, sh, sw,
                    out_ch, mask_flags,
                );
                let actual = super::conv2d_backward_mask_fused_f32(
                    &incoming, &mask, &padded, &weight, batch, in_ch, ph, pw, kh, kw, oh, ow, sh,
                    sw, out_ch, mask_flags,
                );
                for (slot, (a, e)) in [actual.0, actual.1, actual.2]
                    .into_iter()
                    .zip([expected.0, expected.1, expected.2])
                    .enumerate()
                {
                    assert_eq!(a.is_some(), e.is_some(), "case {case} slot {slot} presence");
                    if let (Some(a), Some(e)) = (a, e) {
                        for (i, (x, y)) in a.iter().zip(e.iter()).enumerate() {
                            assert_eq!(
                                x.to_bits(),
                                y.to_bits(),
                                "case {case} slot {slot} element {i} mask {mask_flags:?}: \
                                 fused {x} != unfused {y}"
                            );
                        }
                    }
                }
            }
        }
    }

    /// The planted negative: narrowing BEFORE the multiply must be visibly different.
    ///
    /// Without this, the test above could pass against a kernel that rounded in the wrong order
    /// on inputs where both orders happen to agree, and it would be asserting nothing.
    #[test]
    fn conv2d_mask_fused_f32_rounding_order_is_observable() {
        let incoming: Vec<f64> = (0..4096)
            .map(|i| ((i % 23) as f64 + 1.0) / 3.0 - 3.7)
            .collect();
        let mask: Vec<f32> = (0..4096)
            .map(|i| (((i % 17) as f32) + 1.0) / 7.0 - 1.3)
            .collect();
        let multiply_then_narrow: Vec<f32> = incoming
            .iter()
            .zip(mask.iter())
            .map(|(&g, &m)| (g * f64::from(m)) as f32)
            .collect();
        let narrow_then_multiply: Vec<f32> = incoming
            .iter()
            .zip(mask.iter())
            .map(|(&g, &m)| (g as f32) * m)
            .collect();
        let differing = multiply_then_narrow
            .iter()
            .zip(narrow_then_multiply.iter())
            .filter(|(a, b)| a.to_bits() != b.to_bits())
            .count();
        assert!(
            differing > 0,
            "the two rounding orders must differ on this fixture, or \
             conv2d_mask_fused_f32_matches_the_unfused_narrow_then_backward proves nothing"
        );
    }
}

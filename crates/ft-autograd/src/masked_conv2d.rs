//! Metadata and forward arithmetic for the first-order Conv2d loss-mask fusion.

use crate::TensorNodeId;

/// The original Conv2d inputs and geometry needed to attach a following
/// non-grad elementwise loss mask directly to Conv2d's backward.
#[derive(Clone, Copy, Debug)]
pub struct Conv2dMaskPlan {
    pub padded: TensorNodeId,
    pub weight: TensorNodeId,
    pub bias: Option<TensorNodeId>,
    pub batch: usize,
    pub in_channels: usize,
    pub padded_h: usize,
    pub padded_w: usize,
    pub kernel_h: usize,
    pub kernel_w: usize,
    pub output_h: usize,
    pub output_w: usize,
    pub stride_h: usize,
    pub stride_w: usize,
    pub out_channels: usize,
}

/// Compute the forward value of `conv_output * mask` in index order.
///
/// The backward keeps this product virtual until it builds Conv2d's existing
/// `[flat, out_channels]` GEMM input. Keeping this elementwise forward result
/// explicit preserves the public value of `tensor_mul`.
#[must_use]
pub fn multiply_forward_f64(conv_output: &[f64], mask: &[f64]) -> Vec<f64> {
    assert_eq!(conv_output.len(), mask.len());
    conv_output
        .iter()
        .zip(mask)
        .map(|(&value, &scale)| value * scale)
        .collect()
}

#[cfg(test)]
mod tests {
    use super::multiply_forward_f64;

    #[test]
    fn mixed_signed_mask_preserves_each_elementwise_product() {
        let output = [2.0, -3.0, 4.5, -7.0];
        let mask = [-1.0, 0.0, 0.25, -0.5];
        let fused = multiply_forward_f64(&output, &mask);
        let expected: Vec<f64> = output.iter().zip(mask).map(|(&a, &b)| a * b).collect();
        assert!(
            fused
                .iter()
                .zip(expected)
                .all(|(&actual, expected)| actual.to_bits() == expected.to_bits())
        );
    }
}

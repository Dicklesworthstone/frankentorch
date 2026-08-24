//! Single-kernel ISA probe for the exact-tiled f64 `avg_pool2d` backward path.
//!
//! This deliberately measures no framework setup beyond constructing the fixed input once.
//! The output digest keeps the repeated kernel calls observable for instruction counters.

use std::hint::black_box;

const BATCH: usize = 4;
const CHANNELS: usize = 16;
const HEIGHT: usize = 64;
const WIDTH: usize = 64;

fn iterations() -> usize {
    std::env::var("FT_AVX2_POOL_ICOUNT_ITERS")
        .ok()
        .and_then(|raw| raw.parse().ok())
        .filter(|&count| count > 0)
        .unwrap_or(64)
}

fn input_gradient() -> Vec<f64> {
    let pooled = BATCH * CHANNELS * (HEIGHT / 2) * (WIDTH / 2);
    (0..pooled)
        .map(|index| (index % 97) as f64 * 0.03125 - 1.5)
        .collect()
}

fn hot_kernel(dout: &[f64]) -> Vec<f64> {
    ft_kernel_cpu::avg_pool2d_backward_f64(
        dout,
        BATCH,
        CHANNELS,
        HEIGHT,
        WIDTH,
        2,
        2,
        HEIGHT / 2,
        WIDTH / 2,
        2,
        2,
        0,
        0,
        HEIGHT,
        WIDTH,
        true,
    )
}

fn digest(values: &[f64]) -> u64 {
    values.iter().fold(0_u64, |sum, value| {
        sum.wrapping_add(value.to_bits().rotate_left(17))
    })
}

fn main() {
    let dout = input_gradient();
    let rounds = iterations();

    // Warm the Rayon pool and allocator before the counted work begins.
    black_box(hot_kernel(black_box(&dout)));

    let mut checksum = 0_u64;
    for _ in 0..rounds {
        let output = hot_kernel(black_box(&dout));
        checksum ^= digest(black_box(&output));
    }

    println!(
        "avx2_pool_icount rounds={rounds} input=[{BATCH},{CHANNELS},{HEIGHT},{WIDTH}] checksum={checksum:016x}"
    );
}

#[cfg(test)]
mod tests {
    use super::hot_kernel;

    #[test]
    fn exact_tiled_fast_path_preserves_values_and_normalizes_negative_zero() {
        let negative_zero = ft_kernel_cpu::avg_pool2d_backward_f64(
            &[-0.0],
            1,
            1,
            2,
            2,
            2,
            2,
            1,
            1,
            2,
            2,
            0,
            0,
            2,
            2,
            true,
        );
        assert!(
            negative_zero
                .iter()
                .all(|value| value.to_bits() == 0.0_f64.to_bits())
        );

        let output = ft_kernel_cpu::avg_pool2d_backward_f64(
            &[1.0],
            1,
            1,
            2,
            2,
            2,
            2,
            1,
            1,
            2,
            2,
            0,
            0,
            2,
            2,
            true,
        );
        assert_eq!(output, vec![0.25; 4]);

        // Keep the test coupled to the probe dimensions as well as the minimal shape.
        let probe_output = hot_kernel(&vec![1.0; 4 * 16 * 32 * 32]);
        assert_eq!(probe_output.len(), 4 * 16 * 64 * 64);
        assert!(probe_output.iter().all(|value| *value == 0.25));
    }

    #[test]
    fn exact_tiled_fast_path_matches_scalar_reference_bits_across_a_simd_tail() {
        // `ow == 7` exercises one four-lane vector group and three scalar tail windows.
        let (batch, channels, height, width) = (2, 3, 10, 14);
        let (pooled_height, pooled_width) = (height / 2, width / 2);
        let dout: Vec<f64> = (0..batch * channels * pooled_height * pooled_width)
            .map(|index| match index % 17 {
                0 => -0.0,
                1 => f64::NAN,
                _ => (index as f64 - 51.0) * 0.03125,
            })
            .collect();
        let got = ft_kernel_cpu::avg_pool2d_backward_f64(
            &dout,
            batch,
            channels,
            height,
            width,
            2,
            2,
            pooled_height,
            pooled_width,
            2,
            2,
            0,
            0,
            height,
            width,
            true,
        );

        let mut expected = vec![0.0; batch * channels * height * width];
        for plane in 0..batch * channels {
            for oy in 0..pooled_height {
                for ox in 0..pooled_width {
                    let g = 0.0
                        + dout[plane * pooled_height * pooled_width + oy * pooled_width + ox] / 4.0;
                    let row0 = plane * height * width + (oy * 2) * width;
                    let row1 = row0 + width;
                    let col0 = ox * 2;
                    expected[row0 + col0] = g;
                    expected[row0 + col0 + 1] = g;
                    expected[row1 + col0] = g;
                    expected[row1 + col0 + 1] = g;
                }
            }
        }

        assert_eq!(
            got.iter().map(|value| value.to_bits()).collect::<Vec<_>>(),
            expected
                .iter()
                .map(|value| value.to_bits())
                .collect::<Vec<_>>(),
        );
    }
}

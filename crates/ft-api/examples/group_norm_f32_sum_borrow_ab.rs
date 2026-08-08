//! Same-process evidence for the no-grad f32 `functional_group_norm_sum` input
//! clone removal. OLD materializes the input with `to_vec()` before the scalar
//! kernel; NEW borrows it directly. The arms are interleaved, include an A/A
//! null gate, and report a bootstrap CI for the median speedup.
//! Run: `cargo run --release -p ft-api --example group_norm_f32_sum_borrow_ab`

use std::time::Instant;

use ft_kernel_cpu::group_norm_sum_forward_f32;

const REPS: usize = 31;
const BOOTSTRAP_REPS: usize = 2_000;

fn median(mut values: Vec<f64>) -> f64 {
    values.sort_by(f64::total_cmp);
    values[values.len() / 2]
}

fn next_random(state: &mut u64) -> u64 {
    *state ^= *state << 13;
    *state ^= *state >> 7;
    *state ^= *state << 17;
    *state
}

fn median_ratio_ci(numerator: &[f64], denominator: &[f64]) -> (f64, f64, f64) {
    assert_eq!(numerator.len(), denominator.len());
    let point = median(numerator.to_vec()) / median(denominator.to_vec());
    let mut samples = Vec::with_capacity(BOOTSTRAP_REPS);
    let mut state = 0x4d59_5df4_d0f3_3173_u64;
    for _ in 0..BOOTSTRAP_REPS {
        let mut left = Vec::with_capacity(numerator.len());
        let mut right = Vec::with_capacity(denominator.len());
        for _ in 0..numerator.len() {
            let index = (next_random(&mut state) as usize) % numerator.len();
            left.push(numerator[index]);
            right.push(denominator[index]);
        }
        samples.push(median(left) / median(right));
    }
    samples.sort_by(f64::total_cmp);
    let lower = samples[BOOTSTRAP_REPS * 25 / 1_000];
    let upper = samples[BOOTSTRAP_REPS * 975 / 1_000];
    (point, lower, upper)
}

fn elapsed_ms<F: FnOnce() -> f32>(operation: F) -> f64 {
    let started = Instant::now();
    let result = operation();
    std::hint::black_box(result);
    started.elapsed().as_secs_f64() * 1_000.0
}

fn main() {
    let (batch, channels, height, width, groups) = (8usize, 64usize, 28usize, 28usize, 32usize);
    let spatial = height * width;
    let input: Vec<f32> = (0..batch * channels * spatial)
        .map(|index| ((index % 877) as f32 - 400.0) * 0.002)
        .collect();
    let weight: Vec<f32> = (0..channels)
        .map(|channel| 1.0 + (channel % 13) as f32 * 0.01)
        .collect();
    let bias: Vec<f32> = (0..channels)
        .map(|channel| (channel % 7) as f32 * 0.01 - 0.03)
        .collect();
    let scalar = |values: &[f32]| {
        group_norm_sum_forward_f32(
            values,
            Some(&weight),
            Some(&bias),
            batch,
            groups,
            channels / groups,
            spatial,
            1e-5,
        )
    };

    let old_result = scalar(&input.to_vec());
    let new_result = scalar(&input);
    assert_eq!(
        old_result, new_result,
        "clone and borrow arms must agree exactly"
    );

    // Warm both arms before collecting the paired samples.
    for _ in 0..4 {
        std::hint::black_box(scalar(&input.to_vec()));
        std::hint::black_box(scalar(&input));
    }

    let mut null_a = Vec::with_capacity(REPS);
    let mut null_b = Vec::with_capacity(REPS);
    let mut old = Vec::with_capacity(REPS);
    let mut new = Vec::with_capacity(REPS);
    for sample in 0..REPS {
        // A/A measures the benchmark's own order/noise envelope. Alternate the
        // order so cache warmth cannot favor one label consistently.
        if sample.is_multiple_of(2) {
            null_a.push(elapsed_ms(|| scalar(&input)));
            null_b.push(elapsed_ms(|| scalar(&input)));
            old.push(elapsed_ms(|| scalar(&input.to_vec())));
            new.push(elapsed_ms(|| scalar(&input)));
        } else {
            null_b.push(elapsed_ms(|| scalar(&input)));
            null_a.push(elapsed_ms(|| scalar(&input)));
            new.push(elapsed_ms(|| scalar(&input)));
            old.push(elapsed_ms(|| scalar(&input.to_vec())));
        }
    }

    let (null_ratio, null_low, null_high) = median_ratio_ci(&null_a, &null_b);
    let (speedup, speedup_low, speedup_high) = median_ratio_ci(&old, &new);
    let report = format!(
        "executing_elf_sha256={}\nworkload=group_norm_sum_f32_no_grad [{batch},{channels},{height},{width}] groups={groups} reps={REPS}\na_a_median_ratio={null_ratio:.4} ci95=[{null_low:.4},{null_high:.4}] gate={}\nold_clone_ms={:.4} new_borrow_ms={:.4} old_over_new={speedup:.4} ci95=[{speedup_low:.4},{speedup_high:.4}] decision={}\n",
        ft_api::harness_provenance::executing_elf_sha256(),
        if null_low <= 1.0 && null_high >= 1.0 {
            "PASS"
        } else {
            "FAIL"
        },
        median(old),
        median(new),
        if null_low <= 1.0 && null_high >= 1.0 && speedup_low > 1.0 {
            "KEEP"
        } else {
            "REJECT"
        },
    );
    // RCH may detach child stdout after execution; its worker target directory
    // is synchronized back, so retain the exact same-invocation decision there.
    if let Ok(target_dir) = std::env::var("CARGO_TARGET_DIR") {
        let _ = std::fs::write(
            format!("{target_dir}/group_norm_f32_sum_borrow_ab.txt"),
            &report,
        );
    }
    print!("{report}");
}

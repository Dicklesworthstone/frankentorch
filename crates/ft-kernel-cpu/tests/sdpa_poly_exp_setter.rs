//! `set_sdpa_poly_exp` is a process-global switch read by `sdpa_forward_f32`.
//!
//! This lives in its own integration-test binary (its own process) on purpose: a unit
//! test inside `lib.rs` that flipped the flag would race the other `sdpa_forward_f32`
//! tests, which cargo runs concurrently in one process. Do not move it into `mod tests`.

use ft_kernel_cpu::{sdpa_forward_f32, sdpa_poly_exp, set_sdpa_poly_exp};

fn fill(seed: u64, n: usize) -> Vec<f32> {
    let mut s = seed | 1;
    (0..n)
        .map(|_| {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            ((s >> 40) as f32 / 16_777_216.0) - 0.5
        })
        .collect()
}

#[test]
fn setter_overrides_default_and_stays_within_the_published_accuracy_budget() {
    // The harness must not export FT_SDPA_POLY_EXP; if it did, the default assertion
    // below would be meaningless rather than wrong, so state the precondition.
    assert!(
        std::env::var("FT_SDPA_POLY_EXP").is_err(),
        "this test asserts the OFF default; unset FT_SDPA_POLY_EXP to run it"
    );
    assert!(!sdpa_poly_exp(), "poly softmax must default to off");

    let (num_bh, seq_q, seq_k, d) = (3usize, 96usize, 96usize, 16usize);
    let q = fill(1, num_bh * seq_q * d);
    let k = fill(7, num_bh * seq_k * d);
    let v = fill(13, num_bh * seq_k * d);
    let scale = 1.0 / (d as f32).sqrt();

    let off = sdpa_forward_f32(&q, &k, &v, num_bh, seq_q, seq_k, d, d, scale, false);

    set_sdpa_poly_exp(true);
    assert!(sdpa_poly_exp(), "setter must win over the env default");
    let on = sdpa_forward_f32(&q, &k, &v, num_bh, seq_q, seq_k, d, d, scale, false);

    set_sdpa_poly_exp(false);
    assert!(!sdpa_poly_exp(), "setter must be able to turn it back off");
    let off_again = sdpa_forward_f32(&q, &k, &v, num_bh, seq_q, seq_k, d, d, scale, false);
    assert_eq!(off, off_again, "disabling must restore the exact libm path");

    // `O` is a probability-weighted mean of zero-mean `V` rows, so per-component
    // relative error is dominated by cancellation IN THE REFERENCE. Measure the
    // vector-relative error, which is what the published budget bounds (1.425e-6).
    let num: f64 = off
        .iter()
        .zip(on.iter())
        .map(|(a, b)| f64::from(a - b).powi(2))
        .sum::<f64>()
        .sqrt();
    let den: f64 = off.iter().map(|a| f64::from(*a).powi(2)).sum::<f64>().sqrt();
    let rel = num / den;
    assert!(
        rel <= 1e-5,
        "poly softmax exceeded the accuracy budget: vector rel err {rel:.3e} > 1e-5"
    );
}

//! QR stage-split probe for `frankentorch-ct2yy`.
//!
//! The profiled path calls the same compact-WY QR implementation as
//! `qr_contiguous_f64`; this example asserts bit-for-bit Q/R equality and then
//! reports the measured stage split for the current worker.
//!
//! Run:
//!   rch exec -- cargo run -q -p ft-kernel-cpu --release --example qr_probe

use ft_core::{DType, Device, TensorMeta};
use ft_kernel_cpu::{QrResult, QrStageTimings, qr_contiguous_f64, qr_contiguous_f64_stage_profile};

fn deterministic_matrix(m: usize, n: usize) -> Vec<f64> {
    let mut a = vec![0.0f64; m * n];
    for i in 0..m {
        for j in 0..n {
            a[i * n + j] = (((i * 53 + j * 31) % 97) as f64 - 48.0) * 0.1;
        }
    }
    for j in 0..m.min(n) {
        a[j * n + j] += m as f64;
    }
    a
}

fn fnv1a_values(values: &[f64], mut hash: u64) -> u64 {
    for value in values {
        for byte in value.to_bits().to_le_bytes() {
            hash ^= byte as u64;
            hash = hash.wrapping_mul(0x100000001b3);
        }
    }
    hash
}

fn qr_digest(result: &QrResult) -> u64 {
    let mut hash = 0xcbf29ce484222325u64;
    for value in [result.m as u64, result.n as u64] {
        for byte in value.to_le_bytes() {
            hash ^= byte as u64;
            hash = hash.wrapping_mul(0x100000001b3);
        }
    }
    hash = fnv1a_values(&result.q, hash);
    fnv1a_values(&result.r, hash)
}

fn assert_same_bits(label: &str, expected: &QrResult, actual: &QrResult) {
    assert_eq!(expected.m, actual.m, "{label}: m changed");
    assert_eq!(expected.n, actual.n, "{label}: n changed");
    assert_eq!(expected.q.len(), actual.q.len(), "{label}: q len changed");
    assert_eq!(expected.r.len(), actual.r.len(), "{label}: r len changed");
    for (idx, (&lhs, &rhs)) in expected.q.iter().zip(&actual.q).enumerate() {
        assert_eq!(
            lhs.to_bits(),
            rhs.to_bits(),
            "{label}: q[{idx}] bit pattern changed"
        );
    }
    for (idx, (&lhs, &rhs)) in expected.r.iter().zip(&actual.r).enumerate() {
        assert_eq!(
            lhs.to_bits(),
            rhs.to_bits(),
            "{label}: r[{idx}] bit pattern changed"
        );
    }
}

fn add_timings(total: &mut QrStageTimings, sample: QrStageTimings) {
    total.copy_zeroing_ns += sample.copy_zeroing_ns;
    total.panel_and_t_ns += sample.panel_and_t_ns;
    total.trailing_r_ns += sample.trailing_r_ns;
    total.reverse_q_ns += sample.reverse_q_ns;
    total.total_ns += sample.total_ns;
}

fn averaged(total: QrStageTimings, iterations: u128) -> QrStageTimings {
    QrStageTimings {
        copy_zeroing_ns: total.copy_zeroing_ns / iterations,
        panel_and_t_ns: total.panel_and_t_ns / iterations,
        trailing_r_ns: total.trailing_r_ns / iterations,
        reverse_q_ns: total.reverse_q_ns / iterations,
        total_ns: total.total_ns / iterations,
    }
}

fn pct(part: u128, total: u128) -> f64 {
    if total == 0 {
        0.0
    } else {
        part as f64 * 100.0 / total as f64
    }
}

fn ms(ns: u128) -> f64 {
    ns as f64 / 1_000_000.0
}

fn run_case(m: usize, n: usize, iterations: usize) {
    let label = format!("qr_f64_{m}x{n}_reduced");
    let data = deterministic_matrix(m, n);
    let meta = TensorMeta::from_shape(vec![m, n], DType::F64, Device::Cpu);

    let expected = qr_contiguous_f64(&data, &meta, true).expect("baseline QR should succeed");
    let profile =
        qr_contiguous_f64_stage_profile(&data, &meta, true).expect("profiled QR should succeed");
    assert!(
        profile.used_blocked_path,
        "{label}: expected blocked QR path"
    );
    assert_same_bits(&label, &expected, &profile.result);

    let digest = qr_digest(&profile.result);
    println!("{label} digest={digest:#018x} bit_equal=true");

    let mut timing_sum = QrStageTimings::default();
    add_timings(&mut timing_sum, profile.timings);
    for _ in 1..iterations {
        let sample = qr_contiguous_f64_stage_profile(&data, &meta, true)
            .expect("profiled QR should succeed");
        assert_same_bits(&label, &expected, &sample.result);
        add_timings(&mut timing_sum, sample.timings);
    }
    let avg = averaged(timing_sum, iterations as u128);
    println!(
        "{label} avg_total_ms={:.3} copy_zeroing={:.3}ms/{:.1}% panel_t={:.3}ms/{:.1}% trailing_r={:.3}ms/{:.1}% reverse_q={:.3}ms/{:.1}% unaccounted={:.3}ms/{:.1}%",
        ms(avg.total_ns),
        ms(avg.copy_zeroing_ns),
        pct(avg.copy_zeroing_ns, avg.total_ns),
        ms(avg.panel_and_t_ns),
        pct(avg.panel_and_t_ns, avg.total_ns),
        ms(avg.trailing_r_ns),
        pct(avg.trailing_r_ns, avg.total_ns),
        ms(avg.reverse_q_ns),
        pct(avg.reverse_q_ns, avg.total_ns),
        ms(avg.unaccounted_ns()),
        pct(avg.unaccounted_ns(), avg.total_ns),
    );
}

fn main() {
    println!("threads={}", rayon::current_num_threads());
    run_case(512, 512, 4);
    run_case(2048, 128, 4);
}

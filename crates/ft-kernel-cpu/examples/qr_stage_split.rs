use ft_core::{DType, Device, TensorMeta};
use ft_kernel_cpu::{QrResult, QrStageTimings, qr_contiguous_f64, qr_contiguous_f64_stage_profile};

fn square_matrix(n: usize) -> Vec<f64> {
    let mut a = vec![0.0f64; n * n];
    for i in 0..n {
        for j in 0..n {
            a[i * n + j] = (((i * 53 + j * 31) % 97) as f64 - 48.0) * 0.1;
        }
        a[i * n + i] += n as f64;
    }
    a
}

fn tall_matrix(m: usize, n: usize) -> Vec<f64> {
    let mut a = vec![0.0f64; m * n];
    for i in 0..m {
        for j in 0..n {
            a[i * n + j] = (((i * 53 + j * 31) % 97) as f64 - 48.0) * 0.1;
        }
    }
    for j in 0..n {
        a[j * n + j] += m as f64;
    }
    a
}

fn feed_f64(hash: &mut u64, value: f64) {
    for byte in value.to_bits().to_le_bytes() {
        *hash ^= u64::from(byte);
        *hash = hash.wrapping_mul(0x100000001b3);
    }
}

fn digest(result: &QrResult) -> u64 {
    let mut hash = 0xcbf29ce484222325u64;
    for &value in &result.q {
        feed_f64(&mut hash, value);
    }
    hash ^= 0xff;
    hash = hash.wrapping_mul(0x100000001b3);
    for &value in &result.r {
        feed_f64(&mut hash, value);
    }
    hash
}

fn same_bits(lhs: &[f64], rhs: &[f64]) -> bool {
    lhs.len() == rhs.len()
        && lhs
            .iter()
            .zip(rhs.iter())
            .all(|(&left, &right)| left.to_bits() == right.to_bits())
}

fn same_result(lhs: &QrResult, rhs: &QrResult) -> bool {
    lhs.m == rhs.m && lhs.n == rhs.n && same_bits(&lhs.q, &rhs.q) && same_bits(&lhs.r, &rhs.r)
}

fn median(values: &mut [u128]) -> u128 {
    values.sort_unstable();
    values[values.len() / 2]
}

fn median_timings(samples: &[QrStageTimings]) -> QrStageTimings {
    let mut copy_zeroing = Vec::with_capacity(samples.len());
    let mut panel_and_t = Vec::with_capacity(samples.len());
    let mut trailing_r = Vec::with_capacity(samples.len());
    let mut reverse_q = Vec::with_capacity(samples.len());
    let mut total = Vec::with_capacity(samples.len());
    for sample in samples {
        copy_zeroing.push(sample.copy_zeroing_ns);
        panel_and_t.push(sample.panel_and_t_ns);
        trailing_r.push(sample.trailing_r_ns);
        reverse_q.push(sample.reverse_q_ns);
        total.push(sample.total_ns);
    }
    QrStageTimings {
        copy_zeroing_ns: median(&mut copy_zeroing),
        panel_and_t_ns: median(&mut panel_and_t),
        trailing_r_ns: median(&mut trailing_r),
        reverse_q_ns: median(&mut reverse_q),
        total_ns: median(&mut total),
        ..QrStageTimings::default()
    }
}

fn percent(stage_ns: u128, total_ns: u128) -> f64 {
    if total_ns == 0 {
        0.0
    } else {
        (stage_ns as f64) * 100.0 / (total_ns as f64)
    }
}

fn run_shape(label: &str, m: usize, n: usize, a: &[f64]) {
    let meta = TensorMeta::from_shape(vec![m, n], DType::F64, Device::Cpu);
    let production = qr_contiguous_f64(a, &meta, true).expect("production qr");
    let production_digest = digest(&production);
    let mut samples = Vec::new();
    let mut all_match = true;
    let mut used_blocked = true;
    for _ in 0..7 {
        let profile = qr_contiguous_f64_stage_profile(a, &meta, true).expect("profile qr");
        all_match &= same_result(&production, &profile.result);
        used_blocked &= profile.used_blocked_path;
        samples.push(profile.timings);
    }
    let timings = median_timings(&samples);
    let unaccounted_ns = timings.unaccounted_ns();
    println!(
        concat!(
            "qr_stage_split shape={} rows={} cols={} reduced=true samples=7 ",
            "blocked={} matches_production={} ",
            "digest_fnv64={:016x} ",
            "total_ns={} ",
            "copy_zeroing_ns={} copy_zeroing_pct={:.3} ",
            "panel_and_t_ns={} panel_and_t_pct={:.3} ",
            "trailing_r_ns={} trailing_r_pct={:.3} ",
            "reverse_q_ns={} reverse_q_pct={:.3} ",
            "unaccounted_ns={} unaccounted_pct={:.3}"
        ),
        label,
        m,
        n,
        used_blocked,
        all_match,
        production_digest,
        timings.total_ns,
        timings.copy_zeroing_ns,
        percent(timings.copy_zeroing_ns, timings.total_ns),
        timings.panel_and_t_ns,
        percent(timings.panel_and_t_ns, timings.total_ns),
        timings.trailing_r_ns,
        percent(timings.trailing_r_ns, timings.total_ns),
        timings.reverse_q_ns,
        percent(timings.reverse_q_ns, timings.total_ns),
        unaccounted_ns,
        percent(unaccounted_ns, timings.total_ns)
    );
}

fn main() {
    let square = square_matrix(512);
    run_shape("512x512", 512, 512, &square);

    let tall = tall_matrix(2048, 128);
    run_shape("2048x128", 2048, 128, &tall);
}

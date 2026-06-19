//! Stage breakdown of the blocked QR (frankentorch-ct2yy) at the n where we
//! lose ~13x to torch, so the next lever targets the real hotspot.
//!   rch exec -- cargo run --release -q -p ft-kernel-cpu --example qr_stage_profile_run

use ft_core::{DType, Device, TensorMeta};
use ft_kernel_cpu::qr_contiguous_f64_stage_profile;

fn lcg(n: usize) -> Vec<f64> {
    let mut a = vec![0.0f64; n * n];
    let mut x: u64 = 0x9E3779B97F4A7C15;
    for slot in a.iter_mut() {
        x = x
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        *slot = (x >> 11) as f64 / 9007199254740992.0 * 2.0 - 1.0;
    }
    a
}

fn main() {
    println!("threads={}", rayon::current_num_threads());
    for &n in &[512usize, 1024] {
        let a = lcg(n);
        let m = TensorMeta::from_shape(vec![n, n], DType::F64, Device::Cpu);
        // warm + time a few
        let _ = qr_contiguous_f64_stage_profile(&a, &m, true).unwrap();
        let it = if n <= 512 { 4 } else { 2 };
        let mut acc = (0u128, 0u128, 0u128, 0u128, 0u128);
        for _ in 0..it {
            let p = qr_contiguous_f64_stage_profile(&a, &m, true).unwrap();
            acc.0 += p.timings.total_ns;
            acc.1 += p.timings.panel_and_t_ns;
            acc.2 += p.timings.trailing_r_ns;
            acc.3 += p.timings.reverse_q_ns;
            acc.4 += p.timings.copy_zeroing_ns;
        }
        let d = it as f64 * 1e6;
        println!(
            "n={n:5} total={:.2}ms  panel+T={:.2}ms  trailingR={:.2}ms  reverseQ(buildQ)={:.2}ms  copy/zero={:.2}ms",
            acc.0 as f64 / d,
            acc.1 as f64 / d,
            acc.2 as f64 / d,
            acc.3 as f64 / d,
            acc.4 as f64 / d
        );
    }
}

//! Where does the tred2 reduction's time go — `ggs` matvec or rank-2 update?
//! `frankentorch-wjrqt`.
//!
//! The reduce is 55% of the eigh lane at n=1024 (7155beb8) and the one structural lever left on
//! it is a blocked dsytrd. That lever defers the RANK-2 UPDATE into one rank-2k GEMM per panel
//! and leaves the matvec exactly where it is, so the update's share IS the lever's ceiling.
//! 4b5be636 sized the lever at 1.36x on the lane from an ASSUMED half-and-half split and said so.
//! This measures it.
//!
//! Everything goes to STDERR so a remote runner returns it.

fn fixture(n: usize) -> Vec<f64> {
    // Same generic-spectrum matrix the h2h lane uses under FT_FIXTURE=generic, symmetrised.
    let mut a = vec![0.0f64; n * n];
    for i in 0..n {
        for j in 0..n {
            let h = ((i as i64) * 73 + (j as i64) * 151 + ((i * j) as i64) % 257).rem_euclid(2048);
            a[i * n + j] = h as f64 / 2048.0 - 1.0 + if i == j { 16.0 } else { 0.0 };
        }
    }
    let mut sym = vec![0.0f64; n * n];
    for i in 0..n {
        for j in 0..n {
            sym[i * n + j] = (a[i * n + j] + a[j * n + i]) * 0.5;
        }
    }
    sym
}

fn main() {
    let sizes: Vec<usize> = std::env::var("FT_SIZES")
        .unwrap_or_else(|_| "1024".to_string())
        .split(',')
        .filter_map(|t| t.trim().parse().ok())
        .collect();
    let reps: usize = std::env::var("FT_REPS")
        .ok()
        .and_then(|t| t.trim().parse().ok())
        .unwrap_or(5);
    eprintln!(
        "TRED2_SPLIT rayon_threads={} reps={reps} (first pass discarded, min of the rest)",
        rayon::current_num_threads()
    );
    for &n in &sizes {
        let a = fixture(n);
        // Warm-up pass, discarded: the first pass of any sweep on this fleet runs ~1.23x slower.
        let _ = ft_kernel_cpu::eigh_stage_profile_f64(&a, n);
        let _ = ft_kernel_cpu::tred2_stage_take_ns();
        let mut best = (u128::MAX, 0u128, 0u128, 0u64, 0u64);
        for _ in 0..reps {
            let (r, b, t) = ft_kernel_cpu::eigh_stage_profile_f64(&a, n);
            let (g, u) = ft_kernel_cpu::tred2_stage_take_ns();
            if r < best.0 {
                best = (r, b, t, g, u);
            }
        }
        let (r, b, t, g, u) = best;
        let lane = (r + b + t) as f64;
        let rf = r as f64;
        eprintln!(
            "TRED2_SPLIT n={n:>5} lane={:8.2}ms  reduce={:8.2}ms ({:4.1}% lane)  \
             ggs={:8.2}ms ({:4.1}% reduce)  update={:8.2}ms ({:4.1}% reduce)  \
             other={:8.2}ms ({:4.1}% reduce)  backtransform={:8.2}ms  tql2={:8.2}ms",
            lane / 1e6,
            rf / 1e6,
            100.0 * rf / lane,
            g as f64 / 1e6,
            100.0 * g as f64 / rf,
            u as f64 / 1e6,
            100.0 * u as f64 / rf,
            (rf - g as f64 - u as f64) / 1e6,
            100.0 * (rf - g as f64 - u as f64) / rf,
            b as f64 / 1e6,
            t as f64 / 1e6,
        );
        // The ceiling a blocked dsytrd could reach on the LANE if the update went to zero.
        let upd_share = u as f64 / lane;
        eprintln!(
            "TRED2_SPLIT n={n:>5} ceiling if the rank-2 update became FREE: lane {:.3}x  \
             (update is {:.1}% of the lane)",
            1.0 / (1.0 - upd_share),
            100.0 * upd_share
        );
    }
}

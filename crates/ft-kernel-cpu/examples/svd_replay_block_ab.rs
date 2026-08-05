//! Same-worker A/B for the SVD bidiagonal-QR replay row-block (frankentorch-x53r3),
//! confirming the inferred SVD win. block=1 is the ANCHOR (legacy per-row replay).
//! Times the FULL svd_contiguous_f64 (so the ratio is the end-to-end SVD effect of
//! the replay row-block, diluted by bidiagonalization which is unchanged).
//!   rch exec -- cargo run --release -q -p ft-kernel-cpu --example svd_replay_block_ab
//!
//! EXTENDED for frankentorch-svd-blocked-bidiag-r7jdo. Two gaps in the original
//! sweep mattered once the reduction was blocked and the replay became the
//! largest single phase:
//!   - it stopped at block=16, but the lever is how many times the multi-MB op
//!     stream is re-read from RAM (once per BLOCK), so the interesting range is
//!     wider now that the reduction no longer hides it;
//!   - it measured `full_matrices=false`, which for a square matrix takes the
//!     deferred-left fast path — that never accumulates U, so it exercised only
//!     the V replay and missed the U stream entirely. Both are timed now.

use ft_core::{DType, Device, TensorMeta};
use ft_kernel_cpu::{set_svd_qr_replay_block_override, svd_contiguous_f64};
use std::time::Instant;

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

fn time_svd(a: &[f64], m: &TensorMeta, block: usize, it: usize, full: bool) -> f64 {
    set_svd_qr_replay_block_override(block);
    let _ = svd_contiguous_f64(a, m, full).unwrap(); // warm
    let t = Instant::now();
    for _ in 0..it {
        let _ = svd_contiguous_f64(a, m, full).unwrap();
    }
    t.elapsed().as_secs_f64() * 1e3 / it as f64
}

fn main() {
    const BLOCKS: [usize; 8] = [2, 4, 8, 16, 32, 64, 128, 256];
    println!("threads={}", rayon::current_num_threads());
    for &n in &[512usize, 1024, 2048] {
        let a = lcg(n);
        let m = TensorMeta::from_shape(vec![n, n], DType::F64, Device::Cpu);
        let it = if n <= 512 {
            4
        } else if n <= 1024 {
            2
        } else {
            1
        };
        // `full=false` on a square matrix routes through the deferred-left fast
        // path (V replay only); `full=true` also accumulates U, which is the
        // larger of the two streams.
        for (label, full) in [("reduced", false), ("full   ", true)] {
            let anchor = time_svd(&a, &m, 1, it, full); // per-row
            print!("n={n:5} {label} anchor(block=1)={anchor:9.2}ms");
            let mut best = (0usize, f64::INFINITY);
            for &b in &BLOCKS {
                let t = time_svd(&a, &m, b, it, full);
                if t < best.1 {
                    best = (b, t);
                }
                print!("  b={b}={t:8.2}({:.2}x)", anchor / t);
            }
            println!("  || best b={} at {:.2}ms", best.0, best.1);
        }
    }
    set_svd_qr_replay_block_override(0); // restore the shipped default
}

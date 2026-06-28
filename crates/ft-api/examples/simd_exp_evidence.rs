//! DECISION EVIDENCE (not a shipped path): quantify the SIMD-exp win + parity cost.
//!
//! FT's softmax/log_softmax/cross_entropy/etc. lose to torch because torch
//! vectorizes `exp`; FT uses scalar libm `exp` (bit-exact, parallel) which hits
//! the per-core scalar ceiling. A SIMD `exp` (wide::f32x8) is much faster but does
//! NOT match libm bit-for-bit -> blocked by the "parity absolute" policy for
//! elementwise ops. This microbench measures BOTH the speedup AND the exact parity
//! delta (max abs/rel/ULP vs scalar libm), so the campaign can decide whether to
//! ratify a transcendental-TOLERANCE policy (as it did for eigen/SVD vectors).
use std::time::Instant;
use wide::f32x8;

fn main() {
    let n: usize = std::env::var("N").ok().and_then(|s| s.parse().ok()).unwrap_or(8_000_000);
    // values in a softmax-like range after max-subtraction: [-30, 0]
    let data: Vec<f32> = (0..n)
        .map(|i| {
            let z = (i as u64).wrapping_mul(2862933555777941757).wrapping_add(3037000493);
            -((z >> 40) as f32 / (1u64 << 24) as f32) * 30.0
        })
        .collect();

    let best = |f: &dyn Fn() -> Vec<f32>| {
        let mut b = f64::INFINITY;
        let mut out = Vec::new();
        for _ in 0..7 {
            let t = Instant::now();
            out = f();
            let e = t.elapsed().as_secs_f64() * 1e3;
            if e < b { b = e; }
        }
        (b, out)
    };

    // 1) scalar libm exp (what FT uses today), serial
    let (t_scalar, ref_out) = best(&|| data.iter().map(|&x| x.exp()).collect());

    // 2) SIMD exp (wide::f32x8), serial
    let (t_simd, simd_out) = best(&|| {
        let mut out = vec![0.0f32; n];
        let chunks = n / 8;
        for c in 0..chunks {
            let v = f32x8::from(&data[c * 8..c * 8 + 8]);
            let e = v.exp();
            out[c * 8..c * 8 + 8].copy_from_slice(e.as_array_ref());
        }
        for i in chunks * 8..n {
            out[i] = data[i].exp();
        }
        out
    });

    // 3) SIMD exp + rayon (the actual achievable FT path)
    let (t_simd_par, _) = best(&|| {
        use rayon::prelude::*;
        let mut out = vec![0.0f32; n];
        out.par_chunks_mut(8192).enumerate().for_each(|(ci, dst)| {
            let base = ci * 8192;
            let len = dst.len();
            let full = len / 8;
            for c in 0..full {
                let v = f32x8::from(&data[base + c * 8..base + c * 8 + 8]);
                dst[c * 8..c * 8 + 8].copy_from_slice(v.exp().as_array_ref());
            }
            for i in full * 8..len {
                dst[i] = data[base + i].exp();
            }
        });
        out
    });

    // parity delta vs scalar libm
    let mut max_abs = 0.0f64;
    let mut max_rel = 0.0f64;
    let mut max_ulp = 0u32;
    for (&a, &b) in ref_out.iter().zip(&simd_out) {
        let d = (a as f64 - b as f64).abs();
        if d > max_abs { max_abs = d; }
        let r = d / (a.abs() as f64).max(1e-30);
        if r > max_rel { max_rel = r; }
        let ua = a.to_bits() as i64;
        let ub = b.to_bits() as i64;
        let u = (ua - ub).unsigned_abs() as u32;
        if u > max_ulp { max_ulp = u; }
    }

    println!("SIMD-exp DECISION EVIDENCE, N={n} f32 (single core unless noted), min-of-7");
    println!("  scalar libm exp (FT today): {t_scalar:8.3} ms");
    println!("  SIMD exp (wide f32x8):      {t_simd:8.3} ms  => {:.2}x faster than scalar", t_scalar / t_simd);
    println!("  SIMD exp + rayon:           {t_simd_par:8.3} ms  => {:.2}x faster than scalar", t_scalar / t_simd_par);
    println!("  parity vs libm: max_abs={max_abs:.3e}  max_rel={max_rel:.3e}  max_ulp={max_ulp}");
    println!("  (the per-element exp speedup is the softmax/cross-entropy ceiling; ULP is the tolerance cost)");
}

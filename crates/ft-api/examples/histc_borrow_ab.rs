//! Same-process A/B for the histc/histogram/unique f64-clone-vs-borrow lever.
//! OLD = `tensor_values_lossy_f64` clones the f64 input to an owned Vec, then histc does 3 read-only
//! passes (finite-check, auto-range, bin). NEW = borrow `&[f64]` (no clone), same 3 passes.
//! Measures whether the 128MB clone is a meaningful fraction of histc's serial cost.
//! Run: cargo run --release -p ft-api --example histc_borrow_ab

use std::time::Instant;

// The histc 3-pass over a &[f64] (finite-check + auto-range + bin), mirroring tensor_histc.
fn histc_passes(vals: &[f64], bins: usize) -> Vec<f64> {
    for &v in vals {
        if !v.is_finite() {
            return vec![0.0; bins];
        }
    }
    let mut lo = f64::INFINITY;
    let mut hi = f64::NEG_INFINITY;
    for &v in vals {
        if v < lo {
            lo = v;
        }
        if v > hi {
            hi = v;
        }
    }
    let mut counts = vec![0.0f64; bins];
    let bin_width = (hi - lo) / bins as f64;
    for &v in vals {
        if v >= lo && v <= hi {
            let mut b = ((v - lo) / bin_width) as usize;
            if b >= bins {
                b = bins - 1;
            }
            counts[b] += 1.0;
        }
    }
    counts
}

fn bench<F: Fn() -> usize>(f: F) -> f64 {
    let mut best = f64::INFINITY;
    for _ in 0..9 {
        let t = Instant::now();
        let s = f();
        let el = t.elapsed().as_secs_f64() * 1e3;
        std::hint::black_box(s);
        if el < best {
            best = el;
        }
    }
    best
}

fn main() {
    let bins = 256;
    for &n in &[1usize << 22, 1 << 24, 1 << 26] {
        // 4M, 16M, 64M f64
        let input: Vec<f64> = (0..n).map(|i| (i % 100003) as f64 * 0.01).collect();
        // OLD: clone (to_vec) then passes; NEW: borrow then passes.
        let a = histc_passes(&input.clone(), bins);
        let b = histc_passes(&input, bins);
        let bitmatch = a == b;
        let old_ms = bench(|| {
            let owned = input.clone(); // the tensor_values_lossy_f64 clone
            histc_passes(&owned, bins).len()
        });
        let new_ms = bench(|| histc_passes(&input, bins).len());
        let ratio = old_ms / new_ms;
        println!(
            "  n={:>10} ({:>4}MB f64)  OLD(clone) {:8.3}  NEW(borrow) {:8.3}  = {:.2}x  bitmatch={}",
            n,
            n * 8 / (1 << 20),
            old_ms,
            new_ms,
            ratio,
            bitmatch
        );
    }
}

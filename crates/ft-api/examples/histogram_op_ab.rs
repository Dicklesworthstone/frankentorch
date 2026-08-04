//! Real-op A/B: tensor_histogram on f64 UNWEIGHTED input. OLD = HEAD's generic f64 path (clone via
//! to_vec + SERIAL finite-check/auto-range/bin), replicated inline; NEW = `s.tensor_histogram` (the
//! added F64 fast path: borrow &[f64] + PARALLEL). Same process, min-9.
//! Run: cargo run --release -p ft-api --example histogram_op_ab

use ft_api::FrankenTorchSession;
use ft_core::ExecutionMode;
use std::time::Instant;

// Replica of HEAD's generic f64 histogram input-processing (clone + serial), returns counts.
fn old_histogram(input: &[f64], bins: usize) -> Vec<f64> {
    let vals = input.to_vec(); // tensor_values_lossy_f64 clone
    for &v in &vals {
        if !v.is_finite() {
            return vec![0.0; bins];
        }
    }
    let mut lo = f64::INFINITY;
    let mut hi = f64::NEG_INFINITY;
    for &v in &vals {
        if v < lo {
            lo = v;
        }
        if v > hi {
            hi = v;
        }
    }
    if lo == hi {
        lo -= 0.5;
        hi += 0.5;
    }
    let mut counts = vec![0.0f64; bins];
    let bin_width = (hi - lo) / bins as f64;
    for &v in &vals {
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

fn bench<F: FnMut() -> usize>(mut f: F) -> f64 {
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
    println!(
        "tensor_histogram f64 unweighted (auto-range), min-9:  OLD=clone+serial  NEW=borrow+parallel"
    );
    for &n in &[1usize << 22, 1 << 24, 1 << 26] {
        let input: Vec<f64> = (0..n).map(|i| (i % 100003) as f64 * 0.01).collect();
        let mut s = FrankenTorchSession::new(ExecutionMode::Strict);
        let t = s.tensor_variable(input.clone(), vec![n], false).unwrap();
        let (hist, _edges) = s.tensor_histogram(t, bins, 0.0, 0.0, None, false).unwrap();
        let new_counts = s.tensor_values(hist).unwrap();
        let old_counts = old_histogram(&input, bins);
        let bitmatch = new_counts == old_counts;

        let old_ms = bench(|| old_histogram(&input, bins).len());
        let new_ms = bench(|| {
            let (h, _e) = s.tensor_histogram(t, bins, 0.0, 0.0, None, false).unwrap();
            s.tensor_values(h).unwrap().len()
        });
        let ratio = old_ms / new_ms;
        println!(
            "  n={:>10} ({:>4}MB)  OLD {:9.3}  NEW {:9.3}  = {:.2}x  bitmatch={}",
            n,
            n * 8 / (1 << 20),
            old_ms,
            new_ms,
            ratio,
            bitmatch
        );
    }
}

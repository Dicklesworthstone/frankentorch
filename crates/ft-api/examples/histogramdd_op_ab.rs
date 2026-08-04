//! Real-op A/B: tensor_histogramdd on f64 [N, D]. OLD = HEAD's generic f64 path (clone + SERIAL N-D
//! binning), replicated inline; NEW = `s.tensor_histogramdd` (added F64 fast path: borrow + parallel).
//! Same process, min-9. Run: cargo run --release -p ft-api --example histogramdd_op_ab

use ft_api::FrankenTorchSession;
use ft_core::ExecutionMode;
use std::time::Instant;

// Replica of HEAD's generic f64 histogramdd binning (clone + serial), returns counts.
fn old_histogramdd(input: &[f64], n_samples: usize, n_dims: usize, bins: &[usize]) -> Vec<f64> {
    let vals = input.to_vec(); // tensor_values_lossy_f64 clone
    let dim_ranges: Vec<(f64, f64)> = (0..n_dims)
        .map(|d| {
            let mut lo = f64::INFINITY;
            let mut hi = f64::NEG_INFINITY;
            for s in 0..n_samples {
                let v = vals[s * n_dims + d];
                if v < lo {
                    lo = v;
                }
                if v > hi {
                    hi = v;
                }
            }
            if lo == hi {
                (lo - 0.5, hi + 0.5)
            } else {
                (lo, hi)
            }
        })
        .collect();
    let bin_widths: Vec<f64> = bins
        .iter()
        .zip(&dim_ranges)
        .map(|(&b, &(lo, hi))| (hi - lo) / b as f64)
        .collect();
    let total_bins: usize = bins.iter().product();
    let mut counts = vec![0usize; total_bins];
    for sample in 0..n_samples {
        let mut linear = 0usize;
        let mut ok = true;
        for d in 0..n_dims {
            let v = vals[sample * n_dims + d];
            let (lo, hi) = dim_ranges[d];
            if v < lo || v > hi {
                ok = false;
                break;
            }
            let raw = ((v - lo) / bin_widths[d]) as usize;
            linear = linear * bins[d] + raw.min(bins[d] - 1);
        }
        if ok {
            counts[linear] += 1;
        }
    }
    counts.into_iter().map(|c| c as f64).collect()
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
    println!("tensor_histogramdd f64 (auto-range), min-9:  OLD=clone+serial  NEW=borrow+parallel");
    // (N samples, D dims, bins per dim)
    let cases: Vec<(usize, usize, Vec<usize>)> = vec![
        (1 << 22, 2, vec![64, 64]),
        (1 << 23, 3, vec![32, 32, 32]),
        (1 << 24, 2, vec![128, 128]),
    ];
    for (n_samples, n_dims, bins) in cases {
        let n = n_samples * n_dims;
        let input: Vec<f64> = (0..n).map(|i| (i % 100003) as f64 * 0.01).collect();
        let mut s = FrankenTorchSession::new(ExecutionMode::Strict);
        let t = s
            .tensor_variable(input.clone(), vec![n_samples, n_dims], false)
            .unwrap();
        let (hist, _edges) = s.tensor_histogramdd(t, &bins, None, false).unwrap();
        let new_counts = s.tensor_values(hist).unwrap();
        let old_counts = old_histogramdd(&input, n_samples, n_dims, &bins);
        let bitmatch = new_counts == old_counts;

        let old_ms = bench(|| old_histogramdd(&input, n_samples, n_dims, &bins).len());
        let new_ms = bench(|| {
            let (h, _e) = s.tensor_histogramdd(t, &bins, None, false).unwrap();
            s.tensor_values(h).unwrap().len()
        });
        let ratio = old_ms / new_ms;
        println!(
            "  N={:>9} D={} ({:>4}MB)  OLD {:9.3}  NEW {:9.3}  = {:.2}x  bitmatch={}",
            n_samples,
            n_dims,
            n * 8 / (1 << 20),
            old_ms,
            new_ms,
            ratio,
            bitmatch
        );
    }
}

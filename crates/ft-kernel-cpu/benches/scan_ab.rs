//! Single-process anchored A/B for the cumulative-scan parallelization, to settle
//! whether `cumprod` parallelizes like `cumsum` (cumsum's outer-block Rayon
//! parallelization was accepted as a win in 686ab41b; cumprod was rejected in
//! 66pe as a 1.83x regression — but those were measured across separate processes,
//! so they may be worker-variance/contention confounds).
//!
//! ALL FOUR variants run in ONE bench binary -> ONE rch worker -> one process, so
//! the serial-vs-parallel ratio is genuinely same-worker, and cumsum acts as the
//! KNOWN-GOOD anchor: if cumprod_parallel/cumprod_serial tracks
//! cumsum_parallel/cumsum_serial, cumprod parallelizes identically (the kernels
//! have the same memory pattern: read input, write output, one op/elem).
//!
//!   cargo bench -p ft-kernel-cpu --bench scan_ab

use criterion::{Criterion, black_box, criterion_group, criterion_main};
use rayon::prelude::*;

const N: usize = 8192; // outer lanes (rows)
const D: usize = 1024; // dim=last, inner=1, lane length = D

fn cumsum_serial(input: &[f64], out: &mut [f64]) {
    for outer in 0..N {
        let base = outer * D;
        let mut acc = 0.0;
        for d in 0..D {
            acc += input[base + d];
            out[base + d] = acc;
        }
    }
}

fn cumsum_parallel(input: &[f64], out: &mut [f64]) {
    out.par_chunks_mut(D)
        .enumerate()
        .for_each(|(outer, chunk)| {
            let base = outer * D;
            let mut acc = 0.0;
            for d in 0..D {
                acc += input[base + d];
                chunk[d] = acc;
            }
        });
}

fn cumprod_serial(input: &[f64], out: &mut [f64]) {
    for outer in 0..N {
        let base = outer * D;
        let mut acc = 1.0;
        for d in 0..D {
            acc *= input[base + d];
            out[base + d] = acc;
        }
    }
}

fn cumprod_parallel(input: &[f64], out: &mut [f64]) {
    out.par_chunks_mut(D)
        .enumerate()
        .for_each(|(outer, chunk)| {
            let base = outer * D;
            let mut acc = 1.0;
            for d in 0..D {
                acc *= input[base + d];
                chunk[d] = acc;
            }
        });
}

fn bench(c: &mut Criterion) {
    // Values near 1.0 so the running product stays finite (sum is unaffected).
    let input: Vec<f64> = (0..N * D).map(|i| 0.99 + (i % 11) as f64 * 0.002).collect();
    let mut out = vec![0.0f64; N * D];
    let mut g = c.benchmark_group("scan_ab");
    g.bench_function("cumsum_serial", |b| {
        b.iter(|| {
            cumsum_serial(black_box(&input), &mut out);
            black_box(out[N * D - 1]);
        })
    });
    g.bench_function("cumsum_parallel", |b| {
        b.iter(|| {
            cumsum_parallel(black_box(&input), &mut out);
            black_box(out[N * D - 1]);
        })
    });
    g.bench_function("cumprod_serial", |b| {
        b.iter(|| {
            cumprod_serial(black_box(&input), &mut out);
            black_box(out[N * D - 1]);
        })
    });
    g.bench_function("cumprod_parallel", |b| {
        b.iter(|| {
            cumprod_parallel(black_box(&input), &mut out);
            black_box(out[N * D - 1]);
        })
    });
    g.finish();
}

criterion_group!(benches, bench);
criterion_main!(benches);

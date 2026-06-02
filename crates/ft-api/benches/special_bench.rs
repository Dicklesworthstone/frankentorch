//! Special-function (lgamma / digamma) elementwise throughput. These are
//! compute-bound per element (libm::lgamma; digamma recurrence + series) and are
//! evaluated across the rayon pool for large tensors. Toggle:
//!   baseline:  rch exec -- env RAYON_NUM_THREADS=1 cargo bench -p ft-api --bench special_bench
//!   optimized: rch exec -- cargo bench -p ft-api --bench special_bench

use criterion::{Criterion, black_box, criterion_group, criterion_main};
use ft_api::FrankenTorchSession;
use ft_core::ExecutionMode;

fn bench_special(c: &mut Criterion) {
    let n = 1usize << 20; // 1M elements

    c.bench_function("lgamma_1m", |b| {
        let mut s = FrankenTorchSession::new(ExecutionMode::Strict);
        let x = s.tensor_randn(vec![n], false).unwrap();
        b.iter(|| black_box(s.lgamma_tensor(black_box(x)).unwrap()));
    });

    c.bench_function("digamma_1m", |b| {
        let mut s = FrankenTorchSession::new(ExecutionMode::Strict);
        let x = s.tensor_randn(vec![n], false).unwrap();
        b.iter(|| black_box(s.digamma_tensor(black_box(x)).unwrap()));
    });

    // igamma(a, x): regularized lower incomplete gamma — power series / continued
    // fraction per element (heavy iterative, binary op via par_zip_map_f64). 1M.
    c.bench_function("igamma_1m", |b| {
        let mut s = FrankenTorchSession::new(ExecutionMode::Strict);
        // a > 0, x > 0.
        let a = s.tensor_rand(vec![n], false).unwrap();
        let x = s.tensor_rand(vec![n], false).unwrap();
        b.iter(|| black_box(s.tensor_igamma(black_box(a), black_box(x)).unwrap()));
    });

    // polygamma(2, x): Hurwitz-zeta series per element — the heaviest scalar
    // special function (routed through par_map_f64). 1M elements.
    c.bench_function("polygamma2_1m", |b| {
        let mut s = FrankenTorchSession::new(ExecutionMode::Strict);
        // polygamma is evaluated for x > 0.
        let x = s.tensor_rand(vec![n], false).unwrap();
        b.iter(|| black_box(s.tensor_polygamma(2, black_box(x)).unwrap()));
    });

    // Modified Bessel i0: polynomial + exp per element (autograd-aware forward,
    // routed through par_map_f64). 1M elements.
    c.bench_function("i0_1m", |b| {
        let mut s = FrankenTorchSession::new(ExecutionMode::Strict);
        let x = s.tensor_randn(vec![n], false).unwrap();
        b.iter(|| black_box(s.tensor_i0(black_box(x)).unwrap()));
    });

    // Bessel j0: A&S polynomial / asymptotic approximation per element
    // (no-autograd forward map, routed through par_map_f64). 1M elements.
    c.bench_function("bessel_j0_1m", |b| {
        let mut s = FrankenTorchSession::new(ExecutionMode::Strict);
        let x = s.tensor_randn(vec![n], false).unwrap();
        b.iter(|| black_box(s.tensor_special_bessel_j0(black_box(x)).unwrap()));
    });

    // Bessel k0: exp/log + series per element (heavier than j0). 1M elements.
    c.bench_function("bessel_k0_1m", |b| {
        let mut s = FrankenTorchSession::new(ExecutionMode::Strict);
        // k0 is defined for x > 0.
        let x = s.tensor_rand(vec![n], false).unwrap();
        b.iter(|| black_box(s.tensor_special_modified_bessel_k0(black_box(x)).unwrap()));
    });

    // Orthogonal polynomials: degree-n recurrence per element (the most
    // compute-bound special functions). 256K elements, degree 64.
    let np = 1usize << 18;
    c.bench_function("legendre_p64_256k", |b| {
        let mut s = FrankenTorchSession::new(ExecutionMode::Strict);
        let x = s.tensor_randn(vec![np], false).unwrap();
        b.iter(|| black_box(s.tensor_special_legendre_polynomial_p(black_box(x), 64).unwrap()));
    });
}

criterion_group!(benches, bench_special);
criterion_main!(benches);

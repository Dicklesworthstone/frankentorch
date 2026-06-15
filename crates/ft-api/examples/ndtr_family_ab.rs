//! Inference (no-grad) before/after for the Gaussian-CDF special-function family
//! ndtr / log_ndtr / ndtri. Each OLD path ran a serial `.iter().map(libm::erfc|
//! erfinv ...)` PLUS an unconditional save_for_backward clone (1x numel f64).
//! NEW: production no-grad path — elements mapped over Rayon AND the backward-only
//! clone skipped. Asserts each production output equals its serial reference
//! BIT-FOR-BIT.
//!   cargo run -q --release -p ft-api --example ndtr_family_ab
use ft_api::FrankenTorchSession;
use ft_core::ExecutionMode;
use std::time::Instant;

fn sqrt2_inv() -> f64 {
    1.0 / std::f64::consts::SQRT_2
}

// Serial mirrors of the production forward kernels + the save clone they used to
// run unconditionally.
fn old_ndtr(x: &[f64]) -> Vec<f64> {
    let s = sqrt2_inv();
    let v: Vec<f64> = x.iter().map(|&xi| 0.5 * libm::erfc(-xi * s)).collect();
    let _saved = std::hint::black_box(x.to_vec());
    v
}
fn old_log_ndtr(x: &[f64]) -> Vec<f64> {
    let s = sqrt2_inv();
    let v: Vec<f64> = x.iter().map(|&xi| (0.5 * libm::erfc(-xi * s)).ln()).collect();
    let _saved = std::hint::black_box(x.to_vec());
    v
}

fn bench<F: Fn() -> Vec<f64>>(f: F) -> f64 {
    let mut best = f64::INFINITY;
    for _ in 0..15 {
        let t = Instant::now();
        std::hint::black_box(f());
        best = best.min(t.elapsed().as_secs_f64() * 1e3);
    }
    best
}

fn check(name: &str, got: &[f64], want: &[f64]) {
    assert_eq!(got.len(), want.len());
    for (g, w) in got.iter().zip(want.iter()) {
        assert_eq!(g.to_bits(), w.to_bits(), "{name} parallel != serial");
    }
}

fn main() {
    let nthreads = rayon::current_num_threads();
    let n = 4_000_000usize;
    // ndtr/log_ndtr take any real x; ndtri takes p in (0,1).
    let x: Vec<f64> = (0..n).map(|i| ((i % 401) as f64) * 0.01 - 2.0).collect();
    let p: Vec<f64> = (0..n).map(|i| 0.001 + ((i % 997) as f64) * 0.001).collect();

    let want_ndtr = old_ndtr(&x);
    let want_log = old_log_ndtr(&x);

    let pn = rayon::ThreadPoolBuilder::new().build().unwrap();
    pn.install(|| {
        let mut s = FrankenTorchSession::new(ExecutionMode::Strict);
        // Correctness: production no-grad output must be bit-for-bit == serial ref.
        let v = s.tensor_variable(x.clone(), vec![n], false).unwrap();
        let r = s.tensor_ndtr(v).unwrap();
        check("ndtr", &s.tensor_values(r).unwrap(), &want_ndtr);
        let v = s.tensor_variable(x.clone(), vec![n], false).unwrap();
        let r = s.tensor_log_ndtr(v).unwrap();
        check("log_ndtr", &s.tensor_values(r).unwrap(), &want_log);
        // ndtri serial reference: erfinv_approx is private, so build the serial
        // baseline by running p in chunks BELOW the parallel threshold (4096 <
        // PARALLEL_ELEMENTWISE_MIN=8192), which forces par_map_f64's serial path.
        // Each element maps independently, so concatenated serial chunks are the
        // exact serial reference the full (parallel) call must reproduce.
        let v = s.tensor_variable(p.clone(), vec![n], false).unwrap();
        let r = s.tensor_ndtri(v).unwrap();
        let want_ndtri_serial = {
            let mut s2 = FrankenTorchSession::new(ExecutionMode::Strict);
            let mut out = Vec::with_capacity(n);
            for chunk in p.chunks(4096) {
                let vc = s2
                    .tensor_variable(chunk.to_vec(), vec![chunk.len()], false)
                    .unwrap();
                let rc = s2.tensor_ndtri(vc).unwrap();
                out.extend(s2.tensor_values(rc).unwrap());
            }
            out
        };
        check("ndtri", &s.tensor_values(r).unwrap(), &want_ndtri_serial);

        let new_ndtr = {
            let mut best = f64::INFINITY;
            for _ in 0..15 {
                let v = s.tensor_variable(x.clone(), vec![n], false).unwrap();
                let t = Instant::now();
                std::hint::black_box(s.tensor_ndtr(v).unwrap());
                best = best.min(t.elapsed().as_secs_f64() * 1e3);
            }
            best
        };
        let new_log = {
            let mut best = f64::INFINITY;
            for _ in 0..15 {
                let v = s.tensor_variable(x.clone(), vec![n], false).unwrap();
                let t = Instant::now();
                std::hint::black_box(s.tensor_log_ndtr(v).unwrap());
                best = best.min(t.elapsed().as_secs_f64() * 1e3);
            }
            best
        };
        let old_ndtr_ms = bench(|| old_ndtr(&x));
        let old_log_ms = bench(|| old_log_ndtr(&x));
        println!(
            "ndtr     no-grad n={n} (bit-exact OK): OLD {old_ndtr_ms:.2}ms  NEW({nthreads}t) {new_ndtr:.2}ms  =>  {:.2}x",
            old_ndtr_ms / new_ndtr
        );
        println!(
            "log_ndtr no-grad n={n} (bit-exact OK): OLD {old_log_ms:.2}ms  NEW({nthreads}t) {new_log:.2}ms  =>  {:.2}x",
            old_log_ms / new_log
        );
    });
}

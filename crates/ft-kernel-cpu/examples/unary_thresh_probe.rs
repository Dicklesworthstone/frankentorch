//! Same-size serial-vs-parallel A/B for the scalar-unary map path
//! (`unary_f64`, used by exp/ln/gelu/erf/sigmoid/tanh/...). Settles whether
//! SCALAR_UNARY_PARALLEL_THRESHOLD (524288) is too high.
//!
//!   rch exec -- cargo run --release -q -p ft-kernel-cpu --example unary_thresh_probe
//!
//! FINDING (64-core worker, 3 passes, sigmoid closure ≈ 1 libm exp/elem):
//!   131072  speedup ~1.0-1.3x  (marginal; one pass regressed to 0.95x)
//!   262144  speedup ~1.7-1.96x (consistently UNDER 2.0)
//!   524288  speedup ~2.2-2.87x (consistently >=2.0)
//! => 524288 sits right at the >=2.0x crossover and is CORRECTLY tuned for the
//! Score>=2.0 bar; lowering it only buys sub-2.0 wins. (par_iter().map().collect()
//! is also near-optimal here: par_iter_mut().zip() measured ~3x WORSE, and
//! par_chunks_mut with a fixed chunk regresses at large N.) WARNING: comparing
//! ns/el at 262144-serial vs 524288-parallel is a SIZE CONFOUND, not an A/B.
use rayon::prelude::*;
use std::time::Instant;
#[inline]
fn costly(x: f64) -> f64 {
    1.0 / (1.0 + (-x).exp())
}
fn t<F: FnMut()>(mut f: F, it: usize) -> f64 {
    f();
    let s = Instant::now();
    for _ in 0..it {
        f();
    }
    s.elapsed().as_secs_f64() * 1e6 / it as f64
}
fn main() {
    println!("threads={}", rayon::current_num_threads());
    for &n in &[131072usize, 262144, 524288] {
        let d: Vec<f64> = (0..n).map(|i| ((i * 7 % 1000) as f64) * 0.01 - 5.0).collect();
        let ser = t(|| { let r: Vec<f64> = d.iter().map(|v| costly(*v)).collect(); std::hint::black_box(r); }, 80);
        let par = t(|| { let r: Vec<f64> = d.par_iter().map(|v| costly(*v)).collect(); std::hint::black_box(r); }, 80);
        println!("n={n:<7} serial={ser:7.1}us par={par:7.1}us speedup={:.2}x", ser / par);
    }
}

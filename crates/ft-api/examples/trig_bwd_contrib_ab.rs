//! Isolated lever A/B for the trig/hyperbolic activation BACKWARD contrib maps in
//! ft-autograd (sin/cos/sinh/cosh/asin/acos). Times JUST the contrib map serial
//! vs the order-preserving Rayon map (mirror of tensor_backward_zip_map), ONE
//! process ONE worker, anchored. The grad uses `cos` (sin's derivative) — the
//! LIGHTEST single transcendental of the six, so it lower-bounds the family win.
//! NEW asserted BIT-FOR-BIT == serial.
//!   cargo run -q --release -p ft-api --example trig_bwd_contrib_ab
use rayon::prelude::*;
use std::time::Instant;

fn serial(incoming: &[f64], x: &[f64]) -> Vec<f64> {
    incoming.iter().zip(x.iter()).map(|(&g, &xi)| g * xi.cos()).collect()
}
fn par(incoming: &[f64], x: &[f64]) -> Vec<f64> {
    incoming.par_iter().zip(x.par_iter()).map(|(&g, &xi)| g * xi.cos()).collect()
}

fn bench(label: &str, n: usize) {
    let nthreads = rayon::current_num_threads();
    let x: Vec<f64> = (0..n).map(|i| ((i % 6283) as f64) * 0.001 - 3.14).collect();
    let incoming: Vec<f64> = (0..n).map(|i| 1.0 + ((i % 5) as f64) * 0.1).collect();
    let want = serial(&incoming, &x);
    let got = par(&incoming, &x);
    for (a, b) in want.iter().zip(got.iter()) {
        assert_eq!(a.to_bits(), b.to_bits(), "parallel != serial");
    }
    let mut old = f64::INFINITY;
    for _ in 0..20 {
        let t = Instant::now();
        std::hint::black_box(serial(&incoming, &x));
        old = old.min(t.elapsed().as_secs_f64() * 1e3);
    }
    let mut new = f64::INFINITY;
    for _ in 0..20 {
        let t = Instant::now();
        std::hint::black_box(par(&incoming, &x));
        new = new.min(t.elapsed().as_secs_f64() * 1e3);
    }
    println!("{label} n={n} ({nthreads}t, bit-exact OK): serial {old:.3}ms  par {new:.3}ms  =>  {:.2}x", old / new);
}

fn main() {
    bench("ANCHOR-small", 4096);
    bench("sin-bwd(cos)", 1_000_000);
    bench("sin-bwd(cos)", 4_000_000);
}

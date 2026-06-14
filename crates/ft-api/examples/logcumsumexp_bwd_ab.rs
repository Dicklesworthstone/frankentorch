//! Anchored A/B for the logcumsumexp BACKWARD lane parallelization (grad/training path).
//! The backward is a per-(outer,inner)-lane O(D^2) sum with an exp per pair = compute-bound;
//! lanes are independent. Replicates the production backward math serial vs parallel,
//! times 1t-vs-Nt in ONE process, asserts bit-for-bit equality.
//!   cargo run -q --release -p ft-api --example logcumsumexp_bwd_ab
use rayon::prelude::*;
use std::time::Instant;

fn fwd(x: &[f64], outer: usize, dim: usize, inner: usize) -> Vec<f64> {
    let mut y = vec![0.0; x.len()];
    for o in 0..outer {
        for i in 0..inner {
            let mut rmax = f64::NEG_INFINITY;
            let mut rsum = 0.0;
            for d in 0..dim {
                let idx = o * dim * inner + d * inner + i;
                let v = x[idx];
                if v > rmax {
                    rsum = rsum * (rmax - v).exp() + 1.0;
                    rmax = v;
                } else {
                    rsum += (v - rmax).exp();
                }
                y[idx] = rmax + rsum.ln();
            }
        }
    }
    y
}

fn bwd_lane(base: usize, out: &mut [f64], x: &[f64], y: &[f64], gy: &[f64], dim: usize, inner: usize) {
    for i in 0..inner {
        for k in 0..dim {
            let kl = k * inner + i;
            let xk = x[base + kl];
            let mut acc = 0.0;
            for ii in k..dim {
                let idx = base + ii * inner + i;
                acc += gy[idx] * (xk - y[idx]).exp();
            }
            out[kl] = acc;
        }
    }
}

fn run_best(threads: usize, x: &[f64], y: &[f64], gy: &[f64], outer: usize, dim: usize, inner: usize) -> (f64, Vec<f64>) {
    let lane = dim * inner;
    let pool = rayon::ThreadPoolBuilder::new().num_threads(threads).build().unwrap();
    pool.install(|| {
        let run = || {
            let mut g = vec![0.0; x.len()];
            if threads > 1 {
                g.par_chunks_mut(lane).enumerate().for_each(|(o, c)| bwd_lane(o * lane, c, x, y, gy, dim, inner));
            } else {
                for o in 0..outer { let b = o * lane; bwd_lane(b, &mut g[b..b + lane], x, y, gy, dim, inner); }
            }
            g
        };
        let out = run();
        let mut best = f64::INFINITY;
        for _ in 0..15 { let t = Instant::now(); std::hint::black_box(run()); best = best.min(t.elapsed().as_secs_f64() * 1e3); }
        (best, out)
    })
}

fn main() {
    let (rows, cols) = (16384usize, 256);
    let n = rows * cols;
    let x: Vec<f64> = (0..n).map(|i| ((i % 211) as f64) * 0.01 - 1.0).collect();
    let y = fwd(&x, rows, cols, 1);
    let gy: Vec<f64> = (0..n).map(|i| 1.0 + ((i % 13) as f64) * 1e-3).collect();
    let nthreads = rayon::current_num_threads();

    let (ser, gs) = run_best(1, &x, &y, &gy, rows, cols, 1);
    let (par, gp) = run_best(nthreads.max(2), &x, &y, &gy, rows, cols, 1);
    for (a, b) in gs.iter().zip(gp.iter()) {
        assert_eq!(a.to_bits(), b.to_bits(), "parallel != serial bit-exact");
    }
    println!("logcumsumexp_backward [{rows},{cols}] dim=1 (bit-exact OK): serial(1t) {ser:.2}ms  parallel({nthreads}t) {par:.2}ms  =>  {:.2}x", ser / par);
}

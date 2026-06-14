//! Anchored A/B for the lp_pool_grad_output FORWARD closure (grad/training path).
//! out[k] = (sum_{i in window_k} |x[i]|^p)^(1/p) — per-window independent, powf-heavy.
//! Builds windows for a typical lp_pool2d config, times serial vs parallel over windows
//! (1t-vs-Nt, one process), asserts bit-for-bit equality.
//!   cargo run -q --release -p ft-api --example lp_pool_grad_fwd_ab
use rayon::prelude::*;
use std::time::Instant;

fn run_best(threads: usize, xv: &[f64], windows: &[Vec<usize>], p: f64) -> (f64, Vec<f64>) {
    let pool = rayon::ThreadPoolBuilder::new().num_threads(threads).build().unwrap();
    let lp = |win: &[usize]| -> f64 {
        let mut s = 0.0;
        for &i in win {
            s += xv[i].abs().powf(p);
        }
        s.powf(1.0 / p)
    };
    pool.install(|| {
        let run = || {
            let mut out = vec![0.0; windows.len()];
            if threads > 1 {
                out.par_iter_mut().zip(windows.par_iter()).for_each(|(o, w)| *o = lp(w));
            } else {
                for (o, w) in out.iter_mut().zip(windows.iter()) { *o = lp(w); }
            }
            out
        };
        let res = run();
        let mut best = f64::INFINITY;
        for _ in 0..15 { let t = Instant::now(); std::hint::black_box(run()); best = best.min(t.elapsed().as_secs_f64() * 1e3); }
        (best, res)
    })
}

fn main() {
    let (n, c, h, w) = (16usize, 64, 64, 64);
    let (kh, kw, sh, sw, p) = (2usize, 2, 2, 2, 3.0);
    let xv: Vec<f64> = (0..n * c * h * w).map(|i| ((i % 197) as f64) * 0.01 - 1.0).collect();
    let (h_out, w_out) = ((h - kh) / sh + 1, (w - kw) / sw + 1);
    let mut windows: Vec<Vec<usize>> = Vec::with_capacity(n * c * h_out * w_out);
    for b in 0..n {
        for ch in 0..c {
            let base = b * c * h * w + ch * h * w;
            for oh in 0..h_out {
                for ow in 0..w_out {
                    let mut win = Vec::new();
                    for ih in oh * sh..(oh * sh + kh).min(h) {
                        for iw in ow * sw..(ow * sw + kw).min(w) {
                            win.push(base + ih * w + iw);
                        }
                    }
                    windows.push(win);
                }
            }
        }
    }
    let nthreads = rayon::current_num_threads();
    let (ser, a) = run_best(1, &xv, &windows, p);
    let (par, b) = run_best(nthreads.max(2), &xv, &windows, p);
    for (x, y) in a.iter().zip(b.iter()) {
        assert_eq!(x.to_bits(), y.to_bits(), "parallel != serial bit-exact");
    }
    println!("lp_pool_grad_fwd {} windows p{p} (bit-exact OK): serial(1t) {ser:.2}ms  parallel({nthreads}t) {par:.2}ms  =>  {:.2}x", windows.len(), ser / par);
}

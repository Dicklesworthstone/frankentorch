//! A/B for tensor_multi_margin_loss F64 (reduction="none"). OLD = apply_function-path replica
//! (CLONE input via to_vec as the orig materializes `inputs`, then SERIAL per-sample hinge); NEW =
//! sess.tensor_multi_margin_loss (F64 fast path: borrow + parallel over N). bitmatch verifies the
//! parallel per-sample rows match the serial ones. Run: cargo run --release -p ft-api --example multi_margin_ab

use ft_api::FrankenTorchSession;
use ft_core::ExecutionMode;
use std::time::Instant;

fn old_multi_margin(input: &[f64], target: &[f64], n: usize, c: usize, p: usize, margin: f64) -> Vec<f64> {
    let cloned = input.to_vec(); // orig materializes input via tensor_values
    let mut rows = Vec::with_capacity(n);
    for i in 0..n {
        let y = target[i] as usize;
        let x_y = cloned[i * c + y];
        let mut sum = 0.0_f64;
        for j in 0..c {
            if j != y {
                let mt = margin - x_y + cloned[i * c + j];
                if mt > 0.0 {
                    let lj = if p == 1 { mt } else { mt * mt };
                    sum += lj;
                }
            }
        }
        rows.push(sum / c as f64);
    }
    rows
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
    println!("tensor_multi_margin_loss f64 none, min-9:  OLD=apply_fn replica (clone+serial)  NEW=borrow+parallel");
    let cases: [(&str, usize, usize, usize); 3] =
        [("200k x 128 p1", 200_000, 128, 1), ("200k x 128 p2", 200_000, 128, 2), ("100k x 256 p1", 100_000, 256, 1)];
    for (label, n, c, p) in cases {
        let margin = 1.0_f64;
        let input: Vec<f64> = (0..n * c).map(|i| ((i % 211) as f64 - 100.0) * 0.01).collect();
        let target: Vec<f64> = (0..n).map(|i| (i % c) as f64).collect();

        let mut sess = FrankenTorchSession::new(ExecutionMode::Strict);
        let it = sess.tensor_variable(input.clone(), vec![n, c], false).unwrap();
        let tt = sess.tensor_variable(target.clone(), vec![n], false).unwrap();
        let out = sess.tensor_multi_margin_loss(it, tt, p, margin, None, "none").unwrap();
        let new_out = sess.tensor_values(out).unwrap();
        let old_out = old_multi_margin(&input, &target, n, c, p, margin);
        let bitmatch = new_out == old_out;

        let old_ms = bench(|| old_multi_margin(&input, &target, n, c, p, margin).len());
        let new_ms = bench(|| sess.tensor_multi_margin_loss(it, tt, p, margin, None, "none").unwrap().0);
        println!(
            "  {label:<16} ({:>3}MB in)  OLD {:8.3}  NEW {:8.3}  = {:.2}x  bitmatch={}",
            n * c * 8 / (1 << 20),
            old_ms,
            new_ms,
            old_ms / new_ms,
            bitmatch
        );
    }
}

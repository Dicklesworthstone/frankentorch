//! A/B for tensor_multilabel_margin_loss F64 (reduction="none"). OLD = clone input (models the
//! apply_function save_for_backward to_vec) + SERIAL per-sample hinge; NEW = real op (borrow + parallel
//! over N). Run: cargo run --release -p ft-api --example multilabel_margin_ab

use ft_api::FrankenTorchSession;
use ft_core::ExecutionMode;
use std::time::Instant;

fn old_multilabel(input: &[f64], target: &[f64], n: usize, c: usize) -> Vec<f64> {
    let cloned = input.to_vec(); // models the apply_function save_for_backward to_vec
    let mut losses = Vec::with_capacity(n);
    for i in 0..n {
        let row = i * c;
        let pos: Vec<usize> = (0..c)
            .filter(|&j| target[row + j] >= 0.0)
            .map(|j| target[row + j] as usize)
            .filter(|&idx| idx < c)
            .collect();
        let mut sum = 0.0;
        for &y in &pos {
            let x_y = cloned[row + y];
            for k in 0..c {
                if !pos.contains(&k) {
                    let mt = 1.0 - x_y + cloned[row + k];
                    if mt > 0.0 {
                        sum += mt;
                    }
                }
            }
        }
        losses.push(sum / c as f64);
    }
    losses
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
    println!(
        "tensor_multilabel_margin_loss f64 none, min-9:  OLD=clone+serial  NEW=borrow+parallel"
    );
    let cases: [(&str, usize, usize, usize); 3] = [
        ("50k x 200 pos5", 50_000, 200, 5),
        ("100k x 100 pos4", 100_000, 100, 4),
        ("30k x 400 pos8", 30_000, 400, 8),
    ];
    for (label, n, c, num_pos) in cases {
        let input: Vec<f64> = (0..n * c)
            .map(|i| ((i % 211) as f64 - 100.0) * 0.01)
            .collect();
        // target[i][0..num_pos] = distinct positive class indices; rest = -1.
        let target: Vec<f64> = (0..n * c)
            .map(|idx| {
                let j = idx % c;
                if j < num_pos { j as f64 } else { -1.0 }
            })
            .collect();

        let mut sess = FrankenTorchSession::new(ExecutionMode::Strict);
        let it = sess
            .tensor_variable(input.clone(), vec![n, c], false)
            .unwrap();
        let tt = sess
            .tensor_variable(target.clone(), vec![n, c], false)
            .unwrap();
        let out = sess.tensor_multilabel_margin_loss(it, tt, "none").unwrap();
        let new_out = sess.tensor_values(out).unwrap();
        let old_out = old_multilabel(&input, &target, n, c);
        let bitmatch = new_out == old_out;

        let old_ms = bench(|| old_multilabel(&input, &target, n, c).len());
        let new_ms = bench(|| {
            sess.tensor_multilabel_margin_loss(it, tt, "none")
                .unwrap()
                .0
        });
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

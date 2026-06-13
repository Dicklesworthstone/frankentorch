use ft_api::FrankenTorchSession;
use ft_core::ExecutionMode;
use std::hint::black_box;
use std::time::Instant;
fn t<F: FnMut()>(mut f: F) -> f64 {
    for _ in 0..2 {
        f();
    }
    let mut b = f64::MAX;
    for _ in 0..6 {
        let s = Instant::now();
        for _ in 0..3 {
            f();
        }
        b = b.min(s.elapsed().as_secs_f64() / 3.0 * 1000.0);
    }
    b
}
fn main() {
    let (rows, cols) = (256usize, 4096usize);
    let n = rows * cols;
    let data: Vec<f64> = (0..n)
        .map(|i| ((i * 2654435761usize) % 1_000_003) as f64)
        .collect();
    let lo = 2047usize;
    let hi = 2048usize;
    let frac = 0.5;
    // OLD op logic via public API: parallel kernel sort along dim, narrow, lerp.
    let old = t(|| {
        let mut s = FrankenTorchSession::new(ExecutionMode::Strict);
        let tt = s
            .tensor_variable(data.clone(), vec![rows, cols], false)
            .unwrap();
        let (sorted, _) = s.tensor_sort(tt, 1, false).unwrap();
        let lo_v = s.tensor_narrow(sorted, 1, lo, 1).unwrap();
        let hi_v = s.tensor_narrow(sorted, 1, hi, 1).unwrap();
        black_box(s.tensor_lerp(lo_v, hi_v, frac).unwrap());
    });
    // NEW op.
    let new = t(|| {
        let mut s = FrankenTorchSession::new(ExecutionMode::Strict);
        let tt = s
            .tensor_variable(data.clone(), vec![rows, cols], false)
            .unwrap();
        black_box(s.tensor_quantile_dim(tt, 0.5, 1, false, "linear").unwrap());
    });
    println!(
        "quantile_dim [{rows},{cols}]: old(parallel-sort+narrow) {old:.3}ms | new(quickselect) {new:.3}ms | {:.2}x",
        old / new
    );
}

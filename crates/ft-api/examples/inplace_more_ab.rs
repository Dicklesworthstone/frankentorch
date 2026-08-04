//! Real-op A/B for copysign_/remainder_/xlogy_ in-place F64 fast path.
//! Run: cargo run --release -p ft-api --example inplace_more_ab

use ft_api::FrankenTorchSession;
use ft_core::ExecutionMode;
use std::time::Instant;

fn old_copysign(t: &[f64], o: &[f64]) -> Vec<f64> {
    let tv = t.to_vec();
    let ov = o.to_vec();
    tv.iter()
        .zip(ov.iter())
        .map(|(&m, &s)| m.copysign(s))
        .collect()
}
fn old_remainder(t: &[f64], o: &[f64]) -> Vec<f64> {
    let tv = t.to_vec();
    let ov = o.to_vec();
    tv.iter()
        .zip(ov.iter())
        .map(|(&a, &b)| a - (a / b).floor() * b)
        .collect()
}
fn old_xlogy(t: &[f64], o: &[f64]) -> Vec<f64> {
    let tv = t.to_vec();
    let ov = o.to_vec();
    tv.iter()
        .zip(ov.iter())
        .map(|(&x, &y)| {
            if x == 0.0 && !y.is_nan() {
                0.0
            } else {
                x * y.ln()
            }
        })
        .collect()
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
        "in-place copysign_/remainder_/xlogy_ f64, min-9:  OLD=clone-both+serial  NEW=borrow-both+parallel"
    );
    let n = 1usize << 26; // 512MB
    let a: Vec<f64> = (0..n).map(|i| (i % 211) as f64 - 100.0).collect();
    let b: Vec<f64> = (0..n).map(|i| (i % 173) as f64 + 1.0).collect();

    let mut s = FrankenTorchSession::new(ExecutionMode::Strict);
    let at = s.tensor_variable(a.clone(), vec![n], false).unwrap();
    let bt = s.tensor_variable(b.clone(), vec![n], false).unwrap();
    s.tensor_copysign_(at, bt).unwrap();
    let bm_c = s.tensor_values(at).unwrap() == old_copysign(&a, &b);
    let co = bench(|| old_copysign(&a, &b).len());
    let cn = bench(|| {
        s.tensor_copysign_(at, bt).unwrap();
        s.tensor_values(at).unwrap().len()
    });

    let mut s2 = FrankenTorchSession::new(ExecutionMode::Strict);
    let at2 = s2.tensor_variable(a.clone(), vec![n], false).unwrap();
    let bt2 = s2.tensor_variable(b.clone(), vec![n], false).unwrap();
    s2.tensor_remainder_(at2, bt2).unwrap();
    let bm_r = s2.tensor_values(at2).unwrap() == old_remainder(&a, &b);
    let ro = bench(|| old_remainder(&a, &b).len());
    let rn = bench(|| {
        s2.tensor_remainder_(at2, bt2).unwrap();
        s2.tensor_values(at2).unwrap().len()
    });

    let mut s3 = FrankenTorchSession::new(ExecutionMode::Strict);
    let a3: Vec<f64> = (0..n).map(|i| (i % 211) as f64).collect();
    let at3 = s3.tensor_variable(a3.clone(), vec![n], false).unwrap();
    let bt3 = s3.tensor_variable(b.clone(), vec![n], false).unwrap();
    s3.tensor_xlogy_(at3, bt3).unwrap();
    let bm_x = s3.tensor_values(at3).unwrap() == old_xlogy(&a3, &b);
    let xo = bench(|| old_xlogy(&a3, &b).len());
    let xn = bench(|| {
        s3.tensor_xlogy_(at3, bt3).unwrap();
        s3.tensor_values(at3).unwrap().len()
    });

    println!(
        "  n={} (512MB)  copysign_ {:.2}x (bm={})  remainder_ {:.2}x (bm={})  xlogy_ {:.2}x (bm={})",
        n,
        co / cn,
        bm_c,
        ro / rn,
        bm_r,
        xo / xn,
        bm_x
    );
}

//! Real-op A/B for the in-place binary family F64 fast path (hypot_ transcendental + fmod_ cheap).
//! OLD = clone-both + serial map (HEAD generic); NEW = the borrow-both+parallel fast path.
//! Run: cargo run --release -p ft-api --example inplace_family_ab

use ft_api::FrankenTorchSession;
use ft_core::ExecutionMode;
use std::time::Instant;

fn old_hypot(t: &[f64], o: &[f64]) -> Vec<f64> {
    let tv = t.to_vec();
    let ov = o.to_vec();
    tv.iter().zip(ov.iter()).map(|(&x, &y)| x.hypot(y)).collect()
}
fn old_fmod(t: &[f64], o: &[f64]) -> Vec<f64> {
    let tv = t.to_vec();
    let ov = o.to_vec();
    tv.iter()
        .zip(ov.iter())
        .map(|(&a, &b)| a - (a / b).trunc() * b)
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
    println!("in-place family f64, min-9:  OLD=clone-both+serial  NEW=borrow-both+parallel");
    for &n in &[1usize << 24, 1 << 26] {
        let a: Vec<f64> = (0..n).map(|i| (i % 211) as f64 + 1.0).collect();
        let b: Vec<f64> = (0..n).map(|i| (i % 173) as f64 + 1.0).collect();

        // hypot_
        let mut s = FrankenTorchSession::new(ExecutionMode::Strict);
        let at = s.tensor_variable(a.clone(), vec![n], false).unwrap();
        let bt = s.tensor_variable(b.clone(), vec![n], false).unwrap();
        s.tensor_hypot_(at, bt).unwrap();
        let bm_h = s.tensor_values(at).unwrap() == old_hypot(&a, &b);
        let ho = bench(|| old_hypot(&a, &b).len());
        let hn = bench(|| {
            s.tensor_hypot_(at, bt).unwrap();
            s.tensor_values(at).unwrap().len()
        });

        // fmod_
        let mut s2 = FrankenTorchSession::new(ExecutionMode::Strict);
        let at2 = s2.tensor_variable(a.clone(), vec![n], false).unwrap();
        let bt2 = s2.tensor_variable(b.clone(), vec![n], false).unwrap();
        s2.tensor_fmod_(at2, bt2).unwrap();
        let bm_f = s2.tensor_values(at2).unwrap() == old_fmod(&a, &b);
        let fo = bench(|| old_fmod(&a, &b).len());
        let fn_ = bench(|| {
            s2.tensor_fmod_(at2, bt2).unwrap();
            s2.tensor_values(at2).unwrap().len()
        });

        println!(
            "  n={:>10} ({:>4}MB)  hypot_ {:.2}x (bm={})  fmod_ {:.2}x (bm={})",
            n,
            n * 8 / (1 << 20),
            ho / hn,
            bm_h,
            fo / fn_,
            bm_f
        );
    }
}

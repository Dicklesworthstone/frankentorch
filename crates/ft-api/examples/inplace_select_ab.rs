//! Real-op A/B for in-place masked_fill_ (binary) + where_ (ternary) F64 select fast paths.
//! Run: cargo run --release -p ft-api --example inplace_select_ab

use ft_api::FrankenTorchSession;
use ft_core::ExecutionMode;
use std::time::Instant;

fn old_masked_fill(t: &[f64], m: &[f64], value: f64) -> Vec<f64> {
    let tv = t.to_vec();
    let mv = m.to_vec();
    tv.iter()
        .zip(mv.iter())
        .map(|(&t, &m)| if m != 0.0 { value } else { t })
        .collect()
}
fn old_where(t: &[f64], c: &[f64], o: &[f64]) -> Vec<f64> {
    let tv = t.to_vec();
    let cv = c.to_vec();
    let ov = o.to_vec();
    tv.iter()
        .zip(cv.iter())
        .zip(ov.iter())
        .map(|((&t, &c), &o)| if c != 0.0 { o } else { t })
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
        "in-place masked_fill_/where_ f64, min-9:  OLD=clone-all+serial  NEW=borrow-all+parallel"
    );
    let n = 1usize << 26; // 512MB
    let t: Vec<f64> = (0..n).map(|i| (i % 211) as f64).collect();
    let m: Vec<f64> = (0..n).map(|i| (i % 2) as f64).collect();
    let o: Vec<f64> = (0..n).map(|i| (i % 173) as f64 + 0.5).collect();

    let mut s = FrankenTorchSession::new(ExecutionMode::Strict);
    let tt = s.tensor_variable(t.clone(), vec![n], false).unwrap();
    let mt = s.tensor_variable(m.clone(), vec![n], false).unwrap();
    s.tensor_masked_fill_(tt, mt, -1.0).unwrap();
    let bm_m = s.tensor_values(tt).unwrap() == old_masked_fill(&t, &m, -1.0);
    let mo = bench(|| old_masked_fill(&t, &m, -1.0).len());
    let mn = bench(|| {
        s.tensor_masked_fill_(tt, mt, -1.0).unwrap();
        s.tensor_values(tt).unwrap().len()
    });

    let mut s2 = FrankenTorchSession::new(ExecutionMode::Strict);
    let tt2 = s2.tensor_variable(t.clone(), vec![n], false).unwrap();
    let ct2 = s2.tensor_variable(m.clone(), vec![n], false).unwrap();
    let ot2 = s2.tensor_variable(o.clone(), vec![n], false).unwrap();
    s2.tensor_where_(tt2, ct2, ot2).unwrap();
    let bm_w = s2.tensor_values(tt2).unwrap() == old_where(&t, &m, &o);
    let wo = bench(|| old_where(&t, &m, &o).len());
    let wn = bench(|| {
        s2.tensor_where_(tt2, ct2, ot2).unwrap();
        s2.tensor_values(tt2).unwrap().len()
    });

    println!(
        "  n={} (512MB)  masked_fill_ {:.2}x (bm={})  where_ {:.2}x (bm={})",
        n,
        mo / mn,
        bm_m,
        wo / wn,
        bm_w
    );
}

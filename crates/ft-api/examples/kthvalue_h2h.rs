use ft_api::FrankenTorchSession;
use ft_core::ExecutionMode;
use std::time::Instant;

fn main() {
    let tag = std::env::var("FT_TAG").unwrap_or_else(|_| "FT".into());
    let n = 16_000_000usize;
    // pseudo-random f64 in [0,1), materialized OUTSIDE the timer
    let data: Vec<f64> = (0..n)
        .map(|i| {
            let x = (i as u64)
                .wrapping_mul(2862933555777941757)
                .wrapping_add(3037000493);
            (x >> 11) as f64 / ((1u64 << 53) as f64)
        })
        .collect();
    for k in [n / 2, 1, n] {
        let mut best = f64::INFINITY;
        for _ in 0..5 {
            let mut s = FrankenTorchSession::new(ExecutionMode::Strict);
            let t = s.tensor_variable(data.clone(), vec![n], false).unwrap();
            let t0 = Instant::now();
            let (v, _idx) = s.tensor_kthvalue(t, k).unwrap();
            let ms = t0.elapsed().as_secs_f64() * 1e3;
            if ms < best {
                best = ms;
            }
            std::hint::black_box((&s, v));
        }
        println!("[{tag}] kthvalue k={k} f64 n={n}: {best:.1} ms");
    }
}

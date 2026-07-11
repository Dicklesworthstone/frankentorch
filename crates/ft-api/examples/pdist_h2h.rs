use ft_api::FrankenTorchSession;
use ft_core::ExecutionMode;
use std::time::Instant;

fn main() {
    let tag = std::env::var("FT_TAG").unwrap_or_else(|_| "FT".into());
    let (n, d) = (2000usize, 200usize);
    let a64: Vec<f64> = (0..n * d)
        .map(|i| ((i * 2654435761usize) % 10007) as f64 / 10007.0)
        .collect();
    let a32: Vec<f32> = a64.iter().map(|&x| x as f32).collect();
    for p in [1.0f64, 3.0] {
        for dt in ["f32", "f64"] {
            let mut best = f64::INFINITY;
            for _ in 0..5 {
                let mut s = FrankenTorchSession::new(ExecutionMode::Strict);
                let x = if dt == "f32" {
                    s.tensor_variable_f32(a32.clone(), vec![n, d], false)
                        .unwrap()
                } else {
                    s.tensor_variable(a64.clone(), vec![n, d], false).unwrap()
                };
                let t0 = Instant::now();
                let out = s.tensor_pdist(x, p).unwrap();
                let ms = t0.elapsed().as_secs_f64() * 1e3;
                if ms < best {
                    best = ms;
                }
                std::hint::black_box((&s, out));
            }
            println!("[{tag}] pdist p={p} {dt} [{n},{d}]: {best:.1} ms");
        }
    }
}

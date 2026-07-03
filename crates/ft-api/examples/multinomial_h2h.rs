use ft_api::FrankenTorchSession;
use ft_core::ExecutionMode;
use std::time::Instant;

fn main() {
    let tag = std::env::var("FT_TAG").unwrap_or_else(|_| "FT".into());
    for (b, c) in [(4096usize, 4096usize), (2048, 8192), (4096, 50000)] {
        // weights in (1e-3, 1+1e-3): materialize OUTSIDE the timer
        let data: Vec<f64> = (0..b * c)
            .map(|i| 1e-3 + ((i % 9973) as f64) / 9973.0)
            .collect();
        for ns in [1usize, 8] {
            let mut best = f64::INFINITY;
            for _ in 0..5 {
                let mut s = FrankenTorchSession::new(ExecutionMode::Strict);
                let w = s.tensor_variable(data.clone(), vec![b, c], false).unwrap();
                let t0 = Instant::now();
                let out = s.multinomial(w, ns, true).unwrap();
                let ms = t0.elapsed().as_secs_f64() * 1e3;
                if ms < best {
                    best = ms;
                }
                std::hint::black_box((&s, out));
            }
            println!("[{tag}] multinomial[{b},{c}] ns={ns} repl: {best:.1} ms");
        }
    }
}

use ft_api::FrankenTorchSession;
use ft_core::ExecutionMode;
use std::time::Instant;

fn main() {
    let tag = std::env::var("FT_TAG").unwrap_or_else(|_| "FT".into());
    let n = 16_000_000usize;
    let shape = vec![n];
    macro_rules! bench {
        ($name:expr, $s:ident, $t:ident, $body:expr) => {{
            let mut best = f64::INFINITY;
            for _ in 0..6 {
                let mut $s = FrankenTorchSession::new(ExecutionMode::Strict);
                let $t = $s
                    .tensor_variable(vec![0.0; n], shape.clone(), false)
                    .unwrap();
                let t0 = Instant::now();
                $body;
                let ms = t0.elapsed().as_secs_f64() * 1e3;
                if ms < best {
                    best = ms;
                }
                std::hint::black_box(&$s);
            }
            println!("[{tag}] {:<14} n={n}: {best:.1} ms", $name);
        }};
    }
    bench!("normal_", s, t, s.tensor_normal_(t, 0.0, 1.0).unwrap());
    bench!("uniform_", s, t, s.tensor_uniform_(t, 0.0, 1.0).unwrap());
    bench!("exponential_", s, t, s.tensor_exponential_(t, 1.5).unwrap());
    bench!("cauchy_", s, t, s.tensor_cauchy_(t, 0.0, 1.0).unwrap());
    bench!(
        "log_normal_",
        s,
        t,
        s.tensor_log_normal_(t, 0.0, 1.0).unwrap()
    );
    bench!("geometric_", s, t, s.tensor_geometric_(t, 0.3).unwrap());
}

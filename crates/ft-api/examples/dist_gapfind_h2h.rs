use ft_api::FrankenTorchSession;
use ft_core::ExecutionMode;
use std::time::Instant;

fn main() {
    let n = 16_000_000usize;
    let shape = vec![4000usize, 4000usize];
    // continuous, gamma-based & inverse-cdf
    macro_rules! bench {
        ($name:expr, $body:expr) => {{
            let mut best = f64::INFINITY;
            for _ in 0..5 {
                let mut s = FrankenTorchSession::new(ExecutionMode::Strict);
                let t0 = Instant::now();
                let _y = $body(&mut s);
                let ms = t0.elapsed().as_secs_f64() * 1e3;
                if ms < best {
                    best = ms;
                }
                std::hint::black_box(&s);
            }
            println!("[FT] {:<16} n={n}: {best:.2} ms", $name);
        }};
    }

    bench!("exponential", |s: &mut FrankenTorchSession| s
        .tensor_exponential(1.5, shape.clone(), false)
        .unwrap());
    bench!("gamma(2.5)", |s: &mut FrankenTorchSession| s
        .tensor_gamma(2.5, 1.0, shape.clone(), false)
        .unwrap());
    bench!("gamma(0.5)", |s: &mut FrankenTorchSession| s
        .tensor_gamma(0.5, 1.0, shape.clone(), false)
        .unwrap());
    bench!("chi2(4)", |s: &mut FrankenTorchSession| s
        .tensor_chi2(4.0, shape.clone(), false)
        .unwrap());
    bench!("studentt(5)", |s: &mut FrankenTorchSession| s
        .tensor_studentt(5.0, shape.clone(), false)
        .unwrap());
    bench!("beta(2,3)", |s: &mut FrankenTorchSession| s
        .tensor_beta(2.0, 3.0, shape.clone(), false)
        .unwrap());
    bench!("poisson(4)", |s: &mut FrankenTorchSession| {
        let rate = s
            .tensor_variable(vec![4.0; n], shape.clone(), false)
            .unwrap();
        s.tensor_poisson(rate).unwrap()
    });
    bench!("binomial(20)", |s: &mut FrankenTorchSession| s
        .tensor_binomial(20, 0.3, shape.clone(), false)
        .unwrap());
    // dirichlet: output shape + [K]; use [4000,4000] rows of K=4
    bench!("dirichlet(k=4)", |s: &mut FrankenTorchSession| s
        .tensor_dirichlet(&[1.5, 2.0, 0.7, 1.0], vec![4_000_000], false)
        .unwrap());
}

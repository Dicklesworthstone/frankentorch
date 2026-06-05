//! cdist p=2 (Euclidean): same-binary A/B of the OLD broadcasted path
//! (unsqueeze+expand+sub+abs+pow+sum_dim+pow, materialising the [P,R,M]
//! intermediate) vs the NEW matmul identity ‖x1‖²+‖x2‖²−2·x1·x2ᵀ (tensor_cdist).
//! Both run in one bench binary, so the ratio is immune to worker variance:
//!   cargo bench -p ft-api --bench cdist_bench
//! `cdist_broadcast/PxRxM` reproduces the pre-change op graph; `cdist_mm/PxRxM`
//! drives the production tensor_cdist (now the matmul fast path).

use criterion::{Criterion, black_box, criterion_group, criterion_main};
use ft_api::FrankenTorchSession;
use ft_core::ExecutionMode;

fn bench_cdist(c: &mut Criterion) {
    for &(p, r, m) in &[(256usize, 256usize, 128usize), (512, 512, 64)] {
        let x1v: Vec<f64> = (0..p * m).map(|i| (i as f64 * 0.013).sin()).collect();
        let x2v: Vec<f64> = (0..r * m).map(|i| (i as f64 * 0.017).cos()).collect();

        // Pre-change broadcasted path: unsqueeze+expand+sub+abs+pow+sum_dim+pow,
        // materialising the [P,R,M] intermediate.
        c.bench_function(&format!("cdist_broadcast/{p}x{r}x{m}"), |b| {
            let mut s = FrankenTorchSession::new(ExecutionMode::Strict);
            let x1 = s.tensor_variable(x1v.clone(), vec![p, m], false).unwrap();
            let x2 = s.tensor_variable(x2v.clone(), vec![r, m], false).unwrap();
            b.iter(|| {
                let x1_u = s.tensor_unsqueeze(x1, 1).unwrap();
                let x2_u = s.tensor_unsqueeze(x2, 0).unwrap();
                let target = vec![p, r, m];
                let x1_e = s.tensor_expand(x1_u, target.clone()).unwrap();
                let x2_e = s.tensor_expand(x2_u, target).unwrap();
                let diff = s.tensor_sub(x1_e, x2_e).unwrap();
                let abs = s.tensor_abs(diff).unwrap();
                let pw = s.tensor_pow(abs, 2.0).unwrap();
                let sum = s.tensor_sum_dim(pw, 2).unwrap();
                black_box(s.tensor_pow(sum, 0.5).unwrap());
            });
        });

        c.bench_function(&format!("cdist_mm/{p}x{r}x{m}"), |b| {
            let mut s = FrankenTorchSession::new(ExecutionMode::Strict);
            let x1 = s.tensor_variable(x1v.clone(), vec![p, m], false).unwrap();
            let x2 = s.tensor_variable(x2v.clone(), vec![r, m], false).unwrap();
            b.iter(|| black_box(s.tensor_cdist(black_box(x1), black_box(x2), 2.0).unwrap()));
        });
    }
}

criterion_group!(benches, bench_cdist);
criterion_main!(benches);

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

fn bench_pdist(c: &mut Criterion) {
    for &(n, m) in &[(256usize, 128usize), (512, 64)] {
        let out_len = n * (n - 1) / 2;
        let xv: Vec<f64> = (0..n * m).map(|i| (i as f64 * 0.013).sin()).collect();
        // Pre-change broadcasted path: gather i<j row pairs, then sub+abs+pow+sum_dim+pow.
        let mut ri = Vec::with_capacity(out_len);
        let mut rj = Vec::with_capacity(out_len);
        for i in 0..n {
            for j in (i + 1)..n {
                ri.push(i as f64);
                rj.push(j as f64);
            }
        }
        c.bench_function(&format!("pdist_broadcast/{n}x{m}"), |b| {
            let mut s = FrankenTorchSession::new(ExecutionMode::Strict);
            let x = s.tensor_variable(xv.clone(), vec![n, m], false).unwrap();
            let ri_t = s.tensor_variable(ri.clone(), vec![out_len], false).unwrap();
            let rj_t = s.tensor_variable(rj.clone(), vec![out_len], false).unwrap();
            b.iter(|| {
                let left = s.tensor_index_select(x, 0, ri_t).unwrap();
                let right = s.tensor_index_select(x, 0, rj_t).unwrap();
                let diff = s.tensor_sub(left, right).unwrap();
                let abs = s.tensor_abs(diff).unwrap();
                let pw = s.tensor_pow(abs, 2.0).unwrap();
                let sum = s.tensor_sum_dim(pw, 1).unwrap();
                black_box(s.tensor_pow(sum, 0.5).unwrap());
            });
        });
        c.bench_function(&format!("pdist_mm/{n}x{m}"), |b| {
            let mut s = FrankenTorchSession::new(ExecutionMode::Strict);
            let x = s.tensor_variable(xv.clone(), vec![n, m], false).unwrap();
            b.iter(|| black_box(s.tensor_pdist(black_box(x), 2.0).unwrap()));
        });
    }
}

/// cdist p=1 (Manhattan): the p≠2 path has no matmul identity, so the OLD code
/// materialised the [P,R,M] broadcast difference and reduced it (expand+sub+abs+
/// pow+sum_dim+pow) for BOTH grad and no-grad. `cdist_p1_broadcast` reproduces
/// that op graph; `cdist_p1_fused` drives the production tensor_cdist no-grad
/// fast path (ft_kernel_cpu::cdist_forward_f64, one streaming pass, no [P,R,M]).
fn bench_cdist_p1(c: &mut Criterion) {
    for &(p, r, m) in &[(256usize, 256usize, 128usize), (512, 512, 64)] {
        let x1v: Vec<f64> = (0..p * m).map(|i| (i as f64 * 0.013).sin()).collect();
        let x2v: Vec<f64> = (0..r * m).map(|i| (i as f64 * 0.017).cos()).collect();

        c.bench_function(&format!("cdist_p1_broadcast/{p}x{r}x{m}"), |b| {
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
                let pw = s.tensor_pow(abs, 1.0).unwrap();
                let sum = s.tensor_sum_dim(pw, 2).unwrap();
                black_box(s.tensor_pow(sum, 1.0).unwrap());
            });
        });

        c.bench_function(&format!("cdist_p1_fused/{p}x{r}x{m}"), |b| {
            let mut s = FrankenTorchSession::new(ExecutionMode::Strict);
            let x1 = s.tensor_variable(x1v.clone(), vec![p, m], false).unwrap();
            let x2 = s.tensor_variable(x2v.clone(), vec![r, m], false).unwrap();
            b.iter(|| black_box(s.tensor_cdist(black_box(x1), black_box(x2), 1.0).unwrap()));
        });
    }
}

/// pdist p=1: the p≠2 op-graph materialises the [out_len, M] pair-difference
/// tensor; `pdist_p1_fused` drives the production no-grad fast path
/// (ft_kernel_cpu::pdist_forward_f64).
fn bench_pdist_p1(c: &mut Criterion) {
    for &(n, m) in &[(256usize, 128usize), (512, 64)] {
        let out_len = n * (n - 1) / 2;
        let xv: Vec<f64> = (0..n * m).map(|i| (i as f64 * 0.013).sin()).collect();
        let mut ri = Vec::with_capacity(out_len);
        let mut rj = Vec::with_capacity(out_len);
        for i in 0..n {
            for j in (i + 1)..n {
                ri.push(i as f64);
                rj.push(j as f64);
            }
        }
        c.bench_function(&format!("pdist_p1_broadcast/{n}x{m}"), |b| {
            let mut s = FrankenTorchSession::new(ExecutionMode::Strict);
            let x = s.tensor_variable(xv.clone(), vec![n, m], false).unwrap();
            let ri_t = s.tensor_variable(ri.clone(), vec![out_len], false).unwrap();
            let rj_t = s.tensor_variable(rj.clone(), vec![out_len], false).unwrap();
            b.iter(|| {
                let left = s.tensor_index_select(x, 0, ri_t).unwrap();
                let right = s.tensor_index_select(x, 0, rj_t).unwrap();
                let diff = s.tensor_sub(left, right).unwrap();
                let abs = s.tensor_abs(diff).unwrap();
                let pw = s.tensor_pow(abs, 1.0).unwrap();
                let sum = s.tensor_sum_dim(pw, 1).unwrap();
                black_box(s.tensor_pow(sum, 1.0).unwrap());
            });
        });
        c.bench_function(&format!("pdist_p1_fused/{n}x{m}"), |b| {
            let mut s = FrankenTorchSession::new(ExecutionMode::Strict);
            let x = s.tensor_variable(xv.clone(), vec![n, m], false).unwrap();
            b.iter(|| black_box(s.tensor_pdist(black_box(x), 1.0).unwrap()));
        });
    }
}

criterion_group!(benches, bench_cdist, bench_pdist, bench_cdist_p1, bench_pdist_p1);
criterion_main!(benches);

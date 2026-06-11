//! A/B: public f32 inverse — native f32 LU (b3o90, lu_factor_contiguous_f32 +
//! lu_solve_contiguous_f32 on an identity RHS) vs the old f32->f64->f32 upcast
//! path (full f64 LU). f32 solve deliberately stays on f64 mixed-refine (faster
//! + more accurate), so only inverse is wired native. frankentorch-b3o90.
//!   cargo run -q --release -p ft-api --example lu_f32_probe
use ft_api::FrankenTorchSession;
use ft_core::{DType, ExecutionMode};
use std::time::Instant;

fn diagdom(n: usize) -> Vec<f32> {
    let mut a = vec![0.0f32; n * n];
    for i in 0..n {
        for j in 0..n {
            a[i * n + j] = (((i * 31 + j * 17) % 13) as f32) * 0.1 - 0.6;
        }
        a[i * n + i] += n as f32;
    }
    a
}

fn main() {
    let it = 10;
    for &n in &[256usize, 512] {
        let a = diagdom(n);
        let mut s = FrankenTorchSession::new(ExecutionMode::Strict);

        // ---- inverse ----
        let warm = s.tensor_variable_f32(a.clone(), vec![n, n], false).unwrap();
        let _ = s.tensor_linalg_inv(warm).unwrap();
        let t0 = Instant::now();
        for _ in 0..it {
            let aa = s.tensor_variable_f32(a.clone(), vec![n, n], false).unwrap();
            std::hint::black_box(s.tensor_linalg_inv(aa).unwrap());
        }
        let native = t0.elapsed().as_secs_f64() * 1e3 / it as f64;
        let t1 = Instant::now();
        for _ in 0..it {
            let aa = s.tensor_variable_f32(a.clone(), vec![n, n], false).unwrap();
            let a64 = s.tensor_to_dtype(aa, DType::F64).unwrap();
            let i64 = s.tensor_linalg_inv(a64).unwrap();
            std::hint::black_box(s.tensor_to_dtype(i64, DType::F32).unwrap());
        }
        let up = t1.elapsed().as_secs_f64() * 1e3 / it as f64;
        println!(
            "inverse      f32 {n}x{n}: native={native:.2}ms  f64-upcast={up:.2}ms  speedup={:.2}x",
            up / native
        );
    }
}

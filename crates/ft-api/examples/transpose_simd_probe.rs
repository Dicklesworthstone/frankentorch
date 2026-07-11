//! Correctness + bench of ft_kernel_cpu::transpose_2d_f64 (AVX2 register-blocked) vs scalar ref
//! and vs the current ft-api movedim path. Run: cargo run --release -p ft-api --example transpose_simd_probe
use ft_api::FrankenTorchSession;
use ft_core::ExecutionMode;
use std::time::Instant;

fn scalar_t(src: &[f64], rows: usize, cols: usize) -> Vec<f64> {
    let mut d = vec![0.0; rows * cols];
    for i in 0..rows {
        for j in 0..cols {
            d[j * rows + i] = src[i * cols + j];
        }
    }
    d
}

fn main() {
    // correctness across sizes incl non-multiple-of-4 + edges
    for &(r, c) in &[
        (1usize, 1usize),
        (4, 4),
        (7, 5),
        (5, 7),
        (16, 16),
        (17, 19),
        (4000, 4001),
        (4001, 4000),
        (1, 9),
        (9, 1),
    ] {
        let src: Vec<f64> = (0..r * c).map(|i| (i as f64) * 0.5 - 3.0).collect();
        let got = ft_kernel_cpu::transpose_2d_f64(&src, r, c);
        let want = scalar_t(&src, r, c);
        assert_eq!(got, want, "MISMATCH at {r}x{c}");
    }
    println!("correctness: ALL PASS (bit-exact vs scalar reference)");

    const N: usize = 4000;
    let src: Vec<f64> = (0..N * N).map(|i| ((i % 9973) as f64) - 4986.0).collect();
    let best = |mut f: Box<dyn FnMut()>| {
        let mut b = f64::INFINITY;
        for _ in 0..9 {
            let t = Instant::now();
            f();
            let e = t.elapsed().as_secs_f64() * 1e3;
            if e < b {
                b = e;
            }
        }
        b
    };

    let s = src.clone();
    let simd = best(Box::new(move || {
        let o = ft_kernel_cpu::transpose_2d_f64(&s, N, N);
        std::hint::black_box(&o);
    }));
    println!("SIMD transpose_2d_f64 [{N},{N}]: {simd:.3} ms");

    // current ft-api movedim path (op-only, build outside timer)
    let mut mv = f64::INFINITY;
    for _ in 0..9 {
        let mut sess = FrankenTorchSession::new(ExecutionMode::Strict);
        let x = sess
            .tensor_variable(src.clone(), vec![N, N], false)
            .unwrap();
        let t = Instant::now();
        let _ = sess.tensor_movedim(x, 0, 1);
        let e = t.elapsed().as_secs_f64() * 1e3;
        if e < mv {
            mv = e;
        }
    }
    println!("ft-api movedim (current scalar-tiled): {mv:.3} ms");
    println!("SIMD vs current: {:.2}x", mv / simd);
}

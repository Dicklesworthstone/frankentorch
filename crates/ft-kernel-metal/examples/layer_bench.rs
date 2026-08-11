//! Per-op attribution for one fused encoder layer at turbo shapes.
//!
//! Runs the exact op sequence of `EncoderGpu::forward`'s layer body
//! (seq=1500, d_model=1280, n_heads=20, d_ff=5120) twice over:
//! once as ONE batch (the production shape, one sync), and once with a
//! sync after each op class so the wall time attributes per op.
//!
//! Usage: cargo run --release -p ft-kernel-metal --example layer_bench [reps]

use ft_kernel_metal::fused::{Batch, GpuTensor, LN_EPS};

const SEQ: usize = 1500;
const D: usize = 1280;
const NH: usize = 20;
const DFF: usize = 5120;

fn v(n: usize, s: f32, o: f32) -> Vec<f32> {
    (0..n).map(|i| ((i % 13) as f32) * s - o).collect()
}

fn main() {
    let reps: usize = std::env::args()
        .nth(1)
        .and_then(|a| a.parse().ok())
        .unwrap_or(5);
    if !ft_kernel_metal::fused::is_available() {
        eprintln!("fused Metal pipelines unavailable");
        std::process::exit(1);
    }

    let up = |data: &[f32], r: usize, c: usize| GpuTensor::upload(data, r, c).expect("upload");
    let x = up(&v(SEQ * D, 0.002, 0.9), SEQ, D);
    let ln_g = up(&v(D, 0.001, -0.9), 1, D); // ~1.0 gamma
    let ln_b = up(&v(D, 0.0005, 0.02), 1, D);
    let wq = up(&v(D * D, 0.0004, 0.02), D, D);
    let bq = up(&v(D, 0.001, 0.01), 1, D);
    let wk = up(&v(D * D, 0.0003, 0.015), D, D);
    let wv = up(&v(D * D, 0.00035, 0.018), D, D);
    let bv = up(&v(D, 0.001, 0.01), 1, D);
    let wo = up(&v(D * D, 0.0003, 0.017), D, D);
    let bo = up(&v(D, 0.001, 0.01), 1, D);
    let w1 = up(&v(D * DFF, 0.0002, 0.01), D, DFF);
    let b1 = up(&v(DFF, 0.0005, 0.005), 1, DFF);
    let w2 = up(&v(DFF * D, 0.0002, 0.01), DFF, D);
    let b2 = up(&v(D, 0.001, 0.01), 1, D);

    // Whole layer, one batch, one sync (production shape).
    let mut best_all = f64::MAX;
    for _ in 0..reps {
        let t = std::time::Instant::now();
        let b = Batch::new().unwrap();
        let n1 = b.layernorm(&x, &ln_g, &ln_b, LN_EPS);
        let q = b.matmul_bias(&n1, &wq, Some(&bq));
        let k = b.matmul_bias(&n1, &wk, None);
        let vv = b.matmul_bias(&n1, &wv, Some(&bv));
        let attn = b.mha(&q, &k, &vv, NH);
        let ao = b.matmul_bias(&attn, &wo, Some(&bo));
        let x1 = b.add(&x, &ao);
        let n2 = b.layernorm(&x1, &ln_g, &ln_b, LN_EPS);
        let fc = b.matmul_bias(&n2, &w1, Some(&b1));
        let g = b.gelu(&fc);
        let proj = b.matmul_bias(&g, &w2, Some(&b2));
        let _x2 = b.add(&x1, &proj);
        b.finish();
        best_all = best_all.min(t.elapsed().as_secs_f64() * 1e3);
    }
    println!(
        "whole layer (1 batch, 1 sync): best {best_all:8.2} ms  -> x32 layers = {:7.2} s",
        best_all * 32.0 / 1e3
    );

    // Per-op attribution (separate batch + sync per op class; sync overhead
    // inflates each row a little, so rows can sum above the whole-layer time).
    let time_op = |label: &str, f: &dyn Fn(&Batch) -> GpuTensor| {
        let mut best = f64::MAX;
        for _ in 0..reps {
            let t = std::time::Instant::now();
            let b = Batch::new().unwrap();
            let _o = f(&b);
            b.finish();
            best = best.min(t.elapsed().as_secs_f64() * 1e3);
        }
        println!("  {label:<28} best {best:8.2} ms");
        best
    };

    // Precompute resident inputs for attention timing at real magnitudes.
    let b = Batch::new().unwrap();
    let n1 = b.layernorm(&x, &ln_g, &ln_b, LN_EPS);
    let q = b.matmul_bias(&n1, &wq, Some(&bq));
    let k = b.matmul_bias(&n1, &wk, None);
    let vv = b.matmul_bias(&n1, &wv, Some(&bv));
    let fc = b.matmul_bias(&n1, &w1, Some(&b1));
    b.finish();

    println!("per-op (separate sync each):");
    let mut sum = 0.0;
    sum += time_op("layernorm [1500x1280]", &|b| {
        b.layernorm(&x, &ln_g, &ln_b, LN_EPS)
    });
    sum += time_op("matmul q [1280->1280]", &|b| {
        b.matmul_bias(&n1, &wq, Some(&bq))
    });
    sum += 2.0 * time_op("matmul k/v (x2)", &|b| b.matmul_bias(&n1, &wk, None));
    sum += time_op("mha flash (20 heads)", &|b| b.mha(&q, &k, &vv, NH));
    sum += time_op("matmul out [1280->1280]", &|b| {
        b.matmul_bias(&q, &wo, Some(&bo))
    });
    sum += 2.0 * time_op("residual add (x2)", &|b| b.add(&x, &q));
    sum += time_op("layernorm2", &|b| b.layernorm(&x, &ln_g, &ln_b, LN_EPS));
    sum += time_op("matmul fc [1280->5120]", &|b| {
        b.matmul_bias(&n1, &w1, Some(&b1))
    });
    sum += time_op("gelu [1500x5120]", &|b| b.gelu(&fc));
    sum += time_op("matmul proj [5120->1280]", &|b| {
        b.matmul_bias(&fc, &w2, Some(&b2))
    });
    println!("  {:<28} sum  {sum:8.2} ms", "(op rows)");
}

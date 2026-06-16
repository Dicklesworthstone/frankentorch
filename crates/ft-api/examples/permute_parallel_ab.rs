//! A/B for parallelized cache-blocked permute_slice (frankentorch-permpar). The
//! block-rotation transpose fast path (used by every transpose/permute: attention
//! head reshapes [B,S,H,D]<->[B,H,S,D], NCHW<->NHWC conv layout, batched .mT, plain
//! 2-D transpose) was serial; now it fans the independent planes / output row-tiles
//! over rayon. Pure data movement → bit-for-bit identical (the ft-autograd lib
//! tests prove it). This measures serial-vs-parallel of the SAME kernel in ONE
//! process via rayon pools (1 thread vs all cores), timing ONLY the permute op
//! (fresh session per iter so gmuml's never-freed tape can't OOM, and the input
//! copy stays outside the timed region). Writes the ratios to a repo file.
//!   cargo run -q --release -p ft-api --example permute_parallel_ab
use ft_api::FrankenTorchSession;
use ft_core::ExecutionMode;
use rayon::ThreadPool;
use std::time::Instant;

fn time_op(pool: &ThreadPool, av: &[f64], shape: &[usize], transpose: bool) -> f64 {
    let mut best = f64::INFINITY;
    for _ in 0..25 {
        let mut s = FrankenTorchSession::new(ExecutionMode::Strict);
        let x = s.tensor_variable(av.to_vec(), shape.to_vec(), false).unwrap();
        let t = Instant::now();
        pool.install(|| {
            if transpose {
                s.tensor_transpose(x, 0, 1).unwrap();
            } else {
                s.tensor_permute(x, vec![0, 2, 1, 3]).unwrap();
            }
        });
        best = best.min(t.elapsed().as_secs_f64() * 1e3);
    }
    best
}

fn main() {
    let pool1 = rayon::ThreadPoolBuilder::new().num_threads(1).build().unwrap();
    let pooln = rayon::ThreadPoolBuilder::new().build().unwrap();
    let nthreads = pooln.current_num_threads();
    let mut out = String::new();

    // (1) Attention head permute [B, S, H, D] -> [B, H, S, D] (perm [0,2,1,3]).
    let (b, sq, h, d) = (32usize, 512usize, 16usize, 64usize);
    let av: Vec<f64> = (0..b * sq * h * d).map(|i| (i % 1009) as f64 * 0.001).collect();
    let s1 = time_op(&pool1, &av, &[b, sq, h, d], false);
    let sn = time_op(&pooln, &av, &[b, sq, h, d], false);
    let line1 = format!(
        "attn[32,512,16,64]->[0,2,1,3]: serial {s1:.3} ms / parallel({nthreads}t) {sn:.3} ms / {:.2}x",
        s1 / sn
    );
    eprintln!("{line1}");
    out.push_str(&line1);
    out.push('\n');

    // (2) Plain 2-D transpose [N, N] (single plane -> row-tile parallel path).
    let n = 4096usize;
    let tv: Vec<f64> = (0..n * n).map(|i| (i % 1009) as f64 * 0.001).collect();
    let t1 = time_op(&pool1, &tv, &[n, n], true);
    let tn = time_op(&pooln, &tv, &[n, n], true);
    let line2 = format!(
        "transpose[4096,4096]: serial {t1:.3} ms / parallel({nthreads}t) {tn:.3} ms / {:.2}x",
        t1 / tn
    );
    eprintln!("{line2}");
    out.push_str(&line2);
    out.push('\n');

    // (3) Constant pad of a CNN feature map [N,C,H,W] by 1 on H,W (hot in conv).
    let (pn, pc, ph, pw) = (16usize, 64usize, 64usize, 64usize);
    let pv: Vec<f64> = (0..pn * pc * ph * pw).map(|i| (i % 1009) as f64 * 0.001).collect();
    let pad_op = |pool: &ThreadPool| -> f64 {
        let mut best = f64::INFINITY;
        for _ in 0..25 {
            let mut s = FrankenTorchSession::new(ExecutionMode::Strict);
            let x = s.tensor_variable(pv.clone(), vec![pn, pc, ph, pw], false).unwrap();
            let t = Instant::now();
            pool.install(|| {
                s.tensor_pad(x, &[1, 1, 1, 1], 0.0).unwrap();
            });
            best = best.min(t.elapsed().as_secs_f64() * 1e3);
        }
        best
    };
    let p1 = pad_op(&pool1);
    let pp = pad_op(&pooln);
    let line3 = format!(
        "pad[16,64,64,64]+1hw: serial {p1:.3} ms / parallel({nthreads}t) {pp:.3} ms / {:.2}x",
        p1 / pp
    );
    eprintln!("{line3}");
    out.push_str(&line3);
    out.push('\n');

    let _ = std::fs::write("artifacts/perf/permute_parallel_ab_result.txt", &out);
}

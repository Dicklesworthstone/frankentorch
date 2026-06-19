//! Phase-timing probe for the avg_pool1d 8M train-step gauntlet lane (kgs4.122).
//! Breaks the ~180ms train step into create / forward / sum / backward / grad-read
//! to root-cause the 25x PyTorch gap. BlackThrush.
use ft_api::FrankenTorchSession;
use ft_core::ExecutionMode;
use std::time::Instant;

const N: usize = 8;
const C: usize = 64;
const L: usize = 8192;

fn values() -> Vec<f64> {
    let total = N * C * L;
    (0..total)
        .map(|i| ((i % 251) as f64) * 0.001 - 0.12)
        .collect()
}

fn main() {
    let base = values();
    let shape = vec![N, C, L];
    let iters: usize = std::env::var("ITERS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(8);

    // warm up
    {
        let mut s = FrankenTorchSession::new(ExecutionMode::Strict);
        let x = s
            .tensor_variable(base.clone(), shape.clone(), true)
            .unwrap();
        let out = s.functional_avg_pool1d(x, 2, 2).unwrap();
        let loss = s.tensor_sum(out).unwrap();
        s.tensor_backward(loss).unwrap();
    }

    let mut t_new = 0u128;
    let mut t_var = 0u128;
    let mut t_fwd = 0u128;
    let mut t_sum = 0u128;
    let mut t_bwd = 0u128;
    let total_start = Instant::now();
    for _ in 0..iters {
        let a = Instant::now();
        let mut s = FrankenTorchSession::new(ExecutionMode::Strict);
        t_new += a.elapsed().as_micros();

        let a = Instant::now();
        let x = s
            .tensor_variable(base.clone(), shape.clone(), true)
            .unwrap();
        t_var += a.elapsed().as_micros();

        let a = Instant::now();
        let out = s.functional_avg_pool1d(x, 2, 2).unwrap();
        t_fwd += a.elapsed().as_micros();

        let a = Instant::now();
        let loss = s.tensor_sum(out).unwrap();
        t_sum += a.elapsed().as_micros();

        let a = Instant::now();
        let res = s.tensor_backward(loss).unwrap();
        std::hint::black_box(&res);
        t_bwd += a.elapsed().as_micros();
    }
    let total = total_start.elapsed().as_micros();
    let n = iters as u128;

    // Isolate the raw kernels (no autograd engine) to localize the overhead.
    let output_len = (L - 2) / 2 + 1;
    let dout = vec![1.0f64; N * C * output_len];
    let mut t_kfwd = 0u128;
    let mut t_kbwd = 0u128;
    for _ in 0..iters {
        let a = Instant::now();
        let o = ft_kernel_cpu::avg_pool1d_forward_f64(&base, N, C, L, 2, output_len, 2);
        std::hint::black_box(&o);
        t_kfwd += a.elapsed().as_micros();
        let a = Instant::now();
        let di = ft_kernel_cpu::avg_pool1d_backward_f64(&dout, N, C, L, 2, output_len, 2);
        std::hint::black_box(&di);
        t_kbwd += a.elapsed().as_micros();
    }
    println!("  RAW kfwd    : {:>8.1}", t_kfwd as f64 / n as f64);
    println!("  RAW kbwd    : {:>8.1}", t_kbwd as f64 / n as f64);

    // Control tape: sum(x).backward() on the SAME 4M leaf, NO custom function.
    // Isolates the generic grads-alloc + sum-arm + report/persistent machinery
    // from the avg_pool1d CustomFunction dispatch.
    let mut t_sumonly_bwd = 0u128;
    for _ in 0..iters {
        let mut s = FrankenTorchSession::new(ExecutionMode::Strict);
        let x = s
            .tensor_variable(base.clone(), shape.clone(), true)
            .unwrap();
        let loss = s.tensor_sum(x).unwrap();
        let a = Instant::now();
        let r = s.tensor_backward(loss).unwrap();
        std::hint::black_box(&r);
        t_sumonly_bwd += a.elapsed().as_micros();
    }
    println!(
        "  SUMONLY bwd : {:>8.1}  (4M leaf, no custom fn)",
        t_sumonly_bwd as f64 / n as f64
    );

    // Decompose the raw primitives the generic Sum-arm backward performs on a 4M leaf.
    let m = N * C * L; // 4M
    let (mut t_zero, mut t_fill, mut t_acc_ser, mut t_acc_par, mut t_tovec) =
        (0u128, 0u128, 0u128, 0u128, 0u128);
    for _ in 0..iters {
        let a = Instant::now();
        let g = vec![0.0f64; m];
        std::hint::black_box(&g);
        t_zero += a.elapsed().as_micros();

        let a = Instant::now();
        let contrib = vec![1.0f64; m];
        std::hint::black_box(&contrib);
        t_fill += a.elapsed().as_micros();

        let mut target = vec![0.0f64; m];
        let a = Instant::now();
        for (t, v) in target.iter_mut().zip(contrib.iter()) {
            *t += *v;
        }
        std::hint::black_box(&target);
        t_acc_ser += a.elapsed().as_micros();

        let mut target2 = vec![0.0f64; m];
        let a = Instant::now();
        {
            use rayon::prelude::*;
            target2
                .par_iter_mut()
                .zip(contrib.par_iter())
                .for_each(|(t, v)| *t += *v);
        }
        std::hint::black_box(&target2);
        t_acc_par += a.elapsed().as_micros();

        let a = Instant::now();
        let cp = target.clone();
        std::hint::black_box(&cp);
        t_tovec += a.elapsed().as_micros();
    }
    println!("  -- raw primitives on 4M f64 (per-iter us) --");
    println!("  vec![0.0;m] : {:>8.1}", t_zero as f64 / n as f64);
    println!("  vec![1.0;m] : {:>8.1}", t_fill as f64 / n as f64);
    println!("  acc serial  : {:>8.1}", t_acc_ser as f64 / n as f64);
    println!("  acc rayon   : {:>8.1}", t_acc_par as f64 / n as f64);
    println!("  clone(tovec): {:>8.1}", t_tovec as f64 / n as f64);

    // ---- SAME-PROCESS A/B: old vs new Sum-arm gradient accumulation ----
    // OLD: materialize `vec![scalar; m]` then `target[i] += contrib[i]`.
    // NEW: `target[i] += scalar` lazily (no contrib alloc/fill/read).
    // Interleaved per-iter to share the same worker/contention window.
    let scalar = 1.0f64;
    let reps = iters * 8;
    // Reuse pre-faulted target buffers so first-touch page faults DON'T dominate
    // the timing (that confound makes the bandwidth-bound result worker- and
    // order-dependent — the rao3v trap). This isolates the actual cost removed:
    // the per-backward `vec![scalar; m]` alloc+fill+read of the contribution.
    let mut target_old = vec![0.0f64; m];
    let mut target_new = vec![0.0f64; m];
    for t in target_old.iter_mut() {
        *t = 1.0;
    }
    for t in target_new.iter_mut() {
        *t = 1.0;
    }
    let mut old_best = u128::MAX;
    let mut new_best = u128::MAX;
    let mut old_sum = 0u128;
    let mut new_sum = 0u128;
    for _ in 0..reps {
        let a = Instant::now();
        let contrib = vec![scalar; m];
        for (t, v) in target_old.iter_mut().zip(contrib.iter()) {
            *t += *v;
        }
        std::hint::black_box(&target_old);
        let e_old = a.elapsed().as_micros();
        old_best = old_best.min(e_old);
        old_sum += e_old;

        let a = Instant::now();
        for t in target_new.iter_mut() {
            *t += scalar;
        }
        std::hint::black_box(&target_new);
        let e_new = a.elapsed().as_micros();
        new_best = new_best.min(e_new);
        new_sum += e_new;
    }
    println!("  -- Sum-arm A/B (m=4M, {reps} reps, same process) --");
    println!(
        "  OLD fill+acc: mean {:>8.1}  min {:>8.1}",
        old_sum as f64 / reps as f64,
        old_best as f64
    );
    println!(
        "  NEW lazy acc: mean {:>8.1}  min {:>8.1}",
        new_sum as f64 / reps as f64,
        new_best as f64
    );
    println!(
        "  ratio(min)  : {:>6.2}x",
        old_best as f64 / new_best as f64
    );
    println!(
        "  ratio(mean) : {:>6.2}x",
        (old_sum as f64) / (new_sum as f64)
    );
    println!("avg_pool1d [8,64,8192] train step, {iters} iters, per-iter us:");
    println!("  session_new : {:>8.1}", t_new as f64 / n as f64);
    println!("  tensor_var  : {:>8.1}", t_var as f64 / n as f64);
    println!("  forward     : {:>8.1}", t_fwd as f64 / n as f64);
    println!("  sum         : {:>8.1}", t_sum as f64 / n as f64);
    println!("  backward    : {:>8.1}", t_bwd as f64 / n as f64);
    println!(
        "  TOTAL       : {:>8.1} ({:.3} ms)",
        total as f64 / n as f64,
        total as f64 / n as f64 / 1000.0
    );
}

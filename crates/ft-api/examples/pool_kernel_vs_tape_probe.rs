//! Where do `max_pool3d` (9.39x) and `avg_pool2d` (4.29x) actually lose?
//! `frankentorch-87sz8`, `frankentorch-k1h8g`.
//!
//! Both lanes' forward and backward kernels read as parallel and tight, so the
//! gap is not obvious from the source. This splits each lane three ways:
//!
//!   raw_fwd     the ft-kernel-cpu forward called DIRECTLY, no session, no tape
//!   raw_bwd     the ft-kernel-cpu backward called DIRECTLY
//!   session     the same work through FrankenTorchSession forward+backward
//!
//! `session - (raw_fwd + raw_bwd)` is what the autograd machinery costs on top of
//! the kernels. That single number decides the lever: if the kernels are already
//! near PyTorch's total, the target is the tape; if the kernels are themselves
//! slow, the target is the kernel.
//!
//! Deliberately FrankenTorch-vs-FrankenTorch — it is an attribution probe, not a
//! vs-PyTorch claim. The vs-PyTorch standing lives in
//! `artifacts/perf/frankentorch-kgs4-lane-sweep/`. For reference while reading the
//! output, PyTorch's whole-op numbers there were max_pool3d 0.660 ms and
//! avg_pool2d 1.833 ms.
//!
//! Run: `cargo run --release -p ft-api --features fair-alloc --example pool_kernel_vs_tape_probe`

use std::time::Instant;

use ft_api::FrankenTorchSession;
use ft_core::ExecutionMode;
use rayon::prelude::*;

const REPS: usize = 15;

// max_pool3d lane, shapes from pytorch_gauntlet_bench.
const M_N: usize = 2;
const M_C: usize = 32;
const M_D: usize = 16;
const M_H: usize = 32;
const M_W: usize = 32;
const M_OD: usize = M_D / 2;
const M_OH: usize = M_H / 2;
const M_OW: usize = M_W / 2;

// avg_pool2d lane.
const A_N: usize = 8;
const A_C: usize = 64;
const A_H: usize = 64;
const A_W: usize = 64;
const A_OH: usize = A_H / 2;
const A_OW: usize = A_W / 2;

fn median(mut values: Vec<f64>) -> f64 {
    values.sort_by(f64::total_cmp);
    values[values.len() / 2]
}

fn seq(n: usize) -> Vec<f64> {
    (0..n).map(|i| ((i % 251) as f64) * 0.001 - 0.12).collect()
}

fn time_it<F: FnMut()>(mut f: F) -> f64 {
    let mut samples = Vec::with_capacity(REPS);
    for _ in 0..3 {
        f();
    }
    for _ in 0..REPS {
        let started = Instant::now();
        f();
        samples.push(started.elapsed().as_secs_f64() * 1_000.0);
    }
    median(samples)
}

fn report(lane: &str, raw_fwd: f64, raw_bwd: f64, session: f64, pytorch_whole_op: f64) {
    let kernels = raw_fwd + raw_bwd;
    let tape = session - kernels;
    println!(
        "  {lane:<12} raw_fwd={raw_fwd:7.3}  raw_bwd={raw_bwd:7.3}  kernels={kernels:7.3}  session={session:7.3}  tape_overhead={tape:7.3} ({:.0}% of session)",
        100.0 * tape / session
    );
    println!(
        "               vs PyTorch whole-op {pytorch_whole_op:.3} ms: kernels alone are {:.2}x PyTorch; the tape adds {:.2}x more",
        kernels / pytorch_whole_op,
        tape / pytorch_whole_op
    );
}

fn main() {
    println!(
        "executing_elf_sha256={}",
        ft_api::harness_provenance::executing_elf_sha256()
    );
    println!(
        "allocator={}",
        if cfg!(feature = "fair-alloc") {
            "mimalloc (--features fair-alloc)"
        } else {
            "system (glibc malloc)"
        }
    );
    println!("FrankenTorch-vs-FrankenTorch attribution probe, reps={REPS} median\n");

    // ── max_pool3d ──────────────────────────────────────────────────────────
    let m_input = seq(M_N * M_C * M_D * M_H * M_W);
    let (_, m_args) = ft_kernel_cpu::max_pool3d_forward_with_indices_f64(
        &m_input, M_N, M_C, M_D, M_H, M_W, 2, 2, 2, M_OD, M_OH, M_OW, 2, 2, 2,
    );
    let m_dout = vec![1.0f64; M_N * M_C * M_OD * M_OH * M_OW];

    let m_raw_fwd = time_it(|| {
        std::hint::black_box(ft_kernel_cpu::max_pool3d_forward_with_indices_f64(
            &m_input, M_N, M_C, M_D, M_H, M_W, 2, 2, 2, M_OD, M_OH, M_OW, 2, 2, 2,
        ));
    });
    let m_raw_bwd = time_it(|| {
        std::hint::black_box(ft_kernel_cpu::max_pool3d_backward_from_indices_f64(
            &m_dout, &m_args, M_N, M_C, M_D, M_H, M_W, M_OD, M_OH, M_OW,
        ));
    });
    let m_session = time_it(|| {
        let mut session = FrankenTorchSession::new(ExecutionMode::Strict);
        let x = session
            .tensor_variable(m_input.clone(), vec![M_N, M_C, M_D, M_H, M_W], true)
            .expect("leaf");
        let out = session
            .functional_max_pool3d(x, (2, 2, 2), (2, 2, 2))
            .expect("max_pool3d");
        let loss = session.tensor_sum(out).expect("sum");
        let rep = session.tensor_backward(loss).expect("backward");
        std::hint::black_box(rep.gradient(x).expect("grad").iter().sum::<f64>());
    });

    // ── avg_pool2d ──────────────────────────────────────────────────────────
    let a_input = seq(A_N * A_C * A_H * A_W);
    let a_dout = vec![1.0f64; A_N * A_C * A_OH * A_OW];

    let a_raw_fwd = time_it(|| {
        std::hint::black_box(ft_kernel_cpu::avg_pool2d_forward_f64(
            &a_input, A_N, A_C, A_H, A_W, 2, 2, A_OH, A_OW, 2, 2, 0, 0, A_H, A_W, true,
        ));
    });
    let a_raw_bwd = time_it(|| {
        std::hint::black_box(ft_kernel_cpu::avg_pool2d_backward_f64(
            &a_dout, A_N, A_C, A_H, A_W, 2, 2, A_OH, A_OW, 2, 2, 0, 0, A_H, A_W, true,
        ));
    });
    let a_session = time_it(|| {
        let mut session = FrankenTorchSession::new(ExecutionMode::Strict);
        let x = session
            .tensor_variable(a_input.clone(), vec![A_N, A_C, A_H, A_W], true)
            .expect("leaf");
        let out = session
            .functional_avg_pool2d(x, (2, 2), (2, 2), (0, 0), false, true)
            .expect("avg_pool2d");
        let loss = session.tensor_sum(out).expect("sum");
        let rep = session.tensor_backward(loss).expect("backward");
        std::hint::black_box(rep.gradient(x).expect("grad").iter().sum::<f64>());
    });

    // ── frankentorch-uufyp: split `tape_overhead` into its parts ────────────
    // k1h8g found 91-93% of avg_pool2d's session arm sits outside the kernels. That
    // single number covers at least four different costs, and attacking it as a unit
    // would repeat 8obhh's mistake of attacking "the dense write" as a unit.
    //
    // CUMULATIVE lanes: each adds one stage to the previous one, so successive
    // DIFFERENCES attribute that stage. Reported as differences below.
    let a_shape = vec![A_N, A_C, A_H, A_W];
    let t_sess = time_it(|| {
        let session = FrankenTorchSession::new(ExecutionMode::Strict);
        std::hint::black_box(&session);
    });
    let t_leaf = time_it(|| {
        let mut session = FrankenTorchSession::new(ExecutionMode::Strict);
        let x = session
            .tensor_variable(a_input.clone(), a_shape.clone(), true)
            .expect("leaf");
        std::hint::black_box((&session, x));
    });
    let t_fwd = time_it(|| {
        let mut session = FrankenTorchSession::new(ExecutionMode::Strict);
        let x = session
            .tensor_variable(a_input.clone(), a_shape.clone(), true)
            .expect("leaf");
        let out = session
            .functional_avg_pool2d(x, (2, 2), (2, 2), (0, 0), false, true)
            .expect("avg_pool2d");
        std::hint::black_box((&session, out));
    });
    let t_sum = time_it(|| {
        let mut session = FrankenTorchSession::new(ExecutionMode::Strict);
        let x = session
            .tensor_variable(a_input.clone(), a_shape.clone(), true)
            .expect("leaf");
        let out = session
            .functional_avg_pool2d(x, (2, 2), (2, 2), (0, 0), false, true)
            .expect("avg_pool2d");
        let loss = session.tensor_sum(out).expect("sum");
        std::hint::black_box((&session, loss));
    });
    let t_bwd = time_it(|| {
        let mut session = FrankenTorchSession::new(ExecutionMode::Strict);
        let x = session
            .tensor_variable(a_input.clone(), a_shape.clone(), true)
            .expect("leaf");
        let out = session
            .functional_avg_pool2d(x, (2, 2), (2, 2), (0, 0), false, true)
            .expect("avg_pool2d");
        let loss = session.tensor_sum(out).expect("sum");
        let rep = session.tensor_backward(loss).expect("backward");
        std::hint::black_box((&session, &rep));
    });
    // A control the decomposition needs: how much of `leaf` is just cloning 16 MiB
    // of input, as opposed to anything the session does with it?
    let t_clone = time_it(|| {
        std::hint::black_box(a_input.clone());
    });
    println!("avg_pool2d SESSION DECOMPOSITION (frankentorch-uufyp), cumulative then differenced");
    println!(
        "  cumulative:  session={t_sess:6.3}  +leaf={t_leaf:6.3}  +fwd={t_fwd:6.3}  +sum={t_sum:6.3}  +bwd={t_bwd:6.3}  +grad_fetch_sum={a_session:6.3}"
    );
    println!(
        "  per stage:   session_new={t_sess:6.3}  leaf_build={:6.3}  forward={:6.3}  sum={:6.3}  backward={:6.3}  grad_fetch_sum={:6.3}",
        t_leaf - t_sess,
        t_fwd - t_leaf,
        t_sum - t_fwd,
        t_bwd - t_sum,
        a_session - t_bwd,
    );
    println!(
        "  of which leaf_build is a bare 16 MiB input clone: {t_clone:.3} ms ({:.0}% of leaf_build)",
        100.0 * t_clone / (t_leaf - t_sess).max(f64::MIN_POSITIVE)
    );
    println!(
        "  raw kernels for reference: fwd={a_raw_fwd:.3} bwd={a_raw_bwd:.3}. A stage difference at or\n\
         below the noise of these medians is NOT attributable; treat only the large terms.\n"
    );

    // ── frankentorch-27aci: does the backward STAGE scale with numel? ───────
    // uufyp put ~11 ms in the engine between "backward called" and "kernel returns",
    // against a 0.32 ms kernel. That is either DATA MOVEMENT (extra full-size passes)
    // or FIXED per-backward machinery, and the two want completely different levers.
    // Sweeping numel separates them in one run: linear => movement, flat => fixed.
    //
    // Each size reports the backward stage as (cumulative-through-backward minus
    // cumulative-through-sum), the same differencing the decomposition above uses,
    // plus the raw kernel at that size so the ratio is visible.
    println!("BACKWARD-STAGE NUMEL SWEEP (frankentorch-27aci)");
    println!(
        "{:>6}{:>9}{:>11}{:>11}{:>10}{:>9}",
        "N", "MiB", "bwd_stage", "raw_bwd", "ratio", "ms/MiB"
    );
    // frankentorch-27aci step 1: finer spacing AND both directions. N=8 is exactly the
    // shape every other lane in this probe uses, so by the time an ascending sweep
    // reaches it the allocator has a hot 16 MiB size class — which is a harness
    // artifact hypothesis, not a property of the stage. If the spike sits at N=8 in
    // BOTH directions it is the shape; if it moves, it is ordering/allocator state.
    let mut order: Vec<usize> = vec![5, 6, 7, 8, 9, 10, 12];
    for pass in 0..2 {
        if pass == 1 {
            order.reverse();
        }
        println!(
            "  --- pass {} ({}) ---",
            pass + 1,
            if pass == 0 { "ascending" } else { "descending" }
        );
        for &n in &order {
            let h = A_H;
            let w = A_W;
            let oh = h / 2;
            let ow = w / 2;
            let numel = n * A_C * h * w;
            let mib = (numel * 8) as f64 / (1024.0 * 1024.0);
            let inp = seq(numel);
            let shape = vec![n, A_C, h, w];
            let dout = vec![1.0f64; n * A_C * oh * ow];

            let through_sum = time_it(|| {
                let mut session = FrankenTorchSession::new(ExecutionMode::Strict);
                let x = session
                    .tensor_variable(inp.clone(), shape.clone(), true)
                    .expect("leaf");
                let out = session
                    .functional_avg_pool2d(x, (2, 2), (2, 2), (0, 0), false, true)
                    .expect("avg_pool2d");
                let loss = session.tensor_sum(out).expect("sum");
                std::hint::black_box((&session, loss));
            });
            let through_bwd = time_it(|| {
                let mut session = FrankenTorchSession::new(ExecutionMode::Strict);
                let x = session
                    .tensor_variable(inp.clone(), shape.clone(), true)
                    .expect("leaf");
                let out = session
                    .functional_avg_pool2d(x, (2, 2), (2, 2), (0, 0), false, true)
                    .expect("avg_pool2d");
                let loss = session.tensor_sum(out).expect("sum");
                let rep = session.tensor_backward(loss).expect("backward");
                std::hint::black_box((&session, &rep));
            });
            let raw = time_it(|| {
                std::hint::black_box(ft_kernel_cpu::avg_pool2d_backward_f64(
                    &dout, n, A_C, h, w, 2, 2, oh, ow, 2, 2, 0, 0, h, w, true,
                ));
            });
            let stage = through_bwd - through_sum;
            println!(
                "{n:>6}{mib:>9.0}{stage:>11.3}{raw:>11.3}{:>10.1}x{:>9.3}",
                stage / raw.max(f64::MIN_POSITIVE),
                stage / mib
            );
        }
    }
    println!(
        "  ms/MiB roughly CONSTANT => the stage is DATA MOVEMENT and the count of full-size\n\
         passes is the thing to hunt. ms/MiB falling as size grows => a FIXED per-backward\n\
         cost dominating at small sizes. Neither => look again before choosing a lever.\n"
    );

    // ── the parallel gate ───────────────────────────────────────────────────
    // POOL_FWD_PARALLEL_MIN is 1<<21 = 2_097_152 "input reads", and the gate is
    // `out.len() * kd*kh*kw`. The lane above computes 131072*8 = 1_048_576 —
    // exactly HALF the threshold — so its forward runs SINGLE-THREADED.
    //
    // Doubling the depth crosses the gate. If parallelism is worth having at this
    // size, 2x the work must cost materially LESS than 2x the time. If it costs
    // ~2x, the gate is correctly placed and this is a dead end.
    const M_D2: usize = M_D * 2;
    const M_OD2: usize = M_D2 / 2;
    let m_input2 = seq(M_N * M_C * M_D2 * M_H * M_W);
    let m_raw_fwd_2x = time_it(|| {
        std::hint::black_box(ft_kernel_cpu::max_pool3d_forward_with_indices_f64(
            &m_input2, M_N, M_C, M_D2, M_H, M_W, 2, 2, 2, M_OD2, M_OH, M_OW, 2, 2, 2,
        ));
    });
    let reads_1x = M_N * M_C * M_OD * M_OH * M_OW * 8;
    let reads_2x = M_N * M_C * M_OD2 * M_OH * M_OW * 8;
    println!(
        "parallel-gate probe: 1x reads={reads_1x} {m_raw_fwd:.3} ms | 2x reads={reads_2x} {m_raw_fwd_2x:.3} ms | ratio {:.2}x",
        m_raw_fwd_2x / m_raw_fwd
    );
    println!(
        "  Before frankentorch-87sz8 the 1x shape fell below POOL_FWD_PARALLEL_MIN and ran SERIAL:\n\
           it measured 2.136 ms while 2x the data ran parallel in 0.946 ms — twice the work in 0.44x\n\
           the time, which is what identified the gate as mis-set. With the per-plane clause in place\n\
           both shapes parallelise, so the ratio here should now sit near 1.0 (2x work, ~2x time per\n\
           unit) rather than below it. A ratio far below 1.0 again would mean a shape is stranded.\n"
    );

    // ── max_pool3d BACKWARD attribution (frankentorch-87sz8 next target) ────
    // The backward allocates a dense 8 MiB f64 gradient and scatters only
    // 131072 values into it — 1 element in 8. Three candidate costs:
    //   alloc_only        the zeroed 8 MiB allocation itself
    //   alloc_plus_touch  that allocation plus writing EVERY element once, which
    //                     is the floor for producing a dense buffer at all
    //   raw_bwd           the real backward
    // If raw_bwd ~ alloc_plus_touch the kernel is at its memory floor and the
    // scatter loop is free; if raw_bwd >> alloc_plus_touch, the loop is the cost.
    let din_len = M_N * M_C * M_D * M_H * M_W;
    let bwd_alloc_only = time_it(|| {
        std::hint::black_box(vec![0.0f64; din_len]);
    });
    let bwd_alloc_touch = time_it(|| {
        let mut v = vec![0.0f64; din_len];
        v.par_chunks_mut(M_D * M_H * M_W).for_each(|row| {
            for slot in row.iter_mut() {
                *slot = 1.0;
            }
        });
        std::hint::black_box(v);
    });
    println!(
        "max_pool3d backward attribution: alloc_only={bwd_alloc_only:.3} ms | alloc+touch_every_element={bwd_alloc_touch:.3} ms | raw_bwd={m_raw_bwd:.3} ms"
    );
    println!(
        "                                 scatter work above the dense-buffer floor = {:.3} ms ({:.0}% of raw_bwd); PyTorch's WHOLE op is 0.660 ms\n",
        m_raw_bwd - bwd_alloc_touch,
        100.0 * (m_raw_bwd - bwd_alloc_touch) / m_raw_bwd
    );

    println!("lane          kernel-vs-tape attribution (ms)");
    report("max_pool3d", m_raw_fwd, m_raw_bwd, m_session, 0.660);
    report("avg_pool2d", a_raw_fwd, a_raw_bwd, a_session, 1.833);
    println!(
        "\nNOTE the session arm includes building the leaf and summing the returned gradient, which\n\
         the raw arms do not. A large tape_overhead therefore points at the autograd path AND that\n\
         surrounding work together — split further before choosing a lever."
    );
}

//! frankentorch-3i7c0 STEP 0 gate — does large-buffer allocation churn survive in
//! a REALISTIC training loop, or is it an artifact of the gauntlet's per-iteration
//! input rebuild?
//!
//! Every number motivating in-session buffer pooling comes from
//! `pytorch_gauntlet_bench`, whose lanes rebuild their input tensor inside
//! `b.iter` on every iteration. Real training does not do that: it allocates
//! parameters and an input batch once and reuses them across steps. So the
//! synthetic rebuild may be manufacturing the entire effect. This probe measures
//! the same op under both harness shapes and reports **per-step large-buffer
//! alloc/free traffic** directly, rather than inferring it from a timing delta.
//!
//! The allocator is a process-global, compile-time choice, so the mimalloc arm is
//! a second build of this same file. Run both and compare:
//!
//! ```text
//! cargo run --release -p ft-api --example training_loop_alloc_profile
//! cargo run --release -p ft-api --features fair-alloc --example training_loop_alloc_profile
//! ```
//!
//! Decision rule (from the bead):
//!   PASS — large-buffer traffic persists across steps in the reuse lanes AND the
//!          allocator swap moves those lanes measurably. Pooling is justified.
//!   FAIL — traffic collapses once inputs are reused. Pooling is unjustified;
//!          record the negative evidence and stop.

// A `GlobalAlloc` is an unsafe trait. This is a measurement-only example that
// counts allocations and delegates every operation to the real allocator.
#![allow(unsafe_code)]

use std::alloc::{GlobalAlloc, Layout};
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Instant;

use ft_api::FrankenTorchSession;
use ft_core::ExecutionMode;

// ── counting allocator ──────────────────────────────────────────────────────
// Wraps whichever allocator this build selected and tallies only the LARGE
// blocks — the ones served by mmap/munmap under glibc, whose page faults are the
// entire mechanism under investigation. Small allocations are delegated
// untallied so the counters answer exactly the bead's question.

/// Blocks at or above this size are the mmap-served class this bead is about.
const LARGE: usize = 1 << 20; // 1 MiB

static LARGE_ALLOCS: AtomicU64 = AtomicU64::new(0);
static LARGE_ALLOC_BYTES: AtomicU64 = AtomicU64::new(0);
static LARGE_FREES: AtomicU64 = AtomicU64::new(0);
static LARGE_FREE_BYTES: AtomicU64 = AtomicU64::new(0);

#[cfg(feature = "fair-alloc")]
static INNER: mimalloc::MiMalloc = mimalloc::MiMalloc;
#[cfg(not(feature = "fair-alloc"))]
static INNER: std::alloc::System = std::alloc::System;

const ALLOCATOR_NAME: &str = if cfg!(feature = "fair-alloc") {
    "mimalloc (--features fair-alloc)"
} else {
    "system (default build)"
};

struct Counting;

#[inline]
fn note_alloc(size: usize) {
    if size >= LARGE {
        LARGE_ALLOCS.fetch_add(1, Ordering::Relaxed);
        LARGE_ALLOC_BYTES.fetch_add(size as u64, Ordering::Relaxed);
    }
}

#[inline]
fn note_free(size: usize) {
    if size >= LARGE {
        LARGE_FREES.fetch_add(1, Ordering::Relaxed);
        LARGE_FREE_BYTES.fetch_add(size as u64, Ordering::Relaxed);
    }
}

// SAFETY: every method forwards to `INNER`, a valid `GlobalAlloc`, with the
// layout and pointer it was given. The counters are plain atomics and never
// allocate, so there is no re-entrancy into this allocator.
unsafe impl GlobalAlloc for Counting {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        note_alloc(layout.size());
        unsafe { INNER.alloc(layout) }
    }

    unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
        note_alloc(layout.size());
        unsafe { INNER.alloc_zeroed(layout) }
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        note_free(layout.size());
        unsafe { INNER.dealloc(ptr, layout) }
    }

    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        note_free(layout.size());
        note_alloc(new_size);
        unsafe { INNER.realloc(ptr, layout, new_size) }
    }
}

#[global_allocator]
static GLOBAL: Counting = Counting;

/// Large-buffer traffic observed over one step.
#[derive(Clone, Copy)]
struct Traffic {
    allocs: u64,
    alloc_bytes: u64,
    frees: u64,
    free_bytes: u64,
}

fn snapshot() -> Traffic {
    Traffic {
        allocs: LARGE_ALLOCS.load(Ordering::Relaxed),
        alloc_bytes: LARGE_ALLOC_BYTES.load(Ordering::Relaxed),
        frees: LARGE_FREES.load(Ordering::Relaxed),
        free_bytes: LARGE_FREE_BYTES.load(Ordering::Relaxed),
    }
}

fn delta(before: Traffic, after: Traffic) -> Traffic {
    Traffic {
        allocs: after.allocs - before.allocs,
        alloc_bytes: after.alloc_bytes - before.alloc_bytes,
        frees: after.frees - before.frees,
        free_bytes: after.free_bytes - before.free_bytes,
    }
}

fn median(mut values: Vec<f64>) -> f64 {
    values.sort_by(f64::total_cmp);
    values[values.len() / 2]
}

fn median_u64(mut values: Vec<u64>) -> u64 {
    values.sort_unstable();
    values[values.len() / 2]
}

// ── workload shapes ─────────────────────────────────────────────────────────
// avg_pool1d [8,64,8192] f64 is the exact lane the pooling hypothesis came from:
// a 32 MiB input, cheap compute, so per-iteration mmap churn is a large share of
// the number. The MLP lane is a second, differently-shaped realistic workload so
// the verdict does not rest on one op.
const POOL_N: usize = 8;
const POOL_C: usize = 64;
const POOL_L: usize = 8192;

const MLP_BATCH: usize = 1024;
const MLP_DIM: usize = 1024;

fn pool_input() -> Vec<f64> {
    (0..POOL_N * POOL_C * POOL_L)
        .map(|i| ((i % 251) as f64) * 0.001 - 0.12)
        .collect()
}

fn seq(n: usize, shift: f64) -> Vec<f64> {
    (0..n)
        .map(|i| (((i as f64) * 0.017 + shift).sin()) * 0.02)
        .collect()
}

/// One measured step plus the large-buffer traffic it caused.
struct Step {
    ms: f64,
    traffic: Traffic,
    checksum: f64,
}

/// HARNESS-SHAPED lane: rebuild the input and the session every step, exactly as
/// `pytorch_gauntlet_bench` does inside `b.iter`. This is the arm that produced
/// the numbers motivating the pooling bead.
fn pool_rebuild_step(base: &[f64]) -> Step {
    let before = snapshot();
    let started = Instant::now();
    let owned = base.to_vec();
    let mut session = FrankenTorchSession::new(ExecutionMode::Strict);
    let x = session
        .tensor_variable(owned, vec![POOL_N, POOL_C, POOL_L], true)
        .expect("leaf");
    let pooled = session.functional_avg_pool1d(x, 2, 2).expect("avg_pool1d");
    let loss = session.tensor_sum(pooled).expect("sum");
    let report = session.tensor_backward(loss).expect("backward");
    let checksum = report.gradient(x).expect("grad").iter().sum::<f64>();
    let ms = started.elapsed().as_secs_f64() * 1e3;
    Step {
        ms,
        traffic: delta(before, snapshot()),
        checksum,
    }
}

/// REALISTIC lane: the input tensor is built once and reused; each step frees its
/// own graph generation via `truncate_autograd_graph` and leaves the leaf intact.
/// This is what a training loop actually does with a fixed batch.
fn pool_reuse_step(
    session: &mut FrankenTorchSession,
    x: ft_autograd::TensorNodeId,
    boundary: usize,
) -> Step {
    let before = snapshot();
    let started = Instant::now();
    let pooled = session.functional_avg_pool1d(x, 2, 2).expect("avg_pool1d");
    let loss = session.tensor_sum(pooled).expect("sum");
    let report = session.tensor_backward(loss).expect("backward");
    let checksum = report.gradient(x).expect("grad").iter().sum::<f64>();
    // Clear the leaf's accumulated gradient so every step performs identical
    // work; otherwise later steps accumulate into an already-populated buffer.
    session.zero_grads_tensor(&[x]).expect("zero_grad");
    // Freeing this step's graph generation is PART of the step: it is where the
    // step's transient buffers are returned. Doing it outside the measured
    // window would attribute every free to the gap between windows and report a
    // free count of zero.
    drop(report);
    session.truncate_autograd_graph(boundary);
    let ms = started.elapsed().as_secs_f64() * 1e3;
    Step {
        ms,
        traffic: delta(before, snapshot()),
        checksum,
    }
}

/// REALISTIC lane 2: a parameterised step — weights and batch allocated once,
/// gradients applied in place by an SGD update, graph generation freed each step.
/// Large intermediates (activations and their gradients) are genuinely per-step,
/// so this is the workload that decides whether pooling has anything to pool.
fn mlp_reuse_step(
    session: &mut FrankenTorchSession,
    batch: ft_autograd::TensorNodeId,
    weight: ft_autograd::TensorNodeId,
    bias: ft_autograd::TensorNodeId,
    learning_rate: f64,
    boundary: usize,
) -> Step {
    let before = snapshot();
    let started = Instant::now();
    let projected = session
        .tensor_linear(batch, weight, Some(bias))
        .expect("linear");
    let activated = session.tensor_relu(projected).expect("relu");
    let loss = session.tensor_sum(activated).expect("sum");
    let report = session.tensor_backward(loss).expect("backward");
    let checksum = report.gradient(weight).expect("grad").iter().sum::<f64>();
    for &parameter in &[weight, bias] {
        session
            .tensor_update_param_values_f64_with_accumulated_gradient(
                parameter,
                |gradient, values| {
                    for (value, &g) in values.iter_mut().zip(gradient) {
                        *value -= learning_rate * g;
                    }
                },
            )
            .expect("in-place SGD update");
    }
    session
        .zero_grads_tensor(&[weight, bias])
        .expect("zero_grad");
    // See `pool_reuse_step`: the generation free belongs inside the window.
    drop(report);
    session.truncate_autograd_graph(boundary);
    let ms = started.elapsed().as_secs_f64() * 1e3;
    Step {
        ms,
        traffic: delta(before, snapshot()),
        checksum,
    }
}

fn report_lane(name: &str, steps: &[Step], warmup: usize) {
    let steady = &steps[warmup..];
    let first = steps[0].traffic;
    let allocs = median_u64(steady.iter().map(|s| s.traffic.allocs).collect());
    let alloc_mib = median_u64(steady.iter().map(|s| s.traffic.alloc_bytes).collect()) as f64
        / (1024.0 * 1024.0);
    let frees = median_u64(steady.iter().map(|s| s.traffic.frees).collect());
    let free_mib = median_u64(steady.iter().map(|s| s.traffic.free_bytes).collect()) as f64
        / (1024.0 * 1024.0);
    let ms = median(steady.iter().map(|s| s.ms).collect());
    println!(
        "  {name:<18} step {ms:8.3} ms | steady-state large blocks/step: {allocs:4} alloc ({alloc_mib:8.2} MiB), {frees:4} free ({free_mib:8.2} MiB) | step0 alloc {} ({:.2} MiB)",
        first.allocs,
        first.alloc_bytes as f64 / (1024.0 * 1024.0),
    );
}

fn main() {
    let steps: usize = std::env::var("STEPS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(24);
    let warmup = (steps / 4).max(2);

    println!("frankentorch-3i7c0 STEP 0 — large-buffer churn in a realistic training loop");
    println!("  allocator            {ALLOCATOR_NAME}");
    println!(
        "  executing_elf_sha256 {}",
        ft_api::harness_provenance::executing_elf_sha256()
    );
    println!("  large-block threshold {LARGE} bytes | steps {steps} (first {warmup} discarded)");
    println!(
        "  shapes               avg_pool1d [{POOL_N},{POOL_C},{POOL_L}] f64 ({:.0} MiB input); mlp [{MLP_BATCH},{MLP_DIM}] x [{MLP_DIM},{MLP_DIM}] f64",
        (POOL_N * POOL_C * POOL_L * 8) as f64 / (1024.0 * 1024.0)
    );

    // Lane 1 — harness-shaped rebuild, the arm the pooling claim came from.
    let base = pool_input();
    let mut rebuild = Vec::with_capacity(steps);
    for _ in 0..steps {
        rebuild.push(pool_rebuild_step(&base));
    }

    // Lane 2 — same op, input built once and reused across steps.
    let mut reuse_session = FrankenTorchSession::new(ExecutionMode::Strict);
    let reused_x = reuse_session
        .tensor_variable(pool_input(), vec![POOL_N, POOL_C, POOL_L], true)
        .expect("leaf");
    let reuse_boundary = reuse_session.autograd_graph_node_count();
    let mut reuse = Vec::with_capacity(steps);
    for _ in 0..steps {
        reuse.push(pool_reuse_step(
            &mut reuse_session,
            reused_x,
            reuse_boundary,
        ));
    }

    // Lane 3 — parameterised training step, weights and batch allocated once.
    let mut mlp_session = FrankenTorchSession::new(ExecutionMode::Strict);
    let batch = mlp_session
        .tensor_variable(
            seq(MLP_BATCH * MLP_DIM, 0.0),
            vec![MLP_BATCH, MLP_DIM],
            false,
        )
        .expect("batch");
    let weight = mlp_session
        .tensor_variable(seq(MLP_DIM * MLP_DIM, 1.0), vec![MLP_DIM, MLP_DIM], true)
        .expect("weight");
    let bias = mlp_session
        .tensor_variable(seq(MLP_DIM, 2.0), vec![MLP_DIM], true)
        .expect("bias");
    let mlp_boundary = mlp_session.autograd_graph_node_count();
    let mut mlp = Vec::with_capacity(steps);
    for _ in 0..steps {
        mlp.push(mlp_reuse_step(
            &mut mlp_session,
            batch,
            weight,
            bias,
            1e-6,
            mlp_boundary,
        ));
    }

    // The lanes must actually be computing something, and the two avg_pool1d
    // lanes must agree — otherwise the comparison is between different work.
    assert!(
        rebuild[0].checksum.is_finite() && mlp[0].checksum.is_finite(),
        "workloads must produce finite gradients"
    );
    assert_eq!(
        rebuild[0].checksum.to_bits(),
        reuse[0].checksum.to_bits(),
        "rebuild and reuse lanes must compute the identical avg_pool1d gradient"
    );

    println!("\nper-step large-buffer traffic (median over steady-state steps):");
    report_lane("pool_rebuild", &rebuild, warmup);
    report_lane("pool_reuse", &reuse, warmup);
    report_lane("mlp_reuse", &mlp, warmup);

    let rebuild_mib = median_u64(
        rebuild[warmup..]
            .iter()
            .map(|s| s.traffic.alloc_bytes)
            .collect(),
    ) as f64
        / (1024.0 * 1024.0);
    let reuse_mib = median_u64(
        reuse[warmup..]
            .iter()
            .map(|s| s.traffic.alloc_bytes)
            .collect(),
    ) as f64
        / (1024.0 * 1024.0);
    let mlp_mib = median_u64(
        mlp[warmup..]
            .iter()
            .map(|s| s.traffic.alloc_bytes)
            .collect(),
    ) as f64
        / (1024.0 * 1024.0);
    println!(
        "\nGATE READING: rebuilding the input allocates {rebuild_mib:.2} MiB of large blocks per step;\n\
         reusing it allocates {reuse_mib:.2} MiB; the parameterised step allocates {mlp_mib:.2} MiB.\n\
         Churn SURVIVES input reuse only if the reuse lanes stay materially above zero."
    );
}

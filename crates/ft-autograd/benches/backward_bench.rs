use criterion::{BatchSize, Criterion, black_box, criterion_group, criterion_main};
use ft_autograd::{TensorNodeId, TensorTape};
use ft_core::ExecutionMode;

// Backward over a deep elementwise chain. Each Mul/Div node's backward currently
// clones both operand value buffers (contiguous_values_as_f64 -> Vec) and
// materialises two contribution Vecs (.collect()), so a depth-D chain over
// size-M tensors allocates ~4*D buffers of M f64 per backward. This bench
// isolates that allocation/traffic cost so the zero-alloc f64 binary-op backward
// lever can be measured same-worker.

fn build_mul_chain(depth: usize, size: usize) -> (TensorTape, TensorNodeId) {
    let mut tape = TensorTape::new();
    let x = tape
        .leaf(
            (0..size).map(|i| 1.0 + ((i % 7) as f64) * 0.1).collect(),
            vec![size],
            true,
        )
        .expect("leaf x");
    let mut cur = x;
    for d in 0..depth {
        let w = tape
            .leaf(
                (0..size)
                    .map(|i| 1.0 + (((i + d) % 5) as f64) * 0.01)
                    .collect(),
                vec![size],
                true,
            )
            .expect("leaf w");
        cur = tape.mul(cur, w, ExecutionMode::Strict).expect("mul").0;
    }
    (tape, cur)
}

fn build_div_chain(depth: usize, size: usize) -> (TensorTape, TensorNodeId) {
    let mut tape = TensorTape::new();
    let x = tape
        .leaf(
            (0..size).map(|i| 2.0 + ((i % 7) as f64) * 0.1).collect(),
            vec![size],
            true,
        )
        .expect("leaf x");
    let mut cur = x;
    for d in 0..depth {
        let w = tape
            .leaf(
                (0..size)
                    .map(|i| 1.5 + (((i + d) % 5) as f64) * 0.01)
                    .collect(),
                vec![size],
                true,
            )
            .expect("leaf w");
        cur = tape.div(cur, w, ExecutionMode::Strict).expect("div").0;
    }
    (tape, cur)
}

fn build_add_chain(depth: usize, size: usize) -> (TensorTape, TensorNodeId) {
    let mut tape = TensorTape::new();
    let x = tape
        .leaf(
            (0..size).map(|i| 1.0 + ((i % 7) as f64) * 0.1).collect(),
            vec![size],
            true,
        )
        .expect("leaf x");
    let mut cur = x;
    for d in 0..depth {
        let w = tape
            .leaf(
                (0..size)
                    .map(|i| 1.0 + (((i + d) % 5) as f64) * 0.01)
                    .collect(),
                vec![size],
                true,
            )
            .expect("leaf w");
        cur = tape.add(cur, w, ExecutionMode::Strict).expect("add").0;
    }
    (tape, cur)
}

#[derive(Clone, Copy)]
enum Activation {
    Gelu,
    Silu,
    Elu,
    Erf,
    Erfc,
    Softplus,
    Mish,
}

fn apply_activation(tape: &mut TensorTape, op: Activation, input: TensorNodeId) -> TensorNodeId {
    match op {
        Activation::Gelu => tape.gelu(input, ExecutionMode::Strict).expect("gelu").0,
        Activation::Silu => tape.silu(input, ExecutionMode::Strict).expect("silu").0,
        Activation::Elu => tape.elu(input, ExecutionMode::Strict).expect("elu").0,
        Activation::Erf => tape.erf(input, ExecutionMode::Strict).expect("erf").0,
        Activation::Erfc => tape.erfc(input, ExecutionMode::Strict).expect("erfc").0,
        Activation::Softplus => {
            tape.softplus(input, ExecutionMode::Strict)
                .expect("softplus")
                .0
        }
        Activation::Mish => tape.mish(input, ExecutionMode::Strict).expect("mish").0,
    }
}

fn build_activation_chain(op: Activation, depth: usize, size: usize) -> (TensorTape, TensorNodeId) {
    let mut tape = TensorTape::new();
    let x = tape
        .leaf(
            (0..size)
                .map(|i| -3.0 + ((i % 1024) as f64) * (6.0 / 1023.0))
                .collect(),
            vec![size],
            true,
        )
        .expect("leaf x");
    let mut cur = x;
    for _ in 0..depth {
        cur = apply_activation(&mut tape, op, cur);
    }
    (tape, cur)
}

fn bench_backward(c: &mut Criterion) {
    let mut group = c.benchmark_group("autograd_backward");
    let depth = 256;
    let size = 2048;

    // ANCHOR: Add backward is NOT touched by the zero-alloc lever (it already
    // accumulates `incoming` directly with no contrib Vec / operand clone), so its
    // time tracks pure worker speed. Normalise mul/div by the add ratio across
    // baseline-vs-after runs to cancel shared-worker variance.
    group.bench_function("add_chain_256x2048_anchor", |b| {
        b.iter_batched(
            || build_add_chain(depth, size),
            |(mut tape, root)| {
                let report = tape.backward(root).expect("backward");
                black_box(report.gradient(TensorNodeId(0)).map(|g| g[0]))
            },
            BatchSize::SmallInput,
        );
    });

    group.bench_function("mul_chain_256x2048", |b| {
        b.iter_batched(
            || build_mul_chain(depth, size),
            |(mut tape, root)| {
                let report = tape.backward(root).expect("backward");
                black_box(report.gradient(TensorNodeId(0)).map(|g| g[0]))
            },
            BatchSize::SmallInput,
        );
    });

    group.bench_function("div_chain_256x2048", |b| {
        b.iter_batched(
            || build_div_chain(depth, size),
            |(mut tape, root)| {
                let report = tape.backward(root).expect("backward");
                black_box(report.gradient(TensorNodeId(0)).map(|g| g[0]))
            },
            BatchSize::SmallInput,
        );
    });

    group.finish();
}

fn bench_activation_backward(c: &mut Criterion) {
    let mut group = c.benchmark_group("activation_backward");
    let depth = 16;
    let size = 65536;

    for (name, op) in [
        ("gelu_chain_16x65536", Activation::Gelu),
        ("silu_chain_16x65536", Activation::Silu),
        ("elu_chain_16x65536", Activation::Elu),
        ("erf_chain_16x65536", Activation::Erf),
        ("erfc_chain_16x65536", Activation::Erfc),
        ("softplus_chain_16x65536", Activation::Softplus),
        ("mish_chain_16x65536", Activation::Mish),
    ] {
        group.bench_function(name, |b| {
            b.iter_batched(
                || build_activation_chain(op, depth, size),
                |(mut tape, root)| {
                    let report = tape.backward(root).expect("backward");
                    black_box(report.gradient(TensorNodeId(0)).map(|g| g[0]))
                },
                BatchSize::SmallInput,
            );
        });
    }

    group.finish();
}

criterion_group!(benches, bench_backward, bench_activation_backward);
criterion_main!(benches);

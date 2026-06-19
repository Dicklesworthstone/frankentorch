//! Targeted PyTorch-vs-FrankenTorch gauntlet benches.
//!
//! Run with an interpreter that has CPU PyTorch installed:
//!   PYTORCH_PYTHON=/data/projects/.venvs/frankentorch-pytorch-cpu/bin/python \
//!   CARGO_TARGET_DIR=/data/projects/.rch-targets/frankentorch-cod-b \
//!   cargo bench -p ft-api --bench pytorch_gauntlet_bench -- max_pool1d
//!   cargo bench -p ft-api --bench pytorch_gauntlet_bench -- max_pool3d
//!   cargo bench -p ft-api --bench pytorch_gauntlet_bench -- linear

use std::path::PathBuf;
use std::process::{Command, exit};
use std::time::Duration;

use criterion::{Criterion, black_box, criterion_group, criterion_main};
use ft_api::FrankenTorchSession;
use ft_core::ExecutionMode;

const MAX_POOL1D_N: usize = 8;
const MAX_POOL1D_C: usize = 64;
const MAX_POOL1D_L: usize = 8192;
const MAX_POOL1D_TOTAL: usize = MAX_POOL1D_N * MAX_POOL1D_C * MAX_POOL1D_L;

const MAX_POOL3D_N: usize = 2;
const MAX_POOL3D_C: usize = 32;
const MAX_POOL3D_D: usize = 16;
const MAX_POOL3D_H: usize = 32;
const MAX_POOL3D_W: usize = 32;
const MAX_POOL3D_TOTAL: usize =
    MAX_POOL3D_N * MAX_POOL3D_C * MAX_POOL3D_D * MAX_POOL3D_H * MAX_POOL3D_W;

const LINEAR_BATCH: usize = 32;
const LINEAR_IN_FEATURES: usize = 512;
const LINEAR_HIDDEN: usize = 2048;

fn deterministic_pool1d_values() -> Vec<f64> {
    (0..MAX_POOL1D_TOTAL)
        .map(|idx| (idx % 251) as f64 * 0.001 - 0.12)
        .collect()
}

fn deterministic_pool3d_values() -> Vec<f64> {
    (0..MAX_POOL3D_TOTAL)
        .map(|idx| (idx % 251) as f64 * 0.001 - 0.12)
        .collect()
}

fn deterministic_values(n: usize, shift: f64) -> Vec<f64> {
    (0..n)
        .map(|i| (((i as f64) * 0.017 + shift).sin()) * 0.2)
        .collect()
}

fn pytorch_python() -> String {
    std::env::var("PYTORCH_PYTHON").unwrap_or_else(|_| "python3".to_string())
}

fn pytorch_max_pool1d_script() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("benches/pytorch_max_pool1d_grad.py")
}

fn pytorch_max_pool3d_script() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("benches/pytorch_max_pool3d_grad.py")
}

fn pytorch_linear_train_script() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("benches/pytorch_linear_train.py")
}

fn fail(message: String) -> ! {
    eprintln!("{message}");
    exit(1);
}

fn require<T, E: std::fmt::Debug>(result: Result<T, E>, context: &str) -> T {
    match result {
        Ok(value) => value,
        Err(err) => fail(format!("{context}: {err:?}")),
    }
}

fn run_pytorch_max_pool1d_grad(iterations: u64) -> Duration {
    let output = match Command::new(pytorch_python())
        .arg(pytorch_max_pool1d_script())
        .env("FT_GAUNTLET_ITERS", iterations.to_string())
        .output()
    {
        Ok(output) => output,
        Err(err) => fail(format!("failed to launch PyTorch benchmark: {err:?}")),
    };

    parse_pytorch_elapsed(output, "PyTorch max_pool1d")
}

fn run_pytorch_max_pool3d_grad(iterations: u64) -> Duration {
    let output = match Command::new(pytorch_python())
        .arg(pytorch_max_pool3d_script())
        .env("FT_GAUNTLET_ITERS", iterations.to_string())
        .output()
    {
        Ok(output) => output,
        Err(err) => fail(format!("failed to launch PyTorch benchmark: {err:?}")),
    };

    parse_pytorch_elapsed(output, "PyTorch max_pool3d")
}

fn run_pytorch_linear_train(iterations: u64) -> Duration {
    let output = match Command::new(pytorch_python())
        .arg(pytorch_linear_train_script())
        .env("FT_GAUNTLET_ITERS", iterations.to_string())
        .output()
    {
        Ok(output) => output,
        Err(err) => fail(format!("failed to launch PyTorch benchmark: {err:?}")),
    };

    parse_pytorch_elapsed(output, "PyTorch linear")
}

fn parse_pytorch_elapsed(output: std::process::Output, label: &str) -> Duration {
    if !output.status.success() {
        fail(format!(
            "{label} benchmark failed with status {:?}\nstdout:\n{}\nstderr:\n{}",
            output.status.code(),
            String::from_utf8_lossy(&output.stdout),
            String::from_utf8_lossy(&output.stderr)
        ));
    }

    let stdout = require(
        String::from_utf8(output.stdout),
        "PyTorch benchmark emitted non-UTF8 stdout",
    );
    let seconds: f64 = require(
        stdout.trim().parse(),
        &format!("failed to parse {label} elapsed seconds `{stdout}`"),
    );
    Duration::from_secs_f64(seconds)
}

fn bench_max_pool1d_unit_dout(c: &mut Criterion) {
    let mut group = c.benchmark_group("gauntlet_max_pool1d_grad");
    group.warm_up_time(Duration::from_secs(1));
    group.measurement_time(Duration::from_secs(3));
    group.sample_size(10);

    let values = deterministic_pool1d_values();
    let shape = vec![MAX_POOL1D_N, MAX_POOL1D_C, MAX_POOL1D_L];

    group.bench_function("frankentorch_kgs4_126", |b| {
        b.iter(|| {
            let mut session = FrankenTorchSession::new(ExecutionMode::Strict);
            let x = require(
                session.tensor_variable(black_box(values.clone()), black_box(shape.clone()), true),
                "failed to create FrankenTorch tensor",
            );
            let out = require(
                session.functional_max_pool1d(x, 2, 2),
                "failed to run FrankenTorch max_pool1d",
            );
            let loss = require(
                session.tensor_sum(out),
                "failed to reduce FrankenTorch output",
            );
            black_box(require(
                session.tensor_backward(loss),
                "failed to run FrankenTorch backward",
            ))
        });
    });

    group.bench_function("pytorch_2_12_cpu", |b| {
        b.iter_custom(run_pytorch_max_pool1d_grad);
    });

    group.finish();
}

fn bench_max_pool3d_saved_indices(c: &mut Criterion) {
    let mut group = c.benchmark_group("gauntlet_max_pool3d_grad");
    group.warm_up_time(Duration::from_secs(1));
    group.measurement_time(Duration::from_secs(3));
    group.sample_size(10);

    let values = deterministic_pool3d_values();
    let shape = vec![
        MAX_POOL3D_N,
        MAX_POOL3D_C,
        MAX_POOL3D_D,
        MAX_POOL3D_H,
        MAX_POOL3D_W,
    ];

    group.bench_function("frankentorch_kgs4_117", |b| {
        b.iter(|| {
            let mut session = FrankenTorchSession::new(ExecutionMode::Strict);
            let x = require(
                session.tensor_variable(black_box(values.clone()), black_box(shape.clone()), true),
                "failed to create FrankenTorch tensor",
            );
            let out = require(
                session.functional_max_pool3d(x, (2, 2, 2), (2, 2, 2)),
                "failed to run FrankenTorch max_pool3d",
            );
            let loss = require(
                session.tensor_sum(out),
                "failed to reduce FrankenTorch output",
            );
            black_box(require(
                session.tensor_backward(loss),
                "failed to run FrankenTorch backward",
            ))
        });
    });

    group.bench_function("pytorch_2_12_cpu", |b| {
        b.iter_custom(run_pytorch_max_pool3d_grad);
    });

    group.finish();
}

fn bench_linear_train_hidden_2048(c: &mut Criterion) {
    let mut group = c.benchmark_group("gauntlet_linear_train_hidden_2048");
    group.warm_up_time(Duration::from_secs(1));
    group.measurement_time(Duration::from_secs(3));
    group.sample_size(10);

    let x_values = deterministic_values(LINEAR_BATCH * LINEAR_IN_FEATURES, 0.0);
    let w_values = deterministic_values(LINEAR_HIDDEN * LINEAR_IN_FEATURES, 1.0);
    let bias_values = deterministic_values(LINEAR_HIDDEN, 2.0);

    group.bench_function("frankentorch_kgs4_121", |b| {
        b.iter(|| {
            let mut session = FrankenTorchSession::new(ExecutionMode::Strict);
            let x = require(
                session.tensor_variable(
                    black_box(x_values.clone()),
                    vec![LINEAR_BATCH, LINEAR_IN_FEATURES],
                    true,
                ),
                "failed to create FrankenTorch linear input",
            );
            let w = require(
                session.tensor_variable(
                    black_box(w_values.clone()),
                    vec![LINEAR_HIDDEN, LINEAR_IN_FEATURES],
                    true,
                ),
                "failed to create FrankenTorch linear weight",
            );
            let bias = require(
                session.tensor_variable(black_box(bias_values.clone()), vec![LINEAR_HIDDEN], true),
                "failed to create FrankenTorch linear bias",
            );
            let y = require(
                session.tensor_linear(x, w, Some(bias)),
                "failed to run FrankenTorch linear",
            );
            let loss = require(
                session.tensor_sum(y),
                "failed to reduce FrankenTorch linear",
            );
            black_box(require(
                session.tensor_backward(loss),
                "failed to run FrankenTorch linear backward",
            ))
        });
    });

    group.bench_function("pytorch_2_12_cpu", |b| {
        b.iter_custom(run_pytorch_linear_train);
    });

    group.finish();
}

criterion_group!(
    benches,
    bench_max_pool1d_unit_dout,
    bench_max_pool3d_saved_indices,
    bench_linear_train_hidden_2048
);
criterion_main!(benches);

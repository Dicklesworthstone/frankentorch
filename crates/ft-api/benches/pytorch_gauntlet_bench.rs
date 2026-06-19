//! Targeted PyTorch-vs-FrankenTorch gauntlet benches.
//!
//! Run with an interpreter that has CPU PyTorch installed:
//!   PYTORCH_PYTHON=/data/projects/.venvs/frankentorch-pytorch-cpu/bin/python \
//!   CARGO_TARGET_DIR=/data/projects/.rch-targets/frankentorch-cod-b \
//!   cargo bench -p ft-api --bench pytorch_gauntlet_bench -- max_pool1d

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

fn deterministic_pool_values() -> Vec<f64> {
    (0..MAX_POOL1D_TOTAL)
        .map(|idx| (idx % 251) as f64 * 0.001 - 0.12)
        .collect()
}

fn pytorch_python() -> String {
    std::env::var("PYTORCH_PYTHON").unwrap_or_else(|_| "python3".to_string())
}

fn pytorch_script() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("benches/pytorch_max_pool1d_grad.py")
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
        .arg(pytorch_script())
        .env("FT_GAUNTLET_ITERS", iterations.to_string())
        .output()
    {
        Ok(output) => output,
        Err(err) => fail(format!("failed to launch PyTorch benchmark: {err:?}")),
    };

    if !output.status.success() {
        fail(format!(
            "PyTorch benchmark failed with status {:?}\nstdout:\n{}\nstderr:\n{}",
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
        &format!("failed to parse PyTorch elapsed seconds `{stdout}`"),
    );
    Duration::from_secs_f64(seconds)
}

fn bench_max_pool1d_unit_dout(c: &mut Criterion) {
    let mut group = c.benchmark_group("gauntlet_max_pool1d_grad");
    group.warm_up_time(Duration::from_secs(1));
    group.measurement_time(Duration::from_secs(3));
    group.sample_size(10);

    let values = deterministic_pool_values();
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

criterion_group!(benches, bench_max_pool1d_unit_dout);
criterion_main!(benches);

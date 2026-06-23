//! Targeted PyTorch-vs-FrankenTorch gauntlet benches.
//!
//! Run with an interpreter that has CPU PyTorch installed:
//!   PYTORCH_PYTHON=/data/projects/.venvs/frankentorch-pytorch-cpu/bin/python \
//!   CARGO_TARGET_DIR=/data/projects/.rch-targets/frankentorch-cod-b \
//!   cargo bench -p ft-api --bench pytorch_gauntlet_bench -- avg_pool1d
//!   cargo bench -p ft-api --bench pytorch_gauntlet_bench -- avg_pool2d
//!   cargo bench -p ft-api --bench pytorch_gauntlet_bench -- batch_norm2d_f32
//!   cargo bench -p ft-api --bench pytorch_gauntlet_bench -- conv3d
//!   cargo bench -p ft-api --bench pytorch_gauntlet_bench -- max_pool1d
//!   cargo bench -p ft-api --bench pytorch_gauntlet_bench -- max_pool3d
//!   cargo bench -p ft-api --bench pytorch_gauntlet_bench -- linear
//!   cargo bench -p ft-api --bench pytorch_gauntlet_bench -- sdpa
//!
//! Fair allocator mode for allocator-sensitive FT/PyTorch comparisons:
//!   cargo bench -p ft-api --features fair-alloc --bench pytorch_gauntlet_bench -- avg_pool1d

#[cfg(feature = "fair-alloc")]
#[global_allocator]
static FAIR_ALLOC: mimalloc::MiMalloc = mimalloc::MiMalloc;

use std::path::PathBuf;
use std::process::{Command, exit};
use std::time::Duration;

use criterion::{BatchSize, Criterion, black_box, criterion_group, criterion_main};
use ft_api::FrankenTorchSession;
use ft_core::ExecutionMode;

const MAX_POOL1D_N: usize = 8;
const MAX_POOL1D_C: usize = 64;
const MAX_POOL1D_L: usize = 8192;
const MAX_POOL1D_TOTAL: usize = MAX_POOL1D_N * MAX_POOL1D_C * MAX_POOL1D_L;

const AVG_POOL2D_N: usize = 8;
const AVG_POOL2D_C: usize = 64;
const AVG_POOL2D_H: usize = 64;
const AVG_POOL2D_W: usize = 64;
const AVG_POOL2D_TOTAL: usize = AVG_POOL2D_N * AVG_POOL2D_C * AVG_POOL2D_H * AVG_POOL2D_W;

const BATCH_NORM2D_N: usize = 32;
const BATCH_NORM2D_C: usize = 256;
const BATCH_NORM2D_H: usize = 28;
const BATCH_NORM2D_W: usize = 28;
const BATCH_NORM2D_TOTAL: usize = BATCH_NORM2D_N * BATCH_NORM2D_C * BATCH_NORM2D_H * BATCH_NORM2D_W;

const CONV3D_N: usize = 2;
const CONV3D_C_IN: usize = 32;
const CONV3D_C_OUT: usize = 32;
const CONV3D_D: usize = 8;
const CONV3D_H: usize = 16;
const CONV3D_W: usize = 16;
const CONV3D_K: usize = 3;
const CONV3D_INPUT_TOTAL: usize = CONV3D_N * CONV3D_C_IN * CONV3D_D * CONV3D_H * CONV3D_W;
const CONV3D_WEIGHT_TOTAL: usize = CONV3D_C_OUT * CONV3D_C_IN * CONV3D_K * CONV3D_K * CONV3D_K;

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

const SDPA_BATCH: usize = 2;
const SDPA_HEADS: usize = 8;
const SDPA_SEQ: usize = 512;
const SDPA_D: usize = 64;
const SDPA_TOTAL: usize = SDPA_BATCH * SDPA_HEADS * SDPA_SEQ * SDPA_D;

fn deterministic_pool1d_values() -> Vec<f64> {
    (0..MAX_POOL1D_TOTAL)
        .map(|idx| (idx % 251) as f64 * 0.001 - 0.12)
        .collect()
}

fn deterministic_pool2d_values() -> Vec<f64> {
    (0..AVG_POOL2D_TOTAL)
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

fn deterministic_values_f32(n: usize, shift: f32) -> Vec<f32> {
    (0..n)
        .map(|i| (((i as f32) * 0.017 + shift).sin()) * 0.2)
        .collect()
}

fn pytorch_python() -> String {
    std::env::var("PYTORCH_PYTHON").unwrap_or_else(|_| "python3".to_string())
}

fn pytorch_max_pool1d_script() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("benches/pytorch_max_pool1d_grad.py")
}

fn pytorch_avg_pool1d_script() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("benches/pytorch_avg_pool1d_grad.py")
}

fn pytorch_avg_pool2d_script() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("benches/pytorch_avg_pool2d_grad.py")
}

fn pytorch_batch_norm2d_f32_script() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("benches/pytorch_batch_norm2d_f32_grad.py")
}

fn pytorch_conv3d_script() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("benches/pytorch_conv3d_grad.py")
}

fn pytorch_max_pool3d_script() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("benches/pytorch_max_pool3d_grad.py")
}

fn pytorch_linear_train_script() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("benches/pytorch_linear_train.py")
}

fn pytorch_sdpa_grad_script() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("benches/pytorch_sdpa_grad.py")
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

fn run_pytorch_avg_pool1d_grad(iterations: u64) -> Duration {
    let output = match Command::new(pytorch_python())
        .arg(pytorch_avg_pool1d_script())
        .env("FT_GAUNTLET_ITERS", iterations.to_string())
        .output()
    {
        Ok(output) => output,
        Err(err) => fail(format!("failed to launch PyTorch benchmark: {err:?}")),
    };

    parse_pytorch_elapsed(output, "PyTorch avg_pool1d")
}

fn run_pytorch_avg_pool2d_grad(iterations: u64) -> Duration {
    let output = match Command::new(pytorch_python())
        .arg(pytorch_avg_pool2d_script())
        .env("FT_GAUNTLET_ITERS", iterations.to_string())
        .output()
    {
        Ok(output) => output,
        Err(err) => fail(format!("failed to launch PyTorch benchmark: {err:?}")),
    };

    parse_pytorch_elapsed(output, "PyTorch avg_pool2d")
}

fn run_pytorch_batch_norm2d_f32_grad(iterations: u64) -> Duration {
    let output = match Command::new(pytorch_python())
        .arg(pytorch_batch_norm2d_f32_script())
        .env("FT_GAUNTLET_ITERS", iterations.to_string())
        .output()
    {
        Ok(output) => output,
        Err(err) => fail(format!("failed to launch PyTorch benchmark: {err:?}")),
    };

    parse_pytorch_elapsed(output, "PyTorch batch_norm2d f32")
}

fn run_pytorch_conv3d_grad(iterations: u64) -> Duration {
    let output = match Command::new(pytorch_python())
        .arg(pytorch_conv3d_script())
        .env("FT_GAUNTLET_ITERS", iterations.to_string())
        .output()
    {
        Ok(output) => output,
        Err(err) => fail(format!("failed to launch PyTorch benchmark: {err:?}")),
    };

    parse_pytorch_elapsed(output, "PyTorch conv3d")
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

fn run_pytorch_sdpa_grad(iterations: u64) -> Duration {
    let output = match Command::new(pytorch_python())
        .arg(pytorch_sdpa_grad_script())
        .env("FT_GAUNTLET_ITERS", iterations.to_string())
        .output()
    {
        Ok(output) => output,
        Err(err) => fail(format!("failed to launch PyTorch benchmark: {err:?}")),
    };

    parse_pytorch_elapsed(output, "PyTorch SDPA")
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

fn bench_avg_pool1d_unit_dy(c: &mut Criterion) {
    let mut group = c.benchmark_group("gauntlet_avg_pool1d_grad");
    group.warm_up_time(Duration::from_secs(1));
    group.measurement_time(Duration::from_secs(3));
    group.sample_size(10);

    let values = deterministic_pool1d_values();
    let shape = vec![MAX_POOL1D_N, MAX_POOL1D_C, MAX_POOL1D_L];

    group.bench_function("frankentorch_kgs4_122", |b| {
        b.iter(|| {
            let mut session = FrankenTorchSession::new(ExecutionMode::Strict);
            let x = require(
                session.tensor_variable(black_box(values.clone()), black_box(shape.clone()), true),
                "failed to create FrankenTorch tensor",
            );
            let out = require(
                session.functional_avg_pool1d(x, 2, 2),
                "failed to run FrankenTorch avg_pool1d",
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

    group.bench_function("frankentorch_kgs4_134_fused_sum_loss", |b| {
        b.iter(|| {
            let mut session = FrankenTorchSession::new(ExecutionMode::Strict);
            let x = require(
                session.tensor_variable(black_box(values.clone()), black_box(shape.clone()), true),
                "failed to create FrankenTorch tensor",
            );
            let loss = require(
                session.functional_avg_pool1d_sum(x, 2, 2),
                "failed to run FrankenTorch fused avg_pool1d sum",
            );
            black_box(require(
                session.tensor_backward(loss),
                "failed to run FrankenTorch fused avg_pool1d backward",
            ))
        });
    });

    group.bench_function("pytorch_2_12_cpu", |b| {
        b.iter_custom(run_pytorch_avg_pool1d_grad);
    });

    group.finish();
}

fn bench_avg_pool2d_unit_dy(c: &mut Criterion) {
    let mut group = c.benchmark_group("gauntlet_avg_pool2d_grad");
    group.warm_up_time(Duration::from_secs(1));
    group.measurement_time(Duration::from_secs(3));
    group.sample_size(10);

    let values = deterministic_pool2d_values();
    let shape = vec![AVG_POOL2D_N, AVG_POOL2D_C, AVG_POOL2D_H, AVG_POOL2D_W];

    group.bench_function("frankentorch_kgs4_112", |b| {
        b.iter(|| {
            let mut session = FrankenTorchSession::new(ExecutionMode::Strict);
            let x = require(
                session.tensor_variable(black_box(values.clone()), black_box(shape.clone()), true),
                "failed to create FrankenTorch tensor",
            );
            let out = require(
                session.functional_avg_pool2d(x, (2, 2), (2, 2), (0, 0), false, true),
                "failed to run FrankenTorch avg_pool2d",
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

    group.bench_function("frankentorch_kgs4_147_scalar_sum", |b| {
        b.iter(|| {
            let mut session = FrankenTorchSession::new(ExecutionMode::Strict);
            let x = require(
                session.tensor_variable(black_box(values.clone()), black_box(shape.clone()), true),
                "failed to create FrankenTorch tensor",
            );
            let loss = require(
                session.functional_avg_pool2d_sum(x, (2, 2), (2, 2), (0, 0), false, true),
                "failed to run FrankenTorch fused avg_pool2d sum",
            );
            black_box(require(
                session.tensor_backward(loss),
                "failed to run FrankenTorch fused avg_pool2d backward",
            ))
        });
    });

    group.bench_function("pytorch_2_12_cpu", |b| {
        b.iter_custom(run_pytorch_avg_pool2d_grad);
    });

    group.finish();
}

fn bench_batch_norm2d_f32_unit_dy(c: &mut Criterion) {
    let mut group = c.benchmark_group("gauntlet_batch_norm2d_f32_grad");
    group.warm_up_time(Duration::from_secs(1));
    group.measurement_time(Duration::from_secs(3));
    group.sample_size(10);

    let x_values = deterministic_values_f32(BATCH_NORM2D_TOTAL, 0.0);
    let running_mean = vec![0.0f32; BATCH_NORM2D_C];
    let running_var = vec![1.0f32; BATCH_NORM2D_C];
    let weight_values = deterministic_values_f32(BATCH_NORM2D_C, 1.0);
    let bias_values = deterministic_values_f32(BATCH_NORM2D_C, 2.0);
    let shape = vec![
        BATCH_NORM2D_N,
        BATCH_NORM2D_C,
        BATCH_NORM2D_H,
        BATCH_NORM2D_W,
    ];

    group.bench_function("frankentorch_kgs4_114", |b| {
        b.iter(|| {
            let mut session = FrankenTorchSession::new(ExecutionMode::Strict);
            let x = require(
                session.tensor_variable_f32(black_box(x_values.clone()), shape.clone(), true),
                "failed to create FrankenTorch BatchNorm input",
            );
            let rm = require(
                session.tensor_variable_f32(running_mean.clone(), vec![BATCH_NORM2D_C], false),
                "failed to create FrankenTorch BatchNorm running_mean",
            );
            let rv = require(
                session.tensor_variable_f32(running_var.clone(), vec![BATCH_NORM2D_C], false),
                "failed to create FrankenTorch BatchNorm running_var",
            );
            let weight = require(
                session.tensor_variable_f32(
                    black_box(weight_values.clone()),
                    vec![BATCH_NORM2D_C],
                    true,
                ),
                "failed to create FrankenTorch BatchNorm weight",
            );
            let bias = require(
                session.tensor_variable_f32(
                    black_box(bias_values.clone()),
                    vec![BATCH_NORM2D_C],
                    true,
                ),
                "failed to create FrankenTorch BatchNorm bias",
            );
            let (out, _, _) = require(
                session.functional_batch_norm2d(
                    x,
                    Some(rm),
                    Some(rv),
                    Some(weight),
                    Some(bias),
                    true,
                    0.1,
                    1e-5,
                ),
                "failed to run FrankenTorch BatchNorm2d",
            );
            let loss = require(
                session.tensor_sum(out),
                "failed to reduce FrankenTorch BatchNorm output",
            );
            black_box(require(
                session.tensor_backward(loss),
                "failed to run FrankenTorch BatchNorm backward",
            ))
        });
    });

    group.bench_function("frankentorch_kgs4_136_scalar_sum", |b| {
        b.iter(|| {
            let mut session = FrankenTorchSession::new(ExecutionMode::Strict);
            let x = require(
                session.tensor_variable_f32(black_box(x_values.clone()), shape.clone(), true),
                "failed to create FrankenTorch BatchNorm input",
            );
            let rm = require(
                session.tensor_variable_f32(running_mean.clone(), vec![BATCH_NORM2D_C], false),
                "failed to create FrankenTorch BatchNorm running_mean",
            );
            let rv = require(
                session.tensor_variable_f32(running_var.clone(), vec![BATCH_NORM2D_C], false),
                "failed to create FrankenTorch BatchNorm running_var",
            );
            let weight = require(
                session.tensor_variable_f32(
                    black_box(weight_values.clone()),
                    vec![BATCH_NORM2D_C],
                    true,
                ),
                "failed to create FrankenTorch BatchNorm weight",
            );
            let bias = require(
                session.tensor_variable_f32(
                    black_box(bias_values.clone()),
                    vec![BATCH_NORM2D_C],
                    true,
                ),
                "failed to create FrankenTorch BatchNorm bias",
            );
            let (loss, _, _) = require(
                session.functional_batch_norm2d_sum(
                    x,
                    Some(rm),
                    Some(rv),
                    Some(weight),
                    Some(bias),
                    true,
                    0.1,
                    1e-5,
                ),
                "failed to run scalar FrankenTorch BatchNorm2d",
            );
            black_box(require(
                session.tensor_backward(loss),
                "failed to run scalar FrankenTorch BatchNorm backward",
            ))
        });
    });

    group.bench_function("pytorch_2_12_cpu", |b| {
        b.iter_custom(run_pytorch_batch_norm2d_f32_grad);
    });

    group.finish();
}

fn bench_conv3d_grad(c: &mut Criterion) {
    let mut group = c.benchmark_group("gauntlet_conv3d_grad");
    group.warm_up_time(Duration::from_secs(1));
    group.measurement_time(Duration::from_secs(3));
    group.sample_size(10);

    let x_values = deterministic_values(CONV3D_INPUT_TOTAL, 0.0);
    let w_values = deterministic_values(CONV3D_WEIGHT_TOTAL, 1.0);
    let x_shape = vec![CONV3D_N, CONV3D_C_IN, CONV3D_D, CONV3D_H, CONV3D_W];
    let w_shape = vec![CONV3D_C_OUT, CONV3D_C_IN, CONV3D_K, CONV3D_K, CONV3D_K];

    group.bench_function("frankentorch_kgs4_119", |b| {
        b.iter(|| {
            let mut session = FrankenTorchSession::new(ExecutionMode::Strict);
            let x = require(
                session.tensor_variable(black_box(x_values.clone()), x_shape.clone(), true),
                "failed to create FrankenTorch Conv3d input",
            );
            let w = require(
                session.tensor_variable(black_box(w_values.clone()), w_shape.clone(), true),
                "failed to create FrankenTorch Conv3d weight",
            );
            let out = require(
                session.functional_conv3d(x, w, None, (1, 1, 1), (1, 1, 1)),
                "failed to run FrankenTorch Conv3d",
            );
            let loss = require(
                session.tensor_sum(out),
                "failed to reduce FrankenTorch Conv3d output",
            );
            black_box(require(
                session.tensor_backward(loss),
                "failed to run FrankenTorch Conv3d backward",
            ))
        });
    });

    group.bench_function("pytorch_2_12_cpu", |b| {
        b.iter_custom(run_pytorch_conv3d_grad);
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

    group.bench_function("frankentorch_fused_sum_loss", |b| {
        b.iter(|| {
            let mut session = FrankenTorchSession::new(ExecutionMode::Strict);
            let x = require(
                session.tensor_variable(black_box(values.clone()), black_box(shape.clone()), true),
                "failed to create FrankenTorch tensor",
            );
            let loss = require(
                session.functional_max_pool3d_sum(x, (2, 2, 2), (2, 2, 2)),
                "failed to run FrankenTorch fused max_pool3d sum",
            );
            black_box(require(
                session.tensor_backward(loss),
                "failed to run FrankenTorch fused backward",
            ))
        });
    });

    group.bench_function("pytorch_2_12_cpu", |b| {
        b.iter_custom(run_pytorch_max_pool3d_grad);
    });

    group.finish();
}

fn bench_max_pool3d_stage_probe(c: &mut Criterion) {
    let mut group = c.benchmark_group("gauntlet_max_pool3d_grad_stage");
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
    let out_d = MAX_POOL3D_D / 2;
    let out_h = MAX_POOL3D_H / 2;
    let out_w = MAX_POOL3D_W / 2;
    let dout = vec![1.0_f64; MAX_POOL3D_N * MAX_POOL3D_C * out_d * out_h * out_w];
    let (_, arg_offsets) = ft_kernel_cpu::max_pool3d_forward_with_indices_f64(
        &values,
        MAX_POOL3D_N,
        MAX_POOL3D_C,
        MAX_POOL3D_D,
        MAX_POOL3D_H,
        MAX_POOL3D_W,
        2,
        2,
        2,
        out_d,
        out_h,
        out_w,
        2,
        2,
        2,
    );

    group.bench_function("frankentorch_setup_tensor", |b| {
        b.iter(|| {
            let mut session = FrankenTorchSession::new(ExecutionMode::Strict);
            black_box(require(
                session.tensor_variable(black_box(values.clone()), black_box(shape.clone()), true),
                "failed to create FrankenTorch tensor",
            ))
        });
    });

    group.bench_function("frankentorch_forward_only", |b| {
        b.iter_batched(
            || {
                let mut session = FrankenTorchSession::new(ExecutionMode::Strict);
                let x = require(
                    session.tensor_variable(values.clone(), shape.clone(), true),
                    "failed to create FrankenTorch tensor",
                );
                (session, x)
            },
            |(mut session, x)| {
                black_box(require(
                    session.functional_max_pool3d(x, (2, 2, 2), (2, 2, 2)),
                    "failed to run FrankenTorch max_pool3d",
                ))
            },
            BatchSize::SmallInput,
        );
    });

    group.bench_function("frankentorch_sum_only", |b| {
        b.iter_batched(
            || {
                let mut session = FrankenTorchSession::new(ExecutionMode::Strict);
                let x = require(
                    session.tensor_variable(values.clone(), shape.clone(), true),
                    "failed to create FrankenTorch tensor",
                );
                let out = require(
                    session.functional_max_pool3d(x, (2, 2, 2), (2, 2, 2)),
                    "failed to run FrankenTorch max_pool3d",
                );
                (session, out)
            },
            |(mut session, out)| {
                black_box(require(
                    session.tensor_sum(out),
                    "failed to reduce FrankenTorch output",
                ))
            },
            BatchSize::SmallInput,
        );
    });

    group.bench_function("frankentorch_backward_only", |b| {
        b.iter_batched(
            || {
                let mut session = FrankenTorchSession::new(ExecutionMode::Strict);
                let x = require(
                    session.tensor_variable(values.clone(), shape.clone(), true),
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
                (session, loss)
            },
            |(mut session, loss)| {
                black_box(require(
                    session.tensor_backward(loss),
                    "failed to run FrankenTorch backward",
                ))
            },
            BatchSize::SmallInput,
        );
    });

    group.bench_function("kernel_forward_with_indices", |b| {
        b.iter(|| {
            black_box(ft_kernel_cpu::max_pool3d_forward_with_indices_f64(
                black_box(&values),
                MAX_POOL3D_N,
                MAX_POOL3D_C,
                MAX_POOL3D_D,
                MAX_POOL3D_H,
                MAX_POOL3D_W,
                2,
                2,
                2,
                out_d,
                out_h,
                out_w,
                2,
                2,
                2,
            ))
        });
    });

    group.bench_function("kernel_backward_from_indices", |b| {
        b.iter(|| {
            black_box(ft_kernel_cpu::max_pool3d_backward_from_indices_f64(
                black_box(&dout),
                black_box(&arg_offsets),
                MAX_POOL3D_N,
                MAX_POOL3D_C,
                MAX_POOL3D_D,
                MAX_POOL3D_H,
                MAX_POOL3D_W,
                out_d,
                out_h,
                out_w,
            ))
        });
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

fn bench_sdpa_grad_2x8x512x64(c: &mut Criterion) {
    let mut group = c.benchmark_group("gauntlet_sdpa_grad_2x8x512x64");
    group.warm_up_time(Duration::from_secs(1));
    group.measurement_time(Duration::from_secs(3));
    group.sample_size(10);

    let q_values = deterministic_values(SDPA_TOTAL, 0.0);
    let k_values = deterministic_values(SDPA_TOTAL, 1.0);
    let v_values = deterministic_values(SDPA_TOTAL, 2.0);
    let shape = vec![SDPA_BATCH, SDPA_HEADS, SDPA_SEQ, SDPA_D];

    group.bench_function("frankentorch_kgs4_113", |b| {
        b.iter(|| {
            let mut session = FrankenTorchSession::new(ExecutionMode::Strict);
            let q = require(
                session.tensor_variable(black_box(q_values.clone()), shape.clone(), true),
                "failed to create FrankenTorch SDPA query",
            );
            let k = require(
                session.tensor_variable(black_box(k_values.clone()), shape.clone(), true),
                "failed to create FrankenTorch SDPA key",
            );
            let v = require(
                session.tensor_variable(black_box(v_values.clone()), shape.clone(), true),
                "failed to create FrankenTorch SDPA value",
            );
            let out = require(
                session.scaled_dot_product_attention(q, k, v, None, 0.0, false),
                "failed to run FrankenTorch SDPA",
            );
            let loss = require(
                session.tensor_sum(out),
                "failed to reduce FrankenTorch SDPA",
            );
            black_box(require(
                session.tensor_backward(loss),
                "failed to run FrankenTorch SDPA backward",
            ))
        });
    });

    group.bench_function("pytorch_2_12_cpu", |b| {
        b.iter_custom(run_pytorch_sdpa_grad);
    });

    group.finish();
}

criterion_group!(
    benches,
    bench_avg_pool1d_unit_dy,
    bench_avg_pool2d_unit_dy,
    bench_batch_norm2d_f32_unit_dy,
    bench_conv3d_grad,
    bench_max_pool1d_unit_dout,
    bench_max_pool3d_saved_indices,
    bench_max_pool3d_stage_probe,
    bench_linear_train_hidden_2048,
    bench_sdpa_grad_2x8x512x64
);
criterion_main!(benches);

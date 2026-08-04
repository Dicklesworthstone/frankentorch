//! A/B for grid_sample bilinear BACKWARD (grid_sample_bilinear_backward_f64). The backward was a serial
//! `for n { for h { for w }}` scatter; now it fans the batch across rayon (grad_input[n]/grad_grid[n]
//! are disjoint per batch, so conflict-free + bit-identical). The changed fn IS the whole grid_sample
//! backward cost, so end-to-end backward ratio == the fn ratio. OLD vs NEW selected by the
//! FT_GRIDSAMPLE_SERIAL env gate in one binary (same worker). bitmatch on fresh sessions.
//! Run PLAIN (no pipe): cargo run --release -p ft-api --example gridsample_backward_ab

use ft_api::{FrankenTorchSession, GridSampleMode, GridSamplePaddingMode};
use ft_core::ExecutionMode;
use std::time::Instant;

fn mode() -> GridSampleMode {
    match std::env::var("FT_GRIDSAMPLE_MODE").as_deref() {
        Ok("nearest") => GridSampleMode::Nearest,
        _ => GridSampleMode::Bilinear,
    }
}

fn build(
    batch: usize,
    ch: usize,
    ih: usize,
    iw: usize,
    oh: usize,
    ow: usize,
) -> (Vec<f64>, Vec<f64>) {
    let input: Vec<f64> = (0..batch * ch * ih * iw)
        .map(|i| ((i % 251) as f64 - 125.0) * 0.01)
        .collect();
    // grid in [-1,1] with variety so all interp branches are hit.
    let grid: Vec<f64> = (0..batch * oh * ow * 2)
        .map(|i| ((i % 197) as f64 / 98.0) - 1.0)
        .collect();
    (input, grid)
}

fn run_once(
    input_v: &[f64],
    grid_v: &[f64],
    dims: (usize, usize, usize, usize, usize, usize),
) -> (Vec<f64>, Vec<f64>) {
    let (batch, ch, ih, iw, oh, ow) = dims;
    let mut s = FrankenTorchSession::new(ExecutionMode::Strict);
    let inp = s
        .tensor_variable(input_v.to_vec(), vec![batch, ch, ih, iw], true)
        .unwrap();
    let grd = s
        .tensor_variable(grid_v.to_vec(), vec![batch, oh, ow, 2], true)
        .unwrap();
    let out = s
        .grid_sample_tensor(inp, grd, mode(), GridSamplePaddingMode::Zeros, false)
        .unwrap();
    let loss = s.tensor_sum(out).unwrap();
    let report = s.tensor_backward(loss).unwrap();
    let ig = s.tensor_gradient(&report, inp).unwrap().to_vec();
    let gg = s.tensor_gradient(&report, grd).unwrap().to_vec();
    (ig, gg)
}

fn time_backward(
    input_v: &[f64],
    grid_v: &[f64],
    dims: (usize, usize, usize, usize, usize, usize),
) -> f64 {
    let (batch, ch, ih, iw, oh, ow) = dims;
    let mut s = FrankenTorchSession::new(ExecutionMode::Strict);
    let inp = s
        .tensor_variable(input_v.to_vec(), vec![batch, ch, ih, iw], true)
        .unwrap();
    let grd = s
        .tensor_variable(grid_v.to_vec(), vec![batch, oh, ow, 2], true)
        .unwrap();
    let mut best = f64::INFINITY;
    for _ in 0..9 {
        let out = s
            .grid_sample_tensor(inp, grd, mode(), GridSamplePaddingMode::Zeros, false)
            .unwrap();
        let loss = s.tensor_sum(out).unwrap();
        let t = Instant::now();
        let report = s.tensor_backward(loss).unwrap();
        let el = t.elapsed().as_secs_f64() * 1e3;
        std::hint::black_box(&report);
        if el < best {
            best = el;
        }
    }
    best
}

fn main() {
    // The code path (serial vs parallel) is chosen by the FT_GRIDSAMPLE_SERIAL env var
    // read INSIDE the backward. Run this example twice from the shell:
    //   FT_GRIDSAMPLE_SERIAL=1 ... example  -> OLD (serial)
    //   ... example                         -> NEW (parallel)
    // Compare the two `time` lines (ratio) and confirm the checksums are IDENTICAL (bit-exact).
    let dims = (8usize, 32usize, 64usize, 64usize, 64usize, 64usize);
    let (batch, ch, ih, iw, oh, ow) = dims;
    let (input_v, grid_v) = build(batch, ch, ih, iw, oh, ow);

    let (ig, gg) = run_once(&input_v, &grid_v, dims);
    // Bit-exact checksum via raw bits (order-independent xor-fold) so SERIAL vs PARALLEL
    // runs are directly comparable across the two processes.
    // wrapping_add fold: commutative (order-independent) AND non-cancelling (xor would
    // collapse to 0 for the many duplicate integer grads in the nearest path).
    let mut acc: u64 = 0;
    for &v in ig.iter().chain(gg.iter()) {
        acc = acc.wrapping_add(v.to_bits().wrapping_mul(0x9E3779B97F4A7C15));
    }
    let ms = time_backward(&input_v, &grid_v, dims);
    let mode = if std::env::var_os("FT_GRIDSAMPLE_SERIAL").is_some() {
        "SERIAL"
    } else {
        "PARALLEL"
    };
    println!(
        "grid_sample backward [{mode:<8}] batch={batch} ch={ch} in={ih}x{iw} out={oh}x{ow}  time {ms:8.3}ms  checksum {acc:016x}"
    );
}

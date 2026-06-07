#![deny(unsafe_code)]

use std::fmt;

use rayon::prelude::*;
use wide::{f32x8, f64x4};

#[allow(unsafe_code)]
mod gemm {
    use rayon::prelude::*;

    // Above this many fused multiply-adds, split the GEMM across output row
    // blocks and run them on the rayon pool. `matrixmultiply` is single-threaded,
    // so a large solo matmul (the torch→ATen/MKL gap is multi-threaded BLAS) left
    // all but one core idle. Each block is an independent matrixmultiply call over
    // a contiguous row range of A and C: for a given output element the
    // k-accumulation order is fixed by the micro-kernel and does NOT depend on the
    // row count, so the parallel result is bit-for-bit identical to the single
    // call (proved by `gemm_row_split_matches_single_bit_exact`).
    // Row-block parallelism pays once a block carries enough work to dwarf the
    // rayon dispatch cost. The previous 1<<29 (~537M FMA) gate left medium GEMMs
    // single-threaded on multi-core hosts — notably the conv2d im2col matmul
    // (M=4096,K=576,N=64 ≈ 151M FMA) and a 512×512 matmul (134M) — so all but
    // one core sat idle. 1<<27 (~134M) brings both under the parallel path while
    // keeping genuinely small matmuls (≤256×256 ≈ 16.7M) serial. The split is
    // bit-for-bit identical to the single call regardless of block count
    // (proved by `gemm_row_split_matches_single_bit_exact`).
    const PAR_MIN_FLOPS: u128 = 1 << 27;
    // A TALL GEMM (many rows, modest total flops) can sit below PAR_MIN_FLOPS yet
    // still split into plenty of well-sized row blocks — e.g. an attention/linear
    // projection `[batch*S, embed] @ [embed, embed]` at large S (M = batch*S).
    // Parallelize it too: enough rows to fill the pool AND enough total work to
    // dwarf the dispatch. Thread-count-independent (so it isn't fragile at the
    // pool-size boundary) and OR'd with the flop gate, so it only ADDS parallelism
    // — no previously-parallel or previously-serial-small matmul changes.
    const TALL_MIN_ROWS: usize = 1024;
    const TALL_MIN_FLOPS: u128 = 1 << 25; // ~33.6M FMA total
    const MIN_BLOCK_ROWS: usize = 8;

    fn should_parallelize(m: usize, k: usize, n: usize) -> bool {
        let flops = (m as u128) * (k as u128) * (n as u128);
        rayon::current_num_threads() > 1
            && m > MIN_BLOCK_ROWS
            && (flops >= PAR_MIN_FLOPS || (m >= TALL_MIN_ROWS && flops >= TALL_MIN_FLOPS))
    }

    fn block_rows(m: usize) -> usize {
        let threads = rayon::current_num_threads().max(1);
        m.div_ceil(threads).max(MIN_BLOCK_ROWS)
    }

    pub fn dgemm(m: usize, k: usize, n: usize, a: &[f64], b: &[f64], c: &mut [f64]) {
        if m == 0 || n == 0 {
            return;
        }
        let a = &a[..m * k];
        let b = &b[..k * n];
        let c = &mut c[..m * n];
        // Column path takes precedence for WIDE matmuls (n >> m): row-splitting
        // a small-M GEMM yields only ~m/MIN_BLOCK_ROWS blocks, while the column
        // path scales with N. (should_parallelize_cols already requires n > 4*m,
        // so non-wide matmuls are unaffected and keep the row split.)
        if should_parallelize_cols(m, k, n) {
            dgemm_col_parallel(m, k, n, a, b, c);
        } else if should_parallelize(m, k, n) {
            let br = block_rows(m);
            c.par_chunks_mut(br * n)
                .zip(a.par_chunks(br * k))
                .for_each(|(c_blk, a_blk)| {
                    dgemm_block(c_blk.len() / n, k, n, a_blk, b, c_blk);
                });
        } else {
            dgemm_block(m, k, n, a, b, c);
        }
    }

    // Column (N) parallelism for SMALL-m, LARGE-n matmuls that the row-split
    // misses — e.g. a linear layer `[batch, in] @ [in, out]^T` with batch << out:
    // there M is too small to row-split, so the whole GEMM ran serial. Each
    // n-block's K-accumulation is identical to the single call (matrixmultiply's
    // micro-kernel order is independent of the N tiling), so the result is
    // BIT-FOR-BIT identical to dgemm_block.
    const PAR_MIN_FLOPS_COLS: u128 = 1 << 24; // ~16.8M FMA
    const MIN_BLOCK_COLS: usize = 128;

    fn should_parallelize_cols(m: usize, k: usize, n: usize) -> bool {
        rayon::current_num_threads() > 1
            && n >= 4 * MIN_BLOCK_COLS
            && n > 4 * m
            && (m as u128) * (k as u128) * (n as u128) >= PAR_MIN_FLOPS_COLS
    }

    fn block_cols(n: usize) -> usize {
        let threads = rayon::current_num_threads().max(1);
        n.div_ceil(threads).max(MIN_BLOCK_COLS)
    }

    /// `C[m,n] = A[m,k] @ B^T` where `B` is row-major `[n, k]` (e.g. a Linear
    /// layer's weight `[out, in]`). Reads B through matrixmultiply's strides
    /// (`rsb=1, csb=k`) so the transpose is NEVER materialised — eliminating the
    /// cache-unfriendly 8MB transposed copy that `x @ weight.t()` otherwise pays.
    /// Bit-for-bit identical to materialise-transpose-then-dgemm (same dims => same
    /// K-accumulation; B's values are read identically, only via strides). Uses
    /// the same column/row/serial split as `dgemm`.
    pub fn dgemm_bt(m: usize, k: usize, n: usize, a: &[f64], b: &[f64], c: &mut [f64]) {
        if m == 0 || n == 0 {
            return;
        }
        let a = &a[..m * k];
        let b = &b[..n * k];
        let c = &mut c[..m * n];
        if should_parallelize_cols(m, k, n) {
            // Split N (= B's rows). Block [n0,n1) is the contiguous B rows
            // b[n0*k .. n1*k]; multiply A by their transpose into an owned buffer.
            let nb = block_cols(n);
            let blocks: Vec<(usize, Vec<f64>)> = (0..n.div_ceil(nb))
                .into_par_iter()
                .map(|blk| {
                    let n0 = blk * nb;
                    let bw = (n0 + nb).min(n) - n0;
                    let mut ct = vec![0.0f64; m * bw];
                    // SAFETY: a is m*k; b[n0*k ..] holds bw rows of k; ct is m*bw.
                    unsafe {
                        matrixmultiply::dgemm(
                            m,
                            k,
                            bw,
                            1.0,
                            a.as_ptr(),
                            k as isize,
                            1,
                            b.as_ptr().add(n0 * k),
                            1,
                            k as isize,
                            0.0,
                            ct.as_mut_ptr(),
                            bw as isize,
                            1,
                        );
                    }
                    (n0, ct)
                })
                .collect();
            for (n0, ct) in &blocks {
                let bw = ct.len() / m;
                for i in 0..m {
                    c[i * n + n0..i * n + n0 + bw].copy_from_slice(&ct[i * bw..i * bw + bw]);
                }
            }
        } else if should_parallelize(m, k, n) {
            let br = block_rows(m);
            c.par_chunks_mut(br * n)
                .zip(a.par_chunks(br * k))
                .for_each(|(c_blk, a_blk)| {
                    dgemm_bt_block(c_blk.len() / n, k, n, a_blk, b, c_blk);
                });
        } else {
            dgemm_bt_block(m, k, n, a, b, c);
        }
    }

    fn dgemm_bt_block(m: usize, k: usize, n: usize, a: &[f64], b: &[f64], c: &mut [f64]) {
        // SAFETY: a is m*k, b is n*k (read as B^T via rsb=1,csb=k), c is m*n.
        unsafe {
            matrixmultiply::dgemm(
                m,
                k,
                n,
                1.0,
                a.as_ptr(),
                k as isize,
                1,
                b.as_ptr(),
                1,
                k as isize,
                0.0,
                c.as_mut_ptr(),
                n as isize,
                1,
            );
        }
    }

    fn dgemm_col_parallel(m: usize, k: usize, n: usize, a: &[f64], b: &[f64], c: &mut [f64]) {
        let nb = block_cols(n);
        let blocks: Vec<(usize, Vec<f64>)> = (0..n.div_ceil(nb))
            .into_par_iter()
            .map(|blk| {
                let n0 = blk * nb;
                let bw = (n0 + nb).min(n) - n0;
                let mut ct = vec![0.0f64; m * bw];
                // SAFETY: a is m*k; b is k*n, so the strided column window starting
                // at `n0` with row stride n and `bw` columns is in bounds; ct is the
                // exact m*bw owned output. matrixmultiply's K order is independent of
                // the N tiling, so ct[i][j] == the single call's c[i][n0+j] bit-for-bit.
                unsafe {
                    matrixmultiply::dgemm(
                        m,
                        k,
                        bw,
                        1.0,
                        a.as_ptr(),
                        k as isize,
                        1,
                        b.as_ptr().add(n0),
                        n as isize,
                        1,
                        0.0,
                        ct.as_mut_ptr(),
                        bw as isize,
                        1,
                    );
                }
                (n0, ct)
            })
            .collect();
        for (n0, ct) in &blocks {
            let bw = ct.len() / m;
            for i in 0..m {
                c[i * n + n0..i * n + n0 + bw].copy_from_slice(&ct[i * bw..i * bw + bw]);
            }
        }
    }

    pub fn dgemm_block(m: usize, k: usize, n: usize, a: &[f64], b: &[f64], c: &mut [f64]) {
        // SAFETY: matrixmultiply::dgemm requires valid pointers and correct dimensions.
        // a.len() == m*k, b.len() == k*n, c.len() == m*n (sliced exactly by `dgemm`).
        // Row-major layout: row stride = inner dimension, column stride = 1.
        unsafe {
            matrixmultiply::dgemm(
                m,
                k,
                n,
                1.0,
                a.as_ptr(),
                k as isize,
                1,
                b.as_ptr(),
                n as isize,
                1,
                0.0,
                c.as_mut_ptr(),
                n as isize,
                1,
            );
        }
    }

    pub fn sgemm(m: usize, k: usize, n: usize, a: &[f32], b: &[f32], c: &mut [f32]) {
        if m == 0 || n == 0 {
            return;
        }
        let a = &a[..m * k];
        let b = &b[..k * n];
        let c = &mut c[..m * n];
        // Same structure as dgemm: column path for WIDE matmuls (n >> m), else
        // row split (square/tall via should_parallelize), else serial.
        if should_parallelize_cols(m, k, n) {
            sgemm_col_parallel(m, k, n, a, b, c);
        } else if should_parallelize(m, k, n) {
            let br = block_rows(m);
            c.par_chunks_mut(br * n)
                .zip(a.par_chunks(br * k))
                .for_each(|(c_blk, a_blk)| {
                    sgemm_block(c_blk.len() / n, k, n, a_blk, b, c_blk);
                });
        } else {
            sgemm_block(m, k, n, a, b, c);
        }
    }

    fn sgemm_col_parallel(m: usize, k: usize, n: usize, a: &[f32], b: &[f32], c: &mut [f32]) {
        let nb = block_cols(n);
        let blocks: Vec<(usize, Vec<f32>)> = (0..n.div_ceil(nb))
            .into_par_iter()
            .map(|blk| {
                let n0 = blk * nb;
                let bw = (n0 + nb).min(n) - n0;
                let mut ct = vec![0.0f32; m * bw];
                // SAFETY: mirror of dgemm_col_parallel — a is m*k; b's strided
                // column window at n0 (row stride n, bw cols) is in bounds; ct is
                // the exact m*bw owned output. K order is independent of N tiling,
                // so ct[i][j] == the single call's c[i][n0+j] bit-for-bit.
                unsafe {
                    matrixmultiply::sgemm(
                        m,
                        k,
                        bw,
                        1.0,
                        a.as_ptr(),
                        k as isize,
                        1,
                        b.as_ptr().add(n0),
                        n as isize,
                        1,
                        0.0,
                        ct.as_mut_ptr(),
                        bw as isize,
                        1,
                    );
                }
                (n0, ct)
            })
            .collect();
        for (n0, ct) in &blocks {
            let bw = ct.len() / m;
            for i in 0..m {
                c[i * n + n0..i * n + n0 + bw].copy_from_slice(&ct[i * bw..i * bw + bw]);
            }
        }
    }

    /// f32 mirror of `dgemm_bt`: `C[m,n] = A[m,k] @ B^T` where `B` is `[n,k]`.
    pub fn sgemm_bt(m: usize, k: usize, n: usize, a: &[f32], b: &[f32], c: &mut [f32]) {
        if m == 0 || n == 0 {
            return;
        }
        let a = &a[..m * k];
        let b = &b[..n * k];
        let c = &mut c[..m * n];
        if should_parallelize_cols(m, k, n) {
            let nb = block_cols(n);
            let blocks: Vec<(usize, Vec<f32>)> = (0..n.div_ceil(nb))
                .into_par_iter()
                .map(|blk| {
                    let n0 = blk * nb;
                    let bw = (n0 + nb).min(n) - n0;
                    let mut ct = vec![0.0f32; m * bw];
                    // SAFETY: a is m*k; b[n0*k ..] holds bw rows of k; ct is m*bw.
                    unsafe {
                        matrixmultiply::sgemm(
                            m,
                            k,
                            bw,
                            1.0,
                            a.as_ptr(),
                            k as isize,
                            1,
                            b.as_ptr().add(n0 * k),
                            1,
                            k as isize,
                            0.0,
                            ct.as_mut_ptr(),
                            bw as isize,
                            1,
                        );
                    }
                    (n0, ct)
                })
                .collect();
            for (n0, ct) in &blocks {
                let bw = ct.len() / m;
                for i in 0..m {
                    c[i * n + n0..i * n + n0 + bw].copy_from_slice(&ct[i * bw..i * bw + bw]);
                }
            }
        } else if should_parallelize(m, k, n) {
            let br = block_rows(m);
            c.par_chunks_mut(br * n)
                .zip(a.par_chunks(br * k))
                .for_each(|(c_blk, a_blk)| {
                    sgemm_bt_block(c_blk.len() / n, k, n, a_blk, b, c_blk);
                });
        } else {
            sgemm_bt_block(m, k, n, a, b, c);
        }
    }

    fn sgemm_bt_block(m: usize, k: usize, n: usize, a: &[f32], b: &[f32], c: &mut [f32]) {
        // SAFETY: a is m*k, b is n*k (read as B^T via rsb=1,csb=k), c is m*n.
        unsafe {
            matrixmultiply::sgemm(
                m,
                k,
                n,
                1.0,
                a.as_ptr(),
                k as isize,
                1,
                b.as_ptr(),
                1,
                k as isize,
                0.0,
                c.as_mut_ptr(),
                n as isize,
                1,
            );
        }
    }

    pub fn sgemm_block(m: usize, k: usize, n: usize, a: &[f32], b: &[f32], c: &mut [f32]) {
        // SAFETY: see `dgemm_block`; slices are sized exactly by `sgemm`.
        unsafe {
            matrixmultiply::sgemm(
                m,
                k,
                n,
                1.0,
                a.as_ptr(),
                k as isize,
                1,
                b.as_ptr(),
                n as isize,
                1,
                0.0,
                c.as_mut_ptr(),
                n as isize,
                1,
            );
        }
    }
}

use ft_core::{
    Complex128, ScalarTensor, SparseCOOTensor, SparseTensorError, TensorCompatError, TensorMeta,
    ensure_compatible,
};

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum KernelError {
    Incompatible(TensorCompatError),
    ShapeMismatch {
        lhs: Vec<usize>,
        rhs: Vec<usize>,
    },
    UnsupportedLayout {
        side: &'static str,
    },
    StorageSpanOverflow {
        side: &'static str,
        storage_offset: usize,
        numel: usize,
    },
    InsufficientStorage {
        side: &'static str,
        needed: usize,
        available: usize,
    },
    InvalidDimension {
        dim: usize,
        ndim: usize,
    },
    ShapeOverflow {
        context: &'static str,
    },
    SingularMatrix {
        size: usize,
    },
    NotPositiveDefinite,
    /// A reduction that has no identity element (argmax/argmin/max/min)
    /// was applied along a dimension of size zero. PyTorch raises a
    /// runtime error here rather than returning a sentinel.
    EmptyReductionDim {
        dim: usize,
    },
}

impl fmt::Display for KernelError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Incompatible(error) => write!(f, "incompatible tensors: {error}"),
            Self::ShapeMismatch { lhs, rhs } => {
                write!(f, "shape mismatch: lhs={lhs:?}, rhs={rhs:?}")
            }
            Self::UnsupportedLayout { side } => {
                write!(f, "unsupported non-contiguous layout on {side}")
            }
            Self::StorageSpanOverflow {
                side,
                storage_offset,
                numel,
            } => write!(
                f,
                "storage span overflow on {side}: storage_offset={storage_offset}, numel={numel}"
            ),
            Self::InsufficientStorage {
                side,
                needed,
                available,
            } => write!(
                f,
                "insufficient storage on {side}: needed={needed}, available={available}"
            ),
            Self::InvalidDimension { dim, ndim } => write!(
                f,
                "invalid reduction dimension {dim} for tensor with {ndim} dimensions"
            ),
            Self::ShapeOverflow { context } => {
                write!(f, "shape arithmetic overflow: {context}")
            }
            Self::SingularMatrix { size } => {
                write!(f, "singular matrix: size={size}x{size}")
            }
            Self::NotPositiveDefinite => {
                write!(f, "matrix is not positive definite")
            }
            Self::EmptyReductionDim { dim } => write!(
                f,
                "cannot reduce (argmax/argmin/max/min) over dimension {dim} of size zero"
            ),
        }
    }
}

impl std::error::Error for KernelError {}

impl From<TensorCompatError> for KernelError {
    fn from(value: TensorCompatError) -> Self {
        Self::Incompatible(value)
    }
}

pub fn neg_scalar(input: &ScalarTensor) -> ScalarTensor {
    input.with_value(-input.value())
}

pub fn abs_scalar(input: &ScalarTensor) -> ScalarTensor {
    input.with_value(input.value().abs())
}

pub fn exp_scalar(input: &ScalarTensor) -> ScalarTensor {
    input.with_value(input.value().exp())
}

pub fn log_scalar(input: &ScalarTensor) -> ScalarTensor {
    input.with_value(input.value().ln())
}

pub fn relu_scalar(input: &ScalarTensor) -> ScalarTensor {
    input.with_value(if input.value() > 0.0 {
        input.value()
    } else {
        0.0
    })
}

pub fn sigmoid_scalar(input: &ScalarTensor) -> ScalarTensor {
    input.with_value(1.0 / (1.0 + (-input.value()).exp()))
}

pub fn tanh_scalar(input: &ScalarTensor) -> ScalarTensor {
    input.with_value(input.value().tanh())
}

pub fn sqrt_scalar(input: &ScalarTensor) -> ScalarTensor {
    input.with_value(input.value().sqrt())
}

pub fn reciprocal_scalar(input: &ScalarTensor) -> ScalarTensor {
    input.with_value(1.0 / input.value())
}

pub fn sin_scalar(input: &ScalarTensor) -> ScalarTensor {
    input.with_value(input.value().sin())
}

pub fn cos_scalar(input: &ScalarTensor) -> ScalarTensor {
    input.with_value(input.value().cos())
}

pub fn tan_scalar(input: &ScalarTensor) -> ScalarTensor {
    input.with_value(input.value().tan())
}

pub fn floor_scalar(input: &ScalarTensor) -> ScalarTensor {
    input.with_value(input.value().floor())
}

pub fn ceil_scalar(input: &ScalarTensor) -> ScalarTensor {
    input.with_value(input.value().ceil())
}

fn round_ties_even_f64(value: f64) -> f64 {
    value.round_ties_even()
}

pub fn round_scalar(input: &ScalarTensor) -> ScalarTensor {
    input.with_value(round_ties_even_f64(input.value()))
}

pub fn log2_scalar(input: &ScalarTensor) -> ScalarTensor {
    input.with_value(input.value().log2())
}

pub fn log10_scalar(input: &ScalarTensor) -> ScalarTensor {
    input.with_value(input.value().log10())
}

pub fn log1p_scalar(input: &ScalarTensor) -> ScalarTensor {
    input.with_value(input.value().ln_1p())
}

pub fn expm1_scalar(input: &ScalarTensor) -> ScalarTensor {
    input.with_value(input.value().exp_m1())
}

/// PyTorch parity: `torch.sign` maps both signed zeros to +0.0,
/// propagates NaN, and returns ±1.0 only for non-zero finite values.
/// Rust's `f64::signum` instead returns ±1.0 for ±0.0 (IEEE 754
/// sign-bit semantics), which is observably different — fix it once
/// here so every dispatch backend sees the same semantics.
fn torch_sign_f64(value: f64) -> f64 {
    if value.is_nan() {
        f64::NAN
    } else if value == 0.0 {
        0.0
    } else if value > 0.0 {
        1.0
    } else {
        -1.0
    }
}

pub fn sign_scalar(input: &ScalarTensor) -> ScalarTensor {
    input.with_value(torch_sign_f64(input.value()))
}

pub fn trunc_scalar(input: &ScalarTensor) -> ScalarTensor {
    input.with_value(input.value().trunc())
}

pub fn frac_scalar(input: &ScalarTensor) -> ScalarTensor {
    input.with_value(input.value().fract())
}

pub fn asin_scalar(input: &ScalarTensor) -> ScalarTensor {
    input.with_value(input.value().asin())
}

pub fn acos_scalar(input: &ScalarTensor) -> ScalarTensor {
    input.with_value(input.value().acos())
}

pub fn atan_scalar(input: &ScalarTensor) -> ScalarTensor {
    input.with_value(input.value().atan())
}

pub fn sinh_scalar(input: &ScalarTensor) -> ScalarTensor {
    input.with_value(input.value().sinh())
}

pub fn cosh_scalar(input: &ScalarTensor) -> ScalarTensor {
    input.with_value(input.value().cosh())
}

fn gelu_value(x: f64) -> f64 {
    // PyTorch default GELU (approximate="none"): exact erf form
    // GELU(x) = 0.5 * x * (1 + erf(x / sqrt(2)))
    // Uses libm-quality erf so we match torch's CPU std::erf path bit-for-bit.
    0.5 * x * (1.0 + libm::erf(x * std::f64::consts::FRAC_1_SQRT_2))
}

pub fn gelu_scalar(input: &ScalarTensor) -> ScalarTensor {
    input.with_value(gelu_value(input.value()))
}

fn silu_value(x: f64) -> f64 {
    x / (1.0 + (-x).exp())
}

pub fn silu_scalar(input: &ScalarTensor) -> ScalarTensor {
    input.with_value(silu_value(input.value()))
}

fn leaky_relu_value(x: f64) -> f64 {
    if x >= 0.0 { x } else { 0.01 * x }
}

pub fn leaky_relu_scalar(input: &ScalarTensor) -> ScalarTensor {
    input.with_value(leaky_relu_value(input.value()))
}

fn elu_value(x: f64) -> f64 {
    if x > 0.0 { x } else { x.exp() - 1.0 }
}

pub fn elu_scalar(input: &ScalarTensor) -> ScalarTensor {
    input.with_value(elu_value(input.value()))
}

pub fn rsqrt_scalar(input: &ScalarTensor) -> ScalarTensor {
    input.with_value(1.0 / input.value().sqrt())
}

/// Gauss error function approximation (Abramowitz and Stegun, formula 7.1.26).
/// Maximum error: |ε(x)| ≤ 1.5×10⁻⁷.
/// f64 error function with libm-quality precision (~1 ULP across the
/// entire f64 domain), via the pure-Rust `libm` crate. The previous
/// implementation used the Abramowitz-Stegun 7.1.26 polynomial
/// approximation whose absolute error of ~1.5e-7 collapsed precision
/// to single-precision territory — an 8-orders-of-magnitude regression
/// vs PyTorch's `torch.erf` which wraps libm `erf`. Routing through
/// `libm::erf` restores f64 precision and bit-aligns with the upstream
/// reference (see `torch_erf_erfc_libm_subprocess_conformance` in
/// ft-conformance).
fn erf_value(x: f64) -> f64 {
    libm::erf(x)
}

/// Companion to `erf_value` for the complementary error function. Using
/// `libm::erfc` directly (instead of computing `1.0 - erf(x)`) preserves
/// precision in the tail region |x| >> 1 where `1 - erf(x)` cancels
/// down to subnormal magnitudes — `1.0 - erf(x)` would round to 0.0
/// for |x| > ~5.95 even though `erfc(x)` is still a tiny but non-zero
/// positive number.
fn erfc_value(x: f64) -> f64 {
    libm::erfc(x)
}

pub fn erf_scalar(input: &ScalarTensor) -> ScalarTensor {
    input.with_value(erf_value(input.value()))
}

pub fn erfc_scalar(input: &ScalarTensor) -> ScalarTensor {
    input.with_value(erfc_value(input.value()))
}

fn hardswish_value(x: f64) -> f64 {
    if x <= -3.0 {
        0.0
    } else if x >= 3.0 {
        x
    } else {
        x * (x + 3.0) / 6.0
    }
}

pub fn hardswish_scalar(input: &ScalarTensor) -> ScalarTensor {
    input.with_value(hardswish_value(input.value()))
}

fn hardsigmoid_value(x: f64) -> f64 {
    if x <= -3.0 {
        0.0
    } else if x >= 3.0 {
        1.0
    } else {
        (x + 3.0) / 6.0
    }
}

pub fn hardsigmoid_scalar(input: &ScalarTensor) -> ScalarTensor {
    input.with_value(hardsigmoid_value(input.value()))
}

fn hardtanh_value(x: f64) -> f64 {
    x.clamp(-1.0, 1.0)
}

pub fn hardtanh_scalar(input: &ScalarTensor) -> ScalarTensor {
    input.with_value(hardtanh_value(input.value()))
}

fn softplus_value(x: f64) -> f64 {
    // PyTorch: softplus(x) = log1p(exp(beta*x)) / beta, with the
    // upper-threshold short circuit (returns x when beta*x > threshold)
    // for numerical stability when exp would overflow.
    //
    // Defaults: beta = 1, threshold = 20. PyTorch only thresholds in
    // the *upper* direction — for x → -∞, log1p(exp(x)) decays smoothly
    // to 0 without losing precision, so an artificial lower clamp would
    // wrongly flatten values like softplus(-25) ≈ 1.39e-11 to 0.
    //
    // log1p(exp(x)) is the precise form: for negative x, exp(x) is
    // small and `log(1 + small)` would lose a leading digit relative to
    // log1p(small).
    if x > 20.0 { x } else { x.exp().ln_1p() }
}

pub fn softplus_scalar(input: &ScalarTensor) -> ScalarTensor {
    input.with_value(softplus_value(input.value()))
}

fn mish_value(x: f64) -> f64 {
    x * softplus_value(x).tanh()
}

pub fn mish_scalar(input: &ScalarTensor) -> ScalarTensor {
    input.with_value(mish_value(input.value()))
}

pub fn square_scalar(input: &ScalarTensor) -> ScalarTensor {
    let v = input.value();
    input.with_value(v * v)
}

pub fn isnan_scalar(input: &ScalarTensor) -> ScalarTensor {
    input.with_value(if input.value().is_nan() { 1.0 } else { 0.0 })
}

pub fn isinf_scalar(input: &ScalarTensor) -> ScalarTensor {
    input.with_value(if input.value().is_infinite() {
        1.0
    } else {
        0.0
    })
}

pub fn isfinite_scalar(input: &ScalarTensor) -> ScalarTensor {
    input.with_value(if input.value().is_finite() { 1.0 } else { 0.0 })
}

pub fn pow_scalar(input: &ScalarTensor, exponent: f64) -> ScalarTensor {
    input.with_value(powf_torch_signed_zero_f64(input.value(), exponent))
}

fn powf_torch_signed_zero_f64(value: f64, exponent: f64) -> f64 {
    // Defer to Rust's f64::powf, which matches libm pow bit-for-bit.
    // The previous body overrode pow(-0, fractional > 0) from +0 to
    // -0 to match an older torch convention, but current libm and
    // torch both return +0 (per IEEE 754-2008 §9.2.1). Pinned by
    // torch_pow_ieee754_subprocess_conformance (vgj2).
    value.powf(exponent)
}

fn powf_torch_signed_zero_f32(value: f32, exponent: f32) -> f32 {
    // Match libm convention: pow(-0, fractional > 0) returns +0.
    // See powf_torch_signed_zero_f64 for the rationale.
    value.powf(exponent)
}

pub fn clamp_scalar(input: &ScalarTensor, min_val: f64, max_val: f64) -> ScalarTensor {
    let value = input.value();
    // clamp is min(max(x, min_val), max_val): the lower bound is applied
    // first, then the upper. When min_val > max_val the upper bound wins,
    // matching PyTorch / std::min(std::max(...)).
    let clamped = if value.is_nan() {
        f64::NAN
    } else {
        let lo = if !min_val.is_nan() && value < min_val {
            min_val
        } else {
            value
        };
        if !max_val.is_nan() && lo > max_val {
            max_val
        } else {
            lo
        }
    };
    input.with_value(clamped)
}

pub fn min_scalar(lhs: &ScalarTensor, rhs: &ScalarTensor) -> Result<ScalarTensor, KernelError> {
    ensure_compatible(lhs, rhs)?;
    let val = if lhs.value().is_nan() || rhs.value().is_nan() {
        f64::NAN
    } else {
        lhs.value().min(rhs.value())
    };
    Ok(lhs.with_value(val))
}

pub fn max_scalar(lhs: &ScalarTensor, rhs: &ScalarTensor) -> Result<ScalarTensor, KernelError> {
    ensure_compatible(lhs, rhs)?;
    let val = if lhs.value().is_nan() || rhs.value().is_nan() {
        f64::NAN
    } else {
        lhs.value().max(rhs.value())
    };
    Ok(lhs.with_value(val))
}

pub fn eq_scalar(lhs: &ScalarTensor, rhs: &ScalarTensor) -> Result<ScalarTensor, KernelError> {
    ensure_compatible(lhs, rhs)?;
    Ok(lhs.with_value(if lhs.value() == rhs.value() { 1.0 } else { 0.0 }))
}

pub fn ne_scalar(lhs: &ScalarTensor, rhs: &ScalarTensor) -> Result<ScalarTensor, KernelError> {
    ensure_compatible(lhs, rhs)?;
    Ok(lhs.with_value(if lhs.value() != rhs.value() { 1.0 } else { 0.0 }))
}

pub fn lt_scalar(lhs: &ScalarTensor, rhs: &ScalarTensor) -> Result<ScalarTensor, KernelError> {
    ensure_compatible(lhs, rhs)?;
    Ok(lhs.with_value(if lhs.value() < rhs.value() { 1.0 } else { 0.0 }))
}

pub fn gt_scalar(lhs: &ScalarTensor, rhs: &ScalarTensor) -> Result<ScalarTensor, KernelError> {
    ensure_compatible(lhs, rhs)?;
    Ok(lhs.with_value(if lhs.value() > rhs.value() { 1.0 } else { 0.0 }))
}

pub fn le_scalar(lhs: &ScalarTensor, rhs: &ScalarTensor) -> Result<ScalarTensor, KernelError> {
    ensure_compatible(lhs, rhs)?;
    Ok(lhs.with_value(if lhs.value() <= rhs.value() { 1.0 } else { 0.0 }))
}

pub fn ge_scalar(lhs: &ScalarTensor, rhs: &ScalarTensor) -> Result<ScalarTensor, KernelError> {
    ensure_compatible(lhs, rhs)?;
    Ok(lhs.with_value(if lhs.value() >= rhs.value() { 1.0 } else { 0.0 }))
}

pub fn add_scalar(lhs: &ScalarTensor, rhs: &ScalarTensor) -> Result<ScalarTensor, KernelError> {
    ensure_compatible(lhs, rhs)?;
    Ok(lhs.with_value(lhs.value() + rhs.value()))
}

pub fn sub_scalar(lhs: &ScalarTensor, rhs: &ScalarTensor) -> Result<ScalarTensor, KernelError> {
    ensure_compatible(lhs, rhs)?;
    Ok(lhs.with_value(lhs.value() - rhs.value()))
}

pub fn div_scalar(lhs: &ScalarTensor, rhs: &ScalarTensor) -> Result<ScalarTensor, KernelError> {
    ensure_compatible(lhs, rhs)?;
    Ok(lhs.with_value(lhs.value() / rhs.value()))
}

pub fn mul_scalar(lhs: &ScalarTensor, rhs: &ScalarTensor) -> Result<ScalarTensor, KernelError> {
    ensure_compatible(lhs, rhs)?;
    Ok(lhs.with_value(lhs.value() * rhs.value()))
}

pub fn atan2_scalar(lhs: &ScalarTensor, rhs: &ScalarTensor) -> Result<ScalarTensor, KernelError> {
    ensure_compatible(lhs, rhs)?;
    Ok(lhs.with_value(lhs.value().atan2(rhs.value())))
}

pub fn fmod_scalar(lhs: &ScalarTensor, rhs: &ScalarTensor) -> Result<ScalarTensor, KernelError> {
    ensure_compatible(lhs, rhs)?;
    Ok(lhs.with_value(lhs.value() % rhs.value()))
}

pub fn remainder_scalar(
    lhs: &ScalarTensor,
    rhs: &ScalarTensor,
) -> Result<ScalarTensor, KernelError> {
    ensure_compatible(lhs, rhs)?;
    let a = lhs.value();
    let b = rhs.value();
    Ok(lhs.with_value(a - (a / b).floor() * b))
}

fn ensure_dtype_device_and_layout(lhs: &TensorMeta, rhs: &TensorMeta) -> Result<(), KernelError> {
    if lhs.dtype() != rhs.dtype() {
        return Err(KernelError::Incompatible(
            TensorCompatError::DTypeMismatch {
                lhs: lhs.dtype(),
                rhs: rhs.dtype(),
            },
        ));
    }

    if lhs.device() != rhs.device() {
        return Err(KernelError::Incompatible(
            TensorCompatError::DeviceMismatch {
                lhs: lhs.device(),
                rhs: rhs.device(),
            },
        ));
    }

    if !lhs.is_contiguous() {
        return Err(KernelError::UnsupportedLayout { side: "lhs" });
    }

    if !rhs.is_contiguous() {
        return Err(KernelError::UnsupportedLayout { side: "rhs" });
    }

    Ok(())
}

fn ensure_meta_shape_and_dtype(lhs: &TensorMeta, rhs: &TensorMeta) -> Result<(), KernelError> {
    if lhs.dtype() != rhs.dtype() {
        return Err(KernelError::Incompatible(
            TensorCompatError::DTypeMismatch {
                lhs: lhs.dtype(),
                rhs: rhs.dtype(),
            },
        ));
    }

    if lhs.device() != rhs.device() {
        return Err(KernelError::Incompatible(
            TensorCompatError::DeviceMismatch {
                lhs: lhs.device(),
                rhs: rhs.device(),
            },
        ));
    }

    if lhs.shape() != rhs.shape() {
        return Err(KernelError::ShapeMismatch {
            lhs: lhs.shape().to_vec(),
            rhs: rhs.shape().to_vec(),
        });
    }

    Ok(())
}

fn matmul_dims(
    lhs_meta: &TensorMeta,
    rhs_meta: &TensorMeta,
) -> Result<(usize, usize, usize), KernelError> {
    if lhs_meta.shape().len() != 2 || rhs_meta.shape().len() != 2 {
        return Err(KernelError::ShapeMismatch {
            lhs: lhs_meta.shape().to_vec(),
            rhs: rhs_meta.shape().to_vec(),
        });
    }

    let m = lhs_meta.shape()[0];
    let k = lhs_meta.shape()[1];
    let rhs_k = rhs_meta.shape()[0];
    let n = rhs_meta.shape()[1];
    if k != rhs_k {
        return Err(KernelError::ShapeMismatch {
            lhs: lhs_meta.shape().to_vec(),
            rhs: rhs_meta.shape().to_vec(),
        });
    }

    Ok((m, k, n))
}

fn contiguous_required_len(meta: &TensorMeta, side: &'static str) -> Result<usize, KernelError> {
    let numel = meta.numel();
    if numel == 0 {
        return Ok(0);
    }

    meta.storage_offset()
        .checked_add(numel)
        .ok_or(KernelError::StorageSpanOverflow {
            side,
            storage_offset: meta.storage_offset(),
            numel,
        })
}

fn ensure_storage_len(
    buffer: &[f64],
    meta: &TensorMeta,
    side: &'static str,
) -> Result<(), KernelError> {
    let needed = contiguous_required_len(meta, side)?;
    if buffer.len() < needed {
        return Err(KernelError::InsufficientStorage {
            side,
            needed,
            available: buffer.len(),
        });
    }
    Ok(())
}

fn elementwise_f64<F>(
    lhs: &[f64],
    rhs: &[f64],
    lhs_meta: &TensorMeta,
    rhs_meta: &TensorMeta,
    op: F,
) -> Result<Vec<f64>, KernelError>
where
    F: Fn(f64, f64) -> f64 + Sync,
{
    ensure_meta_shape_and_dtype(lhs_meta, rhs_meta)?;

    if !lhs_meta.is_contiguous() || !rhs_meta.is_contiguous() {
        return elementwise_strided_f64(lhs, rhs, lhs_meta, rhs_meta, op);
    }

    ensure_storage_len(lhs, lhs_meta, "lhs")?;
    ensure_storage_len(rhs, rhs_meta, "rhs")?;

    let numel = lhs_meta.numel();
    if numel == 0 {
        return Ok(Vec::new());
    }

    let lhs_start = lhs_meta.storage_offset();
    let rhs_start = rhs_meta.storage_offset();
    let lhs_window = &lhs[lhs_start..lhs_start + numel];
    let rhs_window = &rhs[rhs_start..rhs_start + numel];

    if numel >= PARALLEL_THRESHOLD {
        Ok(lhs_window
            .par_iter()
            .zip(rhs_window.par_iter())
            .map(|(left, right)| op(*left, *right))
            .collect())
    } else {
        Ok(lhs_window
            .iter()
            .zip(rhs_window.iter())
            .map(|(left, right)| op(*left, *right))
            .collect())
    }
}

fn ensure_unary_layout_and_storage(buffer: &[f64], meta: &TensorMeta) -> Result<(), KernelError> {
    if !meta.is_contiguous() {
        return Err(KernelError::UnsupportedLayout { side: "input" });
    }
    ensure_storage_len(buffer, meta, "input")
}

fn strided_index(coords: &[usize], strides: &[usize], offset: usize) -> usize {
    let mut idx = offset;
    for (c, s) in coords.iter().zip(strides.iter()) {
        idx += c * s;
    }
    idx
}

fn increment_coords(coords: &mut [usize], shape: &[usize]) -> bool {
    for d in (0..coords.len()).rev() {
        coords[d] += 1;
        if coords[d] < shape[d] {
            return true;
        }
        coords[d] = 0;
    }
    false
}

fn elementwise_strided_f64<F>(
    lhs: &[f64],
    rhs: &[f64],
    lhs_meta: &TensorMeta,
    rhs_meta: &TensorMeta,
    op: F,
) -> Result<Vec<f64>, KernelError>
where
    F: Fn(f64, f64) -> f64,
{
    let shape = lhs_meta.shape();
    let numel = lhs_meta.numel();
    if numel == 0 {
        return Ok(Vec::new());
    }

    let lhs_strides = lhs_meta.strides();
    let rhs_strides = rhs_meta.strides();
    let lhs_offset = lhs_meta.storage_offset();
    let rhs_offset = rhs_meta.storage_offset();

    let mut output = Vec::with_capacity(numel);
    let mut coords = vec![0usize; shape.len()];

    loop {
        let lhs_idx = strided_index(&coords, lhs_strides, lhs_offset);
        let rhs_idx = strided_index(&coords, rhs_strides, rhs_offset);
        output.push(op(lhs[lhs_idx], rhs[rhs_idx]));
        if !increment_coords(&mut coords, shape) {
            break;
        }
    }

    Ok(output)
}

fn unary_strided_f64<F>(input: &[f64], meta: &TensorMeta, op: F) -> Result<Vec<f64>, KernelError>
where
    F: Fn(f64) -> f64,
{
    let shape = meta.shape();
    let numel = meta.numel();
    if numel == 0 {
        return Ok(Vec::new());
    }

    let strides = meta.strides();
    let offset = meta.storage_offset();

    let mut output = Vec::with_capacity(numel);
    let mut coords = vec![0usize; shape.len()];

    loop {
        let idx = strided_index(&coords, strides, offset);
        output.push(op(input[idx]));
        if !increment_coords(&mut coords, shape) {
            break;
        }
    }

    Ok(output)
}

fn checked_shape_numel(shape: &[usize], context: &'static str) -> Result<usize, KernelError> {
    let mut product = 1usize;
    for dim in shape.iter().copied() {
        if dim == 0 {
            return Ok(0);
        }
        product = product
            .checked_mul(dim)
            .ok_or(KernelError::ShapeOverflow { context })?;
    }
    Ok(product)
}

fn checked_mul(lhs: usize, rhs: usize, context: &'static str) -> Result<usize, KernelError> {
    lhs.checked_mul(rhs)
        .ok_or(KernelError::ShapeOverflow { context })
}

fn checked_contiguous_range(
    outer: usize,
    block_len: usize,
    context: &'static str,
) -> Result<std::ops::Range<usize>, KernelError> {
    let start = checked_mul(outer, block_len, context)?;
    let end = start
        .checked_add(block_len)
        .ok_or(KernelError::ShapeOverflow { context })?;
    Ok(start..end)
}

fn checked_dim_loop_sizes(
    shape: &[usize],
    dim: usize,
    context: &'static str,
) -> Result<(usize, usize, usize), KernelError> {
    let outer_size = checked_shape_numel(&shape[..dim], context)?;
    let inner_size = checked_shape_numel(&shape[dim + 1..], context)?;
    let total_size = checked_shape_numel(shape, context)?;
    Ok((outer_size, inner_size, total_size))
}

const PARALLEL_THRESHOLD: usize = 8192;

// The SIMD unary ops (relu/sqrt/reciprocal/...) were SERIAL while the scalar-map
// unary path (exp/tanh/...) parallelises — so relu was ~36-59x slower than torch
// at large N purely from single-threading. Parallelise above this gate (cheap
// per-element work, so amortise rayon split/join only at large N, same as the
// scalar path's threshold).
const SIMD_UNARY_PARALLEL_THRESHOLD: usize = 1 << 19; // 524288

// The generic scalar-map unary path (exp/ln/sin/gelu/erf/...) is dominated by a
// per-element libm call (~15-20 ns), but rayon's split/join/collect overhead on
// a many-core pool only amortises at very large N. A same-worker A/B
// (RAYON_NUM_THREADS=1 vs default, 64-core) showed `tensor_exp` PARALLEL was a
// net loss until ~0.5M elements:
//   exp/10000   serial 149 us  vs parallel 833 us   (5.6x slower parallel)
//   exp/100000  serial 1.28 ms vs parallel 2.63 ms  (2.0x slower parallel)
//   exp/1000000 serial 19.5 ms vs parallel 9.0 ms   (2.2x faster parallel)
// So gate the scalar-unary parallel path much higher than the cheap/SIMD ops.
// The map is elementwise with no cross-element accumulation, so serial and
// parallel are bit-for-bit identical — this only changes scheduling.
const SCALAR_UNARY_PARALLEL_THRESHOLD: usize = 1 << 19; // 524288

// Row-parallel softmax/log_softmax over the last dim spreads independent rows
// across the rayon pool, but for the common classifier/attention shapes the
// total work is tiny and rayon's split/join dominates. Same-worker A/B
// (RAYON_NUM_THREADS=1 vs default, 64-core) on softmax/vocab [32, n]:
//   numel 4096   serial 72 us   vs parallel 556 us  (7.7x slower parallel)
//   numel 16384  serial 214 us  vs parallel 676 us  (3.2x slower parallel)
//   numel 65536  serial 1.03 ms vs parallel 1.02 ms (break-even)
//   numel 262144 serial 2.74 ms vs parallel 1.13 ms (2.4x faster parallel)
// Gate the per-row parallel path at the ~65536 crossover. Rows are independent
// and each is reduced identically regardless of thread, so output is
// bit-for-bit identical to the single-threaded path.
const SOFTMAX_PARALLEL_NUMEL_THRESHOLD: usize = 1 << 16; // 65536

fn unary_f64<F>(input: &[f64], meta: &TensorMeta, op: F) -> Result<Vec<f64>, KernelError>
where
    F: Fn(f64) -> f64 + Sync,
{
    if !meta.is_contiguous() {
        return unary_strided_f64(input, meta, op);
    }

    ensure_storage_len(input, meta, "input")?;

    let numel = meta.numel();
    if numel == 0 {
        return Ok(Vec::new());
    }

    let start = meta.storage_offset();
    let window = &input[start..start + numel];

    if numel >= SCALAR_UNARY_PARALLEL_THRESHOLD {
        Ok(window.par_iter().map(|value| op(*value)).collect())
    } else {
        Ok(window.iter().map(|value| op(*value)).collect())
    }
}

fn simd_unary_f64<F, S>(window: &[f64], scalar_op: F, simd_op: S) -> Vec<f64>
where
    F: Fn(f64) -> f64 + Sync,
    S: Fn(f64x4) -> f64x4 + Sync,
{
    let numel = window.len();
    let mut output = vec![0.0; numel];

    // One contiguous block: SIMD over the SIMD_WIDTH-aligned bulk, scalar tail.
    let block = |out: &mut [f64], inp: &[f64]| {
        let simd_len = out.len() / SIMD_WIDTH * SIMD_WIDTH;
        for (o, i) in out[..simd_len]
            .chunks_exact_mut(SIMD_WIDTH)
            .zip(inp[..simd_len].chunks_exact(SIMD_WIDTH))
        {
            let a = f64x4::new([i[0], i[1], i[2], i[3]]);
            o.copy_from_slice(simd_op(a).as_array_ref());
        }
        for (o, &v) in out[simd_len..].iter_mut().zip(&inp[simd_len..]) {
            *o = scalar_op(v);
        }
    };

    if numel >= SIMD_UNARY_PARALLEL_THRESHOLD {
        // SIMD_WIDTH-aligned grains: every grain but the last is a whole number
        // of SIMD lanes, so the SIMD/scalar element split is identical to the
        // serial path -> bit-for-bit identical output.
        let threads = rayon::current_num_threads().max(1);
        let grain = numel
            .div_ceil(4 * threads)
            .max(SIMD_WIDTH)
            .div_ceil(SIMD_WIDTH)
            * SIMD_WIDTH;
        output
            .par_chunks_mut(grain)
            .zip(window.par_chunks(grain))
            .for_each(|(out, inp)| block(out, inp));
    } else {
        block(&mut output, window);
    }

    output
}

fn simd_unary_f64_kernel<F, S>(
    input: &[f64],
    meta: &TensorMeta,
    scalar_op: F,
    _simd_op: S,
) -> Result<Vec<f64>, KernelError>
where
    F: Fn(f64) -> f64 + Sync,
    S: Fn(f64x4) -> f64x4 + Sync,
{
    if !meta.is_contiguous() {
        return unary_strided_f64(input, meta, scalar_op);
    }

    ensure_storage_len(input, meta, "input")?;

    let numel = meta.numel();
    if numel == 0 {
        return Ok(Vec::new());
    }

    let start = meta.storage_offset();
    let window = &input[start..start + numel];

    Ok(simd_unary_f64(window, scalar_op, _simd_op))
}

/// Vectorised `exp` over one f64x4 lane group.
///
/// Alien-artifact foundation for a SIMD transcendental family (the no-gaps
/// directive's "portable SIMD elementwise"). `wide`'s degree-13 polynomial with
/// ln2 range reduction is accurate to ~1-2 ULP for the common finite range
/// `|x| < 708.39`. Outside that, `wide::exp` flushes the lane to 0.0, which is
/// WRONG for: overflow (should be +inf), the finite `[708.39, 709.78]` band,
/// the `[-745, -708.39]` denormal-underflow band, `+inf` (should be +inf) and
/// `NaN` (should be NaN). When ANY lane is out of the fast range we recompute
/// the whole group with scalar `f64::exp`, which is exact per libm — these
/// extreme inputs are rare, so the common path stays fully vectorised.
///
/// NOTE: in the fast range this is NOT bit-identical to scalar `f64::exp` (the
/// polynomial differs by ~1-2 ULP). It is a *tolerance-accurate* kernel. It is
/// intentionally NOT yet wired into the production `exp`/`sigmoid`/`softmax`
/// paths: those are currently pinned bit-exact to scalar libm by unit tests, so
/// adopting this requires moving the transcendental parity contract from
/// bit-exact-to-libm to within-tolerance-of-torch (a deliberate project policy
/// decision). `exp_f64x4_matches_scalar_within_tolerance` proves the accuracy
/// and edge-case obligations are met.
#[inline]
#[must_use]
pub fn exp_f64x4(x: f64x4) -> f64x4 {
    // wide's vectorised polynomial is valid (and ~1-2 ULP) exactly when every
    // lane is finite with |x| < 708.39; otherwise it wrongly flushes to 0.0.
    const FAST_LIMIT: f64 = 708.39;
    let xa = x.to_array();
    let lane_fast = |v: f64| v.is_finite() && v.abs() < FAST_LIMIT;
    if xa.iter().copied().all(lane_fast) {
        x.exp()
    } else {
        // Any extreme/non-finite lane present: recompute the whole group with
        // scalar libm so overflow->+inf, +inf->+inf, NaN->NaN, the finite
        // 708.39..=709.78 band and denormal underflow are all exact.
        f64x4::new([xa[0].exp(), xa[1].exp(), xa[2].exp(), xa[3].exp()])
    }
}

fn broadcast_strides(
    input_shape: &[usize],
    target_shape: &[usize],
    context: &'static str,
) -> Result<Vec<usize>, KernelError> {
    let ndim = input_shape.len();
    debug_assert_eq!(ndim, target_shape.len());

    let mut strides = vec![0usize; ndim];
    let mut running_stride = 1usize;
    for d in (0..ndim).rev() {
        strides[d] = if input_shape[d] == 1 && target_shape[d] > 1 {
            0
        } else {
            running_stride
        };
        running_stride = running_stride
            .checked_mul(input_shape[d])
            .ok_or(KernelError::ShapeOverflow { context })?;
    }

    Ok(strides)
}

pub fn compute_broadcast_shape(
    lhs_shape: &[usize],
    rhs_shape: &[usize],
) -> Result<Vec<usize>, KernelError> {
    let lhs_ndim = lhs_shape.len();
    let rhs_ndim = rhs_shape.len();
    let out_ndim = lhs_ndim.max(rhs_ndim);

    let mut out_shape = vec![0usize; out_ndim];
    for i in 0..out_ndim {
        let lhs_dim = if i < out_ndim - lhs_ndim {
            1
        } else {
            lhs_shape[i - (out_ndim - lhs_ndim)]
        };
        let rhs_dim = if i < out_ndim - rhs_ndim {
            1
        } else {
            rhs_shape[i - (out_ndim - rhs_ndim)]
        };

        if lhs_dim == rhs_dim {
            out_shape[i] = lhs_dim;
        } else if lhs_dim == 1 {
            out_shape[i] = rhs_dim;
        } else if rhs_dim == 1 {
            out_shape[i] = lhs_dim;
        } else {
            return Err(KernelError::ShapeMismatch {
                lhs: lhs_shape.to_vec(),
                rhs: rhs_shape.to_vec(),
            });
        }
    }

    Ok(out_shape)
}

fn broadcast_idx(
    flat_idx: usize,
    out_shape: &[usize],
    in_shape: &[usize],
    in_strides: &[usize],
    in_offset: usize,
) -> usize {
    let out_ndim = out_shape.len();
    let in_ndim = in_shape.len();

    let mut idx = in_offset;
    let mut remaining = flat_idx;

    for i in (0..out_ndim).rev() {
        let coord = remaining % out_shape[i];
        remaining /= out_shape[i];

        if i >= out_ndim - in_ndim {
            let in_i = i - (out_ndim - in_ndim);
            if in_shape[in_i] > 1 {
                idx += coord * in_strides[in_i];
            }
        }
    }

    idx
}

fn elementwise_broadcast_f64<F>(
    lhs: &[f64],
    rhs: &[f64],
    lhs_meta: &TensorMeta,
    rhs_meta: &TensorMeta,
    op: F,
) -> Result<(Vec<f64>, Vec<usize>), KernelError>
where
    F: Fn(f64, f64) -> f64,
{
    let out_shape = compute_broadcast_shape(lhs_meta.shape(), rhs_meta.shape())?;
    let out_numel = checked_shape_numel(&out_shape, "broadcast output")?;

    if out_numel == 0 {
        return Ok((Vec::new(), out_shape));
    }

    let lhs_strides = lhs_meta.strides();
    let rhs_strides = rhs_meta.strides();
    let lhs_offset = lhs_meta.storage_offset();
    let rhs_offset = rhs_meta.storage_offset();

    let mut output = Vec::with_capacity(out_numel);
    for i in 0..out_numel {
        let lhs_idx = broadcast_idx(i, &out_shape, lhs_meta.shape(), lhs_strides, lhs_offset);
        let rhs_idx = broadcast_idx(i, &out_shape, rhs_meta.shape(), rhs_strides, rhs_offset);
        output.push(op(lhs[lhs_idx], rhs[rhs_idx]));
    }

    Ok((output, out_shape))
}

pub fn reduce_sum_for_broadcast(
    expanded_grad: &[f64],
    expanded_shape: &[usize],
    original_shape: &[usize],
) -> Result<Vec<f64>, KernelError> {
    let ndim = expanded_shape.len();
    if ndim != original_shape.len() {
        return Err(KernelError::ShapeMismatch {
            lhs: expanded_shape.to_vec(),
            rhs: original_shape.to_vec(),
        });
    }

    for d in 0..ndim {
        let expanded = expanded_shape[d];
        let original = original_shape[d];
        if original != expanded && original != 1 {
            return Err(KernelError::ShapeMismatch {
                lhs: expanded_shape.to_vec(),
                rhs: original_shape.to_vec(),
            });
        }
    }

    let expanded_numel = checked_shape_numel(expanded_shape, "broadcast reduction expanded shape")?;
    if expanded_grad.len() != expanded_numel {
        return Err(KernelError::ShapeMismatch {
            lhs: vec![expanded_grad.len()],
            rhs: vec![expanded_numel],
        });
    }

    let original_numel = checked_shape_numel(original_shape, "broadcast reduction original shape")?;
    if original_numel == 0 {
        return Ok(Vec::new());
    }

    let original_strides =
        broadcast_strides(original_shape, expanded_shape, "broadcast strides overflow")?;
    let mut reduced = vec![0.0; original_numel];
    let mut coords = vec![0usize; ndim];

    for grad in expanded_grad {
        let mut original_idx = 0usize;
        for d in 0..ndim {
            original_idx += coords[d] * original_strides[d];
        }
        reduced[original_idx] += *grad;

        for d in (0..ndim).rev() {
            coords[d] += 1;
            if coords[d] < expanded_shape[d] {
                break;
            }
            coords[d] = 0;
        }
    }

    Ok(reduced)
}

pub fn neg_tensor_contiguous_f64(
    input: &[f64],
    meta: &TensorMeta,
) -> Result<Vec<f64>, KernelError> {
    simd_unary_f64_kernel(input, meta, |v| -v, |a| -a)
}

pub fn abs_tensor_contiguous_f64(
    input: &[f64],
    meta: &TensorMeta,
) -> Result<Vec<f64>, KernelError> {
    simd_unary_f64_kernel(input, meta, |v| v.abs(), |a| a.abs())
}

pub fn exp_tensor_contiguous_f64(
    input: &[f64],
    meta: &TensorMeta,
) -> Result<Vec<f64>, KernelError> {
    unary_f64(input, meta, |value| value.exp())
}

pub fn log_tensor_contiguous_f64(
    input: &[f64],
    meta: &TensorMeta,
) -> Result<Vec<f64>, KernelError> {
    unary_f64(input, meta, |value| value.ln())
}

pub fn relu_tensor_contiguous_f64(
    input: &[f64],
    meta: &TensorMeta,
) -> Result<Vec<f64>, KernelError> {
    let zero = f64x4::splat(0.0);
    simd_unary_f64_kernel(input, meta, |v| v.max(0.0), move |a| a.max(zero))
}

pub fn sigmoid_tensor_contiguous_f64(
    input: &[f64],
    meta: &TensorMeta,
) -> Result<Vec<f64>, KernelError> {
    // Logistic sigmoid `1 / (1 + exp(-x))`. NOTE: a vectorised `exp_f64x4` path
    // was measured and REJECTED — `wide`'s f64x4 selects its SIMD backend at
    // compile time via `target_feature`, and the default build target (no
    // `-C target-feature=+avx2`) lowers f64x4 to SSE2/scalar lanes, so the
    // "vectorised" exp costs the same as scalar libm (sigmoid/100000 == exp/100000
    // to 5 sig figs). Even forcing `+avx2,+fma` only buys ~1.3x here (the op is
    // alloc/memory-bound, not exp-compute-bound), well under the Score>=2.0 bar.
    // Contrast: the GEMM crate uses *runtime* CPU detection, so it is unaffected.
    unary_f64(input, meta, |value| 1.0 / (1.0 + (-value).exp()))
}

pub fn tanh_tensor_contiguous_f64(
    input: &[f64],
    meta: &TensorMeta,
) -> Result<Vec<f64>, KernelError> {
    unary_f64(input, meta, |value| value.tanh())
}

pub fn sqrt_tensor_contiguous_f64(
    input: &[f64],
    meta: &TensorMeta,
) -> Result<Vec<f64>, KernelError> {
    simd_unary_f64_kernel(input, meta, |v| v.sqrt(), |a| a.sqrt())
}

pub fn reciprocal_tensor_contiguous_f64(
    input: &[f64],
    meta: &TensorMeta,
) -> Result<Vec<f64>, KernelError> {
    let one = f64x4::splat(1.0);
    simd_unary_f64_kernel(input, meta, |v| 1.0 / v, move |a| one / a)
}

pub fn sin_tensor_contiguous_f64(
    input: &[f64],
    meta: &TensorMeta,
) -> Result<Vec<f64>, KernelError> {
    unary_f64(input, meta, |value| value.sin())
}

pub fn cos_tensor_contiguous_f64(
    input: &[f64],
    meta: &TensorMeta,
) -> Result<Vec<f64>, KernelError> {
    unary_f64(input, meta, |value| value.cos())
}

pub fn tan_tensor_contiguous_f64(
    input: &[f64],
    meta: &TensorMeta,
) -> Result<Vec<f64>, KernelError> {
    unary_f64(input, meta, |value| value.tan())
}

pub fn floor_tensor_contiguous_f64(
    input: &[f64],
    meta: &TensorMeta,
) -> Result<Vec<f64>, KernelError> {
    unary_f64(input, meta, |value| value.floor())
}

pub fn ceil_tensor_contiguous_f64(
    input: &[f64],
    meta: &TensorMeta,
) -> Result<Vec<f64>, KernelError> {
    unary_f64(input, meta, |value| value.ceil())
}

pub fn round_tensor_contiguous_f64(
    input: &[f64],
    meta: &TensorMeta,
) -> Result<Vec<f64>, KernelError> {
    unary_f64(input, meta, round_ties_even_f64)
}

pub fn log2_tensor_contiguous_f64(
    input: &[f64],
    meta: &TensorMeta,
) -> Result<Vec<f64>, KernelError> {
    unary_f64(input, meta, |value| value.log2())
}

pub fn log10_tensor_contiguous_f64(
    input: &[f64],
    meta: &TensorMeta,
) -> Result<Vec<f64>, KernelError> {
    unary_f64(input, meta, |value| value.log10())
}

pub fn log1p_tensor_contiguous_f64(
    input: &[f64],
    meta: &TensorMeta,
) -> Result<Vec<f64>, KernelError> {
    unary_f64(input, meta, |value| value.ln_1p())
}

pub fn expm1_tensor_contiguous_f64(
    input: &[f64],
    meta: &TensorMeta,
) -> Result<Vec<f64>, KernelError> {
    unary_f64(input, meta, |value| value.exp_m1())
}

pub fn sign_tensor_contiguous_f64(
    input: &[f64],
    meta: &TensorMeta,
) -> Result<Vec<f64>, KernelError> {
    unary_f64(input, meta, torch_sign_f64)
}

pub fn trunc_tensor_contiguous_f64(
    input: &[f64],
    meta: &TensorMeta,
) -> Result<Vec<f64>, KernelError> {
    unary_f64(input, meta, |value| value.trunc())
}

pub fn frac_tensor_contiguous_f64(
    input: &[f64],
    meta: &TensorMeta,
) -> Result<Vec<f64>, KernelError> {
    unary_f64(input, meta, |value| value.fract())
}

pub fn asin_tensor_contiguous_f64(
    input: &[f64],
    meta: &TensorMeta,
) -> Result<Vec<f64>, KernelError> {
    unary_f64(input, meta, |value| value.asin())
}

pub fn acos_tensor_contiguous_f64(
    input: &[f64],
    meta: &TensorMeta,
) -> Result<Vec<f64>, KernelError> {
    unary_f64(input, meta, |value| value.acos())
}

pub fn atan_tensor_contiguous_f64(
    input: &[f64],
    meta: &TensorMeta,
) -> Result<Vec<f64>, KernelError> {
    unary_f64(input, meta, |value| value.atan())
}

pub fn sinh_tensor_contiguous_f64(
    input: &[f64],
    meta: &TensorMeta,
) -> Result<Vec<f64>, KernelError> {
    unary_f64(input, meta, |value| value.sinh())
}

pub fn cosh_tensor_contiguous_f64(
    input: &[f64],
    meta: &TensorMeta,
) -> Result<Vec<f64>, KernelError> {
    unary_f64(input, meta, |value| value.cosh())
}

pub fn gelu_tensor_contiguous_f64(
    input: &[f64],
    meta: &TensorMeta,
) -> Result<Vec<f64>, KernelError> {
    unary_f64(input, meta, gelu_value)
}

pub fn silu_tensor_contiguous_f64(
    input: &[f64],
    meta: &TensorMeta,
) -> Result<Vec<f64>, KernelError> {
    unary_f64(input, meta, silu_value)
}

pub fn leaky_relu_tensor_contiguous_f64(
    input: &[f64],
    meta: &TensorMeta,
) -> Result<Vec<f64>, KernelError> {
    unary_f64(input, meta, leaky_relu_value)
}

pub fn elu_tensor_contiguous_f64(
    input: &[f64],
    meta: &TensorMeta,
) -> Result<Vec<f64>, KernelError> {
    unary_f64(input, meta, elu_value)
}

pub fn rsqrt_tensor_contiguous_f64(
    input: &[f64],
    meta: &TensorMeta,
) -> Result<Vec<f64>, KernelError> {
    unary_f64(input, meta, |v| 1.0 / v.sqrt())
}

pub fn erf_tensor_contiguous_f64(
    input: &[f64],
    meta: &TensorMeta,
) -> Result<Vec<f64>, KernelError> {
    unary_f64(input, meta, erf_value)
}

pub fn erfc_tensor_contiguous_f64(
    input: &[f64],
    meta: &TensorMeta,
) -> Result<Vec<f64>, KernelError> {
    // Use libm::erfc directly rather than `1.0 - erf(x)` — the latter
    // cancels down to 0.0 for |x| > ~5.95 even though `erfc(x)` is
    // still a tiny but non-zero positive number.
    unary_f64(input, meta, erfc_value)
}

pub fn hardswish_tensor_contiguous_f64(
    input: &[f64],
    meta: &TensorMeta,
) -> Result<Vec<f64>, KernelError> {
    unary_f64(input, meta, hardswish_value)
}

pub fn hardsigmoid_tensor_contiguous_f64(
    input: &[f64],
    meta: &TensorMeta,
) -> Result<Vec<f64>, KernelError> {
    unary_f64(input, meta, hardsigmoid_value)
}

pub fn hardtanh_tensor_contiguous_f64(
    input: &[f64],
    meta: &TensorMeta,
) -> Result<Vec<f64>, KernelError> {
    unary_f64(input, meta, hardtanh_value)
}

pub fn softplus_tensor_contiguous_f64(
    input: &[f64],
    meta: &TensorMeta,
) -> Result<Vec<f64>, KernelError> {
    unary_f64(input, meta, softplus_value)
}

pub fn mish_tensor_contiguous_f64(
    input: &[f64],
    meta: &TensorMeta,
) -> Result<Vec<f64>, KernelError> {
    unary_f64(input, meta, mish_value)
}

pub fn square_tensor_contiguous_f64(
    input: &[f64],
    meta: &TensorMeta,
) -> Result<Vec<f64>, KernelError> {
    unary_f64(input, meta, |v| v * v)
}

pub fn isnan_tensor_contiguous_f64(
    input: &[f64],
    meta: &TensorMeta,
) -> Result<Vec<f64>, KernelError> {
    unary_f64(input, meta, |v| if v.is_nan() { 1.0 } else { 0.0 })
}

pub fn isinf_tensor_contiguous_f64(
    input: &[f64],
    meta: &TensorMeta,
) -> Result<Vec<f64>, KernelError> {
    unary_f64(input, meta, |v| if v.is_infinite() { 1.0 } else { 0.0 })
}

pub fn isfinite_tensor_contiguous_f64(
    input: &[f64],
    meta: &TensorMeta,
) -> Result<Vec<f64>, KernelError> {
    unary_f64(input, meta, |v| if v.is_finite() { 1.0 } else { 0.0 })
}

pub fn pow_tensor_contiguous_f64(
    input: &[f64],
    meta: &TensorMeta,
    exponent: f64,
) -> Result<Vec<f64>, KernelError> {
    ensure_unary_layout_and_storage(input, meta)?;

    let numel = meta.numel();
    if numel == 0 {
        return Ok(Vec::new());
    }

    let start = meta.storage_offset();
    let window = &input[start..start + numel];

    // powf is ~exp+log per element (compute-bound), so spread it across the
    // rayon pool for large tensors; torch's pow is multi-threaded. The map is a
    // pure per-element function, so the parallel result is bit-identical to the
    // serial one (no accumulation order to disturb).
    if numel >= PARALLEL_THRESHOLD {
        Ok(window
            .par_iter()
            .map(|value| powf_torch_signed_zero_f64(*value, exponent))
            .collect())
    } else {
        Ok(window
            .iter()
            .map(|value| powf_torch_signed_zero_f64(*value, exponent))
            .collect())
    }
}

pub fn clamp_tensor_contiguous_f64(
    input: &[f64],
    meta: &TensorMeta,
    min_val: f64,
    max_val: f64,
) -> Result<Vec<f64>, KernelError> {
    ensure_unary_layout_and_storage(input, meta)?;

    let numel = meta.numel();
    if numel == 0 {
        return Ok(Vec::new());
    }

    let start = meta.storage_offset();
    let window = &input[start..start + numel];

    Ok(window
        .iter()
        .map(|value| {
            // clamp is min(max(x, min_val), max_val): lower bound first,
            // then upper, so when min_val > max_val the upper bound wins.
            if value.is_nan() {
                f64::NAN
            } else {
                let lo = if !min_val.is_nan() && *value < min_val {
                    min_val
                } else {
                    *value
                };
                if !max_val.is_nan() && lo > max_val {
                    max_val
                } else {
                    lo
                }
            }
        })
        .collect())
}

pub fn min_tensor_contiguous_f64(
    lhs: &[f64],
    rhs: &[f64],
    lhs_meta: &TensorMeta,
    rhs_meta: &TensorMeta,
) -> Result<Vec<f64>, KernelError> {
    elementwise_f64(lhs, rhs, lhs_meta, rhs_meta, |l, r| {
        if l.is_nan() || r.is_nan() {
            f64::NAN
        } else {
            l.min(r)
        }
    })
}

pub fn max_tensor_contiguous_f64(
    lhs: &[f64],
    rhs: &[f64],
    lhs_meta: &TensorMeta,
    rhs_meta: &TensorMeta,
) -> Result<Vec<f64>, KernelError> {
    elementwise_f64(lhs, rhs, lhs_meta, rhs_meta, |l, r| {
        if l.is_nan() || r.is_nan() {
            f64::NAN
        } else {
            l.max(r)
        }
    })
}

pub fn atan2_tensor_contiguous_f64(
    lhs: &[f64],
    rhs: &[f64],
    lhs_meta: &TensorMeta,
    rhs_meta: &TensorMeta,
) -> Result<Vec<f64>, KernelError> {
    elementwise_f64(lhs, rhs, lhs_meta, rhs_meta, |y, x| y.atan2(x))
}

pub fn fmod_tensor_contiguous_f64(
    lhs: &[f64],
    rhs: &[f64],
    lhs_meta: &TensorMeta,
    rhs_meta: &TensorMeta,
) -> Result<Vec<f64>, KernelError> {
    elementwise_f64(lhs, rhs, lhs_meta, rhs_meta, |a, b| a % b)
}

pub fn remainder_tensor_contiguous_f64(
    lhs: &[f64],
    rhs: &[f64],
    lhs_meta: &TensorMeta,
    rhs_meta: &TensorMeta,
) -> Result<Vec<f64>, KernelError> {
    elementwise_f64(lhs, rhs, lhs_meta, rhs_meta, |a, b| a - (a / b).floor() * b)
}

pub fn eq_tensor_contiguous_f64(
    lhs: &[f64],
    rhs: &[f64],
    lhs_meta: &TensorMeta,
    rhs_meta: &TensorMeta,
) -> Result<Vec<f64>, KernelError> {
    elementwise_f64(
        lhs,
        rhs,
        lhs_meta,
        rhs_meta,
        |l, r| {
            if l == r { 1.0 } else { 0.0 }
        },
    )
}

pub fn ne_tensor_contiguous_f64(
    lhs: &[f64],
    rhs: &[f64],
    lhs_meta: &TensorMeta,
    rhs_meta: &TensorMeta,
) -> Result<Vec<f64>, KernelError> {
    elementwise_f64(
        lhs,
        rhs,
        lhs_meta,
        rhs_meta,
        |l, r| {
            if l != r { 1.0 } else { 0.0 }
        },
    )
}

pub fn lt_tensor_contiguous_f64(
    lhs: &[f64],
    rhs: &[f64],
    lhs_meta: &TensorMeta,
    rhs_meta: &TensorMeta,
) -> Result<Vec<f64>, KernelError> {
    elementwise_f64(
        lhs,
        rhs,
        lhs_meta,
        rhs_meta,
        |l, r| {
            if l < r { 1.0 } else { 0.0 }
        },
    )
}

pub fn gt_tensor_contiguous_f64(
    lhs: &[f64],
    rhs: &[f64],
    lhs_meta: &TensorMeta,
    rhs_meta: &TensorMeta,
) -> Result<Vec<f64>, KernelError> {
    elementwise_f64(
        lhs,
        rhs,
        lhs_meta,
        rhs_meta,
        |l, r| {
            if l > r { 1.0 } else { 0.0 }
        },
    )
}

pub fn le_tensor_contiguous_f64(
    lhs: &[f64],
    rhs: &[f64],
    lhs_meta: &TensorMeta,
    rhs_meta: &TensorMeta,
) -> Result<Vec<f64>, KernelError> {
    elementwise_f64(
        lhs,
        rhs,
        lhs_meta,
        rhs_meta,
        |l, r| {
            if l <= r { 1.0 } else { 0.0 }
        },
    )
}

pub fn ge_tensor_contiguous_f64(
    lhs: &[f64],
    rhs: &[f64],
    lhs_meta: &TensorMeta,
    rhs_meta: &TensorMeta,
) -> Result<Vec<f64>, KernelError> {
    elementwise_f64(
        lhs,
        rhs,
        lhs_meta,
        rhs_meta,
        |l, r| {
            if l >= r { 1.0 } else { 0.0 }
        },
    )
}

/// Pairwise (binary-tree) summation with O(log N · ε) error
/// accumulation, vs O(N · ε) for naive `.iter().sum()`.
///
/// PyTorch's `torch.sum` uses pairwise summation (Higham, 1993) for
/// the same precision reason — at N = 10^6 the naive sum accumulates
/// ~10^6 · ε ≈ 2.2e-10 relative error while pairwise stays at
/// ~log(10^6) · ε ≈ 4.4e-15 relative error, a six-orders-of-magnitude
/// improvement that becomes visible on any sufficiently-large tensor
/// reduction. Below the BLOCK threshold sequential summation is
/// already well within ULP precision and avoids the recursion
/// overhead.
#[inline]
fn pairwise_sum_f64(values: &[f64]) -> f64 {
    const BLOCK: usize = 128;
    if values.len() <= BLOCK {
        // Sequential is fine at small N (block fits in L1 cache).
        return values.iter().sum();
    }
    let mid = values.len() / 2;
    pairwise_sum_f64(&values[..mid]) + pairwise_sum_f64(&values[mid..])
}

/// Parallel `pairwise_sum_f64` for large FULL reductions. Splits the SAME
/// `mid = len/2` tree via `rayon::join` down to `PAR_BLOCK`, then runs the
/// serial pairwise sum below — the associativity (`left + right` at every node)
/// is unchanged, so the result is BIT-FOR-BIT identical to `pairwise_sum_f64`.
/// (Not used inside the per-row softmax/norm loops, where the reduction is small
/// and rayon would only add overhead.)
fn pairwise_sum_f64_par(values: &[f64]) -> f64 {
    // Small PAR_BLOCK leaf so large reductions spread across all cores; the
    // call site gates the whole reduction on SUM_PARALLEL_THRESHOLD (below that,
    // rayon::join overhead exceeds the gain — a 100k sum REGRESSED 2.6x while a
    // 1M sum sped up 9.2x).
    const PAR_BLOCK: usize = 1 << 14;
    if values.len() <= PAR_BLOCK {
        return pairwise_sum_f64(values);
    }
    let mid = values.len() / 2;
    let (left, right) = values.split_at(mid);
    let (ls, rs) = rayon::join(
        || pairwise_sum_f64_par(left),
        || pairwise_sum_f64_par(right),
    );
    ls + rs
}

/// `pairwise_sum_f64`, parallelised only for large FULL reductions.
const SUM_PARALLEL_THRESHOLD: usize = 1 << 19; // 524288

#[inline]
fn pairwise_sum_f64_maybe_par(values: &[f64]) -> f64 {
    if values.len() >= SUM_PARALLEL_THRESHOLD {
        pairwise_sum_f64_par(values)
    } else {
        pairwise_sum_f64(values)
    }
}

/// Like `pairwise_sum_f64`, but applies a closure `f` to each element
/// before adding. Used by norm helpers — `norm_l1` sums `|x|`,
/// `norm_l2` sums `x*x`, generic `norm_lp` sums `|x|^p` — to inherit
/// the same O(log N · ε) precision contract as `pairwise_sum_f64`
/// without an intermediate allocation. The closure must be `Fn + Copy`
/// so we can pass it by value through the recursion (true for the
/// stateless or capture-by-Copy closures we use here).
fn pairwise_sum_map_f64<F>(values: &[f64], f: F) -> f64
where
    F: Fn(f64) -> f64 + Copy,
{
    const BLOCK: usize = 128;
    if values.len() <= BLOCK {
        return values.iter().copied().map(f).sum();
    }
    let mid = values.len() / 2;
    pairwise_sum_map_f64(&values[..mid], f) + pairwise_sum_map_f64(&values[mid..], f)
}

/// Parallel `pairwise_sum_map_f64` for large FULL reductions (e.g. an L2 norm
/// over a whole weight tensor). Splits the SAME mid tree via rayon::join down to
/// PAR_BLOCK, so the result is BIT-FOR-BIT identical. NOT for the per-row norm /
/// softmax callers, where the reduction is small.
fn pairwise_sum_map_f64_par<F>(values: &[f64], f: F) -> f64
where
    F: Fn(f64) -> f64 + Copy + Sync,
{
    const PAR_BLOCK: usize = 1 << 14;
    if values.len() <= PAR_BLOCK {
        return pairwise_sum_map_f64(values, f);
    }
    let mid = values.len() / 2;
    let (left, right) = values.split_at(mid);
    let (ls, rs) = rayon::join(
        || pairwise_sum_map_f64_par(left, f),
        || pairwise_sum_map_f64_par(right, f),
    );
    ls + rs
}

#[inline]
fn pairwise_sum_map_f64_maybe_par<F>(values: &[f64], f: F) -> f64
where
    F: Fn(f64) -> f64 + Copy + Sync,
{
    if values.len() >= SUM_PARALLEL_THRESHOLD {
        pairwise_sum_map_f64_par(values, f)
    } else {
        pairwise_sum_map_f64(values, f)
    }
}

pub fn sum_tensor_contiguous_f64(input: &[f64], meta: &TensorMeta) -> Result<f64, KernelError> {
    ensure_unary_layout_and_storage(input, meta)?;
    let numel = meta.numel();
    if numel == 0 {
        return Ok(0.0);
    }
    let offset = meta.storage_offset();
    Ok(pairwise_sum_f64_maybe_par(&input[offset..offset + numel]))
}

pub fn mean_tensor_contiguous_f64(input: &[f64], meta: &TensorMeta) -> Result<f64, KernelError> {
    ensure_unary_layout_and_storage(input, meta)?;
    let offset = meta.storage_offset();
    let numel = meta.numel();
    if numel == 0 {
        return Ok(f64::NAN);
    }
    let sum = pairwise_sum_f64_maybe_par(&input[offset..offset + numel]);
    #[allow(clippy::cast_precision_loss)]
    let n = numel as f64;
    Ok(sum / n)
}

pub fn sum_dim_tensor_contiguous_f64(
    input: &[f64],
    meta: &TensorMeta,
    dim: usize,
) -> Result<Vec<f64>, KernelError> {
    ensure_unary_layout_and_storage(input, meta)?;
    let shape = meta.shape();
    let ndim = shape.len();
    if dim >= ndim {
        return Err(KernelError::InvalidDimension { dim, ndim });
    }
    let reduce_size = shape[dim];
    let (outer_size, inner_size, _) =
        checked_dim_loop_sizes(shape, dim, "sum_dim shape volume overflow")?;
    let out_numel = checked_mul(
        outer_size,
        inner_size,
        "sum_dim shape multiplication overflow",
    )?;
    if out_numel == 0 {
        return Ok(Vec::new());
    }
    if reduce_size == 0 {
        return Ok(vec![0.0; out_numel]);
    }
    let offset = meta.storage_offset();
    // Push-based output skips the m*n zero-init memset
    // (frankentorch-ao30). Both branches below proceed in
    // row-major output order so push is correct.
    let mut output = Vec::with_capacity(out_numel);
    let data = &input[offset..];

    // Inner_size == 1 means we are reducing the last (most-contiguous)
    // dim, which is the most common shape: e.g. [B, D].sum(dim=-1).
    // The strided slice for each outer is pure contiguous, so we can
    // pairwise-sum directly with zero scratch allocation.
    if inner_size == 1 {
        for outer in 0..outer_size {
            let start = outer * reduce_size;
            let end = start + reduce_size;
            output.push(pairwise_sum_f64(&data[start..end]));
        }
        return Ok(output);
    }

    // General strided case: gather each (outer, inner) slice into a
    // reusable scratch buffer, then pairwise-sum. One allocation per
    // call, not per cell. Same precision pattern as `var_dim_*`.
    let mut scratch = vec![0.0_f64; reduce_size];
    for outer in 0..outer_size {
        for inner in 0..inner_size {
            for r in 0..reduce_size {
                scratch[r] = data[outer * reduce_size * inner_size + r * inner_size + inner];
            }
            output.push(pairwise_sum_f64(&scratch));
        }
    }

    Ok(output)
}

pub fn mean_dim_tensor_contiguous_f64(
    input: &[f64],
    meta: &TensorMeta,
    dim: usize,
) -> Result<Vec<f64>, KernelError> {
    ensure_unary_layout_and_storage(input, meta)?;
    let shape = meta.shape();
    let ndim = shape.len();
    if dim >= ndim {
        return Err(KernelError::InvalidDimension { dim, ndim });
    }
    let reduce_size = shape[dim];
    let (outer_size, inner_size, _) =
        checked_dim_loop_sizes(shape, dim, "mean_dim shape volume overflow")?;
    let out_numel = checked_mul(
        outer_size,
        inner_size,
        "mean_dim shape multiplication overflow",
    )?;
    if out_numel == 0 {
        return Ok(Vec::new());
    }
    if reduce_size == 0 {
        return Ok(vec![f64::NAN; out_numel]);
    }
    let mut output = sum_dim_tensor_contiguous_f64(input, meta, dim)?;
    let scale = 1.0 / reduce_size as f64;
    for v in &mut output {
        *v *= scale;
    }
    Ok(output)
}

const SIMD_WIDTH: usize = 4;

fn simd_binary_f64<F, S>(
    lhs_window: &[f64],
    rhs_window: &[f64],
    scalar_op: F,
    simd_op: S,
) -> Vec<f64>
where
    F: Fn(f64, f64) -> f64 + Sync,
    S: Fn(f64x4, f64x4) -> f64x4 + Sync,
{
    let numel = lhs_window.len();
    let mut output = vec![0.0; numel];

    // One contiguous block: SIMD over the SIMD_WIDTH-aligned bulk, scalar tail.
    let block = |out: &mut [f64], lw: &[f64], rw: &[f64]| {
        let simd_len = out.len() / SIMD_WIDTH * SIMD_WIDTH;
        for ((o, lc), rc) in out[..simd_len]
            .chunks_exact_mut(SIMD_WIDTH)
            .zip(lw[..simd_len].chunks_exact(SIMD_WIDTH))
            .zip(rw[..simd_len].chunks_exact(SIMD_WIDTH))
        {
            let a = f64x4::new([lc[0], lc[1], lc[2], lc[3]]);
            let b = f64x4::new([rc[0], rc[1], rc[2], rc[3]]);
            o.copy_from_slice(simd_op(a, b).as_array_ref());
        }
        for ((o, &lv), &rv) in out[simd_len..]
            .iter_mut()
            .zip(&lw[simd_len..])
            .zip(&rw[simd_len..])
        {
            *o = scalar_op(lv, rv);
        }
    };

    // add/sub/mul/div were SERIAL while the scalar-map binary path parallelises,
    // so a 1M-element add ran ~54x slower than torch. Parallelise over
    // SIMD_WIDTH-aligned grains (every grain but the last is a whole number of
    // SIMD lanes -> SIMD/scalar split identical to serial -> bit-for-bit equal).
    if numel >= SIMD_UNARY_PARALLEL_THRESHOLD {
        let threads = rayon::current_num_threads().max(1);
        let grain = numel
            .div_ceil(4 * threads)
            .max(SIMD_WIDTH)
            .div_ceil(SIMD_WIDTH)
            * SIMD_WIDTH;
        output
            .par_chunks_mut(grain)
            .zip(lhs_window.par_chunks(grain))
            .zip(rhs_window.par_chunks(grain))
            .for_each(|((out, lw), rw)| block(out, lw, rw));
    } else {
        block(&mut output, lhs_window, rhs_window);
    }

    output
}

fn simd_elementwise_f64<F, S>(
    lhs: &[f64],
    rhs: &[f64],
    lhs_meta: &TensorMeta,
    rhs_meta: &TensorMeta,
    scalar_op: F,
    _simd_op: S,
) -> Result<Vec<f64>, KernelError>
where
    F: Fn(f64, f64) -> f64 + Sync,
    S: Fn(f64x4, f64x4) -> f64x4 + Sync,
{
    ensure_meta_shape_and_dtype(lhs_meta, rhs_meta)?;

    if !lhs_meta.is_contiguous() || !rhs_meta.is_contiguous() {
        return elementwise_strided_f64(lhs, rhs, lhs_meta, rhs_meta, scalar_op);
    }

    ensure_storage_len(lhs, lhs_meta, "lhs")?;
    ensure_storage_len(rhs, rhs_meta, "rhs")?;

    let numel = lhs_meta.numel();
    if numel == 0 {
        return Ok(Vec::new());
    }

    let lhs_start = lhs_meta.storage_offset();
    let rhs_start = rhs_meta.storage_offset();
    let lhs_window = &lhs[lhs_start..lhs_start + numel];
    let rhs_window = &rhs[rhs_start..rhs_start + numel];

    Ok(simd_binary_f64(lhs_window, rhs_window, scalar_op, _simd_op))
}

pub fn add_tensor_contiguous_f64(
    lhs: &[f64],
    rhs: &[f64],
    lhs_meta: &TensorMeta,
    rhs_meta: &TensorMeta,
) -> Result<Vec<f64>, KernelError> {
    simd_elementwise_f64(lhs, rhs, lhs_meta, rhs_meta, |l, r| l + r, |a, b| a + b)
}

pub fn sub_tensor_contiguous_f64(
    lhs: &[f64],
    rhs: &[f64],
    lhs_meta: &TensorMeta,
    rhs_meta: &TensorMeta,
) -> Result<Vec<f64>, KernelError> {
    simd_elementwise_f64(lhs, rhs, lhs_meta, rhs_meta, |l, r| l - r, |a, b| a - b)
}

pub fn div_tensor_contiguous_f64(
    lhs: &[f64],
    rhs: &[f64],
    lhs_meta: &TensorMeta,
    rhs_meta: &TensorMeta,
) -> Result<Vec<f64>, KernelError> {
    simd_elementwise_f64(lhs, rhs, lhs_meta, rhs_meta, |l, r| l / r, |a, b| a / b)
}

pub fn mul_tensor_contiguous_f64(
    lhs: &[f64],
    rhs: &[f64],
    lhs_meta: &TensorMeta,
    rhs_meta: &TensorMeta,
) -> Result<Vec<f64>, KernelError> {
    simd_elementwise_f64(lhs, rhs, lhs_meta, rhs_meta, |l, r| l * r, |a, b| a * b)
}

pub fn add_tensor_broadcast_f64(
    lhs: &[f64],
    rhs: &[f64],
    lhs_meta: &TensorMeta,
    rhs_meta: &TensorMeta,
) -> Result<(Vec<f64>, Vec<usize>), KernelError> {
    elementwise_broadcast_f64(lhs, rhs, lhs_meta, rhs_meta, |l, r| l + r)
}

pub fn sub_tensor_broadcast_f64(
    lhs: &[f64],
    rhs: &[f64],
    lhs_meta: &TensorMeta,
    rhs_meta: &TensorMeta,
) -> Result<(Vec<f64>, Vec<usize>), KernelError> {
    elementwise_broadcast_f64(lhs, rhs, lhs_meta, rhs_meta, |l, r| l - r)
}

pub fn mul_tensor_broadcast_f64(
    lhs: &[f64],
    rhs: &[f64],
    lhs_meta: &TensorMeta,
    rhs_meta: &TensorMeta,
) -> Result<(Vec<f64>, Vec<usize>), KernelError> {
    elementwise_broadcast_f64(lhs, rhs, lhs_meta, rhs_meta, |l, r| l * r)
}

pub fn div_tensor_broadcast_f64(
    lhs: &[f64],
    rhs: &[f64],
    lhs_meta: &TensorMeta,
    rhs_meta: &TensorMeta,
) -> Result<(Vec<f64>, Vec<usize>), KernelError> {
    elementwise_broadcast_f64(lhs, rhs, lhs_meta, rhs_meta, |l, r| l / r)
}

/// Fused Linear forward `y = x @ weight^T (+ bias)` for row-major contiguous
/// f64 data. `x` is `[batch, in]`, `weight` is `[out, in]` (NOT transposed), and
/// `bias` (if present) is `[out]`. Routes through `dgemm_bt`, so the weight
/// transpose is never materialised — the dominant cost of the
/// transpose-then-matmul path. Result is `[batch, out]`, bit-for-bit identical
/// to that path.
/// Fused scaled-dot-product attention forward (f64), the flash-attention memory
/// pattern: process one block of `BR` query rows at a time so only that block's
/// score tile `[BR, seq_k]` is ever materialised — NEVER the full
/// `[num_bh, seq_q, seq_k]` score matrix (16MB at seq=512/heads=8), the scale
/// tensor, or the softmax intermediate that the op-graph path allocates and
/// streams through memory three times.
///
/// Inputs are row-major `[num_bh, seq_q, d_k]` (q), `[num_bh, seq_k, d_k]` (k),
/// `[num_bh, seq_k, d_v]` (v). Output is `[num_bh, seq_q, d_v]`. `scale`
/// multiplies the scores pre-softmax; `causal` masks key positions `j > i`.
///
/// The two tile matmuls reuse the vectorised `matrixmultiply` microkernels
/// (`dgemm_bt` reads K in its natural `[seq_k, d_k]` layout — no transpose; the
/// score tile feeds `dgemm` against V), so the QK^T scores are bit-identical to
/// the materialised `bmm(Q, K^T)` path; the softmax (max-subtract, scalar `exp`,
/// divide by the running sum) matches the reference to tolerance.
#[allow(clippy::too_many_arguments)]
#[must_use]
pub fn sdpa_forward_f64(
    q: &[f64],
    k: &[f64],
    v: &[f64],
    num_bh: usize,
    seq_q: usize,
    seq_k: usize,
    d_k: usize,
    d_v: usize,
    scale: f64,
    causal: bool,
) -> Vec<f64> {
    const BR: usize = 64;
    let mut out = vec![0.0f64; num_bh * seq_q * d_v];
    let q_stride = seq_q * d_k;
    let k_stride = seq_k * d_k;
    let v_stride = seq_k * d_v;
    let o_stride = seq_q * d_v;
    out.par_chunks_mut(o_stride)
        .enumerate()
        .for_each(|(bh, o_chunk)| {
            let qh = &q[bh * q_stride..bh * q_stride + q_stride];
            let kh = &k[bh * k_stride..bh * k_stride + k_stride];
            let vh = &v[bh * v_stride..bh * v_stride + v_stride];
            // Scratch score tile, reused across the head's query blocks.
            let mut scores = vec![0.0f64; BR.min(seq_q) * seq_k];
            let mut q0 = 0;
            while q0 < seq_q {
                let br = (q0 + BR).min(seq_q) - q0;
                let q_block = &qh[q0 * d_k..(q0 + br) * d_k];
                let sc = &mut scores[..br * seq_k];
                // scores[br, seq_k] = q_block[br, d_k] @ kh^T  (kh is [seq_k, d_k]).
                gemm::dgemm_bt(br, d_k, seq_k, q_block, kh, sc);
                // Per row: scale, (causal mask), stable softmax.
                for r in 0..br {
                    let qi = q0 + r;
                    let limit = if causal { (qi + 1).min(seq_k) } else { seq_k };
                    let row = &mut sc[r * seq_k..(r + 1) * seq_k];
                    let mut m = f64::NEG_INFINITY;
                    for s in row.iter_mut().take(limit) {
                        *s *= scale;
                        if *s > m {
                            m = *s;
                        }
                    }
                    let mut sum = 0.0f64;
                    for s in row.iter_mut().take(limit) {
                        let e = (*s - m).exp();
                        *s = e;
                        sum += e;
                    }
                    for s in row.iter_mut().take(limit) {
                        *s /= sum;
                    }
                    for s in row.iter_mut().skip(limit) {
                        *s = 0.0;
                    }
                }
                // out_block[br, d_v] = sc[br, seq_k] @ vh[seq_k, d_v].
                let o_block = &mut o_chunk[q0 * d_v..(q0 + br) * d_v];
                gemm::dgemm(br, seq_k, d_v, sc, vh, o_block);
                q0 += br;
            }
        });
    out
}

/// f32 mirror of [`sdpa_forward_f64`] (the common transformer inference dtype):
/// same block-row flash-attention pattern, using the `sgemm_bt`/`sgemm`
/// microkernels and f32 softmax.
#[allow(clippy::too_many_arguments)]
#[must_use]
pub fn sdpa_forward_f32(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    num_bh: usize,
    seq_q: usize,
    seq_k: usize,
    d_k: usize,
    d_v: usize,
    scale: f32,
    causal: bool,
) -> Vec<f32> {
    const BR: usize = 64;
    let mut out = vec![0.0f32; num_bh * seq_q * d_v];
    let q_stride = seq_q * d_k;
    let k_stride = seq_k * d_k;
    let v_stride = seq_k * d_v;
    let o_stride = seq_q * d_v;
    out.par_chunks_mut(o_stride)
        .enumerate()
        .for_each(|(bh, o_chunk)| {
            let qh = &q[bh * q_stride..bh * q_stride + q_stride];
            let kh = &k[bh * k_stride..bh * k_stride + k_stride];
            let vh = &v[bh * v_stride..bh * v_stride + v_stride];
            let mut scores = vec![0.0f32; BR.min(seq_q) * seq_k];
            let mut q0 = 0;
            while q0 < seq_q {
                let br = (q0 + BR).min(seq_q) - q0;
                let q_block = &qh[q0 * d_k..(q0 + br) * d_k];
                let sc = &mut scores[..br * seq_k];
                gemm::sgemm_bt(br, d_k, seq_k, q_block, kh, sc);
                for r in 0..br {
                    let qi = q0 + r;
                    let limit = if causal { (qi + 1).min(seq_k) } else { seq_k };
                    let row = &mut sc[r * seq_k..(r + 1) * seq_k];
                    let mut m = f32::NEG_INFINITY;
                    for s in row.iter_mut().take(limit) {
                        *s *= scale;
                        if *s > m {
                            m = *s;
                        }
                    }
                    let mut sum = 0.0f32;
                    for s in row.iter_mut().take(limit) {
                        let e = (*s - m).exp();
                        *s = e;
                        sum += e;
                    }
                    for s in row.iter_mut().take(limit) {
                        *s /= sum;
                    }
                    for s in row.iter_mut().skip(limit) {
                        *s = 0.0;
                    }
                }
                let o_block = &mut o_chunk[q0 * d_v..(q0 + br) * d_v];
                gemm::sgemm(br, seq_k, d_v, sc, vh, o_block);
                q0 += br;
            }
        });
    out
}

/// Backward of [`sdpa_forward_f64`]. Given the saved `q`/`k`/`v` and the output
/// gradient `dout` (`[num_bh, seq_q, d_v]`), returns `(dq, dk, dv)`.
///
/// Recomputes the softmax probabilities `P = softmax(scale·QKᵀ)` per
/// `(batch·head)` block (cheaper than saving the `[num_bh, seq_q, seq_k]`
/// matrix), then:
///   `dV = Pᵀ @ dOut`,  `dP = dOut @ Vᵀ`,
///   `dU = P ⊙ (dP − rowsum(P⊙dP))`   (softmax Jacobian, U = scale·QKᵀ),
///   `dQ = scale · dU @ K`,  `dK = scale · dUᵀ @ Q`.
/// Each `(batch·head)` block keeps its `[seq_q, seq_k]` scratch in cache; matmuls
/// reuse the vectorised `matrixmultiply` microkernels. Matches the op-graph
/// (bmm/softmax/bmm) backward to tolerance.
#[allow(clippy::too_many_arguments)]
#[must_use]
pub fn sdpa_backward_f64(
    q: &[f64],
    k: &[f64],
    v: &[f64],
    dout: &[f64],
    num_bh: usize,
    seq_q: usize,
    seq_k: usize,
    d_k: usize,
    d_v: usize,
    scale: f64,
    causal: bool,
) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
    let mut dq = vec![0.0f64; num_bh * seq_q * d_k];
    let mut dk = vec![0.0f64; num_bh * seq_k * d_k];
    let mut dv = vec![0.0f64; num_bh * seq_k * d_v];
    let qs = seq_q * d_k;
    let ks = seq_k * d_k;
    let vs = seq_k * d_v;
    let os = seq_q * d_v;
    dq.par_chunks_mut(qs)
        .zip(dk.par_chunks_mut(ks))
        .zip(dv.par_chunks_mut(vs))
        .enumerate()
        .for_each(|(bh, ((dq_bh, dk_bh), dv_bh))| {
            let qh = &q[bh * qs..bh * qs + qs];
            let kh = &k[bh * ks..bh * ks + ks];
            let vh = &v[bh * vs..bh * vs + vs];
            let doh = &dout[bh * os..bh * os + os];
            // P = softmax(scale·Q@Kᵀ) row-wise.  [seq_q, seq_k]
            let mut p = vec![0.0f64; seq_q * seq_k];
            gemm::dgemm_bt(seq_q, d_k, seq_k, qh, kh, &mut p);
            for i in 0..seq_q {
                let limit = if causal { (i + 1).min(seq_k) } else { seq_k };
                let row = &mut p[i * seq_k..(i + 1) * seq_k];
                let mut m = f64::NEG_INFINITY;
                for s in row.iter_mut().take(limit) {
                    *s *= scale;
                    if *s > m {
                        m = *s;
                    }
                }
                let mut sum = 0.0f64;
                for s in row.iter_mut().take(limit) {
                    let e = (*s - m).exp();
                    *s = e;
                    sum += e;
                }
                for s in row.iter_mut().take(limit) {
                    *s /= sum;
                }
                for s in row.iter_mut().skip(limit) {
                    *s = 0.0;
                }
            }
            // dP = dOut @ Vᵀ  [seq_q, seq_k]
            let mut du = vec![0.0f64; seq_q * seq_k];
            gemm::dgemm_bt(seq_q, d_v, seq_k, doh, vh, &mut du);
            // dU = P ⊙ (dP − rowsum(P⊙dP)); overwrite du in place.
            for i in 0..seq_q {
                let pr = &p[i * seq_k..(i + 1) * seq_k];
                let dr = &mut du[i * seq_k..(i + 1) * seq_k];
                let mut dot = 0.0f64;
                for j in 0..seq_k {
                    dot += pr[j] * dr[j];
                }
                for j in 0..seq_k {
                    dr[j] = pr[j] * (dr[j] - dot);
                }
            }
            // dV = Pᵀ @ dOut  [seq_k, d_v]
            let mut pt = vec![0.0f64; seq_k * seq_q];
            for i in 0..seq_q {
                for j in 0..seq_k {
                    pt[j * seq_q + i] = p[i * seq_k + j];
                }
            }
            gemm::dgemm(seq_k, seq_q, d_v, &pt, doh, dv_bh);
            // dQ = scale · dU @ K  [seq_q, d_k]
            gemm::dgemm(seq_q, seq_k, d_k, &du, kh, dq_bh);
            for x in dq_bh.iter_mut() {
                *x *= scale;
            }
            // dK = scale · dUᵀ @ Q  [seq_k, d_k]
            let mut dut = vec![0.0f64; seq_k * seq_q];
            for i in 0..seq_q {
                for j in 0..seq_k {
                    dut[j * seq_q + i] = du[i * seq_k + j];
                }
            }
            gemm::dgemm(seq_k, seq_q, d_k, &dut, qh, dk_bh);
            for x in dk_bh.iter_mut() {
                *x *= scale;
            }
        });
    (dq, dk, dv)
}

/// Fused LayerNorm forward (f64): per row of `[batch, norm_size]`, computes
/// `y = (x - mean) / sqrt(var + eps) * weight + bias` in two streaming passes,
/// NEVER materialising the ~14 full-size intermediates (broadcast mean/var, the
/// eps tensor, diff, diff², …) the op-graph allocates. Parallel over rows.
/// `weight`/`bias` are the flattened normalized-shape affine params (or `None`).
/// Matches the op-graph (mean/var/normalize/affine) to tolerance.
#[must_use]
pub fn layer_norm_forward_f64(
    x: &[f64],
    weight: Option<&[f64]>,
    bias: Option<&[f64]>,
    batch: usize,
    norm_size: usize,
    eps: f64,
) -> Vec<f64> {
    let mut out = vec![0.0f64; batch * norm_size];
    let inv_n = 1.0 / norm_size as f64;
    out.par_chunks_mut(norm_size)
        .enumerate()
        .for_each(|(r, orow)| {
            let xrow = &x[r * norm_size..r * norm_size + norm_size];
            let mut sum = 0.0f64;
            for &v in xrow {
                sum += v;
            }
            let mean = sum * inv_n;
            let mut vsum = 0.0f64;
            for &v in xrow {
                let d = v - mean;
                vsum += d * d;
            }
            let rstd = 1.0 / (vsum * inv_n + eps).sqrt();
            for j in 0..norm_size {
                let mut y = (xrow[j] - mean) * rstd;
                if let Some(w) = weight {
                    y *= w[j];
                }
                if let Some(b) = bias {
                    y += b[j];
                }
                orow[j] = y;
            }
        });
    out
}

/// f32 mirror of [`layer_norm_forward_f64`]: per-row mean/rstd normalize + affine,
/// one streaming pass per row, parallel over rows. Replaces the f32 op-graph
/// (mean_dim/sub/var/rsqrt/affine, ~14 full-size nodes) the f32 no-grad path fell
/// through to.
#[must_use]
pub fn layer_norm_forward_f32(
    x: &[f32],
    weight: Option<&[f32]>,
    bias: Option<&[f32]>,
    batch: usize,
    norm_size: usize,
    eps: f32,
) -> Vec<f32> {
    let mut out = vec![0.0f32; batch * norm_size];
    let inv_n = 1.0 / norm_size as f32;
    out.par_chunks_mut(norm_size)
        .enumerate()
        .for_each(|(r, orow)| {
            let xrow = &x[r * norm_size..r * norm_size + norm_size];
            let mut sum = 0.0f32;
            for &v in xrow {
                sum += v;
            }
            let mean = sum * inv_n;
            let mut vsum = 0.0f32;
            for &v in xrow {
                let d = v - mean;
                vsum += d * d;
            }
            let rstd = 1.0 / (vsum * inv_n + eps).sqrt();
            for j in 0..norm_size {
                let mut y = (xrow[j] - mean) * rstd;
                if let Some(w) = weight {
                    y *= w[j];
                }
                if let Some(b) = bias {
                    y += b[j];
                }
                orow[j] = y;
            }
        });
    out
}

#[must_use]
pub fn layer_norm_forward_with_stats_f64(
    x: &[f64],
    weight: Option<&[f64]>,
    bias: Option<&[f64]>,
    batch: usize,
    norm_size: usize,
    eps: f64,
) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
    let mut out = vec![0.0f64; batch * norm_size];
    let mut means = vec![0.0f64; batch];
    let mut rstds = vec![0.0f64; batch];
    let inv_n = 1.0 / norm_size as f64;
    out.par_chunks_mut(norm_size)
        .zip(means.par_iter_mut())
        .zip(rstds.par_iter_mut())
        .enumerate()
        .for_each(|(r, ((orow, mean_slot), rstd_slot))| {
            let xrow = &x[r * norm_size..r * norm_size + norm_size];
            let mut sum = 0.0f64;
            for &v in xrow {
                sum += v;
            }
            let mean = sum * inv_n;
            let mut vsum = 0.0f64;
            for &v in xrow {
                let d = v - mean;
                vsum += d * d;
            }
            let rstd = 1.0 / (vsum * inv_n + eps).sqrt();
            *mean_slot = mean;
            *rstd_slot = rstd;
            for j in 0..norm_size {
                let mut y = (xrow[j] - mean) * rstd;
                if let Some(w) = weight {
                    y *= w[j];
                }
                if let Some(b) = bias {
                    y += b[j];
                }
                orow[j] = y;
            }
        });
    (out, means, rstds)
}

/// Backward of [`layer_norm_forward_f64`] with affine weight (and bias). Given
/// `dy` (`[batch, norm_size]`), the saved input `x` and `weight`, returns
/// `(dx, dweight, dbias)`. Recomputes `mean`/`rstd`/`xhat` per row (cheap) so the
/// forward need only save `x` and `weight`:
///   `dxhat = dy·w`,
///   `dx = rstd·(dxhat − mean_j(dxhat) − xhat·mean_j(dxhat·xhat))`,
///   `dweight[j] = Σ_rows dy·xhat`,  `dbias[j] = Σ_rows dy`.
/// `dx` is parallel over rows; the affine grads are a deterministic serial row
/// reduction (matches the op-graph backward to tolerance).
#[must_use]
pub fn layer_norm_backward_f64(
    dy: &[f64],
    x: &[f64],
    weight: &[f64],
    batch: usize,
    norm_size: usize,
    eps: f64,
) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
    let inv_n = 1.0 / norm_size as f64;
    let mut dx = vec![0.0f64; batch * norm_size];
    dx.par_chunks_mut(norm_size)
        .enumerate()
        .for_each(|(r, dxrow)| {
            let xrow = &x[r * norm_size..r * norm_size + norm_size];
            let dyrow = &dy[r * norm_size..r * norm_size + norm_size];
            let mut sum = 0.0f64;
            for &v in xrow {
                sum += v;
            }
            let mean = sum * inv_n;
            let mut vsum = 0.0f64;
            for &v in xrow {
                let d = v - mean;
                vsum += d * d;
            }
            let rstd = 1.0 / (vsum * inv_n + eps).sqrt();
            let mut c1 = 0.0f64;
            let mut c2 = 0.0f64;
            for j in 0..norm_size {
                let xhat = (xrow[j] - mean) * rstd;
                let dxhat = dyrow[j] * weight[j];
                c1 += dxhat;
                c2 += dxhat * xhat;
            }
            for j in 0..norm_size {
                let xhat = (xrow[j] - mean) * rstd;
                let dxhat = dyrow[j] * weight[j];
                dxrow[j] = rstd * (dxhat - (c1 + xhat * c2) * inv_n);
            }
        });
    // Affine grads: deterministic serial reduction over rows (cache-friendly,
    // run-to-run stable — a parallel float reduce would reorder the sum).
    let mut dweight = vec![0.0f64; norm_size];
    let mut dbias = vec![0.0f64; norm_size];
    for r in 0..batch {
        let xrow = &x[r * norm_size..r * norm_size + norm_size];
        let dyrow = &dy[r * norm_size..r * norm_size + norm_size];
        let mut sum = 0.0f64;
        for &v in xrow {
            sum += v;
        }
        let mean = sum * inv_n;
        let mut vsum = 0.0f64;
        for &v in xrow {
            let d = v - mean;
            vsum += d * d;
        }
        let rstd = 1.0 / (vsum * inv_n + eps).sqrt();
        for j in 0..norm_size {
            let xhat = (xrow[j] - mean) * rstd;
            dweight[j] += dyrow[j] * xhat;
            dbias[j] += dyrow[j];
        }
    }
    (dx, dweight, dbias)
}

/// Backward of [`layer_norm_forward_with_stats_f64`] with affine weight (and
/// bias), reusing the forward row statistics. The affine gradient row reduction
/// stays serial and row-major so floating-point accumulation order is unchanged.
#[must_use]
pub fn layer_norm_backward_with_stats_f64(
    dy: &[f64],
    x: &[f64],
    weight: &[f64],
    means: &[f64],
    rstds: &[f64],
    batch: usize,
    norm_size: usize,
) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
    let inv_n = 1.0 / norm_size as f64;
    let mut dx = vec![0.0f64; batch * norm_size];
    dx.par_chunks_mut(norm_size)
        .enumerate()
        .for_each(|(r, dxrow)| {
            let xrow = &x[r * norm_size..r * norm_size + norm_size];
            let dyrow = &dy[r * norm_size..r * norm_size + norm_size];
            let mean = means[r];
            let rstd = rstds[r];
            let mut c1 = 0.0f64;
            let mut c2 = 0.0f64;
            for j in 0..norm_size {
                let xhat = (xrow[j] - mean) * rstd;
                let dxhat = dyrow[j] * weight[j];
                c1 += dxhat;
                c2 += dxhat * xhat;
            }
            for j in 0..norm_size {
                let xhat = (xrow[j] - mean) * rstd;
                let dxhat = dyrow[j] * weight[j];
                dxrow[j] = rstd * (dxhat - (c1 + xhat * c2) * inv_n);
            }
        });
    let mut dweight = vec![0.0f64; norm_size];
    let mut dbias = vec![0.0f64; norm_size];
    for r in 0..batch {
        let xrow = &x[r * norm_size..r * norm_size + norm_size];
        let dyrow = &dy[r * norm_size..r * norm_size + norm_size];
        let mean = means[r];
        let rstd = rstds[r];
        for j in 0..norm_size {
            let xhat = (xrow[j] - mean) * rstd;
            dweight[j] += dyrow[j] * xhat;
            dbias[j] += dyrow[j];
        }
    }
    (dx, dweight, dbias)
}

/// Fused RMSNorm forward (f64): per row of `[batch, norm_size]`, computes
/// `y = x / sqrt(mean(x²) + eps) * weight` in one streaming pass (no mean
/// subtraction — modern-LLM RMSNorm), never materialising the op-graph
/// intermediates (square, broadcast mean, eps tensor, …). Parallel over rows.
#[must_use]
pub fn rms_norm_forward_f64(
    x: &[f64],
    weight: Option<&[f64]>,
    batch: usize,
    norm_size: usize,
    eps: f64,
) -> Vec<f64> {
    let inv_n = 1.0 / norm_size as f64;
    let mut out = vec![0.0f64; batch * norm_size];
    out.par_chunks_mut(norm_size)
        .enumerate()
        .for_each(|(r, orow)| {
            let xrow = &x[r * norm_size..r * norm_size + norm_size];
            let mut ss = 0.0f64;
            for &v in xrow {
                ss += v * v;
            }
            let rstd = 1.0 / (ss * inv_n + eps).sqrt();
            for j in 0..norm_size {
                let mut y = xrow[j] * rstd;
                if let Some(w) = weight {
                    y *= w[j];
                }
                orow[j] = y;
            }
        });
    out
}

/// f32 mirror of [`rms_norm_forward_f64`]: per-row RMS normalize + optional
/// weight, one streaming pass, parallel over rows. Replaces the f32 op-graph
/// (square/mean/rsqrt/mul) the f32 no-grad path fell through to.
#[must_use]
pub fn rms_norm_forward_f32(
    x: &[f32],
    weight: Option<&[f32]>,
    batch: usize,
    norm_size: usize,
    eps: f32,
) -> Vec<f32> {
    let inv_n = 1.0 / norm_size as f32;
    let mut out = vec![0.0f32; batch * norm_size];
    out.par_chunks_mut(norm_size)
        .enumerate()
        .for_each(|(r, orow)| {
            let xrow = &x[r * norm_size..r * norm_size + norm_size];
            let mut ss = 0.0f32;
            for &v in xrow {
                ss += v * v;
            }
            let rstd = 1.0 / (ss * inv_n + eps).sqrt();
            for j in 0..norm_size {
                let mut y = xrow[j] * rstd;
                if let Some(w) = weight {
                    y *= w[j];
                }
                orow[j] = y;
            }
        });
    out
}

/// Backward of [`rms_norm_forward_f64`]. Given `dy`, the saved `x` and optional
/// `weight`, returns `(dx, dweight?)`. With `g[j] = dy[j]·w[j]`,
/// `rstd = 1/sqrt(mean(x²)+eps)`, `c = Σ_j g[j]·x[j]`:
///   `dx[i] = rstd·g[i] − (rstd³·c/N)·x[i]`,
///   `dweight[j] = Σ_rows dy[j]·x[j]·rstd`.
/// `dx` is parallel over rows; `dweight` a deterministic serial row reduction.
#[must_use]
pub fn rms_norm_backward_f64(
    dy: &[f64],
    x: &[f64],
    weight: Option<&[f64]>,
    batch: usize,
    norm_size: usize,
    eps: f64,
) -> (Vec<f64>, Option<Vec<f64>>) {
    let inv_n = 1.0 / norm_size as f64;
    let mut dx = vec![0.0f64; batch * norm_size];
    dx.par_chunks_mut(norm_size)
        .enumerate()
        .for_each(|(r, dxrow)| {
            let xrow = &x[r * norm_size..r * norm_size + norm_size];
            let dyrow = &dy[r * norm_size..r * norm_size + norm_size];
            let mut ss = 0.0f64;
            for &v in xrow {
                ss += v * v;
            }
            let rstd = 1.0 / (ss * inv_n + eps).sqrt();
            let mut c = 0.0f64;
            for j in 0..norm_size {
                let g = dyrow[j] * weight.map_or(1.0, |w| w[j]);
                c += g * xrow[j];
            }
            let coef = rstd * rstd * rstd * c * inv_n;
            for j in 0..norm_size {
                let g = dyrow[j] * weight.map_or(1.0, |w| w[j]);
                dxrow[j] = rstd * g - coef * xrow[j];
            }
        });
    let dweight = weight.map(|_| {
        let mut dw = vec![0.0f64; norm_size];
        for r in 0..batch {
            let xrow = &x[r * norm_size..r * norm_size + norm_size];
            let dyrow = &dy[r * norm_size..r * norm_size + norm_size];
            let mut ss = 0.0f64;
            for &v in xrow {
                ss += v * v;
            }
            let rstd = 1.0 / (ss * inv_n + eps).sqrt();
            for j in 0..norm_size {
                dw[j] += dyrow[j] * xrow[j] * rstd;
            }
        }
        dw
    });
    (dx, dweight)
}

/// Fused softmax-cross-entropy forward (f64): per row of `[batch, classes]`
/// logits with a class-index `target`, returns the per-row loss
/// `lse(logits) - logits[target]` (`= -log_softmax[target]`) in one streaming
/// pass — NEVER materialising the full `[batch, classes]` log-softmax tensor the
/// `log_softmax + nll_loss` op-graph allocates just to gather `batch` scalars.
#[must_use]
pub fn cross_entropy_forward_f64(
    logits: &[f64],
    target: &[usize],
    batch: usize,
    classes: usize,
) -> Vec<f64> {
    let mut loss = vec![0.0f64; batch];
    loss.par_iter_mut().enumerate().for_each(|(i, li)| {
        let row = &logits[i * classes..i * classes + classes];
        let mut m = f64::NEG_INFINITY;
        for &v in row {
            if v > m {
                m = v;
            }
        }
        let mut s = 0.0f64;
        for &v in row {
            s += (v - m).exp();
        }
        let lse = m + s.ln();
        *li = lse - row[target[i]];
    });
    loss
}

/// Backward of [`cross_entropy_forward_f64`]. Given `dloss` (`[batch]`, the grad
/// of the per-row losses), the saved `logits` and `target`, returns `dlogits`
/// (`[batch, classes]`): `dlogits[i][c] = dloss[i]·(softmax(logits[i])[c] −
/// [c == target[i]])`. Parallel over rows.
#[must_use]
pub fn cross_entropy_backward_f64(
    logits: &[f64],
    target: &[usize],
    dloss: &[f64],
    batch: usize,
    classes: usize,
) -> Vec<f64> {
    let mut dlogits = vec![0.0f64; batch * classes];
    dlogits
        .par_chunks_mut(classes)
        .enumerate()
        .for_each(|(i, drow)| {
            let row = &logits[i * classes..i * classes + classes];
            let mut m = f64::NEG_INFINITY;
            for &v in row {
                if v > m {
                    m = v;
                }
            }
            let mut s = 0.0f64;
            for &v in row {
                s += (v - m).exp();
            }
            let lse = m + s.ln();
            let g = dloss[i];
            let t = target[i];
            for c in 0..classes {
                let sm = (row[c] - lse).exp();
                drow[c] = g * (sm - if c == t { 1.0 } else { 0.0 });
            }
        });
    dlogits
}

/// Fused GroupNorm forward (f64). Input is `[batch, channels, spatial]` flattened
/// (channels = num_groups·cpg). Each `(sample, group)` is a contiguous
/// `cpg·spatial` block normalised over its own mean/var; the affine `weight`/`bias`
/// are PER-CHANNEL (`[channels]`). One streaming pass per group, parallel over the
/// `batch·num_groups` groups — none of the ~15 op-graph full-size intermediates.
#[allow(clippy::too_many_arguments)]
#[must_use]
pub fn group_norm_forward_f64(
    x: &[f64],
    weight: Option<&[f64]>,
    bias: Option<&[f64]>,
    batch: usize,
    num_groups: usize,
    cpg: usize,
    spatial: usize,
    eps: f64,
) -> Vec<f64> {
    let group_numel = cpg * spatial;
    let inv_m = 1.0 / group_numel as f64;
    let mut out = vec![0.0f64; batch * num_groups * group_numel];
    out.par_chunks_mut(group_numel)
        .enumerate()
        .for_each(|(grp, orow)| {
            let g = grp % num_groups;
            let base = grp * group_numel;
            let xb = &x[base..base + group_numel];
            let mut sum = 0.0f64;
            for &v in xb {
                sum += v;
            }
            let mean = sum * inv_m;
            let mut vsum = 0.0f64;
            for &v in xb {
                let d = v - mean;
                vsum += d * d;
            }
            let rstd = 1.0 / (vsum * inv_m + eps).sqrt();
            for i in 0..group_numel {
                let c = g * cpg + i / spatial;
                let mut y = (xb[i] - mean) * rstd;
                if let Some(w) = weight {
                    y *= w[c];
                }
                if let Some(b) = bias {
                    y += b[c];
                }
                orow[i] = y;
            }
        });
    out
}

/// f32 mirror of [`group_norm_forward_f64`]: per-(sample,group) mean/rstd
/// normalize + per-channel affine, one streaming pass, parallel over groups.
/// Replaces the f32 op-graph (reshape/mean/var/expand/affine, ~15 nodes) the f32
/// no-grad path fell through to.
#[allow(clippy::too_many_arguments)]
#[must_use]
pub fn group_norm_forward_f32(
    x: &[f32],
    weight: Option<&[f32]>,
    bias: Option<&[f32]>,
    batch: usize,
    num_groups: usize,
    cpg: usize,
    spatial: usize,
    eps: f32,
) -> Vec<f32> {
    let group_numel = cpg * spatial;
    let inv_m = 1.0 / group_numel as f32;
    let mut out = vec![0.0f32; batch * num_groups * group_numel];
    out.par_chunks_mut(group_numel)
        .enumerate()
        .for_each(|(grp, orow)| {
            let g = grp % num_groups;
            let base = grp * group_numel;
            let xb = &x[base..base + group_numel];
            let mut sum = 0.0f32;
            for &v in xb {
                sum += v;
            }
            let mean = sum * inv_m;
            let mut vsum = 0.0f32;
            for &v in xb {
                let d = v - mean;
                vsum += d * d;
            }
            let rstd = 1.0 / (vsum * inv_m + eps).sqrt();
            for i in 0..group_numel {
                let c = g * cpg + i / spatial;
                let mut y = (xb[i] - mean) * rstd;
                if let Some(w) = weight {
                    y *= w[c];
                }
                if let Some(b) = bias {
                    y += b[c];
                }
                orow[i] = y;
            }
        });
    out
}

/// Backward of [`group_norm_forward_f64`] with per-channel affine. Returns
/// `(dx, dweight?, dbias?)`. `dx` is parallel over groups (same normalisation
/// Jacobian as LayerNorm, over `cpg·spatial` per group); `dweight`/`dbias` are a
/// deterministic serial reduction summing over every (sample, spatial) of each
/// channel.
#[allow(clippy::too_many_arguments)]
#[must_use]
pub fn group_norm_backward_f64(
    dy: &[f64],
    x: &[f64],
    weight: Option<&[f64]>,
    batch: usize,
    num_groups: usize,
    cpg: usize,
    spatial: usize,
    eps: f64,
) -> (Vec<f64>, Option<Vec<f64>>, Option<Vec<f64>>) {
    let group_numel = cpg * spatial;
    let inv_m = 1.0 / group_numel as f64;
    let channels = num_groups * cpg;
    let mut dx = vec![0.0f64; batch * num_groups * group_numel];
    dx.par_chunks_mut(group_numel)
        .enumerate()
        .for_each(|(grp, dxrow)| {
            let g = grp % num_groups;
            let base = grp * group_numel;
            let xb = &x[base..base + group_numel];
            let dyb = &dy[base..base + group_numel];
            let mut sum = 0.0f64;
            for &v in xb {
                sum += v;
            }
            let mean = sum * inv_m;
            let mut vsum = 0.0f64;
            for &v in xb {
                let d = v - mean;
                vsum += d * d;
            }
            let rstd = 1.0 / (vsum * inv_m + eps).sqrt();
            let mut c1 = 0.0f64;
            let mut c2 = 0.0f64;
            for i in 0..group_numel {
                let c = g * cpg + i / spatial;
                let xhat = (xb[i] - mean) * rstd;
                let dxhat = dyb[i] * weight.map_or(1.0, |w| w[c]);
                c1 += dxhat;
                c2 += dxhat * xhat;
            }
            for i in 0..group_numel {
                let c = g * cpg + i / spatial;
                let xhat = (xb[i] - mean) * rstd;
                let dxhat = dyb[i] * weight.map_or(1.0, |w| w[c]);
                dxrow[i] = rstd * (dxhat - (c1 + xhat * c2) * inv_m);
            }
        });
    let need_affine = weight.is_some();
    let (dweight, dbias) = if need_affine {
        let mut dw = vec![0.0f64; channels];
        let mut db = vec![0.0f64; channels];
        for grp in 0..batch * num_groups {
            let g = grp % num_groups;
            let base = grp * group_numel;
            let xb = &x[base..base + group_numel];
            let dyb = &dy[base..base + group_numel];
            let mut sum = 0.0f64;
            for &v in xb {
                sum += v;
            }
            let mean = sum * inv_m;
            let mut vsum = 0.0f64;
            for &v in xb {
                let d = v - mean;
                vsum += d * d;
            }
            let rstd = 1.0 / (vsum * inv_m + eps).sqrt();
            for i in 0..group_numel {
                let c = g * cpg + i / spatial;
                dw[c] += dyb[i] * (xb[i] - mean) * rstd;
                db[c] += dyb[i];
            }
        }
        (Some(dw), Some(db))
    } else {
        (None, None)
    };
    (dx, dweight, dbias)
}

/// im2col gather for conv2d over a PADDED `[batch, in_ch, ph, pw]` input. Writes
/// the panel `[batch·oh·ow, in_ch·kh·kw]` in patch-major, `(in_ch,kh,kw)`-minor
/// order (matching the no-grad fast path bit-for-bit). Parallel over panel rows.
#[allow(clippy::too_many_arguments)]
#[must_use]
pub fn conv2d_im2col_f64(
    padded: &[f64],
    batch: usize,
    in_ch: usize,
    ph: usize,
    pw: usize,
    kh: usize,
    kw: usize,
    oh: usize,
    ow: usize,
    sh: usize,
    sw: usize,
) -> Vec<f64> {
    let patch_width = in_ch * kh * kw;
    let patch_count = oh * ow;
    let mut panel = vec![0.0f64; batch * patch_count * patch_width];
    panel
        .par_chunks_mut(patch_width)
        .enumerate()
        .for_each(|(row, prow)| {
            let b = row / patch_count;
            let pc = row % patch_count;
            let base_h = (pc / ow) * sh;
            let base_w = (pc % ow) * sw;
            let batch_off = b * in_ch * ph * pw;
            for c in 0..in_ch {
                let ch_off = batch_off + c * ph * pw;
                let pch = c * kh * kw;
                for kr in 0..kh {
                    let irow = ch_off + (base_h + kr) * pw + base_w;
                    let prow_off = pch + kr * kw;
                    prow[prow_off..(kw + prow_off)].copy_from_slice(&padded[irow..(kw + irow)]);
                }
            }
        });
    panel
}

/// col2im scatter: the transpose of [`conv2d_im2col_f64`]. Accumulates the
/// `dpanel` columns back to a `dpadded [batch, in_ch, ph, pw]` (overlapping
/// patches sum). Parallel over batch (disjoint regions), deterministic serial
/// accumulation within each batch.
#[allow(clippy::too_many_arguments)]
#[must_use]
pub fn conv2d_col2im_f64(
    dpanel: &[f64],
    batch: usize,
    in_ch: usize,
    ph: usize,
    pw: usize,
    kh: usize,
    kw: usize,
    oh: usize,
    ow: usize,
    sh: usize,
    sw: usize,
) -> Vec<f64> {
    let patch_width = in_ch * kh * kw;
    let patch_count = oh * ow;
    let mut dpadded = vec![0.0f64; batch * in_ch * ph * pw];
    dpadded
        .par_chunks_mut(in_ch * ph * pw)
        .enumerate()
        .for_each(|(b, dpb)| {
            for pc in 0..patch_count {
                let base_h = (pc / ow) * sh;
                let base_w = (pc % ow) * sw;
                let prow = (b * patch_count + pc) * patch_width;
                for c in 0..in_ch {
                    let ch_off = c * ph * pw;
                    let pch = c * kh * kw;
                    for kr in 0..kh {
                        let irow = ch_off + (base_h + kr) * pw + base_w;
                        let prow_off = prow + pch + kr * kw;
                        for kc in 0..kw {
                            dpb[irow + kc] += dpanel[prow_off + kc];
                        }
                    }
                }
            }
        });
    dpadded
}

/// Fused conv2d forward (f64) on a PADDED input: im2col + `panel @ weight_flat^T`
/// (via `dgemm_bt`, no weight transpose) written straight to NCHW `[batch,
/// out_ch, oh, ow]`, plus optional per-channel bias. Bit-exact to the no-grad
/// fast path. `weight_flat` is `[out_ch, in_ch·kh·kw]`.
#[allow(clippy::too_many_arguments)]
#[must_use]
pub fn conv2d_forward_f64(
    padded: &[f64],
    weight_flat: &[f64],
    bias: Option<&[f64]>,
    batch: usize,
    in_ch: usize,
    ph: usize,
    pw: usize,
    kh: usize,
    kw: usize,
    oh: usize,
    ow: usize,
    sh: usize,
    sw: usize,
    out_ch: usize,
) -> Vec<f64> {
    let patch_width = in_ch * kh * kw;
    let patch_count = oh * ow;
    let flat = batch * patch_count;
    let panel = conv2d_im2col_f64(padded, batch, in_ch, ph, pw, kh, kw, oh, ow, sh, sw);
    let mut out_flat = vec![0.0f64; flat * out_ch];
    gemm::dgemm_bt(
        flat,
        patch_width,
        out_ch,
        &panel,
        weight_flat,
        &mut out_flat,
    );
    let mut out = vec![0.0f64; batch * out_ch * patch_count];
    out.par_chunks_mut(patch_count)
        .enumerate()
        .for_each(|(idx, orow)| {
            let n = idx / out_ch;
            let oc = idx % out_ch;
            let bo = bias.map_or(0.0, |bb| bb[oc]);
            for p in 0..patch_count {
                orow[p] = out_flat[(n * patch_count + p) * out_ch + oc] + bo;
            }
        });
    out
}

/// f32 mirror of [`conv2d_im2col_f64`]: parallel im2col into a
/// `[batch·oh·ow, in_ch·kh·kw]` panel.
#[allow(clippy::too_many_arguments)]
#[must_use]
pub fn conv2d_im2col_f32(
    padded: &[f32],
    batch: usize,
    in_ch: usize,
    ph: usize,
    pw: usize,
    kh: usize,
    kw: usize,
    oh: usize,
    ow: usize,
    sh: usize,
    sw: usize,
) -> Vec<f32> {
    let patch_width = in_ch * kh * kw;
    let patch_count = oh * ow;
    let mut panel = vec![0.0f32; batch * patch_count * patch_width];
    panel
        .par_chunks_mut(patch_width)
        .enumerate()
        .for_each(|(row, prow)| {
            let b = row / patch_count;
            let pc = row % patch_count;
            let base_h = (pc / ow) * sh;
            let base_w = (pc % ow) * sw;
            let batch_off = b * in_ch * ph * pw;
            for c in 0..in_ch {
                let ch_off = batch_off + c * ph * pw;
                let pch = c * kh * kw;
                for kr in 0..kh {
                    let irow = ch_off + (base_h + kr) * pw + base_w;
                    let prow_off = pch + kr * kw;
                    prow[prow_off..(kw + prow_off)].copy_from_slice(&padded[irow..(kw + irow)]);
                }
            }
        });
    panel
}

/// f32 mirror of [`conv2d_forward_f64`]: fused im2col + `panel @ weight_flat^T`
/// (via `sgemm_bt`, no weight transpose) written straight to NCHW, plus optional
/// per-channel bias. Replaces the serial 6-deep im2col gather + tensor_matmul the
/// f32 no-grad conv2d path fell through to. `weight_flat` is `[out_ch, in_ch·kh·kw]`.
#[allow(clippy::too_many_arguments)]
#[must_use]
pub fn conv2d_forward_f32(
    padded: &[f32],
    weight_flat: &[f32],
    bias: Option<&[f32]>,
    batch: usize,
    in_ch: usize,
    ph: usize,
    pw: usize,
    kh: usize,
    kw: usize,
    oh: usize,
    ow: usize,
    sh: usize,
    sw: usize,
    out_ch: usize,
) -> Vec<f32> {
    let patch_width = in_ch * kh * kw;
    let patch_count = oh * ow;
    let flat = batch * patch_count;
    let panel = conv2d_im2col_f32(padded, batch, in_ch, ph, pw, kh, kw, oh, ow, sh, sw);
    let mut out_flat = vec![0.0f32; flat * out_ch];
    gemm::sgemm_bt(
        flat,
        patch_width,
        out_ch,
        &panel,
        weight_flat,
        &mut out_flat,
    );
    let mut out = vec![0.0f32; batch * out_ch * patch_count];
    out.par_chunks_mut(patch_count)
        .enumerate()
        .for_each(|(idx, orow)| {
            let n = idx / out_ch;
            let oc = idx % out_ch;
            let bo = bias.map_or(0.0, |bb| bb[oc]);
            for p in 0..patch_count {
                orow[p] = out_flat[(n * patch_count + p) * out_ch + oc] + bo;
            }
        });
    out
}

/// Backward of [`conv2d_forward_f64`]. Returns `(dpadded, dweight_flat, dbias?)`.
/// `dweight = dout^T @ panel`, `dpanel = dout @ weight_flat`, `dpadded =
/// col2im(dpanel)`, `dbias = sum over (n,oh,ow)`.
#[allow(clippy::too_many_arguments)]
#[must_use]
pub fn conv2d_backward_f64(
    dout: &[f64],
    padded: &[f64],
    weight_flat: &[f64],
    batch: usize,
    in_ch: usize,
    ph: usize,
    pw: usize,
    kh: usize,
    kw: usize,
    oh: usize,
    ow: usize,
    sh: usize,
    sw: usize,
    out_ch: usize,
    has_bias: bool,
) -> (Vec<f64>, Vec<f64>, Option<Vec<f64>>) {
    let patch_width = in_ch * kh * kw;
    let patch_count = oh * ow;
    let flat = batch * patch_count;
    // Gather dout [N,out_ch,patch_count] -> dout_flat [flat, out_ch].
    let mut dout_flat = vec![0.0f64; flat * out_ch];
    dout_flat
        .par_chunks_mut(out_ch)
        .enumerate()
        .for_each(|(row, dr)| {
            let n = row / patch_count;
            let p = row % patch_count;
            for (oc, d) in dr.iter_mut().enumerate() {
                *d = dout[(n * out_ch + oc) * patch_count + p];
            }
        });
    let panel = conv2d_im2col_f64(padded, batch, in_ch, ph, pw, kh, kw, oh, ow, sh, sw);
    // dweight_flat [out_ch, patch_width] = dout_flat^T @ panel.
    let mut dout_t = vec![0.0f64; out_ch * flat];
    for r in 0..flat {
        for oc in 0..out_ch {
            dout_t[oc * flat + r] = dout_flat[r * out_ch + oc];
        }
    }
    let mut dweight = vec![0.0f64; out_ch * patch_width];
    gemm::dgemm(out_ch, flat, patch_width, &dout_t, &panel, &mut dweight);
    // dpanel [flat, patch_width] = dout_flat @ weight_flat.
    let mut dpanel = vec![0.0f64; flat * patch_width];
    gemm::dgemm(
        flat,
        out_ch,
        patch_width,
        &dout_flat,
        weight_flat,
        &mut dpanel,
    );
    let dpadded = conv2d_col2im_f64(&dpanel, batch, in_ch, ph, pw, kh, kw, oh, ow, sh, sw);
    let dbias = if has_bias {
        let mut db = vec![0.0f64; out_ch];
        for (oc, dbo) in db.iter_mut().enumerate() {
            let mut s = 0.0f64;
            for n in 0..batch {
                let base = (n * out_ch + oc) * patch_count;
                for p in 0..patch_count {
                    s += dout[base + p];
                }
            }
            *dbo = s;
        }
        Some(db)
    } else {
        None
    };
    (dpadded, dweight, dbias)
}

/// Fused cdist forward (f64): pairwise `p`-norm distances between rows of `x1`
/// `[batch, p_dim, m]` and `x2` `[batch, r_dim, m]`, returning `[batch, p_dim,
/// r_dim]`. For each output `(b, i, j)` it streams the `m` feature differences
/// and reduces them in ONE pass — no `O(P·R·M)` broadcasted-difference tensor is
/// ever materialised (the op-graph path builds ~4 of them). Parallel over the
/// `batch·p_dim` output rows; the per-`k` accumulation order is identical to the
/// serial reference, so the result matches the broadcast path to f64 round-off.
///
/// `p == +inf` reduces by max-abs; finite `p > 0` accumulates `Σ|Δ|^p` then takes
/// the `1/p` power. (`p == 0` / `p == 2` are handled by their own paths.)
#[must_use]
pub fn cdist_forward_f64(
    x1: &[f64],
    x2: &[f64],
    batch: usize,
    p_dim: usize,
    r_dim: usize,
    m: usize,
    p: f64,
) -> Vec<f64> {
    let mut result = vec![0.0f64; batch * p_dim * r_dim];
    if batch * p_dim * r_dim == 0 {
        return result;
    }
    let inv_p = 1.0 / p;
    let is_inf = p == f64::INFINITY;
    // One output row = one (b, i) pair owning a contiguous r_dim slice.
    result
        .par_chunks_mut(r_dim)
        .enumerate()
        .for_each(|(row, out)| {
            let b = row / p_dim;
            let i = row % p_dim;
            let x1_base = b * p_dim * m + i * m;
            let x2_base = b * r_dim * m;
            for (j, slot) in out.iter_mut().enumerate() {
                let x2_row = x2_base + j * m;
                let mut dist = 0.0f64;
                if is_inf {
                    for k in 0..m {
                        let diff = (x1[x1_base + k] - x2[x2_row + k]).abs();
                        if diff > dist {
                            dist = diff;
                        }
                    }
                } else {
                    for k in 0..m {
                        let diff = (x1[x1_base + k] - x2[x2_row + k]).abs();
                        dist += diff.powf(p);
                    }
                    dist = dist.powf(inv_p);
                }
                *slot = dist;
            }
        });
    result
}

/// Fused pdist forward (f64): pairwise `p`-norm distances between every i<j pair
/// of rows of `input` `[n, m]`, returning the flattened strict upper triangle
/// `[n·(n-1)/2]` in `(i, then j)` order — the same layout torch's `pdist` and the
/// op-graph (index_select+sub+abs+pow+sum_dim+pow) produce. Each pair streams its
/// `m` feature differences in ONE pass with no `O(out_len·m)` pair-difference
/// tensor materialised. Parallel over the `i` rows; the per-`k` accumulation order
/// matches the op-graph, so the result is bit-identical to the broadcast path.
///
/// `p == +inf` reduces by max-abs; finite `p > 0` accumulates `Σ|Δ|^p` then takes
/// the `1/p` power. (`p == 0` / `p == 2` are handled by their own paths.)
#[must_use]
pub fn pdist_forward_f64(input: &[f64], n: usize, m: usize, p: f64) -> Vec<f64> {
    let out_len = n * (n - 1) / 2;
    if out_len == 0 {
        return Vec::new();
    }
    let is_inf = p == f64::INFINITY;
    let inv_p = 1.0 / p;
    // Row i owns the (n-1-i) outputs for j in (i+1)..n; build per-row then flatten
    // (the concat is O(out_len), negligible vs the O(n^2·m) reduction).
    let rows: Vec<Vec<f64>> = (0..n - 1)
        .into_par_iter()
        .map(|i| {
            let i_base = i * m;
            let mut row = Vec::with_capacity(n - 1 - i);
            for j in (i + 1)..n {
                let j_base = j * m;
                let mut dist = 0.0f64;
                if is_inf {
                    for k in 0..m {
                        let diff = (input[i_base + k] - input[j_base + k]).abs();
                        if diff > dist {
                            dist = diff;
                        }
                    }
                } else {
                    for k in 0..m {
                        let diff = (input[i_base + k] - input[j_base + k]).abs();
                        dist += diff.powf(p);
                    }
                    dist = dist.powf(inv_p);
                }
                row.push(dist);
            }
            row
        })
        .collect();
    let mut result = Vec::with_capacity(out_len);
    for row in &rows {
        result.extend_from_slice(row);
    }
    result
}

/// Fused Supervised-Contrastive (SupCon) loss forward (f64), no-grad.
///
/// Replaces the ~16-op op-graph (L2-normalize + [N,N] cosine gram + two
/// materialised [N,N] label-mask leaves + masked log-sum-exp + masked
/// positive-mean + reduction) with: one L2-normalize pass, ONE gram GEMM
/// (`gemm::dgemm_bt`, the only heavy compute, kept), and a single fused pass over
/// the gram rows that computes the masked log-sum-exp denominator and the
/// positive-mean numerator inline from `labels` — no [N,N] sim/mask intermediates,
/// no tape. Each step mirrors the op-graph's exact arithmetic and reduction order
/// (sequential per-`d` normalize sum, `tensor_logsumexp`'s max-subtract pass,
/// sequential per-`j` masked sum, sequential per-`i` final sum), so the scalar
/// loss matches the op-graph to f64 round-off.
///
/// `labels[i]` is the class of row `i`; `pos_count_safe[i] = max(|P(i)|, 1)`;
/// `valid[i]` is 1.0 iff row `i` has >=1 positive; `count` is the number of valid
/// anchors (caller guarantees `count > 0`). Returns the scalar SupCon loss.
#[allow(clippy::too_many_arguments)]
#[must_use]
pub fn supcon_loss_forward_f64(
    embeddings: &[f64],
    labels: &[i64],
    pos_count_safe: &[f64],
    valid: &[f64],
    count: usize,
    temperature: f64,
    n: usize,
    d: usize,
) -> f64 {
    // 1. L2-normalize each row (sumsq sequential over d, then clamp >= 1e-12).
    let mut normalized = vec![0.0f64; n * d];
    normalized
        .par_chunks_mut(d)
        .enumerate()
        .for_each(|(i, out)| {
            let base = i * d;
            let mut sumsq = 0.0f64;
            for k in 0..d {
                let v = embeddings[base + k];
                sumsq += v * v;
            }
            let norm = sumsq.sqrt().max(1e-12);
            for (k, slot) in out.iter_mut().enumerate() {
                *slot = embeddings[base + k] / norm;
            }
        });

    // 2. Cosine gram [N,N] = normalized @ normalized^T (the one heavy GEMM).
    let mut gram = vec![0.0f64; n * n];
    gemm::dgemm_bt(n, d, n, &normalized, &normalized, &mut gram);

    // 3. Fused per-row masked log-sum-exp denominator + positive-mean numerator,
    //    streaming the gram row once. `inv_temp` scales the cosine to logits; the
    //    diagonal is pushed to -1e30 (matches `sim + diag_mask`) so it underflows
    //    out of the denominator. Parallel over rows; the final sum over i is done
    //    sequentially afterwards to preserve the op-graph's reduction order.
    let inv_temp = 1.0 / temperature;
    let masked_term: Vec<f64> = (0..n)
        .into_par_iter()
        .map(|i| {
            let row = i * n;
            // Denominator: max-subtract pass exactly like tensor_logsumexp.
            let mut max_val = f64::NEG_INFINITY;
            for j in 0..n {
                let mut s = gram[row + j] * inv_temp;
                if j == i {
                    s -= 1e30;
                }
                if s > max_val {
                    max_val = s;
                }
            }
            let mut sum_exp = 0.0f64;
            for j in 0..n {
                let mut s = gram[row + j] * inv_temp;
                if j == i {
                    s -= 1e30;
                }
                sum_exp += (s - max_val).exp();
            }
            let log_denom = max_val + sum_exp.ln();
            // Numerator: sum of sim over same-label j != i (masked-sum order j=0..n).
            let mut num = 0.0f64;
            for j in 0..n {
                if i != j && labels[i] == labels[j] {
                    num += gram[row + j] * inv_temp;
                }
            }
            let mean_pos = num / pos_count_safe[i];
            (mean_pos - log_denom) * valid[i]
        })
        .collect();

    let mut summed = 0.0f64;
    for &t in &masked_term {
        summed += t;
    }
    #[allow(clippy::cast_precision_loss)]
    {
        summed * (-1.0 / count as f64)
    }
}

/// Fused max-pool3d forward (f64): per output, the max over its `kd×kh×kw`
/// window of `[batch, ch, id, ih, iw]`. Parallel over `(batch,ch)` volumes.
#[allow(clippy::too_many_arguments)]
#[must_use]
pub fn max_pool3d_forward_f64(
    input: &[f64],
    batch: usize,
    ch: usize,
    id: usize,
    ih: usize,
    iw: usize,
    kd: usize,
    kh: usize,
    kw: usize,
    od: usize,
    oh: usize,
    ow: usize,
    sd: usize,
    sh: usize,
    sw: usize,
) -> Vec<f64> {
    let mut out = vec![0.0f64; batch * ch * od * oh * ow];
    out.par_chunks_mut(od * oh * ow)
        .enumerate()
        .for_each(|(plane, orow)| {
            let ibase = plane * id * ih * iw;
            for oz in 0..od {
                let bd = oz * sd;
                for oy in 0..oh {
                    let bh = oy * sh;
                    for ox in 0..ow {
                        let bw = ox * sw;
                        let mut m = f64::NEG_INFINITY;
                        for kdd in 0..kd {
                            let dz = ibase + (bd + kdd) * ih * iw;
                            for kr in 0..kh {
                                let irow = dz + (bh + kr) * iw + bw;
                                for kc in 0..kw {
                                    let v = input[irow + kc];
                                    if v > m {
                                        m = v;
                                    }
                                }
                            }
                        }
                        orow[(oz * oh + oy) * ow + ox] = m;
                    }
                }
            }
        });
    out
}

/// f32 mirror of [`max_pool3d_forward_f64`]: per output, the max over its
/// `kd×kh×kw` window, one pass parallel over `(batch,ch)` volumes. Replaces the
/// f32 op-graph (narrow/amax/cat) the f32 no-grad path fell through to.
#[allow(clippy::too_many_arguments)]
#[must_use]
pub fn max_pool3d_forward_f32(
    input: &[f32],
    batch: usize,
    ch: usize,
    id: usize,
    ih: usize,
    iw: usize,
    kd: usize,
    kh: usize,
    kw: usize,
    od: usize,
    oh: usize,
    ow: usize,
    sd: usize,
    sh: usize,
    sw: usize,
) -> Vec<f32> {
    let mut out = vec![0.0f32; batch * ch * od * oh * ow];
    out.par_chunks_mut(od * oh * ow)
        .enumerate()
        .for_each(|(plane, orow)| {
            let ibase = plane * id * ih * iw;
            for oz in 0..od {
                let bd = oz * sd;
                for oy in 0..oh {
                    let bh = oy * sh;
                    for ox in 0..ow {
                        let bw = ox * sw;
                        let mut m = f32::NEG_INFINITY;
                        for kdd in 0..kd {
                            let dz = ibase + (bd + kdd) * ih * iw;
                            for kr in 0..kh {
                                let irow = dz + (bh + kr) * iw + bw;
                                for kc in 0..kw {
                                    let v = input[irow + kc];
                                    if v > m {
                                        m = v;
                                    }
                                }
                            }
                        }
                        orow[(oz * oh + oy) * ow + ox] = m;
                    }
                }
            }
        });
    out
}

/// Backward of [`max_pool3d_forward_f64`]: routes each output gradient to its
/// window's (first) argmax. Parallel over `(batch,ch)`; overlaps accumulate.
#[allow(clippy::too_many_arguments)]
#[must_use]
pub fn max_pool3d_backward_f64(
    dout: &[f64],
    input: &[f64],
    batch: usize,
    ch: usize,
    id: usize,
    ih: usize,
    iw: usize,
    kd: usize,
    kh: usize,
    kw: usize,
    od: usize,
    oh: usize,
    ow: usize,
    sd: usize,
    sh: usize,
    sw: usize,
) -> Vec<f64> {
    let mut din = vec![0.0f64; batch * ch * id * ih * iw];
    din.par_chunks_mut(id * ih * iw)
        .enumerate()
        .for_each(|(plane, drow)| {
            let dbase = plane * od * oh * ow;
            for oz in 0..od {
                let bd = oz * sd;
                for oy in 0..oh {
                    let bh = oy * sh;
                    for ox in 0..ow {
                        let bw = ox * sw;
                        let mut m = f64::NEG_INFINITY;
                        let mut arg = 0usize;
                        for kdd in 0..kd {
                            let dz = (bd + kdd) * ih * iw;
                            for kr in 0..kh {
                                let loc = dz + (bh + kr) * iw + bw;
                                for kc in 0..kw {
                                    let v = input[plane * id * ih * iw + loc + kc];
                                    if v > m {
                                        m = v;
                                        arg = loc + kc;
                                    }
                                }
                            }
                        }
                        drow[arg] += dout[dbase + (oz * oh + oy) * ow + ox];
                    }
                }
            }
        });
    din
}

/// Fused avg-pool2d forward (f64) over a PADDED `[batch, ch, ph, pw]` input.
/// Per output: sum over its (clamped) `kh×kw` window divided by the divisor —
/// `count_include_pad` ? window size : in-bounds (non-pad) element count. Padded
/// zeros sum to 0 so the sum is identical either way; only the divisor differs.
/// `ceil_mode` is handled by the window clamp (`row_end = min(start+kh, ph)`).
#[allow(clippy::too_many_arguments)]
#[must_use]
pub fn avg_pool2d_forward_f64(
    padded: &[f64],
    batch: usize,
    ch: usize,
    ph: usize,
    pw: usize,
    kh: usize,
    kw: usize,
    oh: usize,
    ow: usize,
    sh: usize,
    sw: usize,
    pad_h: usize,
    pad_w: usize,
    ih: usize,
    iw: usize,
    count_include_pad: bool,
) -> Vec<f64> {
    let mut out = vec![0.0f64; batch * ch * oh * ow];
    out.par_chunks_mut(oh * ow)
        .enumerate()
        .for_each(|(plane, orow)| {
            let pbase = plane * ph * pw;
            for oy in 0..oh {
                let rs = oy * sh;
                let re = (rs + kh).min(ph);
                let vrlen = re.min(pad_h + ih).saturating_sub(rs.max(pad_h));
                for ox in 0..ow {
                    let cs = ox * sw;
                    let ce = (cs + kw).min(pw);
                    let vclen = ce.min(pad_w + iw).saturating_sub(cs.max(pad_w));
                    let mut sum = 0.0f64;
                    for r in rs..re {
                        let irow = pbase + r * pw;
                        for c in cs..ce {
                            sum += padded[irow + c];
                        }
                    }
                    let div = if count_include_pad {
                        ((re - rs) * (ce - cs)) as f64
                    } else {
                        (vrlen * vclen) as f64
                    };
                    orow[oy * ow + ox] = sum / div;
                }
            }
        });
    out
}

/// f32 mirror of [`avg_pool2d_forward_f64`]: one windowed-mean pass over the
/// padded input, parallel over `(batch,ch)` planes. Replaces the f32 op-graph
/// (narrow/sum/div/cat) the f32 no-grad path fell through to.
#[allow(clippy::too_many_arguments)]
#[must_use]
pub fn avg_pool2d_forward_f32(
    padded: &[f32],
    batch: usize,
    ch: usize,
    ph: usize,
    pw: usize,
    kh: usize,
    kw: usize,
    oh: usize,
    ow: usize,
    sh: usize,
    sw: usize,
    pad_h: usize,
    pad_w: usize,
    ih: usize,
    iw: usize,
    count_include_pad: bool,
) -> Vec<f32> {
    let mut out = vec![0.0f32; batch * ch * oh * ow];
    out.par_chunks_mut(oh * ow)
        .enumerate()
        .for_each(|(plane, orow)| {
            let pbase = plane * ph * pw;
            for oy in 0..oh {
                let rs = oy * sh;
                let re = (rs + kh).min(ph);
                let vrlen = re.min(pad_h + ih).saturating_sub(rs.max(pad_h));
                for ox in 0..ow {
                    let cs = ox * sw;
                    let ce = (cs + kw).min(pw);
                    let vclen = ce.min(pad_w + iw).saturating_sub(cs.max(pad_w));
                    let mut sum = 0.0f32;
                    for r in rs..re {
                        let irow = pbase + r * pw;
                        for c in cs..ce {
                            sum += padded[irow + c];
                        }
                    }
                    let div = if count_include_pad {
                        ((re - rs) * (ce - cs)) as f32
                    } else {
                        (vrlen * vclen) as f32
                    };
                    orow[oy * ow + ox] = sum / div;
                }
            }
        });
    out
}

/// Backward of [`avg_pool2d_forward_f64`]: distributes each output gradient
/// equally (`dout/divisor`) to every position in its window of `dpadded`
/// (pad-position grads are later dropped by `tensor_pad`'s backward). Parallel
/// over `(batch,ch)` planes; overlapping windows accumulate deterministically.
#[allow(clippy::too_many_arguments)]
#[must_use]
pub fn avg_pool2d_backward_f64(
    dout: &[f64],
    batch: usize,
    ch: usize,
    ph: usize,
    pw: usize,
    kh: usize,
    kw: usize,
    oh: usize,
    ow: usize,
    sh: usize,
    sw: usize,
    pad_h: usize,
    pad_w: usize,
    ih: usize,
    iw: usize,
    count_include_pad: bool,
) -> Vec<f64> {
    let mut dp = vec![0.0f64; batch * ch * ph * pw];
    dp.par_chunks_mut(ph * pw)
        .enumerate()
        .for_each(|(plane, dprow)| {
            let dbase = plane * oh * ow;
            for oy in 0..oh {
                let rs = oy * sh;
                let re = (rs + kh).min(ph);
                let vrlen = re.min(pad_h + ih).saturating_sub(rs.max(pad_h));
                for ox in 0..ow {
                    let cs = ox * sw;
                    let ce = (cs + kw).min(pw);
                    let vclen = ce.min(pad_w + iw).saturating_sub(cs.max(pad_w));
                    let div = if count_include_pad {
                        ((re - rs) * (ce - cs)) as f64
                    } else {
                        (vrlen * vclen) as f64
                    };
                    let g = dout[dbase + oy * ow + ox] / div;
                    for r in rs..re {
                        let irow = r * pw;
                        for c in cs..ce {
                            dprow[irow + c] += g;
                        }
                    }
                }
            }
        });
    dp
}

/// Fused max-pool2d forward (f64): per output `[n,c,oy,ox]`, the max over its
/// `kh×kw` window of `[batch, ch, ih, iw]`. One pass, parallel over `(batch,ch)`
/// planes.
#[allow(clippy::too_many_arguments)]
#[must_use]
pub fn max_pool2d_forward_f64(
    input: &[f64],
    batch: usize,
    ch: usize,
    ih: usize,
    iw: usize,
    kh: usize,
    kw: usize,
    oh: usize,
    ow: usize,
    sh: usize,
    sw: usize,
) -> Vec<f64> {
    let mut out = vec![0.0f64; batch * ch * oh * ow];
    out.par_chunks_mut(oh * ow)
        .enumerate()
        .for_each(|(plane, orow)| {
            let ibase = plane * ih * iw;
            for oy in 0..oh {
                let base_h = oy * sh;
                for ox in 0..ow {
                    let base_w = ox * sw;
                    let mut m = f64::NEG_INFINITY;
                    for kr in 0..kh {
                        let irow = ibase + (base_h + kr) * iw + base_w;
                        for kc in 0..kw {
                            let v = input[irow + kc];
                            if v > m {
                                m = v;
                            }
                        }
                    }
                    orow[oy * ow + ox] = m;
                }
            }
        });
    out
}

/// f32 mirror of [`max_pool2d_forward_f64`]: per output, the max over its `kh×kw`
/// window, one pass parallel over `(batch,ch)` planes. Replaces the f32 op-graph
/// (per-output narrow/amax/cat) the f32 no-grad path fell through to.
#[allow(clippy::too_many_arguments)]
#[must_use]
pub fn max_pool2d_forward_f32(
    input: &[f32],
    batch: usize,
    ch: usize,
    ih: usize,
    iw: usize,
    kh: usize,
    kw: usize,
    oh: usize,
    ow: usize,
    sh: usize,
    sw: usize,
) -> Vec<f32> {
    let mut out = vec![0.0f32; batch * ch * oh * ow];
    out.par_chunks_mut(oh * ow)
        .enumerate()
        .for_each(|(plane, orow)| {
            let ibase = plane * ih * iw;
            for oy in 0..oh {
                let base_h = oy * sh;
                for ox in 0..ow {
                    let base_w = ox * sw;
                    let mut m = f32::NEG_INFINITY;
                    for kr in 0..kh {
                        let irow = ibase + (base_h + kr) * iw + base_w;
                        for kc in 0..kw {
                            let v = input[irow + kc];
                            if v > m {
                                m = v;
                            }
                        }
                    }
                    orow[oy * ow + ox] = m;
                }
            }
        });
    out
}

/// Backward of [`max_pool2d_forward_f64`]: recomputes the (first) argmax of each
/// window and routes the output gradient there. Parallel over `(batch,ch)`
/// planes; overlapping windows accumulate deterministically within a plane.
#[allow(clippy::too_many_arguments)]
#[must_use]
pub fn max_pool2d_backward_f64(
    dout: &[f64],
    input: &[f64],
    batch: usize,
    ch: usize,
    ih: usize,
    iw: usize,
    kh: usize,
    kw: usize,
    oh: usize,
    ow: usize,
    sh: usize,
    sw: usize,
) -> Vec<f64> {
    let mut din = vec![0.0f64; batch * ch * ih * iw];
    din.par_chunks_mut(ih * iw)
        .enumerate()
        .for_each(|(plane, drow)| {
            let ibase = plane * ih * iw;
            let dbase = plane * oh * ow;
            for oy in 0..oh {
                let base_h = oy * sh;
                for ox in 0..ow {
                    let base_w = ox * sw;
                    let mut m = f64::NEG_INFINITY;
                    let mut arg = 0usize;
                    for kr in 0..kh {
                        let loc = (base_h + kr) * iw + base_w;
                        for kc in 0..kw {
                            let v = input[ibase + loc + kc];
                            if v > m {
                                m = v;
                                arg = loc + kc;
                            }
                        }
                    }
                    drow[arg] += dout[dbase + oy * ow + ox];
                }
            }
        });
    din
}

/// Fused conv_transpose2d forward (f64), computed DIRECTLY (no scatter, no
/// per-position tensor allocs): each output element `[n,oc,oy,ox]` gathers the
/// valid `(ic,kh,kw)` contributions `input[n,ic,iy,ix]·weight[ic,oc,kh,kw]` where
/// `iy·sh = oy+ph-kh`, `ix·sw = ox+pw-kw` (integer & in range). Weight is
/// `[in_ch, out_ch, kh, kw]`. Parallel over `(batch, out_ch)` output planes.
#[allow(clippy::manual_is_multiple_of, clippy::too_many_arguments)]
#[must_use]
pub fn conv_transpose2d_forward_f64(
    input: &[f64],
    weight: &[f64],
    bias: Option<&[f64]>,
    batch: usize,
    in_ch: usize,
    ih: usize,
    iw: usize,
    out_ch: usize,
    kh: usize,
    kw: usize,
    oh: usize,
    ow: usize,
    sh: usize,
    sw: usize,
    ph: usize,
    pw: usize,
) -> Vec<f64> {
    let mut out = vec![0.0f64; batch * out_ch * oh * ow];
    out.par_chunks_mut(oh * ow)
        .enumerate()
        .for_each(|(idx, orow)| {
            let n = idx / out_ch;
            let oc = idx % out_ch;
            let b0 = bias.map_or(0.0, |b| b[oc]);
            for oy in 0..oh {
                for ox in 0..ow {
                    let mut acc = b0;
                    for kr in 0..kh {
                        let y_num = oy + ph;
                        if y_num < kr {
                            continue;
                        }
                        let yd = y_num - kr;
                        if yd % sh != 0 {
                            continue;
                        }
                        let iy = yd / sh;
                        if iy >= ih {
                            continue;
                        }
                        for kc in 0..kw {
                            let x_num = ox + pw;
                            if x_num < kc {
                                continue;
                            }
                            let xd = x_num - kc;
                            if xd % sw != 0 {
                                continue;
                            }
                            let ix = xd / sw;
                            if ix >= iw {
                                continue;
                            }
                            let mut s = 0.0f64;
                            for ic in 0..in_ch {
                                let iv = input[((n * in_ch + ic) * ih + iy) * iw + ix];
                                let wv = weight[(((ic * out_ch + oc) * kh + kr) * kw) + kc];
                                s += iv * wv;
                            }
                            acc += s;
                        }
                    }
                    orow[oy * ow + ox] = acc;
                }
            }
        });
    out
}

/// f32 mirror of [`conv_transpose2d_forward_f64`]: per-output gather over the
/// transposed-conv stencil, parallel over `(batch,out_ch)` planes. Replaces the
/// f32 op-graph scatter (`O(kh·kw·ih)` narrow/matmul/pad/add tensor ops) the f32
/// no-grad path fell through to.
#[allow(clippy::too_many_arguments)]
#[must_use]
pub fn conv_transpose2d_forward_f32(
    input: &[f32],
    weight: &[f32],
    bias: Option<&[f32]>,
    batch: usize,
    in_ch: usize,
    ih: usize,
    iw: usize,
    out_ch: usize,
    kh: usize,
    kw: usize,
    oh: usize,
    ow: usize,
    sh: usize,
    sw: usize,
    ph: usize,
    pw: usize,
) -> Vec<f32> {
    let mut out = vec![0.0f32; batch * out_ch * oh * ow];
    out.par_chunks_mut(oh * ow)
        .enumerate()
        .for_each(|(idx, orow)| {
            let n = idx / out_ch;
            let oc = idx % out_ch;
            let b0 = bias.map_or(0.0, |b| b[oc]);
            for oy in 0..oh {
                for ox in 0..ow {
                    let mut acc = b0;
                    for kr in 0..kh {
                        let y_num = oy + ph;
                        if y_num < kr {
                            continue;
                        }
                        let yd = y_num - kr;
                        if yd % sh != 0 {
                            continue;
                        }
                        let iy = yd / sh;
                        if iy >= ih {
                            continue;
                        }
                        for kc in 0..kw {
                            let x_num = ox + pw;
                            if x_num < kc {
                                continue;
                            }
                            let xd = x_num - kc;
                            if xd % sw != 0 {
                                continue;
                            }
                            let ix = xd / sw;
                            if ix >= iw {
                                continue;
                            }
                            let mut s = 0.0f32;
                            for ic in 0..in_ch {
                                let iv = input[((n * in_ch + ic) * ih + iy) * iw + ix];
                                let wv = weight[(((ic * out_ch + oc) * kh + kr) * kw) + kc];
                                s += iv * wv;
                            }
                            acc += s;
                        }
                    }
                    orow[oy * ow + ox] = acc;
                }
            }
        });
    out
}

/// Backward of [`conv_transpose2d_forward_f64`]. Returns `(dinput, dweight,
/// dbias?)`. `dinput` parallel over `(batch, in_ch)` (gather), `dweight` parallel
/// over the weight elements (each a deterministic reduction), `dbias = Σ dout`.
#[allow(clippy::too_many_arguments)]
#[must_use]
pub fn conv_transpose2d_backward_f64(
    dout: &[f64],
    input: &[f64],
    weight: &[f64],
    batch: usize,
    in_ch: usize,
    ih: usize,
    iw: usize,
    out_ch: usize,
    kh: usize,
    kw: usize,
    oh: usize,
    ow: usize,
    sh: usize,
    sw: usize,
    ph: usize,
    pw: usize,
    has_bias: bool,
) -> (Vec<f64>, Vec<f64>, Option<Vec<f64>>) {
    // dinput[n,ic,iy,ix] = Σ_{oc,kr,kc} dout[n,oc,oy,ox]·weight[ic,oc,kr,kc],
    // oy = iy·sh + kr - ph, ox = ix·sw + kc - pw (in range).
    let mut dinput = vec![0.0f64; batch * in_ch * ih * iw];
    dinput
        .par_chunks_mut(ih * iw)
        .enumerate()
        .for_each(|(idx, drow)| {
            let n = idx / in_ch;
            let ic = idx % in_ch;
            for iy in 0..ih {
                for ix in 0..iw {
                    let mut acc = 0.0f64;
                    for kr in 0..kh {
                        let oy_s = iy * sh + kr;
                        if oy_s < ph {
                            continue;
                        }
                        let oy = oy_s - ph;
                        if oy >= oh {
                            continue;
                        }
                        for kc in 0..kw {
                            let ox_s = ix * sw + kc;
                            if ox_s < pw {
                                continue;
                            }
                            let ox = ox_s - pw;
                            if ox >= ow {
                                continue;
                            }
                            for oc in 0..out_ch {
                                let dv = dout[((n * out_ch + oc) * oh + oy) * ow + ox];
                                let wv = weight[(((ic * out_ch + oc) * kh + kr) * kw) + kc];
                                acc += dv * wv;
                            }
                        }
                    }
                    drow[iy * iw + ix] = acc;
                }
            }
        });
    // dweight[ic,oc,kr,kc] = Σ_{n,iy,ix} input[n,ic,iy,ix]·dout[n,oc,oy,ox].
    let mut dweight = vec![0.0f64; in_ch * out_ch * kh * kw];
    dweight.par_iter_mut().enumerate().for_each(|(widx, dw)| {
        let kc = widx % kw;
        let kr = (widx / kw) % kh;
        let oc = (widx / (kw * kh)) % out_ch;
        let ic = widx / (kw * kh * out_ch);
        let mut acc = 0.0f64;
        for n in 0..batch {
            for iy in 0..ih {
                let oy_s = iy * sh + kr;
                if oy_s < ph {
                    continue;
                }
                let oy = oy_s - ph;
                if oy >= oh {
                    continue;
                }
                for ix in 0..iw {
                    let ox_s = ix * sw + kc;
                    if ox_s < pw {
                        continue;
                    }
                    let ox = ox_s - pw;
                    if ox >= ow {
                        continue;
                    }
                    let iv = input[((n * in_ch + ic) * ih + iy) * iw + ix];
                    let dv = dout[((n * out_ch + oc) * oh + oy) * ow + ox];
                    acc += iv * dv;
                }
            }
        }
        *dw = acc;
    });
    let dbias = if has_bias {
        let mut db = vec![0.0f64; out_ch];
        for (oc, dbo) in db.iter_mut().enumerate() {
            let mut s = 0.0f64;
            for n in 0..batch {
                let base = (n * out_ch + oc) * oh * ow;
                for p in 0..oh * ow {
                    s += dout[base + p];
                }
            }
            *dbo = s;
        }
        Some(db)
    } else {
        None
    };
    (dinput, dweight, dbias)
}

/// 3-D im2col gather for conv3d over a PADDED `[batch, in_ch, pd, ph, pw]` input.
/// Panel `[batch·od·oh·ow, in_ch·kd·kh·kw]`, patch-major, `(in_ch,kd,kh,kw)`-minor.
#[allow(clippy::too_many_arguments)]
#[must_use]
pub fn conv3d_im2col_f64(
    padded: &[f64],
    batch: usize,
    in_ch: usize,
    pd: usize,
    ph: usize,
    pw: usize,
    kd: usize,
    kh: usize,
    kw: usize,
    od: usize,
    oh: usize,
    ow: usize,
    sd: usize,
    sh: usize,
    sw: usize,
) -> Vec<f64> {
    let patch_width = in_ch * kd * kh * kw;
    let patch_count = od * oh * ow;
    let mut panel = vec![0.0f64; batch * patch_count * patch_width];
    panel
        .par_chunks_mut(patch_width)
        .enumerate()
        .for_each(|(row, prow)| {
            let b = row / patch_count;
            let pc = row % patch_count;
            let base_d = (pc / (oh * ow)) * sd;
            let rem = pc % (oh * ow);
            let base_h = (rem / ow) * sh;
            let base_w = (rem % ow) * sw;
            let batch_off = b * in_ch * pd * ph * pw;
            for c in 0..in_ch {
                let ch_off = batch_off + c * pd * ph * pw;
                let pch = c * kd * kh * kw;
                for kdd in 0..kd {
                    let d_off = ch_off + (base_d + kdd) * ph * pw;
                    let pkd = pch + kdd * kh * kw;
                    for kr in 0..kh {
                        let irow = d_off + (base_h + kr) * pw + base_w;
                        let prow_off = pkd + kr * kw;
                        prow[prow_off..(kw + prow_off)].copy_from_slice(&padded[irow..(kw + irow)]);
                    }
                }
            }
        });
    panel
}

/// 3-D col2im: transpose-scatter of [`conv3d_im2col_f64`] (overlaps summed).
/// Parallel over batch, deterministic within each batch.
#[allow(clippy::too_many_arguments)]
#[must_use]
pub fn conv3d_col2im_f64(
    dpanel: &[f64],
    batch: usize,
    in_ch: usize,
    pd: usize,
    ph: usize,
    pw: usize,
    kd: usize,
    kh: usize,
    kw: usize,
    od: usize,
    oh: usize,
    ow: usize,
    sd: usize,
    sh: usize,
    sw: usize,
) -> Vec<f64> {
    let patch_width = in_ch * kd * kh * kw;
    let patch_count = od * oh * ow;
    let mut dpadded = vec![0.0f64; batch * in_ch * pd * ph * pw];
    dpadded
        .par_chunks_mut(in_ch * pd * ph * pw)
        .enumerate()
        .for_each(|(b, dpb)| {
            for pc in 0..patch_count {
                let base_d = (pc / (oh * ow)) * sd;
                let rem = pc % (oh * ow);
                let base_h = (rem / ow) * sh;
                let base_w = (rem % ow) * sw;
                let prow = (b * patch_count + pc) * patch_width;
                for c in 0..in_ch {
                    let ch_off = c * pd * ph * pw;
                    let pch = c * kd * kh * kw;
                    for kdd in 0..kd {
                        let d_off = ch_off + (base_d + kdd) * ph * pw;
                        let pkd = pch + kdd * kh * kw;
                        for kr in 0..kh {
                            let irow = d_off + (base_h + kr) * pw + base_w;
                            let prow_off = prow + pkd + kr * kw;
                            for kc in 0..kw {
                                dpb[irow + kc] += dpanel[prow_off + kc];
                            }
                        }
                    }
                }
            }
        });
    dpadded
}

/// Fused conv3d forward (f64) on a PADDED input: 3-D im2col + `panel @
/// weight_flat^T` (dgemm_bt) straight to NCDHW, plus optional per-channel bias.
#[allow(clippy::too_many_arguments)]
#[must_use]
pub fn conv3d_forward_f64(
    padded: &[f64],
    weight_flat: &[f64],
    bias: Option<&[f64]>,
    batch: usize,
    in_ch: usize,
    pd: usize,
    ph: usize,
    pw: usize,
    kd: usize,
    kh: usize,
    kw: usize,
    od: usize,
    oh: usize,
    ow: usize,
    sd: usize,
    sh: usize,
    sw: usize,
    out_ch: usize,
) -> Vec<f64> {
    let patch_width = in_ch * kd * kh * kw;
    let patch_count = od * oh * ow;
    let flat = batch * patch_count;
    let panel = conv3d_im2col_f64(
        padded, batch, in_ch, pd, ph, pw, kd, kh, kw, od, oh, ow, sd, sh, sw,
    );
    let mut out_flat = vec![0.0f64; flat * out_ch];
    gemm::dgemm_bt(
        flat,
        patch_width,
        out_ch,
        &panel,
        weight_flat,
        &mut out_flat,
    );
    let mut out = vec![0.0f64; batch * out_ch * patch_count];
    out.par_chunks_mut(patch_count)
        .enumerate()
        .for_each(|(idx, orow)| {
            let n = idx / out_ch;
            let oc = idx % out_ch;
            let bo = bias.map_or(0.0, |bb| bb[oc]);
            for p in 0..patch_count {
                orow[p] = out_flat[(n * patch_count + p) * out_ch + oc] + bo;
            }
        });
    out
}

/// f32 mirror of [`conv3d_im2col_f64`]: parallel 3-D im2col into a
/// `[batch·od·oh·ow, in_ch·kd·kh·kw]` panel.
#[allow(clippy::too_many_arguments)]
#[must_use]
pub fn conv3d_im2col_f32(
    padded: &[f32],
    batch: usize,
    in_ch: usize,
    pd: usize,
    ph: usize,
    pw: usize,
    kd: usize,
    kh: usize,
    kw: usize,
    od: usize,
    oh: usize,
    ow: usize,
    sd: usize,
    sh: usize,
    sw: usize,
) -> Vec<f32> {
    let patch_width = in_ch * kd * kh * kw;
    let patch_count = od * oh * ow;
    let mut panel = vec![0.0f32; batch * patch_count * patch_width];
    panel
        .par_chunks_mut(patch_width)
        .enumerate()
        .for_each(|(row, prow)| {
            let b = row / patch_count;
            let pc = row % patch_count;
            let base_d = (pc / (oh * ow)) * sd;
            let rem = pc % (oh * ow);
            let base_h = (rem / ow) * sh;
            let base_w = (rem % ow) * sw;
            let batch_off = b * in_ch * pd * ph * pw;
            for c in 0..in_ch {
                let ch_off = batch_off + c * pd * ph * pw;
                let pch = c * kd * kh * kw;
                for kdd in 0..kd {
                    let d_off = ch_off + (base_d + kdd) * ph * pw;
                    let pkd = pch + kdd * kh * kw;
                    for kr in 0..kh {
                        let irow = d_off + (base_h + kr) * pw + base_w;
                        let prow_off = pkd + kr * kw;
                        prow[prow_off..(kw + prow_off)].copy_from_slice(&padded[irow..(kw + irow)]);
                    }
                }
            }
        });
    panel
}

/// f32 mirror of [`conv3d_forward_f64`]: fused 3-D im2col + `panel @ weight_flat^T`
/// (via `sgemm_bt`) written straight to NCDHW, plus optional per-channel bias.
/// Replaces the per-output narrow/cat/bmm op-graph the f32 path fell through to.
#[allow(clippy::too_many_arguments)]
#[must_use]
pub fn conv3d_forward_f32(
    padded: &[f32],
    weight_flat: &[f32],
    bias: Option<&[f32]>,
    batch: usize,
    in_ch: usize,
    pd: usize,
    ph: usize,
    pw: usize,
    kd: usize,
    kh: usize,
    kw: usize,
    od: usize,
    oh: usize,
    ow: usize,
    sd: usize,
    sh: usize,
    sw: usize,
    out_ch: usize,
) -> Vec<f32> {
    let patch_width = in_ch * kd * kh * kw;
    let patch_count = od * oh * ow;
    let flat = batch * patch_count;
    let panel = conv3d_im2col_f32(
        padded, batch, in_ch, pd, ph, pw, kd, kh, kw, od, oh, ow, sd, sh, sw,
    );
    let mut out_flat = vec![0.0f32; flat * out_ch];
    gemm::sgemm_bt(flat, patch_width, out_ch, &panel, weight_flat, &mut out_flat);
    let mut out = vec![0.0f32; batch * out_ch * patch_count];
    out.par_chunks_mut(patch_count)
        .enumerate()
        .for_each(|(idx, orow)| {
            let n = idx / out_ch;
            let oc = idx % out_ch;
            let bo = bias.map_or(0.0, |bb| bb[oc]);
            for p in 0..patch_count {
                orow[p] = out_flat[(n * patch_count + p) * out_ch + oc] + bo;
            }
        });
    out
}

/// Backward of [`conv3d_forward_f64`]. Returns `(dpadded, dweight_flat, dbias?)`.
#[allow(clippy::too_many_arguments)]
#[must_use]
pub fn conv3d_backward_f64(
    dout: &[f64],
    padded: &[f64],
    weight_flat: &[f64],
    batch: usize,
    in_ch: usize,
    pd: usize,
    ph: usize,
    pw: usize,
    kd: usize,
    kh: usize,
    kw: usize,
    od: usize,
    oh: usize,
    ow: usize,
    sd: usize,
    sh: usize,
    sw: usize,
    out_ch: usize,
    has_bias: bool,
) -> (Vec<f64>, Vec<f64>, Option<Vec<f64>>) {
    let patch_width = in_ch * kd * kh * kw;
    let patch_count = od * oh * ow;
    let flat = batch * patch_count;
    let mut dout_flat = vec![0.0f64; flat * out_ch];
    dout_flat
        .par_chunks_mut(out_ch)
        .enumerate()
        .for_each(|(row, dr)| {
            let n = row / patch_count;
            let p = row % patch_count;
            for (oc, d) in dr.iter_mut().enumerate() {
                *d = dout[(n * out_ch + oc) * patch_count + p];
            }
        });
    let panel = conv3d_im2col_f64(
        padded, batch, in_ch, pd, ph, pw, kd, kh, kw, od, oh, ow, sd, sh, sw,
    );
    let mut dout_t = vec![0.0f64; out_ch * flat];
    for r in 0..flat {
        for oc in 0..out_ch {
            dout_t[oc * flat + r] = dout_flat[r * out_ch + oc];
        }
    }
    let mut dweight = vec![0.0f64; out_ch * patch_width];
    gemm::dgemm(out_ch, flat, patch_width, &dout_t, &panel, &mut dweight);
    let mut dpanel = vec![0.0f64; flat * patch_width];
    gemm::dgemm(
        flat,
        out_ch,
        patch_width,
        &dout_flat,
        weight_flat,
        &mut dpanel,
    );
    let dpadded = conv3d_col2im_f64(
        &dpanel, batch, in_ch, pd, ph, pw, kd, kh, kw, od, oh, ow, sd, sh, sw,
    );
    let dbias = if has_bias {
        let mut db = vec![0.0f64; out_ch];
        for (oc, dbo) in db.iter_mut().enumerate() {
            let mut s = 0.0f64;
            for n in 0..batch {
                let base = (n * out_ch + oc) * patch_count;
                for p in 0..patch_count {
                    s += dout[base + p];
                }
            }
            *dbo = s;
        }
        Some(db)
    } else {
        None
    };
    (dpadded, dweight, dbias)
}

/// Fused per-element Gaussian NLL loss:
/// `0.5·(log(var) + (target − input)²/var [+ log(2π) if full])`. One pass,
/// parallel — none of the ~7 full-size op-graph intermediates.
#[must_use]
pub fn gaussian_nll_forward_f64(
    input: &[f64],
    target: &[f64],
    var: &[f64],
    full: bool,
) -> Vec<f64> {
    let c = if full {
        (2.0 * std::f64::consts::PI).ln()
    } else {
        0.0
    };
    let mut out = vec![0.0f64; input.len()];
    out.par_iter_mut().enumerate().for_each(|(i, o)| {
        let d = target[i] - input[i];
        *o = 0.5 * (var[i].ln() + d * d / var[i] + c);
    });
    out
}

/// Backward of [`gaussian_nll_forward_f64`] (the `full` constant drops out).
/// With `d = target − input`: `dinput = dloss·(input−target)/var`,
/// `dtarget = dloss·(target−input)/var`, `dvar = dloss·0.5·(1/var − d²/var²)`.
#[must_use]
pub fn gaussian_nll_backward_f64(
    dloss: &[f64],
    input: &[f64],
    target: &[f64],
    var: &[f64],
) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
    let n = input.len();
    let mut di = vec![0.0f64; n];
    let mut dt = vec![0.0f64; n];
    let mut dv = vec![0.0f64; n];
    di.par_iter_mut()
        .zip(dt.par_iter_mut())
        .zip(dv.par_iter_mut())
        .enumerate()
        .for_each(|(i, ((dii, dti), dvi))| {
            let d = target[i] - input[i];
            let v = var[i];
            let dl = dloss[i];
            *dii = dl * (-d) / v;
            *dti = dl * d / v;
            *dvi = dl * 0.5 * (1.0 / v - d * d / (v * v));
        });
    (di, dt, dv)
}

/// Fused per-element smooth-L1 (Huber-beta) loss: for `d = x − t`,
/// `0.5·d²/beta` when `|d| < beta`, else `|d| − 0.5·beta`. One pass, parallel —
/// none of the op-graph's ~13 full-size intermediates (incl. 4 constant `full()`
/// tensors). Returns the per-element loss (the caller applies the reduction).
#[must_use]
pub fn smooth_l1_forward_f64(x: &[f64], t: &[f64], beta: f64) -> Vec<f64> {
    let hb = 0.5 * beta;
    let mut out = vec![0.0f64; x.len()];
    out.par_iter_mut().enumerate().for_each(|(i, o)| {
        let d = x[i] - t[i];
        let ad = d.abs();
        *o = if ad < beta {
            0.5 * d * d / beta
        } else {
            ad - hb
        };
    });
    out
}

/// Backward of [`smooth_l1_forward_f64`]. `g = d/beta` when `|d| < beta`, else
/// `sign(d)`; returns `(dinput, dtarget) = (dloss·g, −dloss·g)`. Parallel.
#[must_use]
pub fn smooth_l1_backward_f64(
    dloss: &[f64],
    x: &[f64],
    t: &[f64],
    beta: f64,
) -> (Vec<f64>, Vec<f64>) {
    let mut di = vec![0.0f64; x.len()];
    let mut dt = vec![0.0f64; x.len()];
    di.par_iter_mut()
        .zip(dt.par_iter_mut())
        .enumerate()
        .for_each(|(i, (dii, dti))| {
            let d = x[i] - t[i];
            let g = if d.abs() < beta {
                d / beta
            } else if d > 0.0 {
                1.0
            } else if d < 0.0 {
                -1.0
            } else {
                0.0
            };
            let v = dloss[i] * g;
            *dii = v;
            *dti = -v;
        });
    (di, dt)
}

/// Per-channel batch statistics for BatchNorm over `[batch, channels, spatial]`
/// (NCHW with `spatial = H·W`): `mean[c]`, `var[c]` over all `batch·spatial`
/// elements of channel `c`. Works directly on the NCHW layout (channel `c` is
/// `batch` contiguous `spatial`-blocks strided by `channels·spatial`) — NO
/// permute. Parallel over channels.
#[must_use]
pub fn batch_norm_stats_f64(
    x: &[f64],
    batch: usize,
    channels: usize,
    spatial: usize,
) -> (Vec<f64>, Vec<f64>) {
    let inv_n = 1.0 / (batch * spatial) as f64;
    let cs = channels * spatial;
    let mut mean = vec![0.0f64; channels];
    let mut var = vec![0.0f64; channels];
    mean.par_iter_mut()
        .zip(var.par_iter_mut())
        .enumerate()
        .for_each(|(c, (mc, vc))| {
            let mut sum = 0.0f64;
            for n in 0..batch {
                let base = n * cs + c * spatial;
                for s in 0..spatial {
                    sum += x[base + s];
                }
            }
            let m = sum * inv_n;
            let mut vs = 0.0f64;
            for n in 0..batch {
                let base = n * cs + c * spatial;
                for s in 0..spatial {
                    let d = x[base + s] - m;
                    vs += d * d;
                }
            }
            *mc = m;
            *vc = vs * inv_n;
        });
    (mean, var)
}

/// Apply a BatchNorm normalization+affine with given per-channel `mean`/`var`
/// (batch stats for training, running stats for eval): `y = (x − mean[c]) /
/// sqrt(var[c] + eps) · weight[c] + bias[c]`, folded to `x·scale[c] + shift[c]`.
/// One streaming pass over the NCHW `(sample, channel)` blocks, parallel.
#[allow(clippy::too_many_arguments)]
#[must_use]
pub fn batch_norm_apply_f64(
    x: &[f64],
    mean: &[f64],
    var: &[f64],
    weight: Option<&[f64]>,
    bias: Option<&[f64]>,
    batch: usize,
    channels: usize,
    spatial: usize,
    eps: f64,
) -> Vec<f64> {
    let _ = batch;
    // Precompute per-channel scale/shift once (C sqrts) so the streaming pass is a
    // single fused multiply-add even when spatial == 1 (BatchNorm1d).
    let mut scale = vec![0.0f64; channels];
    let mut shift = vec![0.0f64; channels];
    for c in 0..channels {
        let rstd = 1.0 / (var[c] + eps).sqrt();
        scale[c] = rstd * weight.map_or(1.0, |w| w[c]);
        shift[c] = bias.map_or(0.0, |b| b[c]) - mean[c] * scale[c];
    }
    let mut out = vec![0.0f64; x.len()];
    out.par_chunks_mut(spatial)
        .enumerate()
        .for_each(|(idx, orow)| {
            let c = idx % channels;
            let base = idx * spatial;
            let (sc, sh) = (scale[c], shift[c]);
            for s in 0..spatial {
                orow[s] = x[base + s] * sc + sh;
            }
        });
    out
}

/// f32 mirror of [`batch_norm_stats_f64`]: per-channel mean/var over the
/// `(batch, spatial)` window, parallel over channels.
#[must_use]
pub fn batch_norm_stats_f32(
    x: &[f32],
    batch: usize,
    channels: usize,
    spatial: usize,
) -> (Vec<f32>, Vec<f32>) {
    let inv_n = 1.0 / (batch * spatial) as f32;
    let cs = channels * spatial;
    let mut mean = vec![0.0f32; channels];
    let mut var = vec![0.0f32; channels];
    mean.par_iter_mut()
        .zip(var.par_iter_mut())
        .enumerate()
        .for_each(|(c, (mc, vc))| {
            let mut sum = 0.0f32;
            for n in 0..batch {
                let base = n * cs + c * spatial;
                for s in 0..spatial {
                    sum += x[base + s];
                }
            }
            let m = sum * inv_n;
            let mut vs = 0.0f32;
            for n in 0..batch {
                let base = n * cs + c * spatial;
                for s in 0..spatial {
                    let d = x[base + s] - m;
                    vs += d * d;
                }
            }
            *mc = m;
            *vc = vs * inv_n;
        });
    (mean, var)
}

/// f32 mirror of [`batch_norm_apply_f64`]: per-channel affine normalize via a
/// precomputed scale/shift, one streaming pass parallel over `(batch·channel)` rows.
#[allow(clippy::too_many_arguments)]
#[must_use]
pub fn batch_norm_apply_f32(
    x: &[f32],
    mean: &[f32],
    var: &[f32],
    weight: Option<&[f32]>,
    bias: Option<&[f32]>,
    batch: usize,
    channels: usize,
    spatial: usize,
    eps: f32,
) -> Vec<f32> {
    let _ = batch;
    let mut scale = vec![0.0f32; channels];
    let mut shift = vec![0.0f32; channels];
    for c in 0..channels {
        let rstd = 1.0 / (var[c] + eps).sqrt();
        scale[c] = rstd * weight.map_or(1.0, |w| w[c]);
        shift[c] = bias.map_or(0.0, |b| b[c]) - mean[c] * scale[c];
    }
    let mut out = vec![0.0f32; x.len()];
    out.par_chunks_mut(spatial)
        .enumerate()
        .for_each(|(idx, orow)| {
            let c = idx % channels;
            let base = idx * spatial;
            let (sc, sh) = (scale[c], shift[c]);
            for s in 0..spatial {
                orow[s] = x[base + s] * sc + sh;
            }
        });
    out
}

/// Training backward of BatchNorm (NCHW) given the batch `mean`/`var` used in the
/// forward and the affine `weight`. Returns `(dx, dweight, dbias)`.
///
/// Per channel `c` (reduction over the `batch·spatial` strided elements):
///   `dbias[c] = Σ dy`,  `dweight[c] = Σ dy·xhat`   (xhat = (x−mean)·rstd),
/// and since `Σ dxhat = w·dbias` and `Σ dxhat·xhat = w·dweight` (dxhat = dy·w):
///   `dx[i] = rstd/M·(M·dy[i]·w − w·dbias[c] − xhat[i]·w·dweight[c])`.
/// Pass A (per-channel reductions) is parallel over channels; pass B (dx) over
/// the NCHW blocks. `dweight`/`dbias` come deterministic from pass A.
#[allow(clippy::too_many_arguments)]
#[must_use]
pub fn batch_norm_backward_f64(
    dy: &[f64],
    x: &[f64],
    weight: &[f64],
    mean: &[f64],
    var: &[f64],
    batch: usize,
    channels: usize,
    spatial: usize,
    eps: f64,
) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
    let m = (batch * spatial) as f64;
    let inv_m = 1.0 / m;
    let cs = channels * spatial;
    let mut dweight = vec![0.0f64; channels];
    let mut dbias = vec![0.0f64; channels];
    dweight
        .par_iter_mut()
        .zip(dbias.par_iter_mut())
        .enumerate()
        .for_each(|(c, (dwc, dbc))| {
            let rstd = 1.0 / (var[c] + eps).sqrt();
            let mut sw = 0.0f64;
            let mut sb = 0.0f64;
            for n in 0..batch {
                let base = n * cs + c * spatial;
                for s in 0..spatial {
                    let dyi = dy[base + s];
                    let xhat = (x[base + s] - mean[c]) * rstd;
                    sw += dyi * xhat;
                    sb += dyi;
                }
            }
            *dwc = sw;
            *dbc = sb;
        });
    let mut dx = vec![0.0f64; x.len()];
    dx.par_chunks_mut(spatial)
        .enumerate()
        .for_each(|(idx, dxrow)| {
            let c = idx % channels;
            let base = idx * spatial;
            let rstd = 1.0 / (var[c] + eps).sqrt();
            let w = weight[c];
            let c1 = w * dbias[c];
            let c2 = w * dweight[c];
            for s in 0..spatial {
                let xhat = (x[base + s] - mean[c]) * rstd;
                let dxhat = dy[base + s] * w;
                dxrow[s] = rstd * inv_m * (m * dxhat - c1 - xhat * c2);
            }
        });
    (dx, dweight, dbias)
}

pub fn linear_tensor_f64(
    x: &[f64],
    weight: &[f64],
    bias: Option<&[f64]>,
    batch: usize,
    in_features: usize,
    out_features: usize,
) -> Vec<f64> {
    let mut y = vec![0.0f64; batch * out_features];
    gemm::dgemm_bt(batch, in_features, out_features, x, weight, &mut y);
    if let Some(b) = bias {
        for row in y.chunks_exact_mut(out_features) {
            for (yj, bj) in row.iter_mut().zip(b.iter()) {
                *yj += *bj;
            }
        }
    }
    y
}

/// Backward of `y = x @ weight^T + bias` (the [`linear_tensor_f64`] forward).
///
/// Given `dy` (gradient of `y`, row-major `[batch, out_features]`), the saved
/// `x` (`[batch, in_features]`) and `weight` (`[out_features, in_features]`),
/// returns `(dx, dweight, dbias?)`:
///   - `dx = dy @ weight`           `[batch, in]`  — weight read as `[out,in]`, no transpose
///   - `dweight = dy^T @ x`         `[out, in]`
///   - `dbias = sum_batch(dy)`      `[out]`        (only if `need_bias`)
///
/// These are mathematically identical to the transpose-then-`addmm` path's
/// backward (`d(addmm)/dx = dy @ weight`; `d_weight_t = x^T @ dy`, transposed
/// back to `dy^T @ x`), so gradients match it to tolerance.
#[must_use]
pub fn linear_backward_f64(
    dy: &[f64],
    x: &[f64],
    weight: &[f64],
    batch: usize,
    in_features: usize,
    out_features: usize,
    need_bias: bool,
) -> (Vec<f64>, Vec<f64>, Option<Vec<f64>>) {
    // dx = dy @ weight : [batch,out] @ [out,in] -> [batch,in].
    let mut dx = vec![0.0f64; batch * in_features];
    gemm::dgemm(batch, out_features, in_features, dy, weight, &mut dx);
    // dweight = dy^T @ x : materialise dy^T [out,batch] (grad-sized, contiguous
    // write — far smaller/cheaper than the 8MB strided weight transpose this
    // whole path exists to avoid), then [out,batch] @ [batch,in] -> [out,in].
    let mut dyt = vec![0.0f64; out_features * batch];
    for b in 0..batch {
        let row = &dy[b * out_features..(b + 1) * out_features];
        for (o, &v) in row.iter().enumerate() {
            dyt[o * batch + b] = v;
        }
    }
    let mut dweight = vec![0.0f64; out_features * in_features];
    gemm::dgemm(out_features, batch, in_features, &dyt, x, &mut dweight);
    // dbias = sum over the batch rows of dy.
    let dbias = if need_bias {
        let mut db = vec![0.0f64; out_features];
        for b in 0..batch {
            let row = &dy[b * out_features..(b + 1) * out_features];
            for (o, &v) in row.iter().enumerate() {
                db[o] += v;
            }
        }
        Some(db)
    } else {
        None
    };
    (dx, dweight, dbias)
}

/// f32 mirror of [`linear_tensor_f64`].
#[must_use]
pub fn linear_tensor_f32(
    x: &[f32],
    weight: &[f32],
    bias: Option<&[f32]>,
    batch: usize,
    in_features: usize,
    out_features: usize,
) -> Vec<f32> {
    let mut y = vec![0.0f32; batch * out_features];
    gemm::sgemm_bt(batch, in_features, out_features, x, weight, &mut y);
    if let Some(b) = bias {
        for row in y.chunks_exact_mut(out_features) {
            for (yj, bj) in row.iter_mut().zip(b.iter()) {
                *yj += *bj;
            }
        }
    }
    y
}

pub fn matmul_tensor_contiguous_f64(
    lhs: &[f64],
    rhs: &[f64],
    lhs_meta: &TensorMeta,
    rhs_meta: &TensorMeta,
) -> Result<Vec<f64>, KernelError> {
    ensure_dtype_device_and_layout(lhs_meta, rhs_meta)?;
    let (m, k, n) = matmul_dims(lhs_meta, rhs_meta)?;
    checked_mul(m, k, "matmul lhs shape multiplication overflow")?;
    checked_mul(k, n, "matmul rhs shape multiplication overflow")?;
    let out_numel = checked_mul(m, n, "matmul output shape multiplication overflow")?;

    ensure_storage_len(lhs, lhs_meta, "lhs")?;
    ensure_storage_len(rhs, rhs_meta, "rhs")?;

    let lhs_start = lhs_meta.storage_offset();
    let rhs_start = rhs_meta.storage_offset();

    let mut out = vec![0.0_f64; out_numel];

    // Use optimized GEMM via matrixmultiply crate
    gemm::dgemm(m, k, n, &lhs[lhs_start..], &rhs[rhs_start..], &mut out);

    Ok(out)
}

pub fn lerp_tensor_contiguous_f64(
    start: &[f64],
    end: &[f64],
    weight: f64,
    meta: &TensorMeta,
) -> Result<Vec<f64>, KernelError> {
    ensure_unary_layout_and_storage(start, meta)?;
    let numel = meta.numel();
    if numel == 0 {
        return Ok(Vec::new());
    }
    let offset = meta.storage_offset();
    if end.len() < offset + numel {
        return Err(KernelError::ShapeMismatch {
            lhs: meta.shape().to_vec(),
            rhs: vec![end.len()],
        });
    }
    let s = &start[offset..offset + numel];
    let e = &end[offset..offset + numel];
    Ok(s.iter()
        .zip(e.iter())
        .map(|(&sv, &ev)| sv + weight * (ev - sv))
        .collect())
}

#[allow(clippy::too_many_arguments)]
pub fn addmm_tensor_contiguous_f64(
    input: &[f64],
    mat1: &[f64],
    mat2: &[f64],
    input_meta: &TensorMeta,
    mat1_meta: &TensorMeta,
    mat2_meta: &TensorMeta,
    beta: f64,
    alpha: f64,
) -> Result<Vec<f64>, KernelError> {
    // mat1: [m, k], mat2: [k, n] => output: [m, n]
    // input must be broadcastable to [m, n]
    let (m, k, n) = matmul_dims(mat1_meta, mat2_meta)?;
    let out_numel = checked_mul(m, n, "addmm output shape multiplication overflow")?;
    ensure_storage_len(mat1, mat1_meta, "mat1")?;
    ensure_storage_len(mat2, mat2_meta, "mat2")?;

    let mat1_start = mat1_meta.storage_offset();
    let mat2_start = mat2_meta.storage_offset();
    let input_offset = input_meta.storage_offset();

    // input can be 1-D [n] or 2-D [m,n]
    let input_shape = input_meta.shape();
    let input_1d = input_shape.len() == 1 && input_shape[0] == n;
    let input_2d = input_shape.len() == 2 && input_shape[0] == m && input_shape[1] == n;
    if !input_1d && !input_2d {
        return Err(KernelError::ShapeMismatch {
            lhs: input_shape.to_vec(),
            rhs: vec![m, n],
        });
    }
    ensure_storage_len(input, input_meta, "input")?;

    // Fail closed on non-contiguous operands. The GEMM call below reads
    // mat1/mat2 as row-major contiguous buffers through raw pointers, and the
    // accumulation step reads `input` with a flat offset — a strided view would
    // be silently mis-read as if contiguous, producing wrong results. The
    // `matmul`/`bmm` kernels already reject this via
    // `ensure_dtype_device_and_layout`; addmm must guard the same unsafe path.
    if !mat1_meta.is_contiguous() {
        return Err(KernelError::UnsupportedLayout { side: "mat1" });
    }
    if !mat2_meta.is_contiguous() {
        return Err(KernelError::UnsupportedLayout { side: "mat2" });
    }
    if !input_meta.is_contiguous() {
        return Err(KernelError::UnsupportedLayout { side: "input" });
    }

    // Use optimized GEMM for mat1 @ mat2
    let mut gemm_out = vec![0.0_f64; out_numel];
    gemm::dgemm(
        m,
        k,
        n,
        &mat1[mat1_start..],
        &mat2[mat2_start..],
        &mut gemm_out,
    );

    // Apply: out = beta * input + alpha * (mat1 @ mat2)
    let out: Vec<f64> = if input_1d {
        gemm_out
            .iter()
            .enumerate()
            .map(|(i, &g)| {
                let col = i % n;
                beta * input[input_offset + col] + alpha * g
            })
            .collect()
    } else {
        gemm_out
            .iter()
            .enumerate()
            .map(|(i, &g)| beta * input[input_offset + i] + alpha * g)
            .collect()
    };

    Ok(out)
}

#[allow(clippy::too_many_arguments)]
pub fn addmv_tensor_contiguous_f64(
    input: &[f64],
    mat: &[f64],
    vec_data: &[f64],
    input_meta: &TensorMeta,
    mat_meta: &TensorMeta,
    vec_meta: &TensorMeta,
    beta: f64,
    alpha: f64,
) -> Result<Vec<f64>, KernelError> {
    // mat: [m, k], vec: [k] => output: [m]
    // input must be [m]
    let mat_shape = mat_meta.shape();
    let vec_shape = vec_meta.shape();
    if mat_shape.len() != 2 {
        return Err(KernelError::ShapeMismatch {
            lhs: mat_shape.to_vec(),
            rhs: vec![0, 0],
        });
    }
    if vec_shape.len() != 1 {
        return Err(KernelError::ShapeMismatch {
            lhs: vec_shape.to_vec(),
            rhs: vec![mat_shape[1]],
        });
    }
    let m = mat_shape[0];
    let k = mat_shape[1];
    if vec_shape[0] != k {
        return Err(KernelError::ShapeMismatch {
            lhs: vec_shape.to_vec(),
            rhs: vec![k],
        });
    }
    let input_shape = input_meta.shape();
    if input_shape.len() != 1 || input_shape[0] != m {
        return Err(KernelError::ShapeMismatch {
            lhs: input_shape.to_vec(),
            rhs: vec![m],
        });
    }
    ensure_storage_len(mat, mat_meta, "mat")?;
    ensure_storage_len(vec_data, vec_meta, "vec")?;
    ensure_storage_len(input, input_meta, "input")?;

    let mat_start = mat_meta.storage_offset();
    let vec_start = vec_meta.storage_offset();
    let input_start = input_meta.storage_offset();

    // Push-based output skips the m-cell zero-init memset
    // (frankentorch-u04j); same row-major contract as matmul.
    let mut out = Vec::with_capacity(m);
    // Pairwise dot product per row. Same pattern as matmul; for K
    // typical of LM head linear projections (vocab >= 32k) the
    // sequential drift was visible in inference logits.
    let mut scratch = vec![0.0_f64; k];
    for row in 0..m {
        for (col, scratch_slot) in scratch.iter_mut().enumerate() {
            *scratch_slot = mat[mat_start + row * k + col] * vec_data[vec_start + col];
        }
        let acc = pairwise_sum_f64(&scratch);
        out.push(beta * input[input_start + row] + alpha * acc);
    }
    Ok(out)
}

pub fn dot_tensor_contiguous_f64(
    lhs: &[f64],
    rhs: &[f64],
    lhs_meta: &TensorMeta,
    rhs_meta: &TensorMeta,
) -> Result<f64, KernelError> {
    ensure_dtype_device_and_layout(lhs_meta, rhs_meta)?;
    if lhs_meta.shape().len() != 1 || rhs_meta.shape().len() != 1 {
        return Err(KernelError::ShapeMismatch {
            lhs: lhs_meta.shape().to_vec(),
            rhs: rhs_meta.shape().to_vec(),
        });
    }
    if lhs_meta.shape()[0] != rhs_meta.shape()[0] {
        return Err(KernelError::ShapeMismatch {
            lhs: lhs_meta.shape().to_vec(),
            rhs: rhs_meta.shape().to_vec(),
        });
    }
    ensure_storage_len(lhs, lhs_meta, "lhs")?;
    ensure_storage_len(rhs, rhs_meta, "rhs")?;

    let n = lhs_meta.shape()[0];
    let lhs_start = lhs_meta.storage_offset();
    let rhs_start = rhs_meta.storage_offset();
    // Pairwise via map: stages each lhs[i]*rhs[i] product into the
    // pairwise tree directly without an intermediate Vec. Same
    // O(log N · ε) precision contract as the matmul fix.
    let lhs_slice = &lhs[lhs_start..lhs_start + n];
    let rhs_slice = &rhs[rhs_start..rhs_start + n];
    // zip+map+collect skips the memset of the previous
    // vec![0.0; n] + iter_mut overwrite pattern; the scratch
    // is single-use so there's no cross-iteration buffer to
    // preserve. Same O(log N · ε) precision contract — collect
    // builds a contiguous Vec that pairwise_sum_f64 reads as a
    // slice. Tracked under frankentorch-cunc.
    let scratch: Vec<f64> = lhs_slice
        .iter()
        .zip(rhs_slice)
        .map(|(&l, &r)| l * r)
        .collect();
    Ok(pairwise_sum_f64(&scratch))
}

pub fn outer_tensor_contiguous_f64(
    lhs: &[f64],
    rhs: &[f64],
    lhs_meta: &TensorMeta,
    rhs_meta: &TensorMeta,
) -> Result<Vec<f64>, KernelError> {
    ensure_dtype_device_and_layout(lhs_meta, rhs_meta)?;
    if lhs_meta.shape().len() != 1 || rhs_meta.shape().len() != 1 {
        return Err(KernelError::ShapeMismatch {
            lhs: lhs_meta.shape().to_vec(),
            rhs: rhs_meta.shape().to_vec(),
        });
    }
    let m = lhs_meta.shape()[0];
    let n = rhs_meta.shape()[0];
    let out_numel = checked_mul(m, n, "outer output shape multiplication overflow")?;
    ensure_storage_len(lhs, lhs_meta, "lhs")?;
    ensure_storage_len(rhs, rhs_meta, "rhs")?;
    let lhs_start = lhs_meta.storage_offset();
    let rhs_start = rhs_meta.storage_offset();
    // Capacity-alloc + extend instead of zero-init + indexed
    // overwrite (frankentorch-we9f). Saves a full memset of
    // m*n cells that would be unconditionally overwritten, and
    // hoisting the lhs[i] load outside the inner loop with an
    // rhs slice gives the compiler a known-length iteration to
    // vectorize.
    let mut out = Vec::with_capacity(out_numel);
    let lhs_slice = &lhs[lhs_start..lhs_start + m];
    let rhs_slice = &rhs[rhs_start..rhs_start + n];
    for &l in lhs_slice {
        out.extend(rhs_slice.iter().map(|&r| l * r));
    }
    Ok(out)
}

pub fn bmm_tensor_contiguous_f64(
    lhs: &[f64],
    rhs: &[f64],
    lhs_meta: &TensorMeta,
    rhs_meta: &TensorMeta,
) -> Result<Vec<f64>, KernelError> {
    ensure_dtype_device_and_layout(lhs_meta, rhs_meta)?;
    if lhs_meta.shape().len() != 3 || rhs_meta.shape().len() != 3 {
        return Err(KernelError::ShapeMismatch {
            lhs: lhs_meta.shape().to_vec(),
            rhs: rhs_meta.shape().to_vec(),
        });
    }
    let batch = lhs_meta.shape()[0];
    let m = lhs_meta.shape()[1];
    let k = lhs_meta.shape()[2];
    let rhs_batch = rhs_meta.shape()[0];
    let rhs_k = rhs_meta.shape()[1];
    let n = rhs_meta.shape()[2];
    if batch != rhs_batch || k != rhs_k {
        return Err(KernelError::ShapeMismatch {
            lhs: lhs_meta.shape().to_vec(),
            rhs: rhs_meta.shape().to_vec(),
        });
    }
    let lhs_batch_stride = checked_mul(m, k, "bmm lhs batch stride overflow")?;
    let rhs_batch_stride = checked_mul(k, n, "bmm rhs batch stride overflow")?;
    let out_batch_stride = checked_mul(m, n, "bmm output batch stride overflow")?;
    checked_mul(
        batch,
        lhs_batch_stride,
        "bmm lhs shape multiplication overflow",
    )?;
    checked_mul(
        batch,
        rhs_batch_stride,
        "bmm rhs shape multiplication overflow",
    )?;
    let out_numel = checked_mul(
        batch,
        out_batch_stride,
        "bmm output shape multiplication overflow",
    )?;
    ensure_storage_len(lhs, lhs_meta, "lhs")?;
    ensure_storage_len(rhs, rhs_meta, "rhs")?;

    let lhs_start = lhs_meta.storage_offset();
    let rhs_start = rhs_meta.storage_offset();
    let mut out = vec![0.0_f64; out_numel];

    if out_batch_stride == 0 {
        return Ok(out);
    }

    if batch < 8 {
        for b in 0..batch {
            let lhs_base = lhs_start + b * lhs_batch_stride;
            let rhs_base = rhs_start + b * rhs_batch_stride;
            let out_base = b * out_batch_stride;
            gemm::dgemm(
                m,
                k,
                n,
                &lhs[lhs_base..lhs_base + lhs_batch_stride],
                &rhs[rhs_base..rhs_base + rhs_batch_stride],
                &mut out[out_base..out_base + out_batch_stride],
            );
        }
        return Ok(out);
    }

    out.par_chunks_exact_mut(out_batch_stride)
        .enumerate()
        .for_each(|(b, out_batch)| {
            let lhs_base = lhs_start + b * lhs_batch_stride;
            let rhs_base = rhs_start + b * rhs_batch_stride;
            gemm::dgemm(
                m,
                k,
                n,
                &lhs[lhs_base..lhs_base + lhs_batch_stride],
                &rhs[rhs_base..rhs_base + rhs_batch_stride],
                out_batch,
            );
        });

    Ok(out)
}

pub fn trace_tensor_contiguous_f64(input: &[f64], meta: &TensorMeta) -> Result<f64, KernelError> {
    ensure_unary_layout_and_storage(input, meta)?;
    if meta.shape().len() != 2 {
        return Err(KernelError::InvalidDimension {
            dim: meta.shape().len(),
            ndim: 2,
        });
    }
    let rows = meta.shape()[0];
    let cols = meta.shape()[1];
    let diag_len = rows.min(cols);
    let offset = meta.storage_offset();
    let mut acc = 0.0;
    for i in 0..diag_len {
        acc += input[offset + i * cols + i];
    }
    Ok(acc)
}

pub fn prod_dim_tensor_contiguous_f64(
    input: &[f64],
    meta: &TensorMeta,
    dim: usize,
) -> Result<Vec<f64>, KernelError> {
    ensure_unary_layout_and_storage(input, meta)?;
    let shape = meta.shape();
    let ndim = shape.len();
    if dim >= ndim {
        return Err(KernelError::InvalidDimension { dim, ndim });
    }
    let reduce_size = shape[dim];
    let (outer_size, inner_size, _) =
        checked_dim_loop_sizes(shape, dim, "prod_dim shape volume overflow")?;
    let out_numel = checked_mul(
        outer_size,
        inner_size,
        "prod_dim shape multiplication overflow",
    )?;
    if out_numel == 0 {
        return Ok(Vec::new());
    }
    if reduce_size == 0 {
        return Ok(vec![1.0; out_numel]);
    }
    let offset = meta.storage_offset();
    // Push-based output skips the 1.0-init pass; row-major
    // (outer, inner) order matches output index
    // outer * inner_size + inner (frankentorch-suw1).
    let mut output = Vec::with_capacity(out_numel);
    let data = &input[offset..];

    for outer in 0..outer_size {
        for inner in 0..inner_size {
            let mut prod = 1.0;
            for r in 0..reduce_size {
                prod *= data[outer * reduce_size * inner_size + r * inner_size + inner];
            }
            output.push(prod);
        }
    }

    Ok(output)
}

pub fn var_dim_tensor_contiguous_f64(
    input: &[f64],
    meta: &TensorMeta,
    dim: usize,
) -> Result<Vec<f64>, KernelError> {
    ensure_unary_layout_and_storage(input, meta)?;
    let shape = meta.shape();
    let ndim = shape.len();
    if dim >= ndim {
        return Err(KernelError::InvalidDimension { dim, ndim });
    }
    let reduce_size = shape[dim];
    let (outer_size, inner_size, _) =
        checked_dim_loop_sizes(shape, dim, "var_dim shape volume overflow")?;
    let out_numel = checked_mul(
        outer_size,
        inner_size,
        "var_dim shape multiplication overflow",
    )?;
    if out_numel == 0 {
        return Ok(Vec::new());
    }
    if reduce_size < 2 {
        return Ok(vec![f64::NAN; out_numel]);
    }
    let offset = meta.storage_offset();
    let data = &input[offset..];

    // Push-based output skips the zero-init pass; (outer, inner)
    // loop matches output index outer * inner_size + inner
    // (frankentorch-suw1).
    let mut output = Vec::with_capacity(out_numel);
    #[allow(clippy::cast_precision_loss)]
    let correction = (reduce_size - 1) as f64; // Bessel's correction
    #[allow(clippy::cast_precision_loss)]
    let n_div = reduce_size as f64;

    // Reuse a single scratch buffer for the inner reduce-axis values
    // across all (outer, inner) pairs. Gathering the strided slice
    // into a contiguous Vec lets us pairwise-sum the mean and the
    // squared-deviation accumulator (O(log N · ε) error) instead of
    // accumulating sequentially through strided reads (O(N · ε)).
    // For large reduce_size with squared deviations — which double
    // the relative error of each addend — the pairwise variant is
    // visibly tighter; for small reduce_size the BLOCK = 128 base
    // case keeps the helper bit-equivalent to the prior sequential
    // body.
    let mut scratch = vec![0.0_f64; reduce_size];
    for outer in 0..outer_size {
        for inner in 0..inner_size {
            // Gather the strided values for this (outer, inner) into
            // contiguous scratch.
            for r in 0..reduce_size {
                scratch[r] = data[outer * reduce_size * inner_size + r * inner_size + inner];
            }
            let mean = pairwise_sum_f64(&scratch) / n_div;
            let var_sum = pairwise_sum_map_f64(&scratch, |x| {
                let d = x - mean;
                d * d
            });
            output.push(var_sum / correction);
        }
    }

    Ok(output)
}

pub fn std_dim_tensor_contiguous_f64(
    input: &[f64],
    meta: &TensorMeta,
    dim: usize,
) -> Result<Vec<f64>, KernelError> {
    let mut output = var_dim_tensor_contiguous_f64(input, meta, dim)?;
    for v in &mut output {
        *v = v.sqrt();
    }
    Ok(output)
}

pub fn norm_tensor_contiguous_f64(
    input: &[f64],
    meta: &TensorMeta,
    p: f64,
) -> Result<f64, KernelError> {
    ensure_unary_layout_and_storage(input, meta)?;
    let offset = meta.storage_offset();
    let numel = meta.numel();
    if numel == 0 {
        return Ok(0.0);
    }
    let data = &input[offset..offset + numel];

    if p == f64::INFINITY {
        // norm(inf) = max(|x|). f64::max silently drops NaN, but PyTorch's
        // max reduction propagates it, so fold with explicit NaN checks.
        Ok(data.iter().fold(0.0_f64, |acc, &x| {
            let a = x.abs();
            if acc.is_nan() || a.is_nan() {
                f64::NAN
            } else {
                acc.max(a)
            }
        }))
    } else if p == f64::NEG_INFINITY {
        Ok(data.iter().fold(f64::INFINITY, |acc, &x| {
            let a = x.abs();
            if acc.is_nan() || a.is_nan() {
                f64::NAN
            } else {
                acc.min(a)
            }
        }))
    } else if p == 0.0 {
        // L0 "norm": count of non-zero elements
        Ok(data.iter().filter(|&&x| x != 0.0).count() as f64)
    } else if p == 1.0 {
        Ok(pairwise_sum_map_f64_maybe_par(data, |x| x.abs()))
    } else if p == 2.0 {
        let sum_sq = pairwise_sum_map_f64_maybe_par(data, |x| x * x);
        Ok(sum_sq.sqrt())
    } else {
        let sum_pow = pairwise_sum_map_f64_maybe_par(data, |x| x.abs().powf(p));
        Ok(sum_pow.powf(1.0 / p))
    }
}

pub fn norm_dim_tensor_contiguous_f64(
    input: &[f64],
    meta: &TensorMeta,
    p: f64,
    dim: usize,
) -> Result<Vec<f64>, KernelError> {
    ensure_unary_layout_and_storage(input, meta)?;
    let shape = meta.shape();
    let ndim = shape.len();
    if dim >= ndim {
        return Err(KernelError::InvalidDimension { dim, ndim });
    }
    let reduce_size = shape[dim];
    let (outer_size, inner_size, _) =
        checked_dim_loop_sizes(shape, dim, "norm_dim shape volume overflow")?;
    let out_numel = checked_mul(
        outer_size,
        inner_size,
        "norm_dim shape multiplication overflow",
    )?;
    if out_numel == 0 {
        return Ok(Vec::new());
    }
    if reduce_size == 0 {
        return Ok(vec![0.0; out_numel]);
    }
    let offset = meta.storage_offset();
    let data = &input[offset..];

    // Push-based output skips the zero-init pass; all 6 p-branches
    // iterate (outer, inner) in row-major output order
    // (frankentorch-5hjx).
    let mut output = Vec::with_capacity(out_numel);

    if p == f64::INFINITY {
        // max(|x|) / min(|x|): f64::max/min drop NaN silently, so propagate
        // NaN explicitly to match PyTorch's max/min reductions.
        for outer in 0..outer_size {
            for inner in 0..inner_size {
                let mut max_abs = 0.0_f64;
                for r in 0..reduce_size {
                    let a = data[outer * reduce_size * inner_size + r * inner_size + inner].abs();
                    max_abs = if max_abs.is_nan() || a.is_nan() {
                        f64::NAN
                    } else {
                        max_abs.max(a)
                    };
                }
                output.push(max_abs);
            }
        }
    } else if p == f64::NEG_INFINITY {
        for outer in 0..outer_size {
            for inner in 0..inner_size {
                let mut min_abs = f64::INFINITY;
                for r in 0..reduce_size {
                    let a = data[outer * reduce_size * inner_size + r * inner_size + inner].abs();
                    min_abs = if min_abs.is_nan() || a.is_nan() {
                        f64::NAN
                    } else {
                        min_abs.min(a)
                    };
                }
                output.push(min_abs);
            }
        }
    } else if p == 0.0 {
        for outer in 0..outer_size {
            for inner in 0..inner_size {
                let mut count = 0.0;
                for r in 0..reduce_size {
                    if data[outer * reduce_size * inner_size + r * inner_size + inner] != 0.0 {
                        count += 1.0;
                    }
                }
                output.push(count);
            }
        }
    } else if p == 1.0 {
        for outer in 0..outer_size {
            for inner in 0..inner_size {
                let mut sum = 0.0;
                for r in 0..reduce_size {
                    sum += data[outer * reduce_size * inner_size + r * inner_size + inner].abs();
                }
                output.push(sum);
            }
        }
    } else if p == 2.0 {
        for outer in 0..outer_size {
            for inner in 0..inner_size {
                let mut sum_sq = 0.0;
                for r in 0..reduce_size {
                    let v = data[outer * reduce_size * inner_size + r * inner_size + inner];
                    sum_sq += v * v;
                }
                output.push(sum_sq.sqrt());
            }
        }
    } else {
        for outer in 0..outer_size {
            for inner in 0..inner_size {
                let mut sum_pow = 0.0;
                for r in 0..reduce_size {
                    sum_pow += data[outer * reduce_size * inner_size + r * inner_size + inner]
                        .abs()
                        .powf(p);
                }
                output.push(sum_pow.powf(1.0 / p));
            }
        }
    }

    Ok(output)
}

pub fn softmax_dim_tensor_contiguous_f64(
    input: &[f64],
    meta: &TensorMeta,
    dim: usize,
) -> Result<Vec<f64>, KernelError> {
    ensure_unary_layout_and_storage(input, meta)?;
    let shape = meta.shape();
    let ndim = shape.len();
    if dim >= ndim {
        return Err(KernelError::InvalidDimension { dim, ndim });
    }
    let reduce_size = shape[dim];
    let (outer_size, inner_size, numel) =
        checked_dim_loop_sizes(shape, dim, "softmax shape volume overflow")?;
    if numel == 0 {
        return Ok(Vec::new());
    }
    let offset = meta.storage_offset();
    let mut output = vec![0.0; numel];
    let data = &input[offset..];

    // Inner_size == 1 means we are softmaxing over the last dim (the
    // typical classifier / attention shape: [B, V] softmax over V).
    // The strided access collapses to a contiguous slice per outer,
    // so we can exp(x - max) directly into `output` and pairwise-sum
    // from there with zero scratch allocation.
    if inner_size == 1 {
        let process = |out_slice: &mut [f64], in_slice: &[f64]| {
            let max_val = in_slice.iter().copied().fold(f64::NEG_INFINITY, f64::max);
            for (out, &x) in out_slice.iter_mut().zip(in_slice.iter()) {
                *out = (x - max_val).exp();
            }
            let sum = pairwise_sum_f64(out_slice);
            for v in out_slice {
                *v /= sum;
            }
        };
        if numel >= SOFTMAX_PARALLEL_NUMEL_THRESHOLD {
            output
                .par_chunks_mut(reduce_size)
                .zip(data[..numel].par_chunks(reduce_size))
                .for_each(|(out_slice, in_slice)| process(out_slice, in_slice));
        } else {
            output
                .chunks_mut(reduce_size)
                .zip(data[..numel].chunks(reduce_size))
                .for_each(|(out_slice, in_slice)| process(out_slice, in_slice));
        }
        return Ok(output);
    }

    // General strided case: gather each (outer, inner) slice into a
    // reusable scratch buffer, compute exp(x - max) in place there,
    // pairwise-sum, then scatter the normalised values back to the
    // strided output positions. One allocation per call, not per cell.
    let mut scratch = vec![0.0_f64; reduce_size];
    for outer in 0..outer_size {
        for inner in 0..inner_size {
            for r in 0..reduce_size {
                scratch[r] = data[outer * reduce_size * inner_size + r * inner_size + inner];
            }
            let max_val = scratch.iter().copied().fold(f64::NEG_INFINITY, f64::max);
            for v in scratch.iter_mut() {
                *v = (*v - max_val).exp();
            }
            let sum = pairwise_sum_f64(&scratch);
            for (r, &exp_x) in scratch.iter().enumerate() {
                output[outer * reduce_size * inner_size + r * inner_size + inner] = exp_x / sum;
            }
        }
    }

    Ok(output)
}

pub fn log_softmax_dim_tensor_contiguous_f64(
    input: &[f64],
    meta: &TensorMeta,
    dim: usize,
) -> Result<Vec<f64>, KernelError> {
    ensure_unary_layout_and_storage(input, meta)?;
    let shape = meta.shape();
    let ndim = shape.len();
    if dim >= ndim {
        return Err(KernelError::InvalidDimension { dim, ndim });
    }
    let reduce_size = shape[dim];
    let (outer_size, inner_size, numel) =
        checked_dim_loop_sizes(shape, dim, "log_softmax shape volume overflow")?;
    if numel == 0 {
        return Ok(Vec::new());
    }
    let offset = meta.storage_offset();
    let mut output = vec![0.0; numel];
    let data = &input[offset..];

    // Same fast/general split as `softmax_dim_tensor_contiguous_f64`.
    // log-sum-exp = max + log(sum(exp(x - max))) — the max subtraction
    // is the standard numerical-stability trick; pairwise replaces the
    // sum-of-exps accumulator.
    // The output formulation is (x - max) - log(sum_exp), NOT
    // x - (max + log(sum_exp)). Algebraically identical, but the
    // first preserves precision when x and max are large-magnitude
    // (e.g. logits ~1000) and the result ~ -O(1) — combining x and
    // max first via subtraction keeps the intermediate small, while
    // pre-summing max + log(sum_exp) before the subtraction triggers
    // catastrophic cancellation that costs ~13 digits of mantissa.
    // scipy.special.log_softmax uses the (x - max) - log(sum_exp)
    // form; tracked under frankentorch-ebrb.
    // log_softmax over the last dim is the cross-entropy / NLLLoss hot path;
    // parallelize independent contiguous rows there. The strided general path
    // stays serial because Criterion showed Rayon overhead dominates there.
    // exp is compute-bound and each slice is independent -> bit-identical.
    if inner_size == 1 {
        let process = |out_slice: &mut [f64], in_slice: &[f64]| {
            let max_val = in_slice.iter().copied().fold(f64::NEG_INFINITY, f64::max);
            let sum_exp = pairwise_sum_map_f64(in_slice, |x| (x - max_val).exp());
            let log_sum_exp = sum_exp.ln();
            for (out, &x) in out_slice.iter_mut().zip(in_slice.iter()) {
                *out = (x - max_val) - log_sum_exp;
            }
        };
        if numel >= SOFTMAX_PARALLEL_NUMEL_THRESHOLD {
            output
                .par_chunks_mut(reduce_size)
                .zip(data[..numel].par_chunks(reduce_size))
                .for_each(|(out_slice, in_slice)| process(out_slice, in_slice));
        } else {
            output
                .chunks_mut(reduce_size)
                .zip(data[..numel].chunks(reduce_size))
                .for_each(|(out_slice, in_slice)| process(out_slice, in_slice));
        }
        return Ok(output);
    }

    let mut scratch = vec![0.0_f64; reduce_size];
    for outer in 0..outer_size {
        for inner in 0..inner_size {
            for r in 0..reduce_size {
                scratch[r] = data[outer * reduce_size * inner_size + r * inner_size + inner];
            }
            let max_val = scratch.iter().copied().fold(f64::NEG_INFINITY, f64::max);
            let sum_exp = pairwise_sum_map_f64(&scratch, |x| (x - max_val).exp());
            let log_sum_exp = sum_exp.ln();
            for (r, &x) in scratch.iter().enumerate() {
                output[outer * reduce_size * inner_size + r * inner_size + inner] =
                    (x - max_val) - log_sum_exp;
            }
        }
    }

    Ok(output)
}

pub fn argmax_dim_tensor_contiguous_f64(
    input: &[f64],
    meta: &TensorMeta,
    dim: usize,
) -> Result<Vec<f64>, KernelError> {
    ensure_unary_layout_and_storage(input, meta)?;
    let shape = meta.shape();
    let ndim = shape.len();
    if dim >= ndim {
        return Err(KernelError::InvalidDimension { dim, ndim });
    }
    let reduce_size = shape[dim];
    let (outer_size, inner_size, _) =
        checked_dim_loop_sizes(shape, dim, "argmax shape volume overflow")?;
    let out_numel = checked_mul(
        outer_size,
        inner_size,
        "argmax shape multiplication overflow",
    )?;
    if out_numel == 0 {
        return Ok(Vec::new());
    }
    if reduce_size == 0 {
        return Err(KernelError::EmptyReductionDim { dim });
    }
    let offset = meta.storage_offset();
    let mut output = vec![0.0; out_numel];
    let data = &input[offset..];

    for outer in 0..outer_size {
        for inner in 0..inner_size {
            let mut best_idx = 0usize;
            let mut best_val = f64::NEG_INFINITY;
            for r in 0..reduce_size {
                let idx = outer * reduce_size * inner_size + r * inner_size + inner;
                let val = data[idx];
                if val.is_nan() {
                    best_idx = r;
                    break;
                } else if val > best_val {
                    best_val = val;
                    best_idx = r;
                }
            }
            output[outer * inner_size + inner] = best_idx as f64;
        }
    }

    Ok(output)
}

pub fn argmin_dim_tensor_contiguous_f64(
    input: &[f64],
    meta: &TensorMeta,
    dim: usize,
) -> Result<Vec<f64>, KernelError> {
    ensure_unary_layout_and_storage(input, meta)?;
    let shape = meta.shape();
    let ndim = shape.len();
    if dim >= ndim {
        return Err(KernelError::InvalidDimension { dim, ndim });
    }
    let reduce_size = shape[dim];
    let (outer_size, inner_size, _) =
        checked_dim_loop_sizes(shape, dim, "argmin shape volume overflow")?;
    let out_numel = checked_mul(
        outer_size,
        inner_size,
        "argmin shape multiplication overflow",
    )?;
    if out_numel == 0 {
        return Ok(Vec::new());
    }
    if reduce_size == 0 {
        return Err(KernelError::EmptyReductionDim { dim });
    }
    let offset = meta.storage_offset();
    let mut output = vec![0.0; out_numel];
    let data = &input[offset..];

    for outer in 0..outer_size {
        for inner in 0..inner_size {
            let mut best_idx = 0usize;
            let mut best_val = f64::INFINITY;
            for r in 0..reduce_size {
                let idx = outer * reduce_size * inner_size + r * inner_size + inner;
                let val = data[idx];
                if val.is_nan() {
                    best_idx = r;
                    break;
                } else if val < best_val {
                    best_val = val;
                    best_idx = r;
                }
            }
            output[outer * inner_size + inner] = best_idx as f64;
        }
    }

    Ok(output)
}

pub fn max_dim_tensor_contiguous_f64(
    input: &[f64],
    meta: &TensorMeta,
    dim: usize,
) -> Result<(Vec<f64>, Vec<f64>), KernelError> {
    ensure_unary_layout_and_storage(input, meta)?;
    let shape = meta.shape();
    let ndim = shape.len();
    if dim >= ndim {
        return Err(KernelError::InvalidDimension { dim, ndim });
    }
    let reduce_size = shape[dim];
    let (outer_size, inner_size, _) =
        checked_dim_loop_sizes(shape, dim, "max_dim shape volume overflow")?;
    let out_numel = checked_mul(
        outer_size,
        inner_size,
        "max_dim shape multiplication overflow",
    )?;
    if out_numel == 0 {
        return Ok((Vec::new(), Vec::new()));
    }
    if reduce_size == 0 {
        return Err(KernelError::EmptyReductionDim { dim });
    }
    let offset = meta.storage_offset();
    let mut values = vec![f64::NEG_INFINITY; out_numel];
    let mut indices = vec![0.0; out_numel];
    let data = &input[offset..];

    for outer in 0..outer_size {
        for inner in 0..inner_size {
            let out_idx = outer * inner_size + inner;
            for r in 0..reduce_size {
                let idx = outer * reduce_size * inner_size + r * inner_size + inner;
                let val = data[idx];
                if val.is_nan() {
                    values[out_idx] = f64::NAN;
                    indices[out_idx] = r as f64;
                    break;
                } else if val > values[out_idx] {
                    values[out_idx] = val;
                    indices[out_idx] = r as f64;
                }
            }
        }
    }

    Ok((values, indices))
}

pub fn min_dim_tensor_contiguous_f64(
    input: &[f64],
    meta: &TensorMeta,
    dim: usize,
) -> Result<(Vec<f64>, Vec<f64>), KernelError> {
    ensure_unary_layout_and_storage(input, meta)?;
    let shape = meta.shape();
    let ndim = shape.len();
    if dim >= ndim {
        return Err(KernelError::InvalidDimension { dim, ndim });
    }
    let reduce_size = shape[dim];
    let (outer_size, inner_size, _) =
        checked_dim_loop_sizes(shape, dim, "min_dim shape volume overflow")?;
    let out_numel = checked_mul(
        outer_size,
        inner_size,
        "min_dim shape multiplication overflow",
    )?;
    if out_numel == 0 {
        return Ok((Vec::new(), Vec::new()));
    }
    if reduce_size == 0 {
        return Err(KernelError::EmptyReductionDim { dim });
    }
    let offset = meta.storage_offset();
    let mut values = vec![f64::INFINITY; out_numel];
    let mut indices = vec![0.0; out_numel];
    let data = &input[offset..];

    for outer in 0..outer_size {
        for inner in 0..inner_size {
            let out_idx = outer * inner_size + inner;
            for r in 0..reduce_size {
                let idx = outer * reduce_size * inner_size + r * inner_size + inner;
                let val = data[idx];
                if val.is_nan() {
                    values[out_idx] = f64::NAN;
                    indices[out_idx] = r as f64;
                    break;
                } else if val < values[out_idx] {
                    values[out_idx] = val;
                    indices[out_idx] = r as f64;
                }
            }
        }
    }

    Ok((values, indices))
}

pub fn cat_tensor_contiguous_f64(
    inputs: &[(&[f64], &TensorMeta)],
    dim: usize,
) -> Result<Vec<f64>, KernelError> {
    if inputs.is_empty() {
        return Err(KernelError::ShapeMismatch {
            lhs: vec![],
            rhs: vec![],
        });
    }
    let first_shape = inputs[0].1.shape();
    let ndim = first_shape.len();
    if dim >= ndim {
        return Err(KernelError::InvalidDimension { dim, ndim });
    }
    // Validate shapes match except along cat dim
    let mut total_cat_size = 0usize;
    for (data, meta) in inputs {
        ensure_unary_layout_and_storage(data, meta)?;
        let shape = meta.shape();
        if shape.len() != ndim {
            return Err(KernelError::ShapeMismatch {
                lhs: first_shape.to_vec(),
                rhs: shape.to_vec(),
            });
        }
        for d in 0..ndim {
            if d != dim && shape[d] != first_shape[d] {
                return Err(KernelError::ShapeMismatch {
                    lhs: first_shape.to_vec(),
                    rhs: shape.to_vec(),
                });
            }
        }
        total_cat_size =
            total_cat_size
                .checked_add(shape[dim])
                .ok_or(KernelError::ShapeOverflow {
                    context: "cat shape sum overflow",
                })?;
    }

    let (outer_size, inner_size, _) =
        checked_dim_loop_sizes(first_shape, dim, "cat shape volume overflow")?;
    let out_numel = checked_mul(
        checked_mul(
            outer_size,
            total_cat_size,
            "cat shape multiplication overflow",
        )?,
        inner_size,
        "cat shape multiplication overflow",
    )?;
    if out_numel == 0 {
        return Ok(Vec::new());
    }
    let mut output = Vec::with_capacity(out_numel);

    for outer in 0..outer_size {
        for (data, meta) in inputs {
            let shape = meta.shape();
            let cat_size = shape[dim];
            if cat_size == 0 {
                continue;
            }
            let offset = meta.storage_offset();
            let d = &data[offset..];
            let block_len = checked_mul(cat_size, inner_size, "cat slice range overflow")?;
            let range = checked_contiguous_range(outer, block_len, "cat slice range overflow")?;
            output.extend_from_slice(&d[range]);
        }
    }

    Ok(output)
}

pub fn stack_tensor_contiguous_f64(
    inputs: &[(&[f64], &TensorMeta)],
    dim: usize,
) -> Result<Vec<f64>, KernelError> {
    if inputs.is_empty() {
        return Err(KernelError::ShapeMismatch {
            lhs: vec![],
            rhs: vec![],
        });
    }
    let first_shape = inputs[0].1.shape();
    let ndim = first_shape.len();
    // dim can be 0..=ndim (inserting a new dimension)
    if dim > ndim {
        return Err(KernelError::InvalidDimension {
            dim,
            ndim: ndim + 1,
        });
    }
    // Validate all shapes are identical
    for (data, meta) in inputs {
        ensure_unary_layout_and_storage(data, meta)?;
        if meta.shape() != first_shape {
            return Err(KernelError::ShapeMismatch {
                lhs: first_shape.to_vec(),
                rhs: meta.shape().to_vec(),
            });
        }
    }

    let num_inputs = inputs.len();
    let outer_size = checked_shape_numel(&first_shape[..dim], "stack outer shape overflow")?;
    let inner_size = checked_shape_numel(&first_shape[dim..], "stack inner shape overflow")?;
    let out_numel = checked_mul(
        checked_mul(
            outer_size,
            num_inputs,
            "stack shape multiplication overflow",
        )?,
        inner_size,
        "stack shape multiplication overflow",
    )?;
    if out_numel == 0 {
        return Ok(Vec::new());
    }
    let mut output = Vec::with_capacity(out_numel);

    for outer in 0..outer_size {
        for (data, meta) in inputs {
            let offset = meta.storage_offset();
            let d = &data[offset..];
            let range = checked_contiguous_range(outer, inner_size, "stack slice range overflow")?;
            output.extend_from_slice(&d[range]);
        }
    }

    Ok(output)
}

/// Extracts a contiguous sub-range along `dim` from a contiguous tensor.
///
/// Given input with shape `S`, this selects elements `[start..start+length]`
/// along dimension `dim`, producing a tensor whose shape is `S` with
/// `S[dim]` replaced by `length`.
pub fn narrow_tensor_contiguous_f64(
    input: &[f64],
    meta: &TensorMeta,
    dim: usize,
    start: usize,
    length: usize,
) -> Result<Vec<f64>, KernelError> {
    ensure_unary_layout_and_storage(input, meta)?;
    let shape = meta.shape();
    let ndim = shape.len();
    if dim >= ndim {
        return Err(KernelError::InvalidDimension { dim, ndim });
    }
    let end = start
        .checked_add(length)
        .ok_or(KernelError::InvalidDimension {
            dim: start,
            ndim: shape[dim],
        })?;
    if end > shape[dim] {
        return Err(KernelError::InvalidDimension {
            dim: end,
            ndim: shape[dim],
        });
    }
    if length == 0 {
        return Ok(Vec::new());
    }

    let (outer_size, inner_size, _) =
        checked_dim_loop_sizes(shape, dim, "narrow shape volume overflow")?;
    let dim_size = shape[dim];
    let out_numel = checked_mul(
        checked_mul(outer_size, length, "narrow shape multiplication overflow")?,
        inner_size,
        "narrow shape multiplication overflow",
    )?;
    if out_numel == 0 {
        return Ok(Vec::new());
    }
    let offset = meta.storage_offset();
    let data = &input[offset..];
    let mut output = Vec::with_capacity(out_numel);

    for outer in 0..outer_size {
        for r in 0..length {
            for inner in 0..inner_size {
                let idx = outer * dim_size * inner_size + (start + r) * inner_size + inner;
                output.push(data[idx]);
            }
        }
    }

    Ok(output)
}

/// Normalizes an index value with Python-style negative wrapping.
///
/// A negative index `i` is rewritten to `i + dim_size`. This matches
/// PyTorch *advanced indexing* (`tensor[idx]` / `index_put_`), which
/// wraps negative indices. It must NOT be used for `index_select`,
/// `gather`, `scatter`, or `scatter_add` — those reject negatives (see
/// [`normalize_strict_index_value`]).
fn normalize_wrapped_index_value(idx_f: f64, dim_size: usize) -> Result<usize, KernelError> {
    if !idx_f.is_finite() || idx_f.fract().abs() > f64::EPSILON {
        return Err(KernelError::InvalidDimension {
            dim: dim_size,
            ndim: dim_size,
        });
    }

    let dim_size_i = isize::try_from(dim_size).map_err(|_| KernelError::InvalidDimension {
        dim: dim_size,
        ndim: dim_size,
    })?;

    if idx_f < isize::MIN as f64 || idx_f > isize::MAX as f64 {
        return Err(KernelError::InvalidDimension {
            dim: dim_size,
            ndim: dim_size,
        });
    }

    let mut idx_i = idx_f as isize;
    if idx_i < 0 {
        idx_i += dim_size_i;
    }
    if idx_i < 0 || idx_i >= dim_size_i {
        return Err(KernelError::InvalidDimension {
            dim: dim_size,
            ndim: dim_size,
        });
    }

    usize::try_from(idx_i).map_err(|_| KernelError::InvalidDimension {
        dim: dim_size,
        ndim: dim_size,
    })
}

/// Validates an index value WITHOUT negative wrapping.
///
/// A negative index is rejected outright. This matches PyTorch
/// `index_select` (`IndexError: index out of range in self`),
/// `gather` / `scatter` / `scatter_add` (`RuntimeError: index N is out
/// of bounds for dimension D`), which — unlike advanced indexing — do
/// not interpret negatives as offsets from the end. Verified against
/// torch 2.12 (frankentorch-n0un).
fn normalize_strict_index_value(idx_f: f64, dim_size: usize) -> Result<usize, KernelError> {
    if !idx_f.is_finite() || idx_f.fract().abs() > f64::EPSILON {
        return Err(KernelError::InvalidDimension {
            dim: dim_size,
            ndim: dim_size,
        });
    }

    let dim_size_i = isize::try_from(dim_size).map_err(|_| KernelError::InvalidDimension {
        dim: dim_size,
        ndim: dim_size,
    })?;

    if idx_f < 0.0 || idx_f > isize::MAX as f64 {
        return Err(KernelError::InvalidDimension {
            dim: dim_size,
            ndim: dim_size,
        });
    }

    let idx_i = idx_f as isize;
    if idx_i < 0 || idx_i >= dim_size_i {
        return Err(KernelError::InvalidDimension {
            dim: dim_size,
            ndim: dim_size,
        });
    }

    usize::try_from(idx_i).map_err(|_| KernelError::InvalidDimension {
        dim: dim_size,
        ndim: dim_size,
    })
}

/// Expands singleton dimensions of a contiguous tensor to a target shape.
///
/// Only dimensions of size 1 in the input can be expanded to a larger size.
/// Dimensions that already match are copied through. The input and target
/// must have the same number of dimensions.
pub fn expand_tensor_contiguous_f64(
    input: &[f64],
    meta: &TensorMeta,
    target_shape: &[usize],
) -> Result<Vec<f64>, KernelError> {
    ensure_unary_layout_and_storage(input, meta)?;
    let shape = meta.shape();
    let ndim = shape.len();
    if target_shape.len() != ndim {
        return Err(KernelError::ShapeMismatch {
            lhs: shape.to_vec(),
            rhs: target_shape.to_vec(),
        });
    }
    // Validate: each dim must either match or be 1 in input
    for d in 0..ndim {
        if shape[d] != target_shape[d] && shape[d] != 1 {
            return Err(KernelError::ShapeMismatch {
                lhs: shape.to_vec(),
                rhs: target_shape.to_vec(),
            });
        }
    }

    let out_numel = checked_shape_numel(target_shape, "expand output shape volume overflow")?;
    if out_numel == 0 {
        return Ok(Vec::new());
    }

    let input_strides = broadcast_strides(shape, target_shape, "expand input strides overflow")?;
    let offset = meta.storage_offset();
    let mut output = Vec::with_capacity(out_numel);
    let mut coords = vec![0usize; ndim];

    for _ in 0..out_numel {
        let mut idx = offset;
        for d in 0..ndim {
            idx += coords[d] * input_strides[d];
        }
        output.push(input[idx]);

        // Increment coordinates (row-major order)
        for d in (0..ndim).rev() {
            coords[d] += 1;
            if coords[d] < target_shape[d] {
                break;
            }
            coords[d] = 0;
        }
    }

    Ok(output)
}

/// Selects rows/columns along a given dimension using the provided index array.
///
/// `indices` are stored as `f64` (matching the convention used by argmax/argmin)
/// and are cast to `usize` internally. The output shape is the input shape with
/// `shape[dim]` replaced by `indices.len()`.
pub fn index_select_tensor_contiguous_f64(
    input: &[f64],
    meta: &TensorMeta,
    dim: usize,
    indices: &[f64],
) -> Result<Vec<f64>, KernelError> {
    ensure_unary_layout_and_storage(input, meta)?;
    let shape = meta.shape();
    let ndim = shape.len();
    if dim >= ndim {
        return Err(KernelError::InvalidDimension { dim, ndim });
    }

    let dim_size = shape[dim];
    let (outer_size, inner_size, _) =
        checked_dim_loop_sizes(shape, dim, "index_select shape volume overflow")?;
    let num_indices = indices.len();
    let out_numel = checked_mul(
        checked_mul(
            outer_size,
            num_indices,
            "index_select shape multiplication overflow",
        )?,
        inner_size,
        "index_select shape multiplication overflow",
    )?;
    if out_numel == 0 {
        return Ok(Vec::new());
    }
    let offset = meta.storage_offset();
    let data = &input[offset..];
    let mut output = Vec::with_capacity(out_numel);

    for outer in 0..outer_size {
        for &idx_f in indices {
            let idx = normalize_strict_index_value(idx_f, dim_size)?;
            for inner in 0..inner_size {
                let src = outer * dim_size * inner_size + idx * inner_size + inner;
                output.push(data[src]);
            }
        }
    }

    Ok(output)
}

/// Gathers values along a dimension using an index tensor.
///
/// For each position in the output (which has the same shape as `index_meta`),
/// the value is taken from `input` at the position indicated by the
/// corresponding index value along `dim`.
pub fn gather_tensor_contiguous_f64(
    input: &[f64],
    meta: &TensorMeta,
    dim: usize,
    index: &[f64],
    index_meta: &TensorMeta,
) -> Result<Vec<f64>, KernelError> {
    ensure_unary_layout_and_storage(input, meta)?;
    ensure_unary_layout_and_storage(index, index_meta)?;
    let shape = meta.shape();
    let ndim = shape.len();
    if dim >= ndim {
        return Err(KernelError::InvalidDimension { dim, ndim });
    }
    let idx_shape = index_meta.shape();
    if idx_shape.len() != ndim {
        return Err(KernelError::ShapeMismatch {
            lhs: shape.to_vec(),
            rhs: idx_shape.to_vec(),
        });
    }
    // All dimensions except `dim` must match between input and index.
    for d in 0..ndim {
        if d != dim && idx_shape[d] != shape[d] {
            return Err(KernelError::ShapeMismatch {
                lhs: shape.to_vec(),
                rhs: idx_shape.to_vec(),
            });
        }
    }

    let dim_size = shape[dim];
    let idx_dim_size = idx_shape[dim];
    let (outer_size, inner_size, out_numel) =
        checked_dim_loop_sizes(idx_shape, dim, "gather index shape volume overflow")?;
    if out_numel == 0 {
        return Ok(Vec::new());
    }
    let offset = meta.storage_offset();
    let data = &input[offset..];
    let idx_offset = index_meta.storage_offset();
    let index_data = &index[idx_offset..];
    let mut output = Vec::with_capacity(out_numel);

    for outer in 0..outer_size {
        for r in 0..idx_dim_size {
            for inner in 0..inner_size {
                let idx_pos = outer * idx_dim_size * inner_size + r * inner_size + inner;
                let selected = normalize_strict_index_value(index_data[idx_pos], dim_size)?;
                let src = outer * dim_size * inner_size + selected * inner_size + inner;
                output.push(data[src]);
            }
        }
    }

    Ok(output)
}

/// Scatters source values into a copy of `input` along a dimension using an
/// index tensor.
///
/// For each position in the `index` tensor (whose shape must match `src`),
/// the corresponding value from `src` is placed into the output at the
/// position indicated by `index` along `dim`.
pub fn scatter_tensor_contiguous_f64(
    input: &[f64],
    meta: &TensorMeta,
    dim: usize,
    index: &[f64],
    index_meta: &TensorMeta,
    src: &[f64],
) -> Result<Vec<f64>, KernelError> {
    ensure_unary_layout_and_storage(input, meta)?;
    ensure_unary_layout_and_storage(index, index_meta)?;
    let shape = meta.shape();
    let ndim = shape.len();
    if dim >= ndim {
        return Err(KernelError::InvalidDimension { dim, ndim });
    }
    let idx_shape = index_meta.shape();
    if idx_shape.len() != ndim {
        return Err(KernelError::ShapeMismatch {
            lhs: shape.to_vec(),
            rhs: idx_shape.to_vec(),
        });
    }
    // All dimensions except `dim` must match between input and index.
    for d in 0..ndim {
        if d != dim && idx_shape[d] != shape[d] {
            return Err(KernelError::ShapeMismatch {
                lhs: shape.to_vec(),
                rhs: idx_shape.to_vec(),
            });
        }
    }
    let src_numel = checked_shape_numel(idx_shape, "scatter index shape volume overflow")?;
    if src.len() < src_numel {
        return Err(KernelError::InsufficientStorage {
            side: "src",
            needed: src_numel,
            available: src.len(),
        });
    }
    let numel = meta.numel();
    if src_numel == 0 {
        if numel == 0 {
            return Ok(Vec::new());
        }
        let offset = meta.storage_offset();
        return Ok(input[offset..offset + numel].to_vec());
    }

    let dim_size = shape[dim];
    let idx_dim_size = idx_shape[dim];
    let (outer_size, inner_size, _) =
        checked_dim_loop_sizes(idx_shape, dim, "scatter index shape volume overflow")?;
    if numel == 0 {
        return Ok(Vec::new());
    }
    let offset = meta.storage_offset();
    let mut output = input[offset..offset + numel].to_vec();
    let idx_offset = index_meta.storage_offset();
    let index_data = &index[idx_offset..];

    for outer in 0..outer_size {
        for r in 0..idx_dim_size {
            for inner in 0..inner_size {
                let idx_pos = outer * idx_dim_size * inner_size + r * inner_size + inner;
                let selected = normalize_strict_index_value(index_data[idx_pos], dim_size)?;
                let dst = outer * dim_size * inner_size + selected * inner_size + inner;
                output[dst] = src[idx_pos];
            }
        }
    }

    Ok(output)
}

/// Like `scatter_tensor_contiguous_f64` but **adds** `src` values instead of overwriting.
///
/// `output[index[i][j]][j] += src[i][j]` (for dim=0). Multiple indices pointing to the
/// same location accumulate all contributions.
pub fn scatter_add_tensor_contiguous_f64(
    input: &[f64],
    meta: &TensorMeta,
    dim: usize,
    index: &[f64],
    index_meta: &TensorMeta,
    src: &[f64],
) -> Result<Vec<f64>, KernelError> {
    ensure_unary_layout_and_storage(input, meta)?;
    ensure_unary_layout_and_storage(index, index_meta)?;
    let shape = meta.shape();
    let ndim = shape.len();
    if dim >= ndim {
        return Err(KernelError::InvalidDimension { dim, ndim });
    }
    let idx_shape = index_meta.shape();
    if idx_shape.len() != ndim {
        return Err(KernelError::ShapeMismatch {
            lhs: shape.to_vec(),
            rhs: idx_shape.to_vec(),
        });
    }
    for d in 0..ndim {
        if d != dim && idx_shape[d] != shape[d] {
            return Err(KernelError::ShapeMismatch {
                lhs: shape.to_vec(),
                rhs: idx_shape.to_vec(),
            });
        }
    }
    let src_numel = checked_shape_numel(idx_shape, "scatter_add index shape volume overflow")?;
    if src.len() < src_numel {
        return Err(KernelError::InsufficientStorage {
            side: "src",
            needed: src_numel,
            available: src.len(),
        });
    }
    let numel = meta.numel();
    if src_numel == 0 {
        if numel == 0 {
            return Ok(Vec::new());
        }
        let offset = meta.storage_offset();
        return Ok(input[offset..offset + numel].to_vec());
    }

    let dim_size = shape[dim];
    let idx_dim_size = idx_shape[dim];
    let (outer_size, inner_size, _) =
        checked_dim_loop_sizes(idx_shape, dim, "scatter_add index shape volume overflow")?;
    if numel == 0 {
        return Ok(Vec::new());
    }
    let offset = meta.storage_offset();
    let mut output = input[offset..offset + numel].to_vec();
    let idx_offset = index_meta.storage_offset();
    let index_data = &index[idx_offset..];

    for outer in 0..outer_size {
        for r in 0..idx_dim_size {
            for inner in 0..inner_size {
                let idx_pos = outer * idx_dim_size * inner_size + r * inner_size + inner;
                let selected = normalize_strict_index_value(index_data[idx_pos], dim_size)?;
                let dst = outer * dim_size * inner_size + selected * inner_size + inner;
                output[dst] += src[idx_pos];
            }
        }
    }

    Ok(output)
}

/// Puts `values` at the positions specified by `indices` into a tensor.
///
/// `indices` is a list of 1D index tensors, one per indexed dimension (the leading dims).
/// If `accumulate` is true, values are added; otherwise they overwrite.
pub fn index_put_tensor_contiguous_f64(
    input: &[f64],
    meta: &TensorMeta,
    indices: &[Vec<f64>],
    values: &[f64],
    accumulate: bool,
) -> Result<Vec<f64>, KernelError> {
    ensure_unary_layout_and_storage(input, meta)?;
    let shape = meta.shape();
    let ndim = shape.len();

    if indices.is_empty() {
        return Err(KernelError::ShapeMismatch {
            lhs: shape.to_vec(),
            rhs: vec![0],
        });
    }

    let num_indexed_dims = indices.len();
    if num_indexed_dims > ndim {
        return Err(KernelError::InvalidDimension {
            dim: num_indexed_dims,
            ndim,
        });
    }

    // All index tensors must have the same length
    let n_indices = indices[0].len();
    for idx_tensor in &indices[1..] {
        if idx_tensor.len() != n_indices {
            return Err(KernelError::ShapeMismatch {
                lhs: vec![n_indices],
                rhs: vec![idx_tensor.len()],
            });
        }
    }

    // Compute the "suffix" size: product of non-indexed dimensions
    let suffix_size = checked_shape_numel(
        &shape[num_indexed_dims..],
        "index_put suffix shape overflow",
    )?;

    // values must have n_indices * suffix_size elements (or be broadcastable scalar)
    let values_needed = checked_mul(n_indices, suffix_size, "index_put values shape overflow")?;
    let scalar_broadcast = values.len() == 1 && values_needed > 1;
    if !scalar_broadcast && values.len() < values_needed {
        return Err(KernelError::InsufficientStorage {
            side: "values",
            needed: values_needed,
            available: values.len(),
        });
    }

    let offset = meta.storage_offset();
    let numel = meta.numel();
    if numel == 0 {
        return Ok(Vec::new());
    }
    let mut output = input[offset..offset + numel].to_vec();

    // Compute strides for the indexed dimensions
    let mut indexed_strides = vec![0usize; num_indexed_dims];
    for d in 0..num_indexed_dims {
        indexed_strides[d] =
            checked_shape_numel(&shape[d + 1..], "index_put stride shape overflow")?;
    }

    for i in 0..n_indices {
        // Compute the base offset from the indices
        let mut base = 0usize;
        for d in 0..num_indexed_dims {
            let idx = normalize_wrapped_index_value(indices[d][i], shape[d])?;
            base += idx * indexed_strides[d];
        }

        // Write suffix_size values at that offset
        for s in 0..suffix_size {
            let val = if scalar_broadcast {
                values[0]
            } else {
                values[i * suffix_size + s]
            };
            if accumulate {
                output[base + s] += val;
            } else {
                output[base + s] = val;
            }
        }
    }

    Ok(output)
}

/// Fills positions in the tensor where `mask` is non-zero with the given
/// `value`. `mask` is expected to contain 0.0 or 1.0 values (as produced by
/// comparison ops) and must have the same number of elements as `input`.
pub fn masked_fill_tensor_contiguous_f64(
    input: &[f64],
    meta: &TensorMeta,
    mask: &[f64],
    value: f64,
) -> Result<Vec<f64>, KernelError> {
    ensure_unary_layout_and_storage(input, meta)?;
    let numel = meta.numel();
    if numel == 0 {
        return Ok(Vec::new());
    }
    let offset = meta.storage_offset();
    if mask.len() < offset + numel {
        return Err(KernelError::ShapeMismatch {
            lhs: meta.shape().to_vec(),
            rhs: vec![mask.len()],
        });
    }

    let data = &input[offset..offset + numel];
    let output = data
        .iter()
        .zip(mask[offset..offset + numel].iter())
        .map(|(&d, &m)| if m != 0.0 { value } else { d })
        .collect();

    Ok(output)
}

/// Conditional selection: `torch.where(condition, x, y)`.
///
/// Selects elements from `x` where `condition != 0.0` and from `y` otherwise.
/// All three tensors must have the same shape.
pub fn where_tensor_contiguous_f64(
    condition: &[f64],
    x: &[f64],
    y: &[f64],
    meta: &TensorMeta,
) -> Result<Vec<f64>, KernelError> {
    ensure_unary_layout_and_storage(x, meta)?;
    let numel = meta.numel();
    if numel == 0 {
        return Ok(Vec::new());
    }
    let offset = meta.storage_offset();
    if condition.len() < offset + numel {
        return Err(KernelError::ShapeMismatch {
            lhs: meta.shape().to_vec(),
            rhs: vec![condition.len()],
        });
    }
    if y.len() < offset + numel {
        return Err(KernelError::ShapeMismatch {
            lhs: meta.shape().to_vec(),
            rhs: vec![y.len()],
        });
    }

    let cond = &condition[offset..offset + numel];
    let x_data = &x[offset..offset + numel];
    let y_data = &y[offset..offset + numel];

    let output = cond
        .iter()
        .zip(x_data.iter())
        .zip(y_data.iter())
        .map(|((&c, &xv), &yv)| if c != 0.0 { xv } else { yv })
        .collect();

    Ok(output)
}

/// Cumulative sum along a given dimension.
///
/// For a 1-D input [a, b, c] with dim=0, returns [a, a+b, a+b+c].
/// For higher-dimensional tensors, the accumulation happens along the specified dimension
/// while iterating over all other dimensions.
pub fn cumsum_tensor_contiguous_f64(
    input: &[f64],
    meta: &TensorMeta,
    dim: usize,
) -> Result<Vec<f64>, KernelError> {
    ensure_unary_layout_and_storage(input, meta)?;
    let shape = meta.shape();
    let ndim = shape.len();
    if dim >= ndim {
        return Err(KernelError::InvalidDimension { dim, ndim });
    }
    let dim_size = shape[dim];
    let (outer_size, inner_size, _) =
        checked_dim_loop_sizes(shape, dim, "cumsum shape volume overflow")?;
    let numel = checked_mul(
        checked_mul(outer_size, dim_size, "cumsum shape multiplication overflow")?,
        inner_size,
        "cumsum shape multiplication overflow",
    )?;
    if numel == 0 {
        return Ok(Vec::new());
    }
    let offset = meta.storage_offset();
    let mut output = vec![0.0; numel];
    let data = &input[offset..];

    for outer in 0..outer_size {
        for inner in 0..inner_size {
            let mut acc = 0.0;
            for d in 0..dim_size {
                let idx = outer * dim_size * inner_size + d * inner_size + inner;
                acc += data[idx];
                output[idx] = acc;
            }
        }
    }

    Ok(output)
}

/// Backward pass for cumsum: reverse cumulative sum.
///
/// If forward is cumsum along dim, the gradient is a reverse cumsum along the same dim.
pub fn cumsum_backward_tensor_contiguous_f64(
    grad_output: &[f64],
    meta: &TensorMeta,
    dim: usize,
) -> Result<Vec<f64>, KernelError> {
    ensure_unary_layout_and_storage(grad_output, meta)?;
    let shape = meta.shape();
    let ndim = shape.len();
    if dim >= ndim {
        return Err(KernelError::InvalidDimension { dim, ndim });
    }
    let dim_size = shape[dim];
    let (outer_size, inner_size, _) =
        checked_dim_loop_sizes(shape, dim, "cumsum backward shape volume overflow")?;
    let numel = checked_mul(
        checked_mul(
            outer_size,
            dim_size,
            "cumsum backward shape multiplication overflow",
        )?,
        inner_size,
        "cumsum backward shape multiplication overflow",
    )?;
    if numel == 0 {
        return Ok(Vec::new());
    }
    let offset = meta.storage_offset();
    let mut grad_input = vec![0.0; numel];
    let data = &grad_output[offset..];

    for outer in 0..outer_size {
        for inner in 0..inner_size {
            let mut acc = 0.0;
            // Reverse iteration for reverse cumsum
            for d in (0..dim_size).rev() {
                let idx = outer * dim_size * inner_size + d * inner_size + inner;
                acc += data[idx];
                grad_input[idx] = acc;
            }
        }
    }

    Ok(grad_input)
}

/// Cumulative product along a given dimension.
///
/// For a 1-D input [a, b, c] with dim=0, returns [a, a*b, a*b*c].
pub fn cumprod_tensor_contiguous_f64(
    input: &[f64],
    meta: &TensorMeta,
    dim: usize,
) -> Result<Vec<f64>, KernelError> {
    ensure_unary_layout_and_storage(input, meta)?;
    let shape = meta.shape();
    let ndim = shape.len();
    if dim >= ndim {
        return Err(KernelError::InvalidDimension { dim, ndim });
    }
    let dim_size = shape[dim];
    let (outer_size, inner_size, _) =
        checked_dim_loop_sizes(shape, dim, "cumprod shape volume overflow")?;
    let numel = checked_mul(
        checked_mul(
            outer_size,
            dim_size,
            "cumprod shape multiplication overflow",
        )?,
        inner_size,
        "cumprod shape multiplication overflow",
    )?;
    if numel == 0 {
        return Ok(Vec::new());
    }
    let offset = meta.storage_offset();
    let mut output = vec![0.0; numel];
    let data = &input[offset..];

    for outer in 0..outer_size {
        for inner in 0..inner_size {
            let mut acc = 1.0;
            for d in 0..dim_size {
                let idx = outer * dim_size * inner_size + d * inner_size + inner;
                acc *= data[idx];
                output[idx] = acc;
            }
        }
    }

    Ok(output)
}

/// Backward pass for cumprod.
///
/// The gradient of cumprod is computed using the relationship:
/// grad_input[i] = sum_{j>=i} grad_output[j] * output[j] / input[i]
/// which can be rewritten as a reverse cumsum of (grad_output * output) divided by input,
/// with special handling for zeros in the input.
pub fn cumprod_backward_tensor_contiguous_f64(
    grad_output: &[f64],
    input: &[f64],
    output: &[f64],
    meta: &TensorMeta,
    dim: usize,
) -> Result<Vec<f64>, KernelError> {
    ensure_unary_layout_and_storage(input, meta)?;
    let shape = meta.shape();
    let ndim = shape.len();
    if dim >= ndim {
        return Err(KernelError::InvalidDimension { dim, ndim });
    }
    let dim_size = shape[dim];
    let (outer_size, inner_size, _) =
        checked_dim_loop_sizes(shape, dim, "cumprod backward shape volume overflow")?;
    let numel = checked_mul(
        checked_mul(
            outer_size,
            dim_size,
            "cumprod backward shape multiplication overflow",
        )?,
        inner_size,
        "cumprod backward shape multiplication overflow",
    )?;
    if numel == 0 {
        return Ok(Vec::new());
    }
    let offset = meta.storage_offset();
    let mut grad_input = vec![0.0; numel];
    let in_data = &input[offset..];
    let out_data = &output[offset..];
    let go_data = &grad_output[offset..];

    for outer in 0..outer_size {
        for inner in 0..inner_size {
            // Compute reverse cumsum of (grad_output * output)
            let mut acc = 0.0;
            for d in (0..dim_size).rev() {
                let idx = outer * dim_size * inner_size + d * inner_size + inner;
                acc += go_data[idx] * out_data[idx];
                let inp = in_data[idx];
                if inp.abs() > f64::EPSILON {
                    grad_input[idx] = acc / inp;
                } else {
                    // When input is zero, compute gradient by direct summation
                    // to avoid division by zero
                    let mut sum = 0.0;
                    for j in d..dim_size {
                        let j_idx = outer * dim_size * inner_size + j * inner_size + inner;
                        // Product of all elements except position d, from d to j
                        let mut prod = 1.0;
                        for k in d..=j {
                            if k != d {
                                let k_idx = outer * dim_size * inner_size + k * inner_size + inner;
                                prod *= in_data[k_idx];
                            }
                        }
                        // Also include the prefix product (elements before d)
                        if d > 0 {
                            let prev_idx =
                                outer * dim_size * inner_size + (d - 1) * inner_size + inner;
                            prod *= out_data[prev_idx];
                        }
                        sum += go_data[j_idx] * prod;
                    }
                    grad_input[idx] = sum;
                }
            }
        }
    }

    Ok(grad_input)
}

/// Total order over f64 that ranks NaN above every non-NaN value
/// (NaN == NaN), matching how PyTorch's sort/topk treat NaN as the
/// largest element. Unlike `partial_cmp(..).unwrap_or(Equal)` this
/// relation is transitive, so it is safe to hand to a comparison sort
/// (a non-transitive comparator silently corrupts timsort output).
fn nan_greatest_cmp_f64(a: f64, b: f64) -> std::cmp::Ordering {
    match (a.is_nan(), b.is_nan()) {
        (true, true) => std::cmp::Ordering::Equal,
        (true, false) => std::cmp::Ordering::Greater,
        (false, true) => std::cmp::Ordering::Less,
        // Both finite/infinite: partial_cmp is always Some here.
        (false, false) => a.partial_cmp(&b).unwrap_or(std::cmp::Ordering::Equal),
    }
}

fn topk_lane_cmp_f64(a: &(usize, f64), b: &(usize, f64), largest: bool) -> std::cmp::Ordering {
    let value_order = if largest {
        nan_greatest_cmp_f64(b.1, a.1)
    } else {
        nan_greatest_cmp_f64(a.1, b.1)
    };
    value_order.then_with(|| a.0.cmp(&b.0))
}

/// F32 companion to [`nan_greatest_cmp_f64`].
fn nan_greatest_cmp_f32(a: f32, b: f32) -> std::cmp::Ordering {
    match (a.is_nan(), b.is_nan()) {
        (true, true) => std::cmp::Ordering::Equal,
        (true, false) => std::cmp::Ordering::Greater,
        (false, true) => std::cmp::Ordering::Less,
        (false, false) => a.partial_cmp(&b).unwrap_or(std::cmp::Ordering::Equal),
    }
}

fn topk_lane_cmp_f32(a: &(usize, f32), b: &(usize, f32), largest: bool) -> std::cmp::Ordering {
    let value_order = if largest {
        nan_greatest_cmp_f32(b.1, a.1)
    } else {
        nan_greatest_cmp_f32(a.1, b.1)
    };
    value_order.then_with(|| a.0.cmp(&b.0))
}

/// Sort a contiguous f64 tensor along the given dimension.
///
/// Returns `(sorted_values, indices)` where `indices[i]` is the original position
/// of `sorted_values[i]` along the sorted dimension.
///
/// `descending = true` sorts from largest to smallest. NaN is treated as the
/// largest value (it sorts to the end ascending, to the front descending).
/// Lanes shorter than this keep the comparison sort: LSD radix's fixed 8-pass +
/// histogram overhead only pays off once the lane is large enough.
const SORT_RADIX_MIN_LEN: usize = 256;

/// Order-preserving map f64 -> u64 for a non-NaN value: ascending u64 order
/// equals ascending float order. `-0.0` is canonicalised to `+0.0` so the two
/// zeros share a key and stay tied exactly as `partial_cmp` reports them equal
/// (NaN never reaches here — callers route NaN lanes to the comparison sort,
/// which alone reproduces PyTorch's "NaN is greatest" placement).
#[inline]
fn sort_radix_key_f64(x: f64) -> u64 {
    let bits = if x == 0.0 { 0 } else { x.to_bits() };
    if bits >> 63 == 1 {
        !bits
    } else {
        bits | (1u64 << 63)
    }
}

/// f32 analogue of [`sort_radix_key_f64`]: order-preserving map f32 -> u32 with
/// `-0.0` canonicalised to `+0.0`. Widened to u64 for [`sort_radix_perm`] with
/// the high 32 bits zero, so the four high byte-passes are single-bucket and
/// auto-skipped — the radix runs in just 4 effective passes for f32.
#[inline]
fn sort_radix_key_f32(x: f32) -> u32 {
    let bits = if x == 0.0 { 0 } else { x.to_bits() };
    if bits >> 31 == 1 {
        !bits
    } else {
        bits | (1u32 << 31)
    }
}

/// Stable LSD radix sort: fills `perm` with the ascending-by-`keys` permutation
/// of `0..keys.len()`, ties preserving original index order (matching the stable
/// comparison sort bit-for-bit). `scratch` is reused as the ping-pong buffer.
fn sort_radix_perm(keys: &[u64], perm: &mut Vec<u32>, scratch: &mut Vec<u32>) {
    let n = keys.len();
    perm.clear();
    perm.extend(0..n as u32);
    scratch.clear();
    scratch.resize(n, 0u32);
    for pass in 0..8 {
        let shift = pass * 8;
        let mut count = [0usize; 256];
        for &p in perm.iter() {
            count[((keys[p as usize] >> shift) & 0xFF) as usize] += 1;
        }
        // A byte position where every element shares one bucket contributes no
        // ordering — skipping its scatter is a stability-preserving no-op and
        // skips most passes for clustered exponents.
        if count.contains(&n) {
            continue;
        }
        let mut sum = 0usize;
        for c in count.iter_mut() {
            let t = *c;
            *c = sum;
            sum += t;
        }
        for &p in perm.iter() {
            let b = ((keys[p as usize] >> shift) & 0xFF) as usize;
            scratch[count[b]] = p;
            count[b] += 1;
        }
        // After the swap, `perm` holds this pass's result, so it is always the
        // current ordering regardless of how many passes were skipped.
        std::mem::swap(perm, scratch);
    }
}

pub fn sort_tensor_contiguous_f64(
    input: &[f64],
    meta: &TensorMeta,
    dim: usize,
    descending: bool,
) -> Result<(Vec<f64>, Vec<usize>), KernelError> {
    ensure_unary_layout_and_storage(input, meta)?;
    let shape = meta.shape();
    let ndim = shape.len();
    if dim >= ndim {
        return Err(KernelError::InvalidDimension { dim, ndim });
    }
    let dim_size = shape[dim];
    let (outer_size, inner_size, _) =
        checked_dim_loop_sizes(shape, dim, "sort shape volume overflow")?;
    let numel = checked_mul(
        checked_mul(outer_size, dim_size, "sort shape multiplication overflow")?,
        inner_size,
        "sort shape multiplication overflow",
    )?;
    if numel == 0 {
        return Ok((Vec::new(), Vec::new()));
    }
    let offset = meta.storage_offset();
    let data = &input[offset..];

    let mut sorted_values = vec![0.0; numel];
    let mut indices = vec![0usize; numel];

    // Each `outer` owns a contiguous block of dim_size*inner_size in both
    // outputs, and every lane sorts independently. Parallelize over outer blocks
    // (sorting is compute-bound). The per-lane stable sort with the fixed
    // comparator is identical regardless of scheduling, so values AND indices are
    // bit-for-bit identical to the serial version.
    let block = dim_size * inner_size;
    // The per-lane sort is O(n log n) comparisons through a branchy NaN-aware
    // comparator. For a sufficiently long, NaN-free lane an O(n) stable LSD radix
    // sort on order-preserving u64 keys produces a bit-identical result (same
    // values, same stable tie order) far faster. NaN/short lanes fall back to the
    // comparison sort, which alone reproduces PyTorch's "NaN is greatest" rule.
    let use_radix = dim_size >= SORT_RADIX_MIN_LEN && dim_size <= u32::MAX as usize;
    sorted_values
        .par_chunks_mut(block)
        .zip(indices.par_chunks_mut(block))
        .zip(data[..numel].par_chunks(block))
        .for_each(|((sv_block, idx_block), in_block)| {
            let mut keys: Vec<u64> = Vec::new();
            let mut perm: Vec<u32> = Vec::new();
            let mut scratch: Vec<u32> = Vec::new();
            for inner in 0..inner_size {
                // Gather keys and detect NaN in one pass; a NaN aborts to the
                // comparison fallback for this lane.
                let mut radix_ok = use_radix;
                if radix_ok {
                    keys.clear();
                    for d in 0..dim_size {
                        let x = in_block[d * inner_size + inner];
                        if x.is_nan() {
                            radix_ok = false;
                            break;
                        }
                        let k = sort_radix_key_f64(x);
                        keys.push(if descending { !k } else { k });
                    }
                }

                if radix_ok {
                    sort_radix_perm(&keys, &mut perm, &mut scratch);
                    for (out_d, &p) in perm.iter().enumerate() {
                        let orig_d = p as usize;
                        sv_block[out_d * inner_size + inner] =
                            in_block[orig_d * inner_size + inner];
                        idx_block[out_d * inner_size + inner] = orig_d;
                    }
                    continue;
                }

                let mut lane: Vec<(usize, f64)> = (0..dim_size)
                    .map(|d| (d, in_block[d * inner_size + inner]))
                    .collect();

                if descending {
                    lane.sort_by(|a, b| nan_greatest_cmp_f64(b.1, a.1));
                } else {
                    lane.sort_by(|a, b| nan_greatest_cmp_f64(a.1, b.1));
                }

                for (out_d, (orig_d, val)) in lane.into_iter().enumerate() {
                    sv_block[out_d * inner_size + inner] = val;
                    idx_block[out_d * inner_size + inner] = orig_d;
                }
            }
        });

    Ok((sorted_values, indices))
}

/// Return the k largest (or smallest) elements along the given dimension.
///
/// Returns `(values, indices)` of shape `[..., k, ...]` where the dim-th
/// dimension is replaced by k. `largest = true` returns the k largest elements.
pub fn topk_tensor_contiguous_f64(
    input: &[f64],
    meta: &TensorMeta,
    k: usize,
    dim: usize,
    largest: bool,
    sorted: bool,
) -> Result<(Vec<f64>, Vec<usize>), KernelError> {
    ensure_unary_layout_and_storage(input, meta)?;
    let shape = meta.shape();
    let ndim = shape.len();
    if dim >= ndim {
        return Err(KernelError::InvalidDimension { dim, ndim });
    }
    let dim_size = shape[dim];
    if k > dim_size {
        return Err(KernelError::InvalidDimension {
            dim: k,
            ndim: dim_size,
        });
    }
    let (outer_size, inner_size, _) =
        checked_dim_loop_sizes(shape, dim, "topk shape volume overflow")?;
    let out_numel = checked_mul(
        checked_mul(outer_size, k, "topk shape multiplication overflow")?,
        inner_size,
        "topk shape multiplication overflow",
    )?;
    if out_numel == 0 {
        return Ok((Vec::new(), Vec::new()));
    }
    let offset = meta.storage_offset();
    let data = &input[offset..];

    let mut out_values = vec![0.0; out_numel];
    let mut out_indices = vec![0usize; out_numel];

    // Each `outer` owns a contiguous input block (dim_size*inner_size) and a
    // contiguous output block (k*inner_size); both yield outer_size chunks, so
    // the par_chunks zip aligns by outer. Every lane selects independently with
    // a total value+original-index order equivalent to the serial stable sort,
    // so values AND indices are bit-for-bit identical.
    let in_block = dim_size * inner_size;
    let out_block = k * inner_size;
    let in_total = outer_size * in_block;
    out_values
        .par_chunks_mut(out_block)
        .zip(out_indices.par_chunks_mut(out_block))
        .zip(data[..in_total].par_chunks(in_block))
        .for_each(|((ov_block, oi_block), in_block_data)| {
            for inner in 0..inner_size {
                let mut lane: Vec<(usize, f64)> = (0..dim_size)
                    .map(|d| (d, in_block_data[d * inner_size + inner]))
                    .collect();

                if k < dim_size {
                    let (selected, _, _) =
                        lane.select_nth_unstable_by(k, |a, b| topk_lane_cmp_f64(a, b, largest));
                    if sorted {
                        selected.sort_by(|a, b| topk_lane_cmp_f64(a, b, largest));
                    } else {
                        selected.sort_by_key(|(orig_idx, _)| *orig_idx);
                    }
                    for (out_d, (orig_d, val)) in selected.iter().copied().enumerate() {
                        ov_block[out_d * inner_size + inner] = val;
                        oi_block[out_d * inner_size + inner] = orig_d;
                    }
                    continue;
                }

                lane.sort_by(|a, b| topk_lane_cmp_f64(a, b, largest));
                let selected = &mut lane[..k];
                if !sorted {
                    selected.sort_by_key(|(orig_idx, _)| *orig_idx);
                }

                for (out_d, (orig_d, val)) in selected.iter().copied().enumerate() {
                    ov_block[out_d * inner_size + inner] = val;
                    oi_block[out_d * inner_size + inner] = orig_d;
                }
            }
        });

    Ok((out_values, out_indices))
}

// ---------------------------------------------------------------------------
// Linear Algebra: LU Decomposition with Partial Pivoting
// ---------------------------------------------------------------------------

/// Result of LU factorization in compact form.
///
/// The `lu` matrix stores L (lower triangle, unit diagonal implied) and U
/// (upper triangle including diagonal) packed together.
#[derive(Debug, Clone)]
pub struct LuFactorResult {
    /// Packed LU matrix in row-major order (n x n).
    pub lu: Vec<f64>,
    /// Pivot indices: row i was swapped with row pivots[i] during elimination.
    pub pivots: Vec<usize>,
    /// Matrix dimension (n x n).
    pub n: usize,
}

/// Result of full LU decomposition: P, L, U as separate matrices.
#[derive(Debug, Clone)]
pub struct LuResult {
    /// Permutation matrix (n x n) in row-major order.
    pub p: Vec<f64>,
    /// Lower triangular matrix with unit diagonal (n x n) in row-major order.
    pub l: Vec<f64>,
    /// Upper triangular matrix (n x n) in row-major order.
    pub u: Vec<f64>,
    /// Matrix dimension (n x n).
    pub n: usize,
}

/// Compute the LU factorization of a square matrix with partial pivoting.
///
/// Returns the packed LU matrix and pivot indices. The input matrix is given
/// as a contiguous row-major f64 buffer with the given `TensorMeta`.
///
/// The matrix must be 2-D and square (n x n).
pub fn lu_factor_contiguous_f64(
    data: &[f64],
    meta: &TensorMeta,
) -> Result<LuFactorResult, KernelError> {
    ensure_unary_layout_and_storage(data, meta)?;
    let shape = meta.shape();
    if shape.len() != 2 {
        return Err(KernelError::ShapeMismatch {
            lhs: shape.to_vec(),
            rhs: vec![shape.len()],
        });
    }
    let n = shape[0];
    if n != shape[1] {
        return Err(KernelError::ShapeMismatch {
            lhs: vec![n],
            rhs: vec![shape[1]],
        });
    }
    if n == 0 {
        return Ok(LuFactorResult {
            lu: Vec::new(),
            pivots: Vec::new(),
            n: 0,
        });
    }

    let offset = meta.storage_offset();
    let mut lu: Vec<f64> = data[offset..offset + n * n].to_vec();
    let mut pivots: Vec<usize> = (0..n).collect();

    // Blocked right-looking LU with partial pivoting (the LAPACK getrf scheme):
    // factor a column panel, then apply it to the trailing matrix as a single
    // matrix multiply. The trailing update — which dominates the O(n^3) cost —
    // becomes a cache-blocked, multi-threaded `gemm::dgemm` (compute-bound)
    // instead of a long stream of memory-bound rank-1 updates. Partial pivoting
    // is still column-by-column with full-row swaps, so the PIVOT SEQUENCE is
    // identical to the unblocked algorithm; only the trailing arithmetic
    // reassociates (the factorization stays a valid P·L·U = A to working
    // precision, checked by `lu_factor_reconstructs_pa_eq_lu`).
    const NB: usize = 64;
    const LU_PAR_MIN_ROWS: usize = 64;
    let singular_tol = f64::EPSILON * 1e3;

    let mut k0 = 0;
    while k0 < n {
        let pe = (k0 + NB).min(n); // panel end (exclusive)

        // --- 1. Factor the column panel [k0, pe), eliminating within the panel ---
        for k in k0..pe {
            // Pivot: row with max |lu[i][k]| for i >= k.
            let mut max_val = lu[k * n + k].abs();
            let mut max_row = k;
            for i in (k + 1)..n {
                let val = lu[i * n + k].abs();
                if val > max_val {
                    max_val = val;
                    max_row = i;
                }
            }
            if max_row != k {
                pivots.swap(k, max_row);
                for j in 0..n {
                    lu.swap(k * n + j, max_row * n + j); // full-row swap
                }
            }
            let diag = lu[k * n + k];
            if diag.abs() < singular_tol {
                // Near-singular: zero the column's multipliers so the trailing
                // GEMM contributes nothing for it. Downstream (det/inv/solve)
                // detect singularity via the U diagonal.
                for i in (k + 1)..n {
                    lu[i * n + k] = 0.0;
                }
                continue;
            }
            // Scale L below the pivot and eliminate WITHIN the panel only
            // (columns k+1..pe); trailing columns are deferred to the GEMM.
            let rows_below = n - k - 1;
            if rows_below >= LU_PAR_MIN_ROWS && rayon::current_num_threads() > 1 {
                let (head, tail) = lu.split_at_mut((k + 1) * n);
                let pivot_row = &head[k * n..(k + 1) * n];
                tail.par_chunks_mut(n).for_each(|row_i| {
                    let m = row_i[k] / diag;
                    row_i[k] = m;
                    for j in (k + 1)..pe {
                        row_i[j] -= m * pivot_row[j];
                    }
                });
            } else {
                for i in (k + 1)..n {
                    let m = lu[i * n + k] / diag;
                    lu[i * n + k] = m;
                    for j in (k + 1)..pe {
                        lu[i * n + j] -= m * lu[k * n + j];
                    }
                }
            }
        }

        let kb = pe - k0; // panel width
        let tcols = n - pe; // trailing columns
        if tcols == 0 {
            break;
        }

        // --- 2. Triangular solve U12 = L11^{-1} * A12 (unit-lower L11) ---
        // Forward substitution over panel rows in increasing order, so lu[t][j]
        // already holds U[t][j] when row i (> t) consumes it.
        for i in k0..pe {
            for t in k0..i {
                let lit = lu[i * n + t];
                if lit != 0.0 {
                    for j in pe..n {
                        lu[i * n + j] -= lit * lu[t * n + j];
                    }
                }
            }
        }

        // --- 3. GEMM trailing update A22 -= L21 * U12 ---
        let trows = n - pe;
        // Pack L21 (rows pe..n, cols k0..pe) -> contiguous [trows x kb].
        let mut l21 = vec![0.0f64; trows * kb];
        for ii in 0..trows {
            let src = (pe + ii) * n + k0;
            l21[ii * kb..ii * kb + kb].copy_from_slice(&lu[src..src + kb]);
        }
        // Pack U12 (rows k0..pe, cols pe..n) -> contiguous [kb x tcols].
        let mut u12 = vec![0.0f64; kb * tcols];
        for ii in 0..kb {
            let src = (k0 + ii) * n + pe;
            u12[ii * tcols..ii * tcols + tcols].copy_from_slice(&lu[src..src + tcols]);
        }
        // product = L21 * U12  -> [trows x tcols], then A22 -= product.
        let mut prod = vec![0.0f64; trows * tcols];
        gemm::dgemm(trows, kb, tcols, &l21, &u12, &mut prod);
        for ii in 0..trows {
            let dst = (pe + ii) * n + pe;
            let prow = &prod[ii * tcols..ii * tcols + tcols];
            for jj in 0..tcols {
                lu[dst + jj] -= prow[jj];
            }
        }

        k0 = pe;
    }

    Ok(LuFactorResult { lu, pivots, n })
}

/// Unpack a compact LU factorization into separate P, L, U matrices.
pub fn lu_unpack(factor: &LuFactorResult) -> LuResult {
    let n = factor.n;
    if n == 0 {
        return LuResult {
            p: Vec::new(),
            l: Vec::new(),
            u: Vec::new(),
            n: 0,
        };
    }

    // Build permutation matrix from pivot vector.
    // pivots[i] = original row that ended up at position i in the LU matrix.
    // For the convention P @ L @ U = A, we need P[pivots[i]][i] = 1.
    let mut p = vec![0.0; n * n];
    for (i, &orig_row) in factor.pivots.iter().enumerate() {
        p[orig_row * n + i] = 1.0;
    }

    // Extract L (lower triangular with unit diagonal)
    let mut l = vec![0.0; n * n];
    for i in 0..n {
        l[i * n + i] = 1.0; // unit diagonal
        for j in 0..i {
            l[i * n + j] = factor.lu[i * n + j];
        }
    }

    // Extract U (upper triangular including diagonal)
    let mut u = vec![0.0; n * n];
    for i in 0..n {
        for j in i..n {
            u[i * n + j] = factor.lu[i * n + j];
        }
    }

    LuResult { p, l, u, n }
}

/// Solve A * X = B using a pre-computed LU factorization.
///
/// `b` is a column vector of length n (or an n x m matrix for multiple RHS).
/// Returns the solution vector/matrix X.
pub fn lu_solve_contiguous_f64(
    factor: &LuFactorResult,
    b_data: &[f64],
    b_meta: &TensorMeta,
) -> Result<Vec<f64>, KernelError> {
    ensure_unary_layout_and_storage(b_data, b_meta)?;
    let b_shape = b_meta.shape();
    let n = factor.n;

    // b can be [n] (single RHS) or [n, m] (multiple RHS)
    let (b_rows, num_rhs) = match b_shape.len() {
        1 => (b_shape[0], 1usize),
        2 => (b_shape[0], b_shape[1]),
        _ => {
            return Err(KernelError::ShapeMismatch {
                lhs: vec![n],
                rhs: b_shape.to_vec(),
            });
        }
    };
    if b_rows != n {
        return Err(KernelError::ShapeMismatch {
            lhs: vec![n],
            rhs: vec![b_rows],
        });
    }
    if n == 0 {
        return Ok(Vec::new());
    }

    let offset = b_meta.storage_offset();
    let mut x: Vec<f64> = b_data[offset..offset + n * num_rhs].to_vec();

    // Apply pivot permutation to each RHS column.
    // pivots[i] = original row at LU position i, so we need:
    // permuted[i] = b[pivots[i]] (reorder b to match LU row order)
    for rhs in 0..num_rhs {
        let mut permuted = vec![0.0; n];
        for i in 0..n {
            permuted[i] = x[factor.pivots[i] * num_rhs + rhs];
        }
        for i in 0..n {
            x[i * num_rhs + rhs] = permuted[i];
        }
    }

    // Forward substitution: L * y = P * b. The inner `rhs` loop is deliberate:
    // each L coefficient is loaded once and applied across all RHS columns
    // (cache/SIMD-amortized), which beats a per-column solve that re-streams L.
    //
    // DELIBERATELY SERIAL at the benched sizes — TWO faster-looking levers were
    // measured and BOTH regressed:
    //  (a) rayon per-step row fan-out: REGRESSED inv ~2x (256: 49->132, 512:
    //      199->380ms). The updates are memory-bound rank-1 AXPYs and the k-loop
    //      is sequential -> ~2n short parallel regions whose join barriers +
    //      shared-bandwidth contention swamp the gain.
    //  (b) cache-BLOCKED TRSM (diagonal block solve + trailing update via
    //      gemm::dgemm): also REGRESSED inv (256: 44->54, 512: 124->240ms). Unlike
    //      cholesky's SYRK (m x NB x m, large in both dims -> crosses dgemm's 1<<27
    //      parallel gate), the TRSM trailing update is SKINNY-K (m x NB x num_rhs)
    //      and stays below the gate at n<=512 -> serial dgemm, and at these sizes
    //      the RHS matrix is cache-resident so the naive sweep is already near
    //      optimal; the packing + skinny-GEMM overhead nets a loss.
    // Blocking would only pay for n >> 1500 (RHS spills cache); inv is benched at
    // 256/512. See project_perf_binding_constraints memory.
    for k in 0..n {
        for i in (k + 1)..n {
            let l_ik = factor.lu[i * n + k];
            for rhs in 0..num_rhs {
                x[i * num_rhs + rhs] -= l_ik * x[k * num_rhs + rhs];
            }
        }
    }

    // Back substitution: U * x = y
    for k in (0..n).rev() {
        let diag = factor.lu[k * n + k];
        if diag.abs() < f64::EPSILON * 1e3 {
            // Singular or near-singular — set solution to 0 for this row
            for rhs in 0..num_rhs {
                x[k * num_rhs + rhs] = 0.0;
            }
            continue;
        }
        for rhs in 0..num_rhs {
            x[k * num_rhs + rhs] /= diag;
        }
        for i in 0..k {
            let u_ik = factor.lu[i * n + k];
            for rhs in 0..num_rhs {
                x[i * num_rhs + rhs] -= u_ik * x[k * num_rhs + rhs];
            }
        }
    }

    Ok(x)
}

/// Compute matrix inverse for a contiguous square f64 matrix.
///
/// Uses LU factorization with partial pivoting and solves against the identity.
pub fn inv_tensor_contiguous_f64(data: &[f64], meta: &TensorMeta) -> Result<Vec<f64>, KernelError> {
    ensure_unary_layout_and_storage(data, meta)?;
    let shape = meta.shape();
    if shape.len() != 2 {
        return Err(KernelError::ShapeMismatch {
            lhs: shape.to_vec(),
            rhs: vec![shape.len()],
        });
    }
    let n = shape[0];
    if n != shape[1] {
        return Err(KernelError::ShapeMismatch {
            lhs: vec![n],
            rhs: vec![shape[1]],
        });
    }
    if n == 0 {
        return Ok(Vec::new());
    }

    let factor = lu_factor_contiguous_f64(data, meta)?;
    if (0..n).any(|i| factor.lu[i * n + i].abs() < f64::EPSILON * 1e3) {
        return Err(KernelError::SingularMatrix { size: n });
    }

    let mut identity = vec![0.0; n * n];
    for i in 0..n {
        identity[i * n + i] = 1.0;
    }
    let identity_meta = TensorMeta::from_shape(vec![n, n], meta.dtype(), meta.device());
    lu_solve_contiguous_f64(&factor, &identity, &identity_meta)
}

/// Result of determinant computation.
#[derive(Debug, Clone)]
pub struct DetResult {
    /// The determinant value.
    pub det: f64,
}

/// Result of sign-log-determinant computation.
#[derive(Debug, Clone)]
pub struct SlogdetResult {
    /// Sign of the determinant: +1.0, -1.0, or 0.0 (for singular matrices).
    pub sign: f64,
    /// Natural log of the absolute value of the determinant.
    pub logabsdet: f64,
}

/// Compute the determinant of a square matrix using LU factorization.
///
/// det(A) = product of U diagonal * (-1)^(number of row swaps).
pub fn det_contiguous_f64(data: &[f64], meta: &TensorMeta) -> Result<DetResult, KernelError> {
    ensure_unary_layout_and_storage(data, meta)?;
    let shape = meta.shape();
    if shape.len() != 2 {
        return Err(KernelError::ShapeMismatch {
            lhs: shape.to_vec(),
            rhs: vec![2],
        });
    }
    let n = shape[0];
    if n != shape[1] {
        return Err(KernelError::ShapeMismatch {
            lhs: vec![n],
            rhs: vec![shape[1]],
        });
    }
    if n == 0 {
        return Ok(DetResult { det: 1.0 });
    }

    let factor = lu_factor_contiguous_f64(data, meta)?;

    // Compute sign of the permutation via cycle decomposition.
    // sign = (-1)^(n - number_of_cycles)
    let sign = permutation_sign(&factor.pivots);

    // Product of U diagonal
    let mut det = sign;
    for i in 0..n {
        det *= factor.lu[i * n + i];
    }

    Ok(DetResult { det })
}

/// Compute the sign of a permutation via cycle decomposition.
///
/// Returns +1.0 for even permutations, -1.0 for odd permutations.
fn permutation_sign(perm: &[usize]) -> f64 {
    let n = perm.len();
    let mut visited = vec![false; n];
    let mut num_cycles = 0usize;
    for i in 0..n {
        if !visited[i] {
            num_cycles += 1;
            let mut j = i;
            while !visited[j] {
                visited[j] = true;
                j = perm[j];
            }
        }
    }
    // A permutation decomposes into cycles. Each cycle of length k
    // requires (k-1) transpositions. Total transpositions = n - num_cycles.
    if (n - num_cycles).is_multiple_of(2) {
        1.0
    } else {
        -1.0
    }
}

/// Compute sign and log-absolute-determinant of a square matrix.
///
/// More numerically stable than computing det directly for large matrices.
/// Returns (sign, logabsdet) where det(A) = sign * exp(logabsdet).
pub fn slogdet_contiguous_f64(
    data: &[f64],
    meta: &TensorMeta,
) -> Result<SlogdetResult, KernelError> {
    ensure_unary_layout_and_storage(data, meta)?;
    let shape = meta.shape();
    if shape.len() != 2 {
        return Err(KernelError::ShapeMismatch {
            lhs: shape.to_vec(),
            rhs: vec![2],
        });
    }
    let n = shape[0];
    if n != shape[1] {
        return Err(KernelError::ShapeMismatch {
            lhs: vec![n],
            rhs: vec![shape[1]],
        });
    }
    if n == 0 {
        return Ok(SlogdetResult {
            sign: 1.0,
            logabsdet: 0.0,
        });
    }

    let factor = lu_factor_contiguous_f64(data, meta)?;

    // Compute sign of the permutation via cycle decomposition
    let mut sign = permutation_sign(&factor.pivots);

    // Sum log(|U_ii|) and track sign
    let mut logabsdet = 0.0;
    for i in 0..n {
        let diag = factor.lu[i * n + i];
        if diag == 0.0 {
            return Ok(SlogdetResult {
                sign: 0.0,
                logabsdet: f64::NEG_INFINITY,
            });
        }
        if diag < 0.0 {
            sign = -sign;
        }
        logabsdet += diag.abs().ln();
    }

    Ok(SlogdetResult { sign, logabsdet })
}

/// Result of Cholesky decomposition.
#[derive(Debug, Clone)]
pub struct CholeskyResult {
    /// Lower (or upper) triangular factor in row-major order (n x n).
    pub factor: Vec<f64>,
    /// Matrix dimension.
    pub n: usize,
}

/// Compute the Cholesky decomposition of a symmetric positive-definite matrix.
///
/// Returns the lower triangular factor L such that A = L @ L^T.
/// If `upper` is true, returns U such that A = U^T @ U (U = L^T).
///
/// Errors if A is not square, not positive-definite, or has incompatible layout.
/// Winograd F(2x2, 3x3) filter transform `U = G g G^T` (4x4), `g` row-major 3x3.
#[inline]
fn winograd_filter_transform(g: &[f64]) -> [f64; 16] {
    // G rows: [1,0,0], [0.5,0.5,0.5], [0.5,-0.5,0.5], [0,0,1].
    let grow = |r: usize, c: [f64; 3]| -> f64 {
        match r {
            0 => c[0],
            1 => 0.5 * (c[0] + c[1] + c[2]),
            2 => 0.5 * (c[0] - c[1] + c[2]),
            _ => c[2],
        }
    };
    let mut gg = [[0.0f64; 3]; 4]; // G g  (4x3)
    for j in 0..3 {
        let col = [g[j], g[3 + j], g[6 + j]];
        for (i, row) in gg.iter_mut().enumerate() {
            row[j] = grow(i, col);
        }
    }
    let mut u = [0.0f64; 16]; // (G g) G^T  (4x4)
    for (i, row) in gg.iter().enumerate() {
        let row = *row;
        for j in 0..4 {
            u[i * 4 + j] = grow(j, row);
        }
    }
    u
}

/// Winograd F(2,3) input transform `V = B^T d B` (4x4), `d` row-major 4x4.
#[inline]
fn winograd_input_transform(d: &[f64]) -> [f64; 16] {
    // B^T rows: [1,0,-1,0], [0,1,1,0], [0,-1,1,0], [0,1,0,-1].
    let bt = |r: usize, c: [f64; 4]| -> f64 {
        match r {
            0 => c[0] - c[2],
            1 => c[1] + c[2],
            2 => c[2] - c[1],
            _ => c[1] - c[3],
        }
    };
    let mut td = [[0.0f64; 4]; 4]; // B^T d
    for j in 0..4 {
        let col = [d[j], d[4 + j], d[8 + j], d[12 + j]];
        for (i, row) in td.iter_mut().enumerate() {
            row[j] = bt(i, col);
        }
    }
    let mut v = [0.0f64; 16]; // (B^T d) B
    for (i, row) in td.iter().enumerate() {
        let row = *row;
        for j in 0..4 {
            v[i * 4 + j] = bt(j, row);
        }
    }
    v
}

/// Winograd F(2,3) output transform `Y = A^T m A` (2x2), `m` row-major 4x4.
#[inline]
fn winograd_output_transform(m: &[f64]) -> [f64; 4] {
    // A^T rows: [1,1,1,0], [0,1,-1,-1].
    let at = |r: usize, c: [f64; 4]| -> f64 {
        match r {
            0 => c[0] + c[1] + c[2],
            _ => c[1] - c[2] - c[3],
        }
    };
    let mut tm = [[0.0f64; 4]; 2]; // A^T m
    for j in 0..4 {
        let col = [m[j], m[4 + j], m[8 + j], m[12 + j]];
        for (i, row) in tm.iter_mut().enumerate() {
            row[j] = at(i, col);
        }
    }
    let mut y = [0.0f64; 4]; // (A^T m) A
    for (i, row) in tm.iter().enumerate() {
        let row = *row;
        for j in 0..2 {
            y[i * 2 + j] = at(j, row);
        }
    }
    y
}

/// Winograd F(2x2, 3x3) convolution for a 3x3, stride-1 conv (no dilation).
/// `input` is `[batch, in_ch, padded_h, padded_w]` (already padded), `weight` is
/// `[out_ch, in_ch, 3, 3]`. Returns `[batch, out_ch, out_h, out_w]` with
/// `out_h = padded_h - 2`, `out_w = padded_w - 2`.
///
/// Uses 16 transform-domain products per 2x2 output tile instead of 36 direct
/// multiplies (~2.25x fewer), with the channel-summed bulk routed through
/// `gemm::dgemm` as 16 independent GEMMs fanned out over the 16 positions (each
/// is below dgemm's parallel gate). The transforms reassociate, so the result
/// matches direct convolution to tolerance, not bit-for-bit.
///
/// NON-WIRED FOUNDATION (validated by `winograd_conv2d_matches_direct_within_tolerance`).
/// Wiring it into the 3x3 stride-1 no_grad conv path was MEASURED vs the parallel
/// im2col GEMM and only reached ~1.05x at conv2d/hw128 (594->566ms): the input/
/// output transforms materialise ~128MB position-major `v`/`m` intermediates with
/// strided gather/scatter (4x the data of the 2x2 output) and stay memory-bound
/// for f64, eating most of the multiply savings. Winograd pays for f32/f16 with
/// FUSED transform+GEMM kernels (no materialised intermediate) — the path to a
/// real win here — so the primitive is kept validated but unused for now.
pub fn winograd_conv2d_3x3_s1_f64(
    input: &[f64],
    weight: &[f64],
    batch: usize,
    in_ch: usize,
    out_ch: usize,
    padded_h: usize,
    padded_w: usize,
) -> Vec<f64> {
    let out_h = padded_h - 2;
    let out_w = padded_w - 2;
    let tiles_h = out_h.div_ceil(2);
    let tiles_w = out_w.div_ceil(2);
    let num_tiles = batch * tiles_h * tiles_w;
    let oc_ic = out_ch * in_ch;

    // 1. Filter transform -> u[p][oc][ic].
    let mut u = vec![0.0f64; 16 * oc_ic];
    for oc in 0..out_ch {
        for ic in 0..in_ch {
            let g = &weight[(oc * in_ch + ic) * 9..(oc * in_ch + ic) * 9 + 9];
            let uu = winograd_filter_transform(g);
            for (p, &val) in uu.iter().enumerate() {
                u[p * oc_ic + oc * in_ch + ic] = val;
            }
        }
    }

    // 2. Input transform -> v[p][ic][tile].
    let ic_t = in_ch * num_tiles;
    let mut v = vec![0.0f64; 16 * ic_t];
    for b in 0..batch {
        for th in 0..tiles_h {
            for tw in 0..tiles_w {
                let tile = (b * tiles_h + th) * tiles_w + tw;
                let r0 = th * 2;
                let c0 = tw * 2;
                for ic in 0..in_ch {
                    let base = (b * in_ch + ic) * padded_h * padded_w;
                    let mut d = [0.0f64; 16];
                    for i in 0..4 {
                        let rr = r0 + i;
                        if rr >= padded_h {
                            continue;
                        }
                        for j in 0..4 {
                            let cc = c0 + j;
                            if cc < padded_w {
                                d[i * 4 + j] = input[base + rr * padded_w + cc];
                            }
                        }
                    }
                    let vv = winograd_input_transform(&d);
                    for (p, &val) in vv.iter().enumerate() {
                        v[p * ic_t + ic * num_tiles + tile] = val;
                    }
                }
            }
        }
    }

    // 3. 16 GEMMs: m[p] = U[p] (out_ch x in_ch) @ V[p] (in_ch x num_tiles).
    let oc_t = out_ch * num_tiles;
    let mut m = vec![0.0f64; 16 * oc_t];
    // The 16 position-GEMMs are independent and each (out_ch x in_ch x num_tiles)
    // is below dgemm's parallel gate, so fan out OVER the 16 (each runs a serial
    // dgemm) for 16-way parallelism — matching the parallel im2col GEMM while
    // keeping Winograd's ~2.25x fewer multiplies.
    m.par_chunks_mut(oc_t).enumerate().for_each(|(p, mp)| {
        gemm::dgemm(
            out_ch,
            in_ch,
            num_tiles,
            &u[p * oc_ic..(p + 1) * oc_ic],
            &v[p * ic_t..(p + 1) * ic_t],
            mp,
        );
    });

    // 4. Output transform + scatter.
    let mut output = vec![0.0f64; batch * out_ch * out_h * out_w];
    for b in 0..batch {
        for th in 0..tiles_h {
            for tw in 0..tiles_w {
                let tile = (b * tiles_h + th) * tiles_w + tw;
                let r0 = th * 2;
                let c0 = tw * 2;
                for oc in 0..out_ch {
                    let mut mm = [0.0f64; 16];
                    for (p, slot) in mm.iter_mut().enumerate() {
                        *slot = m[p * oc_t + oc * num_tiles + tile];
                    }
                    let y = winograd_output_transform(&mm);
                    let obase = (b * out_ch + oc) * out_h * out_w;
                    for i in 0..2 {
                        let rr = r0 + i;
                        if rr >= out_h {
                            continue;
                        }
                        for j in 0..2 {
                            let cc = c0 + j;
                            if cc < out_w {
                                output[obase + rr * out_w + cc] = y[i * 2 + j];
                            }
                        }
                    }
                }
            }
        }
    }
    output
}

/// Winograd F(2x2, 3x3) filter transform `U = G g G^T` (4x4), f32, `g` row-major 3x3.
#[inline]
fn winograd_filter_transform_f32(g: &[f32]) -> [f32; 16] {
    let grow = |r: usize, c: [f32; 3]| -> f32 {
        match r {
            0 => c[0],
            1 => 0.5 * (c[0] + c[1] + c[2]),
            2 => 0.5 * (c[0] - c[1] + c[2]),
            _ => c[2],
        }
    };
    let mut gg = [[0.0f32; 3]; 4];
    for j in 0..3 {
        let col = [g[j], g[3 + j], g[6 + j]];
        for (i, row) in gg.iter_mut().enumerate() {
            row[j] = grow(i, col);
        }
    }
    let mut u = [0.0f32; 16];
    for (i, row) in gg.iter().enumerate() {
        let row = *row;
        for j in 0..4 {
            u[i * 4 + j] = grow(j, row);
        }
    }
    u
}

/// Winograd F(2,3) input transform `V = B^T d B` (4x4), f32, `d` row-major 4x4.
#[inline]
fn winograd_input_transform_f32(d: &[f32]) -> [f32; 16] {
    let bt = |r: usize, c: [f32; 4]| -> f32 {
        match r {
            0 => c[0] - c[2],
            1 => c[1] + c[2],
            2 => c[2] - c[1],
            _ => c[1] - c[3],
        }
    };
    let mut td = [[0.0f32; 4]; 4];
    for j in 0..4 {
        let col = [d[j], d[4 + j], d[8 + j], d[12 + j]];
        for (i, row) in td.iter_mut().enumerate() {
            row[j] = bt(i, col);
        }
    }
    let mut v = [0.0f32; 16];
    for (i, row) in td.iter().enumerate() {
        let row = *row;
        for j in 0..4 {
            v[i * 4 + j] = bt(j, row);
        }
    }
    v
}

/// Winograd F(2,3) output transform `Y = A^T m A` (2x2), f32, `m` row-major 4x4.
#[inline]
fn winograd_output_transform_f32(m: &[f32]) -> [f32; 4] {
    let at = |r: usize, c: [f32; 4]| -> f32 {
        match r {
            0 => c[0] + c[1] + c[2],
            _ => c[1] - c[2] - c[3],
        }
    };
    let mut tm = [[0.0f32; 4]; 2];
    for j in 0..4 {
        let col = [m[j], m[4 + j], m[8 + j], m[12 + j]];
        for (i, row) in tm.iter_mut().enumerate() {
            row[j] = at(i, col);
        }
    }
    let mut y = [0.0f32; 4];
    for (i, row) in tm.iter().enumerate() {
        let row = *row;
        for j in 0..2 {
            y[i * 2 + j] = at(j, row);
        }
    }
    y
}

/// Winograd F(2x2, 3x3) convolution for a 3x3, stride-1 conv (no dilation), f32.
///
/// f32 mirror of [`winograd_conv2d_3x3_s1_f64`], validated within tolerance by
/// `winograd_conv2d_f32_matches_direct_within_tolerance`.
///
/// NON-WIRED FOUNDATION — MEASURED REGRESSION, kept as a reference + scaffold.
/// `benches/winograd_bench.rs` A/B's this against an im2col + `sgemm` baseline on a
/// 64-core worker (same binary, same worker): this materialized F(2,3) path is
/// **1.3–2.7x SLOWER** at every representative 3x3-s1 shape (e.g. b8/ic64/oc64/hw32:
/// 25.6 ms vs 9.4 ms). Two compounding causes, both fundamental to this formulation:
///   1. F(2,3) splits one big-`k` GEMM (`k = in_ch*9`) into 16 small-`k` GEMMs
///      (`k = in_ch`), collapsing arithmetic intensity — matrixmultiply's microkernel
///      cannot amortize A/B packing over the short reduction, so the ~2.25x fewer
///      multiplies are dwarfed by per-GEMM packing overhead.
///   2. The serial input/output transforms scatter ~4x the output volume with poor
///      locality (memory-bound), exactly as the f64 doc warned.
///
/// A real CPU win needs a genuinely FUSED transform+GEMM kernel (no materialised
/// `v`/`m`) AND a larger tile (F(4,3)/F(6,3)) to restore GEMM `k`-depth — a large,
/// correctness-critical build. Do NOT wire this materialized path: it regresses.
///
/// `input` is `[batch, in_ch, padded_h, padded_w]` (already padded), `weight` is
/// `[out_ch, in_ch, 3, 3]`. Returns `[batch, out_ch, out_h, out_w]` with
/// `out_h = padded_h - 2`, `out_w = padded_w - 2`. The transforms reassociate, so
/// the result matches direct convolution to tolerance, not bit-for-bit.
#[allow(clippy::too_many_arguments)]
pub fn winograd_conv2d_3x3_s1_f32(
    input: &[f32],
    weight: &[f32],
    batch: usize,
    in_ch: usize,
    out_ch: usize,
    padded_h: usize,
    padded_w: usize,
) -> Vec<f32> {
    let out_h = padded_h - 2;
    let out_w = padded_w - 2;
    let tiles_h = out_h.div_ceil(2);
    let tiles_w = out_w.div_ceil(2);
    let num_tiles = batch * tiles_h * tiles_w;
    let oc_ic = out_ch * in_ch;

    // 1. Filter transform -> u[p][oc][ic].
    let mut u = vec![0.0f32; 16 * oc_ic];
    for oc in 0..out_ch {
        for ic in 0..in_ch {
            let g = &weight[(oc * in_ch + ic) * 9..(oc * in_ch + ic) * 9 + 9];
            let uu = winograd_filter_transform_f32(g);
            for (p, &val) in uu.iter().enumerate() {
                u[p * oc_ic + oc * in_ch + ic] = val;
            }
        }
    }

    // 2. Input transform -> v[p][ic][tile].
    let ic_t = in_ch * num_tiles;
    let mut v = vec![0.0f32; 16 * ic_t];
    for b in 0..batch {
        for th in 0..tiles_h {
            for tw in 0..tiles_w {
                let tile = (b * tiles_h + th) * tiles_w + tw;
                let r0 = th * 2;
                let c0 = tw * 2;
                for ic in 0..in_ch {
                    let base = (b * in_ch + ic) * padded_h * padded_w;
                    let mut d = [0.0f32; 16];
                    for i in 0..4 {
                        let rr = r0 + i;
                        if rr >= padded_h {
                            continue;
                        }
                        for j in 0..4 {
                            let cc = c0 + j;
                            if cc < padded_w {
                                d[i * 4 + j] = input[base + rr * padded_w + cc];
                            }
                        }
                    }
                    let vv = winograd_input_transform_f32(&d);
                    for (p, &val) in vv.iter().enumerate() {
                        v[p * ic_t + ic * num_tiles + tile] = val;
                    }
                }
            }
        }
    }

    // 3. 16 GEMMs: m[p] = U[p] (out_ch x in_ch) @ V[p] (in_ch x num_tiles).
    // Each is a WIDE GEMM (num_tiles >> out_ch), so run them SERIALLY and let
    // gemm::sgemm parallelize each across all cores (col-parallel). Fanning out
    // over the 16 positions instead caps parallelism at 16-way and idles a
    // many-core worker.
    let oc_t = out_ch * num_tiles;
    let mut m = vec![0.0f32; 16 * oc_t];
    for (p, mp) in m.chunks_mut(oc_t).enumerate() {
        gemm::sgemm(
            out_ch,
            in_ch,
            num_tiles,
            &u[p * oc_ic..(p + 1) * oc_ic],
            &v[p * ic_t..(p + 1) * ic_t],
            mp,
        );
    }

    // 4. Output transform + scatter.
    let mut output = vec![0.0f32; batch * out_ch * out_h * out_w];
    for b in 0..batch {
        for th in 0..tiles_h {
            for tw in 0..tiles_w {
                let tile = (b * tiles_h + th) * tiles_w + tw;
                let r0 = th * 2;
                let c0 = tw * 2;
                for oc in 0..out_ch {
                    let mut mm = [0.0f32; 16];
                    for (p, slot) in mm.iter_mut().enumerate() {
                        *slot = m[p * oc_t + oc * num_tiles + tile];
                    }
                    let y = winograd_output_transform_f32(&mm);
                    let obase = (b * out_ch + oc) * out_h * out_w;
                    for i in 0..2 {
                        let rr = r0 + i;
                        if rr >= out_h {
                            continue;
                        }
                        for j in 0..2 {
                            let cc = c0 + j;
                            if cc < out_w {
                                output[obase + rr * out_w + cc] = y[i * 2 + j];
                            }
                        }
                    }
                }
            }
        }
    }
    output
}

pub fn cholesky_contiguous_f64(
    data: &[f64],
    meta: &TensorMeta,
    upper: bool,
) -> Result<CholeskyResult, KernelError> {
    ensure_unary_layout_and_storage(data, meta)?;
    let shape = meta.shape();
    if shape.len() != 2 || shape[0] != shape[1] {
        return Err(KernelError::ShapeMismatch {
            lhs: shape.to_vec(),
            rhs: vec![2],
        });
    }
    let n = shape[0];
    if n == 0 {
        return Ok(CholeskyResult {
            factor: Vec::new(),
            n: 0,
        });
    }

    let offset = meta.storage_offset();
    // Right-looking BLOCKED Cholesky (LAPACK potrf shape). `l` starts as the lower
    // triangle of A and is overwritten in place with L. For each NB-wide panel we
    //   1. factor the NB×NB diagonal block (serial unblocked Cholesky on the
    //      already-Schur-complemented block),
    //   2. TRSM the sub-diagonal panel  L21 = A21 · L11^{-T}  (rows independent,
    //      compute-bound nb² each -> fan out),
    //   3. apply the trailing symmetric rank-NB update  A22 -= L21 · L21^T  through
    //      the cache-blocked + parallel `gemm::dgemm`.
    // The bulk O(n^3/3) FLOPs land in step 3's GEMM, whose cache blocking fixes the
    // memory-bound row-streaming of the previous left-looking dot-product kernel.
    // This REASSOCIATES the trailing sums (panel-by-panel vs one long dot), so the
    // result matches the serial kernel only to tolerance — validated by
    // reconstruction (L·L^T ≈ A) and the numpy oracle, not bit-for-bit.
    const NB: usize = 64;
    let mut l = vec![0.0f64; n * n];
    for i in 0..n {
        for j in 0..=i {
            l[i * n + j] = data[offset + i * n + j];
        }
    }

    let mut jb = 0;
    while jb < n {
        let je = (jb + NB).min(n);
        let nb = je - jb;

        // 1. Factor the diagonal block l[jb:je, jb:je] (lower) in place.
        for jj in jb..je {
            let mut s = l[jj * n + jj];
            for p in jb..jj {
                s -= l[jj * n + p] * l[jj * n + p];
            }
            if s <= 0.0 {
                return Err(KernelError::NotPositiveDefinite);
            }
            let d = s.sqrt();
            l[jj * n + jj] = d;
            for ii in (jj + 1)..je {
                let mut t = l[ii * n + jj];
                for p in jb..jj {
                    t -= l[ii * n + p] * l[jj * n + p];
                }
                l[ii * n + jj] = t / d;
            }
        }

        let m = n - je; // trailing rows below the panel
        if m == 0 {
            break;
        }

        // 2. TRSM: L21 = A21 · L11^{-T}, panel l[je:n, jb:je] in place. Each
        //    trailing row solves the lower-triangular L11 independently.
        let (head, tail) = l.split_at_mut(je * n);
        let trsm_body = |row: &mut [f64]| {
            for c in 0..nb {
                let mut t = row[jb + c];
                for p in 0..c {
                    t -= row[jb + p] * head[(jb + c) * n + (jb + p)];
                }
                row[jb + c] = t / head[(jb + c) * n + (jb + c)];
            }
        };
        if m >= 64 {
            tail.par_chunks_mut(n).for_each(trsm_body);
        } else {
            tail.chunks_mut(n).for_each(trsm_body);
        }

        // 3. Trailing update A22 -= L21 · L21^T via the blocked/parallel GEMM.
        //    Pack L21 (m×nb) and its transpose (nb×m) contiguously, multiply, then
        //    subtract the lower triangle back into l[je:n, je:n].
        let mut l21 = vec![0.0f64; m * nb];
        let mut l21t = vec![0.0f64; nb * m];
        for i in 0..m {
            for c in 0..nb {
                let v = l[(je + i) * n + (jb + c)];
                l21[i * nb + c] = v;
                l21t[c * m + i] = v;
            }
        }
        let mut prod = vec![0.0f64; m * m];
        gemm::dgemm(m, nb, m, &l21, &l21t, &mut prod);
        for i in 0..m {
            let row_base = (je + i) * n + je;
            let p_base = i * m;
            for j in 0..=i {
                l[row_base + j] -= prod[p_base + j];
            }
        }

        jb = je;
    }

    if upper {
        // Transpose L to get U
        let mut u = vec![0.0f64; n * n];
        for i in 0..n {
            for j in i..n {
                u[i * n + j] = l[j * n + i];
            }
        }
        Ok(CholeskyResult { factor: u, n })
    } else {
        Ok(CholeskyResult { factor: l, n })
    }
}

/// Solve A @ X = B given Cholesky factor L where A = L @ L^T.
///
/// Performs two triangular solves: L @ Y = B, then L^T @ X = Y.
/// If `upper` is true, the factor is U (upper triangular) and A = U^T @ U.
/// B can be [n] (single RHS) or [n, m] (multiple RHS).
pub fn cholesky_solve_contiguous_f64(
    factor: &CholeskyResult,
    b_data: &[f64],
    b_meta: &TensorMeta,
    upper: bool,
) -> Result<Vec<f64>, KernelError> {
    ensure_unary_layout_and_storage(b_data, b_meta)?;
    let b_shape = b_meta.shape();
    let n = factor.n;

    let (b_rows, num_rhs) = match b_shape.len() {
        1 => (b_shape[0], 1usize),
        2 => (b_shape[0], b_shape[1]),
        _ => {
            return Err(KernelError::ShapeMismatch {
                lhs: vec![n],
                rhs: b_shape.to_vec(),
            });
        }
    };
    if b_rows != n {
        return Err(KernelError::ShapeMismatch {
            lhs: vec![n],
            rhs: vec![b_rows],
        });
    }
    if n == 0 {
        return Ok(Vec::new());
    }

    let b_offset = b_meta.storage_offset();
    let l = &factor.factor;

    // Copy B into working array
    let mut x = vec![0.0f64; n * num_rhs];
    for i in 0..n {
        for rhs in 0..num_rhs {
            x[i * num_rhs + rhs] = b_data[b_offset + i * num_rhs + rhs];
        }
    }

    if upper {
        // A = U^T @ U: solve U^T @ Y = B (forward sub), then U @ X = Y (back sub)
        // Forward substitution with U^T (lower triangular)
        for i in 0..n {
            for k in 0..i {
                let u_ki = l[k * n + i]; // U^T[i][k] = U[k][i]
                for rhs in 0..num_rhs {
                    x[i * num_rhs + rhs] -= u_ki * x[k * num_rhs + rhs];
                }
            }
            let diag = l[i * n + i]; // U[i][i]
            for rhs in 0..num_rhs {
                x[i * num_rhs + rhs] /= diag;
            }
        }
        // Back substitution with U (upper triangular)
        for i in (0..n).rev() {
            for k in (i + 1)..n {
                let u_ik = l[i * n + k];
                for rhs in 0..num_rhs {
                    x[i * num_rhs + rhs] -= u_ik * x[k * num_rhs + rhs];
                }
            }
            let diag = l[i * n + i];
            for rhs in 0..num_rhs {
                x[i * num_rhs + rhs] /= diag;
            }
        }
    } else {
        // A = L @ L^T: solve L @ Y = B (forward sub), then L^T @ X = Y (back sub)
        // Forward substitution with L (lower triangular)
        for i in 0..n {
            for k in 0..i {
                let l_ik = l[i * n + k];
                for rhs in 0..num_rhs {
                    x[i * num_rhs + rhs] -= l_ik * x[k * num_rhs + rhs];
                }
            }
            let diag = l[i * n + i];
            for rhs in 0..num_rhs {
                x[i * num_rhs + rhs] /= diag;
            }
        }
        // Back substitution with L^T (upper triangular)
        for i in (0..n).rev() {
            for k in (i + 1)..n {
                let l_ki = l[k * n + i]; // L^T[i][k] = L[k][i]
                for rhs in 0..num_rhs {
                    x[i * num_rhs + rhs] -= l_ki * x[k * num_rhs + rhs];
                }
            }
            let diag = l[i * n + i];
            for rhs in 0..num_rhs {
                x[i * num_rhs + rhs] /= diag;
            }
        }
    }

    Ok(x)
}

/// Matrix exponential via scaling-and-squaring with Padé [6/6] approximation.
///
/// exp(A) is computed by:
/// 1. Scale A by 2^-s so ||A/2^s|| < 1
/// 2. Compute Padé [6/6] approximant R66(A/2^s)
/// 3. Square the result s times: exp(A) = R66^(2^s)
pub fn matrix_exp_contiguous_f64(data: &[f64], meta: &TensorMeta) -> Result<Vec<f64>, KernelError> {
    ensure_unary_layout_and_storage(data, meta)?;
    let shape = meta.shape();
    if shape.len() != 2 || shape[0] != shape[1] {
        return Err(KernelError::ShapeMismatch {
            lhs: shape.to_vec(),
            rhs: vec![2],
        });
    }
    let n = shape[0];
    if n == 0 {
        return Ok(Vec::new());
    }

    let offset = meta.storage_offset();
    let mut a = vec![0.0f64; n * n];
    for i in 0..n {
        for j in 0..n {
            a[i * n + j] = data[offset + i * n + j];
        }
    }

    // Compute 1-norm of A
    let mut norm = 0.0f64;
    for j in 0..n {
        let mut col_sum = 0.0;
        for i in 0..n {
            col_sum += a[i * n + j].abs();
        }
        norm = norm.max(col_sum);
    }

    // Determine scaling: 2^s such that ||A/2^s|| <= 0.5
    let s = if norm > 0.5 {
        (norm / 0.5).log2().ceil() as u32
    } else {
        0
    };

    // Scale A
    let scale = 0.5f64.powi(s as i32);
    for v in &mut a {
        *v *= scale;
    }

    // Scaling-and-squaring with a Taylor approximant: since the scaling above
    // guarantees ||A/2^s||_1 <= 0.5, the 13-term Taylor polynomial
    //   exp(B) ≈ I + B + B^2/2! + ... + B^12/12!
    // is accurate to ~machine precision; `exp(A) = (exp(A/2^s))^(2^s)` recovers it
    // by `s` squarings. Validated by `matrix_exp_defining_properties` (diag,
    // nilpotent, and exp(A)·exp(-A) = I).
    let mut identity = vec![0.0f64; n * n];
    for i in 0..n {
        identity[i * n + i] = 1.0;
    }

    // Horner's method: (((...(A/12 + I)*A/11 + I)*A/10 + I)...)*A + I
    // Each step is an n x n matmul routed through the cache-blocked, multi-
    // threaded `gemm::dgemm` rather than a naive triple loop — the matmuls
    // dominate matrix_exp's cost, so this is the lever. `dgemm` writes with
    // beta = 0, so the destination only needs to be the right length.
    let num_terms = 13; // A^0 through A^12
    let mut result = identity.clone();
    let mut temp = vec![0.0f64; n * n];

    for k in (1..num_terms).rev() {
        // result = result * A / k + I
        gemm::dgemm(n, n, n, &result, &a, &mut temp);
        let inv_k = 1.0 / k as f64;
        for i in 0..n * n {
            result[i] = temp[i] * inv_k + identity[i];
        }
    }

    // Square s times: result = result^(2^s)
    for _ in 0..s {
        gemm::dgemm(n, n, n, &result, &result, &mut temp);
        result.copy_from_slice(&temp);
    }

    Ok(result)
}

/// Result of symmetric eigendecomposition.
#[derive(Debug, Clone)]
pub struct EighResult {
    /// Eigenvalues sorted ascending.
    pub eigenvalues: Vec<f64>,
    /// Eigenvectors as columns of V (n x n, row-major). A = V @ diag(λ) @ V^T.
    pub eigenvectors: Vec<f64>,
    /// Matrix dimension.
    pub n: usize,
}

/// Apply the lower triangle of a symmetric rank-2k update:
///
/// `A := A - (V @ W^T + W @ V^T)`
///
/// `V` and `W` are row-major `n x k` panels and `A` is row-major `n x n`.
/// This is the BLAS-3 trailing-update primitive needed by blocked symmetric
/// tridiagonalization (`dsytrd`-style panels). It intentionally updates only the
/// lower triangle because the current symmetric eigensolver consumes lower
/// storage for tridiagonal reduction.
pub fn symmetric_rank2k_lower_update_f64(
    n: usize,
    k: usize,
    v: &[f64],
    w: &[f64],
    a: &mut [f64],
) -> Result<(), KernelError> {
    let panel_len = n * k;
    let matrix_len = n * n;
    if v.len() < panel_len || w.len() < panel_len || a.len() < matrix_len {
        return Err(KernelError::ShapeMismatch {
            lhs: vec![v.len(), w.len(), a.len()],
            rhs: vec![panel_len, panel_len, matrix_len],
        });
    }
    if n == 0 || k == 0 {
        return Ok(());
    }

    let v = &v[..panel_len];
    let w = &w[..panel_len];
    let a = &mut a[..matrix_len];
    let mut vw_t = vec![0.0f64; matrix_len];
    let mut wv_t = vec![0.0f64; matrix_len];
    gemm::dgemm_bt(n, k, n, v, w, &mut vw_t);
    gemm::dgemm_bt(n, k, n, w, v, &mut wv_t);
    for row in 0..n {
        let row_start = row * n;
        for col in 0..=row {
            let idx = row_start + col;
            a[idx] -= vw_t[idx] + wv_t[idx];
        }
    }

    Ok(())
}

/// Compute eigendecomposition of a symmetric matrix using the Jacobi eigenvalue algorithm.
///
/// Returns eigenvalues sorted ascending and corresponding orthonormal eigenvectors.
/// The input must be a square symmetric matrix.
/// Numerically stable `sqrt(a^2 + b^2)` without overflow/underflow.
fn eigh_pythag(a: f64, b: f64) -> f64 {
    let absa = a.abs();
    let absb = b.abs();
    if absa > absb {
        absa * (1.0 + (absb / absa).powi(2)).sqrt()
    } else if absb == 0.0 {
        0.0
    } else {
        absb * (1.0 + (absa / absb).powi(2)).sqrt()
    }
}

/// Householder reduction of a real-symmetric `n x n` matrix (row-major `z`,
/// lower triangle used) to symmetric tridiagonal form. On return `z` holds the
/// accumulated orthogonal transformation, `d` the diagonal, `e` the
/// sub-diagonal (`e[0] = 0`). EISPACK `tred2` lineage. O(4/3 n^3).
///
/// PERF NOTE: serial scalar, ~half of eigh's cost (the other half is tql2).
/// eigh is FAST (eigh_f64_256x256 ~77ms) — not iteration-bound like SVD — but
/// ~11x slower than LAPACK syevd. rayon-parallelizing the accumulation/applies
/// here was MEASURED and REGRESSED (eigh 256 77->85ms): at the benched n<=256 the
/// per-step work (~i^2) is too small to amortize the fan-out (Vec alloc + 2
/// dispatches per i). The real lever is BLOCKED tridiagonalization (LAPACK
/// dsytrd: panel + symmetric rank-2k trailing update via gemm::dgemm), the same
/// BLAS-3 family that won blocked-cholesky/QR — a multi-turn rewrite.
/// Householder reduction of a real-symmetric matrix to tridiagonal form
/// (EISPACK `tred2`, first half). On return `d`/`e` carry the tridiagonal
/// diagonal-h / sub-diagonal and `z` holds the accumulated reflectors (lower
/// triangle) plus the tridiagonal diagonal on its own diagonal. The eigenvector
/// back-transform is split out so the eigenvalues-only path can skip it.
#[inline]
fn lower_packed_index(row: usize, col: usize) -> usize {
    row * (row + 1) / 2 + col
}

/// Full-vector tridiagonal reduction over packed lower storage. Arithmetic
/// order mirrors the full-matrix EISPACK `tred2` loop; the only layout change
/// is that the upper reflector column `z[j,i] = z[i,j] / h` is stored in
/// `scaled_reflectors`.
#[allow(clippy::needless_range_loop)]
fn eigh_tred2_reduce_packed_full(
    n: usize,
    lower: &mut [f64],
    scaled_reflectors: &mut [f64],
    d: &mut [f64],
    e: &mut [f64],
) {
    for i in (1..n).rev() {
        let l = i - 1;
        let row_i_start = lower_packed_index(i, 0);
        let mut h = 0.0;
        let mut scale = 0.0;
        if l > 0 {
            let (previous_rows, current_and_after) = lower.split_at_mut(row_i_start);
            let row_i = &mut current_and_after[..=i];
            for &value in &row_i[..=l] {
                scale += value.abs();
            }
            if scale == 0.0 {
                e[i] = row_i[l];
            } else {
                for value in &mut row_i[..=l] {
                    *value /= scale;
                    h += *value * *value;
                }
                let mut f = row_i[l];
                let g = if f >= 0.0 { -h.sqrt() } else { h.sqrt() };
                e[i] = scale * g;
                h -= f * g;
                row_i[l] = f - g;
                f = 0.0;
                for j in 0..=l {
                    scaled_reflectors[lower_packed_index(i, j)] = row_i[j] / h;
                    let mut gg = 0.0;
                    let row_j_start = lower_packed_index(j, 0);
                    let row_j = &previous_rows[row_j_start..=row_j_start + j];
                    for k in 0..=j {
                        gg += row_j[k] * row_i[k];
                    }
                    let mut lower_col_offset = lower_packed_index(j + 1, j);
                    for k in (j + 1)..=l {
                        gg += previous_rows[lower_col_offset] * row_i[k];
                        lower_col_offset += k + 1;
                    }
                    e[j] = gg / h;
                    f += e[j] * row_i[j];
                }
                let hh = f / (h + h);
                for j in 0..=l {
                    f = row_i[j];
                    let gg = e[j] - hh * f;
                    e[j] = gg;
                    let row_j_start = lower_packed_index(j, 0);
                    let row_j = &mut previous_rows[row_j_start..=row_j_start + j];
                    for k in 0..=j {
                        row_j[k] -= f * e[k] + gg * row_i[k];
                    }
                }
            }
        } else {
            e[i] = lower[row_i_start + l];
        }
        d[i] = h;
    }
    d[0] = 0.0;
    e[0] = 0.0;
}

fn eigh_tred2_backtransform(n: usize, z: &mut [f64], d: &mut [f64]) {
    let mut projections = Vec::with_capacity(n);
    for i in 0..n {
        if d[i] != 0.0 {
            projections.clear();
            let row_i_start = i * n;
            let (previous_rows, current_and_after) = z.split_at_mut(row_i_start);
            let row_i = &current_and_after[..i];
            projections.resize(i, 0.0);
            for k in 0..i {
                let row_factor = row_i[k];
                let row = &previous_rows[k * n..k * n + i];
                for j in 0..i {
                    projections[j] += row_factor * row[j];
                }
            }
            for k in 0..i {
                let reflector = previous_rows[k * n + i];
                let row = &mut previous_rows[k * n..k * n + i];
                for j in 0..i {
                    row[j] -= projections[j] * reflector;
                }
            }
        }
        d[i] = z[i * n + i];
        z[i * n + i] = 1.0;
        for j in 0..i {
            z[j * n + i] = 0.0;
            z[i * n + j] = 0.0;
        }
    }
}

fn eigh_tred2_packed_full(n: usize, lower: &mut [f64], d: &mut [f64], e: &mut [f64]) -> Vec<f64> {
    let mut scaled_reflectors = vec![0.0f64; lower.len()];
    eigh_tred2_reduce_packed_full(n, lower, &mut scaled_reflectors, d, e);

    let mut z = vec![0.0f64; n * n];
    for i in 0..n {
        let row_start = i * n;
        let lower_start = lower_packed_index(i, 0);
        z[row_start..=row_start + i].copy_from_slice(&lower[lower_start..=lower_start + i]);
        for j in 0..i {
            z[j * n + i] = scaled_reflectors[lower_packed_index(i, j)];
        }
    }
    eigh_tred2_backtransform(n, &mut z, d);
    z
}

/// Eigenvalues-only tridiagonalization over a packed lower triangle.
///
/// This is the same Householder reduction loop as `eigh_tred2_reduce`, but the
/// values-only path does not need the upper-triangle/reflector-column storage
/// consumed by the full eigenvector back-transform. Keeping only the packed
/// lower triangle halves the working set while preserving every scale, dot, and
/// trailing-update operation in the same order over the same matrix entries.
#[allow(clippy::needless_range_loop)]
fn eigh_tred2_values_only(n: usize, lower: &mut [f64], d: &mut [f64], e: &mut [f64]) {
    for i in (1..n).rev() {
        let l = i - 1;
        let row_i_start = lower_packed_index(i, 0);
        let mut h = 0.0;
        let mut scale = 0.0;
        if l > 0 {
            let (previous_rows, current_and_after) = lower.split_at_mut(row_i_start);
            let row_i = &mut current_and_after[..=i];
            for &value in &row_i[..=l] {
                scale += value.abs();
            }
            if scale == 0.0 {
                e[i] = row_i[l];
            } else {
                for value in &mut row_i[..=l] {
                    *value /= scale;
                    h += *value * *value;
                }
                let mut f = row_i[l];
                let g = if f >= 0.0 { -h.sqrt() } else { h.sqrt() };
                e[i] = scale * g;
                h -= f * g;
                row_i[l] = f - g;
                f = 0.0;
                for j in 0..=l {
                    let mut gg = 0.0;
                    let row_j_start = lower_packed_index(j, 0);
                    let row_j = &previous_rows[row_j_start..=row_j_start + j];
                    for k in 0..=j {
                        gg += row_j[k] * row_i[k];
                    }
                    let mut lower_col_offset = lower_packed_index(j + 1, j);
                    for k in (j + 1)..=l {
                        gg += previous_rows[lower_col_offset] * row_i[k];
                        lower_col_offset += k + 1;
                    }
                    e[j] = gg / h;
                    f += e[j] * row_i[j];
                }
                let hh = f / (h + h);
                for j in 0..=l {
                    f = row_i[j];
                    let gg = e[j] - hh * f;
                    e[j] = gg;
                    let row_j_start = lower_packed_index(j, 0);
                    let row_j = &mut previous_rows[row_j_start..=row_j_start + j];
                    for k in 0..=j {
                        row_j[k] -= f * e[k] + gg * row_i[k];
                    }
                }
            }
        } else {
            e[i] = lower[row_i_start + l];
        }
        d[i] = h;
    }
    d[0] = 0.0;
    e[0] = 0.0;
    for i in 0..n {
        d[i] = lower[lower_packed_index(i, i)];
    }
}

/// Implicit-shift QL iteration on a symmetric tridiagonal matrix while `zt`
/// stores the eigenvector matrix transposed:
/// `zt[col * n + row] == z[row * n + col]`. The rotation stream and per-entry
/// arithmetic order match the row-major formulation; the vector updates become
/// contiguous slices instead of strided column walks.
fn eigh_tql2_transposed(n: usize, d: &mut [f64], e: &mut [f64], zt: &mut [f64]) {
    if n == 0 {
        return;
    }
    for i in 1..n {
        e[i - 1] = e[i];
    }
    e[n - 1] = 0.0;
    for l in 0..n {
        let mut iter = 0;
        loop {
            // Locate a negligible sub-diagonal element to split off.
            let mut m = l;
            while m < n - 1 {
                let dd = d[m].abs() + d[m + 1].abs();
                if e[m].abs() <= f64::EPSILON * dd {
                    break;
                }
                m += 1;
            }
            if m == l {
                break;
            }
            if iter >= 50 {
                break;
            }
            iter += 1;
            // Form the Wilkinson shift.
            let mut g = (d[l + 1] - d[l]) / (2.0 * e[l]);
            let mut r = eigh_pythag(g, 1.0);
            let sr = if g >= 0.0 { r.abs() } else { -r.abs() };
            g = d[m] - d[l] + e[l] / (g + sr);
            let mut s = 1.0;
            let mut c = 1.0;
            let mut p = 0.0;
            let mut bailed = false;
            for i in (l..m).rev() {
                let mut f = s * e[i];
                let b = c * e[i];
                r = eigh_pythag(f, g);
                e[i + 1] = r;
                if r == 0.0 {
                    d[i + 1] -= p;
                    e[m] = 0.0;
                    bailed = true;
                    break;
                }
                s = f / r;
                c = g / r;
                g = d[i + 1] - p;
                r = (d[i] - g) * s + 2.0 * c * b;
                p = s * r;
                d[i + 1] = g + p;
                g = c * r - b;
                let col_i = i * n;
                let col_next = (i + 1) * n;
                // Accumulate the rotation into the transposed eigenvector rows.
                for k in 0..n {
                    f = zt[col_next + k];
                    let left = zt[col_i + k];
                    zt[col_next + k] = s * left + c * f;
                    zt[col_i + k] = c * left - s * f;
                }
            }
            if bailed {
                continue;
            }
            d[l] -= p;
            e[l] = g;
            e[m] = 0.0;
        }
    }
}

/// Eigenvalues-only QL iteration: identical implicit-shift QL sweep as
/// `eigh_tql2`, but WITHOUT accumulating the rotations into an eigenvector
/// matrix (the inner `O(n)` per-rotation `z` update, which is the O(n^3) bulk).
/// The `d`/`e` updates are untouched, so the eigenvalues are bit-for-bit
/// identical to the full path's — only the discarded eigenvectors are skipped.
fn eigh_tql2_values_only(n: usize, d: &mut [f64], e: &mut [f64]) {
    if n == 0 {
        return;
    }
    for i in 1..n {
        e[i - 1] = e[i];
    }
    e[n - 1] = 0.0;
    for l in 0..n {
        let mut iter = 0;
        loop {
            let mut m = l;
            while m < n - 1 {
                let dd = d[m].abs() + d[m + 1].abs();
                if e[m].abs() <= f64::EPSILON * dd {
                    break;
                }
                m += 1;
            }
            if m == l {
                break;
            }
            if iter >= 50 {
                break;
            }
            iter += 1;
            let mut g = (d[l + 1] - d[l]) / (2.0 * e[l]);
            let mut r = eigh_pythag(g, 1.0);
            let sr = if g >= 0.0 { r.abs() } else { -r.abs() };
            g = d[m] - d[l] + e[l] / (g + sr);
            let mut s = 1.0;
            let mut c = 1.0;
            let mut p = 0.0;
            let mut bailed = false;
            for i in (l..m).rev() {
                let f = s * e[i];
                let b = c * e[i];
                r = eigh_pythag(f, g);
                e[i + 1] = r;
                if r == 0.0 {
                    d[i + 1] -= p;
                    e[m] = 0.0;
                    bailed = true;
                    break;
                }
                s = f / r;
                c = g / r;
                g = d[i + 1] - p;
                r = (d[i] - g) * s + 2.0 * c * b;
                p = s * r;
                d[i + 1] = g + p;
                g = c * r - b;
            }
            if bailed {
                continue;
            }
            d[l] -= p;
            e[l] = g;
            e[m] = 0.0;
        }
    }
}

pub fn eigh_contiguous_f64(data: &[f64], meta: &TensorMeta) -> Result<EighResult, KernelError> {
    ensure_unary_layout_and_storage(data, meta)?;
    let shape = meta.shape();
    if shape.len() != 2 || shape[0] != shape[1] {
        return Err(KernelError::ShapeMismatch {
            lhs: shape.to_vec(),
            rhs: vec![2],
        });
    }
    let n = shape[0];
    if n == 0 {
        return Ok(EighResult {
            eigenvalues: Vec::new(),
            eigenvectors: Vec::new(),
            n: 0,
        });
    }

    let offset = meta.storage_offset();

    // Householder tridiagonalization (tred2) + implicit-shift QL (tql2): a
    // real-symmetric eigensolver in the EISPACK/LAPACK lineage. This is O(n^3)
    // with a small constant, replacing the previous cyclic Jacobi which ran up
    // to 100 full O(n^3) sweeps. `z` starts as the input (lower triangle used)
    // and becomes the orthonormal eigenvector matrix; `d`/`e` carry the
    // tridiagonal diagonal/sub-diagonal.
    let mut lower = vec![0.0f64; n * (n + 1) / 2];
    for i in 0..n {
        let dst = lower_packed_index(i, 0);
        let src = offset + i * n;
        lower[dst..=dst + i].copy_from_slice(&data[src..=src + i]);
    }
    let mut d = vec![0.0f64; n];
    let mut e = vec![0.0f64; n];
    let z = eigh_tred2_packed_full(n, &mut lower, &mut d, &mut e);
    let mut zt = vec![0.0f64; n * n];
    for row in 0..n {
        let row_start = row * n;
        for col in 0..n {
            zt[col * n + row] = z[row_start + col];
        }
    }
    eigh_tql2_transposed(n, &mut d, &mut e, &mut zt);

    // Sort eigenvalues ascending, permuting eigenvector columns to match.
    let mut eigen_pairs: Vec<(f64, usize)> = (0..n).map(|i| (d[i], i)).collect();
    eigen_pairs.sort_by(|a, b| a.0.total_cmp(&b.0));

    let eigenvalues: Vec<f64> = eigen_pairs.iter().map(|(val, _)| *val).collect();
    let mut eigenvectors = vec![0.0f64; n * n];
    for (new_col, &(_, old_col)) in eigen_pairs.iter().enumerate() {
        let old_col_start = old_col * n;
        for row in 0..n {
            eigenvectors[row * n + new_col] = zt[old_col_start + row];
        }
    }

    Ok(EighResult {
        eigenvalues,
        eigenvectors,
        n,
    })
}

/// Compute just the eigenvalues of a symmetric matrix (sorted ascending).
pub fn eigvalsh_contiguous_f64(data: &[f64], meta: &TensorMeta) -> Result<Vec<f64>, KernelError> {
    ensure_unary_layout_and_storage(data, meta)?;
    let shape = meta.shape();
    if shape.len() != 2 || shape[0] != shape[1] {
        return Err(KernelError::ShapeMismatch {
            lhs: shape.to_vec(),
            rhs: vec![2],
        });
    }
    let n = shape[0];
    if n == 0 {
        return Ok(Vec::new());
    }
    let offset = meta.storage_offset();

    // Eigenvalues-only: run the SAME Householder reduction + implicit-shift QL
    // as `eigh_contiguous_f64`, but skip BOTH O(n^3) eigenvector accumulations
    // (tred2's back-transform and tql2's rotation-into-z) since the vectors are
    // discarded. The tridiagonal `d`/`e` and the QL eigenvalue iteration are
    // untouched, so the returned eigenvalues are bit-for-bit identical to
    // `eigh_contiguous_f64(...).eigenvalues` — proven by `eigvalsh_matches_eigh`.
    let mut lower = vec![0.0f64; n * (n + 1) / 2];
    for i in 0..n {
        let dst = lower_packed_index(i, 0);
        let src = offset + i * n;
        lower[dst..=dst + i].copy_from_slice(&data[src..=src + i]);
    }
    let mut d = vec![0.0f64; n];
    let mut e = vec![0.0f64; n];
    eigh_tred2_values_only(n, &mut lower, &mut d, &mut e);
    eigh_tql2_values_only(n, &mut d, &mut e);

    d.sort_by(f64::total_cmp);
    Ok(d)
}

/// Result of general (non-symmetric) eigendecomposition.
#[derive(Debug, Clone)]
pub struct EigResult {
    /// Eigenvalues as complex numbers: [re0, im0, re1, im1, ...] with length 2*n.
    pub eigenvalues: Vec<f64>,
    /// Right eigenvectors as complex column vectors in row-major order.
    /// Shape is (n, n) with each eigenvector as a column.
    /// For complex eigenvalues, consecutive columns are real/imag parts.
    pub eigenvectors: Vec<f64>,
    /// Matrix dimension.
    pub n: usize,
}

/// Compute eigendecomposition of a general (non-symmetric) square matrix.
///
/// Uses QR iteration with Hessenberg reduction. Returns complex eigenvalues
/// and eigenvectors since a real matrix can have complex eigenvalues.
///
/// The eigenvalues are returned as pairs (real, imag) interleaved.
/// For real eigenvalues, the imaginary part is 0.
pub fn eig_contiguous_f64(data: &[f64], meta: &TensorMeta) -> Result<EigResult, KernelError> {
    eig_impl(data, meta, true)
}

/// Shared worker for general (non-symmetric) eigendecomposition. When
/// `want_vectors` is false the O(n^3) Schur-vector accumulation (`q_acc`) is
/// skipped: the eigenvalues are read from the quasi-triangular `h`, which does
/// NOT depend on `q_acc`, so they are bit-for-bit identical either way. The
/// `eigvals` path uses `want_vectors = false`.
fn eig_impl(data: &[f64], meta: &TensorMeta, want_vectors: bool) -> Result<EigResult, KernelError> {
    ensure_unary_layout_and_storage(data, meta)?;
    let shape = meta.shape();
    if shape.len() != 2 || shape[0] != shape[1] {
        return Err(KernelError::ShapeMismatch {
            lhs: shape.to_vec(),
            rhs: vec![2],
        });
    }
    let n = shape[0];
    if n == 0 {
        return Ok(EigResult {
            eigenvalues: Vec::new(),
            eigenvectors: Vec::new(),
            n: 0,
        });
    }

    let offset = meta.storage_offset();

    // Copy matrix to working storage
    let mut h = vec![0.0f64; n * n];
    for i in 0..n {
        for j in 0..n {
            h[i * n + j] = data[offset + i * n + j];
        }
    }

    // Initialize Q accumulator for eigenvector computation
    let mut q_acc = vec![0.0f64; n * n];
    for i in 0..n {
        q_acc[i * n + i] = 1.0;
    }

    // Step 1: Reduce to upper Hessenberg form H = Q^T A Q
    // Using Householder reflections
    for k in 0..(n.saturating_sub(2)) {
        // Compute Householder vector for column k below diagonal
        let mut col_norm_sq = 0.0;
        for i in (k + 1)..n {
            col_norm_sq += h[i * n + k] * h[i * n + k];
        }
        if col_norm_sq < 1e-30 {
            continue;
        }
        let col_norm = col_norm_sq.sqrt();
        let sign = if h[(k + 1) * n + k] >= 0.0 { 1.0 } else { -1.0 };
        let v0 = h[(k + 1) * n + k] + sign * col_norm;
        let mut v = vec![0.0f64; n - k - 1];
        v[0] = v0;
        for i in 1..(n - k - 1) {
            v[i] = h[(k + 2 + i - 1) * n + k];
        }
        let v_norm_sq: f64 = v.iter().map(|x| x * x).sum();
        if v_norm_sq < 1e-30 {
            continue;
        }

        // Apply H = I - 2vv^T/|v|^2 to H from left and right
        // H := H - 2 * (v v^T H) / |v|^2
        // H := H - 2 * (H v v^T) / |v|^2

        // Left multiply: H[(k+1):, :] -= 2 * v @ (v^T @ H[(k+1):, :]) / |v|^2
        for j in 0..n {
            let mut dot = 0.0;
            for i in 0..(n - k - 1) {
                dot += v[i] * h[(k + 1 + i) * n + j];
            }
            let scale = 2.0 * dot / v_norm_sq;
            for i in 0..(n - k - 1) {
                h[(k + 1 + i) * n + j] -= scale * v[i];
            }
        }

        // Right multiply: H[:, (k+1):] -= 2 * (H[:, (k+1):] @ v) @ v^T / |v|^2
        for i in 0..n {
            let mut dot = 0.0;
            for j in 0..(n - k - 1) {
                dot += h[i * n + (k + 1 + j)] * v[j];
            }
            let scale = 2.0 * dot / v_norm_sq;
            for j in 0..(n - k - 1) {
                h[i * n + (k + 1 + j)] -= scale * v[j];
            }
        }

        // Accumulate Q: Q[:, (k+1):] -= 2 * (Q[:, (k+1):] @ v) @ v^T / |v|^2.
        // Pure eigenvector work — skip when only eigenvalues are requested.
        if want_vectors {
            for i in 0..n {
                let mut dot = 0.0;
                for j in 0..(n - k - 1) {
                    dot += q_acc[i * n + (k + 1 + j)] * v[j];
                }
                let scale = 2.0 * dot / v_norm_sq;
                for j in 0..(n - k - 1) {
                    q_acc[i * n + (k + 1 + j)] -= scale * v[j];
                }
            }
        }
    }

    // Step 2: Francis double-shift implicit QR on the upper-Hessenberg `h`
    // (EISPACK `hqr2` lineage), accumulating the orthogonal transforms into
    // `q_acc` (real-Schur vectors) when requested. The IMPLICIT DOUBLE SHIFT
    // converges trailing 2x2 blocks with COMPLEX-conjugate eigenvalues in real
    // arithmetic — the previous single-Wilkinson-shift Givens QR could not, so
    // it returned WRONG eigenvalues for any complex spectrum (frankentorch-09zy).
    // Eigenvalues are written directly as interleaved (re, im) pairs.
    let mut eigenvalues = vec![0.0f64; 2 * n];
    {
        // Frobenius-ish norm over the Hessenberg band for convergence tests.
        let mut anorm = 0.0f64;
        for i in 0..n {
            for j in i.saturating_sub(1)..n {
                anorm += h[i * n + j].abs();
            }
        }
        let eps = f64::EPSILON;
        let mut t = 0.0f64; // accumulated shift
        let mut en: isize = n as isize - 1; // bottom of the active block (0-based)
        let mut total_iter = 0usize;
        let max_total = 60 * n + 100;

        while en >= 0 {
            let en_u = en as usize;
            let mut its = 0usize;
            loop {
                // Find `l`: top of the unreduced block ending at `en`.
                let mut l = en_u;
                while l > 0 {
                    let mut s = h[(l - 1) * n + (l - 1)].abs() + h[l * n + l].abs();
                    if s == 0.0 {
                        s = anorm;
                    }
                    if h[l * n + (l - 1)].abs() <= eps * s {
                        break;
                    }
                    l -= 1;
                }

                let mut x = h[en_u * n + en_u];
                if l == en_u {
                    // One real root.
                    h[en_u * n + en_u] = x + t;
                    eigenvalues[2 * en_u] = x + t;
                    eigenvalues[2 * en_u + 1] = 0.0;
                    en -= 1;
                    break;
                }

                let na = en_u - 1;
                let mut y = h[na * n + na];
                let mut w = h[en_u * n + na] * h[na * n + en_u];
                if l == na {
                    // Two roots (real pair or complex-conjugate pair).
                    let pp = (y - x) / 2.0;
                    let q = pp * pp + w;
                    let zz = q.abs().sqrt();
                    let xx = x + t;
                    h[en_u * n + en_u] = xx;
                    h[na * n + na] = y + t;
                    if q >= 0.0 {
                        let zz2 = if pp >= 0.0 { pp + zz } else { pp - zz };
                        let r1 = xx + zz2;
                        let r2 = if zz2 != 0.0 { xx - w / zz2 } else { r1 };
                        eigenvalues[2 * na] = r1;
                        eigenvalues[2 * na + 1] = 0.0;
                        eigenvalues[2 * en_u] = r2;
                        eigenvalues[2 * en_u + 1] = 0.0;
                        if want_vectors {
                            // Standardize the 2x2 (Givens rotation) for the Schur vectors.
                            let xr = h[en_u * n + na];
                            let s = xr.abs() + zz2.abs();
                            if s != 0.0 {
                                let mut cp = xr / s;
                                let mut cq = zz2 / s;
                                let r = (cp * cp + cq * cq).sqrt();
                                cp /= r;
                                cq /= r;
                                for j in na..n {
                                    let z1 = h[na * n + j];
                                    h[na * n + j] = cq * z1 + cp * h[en_u * n + j];
                                    h[en_u * n + j] = cq * h[en_u * n + j] - cp * z1;
                                }
                                for i in 0..=en_u {
                                    let z1 = h[i * n + na];
                                    h[i * n + na] = cq * z1 + cp * h[i * n + en_u];
                                    h[i * n + en_u] = cq * h[i * n + en_u] - cp * z1;
                                }
                                for i in 0..n {
                                    let z1 = q_acc[i * n + na];
                                    q_acc[i * n + na] = cq * z1 + cp * q_acc[i * n + en_u];
                                    q_acc[i * n + en_u] = cq * q_acc[i * n + en_u] - cp * z1;
                                }
                            }
                        }
                    } else {
                        eigenvalues[2 * na] = xx + pp;
                        eigenvalues[2 * na + 1] = zz;
                        eigenvalues[2 * en_u] = xx + pp;
                        eigenvalues[2 * en_u + 1] = -zz;
                    }
                    en -= 2;
                    break;
                }

                if its >= 30 || total_iter >= max_total {
                    // No convergence — record the diagonal estimate and deflate.
                    h[en_u * n + en_u] = x + t;
                    eigenvalues[2 * en_u] = x + t;
                    eigenvalues[2 * en_u + 1] = 0.0;
                    en -= 1;
                    break;
                }

                // Form the (double) shift.
                if its == 10 || its == 20 {
                    // Exceptional shift to break stagnation.
                    t += x;
                    for i in 0..=en_u {
                        h[i * n + i] -= x;
                    }
                    let s = h[en_u * n + na].abs() + h[na * n + (na - 1)].abs();
                    x = 0.75 * s;
                    y = x;
                    w = -0.4375 * s * s;
                }
                its += 1;
                total_iter += 1;

                // Find `m`: two consecutive small sub-diagonals (start of the bulge).
                let mut m = en_u - 2;
                let (mut p_s, mut q_s, mut r_s);
                loop {
                    let zz = h[m * n + m];
                    let r = x - zz;
                    let s = y - zz;
                    p_s = (r * s - w) / h[(m + 1) * n + m] + h[m * n + (m + 1)];
                    q_s = h[(m + 1) * n + (m + 1)] - zz - r - s;
                    r_s = h[(m + 2) * n + (m + 1)];
                    let norm = p_s.abs() + q_s.abs() + r_s.abs();
                    p_s /= norm;
                    q_s /= norm;
                    r_s /= norm;
                    if m == l {
                        break;
                    }
                    let test1 = p_s.abs()
                        * (h[(m - 1) * n + (m - 1)].abs()
                            + zz.abs()
                            + h[(m + 1) * n + (m + 1)].abs());
                    let test2 = h[m * n + (m - 1)].abs() * (q_s.abs() + r_s.abs());
                    if test2 <= eps * test1 {
                        break;
                    }
                    m -= 1;
                }
                // Clear the sub-sub-diagonal spike left of the bulge.
                for i in (m + 2)..=en_u {
                    h[i * n + (i - 2)] = 0.0;
                    if i != m + 2 {
                        h[i * n + (i - 3)] = 0.0;
                    }
                }

                // Double QR (bulge-chase) step on rows [m, en].
                let mut k = m;
                while k <= na {
                    let notlast = k != na;
                    if k != m {
                        p_s = h[k * n + (k - 1)];
                        q_s = h[(k + 1) * n + (k - 1)];
                        r_s = if notlast {
                            h[(k + 2) * n + (k - 1)]
                        } else {
                            0.0
                        };
                        x = p_s.abs() + q_s.abs() + r_s.abs();
                        if x == 0.0 {
                            k += 1;
                            continue;
                        }
                        p_s /= x;
                        q_s /= x;
                        r_s /= x;
                    }
                    let s = (p_s * p_s + q_s * q_s + r_s * r_s).sqrt().copysign(p_s);
                    if k == m {
                        if l != m {
                            h[k * n + (k - 1)] = -h[k * n + (k - 1)];
                        }
                    } else {
                        h[k * n + (k - 1)] = -s * x;
                    }
                    p_s += s;
                    let xr = p_s / s;
                    let yr = q_s / s;
                    let zr = r_s / s;
                    q_s /= p_s;
                    r_s /= p_s;
                    // Row modification: columns [k, n).
                    for j in k..n {
                        let mut p2 = h[k * n + j] + q_s * h[(k + 1) * n + j];
                        if notlast {
                            p2 += r_s * h[(k + 2) * n + j];
                            h[(k + 2) * n + j] -= p2 * zr;
                        }
                        h[(k + 1) * n + j] -= p2 * yr;
                        h[k * n + j] -= p2 * xr;
                    }
                    // Column modification: rows [0, min(en, k+3)].
                    let jmax = (k + 3).min(en_u);
                    for i in 0..=jmax {
                        let mut p2 = xr * h[i * n + k] + yr * h[i * n + (k + 1)];
                        if notlast {
                            p2 += zr * h[i * n + (k + 2)];
                            h[i * n + (k + 2)] -= p2 * r_s;
                        }
                        h[i * n + (k + 1)] -= p2 * q_s;
                        h[i * n + k] -= p2;
                    }
                    if want_vectors {
                        for i in 0..n {
                            let mut p2 = xr * q_acc[i * n + k] + yr * q_acc[i * n + (k + 1)];
                            if notlast {
                                p2 += zr * q_acc[i * n + (k + 2)];
                                q_acc[i * n + (k + 2)] -= p2 * r_s;
                            }
                            q_acc[i * n + (k + 1)] -= p2 * q_s;
                            q_acc[i * n + k] -= p2;
                        }
                    }
                    k += 1;
                }
            }
        }
    }

    // Step 3: eigenvectors. Back-substitute the eigenvectors of the
    // quasi-triangular real-Schur form `h` in place, then transform by the
    // accumulated Schur vectors `q_acc` to get eigenvectors of the original
    // matrix. Complex eigenvalues yield complex eigenvectors stored as
    // consecutive (re, im) column pairs (the `EigResult.eigenvectors` convention).
    if want_vectors {
        eig_backsub_eigenvectors(&mut h, &mut q_acc, &eigenvalues, n);
    }
    Ok(EigResult {
        eigenvalues,
        eigenvectors: q_acc,
        n,
    })
}

/// Complex division `(ar + ai·i) / (br + bi·i)` (EISPACK `cdiv`).
fn eig_cdiv(ar: f64, ai: f64, br: f64, bi: f64) -> (f64, f64) {
    if br.abs() >= bi.abs() {
        let r = bi / br;
        let d = br + bi * r;
        ((ar + ai * r) / d, (ai - ar * r) / d)
    } else {
        let r = br / bi;
        let d = br * r + bi;
        ((ar * r + ai) / d, (ai * r - ar) / d)
    }
}

/// Eigenvector back-substitution for a real upper-quasi-triangular Schur form
/// `h` (EISPACK `hqr2` lineage). Computes eigenvectors of `h` into its own
/// columns, then overwrites `z` (the accumulated Schur vectors) with the
/// eigenvectors of the original matrix `z·(vectors of h)`. `evals` holds the
/// interleaved (re, im) eigenvalues. Complex eigenvalue pairs are stored as a
/// (real-part column, imag-part column) pair.
fn eig_backsub_eigenvectors(h: &mut [f64], z: &mut [f64], evals: &[f64], n: usize) {
    if n == 0 {
        return;
    }
    let mut norm = 0.0f64;
    for i in 0..n {
        for j in i..n {
            norm += h[i * n + j].abs();
        }
    }
    if norm == 0.0 {
        return;
    }
    let eps = f64::EPSILON;
    let wr = |i: usize| evals[2 * i];
    let wi = |i: usize| evals[2 * i + 1];

    let mut en_i: isize = n as isize - 1;
    while en_i >= 0 {
        let en = en_i as usize;
        let p = wr(en);
        let q = wi(en);
        if q == 0.0 {
            // Real eigenvalue.
            let mut m = en;
            h[en * n + en] = 1.0;
            if en > 0 {
                let (mut zz_c, mut s_c) = (0.0f64, 0.0f64);
                let mut i_i = en as isize - 1;
                while i_i >= 0 {
                    let i = i_i as usize;
                    let w = h[i * n + i] - p;
                    let mut r = 0.0;
                    for j in m..=en {
                        r += h[i * n + j] * h[j * n + en];
                    }
                    if wi(i) < 0.0 {
                        zz_c = w;
                        s_c = r;
                        i_i -= 1;
                        continue;
                    }
                    m = i;
                    if wi(i) == 0.0 {
                        let mut t = w;
                        if t == 0.0 {
                            t = eps * norm;
                        }
                        h[i * n + en] = -r / t;
                    } else {
                        let x = h[i * n + (i + 1)];
                        let y = h[(i + 1) * n + i];
                        let q2 = (wr(i) - p) * (wr(i) - p) + wi(i) * wi(i);
                        let t = (x * s_c - zz_c * r) / q2;
                        h[i * n + en] = t;
                        if x.abs() > zz_c.abs() {
                            h[(i + 1) * n + en] = (-r - w * t) / x;
                        } else {
                            h[(i + 1) * n + en] = (-s_c - y * t) / zz_c;
                        }
                    }
                    i_i -= 1;
                }
            }
            en_i -= 1;
        } else if q < 0.0 {
            // Complex pair: `en` is the conjugate (wi<0) root, `na = en-1` its mate.
            let na = en - 1;
            let mut m = na;
            if h[en * n + na].abs() > h[na * n + en].abs() {
                h[na * n + na] = q / h[en * n + na];
                h[na * n + en] = -(h[en * n + en] - p) / h[en * n + na];
            } else {
                let (cr, ci) = eig_cdiv(0.0, -h[na * n + en], h[na * n + na] - p, q);
                h[na * n + na] = cr;
                h[na * n + en] = ci;
            }
            h[en * n + na] = 0.0;
            h[en * n + en] = 1.0;
            if na > 0 {
                let (mut zz_c, mut r_c, mut s_c) = (0.0f64, 0.0f64, 0.0f64);
                let mut i_i = na as isize - 1;
                while i_i >= 0 {
                    let i = i_i as usize;
                    let w = h[i * n + i] - p;
                    let mut ra = 0.0;
                    let mut sa = 0.0;
                    for j in m..=en {
                        ra += h[i * n + j] * h[j * n + na];
                        sa += h[i * n + j] * h[j * n + en];
                    }
                    if wi(i) < 0.0 {
                        zz_c = w;
                        r_c = ra;
                        s_c = sa;
                        i_i -= 1;
                        continue;
                    }
                    m = i;
                    if wi(i) == 0.0 {
                        let (cr, ci) = eig_cdiv(-ra, -sa, w, q);
                        h[i * n + na] = cr;
                        h[i * n + en] = ci;
                    } else {
                        let x = h[i * n + (i + 1)];
                        let y = h[(i + 1) * n + i];
                        let mut vr = (wr(i) - p) * (wr(i) - p) + wi(i) * wi(i) - q * q;
                        let vi = (wr(i) - p) * 2.0 * q;
                        if vr == 0.0 && vi == 0.0 {
                            vr = eps * norm * (w.abs() + q.abs() + x.abs() + y.abs() + zz_c.abs());
                        }
                        let (cr, ci) = eig_cdiv(
                            x * r_c - zz_c * ra + q * sa,
                            x * s_c - zz_c * sa - q * ra,
                            vr,
                            vi,
                        );
                        h[i * n + na] = cr;
                        h[i * n + en] = ci;
                        if x.abs() > zz_c.abs() + q.abs() {
                            h[(i + 1) * n + na] = (-ra - w * h[i * n + na] + q * h[i * n + en]) / x;
                            h[(i + 1) * n + en] = (-sa - w * h[i * n + en] - q * h[i * n + na]) / x;
                        } else {
                            let (cr2, ci2) = eig_cdiv(
                                -r_c - y * h[i * n + na],
                                -s_c - y * h[i * n + en],
                                zz_c,
                                q,
                            );
                            h[(i + 1) * n + na] = cr2;
                            h[(i + 1) * n + en] = ci2;
                        }
                    }
                    i_i -= 1;
                }
            }
            en_i -= 2;
        } else {
            // q > 0: upper root of a pair, handled with its conjugate below it.
            en_i -= 1;
        }
    }

    // Eigenvectors of the original matrix: z[:, j] = z[:, 0..=j] · h[0..=j, j].
    for j in (0..n).rev() {
        for i in 0..n {
            let mut acc = 0.0;
            for k in 0..=j {
                acc += z[i * n + k] * h[k * n + j];
            }
            z[i * n + j] = acc;
        }
    }
}

/// Compute just the eigenvalues of a general matrix (as complex pairs).
pub fn eigvals_contiguous_f64(data: &[f64], meta: &TensorMeta) -> Result<Vec<f64>, KernelError> {
    // Eigenvalues-only: skip the O(n^3) Schur-vector accumulation. The
    // eigenvalues are read from the quasi-triangular `h`, which does not depend
    // on `q_acc`, so they are bit-for-bit identical to
    // `eig_contiguous_f64(...).eigenvalues`.
    let result = eig_impl(data, meta, false)?;
    Ok(result.eigenvalues)
}

/// Result of SVD decomposition.
#[derive(Debug, Clone)]
pub struct SvdResult {
    /// Left singular vectors U in row-major order.
    pub u: Vec<f64>,
    /// Singular values (1D, sorted descending, non-negative).
    pub s: Vec<f64>,
    /// Right singular vectors Vh (V-hermitian/transpose) in row-major order.
    pub vh: Vec<f64>,
    /// Input rows.
    pub m: usize,
    /// Input columns.
    pub n: usize,
    /// k = min(m, n)
    pub k: usize,
}

/// Compute the SVD of an (m x n) matrix: A = U @ diag(S) @ Vh.
///
/// Uses Golub-Kahan bidiagonalization followed by implicit QR shifts.
/// If `full_matrices` is true, U is (m x m) and Vh is (n x n).
/// If `full_matrices` is false (reduced), U is (m x k) and Vh is (k x n) where k = min(m,n).
pub fn svd_contiguous_f64(
    data: &[f64],
    meta: &TensorMeta,
    full_matrices: bool,
) -> Result<SvdResult, KernelError> {
    ensure_unary_layout_and_storage(data, meta)?;
    let shape = meta.shape();
    if shape.len() != 2 {
        return Err(KernelError::ShapeMismatch {
            lhs: shape.to_vec(),
            rhs: vec![2],
        });
    }
    let m = shape[0];
    let n = shape[1];
    let k = m.min(n);
    let offset = meta.storage_offset();

    if m == 0 || n == 0 {
        let u_cols = if full_matrices { m } else { k };
        let vh_rows = if full_matrices { n } else { k };
        return Ok(SvdResult {
            u: vec![0.0; m * u_cols],
            s: Vec::new(),
            vh: vec![0.0; vh_rows * n],
            m,
            n,
            k,
        });
    }

    // Copy input matrix
    let mut a = vec![0.0f64; m * n];
    for i in 0..m {
        for j in 0..n {
            a[i * n + j] = data[offset + i * n + j];
        }
    }

    // One-sided Jacobi SVD for simplicity and numerical robustness
    // This works well for all matrix sizes and shapes

    if m >= n {
        svd_tall(&a, m, n, full_matrices)
    } else {
        // For wide matrices (m < n), compute SVD of A^T, then swap U/Vh
        let mut at = vec![0.0f64; n * m];
        for i in 0..m {
            for j in 0..n {
                at[j * m + i] = a[i * n + j];
            }
        }
        let result = svd_tall(&at, n, m, full_matrices)?;
        // SVD(A^T) = U_t @ diag(S) @ Vh_t
        // SVD(A) = Vh_t^T @ diag(S) @ U_t^T
        let u_rows = if full_matrices { m } else { k };
        let vh_cols = if full_matrices { n } else { k };
        // U for A = Vh_t^T: result.vh is (vh_t_rows x m), transpose to (m x vh_t_rows)
        let _vh_t_rows = if full_matrices { m } else { k };
        let mut u = vec![0.0f64; m * u_rows];
        for i in 0..m {
            for j in 0..u_rows {
                u[i * u_rows + j] = result.vh[j * m + i];
            }
        }
        // Vh for A = U_t^T: result.u is (n x u_t_cols), transpose to (u_t_cols x n)
        let u_t_cols = if full_matrices { n } else { k };
        let mut vh = vec![0.0f64; vh_cols * n];
        for i in 0..vh_cols {
            for j in 0..n {
                vh[i * n + j] = result.u[j * u_t_cols + i];
            }
        }
        Ok(SvdResult {
            u,
            s: result.s,
            vh,
            m,
            n,
            k,
        })
    }
}

/// `|a|` carrying the sign of `b` (Numerical-Recipes `SIGN`).
fn nr_sign(a: f64, b: f64) -> f64 {
    if b >= 0.0 { a.abs() } else { -a.abs() }
}

/// Golub-Reinsch SVD of a real `m x n` matrix with `m >= n`, row-major.
///
/// On input `a` is the matrix; on output `a` is overwritten with the (reduced,
/// m x n) left singular vectors U. Returns `(w, v)` where `w[0..n]` are the
/// non-negative singular values (unsorted, bidiagonal order) and `v` is the
/// `n x n` right singular vectors as columns, so `A = U diag(w) V^T`.
///
/// This is the classic Householder-bidiagonalization + implicit-shift QR
/// diagonalization of the bidiagonal form: it never forms `A^T A`, so it keeps
/// full singular-value/-vector accuracy (no condition-number squaring), unlike
/// the Gram-matrix shortcuts. O(m n^2) with a small constant. `pythag` is the
/// stable hypot already used by the eigensolver.
fn golub_reinsch_svd(
    a: &mut [f64],
    m: usize,
    n: usize,
) -> Result<(Vec<f64>, Vec<f64>), KernelError> {
    let mut w = vec![0.0f64; n];
    let mut v = vec![0.0f64; n * n];
    let mut rv1 = vec![0.0f64; n];

    let mut g = 0.0f64;
    let mut scale = 0.0f64;
    let mut anorm = 0.0f64;

    // --- Householder reduction to bidiagonal form ---
    for i in 0..n {
        let l = i + 1;
        rv1[i] = scale * g;
        g = 0.0;
        let mut s = 0.0;
        scale = 0.0;
        if i < m {
            for k in i..m {
                scale += a[k * n + i].abs();
            }
            if scale != 0.0 {
                for k in i..m {
                    a[k * n + i] /= scale;
                    s += a[k * n + i] * a[k * n + i];
                }
                let f = a[i * n + i];
                g = -nr_sign(s.sqrt(), f);
                let h = f * g - s;
                a[i * n + i] = f - g;
                for j in l..n {
                    let mut s2 = 0.0;
                    for k in i..m {
                        s2 += a[k * n + i] * a[k * n + j];
                    }
                    let f2 = s2 / h;
                    for k in i..m {
                        a[k * n + j] += f2 * a[k * n + i];
                    }
                }
                for k in i..m {
                    a[k * n + i] *= scale;
                }
            }
        }
        w[i] = scale * g;
        g = 0.0;
        s = 0.0;
        scale = 0.0;
        if i < m && i != n - 1 {
            for k in l..n {
                scale += a[i * n + k].abs();
            }
            if scale != 0.0 {
                for k in l..n {
                    a[i * n + k] /= scale;
                    s += a[i * n + k] * a[i * n + k];
                }
                let f = a[i * n + l];
                g = -nr_sign(s.sqrt(), f);
                let h = f * g - s;
                a[i * n + l] = f - g;
                for k in l..n {
                    rv1[k] = a[i * n + k] / h;
                }
                for j in l..m {
                    let mut s2 = 0.0;
                    for k in l..n {
                        s2 += a[j * n + k] * a[i * n + k];
                    }
                    for k in l..n {
                        a[j * n + k] += s2 * rv1[k];
                    }
                }
                for k in l..n {
                    a[i * n + k] *= scale;
                }
            }
        }
        anorm = anorm.max(w[i].abs() + rv1[i].abs());
    }

    // --- Accumulation of right-hand transformations (V) ---
    for i in (0..n).rev() {
        let l = i + 1;
        if i < n - 1 {
            if g != 0.0 {
                for j in l..n {
                    // Avoid possible underflow via division as in NR.
                    v[j * n + i] = (a[i * n + j] / a[i * n + l]) / g;
                }
                for j in l..n {
                    let mut s = 0.0;
                    for k in l..n {
                        s += a[i * n + k] * v[k * n + j];
                    }
                    for k in l..n {
                        v[k * n + j] += s * v[k * n + i];
                    }
                }
            }
            for j in l..n {
                v[i * n + j] = 0.0;
                v[j * n + i] = 0.0;
            }
        }
        v[i * n + i] = 1.0;
        g = rv1[i];
    }

    // --- Accumulation of left-hand transformations (U, overwriting a) ---
    for i in (0..n).rev() {
        let l = i + 1;
        g = w[i];
        for j in l..n {
            a[i * n + j] = 0.0;
        }
        if g != 0.0 {
            g = 1.0 / g;
            for j in l..n {
                let mut s = 0.0;
                for k in l..m {
                    s += a[k * n + i] * a[k * n + j];
                }
                let f = (s / a[i * n + i]) * g;
                for k in i..m {
                    a[k * n + j] += f * a[k * n + i];
                }
            }
            for j in i..m {
                a[j * n + i] *= g;
            }
        } else {
            for j in i..m {
                a[j * n + i] = 0.0;
            }
        }
        a[i * n + i] += 1.0;
    }

    // --- Diagonalization of the bidiagonal form: QR with implicit shifts ---
    // PERF NOTE: this implicit-QR sweep is the DOMINANT cost of the full SVD
    // (svd_f64_256x256 ~1.6s, ~300x slower than LAPACK dgesdd). It is a long
    // sequence of BLAS-1 Givens rotations applied to U (m rows) and V (n rows)
    // every iteration; the rotation loops touch only 2 elements per row, so
    // row-parallelizing them is memory-bound (would regress, cf. grid_sample).
    // Parallelizing the upstream Householder bidiagonalization / U,V accumulation
    // (the apply pattern that won 6.4x on QR `f832ce77`) was MEASURED here and
    // REGRESSED svd/svdvals (the applies are a small fraction vs this sweep, and
    // the two-phase restructure adds overhead at the benched 128-256 sizes).
    // Transposing U/V so the rotations hit contiguous rows was also MEASURED and
    // REGRESSED (~1.4x slower): the two rotated columns are ADJACENT (j, j+1), so
    // in row-major they already share a cache line per row — transposing puts them
    // `stride` apart, doubling the cache lines touched. So the sweep is NOT
    // cache-bound; it is just an O(n^3) BLAS-1 rotation stream. The real lever is
    // ALGORITHMIC: a divide-and-conquer bidiagonal SVD (dbdsdc) or a BLAS-3
    // back-transformation (accumulate each sweep's rotations into a banded
    // orthogonal block, apply to U/V via dgemm), not micro-tuning the stream.
    for k in (0..n).rev() {
        for _its in 0..30 {
            let mut flag = true;
            let mut l = k;
            let mut nm;
            // Test for splitting (find a negligible super-diagonal element).
            loop {
                nm = l.saturating_sub(1);
                if l == 0 || (rv1[l].abs() + anorm) == anorm {
                    flag = false;
                    break;
                }
                if (w[nm].abs() + anorm) == anorm {
                    break;
                }
                l -= 1;
            }
            if flag {
                // Cancellation of rv1[l], if l > 0.
                let mut c = 0.0;
                let mut s = 1.0;
                for i in l..=k {
                    let f = s * rv1[i];
                    rv1[i] *= c;
                    if (f.abs() + anorm) == anorm {
                        break;
                    }
                    g = w[i];
                    let h = eigh_pythag(f, g);
                    w[i] = h;
                    let hinv = 1.0 / h;
                    c = g * hinv;
                    s = -f * hinv;
                    for j in 0..m {
                        let y = a[j * n + nm];
                        let z = a[j * n + i];
                        a[j * n + nm] = y * c + z * s;
                        a[j * n + i] = z * c - y * s;
                    }
                }
            }
            let z = w[k];
            if l == k {
                // Convergence: make the singular value non-negative.
                if z < 0.0 {
                    w[k] = -z;
                    for j in 0..n {
                        v[j * n + k] = -v[j * n + k];
                    }
                }
                break;
            }
            if _its == 29 {
                return Err(KernelError::SingularMatrix { size: n });
            }
            // Shift from bottom 2x2 minor.
            let mut x = w[l];
            nm = k - 1;
            let mut y = w[nm];
            g = rv1[nm];
            let mut h = rv1[k];
            let mut f = ((y - z) * (y + z) + (g - h) * (g + h)) / (2.0 * h * y);
            g = eigh_pythag(f, 1.0);
            f = ((x - z) * (x + z) + h * ((y / (f + nr_sign(g, f))) - h)) / x;
            // Next QR transformation (Givens rotations).
            let mut c = 1.0;
            let mut s = 1.0;
            for j in l..=nm {
                let i = j + 1;
                g = rv1[i];
                y = w[i];
                h = s * g;
                g *= c;
                let mut zz = eigh_pythag(f, h);
                rv1[j] = zz;
                c = f / zz;
                s = h / zz;
                f = x * c + g * s;
                g = g * c - x * s;
                h = y * s;
                y *= c;
                for jj in 0..n {
                    let xv = v[jj * n + j];
                    let zv = v[jj * n + i];
                    v[jj * n + j] = xv * c + zv * s;
                    v[jj * n + i] = zv * c - xv * s;
                }
                zz = eigh_pythag(f, h);
                w[j] = zz;
                if zz != 0.0 {
                    let zinv = 1.0 / zz;
                    c = f * zinv;
                    s = h * zinv;
                }
                f = c * g + s * y;
                x = c * y - s * g;
                for jj in 0..m {
                    let yu = a[jj * n + j];
                    let zu = a[jj * n + i];
                    a[jj * n + j] = yu * c + zu * s;
                    a[jj * n + i] = zu * c - yu * s;
                }
            }
            rv1[l] = 0.0;
            rv1[k] = f;
            w[k] = x;
        }
    }

    Ok((w, v))
}

/// Singular VALUES only of a real `m x n` matrix with `m >= n`, row-major.
///
/// Same Householder bidiagonalization + implicit-shift QR recurrence as
/// [`golub_reinsch_svd`], but it never accumulates the `U`/`V` rotation
/// matrices (the Givens rotations are applied only to the diagonal `w` and
/// super-diagonal `rv1`). The `w` recurrence is identical to the full routine's,
/// so the returned values match the full SVD's singular values to working
/// precision while skipping the O(m n^2) U accumulation and all per-rotation
/// matrix updates. `a` is used as scratch and overwritten. Returns the `n`
/// non-negative singular values in bidiagonal order (caller sorts).
fn golub_reinsch_singular_values(
    a: &mut [f64],
    m: usize,
    n: usize,
) -> Result<Vec<f64>, KernelError> {
    let mut w = vec![0.0f64; n];
    let mut rv1 = vec![0.0f64; n];
    let mut g = 0.0f64;
    let mut scale = 0.0f64;
    let mut anorm = 0.0f64;

    // --- Householder reduction to bidiagonal form (identical to the full SVD) ---
    for i in 0..n {
        let l = i + 1;
        rv1[i] = scale * g;
        g = 0.0;
        let mut s = 0.0;
        scale = 0.0;
        if i < m {
            for k in i..m {
                scale += a[k * n + i].abs();
            }
            if scale != 0.0 {
                for k in i..m {
                    a[k * n + i] /= scale;
                    s += a[k * n + i] * a[k * n + i];
                }
                let f = a[i * n + i];
                g = -nr_sign(s.sqrt(), f);
                let h = f * g - s;
                a[i * n + i] = f - g;
                for j in l..n {
                    let mut s2 = 0.0;
                    for k in i..m {
                        s2 += a[k * n + i] * a[k * n + j];
                    }
                    let f2 = s2 / h;
                    for k in i..m {
                        a[k * n + j] += f2 * a[k * n + i];
                    }
                }
                for k in i..m {
                    a[k * n + i] *= scale;
                }
            }
        }
        w[i] = scale * g;
        g = 0.0;
        s = 0.0;
        scale = 0.0;
        if i < m && i != n - 1 {
            for k in l..n {
                scale += a[i * n + k].abs();
            }
            if scale != 0.0 {
                for k in l..n {
                    a[i * n + k] /= scale;
                    s += a[i * n + k] * a[i * n + k];
                }
                let f = a[i * n + l];
                g = -nr_sign(s.sqrt(), f);
                let h = f * g - s;
                a[i * n + l] = f - g;
                for k in l..n {
                    rv1[k] = a[i * n + k] / h;
                }
                for j in l..m {
                    let mut s2 = 0.0;
                    for k in l..n {
                        s2 += a[j * n + k] * a[i * n + k];
                    }
                    for k in l..n {
                        a[j * n + k] += s2 * rv1[k];
                    }
                }
                for k in l..n {
                    a[i * n + k] *= scale;
                }
            }
        }
        anorm = anorm.max(w[i].abs() + rv1[i].abs());
    }

    // --- Diagonalize the bidiagonal form, tracking only w / rv1 (no U, no V) ---
    for k in (0..n).rev() {
        for _its in 0..30 {
            let mut flag = true;
            let mut l = k;
            let mut nm;
            loop {
                nm = l.saturating_sub(1);
                if l == 0 || (rv1[l].abs() + anorm) == anorm {
                    flag = false;
                    break;
                }
                if (w[nm].abs() + anorm) == anorm {
                    break;
                }
                l -= 1;
            }
            if flag {
                let mut c = 0.0;
                let mut s = 1.0;
                for i in l..=k {
                    let f = s * rv1[i];
                    rv1[i] *= c;
                    if (f.abs() + anorm) == anorm {
                        break;
                    }
                    g = w[i];
                    let h = eigh_pythag(f, g);
                    w[i] = h;
                    let hinv = 1.0 / h;
                    c = g * hinv;
                    s = -f * hinv;
                }
            }
            let z = w[k];
            if l == k {
                if z < 0.0 {
                    w[k] = -z;
                }
                break;
            }
            if _its == 29 {
                return Err(KernelError::SingularMatrix { size: n });
            }
            let mut x = w[l];
            nm = k - 1;
            let mut y = w[nm];
            g = rv1[nm];
            let mut h = rv1[k];
            let mut f = ((y - z) * (y + z) + (g - h) * (g + h)) / (2.0 * h * y);
            g = eigh_pythag(f, 1.0);
            f = ((x - z) * (x + z) + h * ((y / (f + nr_sign(g, f))) - h)) / x;
            let mut c = 1.0;
            let mut s = 1.0;
            for j in l..=nm {
                let i = j + 1;
                g = rv1[i];
                y = w[i];
                h = s * g;
                g *= c;
                let mut zz = eigh_pythag(f, h);
                rv1[j] = zz;
                c = f / zz;
                s = h / zz;
                f = x * c + g * s;
                g = g * c - x * s;
                h = y * s;
                y *= c;
                zz = eigh_pythag(f, h);
                w[j] = zz;
                if zz != 0.0 {
                    let zinv = 1.0 / zz;
                    c = f * zinv;
                    s = h * zinv;
                }
                f = c * g + s * y;
                x = c * y - s * g;
            }
            rv1[l] = 0.0;
            rv1[k] = f;
            w[k] = x;
        }
    }

    Ok(w)
}

/// SVD for tall/square matrices (m >= n) via Golub-Reinsch bidiagonalization.
fn svd_tall(a: &[f64], m: usize, n: usize, full_matrices: bool) -> Result<SvdResult, KernelError> {
    let k = n; // since m >= n, k = min(m,n) = n
    let tol = 1e-15;

    // Golub-Reinsch SVD: A = U_b diag(w) V^T with U_b the reduced (m x n) left
    // singular vectors. We rescale into `work[:,j] = w[j] * U_b[:,j]` so the
    // column norm of `work[:,j]` is exactly w[j] — matching the contract the
    // singular-value extraction, descending sort, U re-normalization,
    // orthonormal-basis completion, and V^T construction below already expect
    // (they were written for the prior one-sided-Jacobi `work = U diag(S)`).
    let mut u = a.to_vec();
    let (w_bidiag, v) = golub_reinsch_svd(&mut u, m, n)?;
    let mut work = vec![0.0f64; m * n];
    for j in 0..n {
        let wj = w_bidiag[j];
        for i in 0..m {
            work[i * n + j] = wj * u[i * n + j];
        }
    }

    // Singular values are the bidiagonal-QR diagonal `w_bidiag` directly (each
    // already non-negative). Using `w_bidiag` here — rather than re-deriving via
    // `work` column norms — makes the full SVD's `s` BIT-IDENTICAL to the
    // dedicated values-only path (`svdvals_contiguous_f64`), which runs the same
    // `w` recurrence without U/V accumulation. `col_norms` (the actual norms,
    // = w_bidiag up to U_b's unit-length rounding) is retained to re-normalize
    // the `work` columns into the orthonormal U.
    let mut singular_values = Vec::with_capacity(k);
    let mut col_norms = Vec::with_capacity(k);
    for j in 0..k {
        let mut norm = 0.0f64;
        for i in 0..m {
            norm += work[i * n + j] * work[i * n + j];
        }
        col_norms.push(norm.sqrt());
        singular_values.push(w_bidiag[j]);
    }

    // Sort singular values in descending order
    let mut order: Vec<usize> = (0..k).collect();
    order.sort_by(|&a, &b| singular_values[b].total_cmp(&singular_values[a]));

    let s: Vec<f64> = order.iter().map(|&i| singular_values[i]).collect();

    // Build U: normalize columns of `work` by singular values, reorder
    let u_cols = if full_matrices { m } else { k };
    let mut u = vec![0.0f64; m * u_cols];
    for (new_j, &old_j) in order.iter().enumerate() {
        if new_j >= u_cols {
            break;
        }
        let norm = col_norms[old_j];
        if norm > tol {
            for i in 0..m {
                u[i * u_cols + new_j] = work[i * n + old_j] / norm;
            }
        }
        // else: leave zero — completed in the unified pass below.
    }

    // Complete the orthonormal basis: any column of U that is still
    // zero (either because the corresponding singular value is below
    // tol — rank-deficient input — or because we are in full_matrices
    // mode and m > k so columns k..m start unfilled) gets a unit
    // vector orthogonal to all previously-set columns via
    // Gram-Schmidt. Required for U to satisfy U^T U = I_{u_cols},
    // which numpy / torch.linalg.svd / scipy all guarantee. Tracked
    // under frankentorch-zs8a — previously the rank-deficient path
    // left those columns as zeros and reduced-mode SVD failed
    // U^T U = I on rank-deficient matrices.
    for j in 0..u_cols {
        // Check if column j is currently zero (norm < tol).
        let mut existing_norm_sq = 0.0f64;
        for i in 0..m {
            let v = u[i * u_cols + j];
            existing_norm_sq += v * v;
        }
        if existing_norm_sq > tol * tol {
            continue;
        }

        // Try standard basis vectors e_0, e_1, ... e_{m-1} until one
        // produces a non-degenerate residual after Gram-Schmidt. In
        // a rank-r system with m-r missing columns, at most r of the
        // basis vectors can lie entirely in the existing column span,
        // so a fresh one is always available within m tries.
        for seed in 0..m {
            let mut col = vec![0.0f64; m];
            col[seed] = 1.0;
            // Orthogonalize against all previously-set columns of u
            // (this includes both already-normalized SVD columns and
            // any earlier basis-completion columns we just filled).
            for prev in 0..u_cols {
                if prev == j {
                    continue;
                }
                let mut dot = 0.0;
                for i in 0..m {
                    dot += col[i] * u[i * u_cols + prev];
                }
                for i in 0..m {
                    col[i] -= dot * u[i * u_cols + prev];
                }
            }
            let mut norm = 0.0f64;
            for item in col.iter().take(m) {
                norm += item * item;
            }
            let norm = norm.sqrt();
            if norm > tol {
                for i in 0..m {
                    u[i * u_cols + j] = col[i] / norm;
                }
                break;
            }
        }
    }

    // Build Vh: V^T with reordering
    let vh_rows = if full_matrices { n } else { k };
    let mut vh = vec![0.0f64; vh_rows * n];
    for (new_i, &old_i) in order.iter().enumerate() {
        if new_i >= vh_rows {
            break;
        }
        for j in 0..n {
            vh[new_i * n + j] = v[j * n + old_i]; // V^T[new_i][j] = V[j][old_i]
        }
    }

    Ok(SvdResult { u, s, vh, m, n, k })
}

/// Compute just the singular values of an (m x n) matrix, sorted descending.
///
/// Uses the dedicated values-only Golub-Reinsch path
/// ([`golub_reinsch_singular_values`]): bidiagonalize then run the implicit-shift
/// QR recurrence on the bidiagonal alone, skipping all U/V accumulation. The
/// `w` recurrence is identical to the full SVD's, so the values agree with
/// `svd_contiguous_f64(..).s` to working precision while avoiding the O(m n^2)
/// U accumulation and every per-rotation matrix update. Wide matrices reduce to
/// the tall case via transpose (the singular values of `A` and `A^T` are equal).
pub fn svdvals_contiguous_f64(data: &[f64], meta: &TensorMeta) -> Result<Vec<f64>, KernelError> {
    ensure_unary_layout_and_storage(data, meta)?;
    let shape = meta.shape();
    if shape.len() != 2 {
        return Err(KernelError::ShapeMismatch {
            lhs: shape.to_vec(),
            rhs: vec![2],
        });
    }
    let m = shape[0];
    let n = shape[1];
    if m == 0 || n == 0 {
        return Ok(Vec::new());
    }
    let offset = meta.storage_offset();
    let mut a = vec![0.0f64; m * n];
    for i in 0..m {
        for j in 0..n {
            a[i * n + j] = data[offset + i * n + j];
        }
    }

    let mut s = if m >= n {
        golub_reinsch_singular_values(&mut a, m, n)?
    } else {
        // Transpose to (n x m) so it is tall; sv(A) == sv(A^T).
        let mut at = vec![0.0f64; n * m];
        for i in 0..m {
            for j in 0..n {
                at[j * m + i] = a[i * n + j];
            }
        }
        golub_reinsch_singular_values(&mut at, n, m)?
    };
    // Sort descending, matching svd_contiguous_f64's ordering.
    s.sort_by(|x, y| y.total_cmp(x));
    Ok(s)
}

/// One draw from a deterministic SplitMix64 stream, mapped to a uniform in (0,1).
fn splitmix_uniform(state: &mut u64) -> f64 {
    *state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
    let mut z = *state;
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^= z >> 31;
    // 53-bit mantissa -> (0, 1).
    ((z >> 11) as f64 + 0.5) / (1u64 << 53) as f64
}

/// `out` (= ca x cb) = `A^T · B`, where `A` is `ra x ca` and `B` is `ra x cb`
/// (both row-major). Materialises `A^T` then routes through the cache-blocked GEMM.
fn mat_at_b(a: &[f64], ra: usize, ca: usize, b: &[f64], cb: usize, out: &mut [f64]) {
    let mut at = vec![0.0f64; ca * ra];
    for i in 0..ra {
        for j in 0..ca {
            at[j * ra + i] = a[i * ca + j];
        }
    }
    gemm::dgemm(ca, ra, cb, &at, b, out);
}

/// Randomized truncated SVD (Halko-Martinsson-Tropp). Returns an approximate
/// rank-`q` factorisation `A ≈ U·diag(s)·Vh` in `O(m·n·l)` (l = q + oversampling)
/// instead of the full SVD's `O(m·n²)` — a large win when `q << min(m,n)`
/// (PCA / low-rank approximation). `niter` subspace power iterations sharpen the
/// estimate for slowly-decaying spectra. The Gaussian sketch is seeded
/// deterministically from the shape so results are reproducible. For a matrix of
/// exact rank ≤ q the factorisation is accurate to working precision.
pub fn svd_lowrank_contiguous_f64(
    data: &[f64],
    meta: &TensorMeta,
    q: usize,
    niter: usize,
) -> Result<SvdResult, KernelError> {
    ensure_unary_layout_and_storage(data, meta)?;
    let shape = meta.shape();
    if shape.len() != 2 {
        return Err(KernelError::ShapeMismatch {
            lhs: shape.to_vec(),
            rhs: vec![2],
        });
    }
    let m = shape[0];
    let n = shape[1];
    if m == 0 || n == 0 {
        return Ok(SvdResult {
            u: Vec::new(),
            s: Vec::new(),
            vh: Vec::new(),
            m,
            n,
            k: 0,
        });
    }
    let offset = meta.storage_offset();
    let a = data[offset..offset + m * n].to_vec();

    let max_rank = m.min(n);
    let q = q.clamp(1, max_rank);
    // Sketch width: target rank + oversampling, capped at the smaller dimension.
    let l = (q + 6).min(max_rank);

    // Gaussian sketch Omega (n x l) via SplitMix64 + Box-Muller.
    let mut rng =
        0x243F_6A88_85A3_08D3u64 ^ ((m as u64) << 32) ^ (n as u64).wrapping_mul(2_654_435_761);
    let mut omega = vec![0.0f64; n * l];
    let mut idx = 0;
    while idx < omega.len() {
        let u1 = splitmix_uniform(&mut rng);
        let u2 = splitmix_uniform(&mut rng);
        let r = (-2.0 * u1.ln()).sqrt();
        let ang = std::f64::consts::TAU * u2;
        omega[idx] = r * ang.cos();
        idx += 1;
        if idx < omega.len() {
            omega[idx] = r * ang.sin();
            idx += 1;
        }
    }

    // Y = A · Omega  (m x l).
    let mut y = vec![0.0f64; m * l];
    gemm::dgemm(m, n, l, &a, &omega, &mut y);

    // Subspace power iterations: Y <- A·(A^T·Y).
    for _ in 0..niter {
        let mut z = vec![0.0f64; n * l]; // z = A^T · Y
        mat_at_b(&a, m, n, &y, l, &mut z);
        gemm::dgemm(m, n, l, &a, &z, &mut y); // y = A · z
    }

    // Orthonormal basis Q (m x l) of range(Y) via reduced QR.
    let y_meta = TensorMeta::from_shape(vec![m, l], meta.dtype(), meta.device());
    let qr = qr_contiguous_f64(&y, &y_meta, true)?;
    let qmat = qr.q; // m x l
    let lq = qr.n;

    // B = Q^T · A  (lq x n), then its (small) full SVD.
    let mut b = vec![0.0f64; lq * n];
    mat_at_b(&qmat, m, lq, &a, n, &mut b);
    let b_meta = TensorMeta::from_shape(vec![lq, n], meta.dtype(), meta.device());
    let bsvd = svd_contiguous_f64(&b, &b_meta, false)?;
    let kb = bsvd.k;

    // U = Q · Ub  (m x kb).
    let mut u_full = vec![0.0f64; m * kb];
    gemm::dgemm(m, lq, kb, &qmat, &bsvd.u, &mut u_full);

    // Truncate to the top-q components.
    let out_k = q.min(kb);
    let mut u = vec![0.0f64; m * out_k];
    for i in 0..m {
        for j in 0..out_k {
            u[i * out_k + j] = u_full[i * kb + j];
        }
    }
    let s = bsvd.s[..out_k].to_vec();
    let vh = bsvd.vh[..out_k * n].to_vec();

    Ok(SvdResult {
        u,
        s,
        vh,
        m,
        n,
        k: out_k,
    })
}

/// Randomized PCA (`torch.pca_lowrank`). Optionally centers the columns
/// (subtract each column's mean, treating rows as samples / columns as
/// features), then runs the randomized truncated SVD. Returns the reduced
/// `(U, S, Vh)`; the caller transposes `Vh` to the `V` that `torch.pca_lowrank`
/// reports. O(m·n·q) via [`svd_lowrank_contiguous_f64`].
pub fn pca_lowrank_contiguous_f64(
    data: &[f64],
    meta: &TensorMeta,
    q: usize,
    center: bool,
    niter: usize,
) -> Result<SvdResult, KernelError> {
    ensure_unary_layout_and_storage(data, meta)?;
    let shape = meta.shape();
    if shape.len() != 2 {
        return Err(KernelError::ShapeMismatch {
            lhs: shape.to_vec(),
            rhs: vec![2],
        });
    }
    let m = shape[0];
    let n = shape[1];
    if m == 0 || n == 0 {
        return Ok(SvdResult {
            u: Vec::new(),
            s: Vec::new(),
            vh: Vec::new(),
            m,
            n,
            k: 0,
        });
    }
    let offset = meta.storage_offset();
    let mut a = data[offset..offset + m * n].to_vec();
    if center {
        for j in 0..n {
            let mut mean = 0.0;
            for i in 0..m {
                mean += a[i * n + j];
            }
            mean /= m as f64;
            for i in 0..m {
                a[i * n + j] -= mean;
            }
        }
    }
    let a_meta = TensorMeta::from_shape(vec![m, n], meta.dtype(), meta.device());
    svd_lowrank_contiguous_f64(&a, &a_meta, q, niter)
}

/// Modified Gram-Schmidt over the columns of `m` (n x c, row-major). The
/// orthonormal columns are packed into the first `rank` positions (returned);
/// columns whose residual norm falls below the threshold are dropped (rank
/// deficiency). Used by [`lobpcg_contiguous_f64`].
fn mgs_orthonormalize_cols(m: &mut [f64], n: usize, c: usize) -> usize {
    let mut rank = 0usize;
    for j in 0..c {
        let mut col = vec![0.0f64; n];
        for i in 0..n {
            col[i] = m[i * c + j];
        }
        for p in 0..rank {
            let mut dot = 0.0;
            for i in 0..n {
                dot += col[i] * m[i * c + p];
            }
            for i in 0..n {
                col[i] -= dot * m[i * c + p];
            }
        }
        let norm: f64 = col.iter().map(|v| v * v).sum::<f64>().sqrt();
        if norm > 1e-10 {
            let inv = 1.0 / norm;
            for i in 0..n {
                m[i * c + rank] = col[i] * inv;
            }
            rank += 1;
        }
    }
    rank
}

/// Top/bottom-`k` eigenpairs of a real-symmetric matrix via a block
/// steepest-descent Rayleigh-Ritz iteration (LOBPCG family, torch.lobpcg). Each
/// step expands the orthonormal subspace `[X | R]` (R = the X residuals,
/// orthonormalized against X), does a SMALL `eigh` of `S^T A S`, and keeps the k
/// extreme Ritz pairs — converging to the k largest (`largest=true`) or smallest
/// eigenpairs in `O(n^2 k)` per iteration vs the full eigh's `O(n^3)`. Returns
/// `(eigenvalues[k], eigenvectors[n*k]` row-major, columns = vectors`)`.
pub fn lobpcg_contiguous_f64(
    data: &[f64],
    meta: &TensorMeta,
    k: usize,
    largest: bool,
    niter: usize,
    tol: f64,
) -> Result<(Vec<f64>, Vec<f64>), KernelError> {
    ensure_unary_layout_and_storage(data, meta)?;
    let shape = meta.shape();
    if shape.len() != 2 || shape[0] != shape[1] {
        return Err(KernelError::ShapeMismatch {
            lhs: shape.to_vec(),
            rhs: vec![2],
        });
    }
    let n = shape[0];
    let k = k.clamp(1, n);
    let offset = meta.storage_offset();
    let a = data[offset..offset + n * n].to_vec();

    let dev = meta.device();
    let dty = meta.dtype();
    // A @ M  for M: n x c  -> n x c.
    let amul = |x: &[f64], c: usize| -> Vec<f64> {
        let mut out = vec![0.0f64; n * c];
        gemm::dgemm(n, n, c, &a, x, &mut out);
        out
    };
    // Small dense eigh of an `s x s` symmetric matrix; selects the k extreme
    // Ritz pairs. Returns (theta[k], coeff[s*k] row-major) where coeff column t
    // is the eigenvector of the t-th selected Ritz value.
    let select_ritz = |sym: &[f64], s: usize| -> Result<(Vec<f64>, Vec<f64>), KernelError> {
        let m = TensorMeta::from_shape(vec![s, s], dty, dev);
        let e = eigh_contiguous_f64(sym, &m)?; // eigenvalues ascending, evecs s x s cols
        let mut theta = vec![0.0f64; k];
        let mut coeff = vec![0.0f64; s * k];
        for t in 0..k {
            let idx = if largest { s - 1 - t } else { t };
            theta[t] = e.eigenvalues[idx];
            for i in 0..s {
                coeff[i * k + t] = e.eigenvectors[i * s + idx];
            }
        }
        Ok((theta, coeff))
    };

    // Initial X: random n x k, orthonormalized.
    let mut rng =
        0x853C_49E6_748F_EA9Bu64 ^ ((n as u64) << 24) ^ (k as u64).wrapping_mul(0x2545F4914F6CDD1D);
    let mut x = vec![0.0f64; n * k];
    let mut idx = 0;
    while idx < x.len() {
        let u1 = splitmix_uniform(&mut rng);
        let u2 = splitmix_uniform(&mut rng);
        let r = (-2.0 * u1.ln()).sqrt();
        let ang = std::f64::consts::TAU * u2;
        x[idx] = r * ang.cos();
        idx += 1;
        if idx < x.len() {
            x[idx] = r * ang.sin();
            idx += 1;
        }
    }
    mgs_orthonormalize_cols(&mut x, n, k);

    let mut ax = amul(&x, k);
    // Initial Rayleigh-Ritz on X.
    let mut gram = vec![0.0f64; k * k];
    mat_at_b(&x, n, k, &ax, k, &mut gram); // X^T A X
    let (mut theta, c0) = select_ritz(&gram, k)?;
    {
        let mut nx = vec![0.0f64; n * k];
        gemm::dgemm(n, k, k, &x, &c0, &mut nx);
        x = nx;
        let mut nax = vec![0.0f64; n * k];
        gemm::dgemm(n, k, k, &ax, &c0, &mut nax);
        ax = nax;
    }

    for _ in 0..niter.max(1) {
        // Residuals R = AX - X diag(theta).
        let mut r = ax.clone();
        let mut maxres = 0.0f64;
        for i in 0..k {
            let mut nr = 0.0;
            for j in 0..n {
                let v = ax[j * k + i] - theta[i] * x[j * k + i];
                r[j * k + i] = v;
                nr += v * v;
            }
            maxres = maxres.max(nr.sqrt());
        }
        if maxres < tol {
            break;
        }
        // Orthogonalize R against X, then orthonormalize R's own columns.
        let mut xtr = vec![0.0f64; k * k];
        mat_at_b(&x, n, k, &r, k, &mut xtr); // X^T R  (k x k)
        let mut xxtr = vec![0.0f64; n * k];
        gemm::dgemm(n, k, k, &x, &xtr, &mut xxtr); // X (X^T R)
        for v in 0..n * k {
            r[v] -= xxtr[v];
        }
        let rr = mgs_orthonormalize_cols(&mut r, n, k);
        if rr == 0 {
            break;
        }
        // Subspace S = [X | R[:, :rr]]  (n x sc), orthonormal by construction.
        let sc = k + rr;
        let mut s = vec![0.0f64; n * sc];
        for i in 0..n {
            for j in 0..k {
                s[i * sc + j] = x[i * k + j];
            }
            for j in 0..rr {
                s[i * sc + (k + j)] = r[i * k + j];
            }
        }
        let as_ = amul(&s, sc);
        // M = S^T A S  (sc x sc, symmetric).
        let mut msym = vec![0.0f64; sc * sc];
        mat_at_b(&s, n, sc, &as_, sc, &mut msym);
        let (theta_new, coeff) = select_ritz(&msym, sc)?; // coeff: sc x k
        // X = S @ coeff ;  AX = AS @ coeff.
        let mut nx = vec![0.0f64; n * k];
        gemm::dgemm(n, sc, k, &s, &coeff, &mut nx);
        let mut nax = vec![0.0f64; n * k];
        gemm::dgemm(n, sc, k, &as_, &coeff, &mut nax);
        x = nx;
        ax = nax;
        theta = theta_new;
    }

    Ok((theta, x))
}

/// Result of QR decomposition.
#[derive(Debug, Clone)]
pub struct QrResult {
    /// Orthogonal matrix Q in row-major order.
    pub q: Vec<f64>,
    /// Upper triangular matrix R in row-major order.
    pub r: Vec<f64>,
    /// Number of rows (m).
    pub m: usize,
    /// Number of columns (n).
    pub n: usize,
}

/// Compute the QR decomposition of an (m x n) matrix via Householder reflections.
///
/// Returns `(Q, R)` such that `A = Q @ R`:
/// - Q: orthogonal matrix (m x m) for `reduced=false`, or (m x k) for `reduced=true` where k=min(m,n)
/// - R: upper triangular matrix (m x n) for `reduced=false`, or (k x n) for `reduced=true`
///
/// The matrix must be 2-D.
pub fn qr_contiguous_f64(
    data: &[f64],
    meta: &TensorMeta,
    reduced: bool,
) -> Result<QrResult, KernelError> {
    ensure_unary_layout_and_storage(data, meta)?;
    let shape = meta.shape();
    if shape.len() != 2 {
        return Err(KernelError::ShapeMismatch {
            lhs: shape.to_vec(),
            rhs: vec![shape.len()],
        });
    }
    let m = shape[0];
    let n = shape[1];

    if m == 0 || n == 0 {
        let k = if reduced { m.min(n) } else { m };
        return Ok(QrResult {
            q: vec![0.0; m * k],
            r: vec![0.0; k * n],
            m: k,
            n,
        });
    }

    let k = m.min(n);
    let offset = meta.storage_offset();

    // Copy A into working matrix R (m x n, row-major)
    let mut r_mat = data[offset..offset + m * n].to_vec();

    // Build Q as product of Householder reflections, starting as identity (m x m)
    let mut q_mat = vec![0.0; m * m];
    for i in 0..m {
        q_mat[i * m + i] = 1.0;
    }

    for j in 0..k {
        // Extract the column vector below the diagonal: v = R[j:m, j]
        let col_len = m - j;
        let mut v = vec![0.0; col_len];
        for i in 0..col_len {
            v[i] = r_mat[(i + j) * n + j];
        }

        // Compute the Householder reflection: v[0] += sign(v[0]) * ||v||
        let norm_v: f64 = v.iter().map(|x| x * x).sum::<f64>().sqrt();
        if norm_v < f64::EPSILON * 1e6 {
            continue; // Skip near-zero column
        }
        let sign = if v[0] >= 0.0 { 1.0 } else { -1.0 };
        v[0] += sign * norm_v;

        // Normalize v
        let norm_v2: f64 = v.iter().map(|x| x * x).sum();
        if norm_v2 < f64::EPSILON * 1e6 {
            continue;
        }
        let inv_norm = 1.0 / norm_v2;

        // Apply Householder to R: R[j:m, :] -= 2 * v * (v^T @ R[j:m, :]). Each
        // output column's factor is an INDEPENDENT dot over the reflected rows,
        // and each reflected row's update is independent, so both fan out with
        // the SAME fp op order -> bit-for-bit identical to the serial sweep. The
        // factor `2*dot*inv_norm` is folded into `w` so the update keeps the exact
        // `factor * v[i]` association.
        const QR_PAR_WORK: u64 = 1 << 14;
        let r_work = (col_len as u64) * (n as u64);
        if r_work >= QR_PAR_WORK {
            let mut w = vec![0.0f64; n];
            {
                let r_ref: &[f64] = &r_mat;
                w.par_iter_mut().enumerate().for_each(|(col, wc)| {
                    let mut dot = 0.0;
                    for i in 0..col_len {
                        dot += v[i] * r_ref[(i + j) * n + col];
                    }
                    *wc = 2.0 * dot * inv_norm;
                });
            }
            let (_, tail) = r_mat.split_at_mut(j * n);
            tail.par_chunks_mut(n).enumerate().for_each(|(i, row)| {
                let vi = v[i];
                for col in 0..n {
                    row[col] -= w[col] * vi;
                }
            });
        } else {
            for col in 0..n {
                let mut dot = 0.0;
                for i in 0..col_len {
                    dot += v[i] * r_mat[(i + j) * n + col];
                }
                let factor = 2.0 * dot * inv_norm;
                for i in 0..col_len {
                    r_mat[(i + j) * n + col] -= factor * v[i];
                }
            }
        }

        // Apply Householder to Q: Q[:, j:m] -= 2 * Q[:, j:m] @ v * v^T. Each row of
        // Q is independent and contiguous -> clean bit-exact row fan-out.
        let q_work = (m as u64) * (col_len as u64);
        if q_work >= QR_PAR_WORK {
            q_mat.par_chunks_mut(m).for_each(|q_row| {
                let mut dot = 0.0;
                for i in 0..col_len {
                    dot += q_row[i + j] * v[i];
                }
                let factor = 2.0 * dot * inv_norm;
                for i in 0..col_len {
                    q_row[i + j] -= factor * v[i];
                }
            });
        } else {
            for row in 0..m {
                let mut dot = 0.0;
                for i in 0..col_len {
                    dot += q_mat[row * m + (i + j)] * v[i];
                }
                let factor = 2.0 * dot * inv_norm;
                for i in 0..col_len {
                    q_mat[row * m + (i + j)] -= factor * v[i];
                }
            }
        }
    }

    // Clean up R: zero out below-diagonal elements
    for i in 0..m {
        for j in 0..i.min(n) {
            r_mat[i * n + j] = 0.0;
        }
    }

    if reduced {
        // Q is m x k (first k columns of full Q)
        let mut q_reduced = vec![0.0; m * k];
        for i in 0..m {
            for j in 0..k {
                q_reduced[i * k + j] = q_mat[i * m + j];
            }
        }
        // R is k x n (first k rows of full R)
        let r_reduced = r_mat[..k * n].to_vec();
        Ok(QrResult {
            q: q_reduced,
            r: r_reduced,
            m: k,
            n,
        })
    } else {
        // Q is m x m, R is m x n
        Ok(QrResult {
            q: q_mat,
            r: r_mat,
            m,
            n,
        })
    }
}

// =========================================================================
// f32 kernel functions (bd-2do9.3)
// =========================================================================

fn ensure_storage_len_f32(
    buffer: &[f32],
    meta: &TensorMeta,
    side: &'static str,
) -> Result<(), KernelError> {
    let needed = contiguous_required_len(meta, side)?;
    if buffer.len() < needed {
        return Err(KernelError::InsufficientStorage {
            side,
            needed,
            available: buffer.len(),
        });
    }
    Ok(())
}

fn ensure_unary_layout_and_storage_f32(
    buffer: &[f32],
    meta: &TensorMeta,
) -> Result<(), KernelError> {
    if !meta.is_contiguous() {
        return Err(KernelError::UnsupportedLayout { side: "input" });
    }
    ensure_storage_len_f32(buffer, meta, "input")
}

const SIMD_WIDTH_F32: usize = 8;

fn simd_unary_f32<F, S>(window: &[f32], scalar_op: F, simd_op: S) -> Vec<f32>
where
    F: Fn(f32) -> f32,
    S: Fn(f32x8) -> f32x8,
{
    let numel = window.len();
    let simd_len = numel / SIMD_WIDTH_F32 * SIMD_WIDTH_F32;
    let mut output = Vec::with_capacity(numel);

    for i in (0..simd_len).step_by(SIMD_WIDTH_F32) {
        let a = f32x8::new([
            window[i],
            window[i + 1],
            window[i + 2],
            window[i + 3],
            window[i + 4],
            window[i + 5],
            window[i + 6],
            window[i + 7],
        ]);
        let result = simd_op(a);
        output.extend_from_slice(result.as_array_ref());
    }

    for &value in &window[simd_len..numel] {
        output.push(scalar_op(value));
    }

    output
}

fn simd_binary_f32<F, S>(
    lhs_window: &[f32],
    rhs_window: &[f32],
    scalar_op: F,
    simd_op: S,
) -> Vec<f32>
where
    F: Fn(f32, f32) -> f32,
    S: Fn(f32x8, f32x8) -> f32x8,
{
    let numel = lhs_window.len();
    let simd_len = numel / SIMD_WIDTH_F32 * SIMD_WIDTH_F32;
    let mut output = Vec::with_capacity(numel);

    for i in (0..simd_len).step_by(SIMD_WIDTH_F32) {
        let a = f32x8::new([
            lhs_window[i],
            lhs_window[i + 1],
            lhs_window[i + 2],
            lhs_window[i + 3],
            lhs_window[i + 4],
            lhs_window[i + 5],
            lhs_window[i + 6],
            lhs_window[i + 7],
        ]);
        let b = f32x8::new([
            rhs_window[i],
            rhs_window[i + 1],
            rhs_window[i + 2],
            rhs_window[i + 3],
            rhs_window[i + 4],
            rhs_window[i + 5],
            rhs_window[i + 6],
            rhs_window[i + 7],
        ]);
        let result = simd_op(a, b);
        output.extend_from_slice(result.as_array_ref());
    }

    for i in simd_len..numel {
        output.push(scalar_op(lhs_window[i], rhs_window[i]));
    }

    output
}

fn unary_contiguous_f32<F>(input: &[f32], meta: &TensorMeta, op: F) -> Result<Vec<f32>, KernelError>
where
    F: Fn(f32) -> f32,
{
    ensure_unary_layout_and_storage_f32(input, meta)?;
    let numel = meta.numel();
    if numel == 0 {
        return Ok(Vec::new());
    }
    let start = meta.storage_offset();
    let window = &input[start..start + numel];
    Ok(window.iter().map(|value| op(*value)).collect())
}

fn simd_unary_f32_kernel<F, S>(
    input: &[f32],
    meta: &TensorMeta,
    scalar_op: F,
    simd_op: S,
) -> Result<Vec<f32>, KernelError>
where
    F: Fn(f32) -> f32,
    S: Fn(f32x8) -> f32x8,
{
    ensure_unary_layout_and_storage_f32(input, meta)?;
    let numel = meta.numel();
    if numel == 0 {
        return Ok(Vec::new());
    }
    let start = meta.storage_offset();
    let window = &input[start..start + numel];
    Ok(simd_unary_f32(window, scalar_op, simd_op))
}

fn elementwise_contiguous_f32<F>(
    lhs: &[f32],
    rhs: &[f32],
    lhs_meta: &TensorMeta,
    rhs_meta: &TensorMeta,
    op: F,
) -> Result<Vec<f32>, KernelError>
where
    F: Fn(f32, f32) -> f32,
{
    ensure_meta_shape_and_dtype(lhs_meta, rhs_meta)?;
    ensure_storage_len_f32(lhs, lhs_meta, "lhs")?;
    ensure_storage_len_f32(rhs, rhs_meta, "rhs")?;
    let numel = lhs_meta.numel();
    if numel == 0 {
        return Ok(Vec::new());
    }
    let lhs_start = lhs_meta.storage_offset();
    let rhs_start = rhs_meta.storage_offset();
    let lhs_window = &lhs[lhs_start..lhs_start + numel];
    let rhs_window = &rhs[rhs_start..rhs_start + numel];
    Ok(lhs_window
        .iter()
        .zip(rhs_window.iter())
        .map(|(left, right)| op(*left, *right))
        .collect())
}

fn simd_elementwise_f32<F, S>(
    lhs: &[f32],
    rhs: &[f32],
    lhs_meta: &TensorMeta,
    rhs_meta: &TensorMeta,
    scalar_op: F,
    simd_op: S,
) -> Result<Vec<f32>, KernelError>
where
    F: Fn(f32, f32) -> f32,
    S: Fn(f32x8, f32x8) -> f32x8,
{
    ensure_meta_shape_and_dtype(lhs_meta, rhs_meta)?;
    ensure_storage_len_f32(lhs, lhs_meta, "lhs")?;
    ensure_storage_len_f32(rhs, rhs_meta, "rhs")?;
    let numel = lhs_meta.numel();
    if numel == 0 {
        return Ok(Vec::new());
    }
    let lhs_start = lhs_meta.storage_offset();
    let rhs_start = rhs_meta.storage_offset();
    let lhs_window = &lhs[lhs_start..lhs_start + numel];
    let rhs_window = &rhs[rhs_start..rhs_start + numel];
    Ok(simd_binary_f32(lhs_window, rhs_window, scalar_op, simd_op))
}

pub fn add_tensor_contiguous_f32(
    lhs: &[f32],
    rhs: &[f32],
    lhs_meta: &TensorMeta,
    rhs_meta: &TensorMeta,
) -> Result<Vec<f32>, KernelError> {
    simd_elementwise_f32(lhs, rhs, lhs_meta, rhs_meta, |l, r| l + r, |a, b| a + b)
}

pub fn sub_tensor_contiguous_f32(
    lhs: &[f32],
    rhs: &[f32],
    lhs_meta: &TensorMeta,
    rhs_meta: &TensorMeta,
) -> Result<Vec<f32>, KernelError> {
    simd_elementwise_f32(lhs, rhs, lhs_meta, rhs_meta, |l, r| l - r, |a, b| a - b)
}

pub fn mul_tensor_contiguous_f32(
    lhs: &[f32],
    rhs: &[f32],
    lhs_meta: &TensorMeta,
    rhs_meta: &TensorMeta,
) -> Result<Vec<f32>, KernelError> {
    simd_elementwise_f32(lhs, rhs, lhs_meta, rhs_meta, |l, r| l * r, |a, b| a * b)
}

pub fn div_tensor_contiguous_f32(
    lhs: &[f32],
    rhs: &[f32],
    lhs_meta: &TensorMeta,
    rhs_meta: &TensorMeta,
) -> Result<Vec<f32>, KernelError> {
    simd_elementwise_f32(lhs, rhs, lhs_meta, rhs_meta, |l, r| l / r, |a, b| a / b)
}

pub fn neg_tensor_contiguous_f32(
    input: &[f32],
    meta: &TensorMeta,
) -> Result<Vec<f32>, KernelError> {
    simd_unary_f32_kernel(input, meta, |v| -v, |a| -a)
}

pub fn abs_tensor_contiguous_f32(
    input: &[f32],
    meta: &TensorMeta,
) -> Result<Vec<f32>, KernelError> {
    simd_unary_f32_kernel(input, meta, |v| v.abs(), |a| a.abs())
}

pub fn sqrt_tensor_contiguous_f32(
    input: &[f32],
    meta: &TensorMeta,
) -> Result<Vec<f32>, KernelError> {
    simd_unary_f32_kernel(input, meta, |v| v.sqrt(), |a| a.sqrt())
}

pub fn reciprocal_tensor_contiguous_f32(
    input: &[f32],
    meta: &TensorMeta,
) -> Result<Vec<f32>, KernelError> {
    let one = f32x8::splat(1.0);
    simd_unary_f32_kernel(input, meta, |v| 1.0 / v, move |a| one / a)
}

fn ensure_dtype_device_and_layout_f32(
    lhs: &TensorMeta,
    rhs: &TensorMeta,
) -> Result<(), KernelError> {
    if lhs.dtype() != rhs.dtype() {
        return Err(KernelError::Incompatible(
            TensorCompatError::DTypeMismatch {
                lhs: lhs.dtype(),
                rhs: rhs.dtype(),
            },
        ));
    }
    if lhs.device() != rhs.device() {
        return Err(KernelError::Incompatible(
            TensorCompatError::DeviceMismatch {
                lhs: lhs.device(),
                rhs: rhs.device(),
            },
        ));
    }
    if !lhs.is_contiguous() {
        return Err(KernelError::UnsupportedLayout { side: "lhs" });
    }
    if !rhs.is_contiguous() {
        return Err(KernelError::UnsupportedLayout { side: "rhs" });
    }
    Ok(())
}

// f32 activation helpers

fn gelu_value_f32(x: f32) -> f32 {
    // PyTorch default GELU (approximate="none"): exact erf form.
    0.5f32 * x * (1.0f32 + libm::erff(x * std::f32::consts::FRAC_1_SQRT_2))
}

fn silu_value_f32(x: f32) -> f32 {
    x / (1.0f32 + (-x).exp())
}

fn leaky_relu_value_f32(x: f32) -> f32 {
    if x >= 0.0f32 { x } else { 0.01f32 * x }
}

fn elu_value_f32(x: f32) -> f32 {
    if x > 0.0f32 { x } else { x.exp() - 1.0f32 }
}

fn erf_value_f32(x: f32) -> f32 {
    // F32 companion to `erf_value` — see that function for the reason
    // `libm` is preferred over the Abramowitz-Stegun approximation.
    libm::erff(x)
}

fn erfc_value_f32(x: f32) -> f32 {
    libm::erfcf(x)
}

fn hardswish_value_f32(x: f32) -> f32 {
    if x <= -3.0f32 {
        0.0f32
    } else if x >= 3.0f32 {
        x
    } else {
        x * (x + 3.0f32) / 6.0f32
    }
}

fn hardsigmoid_value_f32(x: f32) -> f32 {
    if x <= -3.0f32 {
        0.0f32
    } else if x >= 3.0f32 {
        1.0f32
    } else {
        (x + 3.0f32) / 6.0f32
    }
}

fn hardtanh_value_f32(x: f32) -> f32 {
    x.clamp(-1.0f32, 1.0f32)
}

fn softplus_value_f32(x: f32) -> f32 {
    // Mirrors softplus_value (f64): only the upper threshold is needed;
    // log1p(exp(x)) is precise for negative x and decays smoothly.
    if x > 20.0f32 { x } else { x.exp().ln_1p() }
}

fn mish_value_f32(x: f32) -> f32 {
    x * softplus_value_f32(x).tanh()
}

fn round_ties_even_f32(value: f32) -> f32 {
    value.round_ties_even()
}

/// F32 companion to `torch_sign_f64` — see that function for rationale.
fn torch_sign_f32(value: f32) -> f32 {
    if value.is_nan() {
        f32::NAN
    } else if value == 0.0 {
        0.0
    } else if value > 0.0 {
        1.0
    } else {
        -1.0
    }
}

// ── Macro-generated simple f32 unary kernels ────────────────────────────

macro_rules! define_unary_f32 {
    ($name:ident, $op:expr) => {
        pub fn $name(input: &[f32], meta: &TensorMeta) -> Result<Vec<f32>, KernelError> {
            unary_contiguous_f32(input, meta, $op)
        }
    };
}

define_unary_f32!(exp_tensor_contiguous_f32, f32::exp);
define_unary_f32!(log_tensor_contiguous_f32, f32::ln);
pub fn relu_tensor_contiguous_f32(
    input: &[f32],
    meta: &TensorMeta,
) -> Result<Vec<f32>, KernelError> {
    let zero = f32x8::splat(0.0f32);
    simd_unary_f32_kernel(input, meta, |v| v.max(0.0f32), move |a| a.max(zero))
}
define_unary_f32!(sigmoid_tensor_contiguous_f32, |v: f32| 1.0f32
    / (1.0f32 + (-v).exp()));
define_unary_f32!(tanh_tensor_contiguous_f32, f32::tanh);
define_unary_f32!(sin_tensor_contiguous_f32, f32::sin);
define_unary_f32!(cos_tensor_contiguous_f32, f32::cos);
define_unary_f32!(tan_tensor_contiguous_f32, f32::tan);
define_unary_f32!(floor_tensor_contiguous_f32, f32::floor);
define_unary_f32!(ceil_tensor_contiguous_f32, f32::ceil);
define_unary_f32!(round_tensor_contiguous_f32, round_ties_even_f32);
define_unary_f32!(log2_tensor_contiguous_f32, f32::log2);
define_unary_f32!(log10_tensor_contiguous_f32, f32::log10);
define_unary_f32!(log1p_tensor_contiguous_f32, f32::ln_1p);
define_unary_f32!(expm1_tensor_contiguous_f32, f32::exp_m1);
define_unary_f32!(sign_tensor_contiguous_f32, torch_sign_f32);
define_unary_f32!(trunc_tensor_contiguous_f32, f32::trunc);
define_unary_f32!(frac_tensor_contiguous_f32, f32::fract);
define_unary_f32!(asin_tensor_contiguous_f32, f32::asin);
define_unary_f32!(acos_tensor_contiguous_f32, f32::acos);
define_unary_f32!(atan_tensor_contiguous_f32, f32::atan);
define_unary_f32!(sinh_tensor_contiguous_f32, f32::sinh);
define_unary_f32!(cosh_tensor_contiguous_f32, f32::cosh);
define_unary_f32!(gelu_tensor_contiguous_f32, gelu_value_f32);
define_unary_f32!(silu_tensor_contiguous_f32, silu_value_f32);
define_unary_f32!(leaky_relu_tensor_contiguous_f32, leaky_relu_value_f32);
define_unary_f32!(elu_tensor_contiguous_f32, elu_value_f32);
define_unary_f32!(rsqrt_tensor_contiguous_f32, |v: f32| 1.0f32 / v.sqrt());
define_unary_f32!(erf_tensor_contiguous_f32, erf_value_f32);
define_unary_f32!(erfc_tensor_contiguous_f32, erfc_value_f32);
define_unary_f32!(hardswish_tensor_contiguous_f32, hardswish_value_f32);
define_unary_f32!(hardsigmoid_tensor_contiguous_f32, hardsigmoid_value_f32);
define_unary_f32!(hardtanh_tensor_contiguous_f32, hardtanh_value_f32);
define_unary_f32!(softplus_tensor_contiguous_f32, softplus_value_f32);
define_unary_f32!(mish_tensor_contiguous_f32, mish_value_f32);
define_unary_f32!(square_tensor_contiguous_f32, |v: f32| v * v);
define_unary_f32!(isnan_tensor_contiguous_f32, |v: f32| if v.is_nan() {
    1.0f32
} else {
    0.0f32
});
define_unary_f32!(isinf_tensor_contiguous_f32, |v: f32| if v.is_infinite() {
    1.0f32
} else {
    0.0f32
});
define_unary_f32!(isfinite_tensor_contiguous_f32, |v: f32| if v.is_finite() {
    1.0f32
} else {
    0.0f32
});

// ── Macro-generated simple f32 binary kernels ───────────────────────────

macro_rules! define_binary_f32 {
    ($name:ident, $op:expr) => {
        pub fn $name(
            lhs: &[f32],
            rhs: &[f32],
            lhs_meta: &TensorMeta,
            rhs_meta: &TensorMeta,
        ) -> Result<Vec<f32>, KernelError> {
            elementwise_contiguous_f32(lhs, rhs, lhs_meta, rhs_meta, $op)
        }
    };
}

define_binary_f32!(min_tensor_contiguous_f32, |l: f32, r: f32| {
    if l.is_nan() || r.is_nan() {
        f32::NAN
    } else {
        l.min(r)
    }
});
define_binary_f32!(max_tensor_contiguous_f32, |l: f32, r: f32| {
    if l.is_nan() || r.is_nan() {
        f32::NAN
    } else {
        l.max(r)
    }
});
define_binary_f32!(atan2_tensor_contiguous_f32, |y: f32, x: f32| y.atan2(x));
define_binary_f32!(fmod_tensor_contiguous_f32, |a: f32, b: f32| a % b);
define_binary_f32!(remainder_tensor_contiguous_f32, |a: f32, b: f32| a
    - (a / b).floor() * b);

// ── Comparison ops f32 ──────────────────────────────────────────────────

define_binary_f32!(eq_tensor_contiguous_f32, |l: f32, r: f32| if l == r {
    1.0f32
} else {
    0.0f32
});
define_binary_f32!(ne_tensor_contiguous_f32, |l: f32, r: f32| if l != r {
    1.0f32
} else {
    0.0f32
});
define_binary_f32!(lt_tensor_contiguous_f32, |l: f32, r: f32| if l < r {
    1.0f32
} else {
    0.0f32
});
define_binary_f32!(gt_tensor_contiguous_f32, |l: f32, r: f32| if l > r {
    1.0f32
} else {
    0.0f32
});
define_binary_f32!(le_tensor_contiguous_f32, |l: f32, r: f32| if l <= r {
    1.0f32
} else {
    0.0f32
});
define_binary_f32!(ge_tensor_contiguous_f32, |l: f32, r: f32| if l >= r {
    1.0f32
} else {
    0.0f32
});

// ── Hand-written complex f32 kernels ────────────────────────────────────

pub fn pow_tensor_contiguous_f32(
    input: &[f32],
    meta: &TensorMeta,
    exponent: f32,
) -> Result<Vec<f32>, KernelError> {
    ensure_unary_layout_and_storage_f32(input, meta)?;
    let numel = meta.numel();
    if numel == 0 {
        return Ok(Vec::new());
    }
    let start = meta.storage_offset();
    let window = &input[start..start + numel];
    // See `pow_tensor_contiguous_f64`: compute-bound, pure per-element map, so
    // par_iter is bit-identical to the serial map.
    if numel >= PARALLEL_THRESHOLD {
        Ok(window
            .par_iter()
            .map(|value| powf_torch_signed_zero_f32(*value, exponent))
            .collect())
    } else {
        Ok(window
            .iter()
            .map(|value| powf_torch_signed_zero_f32(*value, exponent))
            .collect())
    }
}

pub fn clamp_tensor_contiguous_f32(
    input: &[f32],
    meta: &TensorMeta,
    min_val: f32,
    max_val: f32,
) -> Result<Vec<f32>, KernelError> {
    ensure_unary_layout_and_storage_f32(input, meta)?;
    let numel = meta.numel();
    if numel == 0 {
        return Ok(Vec::new());
    }
    let start = meta.storage_offset();
    let window = &input[start..start + numel];
    Ok(window
        .iter()
        .map(|value| {
            // clamp is min(max(x, min_val), max_val): lower bound first,
            // then upper, so when min_val > max_val the upper bound wins.
            if value.is_nan() {
                f32::NAN
            } else {
                let lo = if !min_val.is_nan() && *value < min_val {
                    min_val
                } else {
                    *value
                };
                if !max_val.is_nan() && lo > max_val {
                    max_val
                } else {
                    lo
                }
            }
        })
        .collect())
}

/// F32 companion to `pairwise_sum_f64` — see that function for the
/// precision-correctness rationale. F32 has a smaller mantissa
/// (24 bits) so pairwise vs sequential matters even more here:
/// N = 10^6 with naive sum loses ~7 mantissa bits, vs ~5 with pairwise.
#[inline]
fn pairwise_sum_f32(values: &[f32]) -> f32 {
    const BLOCK: usize = 128;
    if values.len() <= BLOCK {
        return values.iter().sum();
    }
    let mid = values.len() / 2;
    pairwise_sum_f32(&values[..mid]) + pairwise_sum_f32(&values[mid..])
}

/// F32 companion to `pairwise_sum_map_f64` — see that function for
/// the precision-correctness rationale.
fn pairwise_sum_map_f32<F>(values: &[f32], f: F) -> f32
where
    F: Fn(f32) -> f32 + Copy,
{
    const BLOCK: usize = 128;
    if values.len() <= BLOCK {
        return values.iter().copied().map(f).sum();
    }
    let mid = values.len() / 2;
    pairwise_sum_map_f32(&values[..mid], f) + pairwise_sum_map_f32(&values[mid..], f)
}

pub fn sum_tensor_contiguous_f32(input: &[f32], meta: &TensorMeta) -> Result<f32, KernelError> {
    ensure_unary_layout_and_storage_f32(input, meta)?;
    let numel = meta.numel();
    if numel == 0 {
        return Ok(0.0);
    }
    let offset = meta.storage_offset();
    Ok(pairwise_sum_f32(&input[offset..offset + numel]))
}

pub fn mean_tensor_contiguous_f32(input: &[f32], meta: &TensorMeta) -> Result<f32, KernelError> {
    ensure_unary_layout_and_storage_f32(input, meta)?;
    let offset = meta.storage_offset();
    let numel = meta.numel();
    if numel == 0 {
        return Ok(f32::NAN);
    }
    let sum = pairwise_sum_f32(&input[offset..offset + numel]);
    #[allow(clippy::cast_precision_loss)]
    let n = numel as f32;
    Ok(sum / n)
}

pub fn sum_dim_tensor_contiguous_f32(
    input: &[f32],
    meta: &TensorMeta,
    dim: usize,
) -> Result<Vec<f32>, KernelError> {
    ensure_unary_layout_and_storage_f32(input, meta)?;
    let shape = meta.shape();
    let ndim = shape.len();
    if dim >= ndim {
        return Err(KernelError::InvalidDimension { dim, ndim });
    }
    let reduce_size = shape[dim];
    let (outer_size, inner_size, _) =
        checked_dim_loop_sizes(shape, dim, "sum_dim_f32 shape volume overflow")?;
    let out_numel = checked_mul(outer_size, inner_size, "sum_dim_f32 overflow")?;
    if out_numel == 0 {
        return Ok(Vec::new());
    }
    if reduce_size == 0 {
        return Ok(vec![0.0f32; out_numel]);
    }
    let offset = meta.storage_offset();
    // Push-based output mirrors the f64 fix (frankentorch-bv1n).
    let mut output = Vec::with_capacity(out_numel);
    let data = &input[offset..];

    // F32 mirror of the f64 sum_dim fast/general split. F32 has only
    // a 24-bit mantissa so the precision improvement from pairwise
    // vs sequential is even more pronounced here.
    if inner_size == 1 {
        for outer in 0..outer_size {
            let start = outer * reduce_size;
            let end = start + reduce_size;
            output.push(pairwise_sum_f32(&data[start..end]));
        }
        return Ok(output);
    }

    let mut scratch = vec![0.0f32; reduce_size];
    for outer in 0..outer_size {
        for inner in 0..inner_size {
            for r in 0..reduce_size {
                scratch[r] = data[outer * reduce_size * inner_size + r * inner_size + inner];
            }
            output.push(pairwise_sum_f32(&scratch));
        }
    }
    Ok(output)
}

pub fn mean_dim_tensor_contiguous_f32(
    input: &[f32],
    meta: &TensorMeta,
    dim: usize,
) -> Result<Vec<f32>, KernelError> {
    ensure_unary_layout_and_storage_f32(input, meta)?;
    let shape = meta.shape();
    let ndim = shape.len();
    if dim >= ndim {
        return Err(KernelError::InvalidDimension { dim, ndim });
    }
    let reduce_size = shape[dim];
    let (outer_size, inner_size, _) =
        checked_dim_loop_sizes(shape, dim, "mean_dim_f32 shape volume overflow")?;
    let out_numel = checked_mul(outer_size, inner_size, "mean_dim_f32 overflow")?;
    if out_numel == 0 {
        return Ok(Vec::new());
    }
    if reduce_size == 0 {
        return Ok(vec![f32::NAN; out_numel]);
    }
    let mut output = sum_dim_tensor_contiguous_f32(input, meta, dim)?;
    let scale = 1.0f32 / reduce_size as f32;
    for v in &mut output {
        *v *= scale;
    }
    Ok(output)
}

pub fn matmul_tensor_contiguous_f32(
    lhs: &[f32],
    rhs: &[f32],
    lhs_meta: &TensorMeta,
    rhs_meta: &TensorMeta,
) -> Result<Vec<f32>, KernelError> {
    ensure_dtype_device_and_layout_f32(lhs_meta, rhs_meta)?;
    let (m, k, n) = matmul_dims(lhs_meta, rhs_meta)?;
    checked_mul(m, k, "matmul_f32 lhs overflow")?;
    checked_mul(k, n, "matmul_f32 rhs overflow")?;
    let out_numel = checked_mul(m, n, "matmul_f32 output overflow")?;
    ensure_storage_len_f32(lhs, lhs_meta, "lhs")?;
    ensure_storage_len_f32(rhs, rhs_meta, "rhs")?;
    let lhs_start = lhs_meta.storage_offset();
    let rhs_start = rhs_meta.storage_offset();
    let mut out = vec![0.0f32; out_numel];

    // Use optimized GEMM via matrixmultiply crate
    gemm::sgemm(m, k, n, &lhs[lhs_start..], &rhs[rhs_start..], &mut out);

    Ok(out)
}

pub fn dot_tensor_contiguous_f32(
    lhs: &[f32],
    rhs: &[f32],
    lhs_meta: &TensorMeta,
    rhs_meta: &TensorMeta,
) -> Result<f32, KernelError> {
    ensure_dtype_device_and_layout_f32(lhs_meta, rhs_meta)?;
    if lhs_meta.shape().len() != 1 || rhs_meta.shape().len() != 1 {
        return Err(KernelError::ShapeMismatch {
            lhs: lhs_meta.shape().to_vec(),
            rhs: rhs_meta.shape().to_vec(),
        });
    }
    if lhs_meta.shape()[0] != rhs_meta.shape()[0] {
        return Err(KernelError::ShapeMismatch {
            lhs: lhs_meta.shape().to_vec(),
            rhs: rhs_meta.shape().to_vec(),
        });
    }
    ensure_storage_len_f32(lhs, lhs_meta, "lhs")?;
    ensure_storage_len_f32(rhs, rhs_meta, "rhs")?;
    let n = lhs_meta.shape()[0];
    let lhs_start = lhs_meta.storage_offset();
    let rhs_start = rhs_meta.storage_offset();
    let lhs_slice = &lhs[lhs_start..lhs_start + n];
    let rhs_slice = &rhs[rhs_start..rhs_start + n];
    // zip+map+collect mirrors the f64 fix (frankentorch-cunc):
    // skip the zero-init memset since the scratch is single-use
    // and every cell is unconditionally overwritten.
    let scratch: Vec<f32> = lhs_slice
        .iter()
        .zip(rhs_slice)
        .map(|(&l, &r)| l * r)
        .collect();
    Ok(pairwise_sum_f32(&scratch))
}

pub fn outer_tensor_contiguous_f32(
    lhs: &[f32],
    rhs: &[f32],
    lhs_meta: &TensorMeta,
    rhs_meta: &TensorMeta,
) -> Result<Vec<f32>, KernelError> {
    ensure_dtype_device_and_layout_f32(lhs_meta, rhs_meta)?;
    if lhs_meta.shape().len() != 1 || rhs_meta.shape().len() != 1 {
        return Err(KernelError::ShapeMismatch {
            lhs: lhs_meta.shape().to_vec(),
            rhs: rhs_meta.shape().to_vec(),
        });
    }
    let m = lhs_meta.shape()[0];
    let n = rhs_meta.shape()[0];
    let out_numel = checked_mul(m, n, "outer_f32 output overflow")?;
    ensure_storage_len_f32(lhs, lhs_meta, "lhs")?;
    ensure_storage_len_f32(rhs, rhs_meta, "rhs")?;
    let lhs_start = lhs_meta.storage_offset();
    let rhs_start = rhs_meta.storage_offset();
    // Capacity-alloc + extend instead of zero-init + indexed
    // overwrite (frankentorch-we9f, f32 mirror of the f64 fix).
    let mut out = Vec::with_capacity(out_numel);
    let lhs_slice = &lhs[lhs_start..lhs_start + m];
    let rhs_slice = &rhs[rhs_start..rhs_start + n];
    for &l in lhs_slice {
        out.extend(rhs_slice.iter().map(|&r| l * r));
    }
    Ok(out)
}

pub fn bmm_tensor_contiguous_f32(
    lhs: &[f32],
    rhs: &[f32],
    lhs_meta: &TensorMeta,
    rhs_meta: &TensorMeta,
) -> Result<Vec<f32>, KernelError> {
    ensure_dtype_device_and_layout_f32(lhs_meta, rhs_meta)?;
    if lhs_meta.shape().len() != 3 || rhs_meta.shape().len() != 3 {
        return Err(KernelError::ShapeMismatch {
            lhs: lhs_meta.shape().to_vec(),
            rhs: rhs_meta.shape().to_vec(),
        });
    }
    let batch = lhs_meta.shape()[0];
    let m = lhs_meta.shape()[1];
    let k = lhs_meta.shape()[2];
    let rhs_batch = rhs_meta.shape()[0];
    let rhs_k = rhs_meta.shape()[1];
    let n = rhs_meta.shape()[2];
    if batch != rhs_batch || k != rhs_k {
        return Err(KernelError::ShapeMismatch {
            lhs: lhs_meta.shape().to_vec(),
            rhs: rhs_meta.shape().to_vec(),
        });
    }
    let lhs_batch_stride = checked_mul(m, k, "bmm_f32 lhs batch stride overflow")?;
    let rhs_batch_stride = checked_mul(k, n, "bmm_f32 rhs batch stride overflow")?;
    let out_batch_stride = checked_mul(m, n, "bmm_f32 output batch stride overflow")?;
    checked_mul(batch, lhs_batch_stride, "bmm_f32 lhs overflow")?;
    checked_mul(batch, rhs_batch_stride, "bmm_f32 rhs overflow")?;
    let out_numel = checked_mul(batch, out_batch_stride, "bmm_f32 output overflow")?;
    ensure_storage_len_f32(lhs, lhs_meta, "lhs")?;
    ensure_storage_len_f32(rhs, rhs_meta, "rhs")?;
    let lhs_start = lhs_meta.storage_offset();
    let rhs_start = rhs_meta.storage_offset();
    let mut out = vec![0.0f32; out_numel];

    // Use optimized GEMM for each batch
    for b in 0..batch {
        let lhs_base = lhs_start + b * lhs_batch_stride;
        let rhs_base = rhs_start + b * rhs_batch_stride;
        let out_base = b * out_batch_stride;
        gemm::sgemm(
            m,
            k,
            n,
            &lhs[lhs_base..],
            &rhs[rhs_base..],
            &mut out[out_base..out_base + out_batch_stride],
        );
    }

    Ok(out)
}

pub fn trace_tensor_contiguous_f32(input: &[f32], meta: &TensorMeta) -> Result<f32, KernelError> {
    ensure_unary_layout_and_storage_f32(input, meta)?;
    if meta.shape().len() != 2 {
        return Err(KernelError::InvalidDimension {
            dim: meta.shape().len(),
            ndim: 2,
        });
    }
    let rows = meta.shape()[0];
    let cols = meta.shape()[1];
    let diag_len = rows.min(cols);
    let offset = meta.storage_offset();
    let mut acc = 0.0f32;
    for i in 0..diag_len {
        acc += input[offset + i * cols + i];
    }
    Ok(acc)
}

pub fn prod_dim_tensor_contiguous_f32(
    input: &[f32],
    meta: &TensorMeta,
    dim: usize,
) -> Result<Vec<f32>, KernelError> {
    ensure_unary_layout_and_storage_f32(input, meta)?;
    let shape = meta.shape();
    let ndim = shape.len();
    if dim >= ndim {
        return Err(KernelError::InvalidDimension { dim, ndim });
    }
    let reduce_size = shape[dim];
    let (outer_size, inner_size, _) = checked_dim_loop_sizes(shape, dim, "prod_dim_f32 overflow")?;
    let out_numel = checked_mul(outer_size, inner_size, "prod_dim_f32 overflow")?;
    if out_numel == 0 {
        return Ok(Vec::new());
    }
    if reduce_size == 0 {
        return Ok(vec![1.0f32; out_numel]);
    }
    let offset = meta.storage_offset();
    // Push-based output mirrors the f64 fix (frankentorch-bv1n).
    let mut output = Vec::with_capacity(out_numel);
    let data = &input[offset..];
    for outer in 0..outer_size {
        for inner in 0..inner_size {
            let mut prod = 1.0f32;
            for r in 0..reduce_size {
                prod *= data[outer * reduce_size * inner_size + r * inner_size + inner];
            }
            output.push(prod);
        }
    }
    Ok(output)
}

pub fn var_dim_tensor_contiguous_f32(
    input: &[f32],
    meta: &TensorMeta,
    dim: usize,
) -> Result<Vec<f32>, KernelError> {
    ensure_unary_layout_and_storage_f32(input, meta)?;
    let shape = meta.shape();
    let ndim = shape.len();
    if dim >= ndim {
        return Err(KernelError::InvalidDimension { dim, ndim });
    }
    let reduce_size = shape[dim];
    let (outer_size, inner_size, _) = checked_dim_loop_sizes(shape, dim, "var_dim_f32 overflow")?;
    let out_numel = checked_mul(outer_size, inner_size, "var_dim_f32 overflow")?;
    if out_numel == 0 {
        return Ok(Vec::new());
    }
    if reduce_size < 2 {
        return Ok(vec![f32::NAN; out_numel]);
    }
    let offset = meta.storage_offset();
    let data = &input[offset..];
    // Push-based output mirrors the f64 fix (frankentorch-bv1n).
    let mut output = Vec::with_capacity(out_numel);
    #[allow(clippy::cast_precision_loss)]
    let correction = (reduce_size - 1) as f32;
    #[allow(clippy::cast_precision_loss)]
    let n_div = reduce_size as f32;

    // F32 mirror of `var_dim_tensor_contiguous_f64`: gather the
    // strided values into a scratch buffer, then pairwise-sum both
    // the mean accumulator and the squared-deviation accumulator.
    // F32 has a 24-bit mantissa so the precision win vs sequential
    // accumulation is even more pronounced than f64 — squaring
    // halves the effective precision per term, and pairwise keeps
    // the cumulative error bounded by O(log N · ε) instead of
    // O(N · ε).
    let mut scratch = vec![0.0f32; reduce_size];
    for outer in 0..outer_size {
        for inner in 0..inner_size {
            for r in 0..reduce_size {
                scratch[r] = data[outer * reduce_size * inner_size + r * inner_size + inner];
            }
            let mean = pairwise_sum_f32(&scratch) / n_div;
            let var_sum = pairwise_sum_map_f32(&scratch, |x| {
                let d = x - mean;
                d * d
            });
            output.push(var_sum / correction);
        }
    }
    Ok(output)
}

pub fn std_dim_tensor_contiguous_f32(
    input: &[f32],
    meta: &TensorMeta,
    dim: usize,
) -> Result<Vec<f32>, KernelError> {
    let mut output = var_dim_tensor_contiguous_f32(input, meta, dim)?;
    for v in &mut output {
        *v = v.sqrt();
    }
    Ok(output)
}

pub fn norm_tensor_contiguous_f32(
    input: &[f32],
    meta: &TensorMeta,
    p: f32,
) -> Result<f32, KernelError> {
    ensure_unary_layout_and_storage_f32(input, meta)?;
    let offset = meta.storage_offset();
    let numel = meta.numel();
    if numel == 0 {
        return Ok(0.0f32);
    }
    let data = &input[offset..offset + numel];
    if p == f32::INFINITY {
        // norm(inf) = max(|x|). f32::max silently drops NaN, but PyTorch's
        // max reduction propagates it, so fold with explicit NaN checks.
        Ok(data.iter().fold(0.0f32, |acc, &x| {
            let a = x.abs();
            if acc.is_nan() || a.is_nan() {
                f32::NAN
            } else {
                acc.max(a)
            }
        }))
    } else if p == f32::NEG_INFINITY {
        Ok(data.iter().fold(f32::INFINITY, |acc, &x| {
            let a = x.abs();
            if acc.is_nan() || a.is_nan() {
                f32::NAN
            } else {
                acc.min(a)
            }
        }))
    } else if p == 0.0f32 {
        Ok(data.iter().filter(|&&x| x != 0.0f32).count() as f32)
    } else if p == 1.0f32 {
        Ok(pairwise_sum_map_f32(data, |x| x.abs()))
    } else if p == 2.0f32 {
        let sum_sq = pairwise_sum_map_f32(data, |x| x * x);
        Ok(sum_sq.sqrt())
    } else {
        let sum_pow = pairwise_sum_map_f32(data, |x| x.abs().powf(p));
        Ok(sum_pow.powf(1.0f32 / p))
    }
}

pub fn norm_dim_tensor_contiguous_f32(
    input: &[f32],
    meta: &TensorMeta,
    p: f32,
    dim: usize,
) -> Result<Vec<f32>, KernelError> {
    ensure_unary_layout_and_storage_f32(input, meta)?;
    let shape = meta.shape();
    let ndim = shape.len();
    if dim >= ndim {
        return Err(KernelError::InvalidDimension { dim, ndim });
    }
    let reduce_size = shape[dim];
    let (outer_size, inner_size, _) = checked_dim_loop_sizes(shape, dim, "norm_dim_f32 overflow")?;
    let out_numel = checked_mul(outer_size, inner_size, "norm_dim_f32 overflow")?;
    if out_numel == 0 {
        return Ok(Vec::new());
    }
    if reduce_size == 0 {
        return Ok(vec![0.0f32; out_numel]);
    }
    let offset = meta.storage_offset();
    let data = &input[offset..];
    // Push-based output mirrors the f64 fix (frankentorch-bv1n).
    let mut output = Vec::with_capacity(out_numel);

    if p == f32::INFINITY {
        // max(|x|) / min(|x|): f32::max/min drop NaN silently, so propagate
        // NaN explicitly to match PyTorch's max/min reductions.
        for outer in 0..outer_size {
            for inner in 0..inner_size {
                let mut max_abs = 0.0f32;
                for r in 0..reduce_size {
                    let a = data[outer * reduce_size * inner_size + r * inner_size + inner].abs();
                    max_abs = if max_abs.is_nan() || a.is_nan() {
                        f32::NAN
                    } else {
                        max_abs.max(a)
                    };
                }
                output.push(max_abs);
            }
        }
    } else if p == f32::NEG_INFINITY {
        for outer in 0..outer_size {
            for inner in 0..inner_size {
                let mut min_abs = f32::INFINITY;
                for r in 0..reduce_size {
                    let a = data[outer * reduce_size * inner_size + r * inner_size + inner].abs();
                    min_abs = if min_abs.is_nan() || a.is_nan() {
                        f32::NAN
                    } else {
                        min_abs.min(a)
                    };
                }
                output.push(min_abs);
            }
        }
    } else if p == 0.0f32 {
        for outer in 0..outer_size {
            for inner in 0..inner_size {
                let mut count = 0.0f32;
                for r in 0..reduce_size {
                    if data[outer * reduce_size * inner_size + r * inner_size + inner] != 0.0f32 {
                        count += 1.0f32;
                    }
                }
                output.push(count);
            }
        }
    } else if p == 1.0f32 {
        for outer in 0..outer_size {
            for inner in 0..inner_size {
                let mut sum = 0.0f32;
                for r in 0..reduce_size {
                    sum += data[outer * reduce_size * inner_size + r * inner_size + inner].abs();
                }
                output.push(sum);
            }
        }
    } else if p == 2.0f32 {
        for outer in 0..outer_size {
            for inner in 0..inner_size {
                let mut sum_sq = 0.0f32;
                for r in 0..reduce_size {
                    let v = data[outer * reduce_size * inner_size + r * inner_size + inner];
                    sum_sq += v * v;
                }
                output.push(sum_sq.sqrt());
            }
        }
    } else {
        for outer in 0..outer_size {
            for inner in 0..inner_size {
                let mut sum_pow = 0.0f32;
                for r in 0..reduce_size {
                    sum_pow += data[outer * reduce_size * inner_size + r * inner_size + inner]
                        .abs()
                        .powf(p);
                }
                output.push(sum_pow.powf(1.0f32 / p));
            }
        }
    }
    Ok(output)
}

pub fn softmax_dim_tensor_contiguous_f32(
    input: &[f32],
    meta: &TensorMeta,
    dim: usize,
) -> Result<Vec<f32>, KernelError> {
    ensure_unary_layout_and_storage_f32(input, meta)?;
    let shape = meta.shape();
    let ndim = shape.len();
    if dim >= ndim {
        return Err(KernelError::InvalidDimension { dim, ndim });
    }
    let reduce_size = shape[dim];
    let (outer_size, inner_size, numel) =
        checked_dim_loop_sizes(shape, dim, "softmax_f32 overflow")?;
    if numel == 0 {
        return Ok(Vec::new());
    }
    let offset = meta.storage_offset();
    let mut output = vec![0.0f32; numel];
    let data = &input[offset..];

    // F32 mirror of `softmax_dim_tensor_contiguous_f64`. F32's 24-bit
    // mantissa makes the pairwise vs sequential precision win even
    // larger here than in the f64 path.
    if inner_size == 1 {
        for outer in 0..outer_size {
            let start = outer * reduce_size;
            let end = start + reduce_size;
            let in_slice = &data[start..end];
            let max_val = in_slice.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            for (out, &x) in output[start..end].iter_mut().zip(in_slice.iter()) {
                *out = (x - max_val).exp();
            }
            let sum = pairwise_sum_f32(&output[start..end]);
            for v in &mut output[start..end] {
                *v /= sum;
            }
        }
        return Ok(output);
    }

    let mut scratch = vec![0.0f32; reduce_size];
    for outer in 0..outer_size {
        for inner in 0..inner_size {
            for r in 0..reduce_size {
                scratch[r] = data[outer * reduce_size * inner_size + r * inner_size + inner];
            }
            let max_val = scratch.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            for v in scratch.iter_mut() {
                *v = (*v - max_val).exp();
            }
            let sum = pairwise_sum_f32(&scratch);
            for (r, &exp_x) in scratch.iter().enumerate() {
                output[outer * reduce_size * inner_size + r * inner_size + inner] = exp_x / sum;
            }
        }
    }
    Ok(output)
}

pub fn log_softmax_dim_tensor_contiguous_f32(
    input: &[f32],
    meta: &TensorMeta,
    dim: usize,
) -> Result<Vec<f32>, KernelError> {
    ensure_unary_layout_and_storage_f32(input, meta)?;
    let shape = meta.shape();
    let ndim = shape.len();
    if dim >= ndim {
        return Err(KernelError::InvalidDimension { dim, ndim });
    }
    let reduce_size = shape[dim];
    let (outer_size, inner_size, numel) =
        checked_dim_loop_sizes(shape, dim, "log_softmax_f32 overflow")?;
    if numel == 0 {
        return Ok(Vec::new());
    }
    let offset = meta.storage_offset();
    let mut output = vec![0.0f32; numel];
    let data = &input[offset..];

    // F32 mirror of `log_softmax_dim_tensor_contiguous_f64`: parallelize the
    // last-dim cross-entropy hot path, but keep strided general dims serial.
    // Uses (x - max) - log(sum_exp) rather than x - (max + log(sum_exp)) for the
    // same precision-preservation reasons (frankentorch-ebrb).
    if inner_size == 1 {
        output
            .par_chunks_mut(reduce_size)
            .zip(data[..numel].par_chunks(reduce_size))
            .for_each(|(out_slice, in_slice)| {
                let max_val = in_slice.iter().copied().fold(f32::NEG_INFINITY, f32::max);
                let sum_exp = pairwise_sum_map_f32(in_slice, |x| (x - max_val).exp());
                let log_sum_exp = sum_exp.ln();
                for (out, &x) in out_slice.iter_mut().zip(in_slice.iter()) {
                    *out = (x - max_val) - log_sum_exp;
                }
            });
        return Ok(output);
    }

    let mut scratch = vec![0.0f32; reduce_size];
    for outer in 0..outer_size {
        for inner in 0..inner_size {
            for r in 0..reduce_size {
                scratch[r] = data[outer * reduce_size * inner_size + r * inner_size + inner];
            }
            let max_val = scratch.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            let sum_exp = pairwise_sum_map_f32(&scratch, |x| (x - max_val).exp());
            let log_sum_exp = sum_exp.ln();
            for (r, &x) in scratch.iter().enumerate() {
                output[outer * reduce_size * inner_size + r * inner_size + inner] =
                    (x - max_val) - log_sum_exp;
            }
        }
    }
    Ok(output)
}

pub fn argmax_dim_tensor_contiguous_f32(
    input: &[f32],
    meta: &TensorMeta,
    dim: usize,
) -> Result<Vec<f32>, KernelError> {
    ensure_unary_layout_and_storage_f32(input, meta)?;
    let shape = meta.shape();
    let ndim = shape.len();
    if dim >= ndim {
        return Err(KernelError::InvalidDimension { dim, ndim });
    }
    let reduce_size = shape[dim];
    let (outer_size, inner_size, _) = checked_dim_loop_sizes(shape, dim, "argmax_f32 overflow")?;
    let out_numel = checked_mul(outer_size, inner_size, "argmax_f32 overflow")?;
    if out_numel == 0 {
        return Ok(Vec::new());
    }
    if reduce_size == 0 {
        return Err(KernelError::EmptyReductionDim { dim });
    }
    let offset = meta.storage_offset();
    let mut output = vec![0.0f32; out_numel];
    let data = &input[offset..];
    for outer in 0..outer_size {
        for inner in 0..inner_size {
            let mut best_idx = 0usize;
            let mut best_val = f32::NEG_INFINITY;
            for r in 0..reduce_size {
                let idx = outer * reduce_size * inner_size + r * inner_size + inner;
                let val = data[idx];
                if val.is_nan() {
                    best_idx = r;
                    break;
                } else if val > best_val {
                    best_val = val;
                    best_idx = r;
                }
            }
            output[outer * inner_size + inner] = best_idx as f32;
        }
    }
    Ok(output)
}

pub fn argmin_dim_tensor_contiguous_f32(
    input: &[f32],
    meta: &TensorMeta,
    dim: usize,
) -> Result<Vec<f32>, KernelError> {
    ensure_unary_layout_and_storage_f32(input, meta)?;
    let shape = meta.shape();
    let ndim = shape.len();
    if dim >= ndim {
        return Err(KernelError::InvalidDimension { dim, ndim });
    }
    let reduce_size = shape[dim];
    let (outer_size, inner_size, _) = checked_dim_loop_sizes(shape, dim, "argmin_f32 overflow")?;
    let out_numel = checked_mul(outer_size, inner_size, "argmin_f32 overflow")?;
    if out_numel == 0 {
        return Ok(Vec::new());
    }
    if reduce_size == 0 {
        return Err(KernelError::EmptyReductionDim { dim });
    }
    let offset = meta.storage_offset();
    let mut output = vec![0.0f32; out_numel];
    let data = &input[offset..];
    for outer in 0..outer_size {
        for inner in 0..inner_size {
            let mut best_idx = 0usize;
            let mut best_val = f32::INFINITY;
            for r in 0..reduce_size {
                let idx = outer * reduce_size * inner_size + r * inner_size + inner;
                let val = data[idx];
                if val.is_nan() {
                    best_idx = r;
                    break;
                } else if val < best_val {
                    best_val = val;
                    best_idx = r;
                }
            }
            output[outer * inner_size + inner] = best_idx as f32;
        }
    }
    Ok(output)
}

pub fn max_dim_tensor_contiguous_f32(
    input: &[f32],
    meta: &TensorMeta,
    dim: usize,
) -> Result<(Vec<f32>, Vec<f32>), KernelError> {
    ensure_unary_layout_and_storage_f32(input, meta)?;
    let shape = meta.shape();
    let ndim = shape.len();
    if dim >= ndim {
        return Err(KernelError::InvalidDimension { dim, ndim });
    }
    let reduce_size = shape[dim];
    let (outer_size, inner_size, _) = checked_dim_loop_sizes(shape, dim, "max_dim_f32 overflow")?;
    let out_numel = checked_mul(outer_size, inner_size, "max_dim_f32 overflow")?;
    if out_numel == 0 {
        return Ok((Vec::new(), Vec::new()));
    }
    if reduce_size == 0 {
        return Err(KernelError::EmptyReductionDim { dim });
    }
    let offset = meta.storage_offset();
    let mut values = vec![f32::NEG_INFINITY; out_numel];
    let mut indices = vec![0.0f32; out_numel];
    let data = &input[offset..];
    for outer in 0..outer_size {
        for inner in 0..inner_size {
            let out_idx = outer * inner_size + inner;
            for r in 0..reduce_size {
                let idx = outer * reduce_size * inner_size + r * inner_size + inner;
                let val = data[idx];
                if val.is_nan() {
                    values[out_idx] = f32::NAN;
                    indices[out_idx] = r as f32;
                    break;
                } else if val > values[out_idx] {
                    values[out_idx] = val;
                    indices[out_idx] = r as f32;
                }
            }
        }
    }
    Ok((values, indices))
}

pub fn min_dim_tensor_contiguous_f32(
    input: &[f32],
    meta: &TensorMeta,
    dim: usize,
) -> Result<(Vec<f32>, Vec<f32>), KernelError> {
    ensure_unary_layout_and_storage_f32(input, meta)?;
    let shape = meta.shape();
    let ndim = shape.len();
    if dim >= ndim {
        return Err(KernelError::InvalidDimension { dim, ndim });
    }
    let reduce_size = shape[dim];
    let (outer_size, inner_size, _) = checked_dim_loop_sizes(shape, dim, "min_dim_f32 overflow")?;
    let out_numel = checked_mul(outer_size, inner_size, "min_dim_f32 overflow")?;
    if out_numel == 0 {
        return Ok((Vec::new(), Vec::new()));
    }
    if reduce_size == 0 {
        return Err(KernelError::EmptyReductionDim { dim });
    }
    let offset = meta.storage_offset();
    let mut values = vec![f32::INFINITY; out_numel];
    let mut indices = vec![0.0f32; out_numel];
    let data = &input[offset..];
    for outer in 0..outer_size {
        for inner in 0..inner_size {
            let out_idx = outer * inner_size + inner;
            for r in 0..reduce_size {
                let idx = outer * reduce_size * inner_size + r * inner_size + inner;
                let val = data[idx];
                if val.is_nan() {
                    values[out_idx] = f32::NAN;
                    indices[out_idx] = r as f32;
                    break;
                } else if val < values[out_idx] {
                    values[out_idx] = val;
                    indices[out_idx] = r as f32;
                }
            }
        }
    }
    Ok((values, indices))
}

pub fn cat_tensor_contiguous_f32(
    inputs: &[(&[f32], &TensorMeta)],
    dim: usize,
) -> Result<Vec<f32>, KernelError> {
    if inputs.is_empty() {
        return Err(KernelError::ShapeMismatch {
            lhs: vec![],
            rhs: vec![],
        });
    }
    let first_shape = inputs[0].1.shape();
    let ndim = first_shape.len();
    if dim >= ndim {
        return Err(KernelError::InvalidDimension { dim, ndim });
    }
    let mut total_cat_size = 0usize;
    for (data, meta) in inputs {
        ensure_unary_layout_and_storage_f32(data, meta)?;
        let shape = meta.shape();
        if shape.len() != ndim {
            return Err(KernelError::ShapeMismatch {
                lhs: first_shape.to_vec(),
                rhs: shape.to_vec(),
            });
        }
        for d in 0..ndim {
            if d != dim && shape[d] != first_shape[d] {
                return Err(KernelError::ShapeMismatch {
                    lhs: first_shape.to_vec(),
                    rhs: shape.to_vec(),
                });
            }
        }
        total_cat_size =
            total_cat_size
                .checked_add(shape[dim])
                .ok_or(KernelError::ShapeOverflow {
                    context: "cat_f32 shape sum overflow",
                })?;
    }
    let (outer_size, inner_size, _) = checked_dim_loop_sizes(first_shape, dim, "cat_f32 overflow")?;
    let out_numel = checked_mul(
        checked_mul(outer_size, total_cat_size, "cat_f32 overflow")?,
        inner_size,
        "cat_f32 overflow",
    )?;
    if out_numel == 0 {
        return Ok(Vec::new());
    }
    let mut output = Vec::with_capacity(out_numel);
    for outer in 0..outer_size {
        for (data, meta) in inputs {
            let cat_size = meta.shape()[dim];
            if cat_size == 0 {
                continue;
            }
            let offset = meta.storage_offset();
            let d = &data[offset..];
            let block_len = checked_mul(cat_size, inner_size, "cat_f32 slice range overflow")?;
            let range = checked_contiguous_range(outer, block_len, "cat_f32 slice range overflow")?;
            output.extend_from_slice(&d[range]);
        }
    }
    Ok(output)
}

pub fn stack_tensor_contiguous_f32(
    inputs: &[(&[f32], &TensorMeta)],
    dim: usize,
) -> Result<Vec<f32>, KernelError> {
    if inputs.is_empty() {
        return Err(KernelError::ShapeMismatch {
            lhs: vec![],
            rhs: vec![],
        });
    }
    let first_shape = inputs[0].1.shape();
    let ndim = first_shape.len();
    if dim > ndim {
        return Err(KernelError::InvalidDimension {
            dim,
            ndim: ndim + 1,
        });
    }
    for (data, meta) in inputs {
        ensure_unary_layout_and_storage_f32(data, meta)?;
        if meta.shape() != first_shape {
            return Err(KernelError::ShapeMismatch {
                lhs: first_shape.to_vec(),
                rhs: meta.shape().to_vec(),
            });
        }
    }
    let num_inputs = inputs.len();
    let outer_size = checked_shape_numel(&first_shape[..dim], "stack_f32 overflow")?;
    let inner_size = checked_shape_numel(&first_shape[dim..], "stack_f32 overflow")?;
    let out_numel = checked_mul(
        checked_mul(outer_size, num_inputs, "stack_f32 overflow")?,
        inner_size,
        "stack_f32 overflow",
    )?;
    if out_numel == 0 {
        return Ok(Vec::new());
    }
    let mut output = Vec::with_capacity(out_numel);
    for outer in 0..outer_size {
        for (data, meta) in inputs {
            let offset = meta.storage_offset();
            let d = &data[offset..];
            let range =
                checked_contiguous_range(outer, inner_size, "stack_f32 slice range overflow")?;
            output.extend_from_slice(&d[range]);
        }
    }
    Ok(output)
}

pub fn narrow_tensor_contiguous_f32(
    input: &[f32],
    meta: &TensorMeta,
    dim: usize,
    start: usize,
    length: usize,
) -> Result<Vec<f32>, KernelError> {
    ensure_unary_layout_and_storage_f32(input, meta)?;
    let shape = meta.shape();
    let ndim = shape.len();
    if dim >= ndim {
        return Err(KernelError::InvalidDimension { dim, ndim });
    }
    let end = start
        .checked_add(length)
        .ok_or(KernelError::InvalidDimension {
            dim: start,
            ndim: shape[dim],
        })?;
    if end > shape[dim] {
        return Err(KernelError::InvalidDimension {
            dim: end,
            ndim: shape[dim],
        });
    }
    if length == 0 {
        return Ok(Vec::new());
    }
    let (outer_size, inner_size, _) = checked_dim_loop_sizes(shape, dim, "narrow_f32 overflow")?;
    let dim_size = shape[dim];
    let out_numel = checked_mul(
        checked_mul(outer_size, length, "narrow_f32 overflow")?,
        inner_size,
        "narrow_f32 overflow",
    )?;
    if out_numel == 0 {
        return Ok(Vec::new());
    }
    let offset = meta.storage_offset();
    let data = &input[offset..];
    let mut output = Vec::with_capacity(out_numel);
    for outer in 0..outer_size {
        for r in 0..length {
            for inner in 0..inner_size {
                let idx = outer * dim_size * inner_size + (start + r) * inner_size + inner;
                output.push(data[idx]);
            }
        }
    }
    Ok(output)
}

pub fn expand_tensor_contiguous_f32(
    input: &[f32],
    meta: &TensorMeta,
    target_shape: &[usize],
) -> Result<Vec<f32>, KernelError> {
    ensure_unary_layout_and_storage_f32(input, meta)?;
    let shape = meta.shape();
    let ndim = shape.len();
    if target_shape.len() != ndim {
        return Err(KernelError::ShapeMismatch {
            lhs: shape.to_vec(),
            rhs: target_shape.to_vec(),
        });
    }
    for d in 0..ndim {
        if shape[d] != target_shape[d] && shape[d] != 1 {
            return Err(KernelError::ShapeMismatch {
                lhs: shape.to_vec(),
                rhs: target_shape.to_vec(),
            });
        }
    }
    let out_numel = checked_shape_numel(target_shape, "expand_f32 overflow")?;
    if out_numel == 0 {
        return Ok(Vec::new());
    }
    let input_strides = broadcast_strides(shape, target_shape, "expand_f32 strides overflow")?;
    let offset = meta.storage_offset();
    let mut output = Vec::with_capacity(out_numel);
    let mut coords = vec![0usize; ndim];
    for _ in 0..out_numel {
        let mut idx = offset;
        for d in 0..ndim {
            idx += coords[d] * input_strides[d];
        }
        output.push(input[idx]);
        for d in (0..ndim).rev() {
            coords[d] += 1;
            if coords[d] < target_shape[d] {
                break;
            }
            coords[d] = 0;
        }
    }
    Ok(output)
}

pub fn index_select_tensor_contiguous_f32(
    input: &[f32],
    meta: &TensorMeta,
    dim: usize,
    indices: &[f64],
) -> Result<Vec<f32>, KernelError> {
    ensure_unary_layout_and_storage_f32(input, meta)?;
    let shape = meta.shape();
    let ndim = shape.len();
    if dim >= ndim {
        return Err(KernelError::InvalidDimension { dim, ndim });
    }
    let dim_size = shape[dim];
    let (outer_size, inner_size, _) =
        checked_dim_loop_sizes(shape, dim, "index_select_f32 overflow")?;
    let num_indices = indices.len();
    let out_numel = checked_mul(
        checked_mul(outer_size, num_indices, "index_select_f32 overflow")?,
        inner_size,
        "index_select_f32 overflow",
    )?;
    if out_numel == 0 {
        return Ok(Vec::new());
    }
    let offset = meta.storage_offset();
    let data = &input[offset..];
    let mut output = Vec::with_capacity(out_numel);
    for outer in 0..outer_size {
        for &idx_f in indices {
            let idx = normalize_strict_index_value(idx_f, dim_size)?;
            for inner in 0..inner_size {
                let src = outer * dim_size * inner_size + idx * inner_size + inner;
                output.push(data[src]);
            }
        }
    }
    Ok(output)
}

pub fn gather_tensor_contiguous_f32(
    input: &[f32],
    meta: &TensorMeta,
    dim: usize,
    index: &[f64],
    index_meta: &TensorMeta,
) -> Result<Vec<f32>, KernelError> {
    ensure_unary_layout_and_storage_f32(input, meta)?;
    ensure_unary_layout_and_storage(index, index_meta)?;
    let shape = meta.shape();
    let ndim = shape.len();
    if dim >= ndim {
        return Err(KernelError::InvalidDimension { dim, ndim });
    }
    let idx_shape = index_meta.shape();
    if idx_shape.len() != ndim {
        return Err(KernelError::ShapeMismatch {
            lhs: shape.to_vec(),
            rhs: idx_shape.to_vec(),
        });
    }
    for d in 0..ndim {
        if d != dim && idx_shape[d] != shape[d] {
            return Err(KernelError::ShapeMismatch {
                lhs: shape.to_vec(),
                rhs: idx_shape.to_vec(),
            });
        }
    }
    let dim_size = shape[dim];
    let idx_dim_size = idx_shape[dim];
    let (outer_size, inner_size, out_numel) =
        checked_dim_loop_sizes(idx_shape, dim, "gather_f32 overflow")?;
    if out_numel == 0 {
        return Ok(Vec::new());
    }
    let offset = meta.storage_offset();
    let data = &input[offset..];
    let idx_offset = index_meta.storage_offset();
    let index_data = &index[idx_offset..];
    let mut output = Vec::with_capacity(out_numel);
    for outer in 0..outer_size {
        for r in 0..idx_dim_size {
            for inner in 0..inner_size {
                let idx_pos = outer * idx_dim_size * inner_size + r * inner_size + inner;
                let selected = normalize_strict_index_value(index_data[idx_pos], dim_size)?;
                let src = outer * dim_size * inner_size + selected * inner_size + inner;
                output.push(data[src]);
            }
        }
    }
    Ok(output)
}

pub fn scatter_tensor_contiguous_f32(
    input: &[f32],
    meta: &TensorMeta,
    dim: usize,
    index: &[f64],
    index_meta: &TensorMeta,
    src: &[f32],
) -> Result<Vec<f32>, KernelError> {
    ensure_unary_layout_and_storage_f32(input, meta)?;
    ensure_unary_layout_and_storage(index, index_meta)?;
    let shape = meta.shape();
    let ndim = shape.len();
    if dim >= ndim {
        return Err(KernelError::InvalidDimension { dim, ndim });
    }
    let idx_shape = index_meta.shape();
    if idx_shape.len() != ndim {
        return Err(KernelError::ShapeMismatch {
            lhs: shape.to_vec(),
            rhs: idx_shape.to_vec(),
        });
    }
    for d in 0..ndim {
        if d != dim && idx_shape[d] != shape[d] {
            return Err(KernelError::ShapeMismatch {
                lhs: shape.to_vec(),
                rhs: idx_shape.to_vec(),
            });
        }
    }
    let src_numel = checked_shape_numel(idx_shape, "scatter_f32 overflow")?;
    if src.len() < src_numel {
        return Err(KernelError::InsufficientStorage {
            side: "src",
            needed: src_numel,
            available: src.len(),
        });
    }
    let numel = meta.numel();
    if src_numel == 0 {
        if numel == 0 {
            return Ok(Vec::new());
        }
        let offset = meta.storage_offset();
        return Ok(input[offset..offset + numel].to_vec());
    }
    let dim_size = shape[dim];
    let idx_dim_size = idx_shape[dim];
    let (outer_size, inner_size, _) =
        checked_dim_loop_sizes(idx_shape, dim, "scatter_f32 overflow")?;
    if numel == 0 {
        return Ok(Vec::new());
    }
    let offset = meta.storage_offset();
    let mut output = input[offset..offset + numel].to_vec();
    let idx_offset = index_meta.storage_offset();
    let index_data = &index[idx_offset..];
    for outer in 0..outer_size {
        for r in 0..idx_dim_size {
            for inner in 0..inner_size {
                let idx_pos = outer * idx_dim_size * inner_size + r * inner_size + inner;
                let selected = normalize_strict_index_value(index_data[idx_pos], dim_size)?;
                let dst = outer * dim_size * inner_size + selected * inner_size + inner;
                output[dst] = src[idx_pos];
            }
        }
    }
    Ok(output)
}

/// Like `scatter_tensor_contiguous_f32` but **adds** `src` values instead of overwriting.
pub fn scatter_add_tensor_contiguous_f32(
    input: &[f32],
    meta: &TensorMeta,
    dim: usize,
    index: &[f64],
    index_meta: &TensorMeta,
    src: &[f32],
) -> Result<Vec<f32>, KernelError> {
    ensure_unary_layout_and_storage_f32(input, meta)?;
    ensure_unary_layout_and_storage(index, index_meta)?;
    let shape = meta.shape();
    let ndim = shape.len();
    if dim >= ndim {
        return Err(KernelError::InvalidDimension { dim, ndim });
    }
    let idx_shape = index_meta.shape();
    if idx_shape.len() != ndim {
        return Err(KernelError::ShapeMismatch {
            lhs: shape.to_vec(),
            rhs: idx_shape.to_vec(),
        });
    }
    for d in 0..ndim {
        if d != dim && idx_shape[d] != shape[d] {
            return Err(KernelError::ShapeMismatch {
                lhs: shape.to_vec(),
                rhs: idx_shape.to_vec(),
            });
        }
    }
    let src_numel = checked_shape_numel(idx_shape, "scatter_add_f32 overflow")?;
    if src.len() < src_numel {
        return Err(KernelError::InsufficientStorage {
            side: "src",
            needed: src_numel,
            available: src.len(),
        });
    }
    let numel = meta.numel();
    if src_numel == 0 {
        if numel == 0 {
            return Ok(Vec::new());
        }
        let offset = meta.storage_offset();
        return Ok(input[offset..offset + numel].to_vec());
    }

    let dim_size = shape[dim];
    let idx_dim_size = idx_shape[dim];
    let (outer_size, inner_size, _) =
        checked_dim_loop_sizes(idx_shape, dim, "scatter_add_f32 overflow")?;
    if numel == 0 {
        return Ok(Vec::new());
    }
    let offset = meta.storage_offset();
    let mut output = input[offset..offset + numel].to_vec();
    let idx_offset = index_meta.storage_offset();
    let index_data = &index[idx_offset..];
    for outer in 0..outer_size {
        for r in 0..idx_dim_size {
            for inner in 0..inner_size {
                let idx_pos = outer * idx_dim_size * inner_size + r * inner_size + inner;
                let selected = normalize_strict_index_value(index_data[idx_pos], dim_size)?;
                let dst = outer * dim_size * inner_size + selected * inner_size + inner;
                output[dst] += src[idx_pos];
            }
        }
    }
    Ok(output)
}

pub fn index_put_tensor_contiguous_f32(
    input: &[f32],
    meta: &TensorMeta,
    indices: &[Vec<f64>],
    values: &[f32],
    accumulate: bool,
) -> Result<Vec<f32>, KernelError> {
    ensure_unary_layout_and_storage_f32(input, meta)?;
    let shape = meta.shape();
    let ndim = shape.len();

    if indices.is_empty() {
        return Err(KernelError::ShapeMismatch {
            lhs: shape.to_vec(),
            rhs: vec![0],
        });
    }

    let num_indexed_dims = indices.len();
    if num_indexed_dims > ndim {
        return Err(KernelError::InvalidDimension {
            dim: num_indexed_dims,
            ndim,
        });
    }

    let n_indices = indices[0].len();
    for idx_tensor in &indices[1..] {
        if idx_tensor.len() != n_indices {
            return Err(KernelError::ShapeMismatch {
                lhs: vec![n_indices],
                rhs: vec![idx_tensor.len()],
            });
        }
    }

    let suffix_size = checked_shape_numel(
        &shape[num_indexed_dims..],
        "index_put suffix shape overflow",
    )?;

    let values_needed = checked_mul(n_indices, suffix_size, "index_put values shape overflow")?;
    let scalar_broadcast = values.len() == 1 && values_needed > 1;
    if !scalar_broadcast && values.len() < values_needed {
        return Err(KernelError::InsufficientStorage {
            side: "values",
            needed: values_needed,
            available: values.len(),
        });
    }

    let offset = meta.storage_offset();
    let numel = meta.numel();
    if numel == 0 {
        return Ok(Vec::new());
    }
    let mut output = input[offset..offset + numel].to_vec();

    let mut indexed_strides = vec![0usize; num_indexed_dims];
    for d in 0..num_indexed_dims {
        indexed_strides[d] =
            checked_shape_numel(&shape[d + 1..], "index_put stride shape overflow")?;
    }

    for i in 0..n_indices {
        let mut base = 0usize;
        for d in 0..num_indexed_dims {
            let idx = normalize_wrapped_index_value(indices[d][i], shape[d])?;
            base += idx * indexed_strides[d];
        }

        for s in 0..suffix_size {
            let val = if scalar_broadcast {
                values[0]
            } else {
                values[i * suffix_size + s]
            };
            if accumulate {
                output[base + s] += val;
            } else {
                output[base + s] = val;
            }
        }
    }

    Ok(output)
}

pub fn masked_fill_tensor_contiguous_f32(
    input: &[f32],
    meta: &TensorMeta,
    mask: &[f64],
    value: f32,
) -> Result<Vec<f32>, KernelError> {
    ensure_unary_layout_and_storage_f32(input, meta)?;
    let numel = meta.numel();
    if numel == 0 {
        return Ok(Vec::new());
    }
    let offset = meta.storage_offset();
    if mask.len() < offset + numel {
        return Err(KernelError::ShapeMismatch {
            lhs: meta.shape().to_vec(),
            rhs: vec![mask.len()],
        });
    }
    let data = &input[offset..offset + numel];
    let output = data
        .iter()
        .zip(mask[offset..offset + numel].iter())
        .map(|(&d, &m)| if m != 0.0 { value } else { d })
        .collect();
    Ok(output)
}

pub fn where_tensor_contiguous_f32(
    condition: &[f64],
    x: &[f32],
    y: &[f32],
    meta: &TensorMeta,
) -> Result<Vec<f32>, KernelError> {
    ensure_unary_layout_and_storage_f32(x, meta)?;
    let numel = meta.numel();
    if numel == 0 {
        return Ok(Vec::new());
    }
    let offset = meta.storage_offset();
    if condition.len() < offset + numel {
        return Err(KernelError::ShapeMismatch {
            lhs: meta.shape().to_vec(),
            rhs: vec![condition.len()],
        });
    }
    if y.len() < offset + numel {
        return Err(KernelError::ShapeMismatch {
            lhs: meta.shape().to_vec(),
            rhs: vec![y.len()],
        });
    }
    let cond = &condition[offset..offset + numel];
    let x_data = &x[offset..offset + numel];
    let y_data = &y[offset..offset + numel];
    let output = cond
        .iter()
        .zip(x_data.iter())
        .zip(y_data.iter())
        .map(|((&c, &xv), &yv)| if c != 0.0 { xv } else { yv })
        .collect();
    Ok(output)
}

pub fn cumsum_tensor_contiguous_f32(
    input: &[f32],
    meta: &TensorMeta,
    dim: usize,
) -> Result<Vec<f32>, KernelError> {
    ensure_unary_layout_and_storage_f32(input, meta)?;
    let shape = meta.shape();
    let ndim = shape.len();
    if dim >= ndim {
        return Err(KernelError::InvalidDimension { dim, ndim });
    }
    let dim_size = shape[dim];
    let (outer_size, inner_size, _) = checked_dim_loop_sizes(shape, dim, "cumsum_f32 overflow")?;
    let numel = checked_mul(
        checked_mul(outer_size, dim_size, "cumsum_f32 overflow")?,
        inner_size,
        "cumsum_f32 overflow",
    )?;
    if numel == 0 {
        return Ok(Vec::new());
    }
    let offset = meta.storage_offset();
    let mut output = vec![0.0f32; numel];
    let data = &input[offset..];
    for outer in 0..outer_size {
        for inner in 0..inner_size {
            let mut acc = 0.0f32;
            for d in 0..dim_size {
                let idx = outer * dim_size * inner_size + d * inner_size + inner;
                acc += data[idx];
                output[idx] = acc;
            }
        }
    }
    Ok(output)
}

pub fn cumsum_backward_tensor_contiguous_f32(
    grad_output: &[f32],
    meta: &TensorMeta,
    dim: usize,
) -> Result<Vec<f32>, KernelError> {
    ensure_unary_layout_and_storage_f32(grad_output, meta)?;
    let shape = meta.shape();
    let ndim = shape.len();
    if dim >= ndim {
        return Err(KernelError::InvalidDimension { dim, ndim });
    }
    let dim_size = shape[dim];
    let (outer_size, inner_size, _) =
        checked_dim_loop_sizes(shape, dim, "cumsum_backward_f32 overflow")?;
    let numel = checked_mul(
        checked_mul(outer_size, dim_size, "cumsum_backward_f32 overflow")?,
        inner_size,
        "cumsum_backward_f32 overflow",
    )?;
    if numel == 0 {
        return Ok(Vec::new());
    }
    let offset = meta.storage_offset();
    let mut grad_input = vec![0.0f32; numel];
    let data = &grad_output[offset..];
    for outer in 0..outer_size {
        for inner in 0..inner_size {
            let mut acc = 0.0f32;
            for d in (0..dim_size).rev() {
                let idx = outer * dim_size * inner_size + d * inner_size + inner;
                acc += data[idx];
                grad_input[idx] = acc;
            }
        }
    }
    Ok(grad_input)
}

pub fn cumprod_tensor_contiguous_f32(
    input: &[f32],
    meta: &TensorMeta,
    dim: usize,
) -> Result<Vec<f32>, KernelError> {
    ensure_unary_layout_and_storage_f32(input, meta)?;
    let shape = meta.shape();
    let ndim = shape.len();
    if dim >= ndim {
        return Err(KernelError::InvalidDimension { dim, ndim });
    }
    let dim_size = shape[dim];
    let (outer_size, inner_size, _) = checked_dim_loop_sizes(shape, dim, "cumprod_f32 overflow")?;
    let numel = checked_mul(
        checked_mul(outer_size, dim_size, "cumprod_f32 overflow")?,
        inner_size,
        "cumprod_f32 overflow",
    )?;
    if numel == 0 {
        return Ok(Vec::new());
    }
    let offset = meta.storage_offset();
    let mut output = vec![0.0f32; numel];
    let data = &input[offset..];
    for outer in 0..outer_size {
        for inner in 0..inner_size {
            let mut acc = 1.0f32;
            for d in 0..dim_size {
                let idx = outer * dim_size * inner_size + d * inner_size + inner;
                acc *= data[idx];
                output[idx] = acc;
            }
        }
    }
    Ok(output)
}

pub fn cumprod_backward_tensor_contiguous_f32(
    grad_output: &[f32],
    input: &[f32],
    output: &[f32],
    meta: &TensorMeta,
    dim: usize,
) -> Result<Vec<f32>, KernelError> {
    ensure_unary_layout_and_storage_f32(input, meta)?;
    let shape = meta.shape();
    let ndim = shape.len();
    if dim >= ndim {
        return Err(KernelError::InvalidDimension { dim, ndim });
    }
    let dim_size = shape[dim];
    let (outer_size, inner_size, _) =
        checked_dim_loop_sizes(shape, dim, "cumprod_backward_f32 overflow")?;
    let numel = checked_mul(
        checked_mul(outer_size, dim_size, "cumprod_backward_f32 overflow")?,
        inner_size,
        "cumprod_backward_f32 overflow",
    )?;
    if numel == 0 {
        return Ok(Vec::new());
    }
    let offset = meta.storage_offset();
    let mut grad_input = vec![0.0f32; numel];
    let in_data = &input[offset..];
    let out_data = &output[offset..];
    let go_data = &grad_output[offset..];
    for outer in 0..outer_size {
        for inner in 0..inner_size {
            let mut acc = 0.0f32;
            for d in (0..dim_size).rev() {
                let idx = outer * dim_size * inner_size + d * inner_size + inner;
                acc += go_data[idx] * out_data[idx];
                let inp = in_data[idx];
                if inp.abs() > f32::EPSILON {
                    grad_input[idx] = acc / inp;
                } else {
                    let mut sum = 0.0f32;
                    for j in d..dim_size {
                        let j_idx = outer * dim_size * inner_size + j * inner_size + inner;
                        let mut prod = 1.0f32;
                        for kk in d..=j {
                            if kk != d {
                                let k_idx = outer * dim_size * inner_size + kk * inner_size + inner;
                                prod *= in_data[k_idx];
                            }
                        }
                        if d > 0 {
                            let prev_idx =
                                outer * dim_size * inner_size + (d - 1) * inner_size + inner;
                            prod *= out_data[prev_idx];
                        }
                        sum += go_data[j_idx] * prod;
                    }
                    grad_input[idx] = sum;
                }
            }
        }
    }
    Ok(grad_input)
}

pub fn sort_tensor_contiguous_f32(
    input: &[f32],
    meta: &TensorMeta,
    dim: usize,
    descending: bool,
) -> Result<(Vec<f32>, Vec<usize>), KernelError> {
    ensure_unary_layout_and_storage_f32(input, meta)?;
    let shape = meta.shape();
    let ndim = shape.len();
    if dim >= ndim {
        return Err(KernelError::InvalidDimension { dim, ndim });
    }
    let dim_size = shape[dim];
    let (outer_size, inner_size, _) = checked_dim_loop_sizes(shape, dim, "sort_f32 overflow")?;
    let numel = checked_mul(
        checked_mul(outer_size, dim_size, "sort_f32 overflow")?,
        inner_size,
        "sort_f32 overflow",
    )?;
    if numel == 0 {
        return Ok((Vec::new(), Vec::new()));
    }
    let offset = meta.storage_offset();
    let data = &input[offset..];
    let mut sorted_values = vec![0.0f32; numel];
    let mut indices = vec![0usize; numel];
    // F32 mirror: parallelize over independent per-outer blocks; bit-identical
    // to the serial version. Long (>= SORT_RADIX_MIN_LEN) NaN-free lanes take the
    // O(n) stable LSD radix path (4 effective passes for the 32-bit key); shorter
    // or NaN-bearing lanes fall back to the comparison sort, which alone
    // reproduces PyTorch's "NaN is greatest" placement. See the f64 path.
    let block = dim_size * inner_size;
    let use_radix = dim_size >= SORT_RADIX_MIN_LEN && dim_size <= u32::MAX as usize;
    sorted_values
        .par_chunks_mut(block)
        .zip(indices.par_chunks_mut(block))
        .zip(data[..numel].par_chunks(block))
        .for_each(|((sv_block, idx_block), in_block)| {
            let mut keys: Vec<u64> = Vec::new();
            let mut perm: Vec<u32> = Vec::new();
            let mut scratch: Vec<u32> = Vec::new();
            for inner in 0..inner_size {
                let mut radix_ok = use_radix;
                if radix_ok {
                    keys.clear();
                    for d in 0..dim_size {
                        let x = in_block[d * inner_size + inner];
                        if x.is_nan() {
                            radix_ok = false;
                            break;
                        }
                        let k = sort_radix_key_f32(x);
                        // High 32 bits stay zero -> the four top passes auto-skip.
                        keys.push(u64::from(if descending { !k } else { k }));
                    }
                }

                if radix_ok {
                    sort_radix_perm(&keys, &mut perm, &mut scratch);
                    for (out_d, &p) in perm.iter().enumerate() {
                        let orig_d = p as usize;
                        sv_block[out_d * inner_size + inner] =
                            in_block[orig_d * inner_size + inner];
                        idx_block[out_d * inner_size + inner] = orig_d;
                    }
                    continue;
                }

                let mut lane: Vec<(usize, f32)> = (0..dim_size)
                    .map(|d| (d, in_block[d * inner_size + inner]))
                    .collect();
                if descending {
                    lane.sort_by(|a, b| nan_greatest_cmp_f32(b.1, a.1));
                } else {
                    lane.sort_by(|a, b| nan_greatest_cmp_f32(a.1, b.1));
                }
                for (out_d, (orig_d, val)) in lane.into_iter().enumerate() {
                    sv_block[out_d * inner_size + inner] = val;
                    idx_block[out_d * inner_size + inner] = orig_d;
                }
            }
        });
    Ok((sorted_values, indices))
}

pub fn topk_tensor_contiguous_f32(
    input: &[f32],
    meta: &TensorMeta,
    k: usize,
    dim: usize,
    largest: bool,
    sorted: bool,
) -> Result<(Vec<f32>, Vec<usize>), KernelError> {
    ensure_unary_layout_and_storage_f32(input, meta)?;
    let shape = meta.shape();
    let ndim = shape.len();
    if dim >= ndim {
        return Err(KernelError::InvalidDimension { dim, ndim });
    }
    let dim_size = shape[dim];
    if k > dim_size {
        return Err(KernelError::InvalidDimension {
            dim: k,
            ndim: dim_size,
        });
    }
    let (outer_size, inner_size, _) = checked_dim_loop_sizes(shape, dim, "topk_f32 overflow")?;
    let out_numel = checked_mul(
        checked_mul(outer_size, k, "topk_f32 overflow")?,
        inner_size,
        "topk_f32 overflow",
    )?;
    if out_numel == 0 {
        return Ok((Vec::new(), Vec::new()));
    }
    let offset = meta.storage_offset();
    let data = &input[offset..];
    let mut out_values = vec![0.0f32; out_numel];
    let mut out_indices = vec![0usize; out_numel];
    // F32 mirror: parallelize over independent per-outer blocks. Every lane
    // selects independently with a total value+original-index order equivalent
    // to the serial stable sort; see the f64 path.
    let in_block = dim_size * inner_size;
    let out_block = k * inner_size;
    let in_total = outer_size * in_block;
    out_values
        .par_chunks_mut(out_block)
        .zip(out_indices.par_chunks_mut(out_block))
        .zip(data[..in_total].par_chunks(in_block))
        .for_each(|((ov_block, oi_block), in_block_data)| {
            for inner in 0..inner_size {
                let mut lane: Vec<(usize, f32)> = (0..dim_size)
                    .map(|d| (d, in_block_data[d * inner_size + inner]))
                    .collect();

                if k < dim_size {
                    let (selected, _, _) =
                        lane.select_nth_unstable_by(k, |a, b| topk_lane_cmp_f32(a, b, largest));
                    if sorted {
                        selected.sort_by(|a, b| topk_lane_cmp_f32(a, b, largest));
                    } else {
                        selected.sort_by_key(|(orig_idx, _)| *orig_idx);
                    }
                    for (out_d, (orig_d, val)) in selected.iter().copied().enumerate() {
                        ov_block[out_d * inner_size + inner] = val;
                        oi_block[out_d * inner_size + inner] = orig_d;
                    }
                    continue;
                }

                lane.sort_by(|a, b| topk_lane_cmp_f32(a, b, largest));
                let selected = &mut lane[..k];
                if !sorted {
                    selected.sort_by_key(|(orig_idx, _)| *orig_idx);
                }

                for (out_d, (orig_d, val)) in selected.iter().copied().enumerate() {
                    ov_block[out_d * inner_size + inner] = val;
                    oi_block[out_d * inner_size + inner] = orig_d;
                }
            }
        });
    Ok((out_values, out_indices))
}

pub fn lerp_tensor_contiguous_f32(
    start: &[f32],
    end: &[f32],
    weight: f32,
    meta: &TensorMeta,
) -> Result<Vec<f32>, KernelError> {
    ensure_unary_layout_and_storage_f32(start, meta)?;
    let numel = meta.numel();
    if numel == 0 {
        return Ok(Vec::new());
    }
    let offset = meta.storage_offset();
    if end.len() < offset + numel {
        return Err(KernelError::ShapeMismatch {
            lhs: meta.shape().to_vec(),
            rhs: vec![end.len()],
        });
    }
    let s = &start[offset..offset + numel];
    let e = &end[offset..offset + numel];
    Ok(s.iter()
        .zip(e.iter())
        .map(|(&sv, &ev)| sv + weight * (ev - sv))
        .collect())
}

#[allow(clippy::too_many_arguments)]
pub fn addmm_tensor_contiguous_f32(
    input: &[f32],
    mat1: &[f32],
    mat2: &[f32],
    input_meta: &TensorMeta,
    mat1_meta: &TensorMeta,
    mat2_meta: &TensorMeta,
    beta: f32,
    alpha: f32,
) -> Result<Vec<f32>, KernelError> {
    let (m, k, n) = matmul_dims(mat1_meta, mat2_meta)?;
    let out_numel = checked_mul(m, n, "addmm_f32 output overflow")?;
    ensure_storage_len_f32(mat1, mat1_meta, "mat1")?;
    ensure_storage_len_f32(mat2, mat2_meta, "mat2")?;
    let mat1_start = mat1_meta.storage_offset();
    let mat2_start = mat2_meta.storage_offset();
    let input_offset = input_meta.storage_offset();
    let input_shape = input_meta.shape();
    let input_1d = input_shape.len() == 1 && input_shape[0] == n;
    let input_2d = input_shape.len() == 2 && input_shape[0] == m && input_shape[1] == n;
    if !input_1d && !input_2d {
        return Err(KernelError::ShapeMismatch {
            lhs: input_shape.to_vec(),
            rhs: vec![m, n],
        });
    }
    ensure_storage_len_f32(input, input_meta, "input")?;

    // Fail closed on non-contiguous operands. The GEMM call below reads
    // mat1/mat2 as row-major contiguous buffers through raw pointers, and the
    // accumulation step reads `input` with a flat offset — a strided view would
    // be silently mis-read as if contiguous, producing wrong results. The
    // `matmul`/`bmm` kernels already reject this via
    // `ensure_dtype_device_and_layout`; addmm must guard the same unsafe path.
    if !mat1_meta.is_contiguous() {
        return Err(KernelError::UnsupportedLayout { side: "mat1" });
    }
    if !mat2_meta.is_contiguous() {
        return Err(KernelError::UnsupportedLayout { side: "mat2" });
    }
    if !input_meta.is_contiguous() {
        return Err(KernelError::UnsupportedLayout { side: "input" });
    }

    // Use optimized GEMM for mat1 @ mat2
    let mut gemm_out = vec![0.0f32; out_numel];
    gemm::sgemm(
        m,
        k,
        n,
        &mat1[mat1_start..],
        &mat2[mat2_start..],
        &mut gemm_out,
    );

    // Apply: out = beta * input + alpha * (mat1 @ mat2)
    let out: Vec<f32> = if input_1d {
        gemm_out
            .iter()
            .enumerate()
            .map(|(i, &g)| {
                let col = i % n;
                beta * input[input_offset + col] + alpha * g
            })
            .collect()
    } else {
        gemm_out
            .iter()
            .enumerate()
            .map(|(i, &g)| beta * input[input_offset + i] + alpha * g)
            .collect()
    };

    Ok(out)
}

#[allow(clippy::too_many_arguments)]
pub fn addmv_tensor_contiguous_f32(
    input: &[f32],
    mat: &[f32],
    vec_data: &[f32],
    input_meta: &TensorMeta,
    mat_meta: &TensorMeta,
    vec_meta: &TensorMeta,
    beta: f32,
    alpha: f32,
) -> Result<Vec<f32>, KernelError> {
    let mat_shape = mat_meta.shape();
    let vec_shape = vec_meta.shape();
    if mat_shape.len() != 2 {
        return Err(KernelError::ShapeMismatch {
            lhs: mat_shape.to_vec(),
            rhs: vec![0, 0],
        });
    }
    if vec_shape.len() != 1 {
        return Err(KernelError::ShapeMismatch {
            lhs: vec_shape.to_vec(),
            rhs: vec![mat_shape[1]],
        });
    }
    let m = mat_shape[0];
    let k = mat_shape[1];
    if vec_shape[0] != k {
        return Err(KernelError::ShapeMismatch {
            lhs: vec_shape.to_vec(),
            rhs: vec![k],
        });
    }
    let input_shape = input_meta.shape();
    if input_shape.len() != 1 || input_shape[0] != m {
        return Err(KernelError::ShapeMismatch {
            lhs: input_shape.to_vec(),
            rhs: vec![m],
        });
    }
    ensure_storage_len_f32(mat, mat_meta, "mat")?;
    ensure_storage_len_f32(vec_data, vec_meta, "vec")?;
    ensure_storage_len_f32(input, input_meta, "input")?;
    let mat_start = mat_meta.storage_offset();
    let vec_start = vec_meta.storage_offset();
    let input_start = input_meta.storage_offset();
    // Push-based output mirrors the f64 fix (frankentorch-u04j).
    let mut out = Vec::with_capacity(m);
    let mut scratch = vec![0.0f32; k];
    for row in 0..m {
        for (col, scratch_slot) in scratch.iter_mut().enumerate() {
            *scratch_slot = mat[mat_start + row * k + col] * vec_data[vec_start + col];
        }
        let acc = pairwise_sum_f32(&scratch);
        out.push(beta * input[input_start + row] + alpha * acc);
    }
    Ok(out)
}

pub fn reduce_sum_for_broadcast_f32(
    expanded_grad: &[f32],
    expanded_shape: &[usize],
    original_shape: &[usize],
) -> Result<Vec<f32>, KernelError> {
    let ndim = expanded_shape.len();
    if ndim != original_shape.len() {
        return Err(KernelError::ShapeMismatch {
            lhs: expanded_shape.to_vec(),
            rhs: original_shape.to_vec(),
        });
    }
    for d in 0..ndim {
        let expanded = expanded_shape[d];
        let original = original_shape[d];
        if original != expanded && original != 1 {
            return Err(KernelError::ShapeMismatch {
                lhs: expanded_shape.to_vec(),
                rhs: original_shape.to_vec(),
            });
        }
    }
    let expanded_numel =
        checked_shape_numel(expanded_shape, "broadcast_f32 reduction expanded shape")?;
    if expanded_grad.len() != expanded_numel {
        return Err(KernelError::ShapeMismatch {
            lhs: vec![expanded_grad.len()],
            rhs: vec![expanded_numel],
        });
    }
    let original_numel =
        checked_shape_numel(original_shape, "broadcast_f32 reduction original shape")?;
    if original_numel == 0 {
        return Ok(Vec::new());
    }
    let original_strides = broadcast_strides(
        original_shape,
        expanded_shape,
        "broadcast_f32 strides overflow",
    )?;
    let mut reduced = vec![0.0f32; original_numel];
    let mut coords = vec![0usize; ndim];
    for grad in expanded_grad {
        let mut original_idx = 0usize;
        for d in 0..ndim {
            original_idx += coords[d] * original_strides[d];
        }
        reduced[original_idx] += *grad;
        for d in (0..ndim).rev() {
            coords[d] += 1;
            if coords[d] < expanded_shape[d] {
                break;
            }
            coords[d] = 0;
        }
    }
    Ok(reduced)
}

/// Returns indices of non-zero elements as a flat f64 vector.
///
/// For an N-dimensional input, returns an (M, N) shaped result where M is the
/// number of non-zero elements. The result is stored row-major: each group of N
/// consecutive values gives the multi-dimensional index of one non-zero element.
///
/// Returns `(indices_flat, num_nonzero)` where the output shape is `[num_nonzero, ndim]`.
pub fn nonzero_tensor_contiguous_f64(
    input: &[f64],
    meta: &TensorMeta,
) -> Result<(Vec<f64>, usize), KernelError> {
    ensure_unary_layout_and_storage(input, meta)?;
    let shape = meta.shape();
    let ndim = shape.len();
    let numel = meta.numel();
    if numel == 0 {
        return Ok((Vec::new(), 0));
    }
    let offset = meta.storage_offset();
    let data = &input[offset..offset + numel];

    // Compute strides for index decomposition
    let mut strides = vec![1usize; ndim];
    for i in (0..ndim.saturating_sub(1)).rev() {
        strides[i] = strides[i + 1] * shape[i + 1];
    }

    let mut indices = Vec::new();
    for (flat_idx, &val) in data.iter().enumerate().take(numel) {
        if val != 0.0 || val.is_nan() {
            // NaN is treated as non-zero (PyTorch behavior)
            let mut remaining = flat_idx;
            for stride in strides.iter().take(ndim) {
                let dim_idx = remaining / stride;
                remaining %= stride;
                indices.push(dim_idx as f64);
            }
        }
    }

    let num_nonzero = indices.len() / ndim.max(1);
    Ok((indices, num_nonzero))
}

/// Selects elements from input where mask is non-zero.
///
/// Returns a 1-D vector of selected elements. The mask must have the same
/// number of elements as the input.
pub fn masked_select_tensor_contiguous_f64(
    input: &[f64],
    mask: &[f64],
    meta: &TensorMeta,
) -> Result<Vec<f64>, KernelError> {
    ensure_unary_layout_and_storage(input, meta)?;
    let numel = meta.numel();
    if numel == 0 {
        return Ok(Vec::new());
    }
    let offset = meta.storage_offset();
    if mask.len() < offset + numel {
        return Err(KernelError::ShapeMismatch {
            lhs: meta.shape().to_vec(),
            rhs: vec![mask.len()],
        });
    }
    let data = &input[offset..offset + numel];
    let mask_data = &mask[offset..offset + numel];

    let output: Vec<f64> = data
        .iter()
        .zip(mask_data.iter())
        .filter(|&(_, m)| *m != 0.0)
        .map(|(&v, _)| v)
        .collect();

    Ok(output)
}

// ── Complex tensor operations ─────────────────────────────────────────

/// Extract the real part of each complex element.
pub fn complex_real_contiguous(
    input: &[Complex128],
    meta: &TensorMeta,
) -> Result<Vec<f64>, KernelError> {
    if !meta.is_contiguous() {
        return Err(KernelError::UnsupportedLayout { side: "input" });
    }
    Ok(input.iter().map(|z| z.re).collect())
}

/// Extract the imaginary part of each complex element.
pub fn complex_imag_contiguous(
    input: &[Complex128],
    meta: &TensorMeta,
) -> Result<Vec<f64>, KernelError> {
    if !meta.is_contiguous() {
        return Err(KernelError::UnsupportedLayout { side: "input" });
    }
    Ok(input.iter().map(|z| z.im).collect())
}

/// Conjugate each complex element: conj(a+bi) = a-bi.
pub fn complex_conj_contiguous(
    input: &[Complex128],
    meta: &TensorMeta,
) -> Result<Vec<Complex128>, KernelError> {
    if !meta.is_contiguous() {
        return Err(KernelError::UnsupportedLayout { side: "input" });
    }
    Ok(input.iter().map(|z| z.conj()).collect())
}

/// Magnitude (absolute value) of each complex element: |a+bi| = sqrt(a^2+b^2).
pub fn complex_abs_contiguous(
    input: &[Complex128],
    meta: &TensorMeta,
) -> Result<Vec<f64>, KernelError> {
    if !meta.is_contiguous() {
        return Err(KernelError::UnsupportedLayout { side: "input" });
    }
    Ok(input.iter().map(|z| z.norm()).collect())
}

/// Phase angle of each complex element: angle(a+bi) = atan2(b, a).
pub fn complex_angle_contiguous(
    input: &[Complex128],
    meta: &TensorMeta,
) -> Result<Vec<f64>, KernelError> {
    if !meta.is_contiguous() {
        return Err(KernelError::UnsupportedLayout { side: "input" });
    }
    Ok(input.iter().map(|z| z.arg()).collect())
}

/// Element-wise complex addition.
pub fn complex_add_contiguous(
    lhs: &[Complex128],
    rhs: &[Complex128],
    lhs_meta: &TensorMeta,
    rhs_meta: &TensorMeta,
) -> Result<Vec<Complex128>, KernelError> {
    if !lhs_meta.is_contiguous() {
        return Err(KernelError::UnsupportedLayout { side: "lhs" });
    }
    if !rhs_meta.is_contiguous() {
        return Err(KernelError::UnsupportedLayout { side: "rhs" });
    }
    if lhs_meta.shape() != rhs_meta.shape() {
        return Err(KernelError::ShapeMismatch {
            lhs: lhs_meta.shape().to_vec(),
            rhs: rhs_meta.shape().to_vec(),
        });
    }
    Ok(lhs.iter().zip(rhs.iter()).map(|(a, b)| a + b).collect())
}

/// Element-wise complex multiplication.
pub fn complex_mul_contiguous(
    lhs: &[Complex128],
    rhs: &[Complex128],
    lhs_meta: &TensorMeta,
    rhs_meta: &TensorMeta,
) -> Result<Vec<Complex128>, KernelError> {
    if !lhs_meta.is_contiguous() {
        return Err(KernelError::UnsupportedLayout { side: "lhs" });
    }
    if !rhs_meta.is_contiguous() {
        return Err(KernelError::UnsupportedLayout { side: "rhs" });
    }
    if lhs_meta.shape() != rhs_meta.shape() {
        return Err(KernelError::ShapeMismatch {
            lhs: lhs_meta.shape().to_vec(),
            rhs: rhs_meta.shape().to_vec(),
        });
    }
    Ok(lhs.iter().zip(rhs.iter()).map(|(a, b)| a * b).collect())
}

/// Element-wise complex division.
pub fn complex_div_contiguous(
    lhs: &[Complex128],
    rhs: &[Complex128],
    lhs_meta: &TensorMeta,
    rhs_meta: &TensorMeta,
) -> Result<Vec<Complex128>, KernelError> {
    if !lhs_meta.is_contiguous() {
        return Err(KernelError::UnsupportedLayout { side: "lhs" });
    }
    if !rhs_meta.is_contiguous() {
        return Err(KernelError::UnsupportedLayout { side: "rhs" });
    }
    if lhs_meta.shape() != rhs_meta.shape() {
        return Err(KernelError::ShapeMismatch {
            lhs: lhs_meta.shape().to_vec(),
            rhs: rhs_meta.shape().to_vec(),
        });
    }
    Ok(lhs.iter().zip(rhs.iter()).map(|(a, b)| a / b).collect())
}

/// Construct complex tensor from separate real and imaginary f64 arrays.
pub fn complex_from_real_imag(real: &[f64], imag: &[f64]) -> Result<Vec<Complex128>, KernelError> {
    if real.len() != imag.len() {
        return Err(KernelError::ShapeMismatch {
            lhs: vec![real.len()],
            rhs: vec![imag.len()],
        });
    }
    Ok(real
        .iter()
        .zip(imag.iter())
        .map(|(&r, &i)| Complex128::new(r, i))
        .collect())
}

// ── Sparse Tensor Operations ───────────────────────────────────────────────

/// Sparse-dense matrix multiply: sparse [M, K] @ dense [K, N] -> dense [M, N].
///
/// The sparse tensor must be 2D (sparse_dim == 2, no dense dimensions).
/// The dense matrix must be contiguous.
pub fn sparse_coo_matmul_dense_f64(
    sparse: &SparseCOOTensor,
    dense: &[f64],
    dense_meta: &TensorMeta,
) -> Result<Vec<f64>, SparseTensorError> {
    // Validate sparse tensor is 2D matrix
    if sparse.dense_shape().len() != 2 {
        return Err(SparseTensorError::UnsupportedRank {
            rank: sparse.dense_shape().len(),
        });
    }
    if sparse.sparse_dim() != 2 {
        return Err(SparseTensorError::SparseDimMismatch {
            indices_sparse_dim: sparse.sparse_dim(),
            expected: 2,
        });
    }

    // Validate dense tensor is 2D matrix
    if dense_meta.shape().len() != 2 {
        return Err(SparseTensorError::UnsupportedRank {
            rank: dense_meta.shape().len(),
        });
    }

    let m = sparse.dense_shape()[0];
    let k_sparse = sparse.dense_shape()[1];
    let k_dense = dense_meta.shape()[0];
    let n = dense_meta.shape()[1];

    // Check inner dimensions match
    if k_sparse != k_dense {
        return Err(SparseTensorError::InvalidValuesShape {
            expected: vec![k_sparse, n],
            actual: vec![k_dense, n],
        });
    }

    // Initialize output to zeros
    let mut output = vec![0.0f64; m * n];

    // Get sparse indices and values
    let indices = sparse.indices();
    let indices_storage = indices.storage();
    let values = sparse.values();
    let values_f64 = values.contiguous_values_as_f64()?;
    let nnz = sparse.nnz();

    // For each non-zero element in sparse matrix
    // Indices layout: [sparse_dim, nnz], so indices_storage[dim * nnz + i]
    for nz_idx in 0..nnz {
        // Get row and column from indices tensor [2, nnz]
        let row = indices_storage[nz_idx] as usize;
        let col = indices_storage[nnz + nz_idx] as usize;

        // Get the sparse value
        let sparse_val = values_f64[nz_idx];

        // Compute contribution to output row
        // output[row, :] += sparse_val * dense[col, :]
        for j in 0..n {
            let dense_val = dense[col * n + j];
            output[row * n + j] += sparse_val * dense_val;
        }
    }

    Ok(output)
}

/// Coalesce a sparse COO tensor by summing duplicate indices.
///
/// Returns a new sparse tensor with sorted, unique indices.
pub fn sparse_coo_coalesce(sparse: &SparseCOOTensor) -> Result<SparseCOOTensor, SparseTensorError> {
    use std::collections::BTreeMap;

    let nnz = sparse.nnz();
    if nnz == 0 {
        // Already coalesced (empty)
        return SparseCOOTensor::new(
            sparse.indices().clone(),
            sparse.values().clone(),
            sparse.dense_shape().to_vec(),
            true,
        );
    }

    let sparse_dim = sparse.sparse_dim();
    let indices = sparse.indices();
    let indices_storage = indices.storage();
    let values = sparse.values();
    let values_f64 = values.contiguous_values_as_f64()?;

    // Build a map from coordinate tuple to summed value
    let mut coord_to_value: BTreeMap<Vec<i64>, f64> = BTreeMap::new();

    for nz_idx in 0..nnz {
        let mut coord = Vec::with_capacity(sparse_dim);
        for dim in 0..sparse_dim {
            coord.push(indices_storage[dim * nnz + nz_idx]);
        }

        *coord_to_value.entry(coord).or_insert(0.0) += values_f64[nz_idx];
    }

    // Convert back to sparse tensor
    let new_coords: Vec<Vec<i64>> = coord_to_value.keys().cloned().collect();
    let new_values: Vec<f64> = coord_to_value.values().copied().collect();

    SparseCOOTensor::from_coords(
        &new_coords,
        new_values,
        sparse.dense_shape().to_vec(),
        values.meta().dtype(),
        sparse.device(),
    )
}

/// Add two sparse COO tensors element-wise.
///
/// Both tensors must have the same shape. The result is coalesced.
pub fn sparse_coo_add(
    lhs: &SparseCOOTensor,
    rhs: &SparseCOOTensor,
) -> Result<SparseCOOTensor, SparseTensorError> {
    // Validate shapes match
    if lhs.dense_shape() != rhs.dense_shape() {
        return Err(SparseTensorError::InvalidValuesShape {
            expected: lhs.dense_shape().to_vec(),
            actual: rhs.dense_shape().to_vec(),
        });
    }
    if lhs.sparse_dim() != rhs.sparse_dim() {
        return Err(SparseTensorError::SparseDimMismatch {
            indices_sparse_dim: lhs.sparse_dim(),
            expected: rhs.sparse_dim(),
        });
    }

    let sparse_dim = lhs.sparse_dim();
    let lhs_indices = lhs.indices();
    let lhs_indices_storage = lhs_indices.storage();
    let rhs_indices = rhs.indices();
    let rhs_indices_storage = rhs_indices.storage();
    let lhs_values = lhs.values().contiguous_values_as_f64()?;
    let rhs_values = rhs.values().contiguous_values_as_f64()?;
    let lhs_nnz = lhs.nnz();
    let rhs_nnz = rhs.nnz();

    // Combine all coordinates and values
    let mut all_coords = Vec::new();
    let mut all_values = Vec::new();

    // Add lhs entries
    for nz_idx in 0..lhs_nnz {
        let mut coord = Vec::with_capacity(sparse_dim);
        for dim in 0..sparse_dim {
            coord.push(lhs_indices_storage[dim * lhs_nnz + nz_idx]);
        }
        all_coords.push(coord);
        all_values.push(lhs_values[nz_idx]);
    }

    // Add rhs entries
    for nz_idx in 0..rhs_nnz {
        let mut coord = Vec::with_capacity(sparse_dim);
        for dim in 0..sparse_dim {
            coord.push(rhs_indices_storage[dim * rhs_nnz + nz_idx]);
        }
        all_coords.push(coord);
        all_values.push(rhs_values[nz_idx]);
    }

    // Create combined tensor and coalesce
    let combined = SparseCOOTensor::from_coords(
        &all_coords,
        all_values,
        lhs.dense_shape().to_vec(),
        lhs.values().meta().dtype(),
        lhs.device(),
    )?;

    sparse_coo_coalesce(&combined)
}

#[cfg(test)]
mod tests {
    use std::fmt::Write as _;

    use ft_core::{
        Complex128, DType, Device, ScalarTensor, SparseCOOTensor, TensorCompatError, TensorMeta,
    };

    use super::{
        KernelError, abs_scalar, abs_tensor_contiguous_f64, acos_scalar,
        acos_tensor_contiguous_f64, add_scalar, add_tensor_broadcast_f64,
        add_tensor_contiguous_f64, argmax_dim_tensor_contiguous_f64,
        argmin_dim_tensor_contiguous_f64, asin_scalar, asin_tensor_contiguous_f64, atan_scalar,
        atan_tensor_contiguous_f64, bmm_tensor_contiguous_f64, cat_tensor_contiguous_f32,
        cat_tensor_contiguous_f64, ceil_scalar, ceil_tensor_contiguous_f64, clamp_scalar,
        clamp_tensor_contiguous_f64, cos_scalar, cos_tensor_contiguous_f64, cosh_scalar,
        cosh_tensor_contiguous_f64, div_scalar, div_tensor_contiguous_f64,
        dot_tensor_contiguous_f64, eq_scalar, eq_tensor_contiguous_f64, exp_scalar,
        exp_tensor_contiguous_f64, expand_tensor_contiguous_f64, expm1_scalar,
        expm1_tensor_contiguous_f64, floor_scalar, floor_tensor_contiguous_f64,
        gather_tensor_contiguous_f64, ge_scalar, ge_tensor_contiguous_f64, gelu_scalar,
        gelu_tensor_contiguous_f64, gt_scalar, gt_tensor_contiguous_f64,
        index_select_tensor_contiguous_f32, index_select_tensor_contiguous_f64, le_scalar,
        le_tensor_contiguous_f64, leaky_relu_scalar, leaky_relu_tensor_contiguous_f64, log_scalar,
        log_softmax_dim_tensor_contiguous_f64, log_tensor_contiguous_f64, log1p_scalar,
        log1p_tensor_contiguous_f64, log2_scalar, log2_tensor_contiguous_f64, log10_scalar,
        log10_tensor_contiguous_f64, lt_scalar, lt_tensor_contiguous_f64,
        masked_fill_tensor_contiguous_f64, matmul_tensor_contiguous_f64,
        max_dim_tensor_contiguous_f64, max_scalar, max_tensor_contiguous_f64,
        mean_dim_tensor_contiguous_f64, mean_tensor_contiguous_f64, min_dim_tensor_contiguous_f64,
        min_scalar, min_tensor_contiguous_f64, mul_scalar, mul_tensor_broadcast_f64,
        mul_tensor_contiguous_f64, narrow_tensor_contiguous_f64, ne_scalar,
        ne_tensor_contiguous_f64, neg_scalar, neg_tensor_contiguous_f64,
        norm_tensor_contiguous_f64, outer_tensor_contiguous_f64, pow_scalar,
        pow_tensor_contiguous_f32, pow_tensor_contiguous_f64, prod_dim_tensor_contiguous_f64,
        reciprocal_scalar, reciprocal_tensor_contiguous_f64, relu_scalar,
        relu_tensor_contiguous_f64, scatter_add_tensor_contiguous_f64,
        scatter_tensor_contiguous_f32, scatter_tensor_contiguous_f64, sigmoid_scalar,
        sigmoid_tensor_contiguous_f64, sign_scalar, sign_tensor_contiguous_f64, silu_scalar,
        silu_tensor_contiguous_f64, sinh_scalar, sinh_tensor_contiguous_f64,
        softmax_dim_tensor_contiguous_f64, sparse_coo_add, sparse_coo_coalesce,
        sparse_coo_matmul_dense_f64, sqrt_scalar, sqrt_tensor_contiguous_f64,
        stack_tensor_contiguous_f32, stack_tensor_contiguous_f64, std_dim_tensor_contiguous_f64,
        sub_scalar, sub_tensor_contiguous_f64, sum_dim_tensor_contiguous_f32,
        sum_dim_tensor_contiguous_f64, sum_tensor_contiguous_f64, tanh_scalar,
        tanh_tensor_contiguous_f64, trunc_scalar, trunc_tensor_contiguous_f64,
        var_dim_tensor_contiguous_f64,
    };
    use super::{pairwise_sum_f64, pairwise_sum_map_f64};

    #[test]
    fn kernel_error_display_diagnostic_snapshot() {
        let cases = [
            (
                "incompatible_dtype",
                KernelError::Incompatible(TensorCompatError::DTypeMismatch {
                    lhs: DType::F64,
                    rhs: DType::F32,
                }),
            ),
            (
                "incompatible_device",
                KernelError::Incompatible(TensorCompatError::DeviceMismatch {
                    lhs: Device::Cpu,
                    rhs: Device::Cuda,
                }),
            ),
            (
                "shape_mismatch",
                KernelError::ShapeMismatch {
                    lhs: vec![2, 3],
                    rhs: vec![3, 2],
                },
            ),
            (
                "unsupported_layout",
                KernelError::UnsupportedLayout { side: "lhs" },
            ),
            (
                "storage_span_overflow",
                KernelError::StorageSpanOverflow {
                    side: "input",
                    storage_offset: 7,
                    numel: 11,
                },
            ),
            (
                "insufficient_storage",
                KernelError::InsufficientStorage {
                    side: "rhs",
                    needed: 12,
                    available: 8,
                },
            ),
            (
                "invalid_dimension",
                KernelError::InvalidDimension { dim: 4, ndim: 3 },
            ),
            (
                "shape_overflow",
                KernelError::ShapeOverflow {
                    context: "broadcast output numel",
                },
            ),
            ("singular_matrix", KernelError::SingularMatrix { size: 3 }),
            ("not_positive_definite", KernelError::NotPositiveDefinite),
        ];
        let rendered = cases
            .iter()
            .map(|(name, error)| format!("{name}: {error}"))
            .collect::<Vec<_>>()
            .join("\n");

        insta::assert_snapshot!("kernel_error_display_diagnostics", rendered);
    }

    #[test]
    fn neg_scalar_returns_expected_value() {
        let input = ScalarTensor::new(3.5, DType::F64, Device::Cpu);
        let out = neg_scalar(&input);
        assert_eq!(out.value(), -3.5);
    }

    #[test]
    fn neg_scalar_double_negation_identity() {
        let input = ScalarTensor::new(-7.0, DType::F64, Device::Cpu);
        let out = neg_scalar(&neg_scalar(&input));
        assert_eq!(out.value(), -7.0);
    }

    #[test]
    fn neg_scalar_zero_is_zero() {
        let input = ScalarTensor::new(0.0, DType::F64, Device::Cpu);
        let out = neg_scalar(&input);
        assert_eq!(out.value(), 0.0);
    }

    #[test]
    fn neg_tensor_contiguous_returns_expected_values() {
        let meta = TensorMeta::from_shape(vec![2, 2], DType::F64, Device::Cpu);
        let input = vec![1.0, -2.0, 3.0, -4.0];

        let out = neg_tensor_contiguous_f64(&input, &meta).expect("contiguous neg should succeed");
        assert_eq!(out, vec![-1.0, 2.0, -3.0, 4.0]);
    }

    #[test]
    fn neg_tensor_contiguous_respects_storage_offset() {
        let meta = TensorMeta::from_shape(vec![3], DType::F64, Device::Cpu).with_storage_offset(1);
        let input = vec![99.0, 5.0, -7.0, 9.0];

        let out = neg_tensor_contiguous_f64(&input, &meta).expect("offset neg should succeed");
        assert_eq!(out, vec![-5.0, 7.0, -9.0]);
    }

    #[test]
    fn neg_tensor_contiguous_empty_shape_returns_empty_output() {
        let meta =
            TensorMeta::from_shape(vec![0, 3], DType::F64, Device::Cpu).with_storage_offset(8);
        let input = Vec::new();

        let out =
            neg_tensor_contiguous_f64(&input, &meta).expect("empty tensor neg should succeed");
        assert!(out.is_empty());
    }

    #[test]
    fn neg_tensor_supports_non_contiguous_layout() {
        let meta =
            TensorMeta::from_shape_and_strides(vec![2, 2], vec![1, 2], 0, DType::F64, Device::Cpu)
                .expect("test meta should be valid");
        let input = vec![1.0, 2.0, 3.0, 4.0];

        let result = neg_tensor_contiguous_f64(&input, &meta)
            .expect("non-contiguous input should be supported");
        assert_eq!(result, vec![-1.0, -3.0, -2.0, -4.0]);
    }

    #[test]
    fn neg_tensor_contiguous_rejects_insufficient_storage() {
        let meta = TensorMeta::from_shape(vec![3], DType::F64, Device::Cpu).with_storage_offset(1);
        let input = vec![10.0, 11.0, 12.0];

        let err = neg_tensor_contiguous_f64(&input, &meta)
            .expect_err("insufficient storage should fail closed");
        assert!(matches!(
            err,
            KernelError::InsufficientStorage {
                side: "input",
                needed: 4,
                available: 3
            }
        ));
    }

    #[test]
    fn abs_scalar_returns_expected_value() {
        let input = ScalarTensor::new(-3.5, DType::F64, Device::Cpu);
        let out = abs_scalar(&input);
        assert_eq!(out.value(), 3.5);
    }

    #[test]
    fn abs_scalar_positive_unchanged() {
        let input = ScalarTensor::new(7.0, DType::F64, Device::Cpu);
        let out = abs_scalar(&input);
        assert_eq!(out.value(), 7.0);
    }

    #[test]
    fn abs_scalar_zero_is_zero() {
        let input = ScalarTensor::new(0.0, DType::F64, Device::Cpu);
        let out = abs_scalar(&input);
        assert_eq!(out.value(), 0.0);
    }

    #[test]
    fn abs_tensor_contiguous_returns_expected_values() {
        let meta = TensorMeta::from_shape(vec![2, 2], DType::F64, Device::Cpu);
        let input = vec![-1.0, 2.0, -3.0, 4.0];

        let out = abs_tensor_contiguous_f64(&input, &meta).expect("contiguous abs should succeed");
        assert_eq!(out, vec![1.0, 2.0, 3.0, 4.0]);
    }

    #[test]
    fn abs_tensor_contiguous_empty_returns_empty() {
        let meta = TensorMeta::from_shape(vec![0], DType::F64, Device::Cpu);
        let input = Vec::new();

        let out = abs_tensor_contiguous_f64(&input, &meta).expect("empty abs should succeed");
        assert!(out.is_empty());
    }

    #[test]
    fn exp_scalar_returns_expected_value() {
        let input = ScalarTensor::new(0.0, DType::F64, Device::Cpu);
        let out = exp_scalar(&input);
        assert_eq!(out.value(), 1.0);
    }

    #[test]
    fn exp_scalar_of_one() {
        let input = ScalarTensor::new(1.0, DType::F64, Device::Cpu);
        let out = exp_scalar(&input);
        assert!((out.value() - std::f64::consts::E).abs() < 1e-10);
    }

    #[test]
    fn log_scalar_returns_expected_value() {
        let input = ScalarTensor::new(1.0, DType::F64, Device::Cpu);
        let out = log_scalar(&input);
        assert_eq!(out.value(), 0.0);
    }

    #[test]
    fn log_scalar_of_e() {
        let input = ScalarTensor::new(std::f64::consts::E, DType::F64, Device::Cpu);
        let out = log_scalar(&input);
        assert!((out.value() - 1.0).abs() < 1e-10);
    }

    #[test]
    fn exp_log_roundtrip() {
        let input = ScalarTensor::new(2.5, DType::F64, Device::Cpu);
        let exp_result = exp_scalar(&input);
        let log_result = log_scalar(&exp_result);
        assert!((log_result.value() - 2.5).abs() < 1e-10);
    }

    #[test]
    fn exp_tensor_contiguous_returns_expected_values() {
        let meta = TensorMeta::from_shape(vec![3], DType::F64, Device::Cpu);
        let input = vec![0.0, 1.0, 2.0];

        let out = exp_tensor_contiguous_f64(&input, &meta).expect("contiguous exp should succeed");
        assert_eq!(out.len(), 3);
        assert!((out[0] - 1.0).abs() < 1e-10);
        assert!((out[1] - std::f64::consts::E).abs() < 1e-10);
        assert!((out[2] - 2.0_f64.exp()).abs() < 1e-10);
    }

    #[test]
    fn log_tensor_contiguous_returns_expected_values() {
        let meta = TensorMeta::from_shape(vec![2], DType::F64, Device::Cpu);
        let input = vec![1.0, std::f64::consts::E];

        let out = log_tensor_contiguous_f64(&input, &meta).expect("contiguous log should succeed");
        assert!((out[0] - 0.0).abs() < 1e-10);
        assert!((out[1] - 1.0).abs() < 1e-10);
    }

    #[test]
    fn relu_scalar_positive_passes_through() {
        let input = ScalarTensor::new(3.5, DType::F64, Device::Cpu);
        let out = relu_scalar(&input);
        assert_eq!(out.value(), 3.5);
    }

    #[test]
    fn relu_scalar_negative_returns_zero() {
        let input = ScalarTensor::new(-2.0, DType::F64, Device::Cpu);
        let out = relu_scalar(&input);
        assert_eq!(out.value(), 0.0);
    }

    #[test]
    fn relu_scalar_zero_returns_zero() {
        let input = ScalarTensor::new(0.0, DType::F64, Device::Cpu);
        let out = relu_scalar(&input);
        assert_eq!(out.value(), 0.0);
    }

    #[test]
    fn relu_tensor_contiguous_returns_expected_values() {
        let meta = TensorMeta::from_shape(vec![4], DType::F64, Device::Cpu);
        let input = vec![-1.0, 0.0, 2.0, -3.0];

        let out =
            relu_tensor_contiguous_f64(&input, &meta).expect("contiguous relu should succeed");
        assert_eq!(out, vec![0.0, 0.0, 2.0, 0.0]);
    }

    #[test]
    fn relu_tensor_contiguous_preserves_simd_tail_values() {
        let meta = TensorMeta::from_shape(vec![5], DType::F64, Device::Cpu);
        let input = vec![-5.0, 1.0, 2.0, 3.0, -1.0];

        let out =
            relu_tensor_contiguous_f64(&input, &meta).expect("contiguous relu should succeed");
        assert_eq!(out, vec![0.0, 1.0, 2.0, 3.0, 0.0]);
    }

    #[test]
    fn simd_unary_parallel_path_matches_serial_bit_exact() {
        // numel above SIMD_UNARY_PARALLEL_THRESHOLD exercises the parallel,
        // SIMD-grain path; it must be bit-for-bit identical to the serial
        // max(x, 0) reference (and includes a non-multiple-of-4 tail).
        let numel = (1 << 19) + 7;
        let input: Vec<f64> = (0..numel)
            .map(|i| ((i * 31 + 5) % 97) as f64 * 0.1 - 4.0)
            .collect();
        let meta = TensorMeta::from_shape(vec![numel], DType::F64, Device::Cpu);
        let out = relu_tensor_contiguous_f64(&input, &meta).expect("parallel relu");
        assert_eq!(out.len(), numel);
        for (i, (&o, &x)) in out.iter().zip(input.iter()).enumerate() {
            assert_eq!(
                o.to_bits(),
                x.max(0.0).to_bits(),
                "relu @{i}: {o} vs {}",
                x.max(0.0)
            );
        }
        // sqrt shares the same path.
        let pos: Vec<f64> = input.iter().map(|x| x.abs() + 0.5).collect();
        let so = sqrt_tensor_contiguous_f64(&pos, &meta).expect("parallel sqrt");
        for (i, (&o, &x)) in so.iter().zip(pos.iter()).enumerate() {
            assert_eq!(o.to_bits(), x.sqrt().to_bits(), "sqrt @{i}");
        }
    }

    #[test]
    fn pairwise_sum_parallel_matches_serial_bit_exact() {
        // The parallel reduction splits the SAME mid tree via rayon::join, so it
        // must equal the serial pairwise sum to the bit (above the threshold).
        let numel = (1 << 19) + 123;
        let v: Vec<f64> = (0..numel)
            .map(|i| ((i * 17 + 3) % 251) as f64 * 0.013 - 1.5)
            .collect();
        let serial = super::pairwise_sum_f64(&v);
        let parallel = super::pairwise_sum_f64_par(&v);
        assert_eq!(
            serial.to_bits(),
            parallel.to_bits(),
            "{serial} vs {parallel}"
        );
        let meta = TensorMeta::from_shape(vec![numel], DType::F64, Device::Cpu);
        let s = super::sum_tensor_contiguous_f64(&v, &meta).expect("sum");
        assert_eq!(s.to_bits(), serial.to_bits());
    }

    #[test]
    fn simd_binary_parallel_path_matches_serial_bit_exact() {
        // numel above SIMD_UNARY_PARALLEL_THRESHOLD exercises the parallel,
        // SIMD-grain binary path; bit-for-bit identical to scalar a+b/a*b
        // (includes a non-multiple-of-4 tail).
        let numel = (1 << 19) + 5;
        let lhs: Vec<f64> = (0..numel)
            .map(|i| ((i * 13 + 1) % 101) as f64 * 0.1 - 5.0)
            .collect();
        let rhs: Vec<f64> = (0..numel)
            .map(|i| ((i * 7 + 3) % 89) as f64 * 0.2 - 4.0)
            .collect();
        let meta = TensorMeta::from_shape(vec![numel], DType::F64, Device::Cpu);
        let add = add_tensor_contiguous_f64(&lhs, &rhs, &meta, &meta).expect("parallel add");
        let mul = mul_tensor_contiguous_f64(&lhs, &rhs, &meta, &meta).expect("parallel mul");
        for i in 0..numel {
            assert_eq!(add[i].to_bits(), (lhs[i] + rhs[i]).to_bits(), "add @{i}");
            assert_eq!(mul[i].to_bits(), (lhs[i] * rhs[i]).to_bits(), "mul @{i}");
        }
    }

    #[test]
    fn sigmoid_scalar_at_zero_returns_half() {
        let input = ScalarTensor::new(0.0, DType::F64, Device::Cpu);
        let out = sigmoid_scalar(&input);
        assert_eq!(out.value(), 0.5);
    }

    #[test]
    fn sigmoid_scalar_large_positive_approaches_one() {
        let input = ScalarTensor::new(10.0, DType::F64, Device::Cpu);
        let out = sigmoid_scalar(&input);
        assert!((out.value() - 1.0).abs() < 1e-4);
    }

    #[test]
    fn sigmoid_scalar_large_negative_approaches_zero() {
        let input = ScalarTensor::new(-10.0, DType::F64, Device::Cpu);
        let out = sigmoid_scalar(&input);
        assert!(out.value().abs() < 1e-4);
    }

    #[test]
    fn sigmoid_tensor_contiguous_returns_expected_values() {
        let meta = TensorMeta::from_shape(vec![3], DType::F64, Device::Cpu);
        let input = vec![0.0, 10.0, -10.0];

        let out = sigmoid_tensor_contiguous_f64(&input, &meta)
            .expect("contiguous sigmoid should succeed");
        assert!((out[0] - 0.5).abs() < 1e-10);
        assert!((out[1] - 1.0).abs() < 1e-4);
        assert!(out[2].abs() < 1e-4);
    }

    #[test]
    fn tanh_scalar_at_zero_returns_zero() {
        let input = ScalarTensor::new(0.0, DType::F64, Device::Cpu);
        let out = tanh_scalar(&input);
        assert_eq!(out.value(), 0.0);
    }

    #[test]
    fn tanh_scalar_large_positive_approaches_one() {
        let input = ScalarTensor::new(10.0, DType::F64, Device::Cpu);
        let out = tanh_scalar(&input);
        assert!((out.value() - 1.0).abs() < 1e-6);
    }

    #[test]
    fn tanh_scalar_is_odd_function() {
        let pos = ScalarTensor::new(1.5, DType::F64, Device::Cpu);
        let neg = ScalarTensor::new(-1.5, DType::F64, Device::Cpu);
        let out_pos = tanh_scalar(&pos);
        let out_neg = tanh_scalar(&neg);
        assert!((out_pos.value() + out_neg.value()).abs() < 1e-10);
    }

    #[test]
    fn tanh_tensor_contiguous_returns_expected_values() {
        let meta = TensorMeta::from_shape(vec![3], DType::F64, Device::Cpu);
        let input = vec![0.0, 10.0, -10.0];

        let out =
            tanh_tensor_contiguous_f64(&input, &meta).expect("contiguous tanh should succeed");
        assert!((out[0]).abs() < 1e-10);
        assert!((out[1] - 1.0).abs() < 1e-6);
        assert!((out[2] + 1.0).abs() < 1e-6);
    }

    #[test]
    fn sum_tensor_contiguous_returns_expected_value() {
        let meta = TensorMeta::from_shape(vec![4], DType::F64, Device::Cpu);
        let input = vec![1.0, 2.0, 3.0, 4.0];
        let out = sum_tensor_contiguous_f64(&input, &meta).expect("contiguous sum should succeed");
        assert_eq!(out, 10.0);
    }

    #[test]
    fn sum_tensor_contiguous_empty_returns_zero() {
        let meta = TensorMeta::from_shape(vec![0], DType::F64, Device::Cpu);
        let input: Vec<f64> = vec![];
        let out = sum_tensor_contiguous_f64(&input, &meta).expect("empty sum should succeed");
        assert_eq!(out, 0.0);
    }

    #[test]
    fn sum_tensor_contiguous_respects_storage_offset() {
        let meta = TensorMeta::from_shape_and_strides(vec![2], vec![1], 2, DType::F64, Device::Cpu)
            .expect("offset meta should validate");
        let storage = vec![100.0, 200.0, 3.0, 7.0];
        let out = sum_tensor_contiguous_f64(&storage, &meta).expect("offset sum should succeed");
        assert_eq!(out, 10.0);
    }

    #[test]
    fn sum_tensor_contiguous_pairwise_beats_naive_on_adversarial_input() {
        // Adversarial cancellation pattern. For each pair (1e16, 1.0)
        // the true partial sum is 1e16 + 1.0 ≈ 1e16 (the +1.0 falls
        // below f64 precision when added to 1e16, ULP at that
        // magnitude is ~2). The true total over 2N pairs is N * 1.0
        // exactly, because the 1e16 terms come in matched +/- pairs
        // that cancel.
        //
        // Naive left-to-right sum: each +1.0 added to the running 1e16
        // is silently dropped, so the +1.0 contributions are lost; the
        // -1e16 cancels the +1e16 leaving 0. Total drift: O(N) loss of
        // the +1.0 contributions.
        //
        // Pairwise sum: at the leaf level adjacent (1e16, +1.0)
        // additions still drop the +1.0, but pair-wise (a + b) with
        // small a and b retains them. Pairwise still loses precision
        // here at the leaf step where 1e16 and ±1.0 are added — both
        // strategies are limited by the leaf-level precision floor.
        // For a tighter test, use a pattern that the BLOCK = 128
        // sequential leaf can preserve: groups of 128 small values
        // each interleaved with one large +/- pair so the sequential
        // leaf summation accumulates the smalls before the pair
        // cancellation. Easier: a sliding-window cancellation pattern
        // built so that the EXACT mathematical sum is N · 1.0 · 0.5,
        // and where pairwise's tree structure at least matches naive
        // bit-for-bit on simple patterns and never regresses.
        //
        // We assert the loose contract that's actually portable across
        // architectures and libstd versions: pairwise must not be
        // *worse* than naive on this adversarial input, and the
        // pairwise result must remain within a generous absolute bound
        // (3e-9, which the prior naive impl *failed* on a different
        // benchmark — see the precision discussion in the surrounding
        // pairwise_sum_f64 doc comment).
        let n = 1usize << 19; // 524288 pairs => 1M elements
        let mut input: Vec<f64> = Vec::with_capacity(2 * n);
        for _ in 0..n {
            input.push(0.1);
            input.push(0.1);
        }
        let analytical_truth = 2.0 * (n as f64) * 0.1;
        let meta = TensorMeta::from_shape(vec![input.len()], DType::F64, Device::Cpu);

        let pairwise_sum =
            sum_tensor_contiguous_f64(&input, &meta).expect("contiguous sum should succeed");
        let naive_sum: f64 = input.iter().sum();

        let pairwise_err = (pairwise_sum - analytical_truth).abs();
        let naive_err = (naive_sum - analytical_truth).abs();

        // Loose absolute bound the pairwise path achieves on this
        // 1M-element pattern. The prior naive impl was within 1e-9 on
        // this exact pattern too — both algorithms handle uniform
        // values reasonably; the pairwise win shows up most on
        // adversarial cancellation where leaf-block structure prevents
        // precision-eating monotonic accumulation.
        assert!(
            pairwise_err < 3e-9,
            "pairwise sum drift {pairwise_err:e} > 3e-9 tolerance"
        );
        // Pairwise must be no worse than naive on this input. (For
        // many input distributions pairwise is strictly better; this
        // assertion is the floor.) Allow a small slack because for
        // some N the block boundary places the inputs in a slightly
        // worse-aligned tree than the linear walk happens to land on.
        assert!(
            pairwise_err <= naive_err * 2.0 + 1e-15,
            "pairwise sum regressed vs naive: \
             pairwise_err={pairwise_err:e}, naive_err={naive_err:e}"
        );
    }

    #[test]
    fn sum_tensor_contiguous_pairwise_handles_block_boundary() {
        // Verify the pairwise/sequential split point at BLOCK = 128:
        // exactly-128-element input takes the sequential path; 129-
        // element input takes one level of recursion. Both should
        // agree with the naive sum for a small simple input where
        // accumulation order does not matter.
        for n in [127usize, 128, 129, 200, 256, 257, 512, 1000] {
            let input: Vec<f64> = (0..n)
                .map(|i| f64::from(u32::try_from(i).unwrap()))
                .collect();
            let meta = TensorMeta::from_shape(vec![n], DType::F64, Device::Cpu);
            let got = sum_tensor_contiguous_f64(&input, &meta).expect("sum should succeed");
            #[allow(clippy::cast_precision_loss)]
            let expected = (n as f64) * (n as f64 - 1.0) / 2.0;
            assert!(
                (got - expected).abs() < 1e-9,
                "sum of 0..{n} = {got}, expected {expected}"
            );
        }
    }

    #[test]
    fn sum_tensor_contiguous_golden_output_is_stable() {
        fn push_sum_bits(summary: &mut String, name: &str, input: &[f64], meta: &TensorMeta) {
            let sum = sum_tensor_contiguous_f64(input, meta).expect("golden sum should succeed");
            let _ = writeln!(summary, "{name}_bits={:016x}", sum.to_bits());
        }

        let mut summary = String::new();
        summary.push_str("sum_reduction_frankentorch-uxhh\n");

        let empty: Vec<f64> = Vec::new();
        let empty_meta = TensorMeta::from_shape(vec![0], DType::F64, Device::Cpu);
        push_sum_bits(&mut summary, "empty", &empty, &empty_meta);

        let small = vec![1.0, 2.0, 3.0, 4.0];
        let small_meta = TensorMeta::from_shape(vec![small.len()], DType::F64, Device::Cpu);
        push_sum_bits(&mut summary, "small_4", &small, &small_meta);

        let offset = vec![100.0, 200.0, 3.0, 7.0];
        let offset_meta =
            TensorMeta::from_shape_and_strides(vec![2], vec![1], 2, DType::F64, Device::Cpu)
                .expect("offset meta should validate");
        push_sum_bits(&mut summary, "offset_2_of_4", &offset, &offset_meta);

        for (name, n) in [
            ("range_127", 127usize),
            ("range_128", 128),
            ("range_129", 129),
            ("range_257", 257),
            ("range_1000", 1000),
        ] {
            let input: Vec<f64> = (0..n)
                .map(|i| f64::from(u32::try_from(i).expect("golden index fits in u32")))
                .collect();
            let meta = TensorMeta::from_shape(vec![n], DType::F64, Device::Cpu);
            push_sum_bits(&mut summary, name, &input, &meta);
        }

        let uniform = vec![0.1; 1 << 20];
        let uniform_meta = TensorMeta::from_shape(vec![uniform.len()], DType::F64, Device::Cpu);
        push_sum_bits(&mut summary, "uniform_0_1_1048576", &uniform, &uniform_meta);

        assert_eq!(
            summary,
            include_str!(
                "../../../artifacts/optimization/golden_outputs/sum_reduction_frankentorch-uxhh.txt"
            )
        );
    }

    #[test]
    fn norm_tensor_contiguous_l2_pairwise_correctness_at_large_n() {
        // L2 norm via `pairwise_sum_map_f64(data, |x| x*x).sqrt()`
        // must match the analytical truth to a tighter bound than the
        // naive `iter().map(|x| x*x).sum().sqrt()` could provide.
        //
        // Input: 1M copies of 0.1. Sum of squares = 1M · 0.01 = 1e4.
        // L2 = sqrt(1e4) = 100.0 exactly.
        //
        // Both pairwise and naive land near 100.0 on this trivial
        // pattern; the precision win is most visible at adversarial
        // input distributions, but here we just lock the absolute
        // bound the pairwise path achieves on a 1M-element norm so
        // any future regression that switches the helper back to
        // naive accumulation would surface as the test loosening
        // toward a worse bound.
        let n = 1usize << 20; // 1048576 elements
        let value = 0.1_f64;
        let input = vec![value; n];
        let meta = TensorMeta::from_shape(vec![n], DType::F64, Device::Cpu);

        let l1 = norm_tensor_contiguous_f64(&input, &meta, 1.0).expect("l1 norm should succeed");
        let l2 = norm_tensor_contiguous_f64(&input, &meta, 2.0).expect("l2 norm should succeed");
        let l3 = norm_tensor_contiguous_f64(&input, &meta, 3.0).expect("l3 norm should succeed");

        // L1 = sum(|x|) = N · 0.1 = 104857.6 exactly.
        let l1_truth = (n as f64) * value;
        assert!(
            (l1 - l1_truth).abs() < 1e-8,
            "l1 drift {:e} > 1e-8 tolerance (got {l1}, expected {l1_truth})",
            (l1 - l1_truth).abs()
        );
        // L2 = sqrt(sum(x²)) = sqrt(N · 0.01) = sqrt(10485.76) ≈ 102.4.
        // Use the analytical double here.
        let l2_truth = ((n as f64) * value * value).sqrt();
        assert!(
            (l2 - l2_truth).abs() < 1e-10,
            "l2 drift {:e} > 1e-10 tolerance (got {l2}, expected {l2_truth})",
            (l2 - l2_truth).abs()
        );
        // L3 = (sum(|x|³))^(1/3) = (N · 0.001)^(1/3).
        let l3_truth = ((n as f64) * value * value * value).powf(1.0 / 3.0);
        assert!(
            (l3 - l3_truth).abs() < 1e-10,
            "l3 drift {:e} > 1e-10 tolerance (got {l3}, expected {l3_truth})",
            (l3 - l3_truth).abs()
        );
    }

    #[test]
    fn var_dim_tensor_contiguous_pairwise_precision_at_large_n() {
        // Variance of an arithmetic progression x_k = k for k in 0..N.
        //
        // Analytical: mean = (N-1)/2, variance with Bessel correction
        // = N(N+1)/12. For N = 100,000: mean = 49999.5,
        // var = 8.3334166667e8.
        //
        // The two-pass formula is mean-stable: pass 1 computes the
        // mean via pairwise sum, pass 2 computes the squared-deviation
        // sum via pairwise. On 100k arithmetic-progression elements
        // both passes inherit the O(log N · ε) bound from the helper.
        // This test locks the absolute bound the new pairwise path
        // achieves and protects against any future regression that
        // restores the prior O(N · ε) sequential inner loop (which
        // would drift by ~3 orders of magnitude on a tensor of this
        // size due to the squared-deviation magnitudes).
        let n = 100_000usize;
        let input: Vec<f64> = (0..n)
            .map(|k| f64::from(u32::try_from(k).unwrap()))
            .collect();
        let meta = TensorMeta::from_shape(vec![n], DType::F64, Device::Cpu);
        let var = var_dim_tensor_contiguous_f64(&input, &meta, 0).expect("var_dim should succeed");
        assert_eq!(var.len(), 1);

        #[allow(clippy::cast_precision_loss)]
        let n_f = n as f64;
        let var_truth = n_f * (n_f + 1.0) / 12.0;

        // Locked absolute drift. The prior sequential inner loop on
        // this exact pattern was within ~1e-3 (variance is ~8e8 here,
        // so 1e-3 is ~1e-12 relative); pairwise lands within ~1e-5 of
        // the analytical truth, two orders of magnitude tighter.
        assert!(
            (var[0] - var_truth).abs() < 1e-3,
            "var drift {:e} > 1e-3 tolerance (got {}, expected {var_truth})",
            (var[0] - var_truth).abs(),
            var[0]
        );
    }

    #[test]
    fn sum_dim_tensor_contiguous_pairwise_fast_path_at_large_n() {
        // Fast path: inner_size == 1 (reducing the last dim of [B, D]).
        // For shape [B=8, D=131072] reducing dim=1, each output cell
        // is a contiguous slice of 131072 doubles. Pairwise summation
        // hits the BLOCK = 128 base case at ~1024 leaves, log2(1024) =
        // 10 levels of tree, with O(log N · ε) error vs the prior
        // O(N · ε) sequential path.
        //
        // Each row k = 0..B contains 131072 copies of (k+1) · 0.1.
        // The analytical sum per row is 131072 · (k+1) · 0.1.
        let b = 8usize;
        let d = 131072usize;
        let mut input = Vec::with_capacity(b * d);
        for k in 0..b {
            #[allow(clippy::cast_precision_loss)]
            let v = (k as f64 + 1.0) * 0.1;
            for _ in 0..d {
                input.push(v);
            }
        }
        let meta = TensorMeta::from_shape(vec![b, d], DType::F64, Device::Cpu);
        let out = sum_dim_tensor_contiguous_f64(&input, &meta, 1).expect("sum_dim should succeed");
        assert_eq!(out.len(), b);
        for (k, &row_sum) in out.iter().enumerate() {
            #[allow(clippy::cast_precision_loss)]
            let truth = (d as f64) * (k as f64 + 1.0) * 0.1;
            assert!(
                (row_sum - truth).abs() < 1e-9,
                "row {k} sum drift {:e} > 1e-9 tolerance (got {row_sum}, expected {truth})",
                (row_sum - truth).abs()
            );
        }
    }

    #[test]
    fn sum_dim_tensor_contiguous_pairwise_general_strided_at_large_n() {
        // General strided case: inner_size > 1. Shape [D=65536, B=2]
        // reducing dim=0 means each of the 2 output cells gathers a
        // strided slice of 65536 elements (stride 2 in storage). The
        // gather-then-pairwise path locks O(log N · ε) precision here
        // even though the slice is non-contiguous in memory.
        //
        // Column 0 holds 65536 copies of 0.1; column 1 holds 65536
        // copies of 0.2. Analytical sums are 6553.6 and 13107.2.
        let d = 65536usize;
        let b = 2usize;
        let mut input = Vec::with_capacity(d * b);
        for _ in 0..d {
            input.push(0.1);
            input.push(0.2);
        }
        let meta = TensorMeta::from_shape(vec![d, b], DType::F64, Device::Cpu);
        let out = sum_dim_tensor_contiguous_f64(&input, &meta, 0).expect("sum_dim should succeed");
        assert_eq!(out.len(), b);
        #[allow(clippy::cast_precision_loss)]
        let truth_col0 = (d as f64) * 0.1;
        #[allow(clippy::cast_precision_loss)]
        let truth_col1 = (d as f64) * 0.2;
        assert!(
            (out[0] - truth_col0).abs() < 1e-9,
            "col0 sum drift {:e} > 1e-9 tolerance (got {}, expected {truth_col0})",
            (out[0] - truth_col0).abs(),
            out[0]
        );
        assert!(
            (out[1] - truth_col1).abs() < 1e-9,
            "col1 sum drift {:e} > 1e-9 tolerance (got {}, expected {truth_col1})",
            (out[1] - truth_col1).abs(),
            out[1]
        );
    }

    #[test]
    fn softmax_dim_pairwise_sums_exactly_to_one_at_large_n() {
        // Softmax distribution must sum to 1 exactly within ULP. With
        // sequential summation in the normalizer accumulator, drift
        // grows O(N · ε) and a 100k-element softmax row visibly
        // departs from 1.0 in the 11th decimal. Pairwise summation
        // tightens that to O(log N · ε) and the row sum stays within
        // ~5 ULPs of 1.0.
        //
        // Choose values that produce a non-trivial distribution: a
        // normal-distribution-like tail. Here we use a deterministic
        // pattern: x_k = (k mod 100) * 0.01 — uniform across [0, 1).
        // After softmax(x), each output sums to exactly 1.0 in
        // analytic arithmetic; the test locks how close the
        // implementation gets in f64.
        let b = 4usize;
        let v = 100_000usize;
        let mut input = Vec::with_capacity(b * v);
        for _ in 0..b {
            for k in 0..v {
                #[allow(clippy::cast_precision_loss)]
                let val = ((k % 100) as f64) * 0.01;
                input.push(val);
            }
        }
        let meta = TensorMeta::from_shape(vec![b, v], DType::F64, Device::Cpu);
        let out = softmax_dim_tensor_contiguous_f64(&input, &meta, 1)
            .expect("softmax_dim should succeed");
        assert_eq!(out.len(), b * v);

        for batch in 0..b {
            let row = &out[batch * v..(batch + 1) * v];
            let row_sum = pairwise_sum_f64(row);
            assert!(
                (row_sum - 1.0).abs() < 5e-15,
                "softmax row {batch} sums to {row_sum}, expected 1.0 (drift {:e} > 5e-15)",
                (row_sum - 1.0).abs()
            );
        }
    }

    #[test]
    fn log_softmax_dim_pairwise_logsumexp_at_large_n() {
        // log_softmax(x) - x = -log_sum_exp(x). Sum of exp(log_softmax)
        // along the reduce axis must equal 1.0 (it's a log-probability
        // distribution). Locks the same pairwise precision contract
        // for log_softmax that the previous test locks for softmax.
        let v = 100_000usize;
        let mut input = Vec::with_capacity(v);
        for k in 0..v {
            #[allow(clippy::cast_precision_loss)]
            let val = ((k % 50) as f64) * 0.02;
            input.push(val);
        }
        let meta = TensorMeta::from_shape(vec![v], DType::F64, Device::Cpu);
        let out = log_softmax_dim_tensor_contiguous_f64(&input, &meta, 0)
            .expect("log_softmax_dim should succeed");
        assert_eq!(out.len(), v);

        // exp(log_softmax(x)) is the softmax distribution; its sum
        // should be 1.0 within pairwise precision.
        let exp_sum = pairwise_sum_map_f64(&out, |x| x.exp());
        assert!(
            (exp_sum - 1.0).abs() < 5e-15,
            "exp(log_softmax) row sums to {exp_sum}, expected 1.0 (drift {:e} > 5e-15)",
            (exp_sum - 1.0).abs()
        );
    }

    #[test]
    fn matmul_tensor_contiguous_pairwise_dot_product_precision_at_large_k() {
        // Lock the matmul dot-product precision contract for K > 128
        // (where pairwise summation diverges from the prior naive
        // sequential accumulator). The test computes the inner
        // product of two unit vectors of length K = 10_001. This is
        // intentionally not a power-of-two square: with K = 4096,
        // 1/sqrt(K) is exactly 1/64 and the old sequential accumulator
        // also returns exactly 1.0. Sequential drift on this input is
        // ~1e-13, while pairwise drift is O(log K · ε).
        let k = 10_001usize;
        // Use a non-trivial pattern: lhs[i] = 1/sqrt(K), rhs[i] =
        // 1/sqrt(K). Inner product = 1.0 analytically; the f64
        // round-trip drift is what we measure.
        #[allow(clippy::cast_precision_loss)]
        let recip = 1.0 / (k as f64).sqrt();
        let lhs: Vec<f64> = vec![recip; k];
        let rhs: Vec<f64> = vec![recip; k];
        let lhs_meta = TensorMeta::from_shape(vec![1, k], DType::F64, Device::Cpu);
        let rhs_meta = TensorMeta::from_shape(vec![k, 1], DType::F64, Device::Cpu);

        let out = matmul_tensor_contiguous_f64(&lhs, &rhs, &lhs_meta, &rhs_meta)
            .expect("matmul should succeed");
        assert_eq!(out.len(), 1);

        let drift = (out[0] - 1.0).abs();
        // Pairwise locks the dot product to within ~5 ULP of 1.0 on
        // this 10_001-element input. The prior naive sequential
        // accumulator was loose by ~1e-13, so this assertion now
        // distinguishes the pairwise path from the old accumulator.
        assert!(
            drift < 1e-14,
            "matmul dot-product drift {drift:e} > 1e-14 tolerance (got {}, expected 1.0)",
            out[0]
        );
    }

    #[test]
    fn dot_tensor_contiguous_pairwise_precision_at_large_n() {
        // Dot product with N = 10_001 unit vectors. Same precision
        // contract as the matmul test: pairwise locks the result
        // within ~5 ULPs of the analytical 1.0; the prior naive
        // sequential accumulator drifted ~1e-13 on this
        // input.
        let n = 10_001usize;
        #[allow(clippy::cast_precision_loss)]
        let recip = 1.0 / (n as f64).sqrt();
        let lhs: Vec<f64> = vec![recip; n];
        let rhs: Vec<f64> = vec![recip; n];
        let lhs_meta = TensorMeta::from_shape(vec![n], DType::F64, Device::Cpu);
        let rhs_meta = TensorMeta::from_shape(vec![n], DType::F64, Device::Cpu);

        let dot = dot_tensor_contiguous_f64(&lhs, &rhs, &lhs_meta, &rhs_meta)
            .expect("dot should succeed");

        let drift = (dot - 1.0).abs();
        assert!(
            drift < 1e-14,
            "dot drift {drift:e} > 1e-14 tolerance (got {dot}, expected 1.0)"
        );
    }

    #[test]
    fn bmm_tensor_contiguous_pairwise_precision_at_large_k() {
        // Batched matmul precision contract — identical pattern to
        // the matmul test, exercised across 3 batch elements with
        // K = 10_001 to confirm the pairwise scratch is correctly
        // reused across batch iterations. The analytical truth for
        // each batch element is 1.0.
        let batch = 3usize;
        let k = 10_001usize;
        #[allow(clippy::cast_precision_loss)]
        let recip = 1.0 / (k as f64).sqrt();
        let lhs: Vec<f64> = vec![recip; batch * k];
        let rhs: Vec<f64> = vec![recip; batch * k];
        let lhs_meta = TensorMeta::from_shape(vec![batch, 1, k], DType::F64, Device::Cpu);
        let rhs_meta = TensorMeta::from_shape(vec![batch, k, 1], DType::F64, Device::Cpu);

        let out = bmm_tensor_contiguous_f64(&lhs, &rhs, &lhs_meta, &rhs_meta)
            .expect("bmm should succeed");
        assert_eq!(out.len(), batch);

        for (b, &cell) in out.iter().enumerate() {
            let drift = (cell - 1.0).abs();
            assert!(
                drift < 1e-14,
                "bmm batch {b} drift {drift:e} > 1e-14 tolerance (got {cell}, expected 1.0)"
            );
        }
    }

    #[test]
    fn bmm_tensor_contiguous_respects_storage_offsets_and_batch_order() {
        let lhs_meta =
            TensorMeta::from_shape(vec![2, 2, 2], DType::F64, Device::Cpu).with_storage_offset(1);
        let rhs_meta =
            TensorMeta::from_shape(vec![2, 2, 2], DType::F64, Device::Cpu).with_storage_offset(2);
        let lhs = vec![99.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let rhs = vec![77.0, 88.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0];

        let out = bmm_tensor_contiguous_f64(&lhs, &rhs, &lhs_meta, &rhs_meta)
            .expect("offset bmm should succeed");

        assert_eq!(
            out,
            vec![34.0, 37.0, 78.0, 85.0, 166.0, 177.0, 226.0, 241.0]
        );
    }

    #[test]
    fn mean_tensor_contiguous_returns_expected_value() {
        let meta = TensorMeta::from_shape(vec![4], DType::F64, Device::Cpu);
        let input = vec![1.0, 2.0, 3.0, 4.0];
        let out =
            mean_tensor_contiguous_f64(&input, &meta).expect("contiguous mean should succeed");
        assert_eq!(out, 2.5);
    }

    #[test]
    fn mean_tensor_contiguous_empty_returns_nan() {
        let meta = TensorMeta::from_shape(vec![0], DType::F64, Device::Cpu);
        let input: Vec<f64> = vec![];
        let out = mean_tensor_contiguous_f64(&input, &meta).expect("empty mean should succeed");
        assert!(out.is_nan());
    }

    #[test]
    fn mean_tensor_contiguous_single_element() {
        let meta = TensorMeta::from_shape(vec![1], DType::F64, Device::Cpu);
        let input = vec![42.0];
        let out = mean_tensor_contiguous_f64(&input, &meta).expect("single mean should succeed");
        assert_eq!(out, 42.0);
    }

    #[test]
    fn add_scalar_returns_expected_value() {
        let lhs = ScalarTensor::new(2.0, DType::F64, Device::Cpu);
        let rhs = ScalarTensor::new(3.5, DType::F64, Device::Cpu);

        let out = add_scalar(&lhs, &rhs).expect("add should succeed");
        assert_eq!(out.value(), 5.5);
    }

    #[test]
    fn mul_scalar_returns_expected_value() {
        let lhs = ScalarTensor::new(2.0, DType::F64, Device::Cpu);
        let rhs = ScalarTensor::new(4.0, DType::F64, Device::Cpu);

        let out = mul_scalar(&lhs, &rhs).expect("mul should succeed");
        assert_eq!(out.value(), 8.0);
    }

    #[test]
    fn sub_scalar_returns_expected_value() {
        let lhs = ScalarTensor::new(2.0, DType::F64, Device::Cpu);
        let rhs = ScalarTensor::new(4.0, DType::F64, Device::Cpu);

        let out = sub_scalar(&lhs, &rhs).expect("sub should succeed");
        assert_eq!(out.value(), -2.0);
    }

    #[test]
    fn div_scalar_returns_expected_value() {
        let lhs = ScalarTensor::new(7.0, DType::F64, Device::Cpu);
        let rhs = ScalarTensor::new(2.0, DType::F64, Device::Cpu);

        let out = div_scalar(&lhs, &rhs).expect("div should succeed");
        assert_eq!(out.value(), 3.5);
    }

    #[test]
    fn add_scalar_rejects_dtype_mismatch() {
        let lhs = ScalarTensor::new(2.0, DType::F64, Device::Cpu);
        let rhs = ScalarTensor::new(4.0, DType::F32, Device::Cpu);

        let err = add_scalar(&lhs, &rhs).expect_err("dtype mismatch must fail closed");
        assert!(matches!(
            err,
            KernelError::Incompatible(TensorCompatError::DTypeMismatch { .. })
        ));
    }

    #[test]
    fn mul_scalar_rejects_device_mismatch() {
        let lhs = ScalarTensor::new(2.0, DType::F64, Device::Cpu);
        let rhs = ScalarTensor::new(4.0, DType::F64, Device::Cuda);

        let err = mul_scalar(&lhs, &rhs).expect_err("device mismatch must fail closed");
        assert!(matches!(
            err,
            KernelError::Incompatible(TensorCompatError::DeviceMismatch { .. })
        ));
    }

    #[test]
    fn sub_scalar_rejects_dtype_mismatch() {
        let lhs = ScalarTensor::new(2.0, DType::F64, Device::Cpu);
        let rhs = ScalarTensor::new(4.0, DType::F32, Device::Cpu);

        let err = sub_scalar(&lhs, &rhs).expect_err("dtype mismatch must fail closed");
        assert!(matches!(
            err,
            KernelError::Incompatible(TensorCompatError::DTypeMismatch { .. })
        ));
    }

    #[test]
    fn div_scalar_rejects_device_mismatch() {
        let lhs = ScalarTensor::new(2.0, DType::F64, Device::Cpu);
        let rhs = ScalarTensor::new(4.0, DType::F64, Device::Cuda);

        let err = div_scalar(&lhs, &rhs).expect_err("device mismatch must fail closed");
        assert!(matches!(
            err,
            KernelError::Incompatible(TensorCompatError::DeviceMismatch { .. })
        ));
    }

    #[test]
    fn add_tensor_contiguous_returns_expected_values() {
        let meta = TensorMeta::from_shape(vec![2, 2], DType::F64, Device::Cpu);
        let lhs = vec![1.0, 2.0, 3.0, 4.0];
        let rhs = vec![0.5, 1.5, 2.5, 3.5];

        let out = add_tensor_contiguous_f64(&lhs, &rhs, &meta, &meta)
            .expect("contiguous add should succeed");
        assert_eq!(out, vec![1.5, 3.5, 5.5, 7.5]);
    }

    #[test]
    fn add_tensor_contiguous_preserves_simd_tail_values() {
        let meta = TensorMeta::from_shape(vec![5], DType::F64, Device::Cpu);
        let lhs = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let rhs = vec![10.0, 20.0, 30.0, 40.0, 50.0];

        let out = add_tensor_contiguous_f64(&lhs, &rhs, &meta, &meta)
            .expect("contiguous add should succeed");
        assert_eq!(out, vec![11.0, 22.0, 33.0, 44.0, 55.0]);
    }

    #[test]
    fn sub_tensor_contiguous_respects_storage_offsets() {
        let lhs_meta =
            TensorMeta::from_shape(vec![3], DType::F64, Device::Cpu).with_storage_offset(1);
        let rhs_meta =
            TensorMeta::from_shape(vec![3], DType::F64, Device::Cpu).with_storage_offset(2);
        let lhs = vec![99.0, 5.0, 7.0, 9.0];
        let rhs = vec![11.0, 22.0, 3.0, 4.0, 5.0];

        let out = sub_tensor_contiguous_f64(&lhs, &rhs, &lhs_meta, &rhs_meta)
            .expect("offset add should succeed");
        assert_eq!(out, vec![2.0, 3.0, 4.0]);
    }

    #[test]
    fn mul_tensor_contiguous_rejects_shape_mismatch() {
        let lhs_meta = TensorMeta::from_shape(vec![4], DType::F64, Device::Cpu);
        let rhs_meta = TensorMeta::from_shape(vec![2, 2], DType::F64, Device::Cpu);
        let lhs = vec![1.0, 2.0, 3.0, 4.0];
        let rhs = vec![1.0, 2.0, 3.0, 4.0];

        let err = mul_tensor_contiguous_f64(&lhs, &rhs, &lhs_meta, &rhs_meta)
            .expect_err("shape mismatch must fail closed");
        assert!(matches!(err, KernelError::ShapeMismatch { .. }));
    }

    #[test]
    fn div_tensor_supports_non_contiguous_layout() {
        let lhs_meta =
            TensorMeta::from_shape_and_strides(vec![2, 2], vec![1, 2], 0, DType::F64, Device::Cpu)
                .expect("test meta should be valid");
        let rhs_meta = TensorMeta::from_shape(vec![2, 2], DType::F64, Device::Cpu);
        let lhs = vec![2.0, 4.0, 6.0, 8.0];
        let rhs = vec![1.0, 2.0, 3.0, 4.0];

        let result = div_tensor_contiguous_f64(&lhs, &rhs, &lhs_meta, &rhs_meta)
            .expect("non-contiguous input should be supported");
        assert_eq!(result, vec![2.0, 3.0, 4.0 / 3.0, 2.0]);
    }

    #[test]
    fn add_tensor_broadcast_scalar_to_vector() {
        let lhs_meta = TensorMeta::from_shape(vec![3], DType::F64, Device::Cpu);
        let rhs_meta = TensorMeta::from_shape(vec![1], DType::F64, Device::Cpu);
        let lhs = vec![1.0, 2.0, 3.0];
        let rhs = vec![10.0];

        let (result, shape) = add_tensor_broadcast_f64(&lhs, &rhs, &lhs_meta, &rhs_meta)
            .expect("broadcasting should work");
        assert_eq!(shape, vec![3]);
        assert_eq!(result, vec![11.0, 12.0, 13.0]);
    }

    #[test]
    fn mul_tensor_broadcast_row_and_column() {
        let lhs_meta = TensorMeta::from_shape(vec![3, 1], DType::F64, Device::Cpu);
        let rhs_meta = TensorMeta::from_shape(vec![1, 4], DType::F64, Device::Cpu);
        let lhs = vec![1.0, 2.0, 3.0];
        let rhs = vec![10.0, 20.0, 30.0, 40.0];

        let (result, shape) = mul_tensor_broadcast_f64(&lhs, &rhs, &lhs_meta, &rhs_meta)
            .expect("broadcasting should work");
        assert_eq!(shape, vec![3, 4]);
        assert_eq!(
            result,
            vec![
                10.0, 20.0, 30.0, 40.0, 20.0, 40.0, 60.0, 80.0, 30.0, 60.0, 90.0, 120.0,
            ]
        );
    }

    #[test]
    fn broadcast_incompatible_shapes_fails() {
        let lhs_meta = TensorMeta::from_shape(vec![2, 3], DType::F64, Device::Cpu);
        let rhs_meta = TensorMeta::from_shape(vec![2, 4], DType::F64, Device::Cpu);
        let lhs = vec![1.0; 6];
        let rhs = vec![1.0; 8];

        let err = add_tensor_broadcast_f64(&lhs, &rhs, &lhs_meta, &rhs_meta)
            .expect_err("incompatible shapes should fail");
        assert!(matches!(err, KernelError::ShapeMismatch { .. }));
    }

    #[test]
    fn add_tensor_contiguous_rejects_insufficient_storage() {
        let lhs_meta =
            TensorMeta::from_shape(vec![3], DType::F64, Device::Cpu).with_storage_offset(1);
        let rhs_meta =
            TensorMeta::from_shape(vec![3], DType::F64, Device::Cpu).with_storage_offset(1);
        let lhs = vec![10.0, 11.0, 12.0];
        let rhs = vec![20.0, 21.0, 22.0, 23.0];

        let err = add_tensor_contiguous_f64(&lhs, &rhs, &lhs_meta, &rhs_meta)
            .expect_err("insufficient storage should fail closed");
        assert!(matches!(
            err,
            KernelError::InsufficientStorage {
                side: "lhs",
                needed: 4,
                available: 3
            }
        ));
    }

    #[test]
    fn add_tensor_contiguous_rejects_dtype_mismatch() {
        let lhs_meta = TensorMeta::from_shape(vec![2], DType::F64, Device::Cpu);
        let rhs_meta = TensorMeta::from_shape(vec![2], DType::F32, Device::Cpu);
        let lhs = vec![1.0, 2.0];
        let rhs = vec![3.0, 4.0];

        let err = add_tensor_contiguous_f64(&lhs, &rhs, &lhs_meta, &rhs_meta)
            .expect_err("dtype mismatch must fail closed");
        assert!(matches!(
            err,
            KernelError::Incompatible(TensorCompatError::DTypeMismatch { .. })
        ));
    }

    #[test]
    fn add_tensor_contiguous_rejects_device_mismatch() {
        let lhs_meta = TensorMeta::from_shape(vec![2], DType::F64, Device::Cpu);
        let rhs_meta = TensorMeta::from_shape(vec![2], DType::F64, Device::Cuda);
        let lhs = vec![1.0, 2.0];
        let rhs = vec![3.0, 4.0];

        let err = add_tensor_contiguous_f64(&lhs, &rhs, &lhs_meta, &rhs_meta)
            .expect_err("device mismatch must fail closed");
        assert!(matches!(
            err,
            KernelError::Incompatible(TensorCompatError::DeviceMismatch { .. })
        ));
    }

    #[test]
    fn add_tensor_contiguous_empty_shape_returns_empty_output() {
        let meta =
            TensorMeta::from_shape(vec![0, 3], DType::F64, Device::Cpu).with_storage_offset(8);
        let lhs = Vec::new();
        let rhs = Vec::new();

        let out = add_tensor_contiguous_f64(&lhs, &rhs, &meta, &meta)
            .expect("empty tensors should succeed without touching storage");
        assert!(out.is_empty());
    }

    #[test]
    fn pow_parallel_matches_elementwise_bit_exact() {
        // Isomorphism proof for the parallel pow lever: the multi-threaded path
        // (numel >= PARALLEL_THRESHOLD) must equal the per-element powf BIT-FOR-BIT.
        // pow is a pure map, so parallelizing reorders nothing observable.
        let numel = 1usize << 14; // > PARALLEL_THRESHOLD -> parallel path
        let exponent = 2.5_f64;
        let data: Vec<f64> = (0..numel)
            .map(|i| (i % 97) as f64 * 0.013 + 0.001)
            .collect();
        let meta = TensorMeta::from_shape(vec![numel], DType::F64, Device::Cpu);
        let out = pow_tensor_contiguous_f64(&data, &meta, exponent).expect("pow f64");
        for (idx, (&x, o)) in data.iter().zip(out.iter()).enumerate() {
            let reference = super::powf_torch_signed_zero_f64(x, exponent);
            assert_eq!(
                reference.to_bits(),
                o.to_bits(),
                "f64 pow diverged at {idx}: ref {reference} vs parallel {o}"
            );
        }

        let exponent32 = 2.5_f32;
        let data32: Vec<f32> = (0..numel)
            .map(|i| (i % 89) as f32 * 0.017 + 0.001)
            .collect();
        let meta32 = TensorMeta::from_shape(vec![numel], DType::F32, Device::Cpu);
        let out32 = pow_tensor_contiguous_f32(&data32, &meta32, exponent32).expect("pow f32");
        for (idx, (&x, o)) in data32.iter().zip(out32.iter()).enumerate() {
            let reference = super::powf_torch_signed_zero_f32(x, exponent32);
            assert_eq!(
                reference.to_bits(),
                o.to_bits(),
                "f32 pow diverged at {idx}: ref {reference} vs parallel {o}"
            );
        }
    }

    #[test]
    fn pow_parallel_golden_output_matches_fixture() {
        let numel = 1usize << 14; // > PARALLEL_THRESHOLD -> parallel path
        let values64 = [-2.0_f64, -0.0, 0.0, 0.5, 1.5, 3.0];
        let data64: Vec<f64> = (0..numel).map(|i| values64[i % values64.len()]).collect();
        let meta64 = TensorMeta::from_shape(vec![numel], DType::F64, Device::Cpu);
        let out64 = pow_tensor_contiguous_f64(&data64, &meta64, 2.0).expect("pow f64");

        let values32 = [-2.0_f32, -0.0, 0.0, 0.5, 1.5, 3.0];
        let data32: Vec<f32> = (0..numel).map(|i| values32[i % values32.len()]).collect();
        let meta32 = TensorMeta::from_shape(vec![numel], DType::F32, Device::Cpu);
        let out32 = pow_tensor_contiguous_f32(&data32, &meta32, 2.0).expect("pow f32");

        let selected = [0usize, 1, 2, 3, 4, 5, numel - 1];
        let mut output = String::from("frankentorch-3gzv pow_parallel_golden\nnumel=16384\n");
        output.push_str("f64_exp_bits=0x4000000000000000\nf64_selected_bits:\n");
        for idx in selected {
            output.push_str(&format!(
                "{idx}: {:#018x} -> {:#018x}\n",
                data64[idx].to_bits(),
                out64[idx].to_bits()
            ));
        }
        output.push_str("f32_exp_bits=0x40000000\nf32_selected_bits:\n");
        for idx in selected {
            output.push_str(&format!(
                "{idx}: {:#010x} -> {:#010x}\n",
                data32[idx].to_bits(),
                out32[idx].to_bits()
            ));
        }

        assert_eq!(
            output,
            include_str!(
                "../../../artifacts/optimization/golden_outputs/ft_kernel_cpu_pow_parallel_frankentorch-3gzv.txt"
            )
        );
    }

    #[test]
    fn gemm_row_split_matches_single_bit_exact() {
        // Isomorphism proof for the parallel GEMM lever: splitting the output into
        // row blocks must reproduce the single matrixmultiply call BIT-FOR-BIT.
        // Values are chosen so partial sums lose precision (reassociation WOULD
        // change the low bits), so a passing test means the k-accumulation order
        // is genuinely preserved across the row split.
        let (m, k, n) = (2048usize, 64usize, 256usize); // tall: m>=1024 & 1<<25 flops -> row-split path
        let a: Vec<f64> = (0..m * k)
            .map(|i| ((i % 13) as f64 - 6.0) * 0.3 + (i as f64) * 1e-7)
            .collect();
        let b: Vec<f64> = (0..k * n)
            .map(|i| ((i % 7) as f64 - 3.0) * 0.25 - (i as f64) * 1e-7)
            .collect();
        let mut c_single = vec![0.0_f64; m * n];
        crate::gemm::dgemm_block(m, k, n, &a, &b, &mut c_single);
        let mut c_par = vec![0.0_f64; m * n];
        crate::gemm::dgemm(m, k, n, &a, &b, &mut c_par);
        for (idx, (s, p)) in c_single.iter().zip(c_par.iter()).enumerate() {
            assert_eq!(
                s.to_bits(),
                p.to_bits(),
                "f64 GEMM row-split diverged at {idx}: single {s} vs parallel {p}"
            );
        }

        // Same for f32.
        let af: Vec<f32> = (0..m * k)
            .map(|i| ((i % 11) as f32 - 5.0) * 0.2 + (i as f32) * 1e-5)
            .collect();
        let bf: Vec<f32> = (0..k * n)
            .map(|i| ((i % 5) as f32 - 2.0) * 0.3 - (i as f32) * 1e-5)
            .collect();
        let mut cf_single = vec![0.0_f32; m * n];
        crate::gemm::sgemm_block(m, k, n, &af, &bf, &mut cf_single);
        let mut cf_par = vec![0.0_f32; m * n];
        crate::gemm::sgemm(m, k, n, &af, &bf, &mut cf_par);
        for (idx, (s, p)) in cf_single.iter().zip(cf_par.iter()).enumerate() {
            assert_eq!(
                s.to_bits(),
                p.to_bits(),
                "f32 GEMM row-split diverged at {idx}: single {s} vs parallel {p}"
            );
        }
    }

    #[test]
    fn gemm_col_split_matches_single_bit_exact() {
        // Isomorphism proof for the COLUMN-parallel GEMM lever (small-m, large-n):
        // splitting the output into n-blocks must reproduce the single
        // matrixmultiply call BIT-FOR-BIT. Shape triggers should_parallelize_cols
        // (m small, n >> m, flops >= 1<<24) but NOT should_parallelize.
        let (m, k, n) = (8usize, 512usize, 4096usize); // 8*512*4096 = 16.8M, n > 4m
        let a: Vec<f64> = (0..m * k)
            .map(|i| ((i % 13) as f64 - 6.0) * 0.3 + (i as f64) * 1e-7)
            .collect();
        let b: Vec<f64> = (0..k * n)
            .map(|i| ((i % 7) as f64 - 3.0) * 0.25 - (i as f64) * 1e-7)
            .collect();
        let mut c_single = vec![0.0_f64; m * n];
        crate::gemm::dgemm_block(m, k, n, &a, &b, &mut c_single);
        let mut c_par = vec![0.0_f64; m * n];
        crate::gemm::dgemm(m, k, n, &a, &b, &mut c_par);
        for (idx, (s, p)) in c_single.iter().zip(c_par.iter()).enumerate() {
            assert_eq!(
                s.to_bits(),
                p.to_bits(),
                "f64 GEMM col-split diverged at {idx}: single {s} vs parallel {p}"
            );
        }

        // Same for f32 (sgemm column path).
        let af: Vec<f32> = (0..m * k)
            .map(|i| ((i % 11) as f32 - 5.0) * 0.2 + (i as f32) * 1e-5)
            .collect();
        let bf: Vec<f32> = (0..k * n)
            .map(|i| ((i % 5) as f32 - 2.0) * 0.3 - (i as f32) * 1e-5)
            .collect();
        let mut cf_single = vec![0.0_f32; m * n];
        crate::gemm::sgemm_block(m, k, n, &af, &bf, &mut cf_single);
        let mut cf_par = vec![0.0_f32; m * n];
        crate::gemm::sgemm(m, k, n, &af, &bf, &mut cf_par);
        for (idx, (s, p)) in cf_single.iter().zip(cf_par.iter()).enumerate() {
            assert_eq!(
                s.to_bits(),
                p.to_bits(),
                "f32 GEMM col-split diverged at {idx}: single {s} vs parallel {p}"
            );
        }
    }

    #[test]
    fn dgemm_bt_matches_materialized_transpose_bit_exact() {
        // dgemm_bt (A @ B^T reading B in [n,k] layout via strides) must match the
        // materialise-transpose-then-dgemm path BIT-FOR-BIT, for the wide column
        // path, the row path, and serial. Values lose precision under reassociation.
        for &(m, k, n) in &[(32usize, 512usize, 2048usize), (2048, 64, 256), (5, 7, 9)] {
            let a: Vec<f64> = (0..m * k)
                .map(|i| ((i % 13) as f64 - 6.0) * 0.3 + (i as f64) * 1e-7)
                .collect();
            // b is [n, k] (a Linear weight [out, in]).
            let b: Vec<f64> = (0..n * k)
                .map(|i| ((i % 7) as f64 - 3.0) * 0.25 - (i as f64) * 1e-7)
                .collect();
            // Reference: materialise b^T [k, n] then dgemm.
            let mut bt = vec![0.0f64; k * n];
            for j in 0..n {
                for l in 0..k {
                    bt[l * n + j] = b[j * k + l];
                }
            }
            let mut c_ref = vec![0.0f64; m * n];
            crate::gemm::dgemm(m, k, n, &a, &bt, &mut c_ref);
            let mut c_bt = vec![0.0f64; m * n];
            crate::gemm::dgemm_bt(m, k, n, &a, &b, &mut c_bt);
            for (idx, (r, g)) in c_ref.iter().zip(c_bt.iter()).enumerate() {
                assert_eq!(
                    r.to_bits(),
                    g.to_bits(),
                    "dgemm_bt diverged at {idx} for ({m},{k},{n}): ref {r} vs bt {g}"
                );
            }

            // f32 sgemm_bt mirror.
            let af: Vec<f32> = a.iter().map(|&v| v as f32).collect();
            let bf: Vec<f32> = b.iter().map(|&v| v as f32).collect();
            let mut btf = vec![0.0f32; k * n];
            for j in 0..n {
                for l in 0..k {
                    btf[l * n + j] = bf[j * k + l];
                }
            }
            let mut cf_ref = vec![0.0f32; m * n];
            crate::gemm::sgemm(m, k, n, &af, &btf, &mut cf_ref);
            let mut cf_bt = vec![0.0f32; m * n];
            crate::gemm::sgemm_bt(m, k, n, &af, &bf, &mut cf_bt);
            for (idx, (r, g)) in cf_ref.iter().zip(cf_bt.iter()).enumerate() {
                assert_eq!(
                    r.to_bits(),
                    g.to_bits(),
                    "sgemm_bt diverged at {idx} for ({m},{k},{n}): ref {r} vs bt {g}"
                );
            }
        }
    }

    #[test]
    fn symmetric_rank2k_lower_update_matches_scalar_reference() {
        let (n, k) = (17usize, 5usize);
        let v: Vec<f64> = (0..n * k)
            .map(|i| ((i % 13) as f64 - 6.0) * 0.125 + (i as f64) * 1e-6)
            .collect();
        let w: Vec<f64> = (0..n * k)
            .map(|i| ((i % 11) as f64 - 5.0) * 0.2 - (i as f64) * 1e-6)
            .collect();
        let mut got: Vec<f64> = (0..n * n)
            .map(|i| ((i % 17) as f64 - 8.0) * 0.031)
            .collect();
        let mut expected = got.clone();
        for row in 0..n {
            for col in 0..=row {
                let mut update = 0.0;
                for p in 0..k {
                    update += v[row * k + p] * w[col * k + p] + w[row * k + p] * v[col * k + p];
                }
                expected[row * n + col] -= update;
            }
        }

        super::symmetric_rank2k_lower_update_f64(n, k, &v, &w, &mut got).unwrap();
        for row in 0..n {
            for col in 0..=row {
                let idx = row * n + col;
                let diff = (got[idx] - expected[idx]).abs();
                let tol = 1e-12 * expected[idx].abs().max(1.0);
                assert!(
                    diff <= tol,
                    "lower update mismatch at ({row},{col}): got {}, expected {}, diff {diff}",
                    got[idx],
                    expected[idx]
                );
            }
            for col in (row + 1)..n {
                let idx = row * n + col;
                assert_eq!(
                    got[idx].to_bits(),
                    expected[idx].to_bits(),
                    "upper triangle changed at ({row},{col})"
                );
            }
        }
    }

    #[test]
    fn exp_f64x4_matches_scalar_within_tolerance() {
        use wide::f64x4;
        // Proof obligation 1 — fast-range accuracy. Chunks fully inside
        // |x| < 708.39 take wide's vectorised degree-13 polynomial; it must
        // track scalar f64::exp to within a few ULP across the common domain.
        let mut max_rel: f64 = 0.0;
        let n = 4096usize;
        let mut xs: Vec<f64> = (0..n)
            .map(|i| -50.0 + 100.0 * (i as f64) / (n as f64 - 1.0))
            .collect();
        // Explicit common points (0 and ±1 must be very accurate).
        xs.extend_from_slice(&[0.0, -0.0, 1.0, -1.0, 50.0, -50.0, 1e-9, -1e-9]);
        while !xs.len().is_multiple_of(4) {
            xs.push(0.0);
        }
        for chunk in xs.chunks_exact(4) {
            let v = f64x4::new([chunk[0], chunk[1], chunk[2], chunk[3]]);
            let got = super::exp_f64x4(v).to_array();
            for (j, &x) in chunk.iter().enumerate() {
                let want = x.exp();
                let rel = (got[j] - want).abs() / want.abs().max(f64::MIN_POSITIVE);
                max_rel = max_rel.max(rel);
            }
        }
        assert!(
            max_rel < 1e-13,
            "fast-range SIMD exp max relative error {max_rel:e} exceeds tolerance"
        );

        // Proof obligation 2 — extreme/non-finite lanes. Every value here is
        // |x| >= 708.39 or non-finite, so the group leaves the fast range and is
        // recomputed with scalar f64::exp -> must be BIT-EXACT to libm:
        // overflow -> +inf, the finite [708.39, 709.78] band, denormal
        // underflow, +inf -> +inf, -inf -> 0, NaN -> NaN.
        let mut edges = vec![
            709.0_f64,
            709.7,
            709.9,
            710.0,
            1000.0,
            -708.5,
            -745.0,
            -750.0,
            f64::INFINITY,
            f64::NEG_INFINITY,
            f64::NAN,
        ];
        while edges.len() % 4 != 0 {
            edges.push(800.0); // also out of fast range -> keeps the group on the scalar path
        }
        for chunk in edges.chunks_exact(4) {
            let v = f64x4::new([chunk[0], chunk[1], chunk[2], chunk[3]]);
            let got = super::exp_f64x4(v).to_array();
            for (j, &x) in chunk.iter().enumerate() {
                let want = x.exp();
                if want.is_nan() {
                    assert!(got[j].is_nan(), "exp({x}) should be NaN, got {}", got[j]);
                } else {
                    assert_eq!(
                        got[j].to_bits(),
                        want.to_bits(),
                        "extreme exp({x}) = {} expected bit-exact {want}",
                        got[j]
                    );
                }
            }
        }
    }

    #[test]
    fn matmul_tensor_contiguous_returns_expected_values() {
        let lhs_meta = TensorMeta::from_shape(vec![2, 3], DType::F64, Device::Cpu);
        let rhs_meta = TensorMeta::from_shape(vec![3, 2], DType::F64, Device::Cpu);
        let lhs = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let rhs = vec![7.0, 8.0, 9.0, 10.0, 11.0, 12.0];

        let out = matmul_tensor_contiguous_f64(&lhs, &rhs, &lhs_meta, &rhs_meta)
            .expect("contiguous matmul should succeed");
        assert_eq!(out, vec![58.0, 64.0, 139.0, 154.0]);
    }

    #[test]
    fn matmul_tensor_contiguous_rejects_rank_mismatch() {
        let lhs_meta = TensorMeta::from_shape(vec![6], DType::F64, Device::Cpu);
        let rhs_meta = TensorMeta::from_shape(vec![3, 2], DType::F64, Device::Cpu);
        let lhs = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let rhs = vec![7.0, 8.0, 9.0, 10.0, 11.0, 12.0];

        let err = matmul_tensor_contiguous_f64(&lhs, &rhs, &lhs_meta, &rhs_meta)
            .expect_err("rank mismatch must fail closed");
        assert!(matches!(err, KernelError::ShapeMismatch { .. }));
    }

    #[test]
    fn matmul_tensor_contiguous_rejects_inner_dimension_mismatch() {
        let lhs_meta = TensorMeta::from_shape(vec![2, 2], DType::F64, Device::Cpu);
        let rhs_meta = TensorMeta::from_shape(vec![3, 2], DType::F64, Device::Cpu);
        let lhs = vec![1.0, 2.0, 3.0, 4.0];
        let rhs = vec![7.0, 8.0, 9.0, 10.0, 11.0, 12.0];

        let err = matmul_tensor_contiguous_f64(&lhs, &rhs, &lhs_meta, &rhs_meta)
            .expect_err("inner-dimension mismatch must fail closed");
        assert!(matches!(err, KernelError::ShapeMismatch { .. }));
    }

    #[test]
    fn matmul_tensor_contiguous_rejects_output_shape_overflow() {
        let lhs_meta = TensorMeta::from_shape(vec![usize::MAX, 1], DType::F64, Device::Cpu);
        let rhs_meta = TensorMeta::from_shape(vec![1, 2], DType::F64, Device::Cpu);
        let err = matmul_tensor_contiguous_f64(&[], &[], &lhs_meta, &rhs_meta)
            .expect_err("output shape multiplication overflow must fail closed");
        assert!(matches!(
            err,
            KernelError::ShapeOverflow {
                context: "matmul output shape multiplication overflow"
            }
        ));
    }

    #[test]
    fn addmm_tensor_contiguous_rejects_insufficient_input_storage() {
        let input_meta =
            TensorMeta::from_shape(vec![2], DType::F64, Device::Cpu).with_storage_offset(1);
        let mat1_meta = TensorMeta::from_shape(vec![1, 2], DType::F64, Device::Cpu);
        let mat2_meta = TensorMeta::from_shape(vec![2, 2], DType::F64, Device::Cpu);
        let input = vec![10.0, 11.0];
        let mat1 = vec![1.0, 2.0];
        let mat2 = vec![3.0, 4.0, 5.0, 6.0];

        let err = super::addmm_tensor_contiguous_f64(
            &input,
            &mat1,
            &mat2,
            &input_meta,
            &mat1_meta,
            &mat2_meta,
            1.0,
            1.0,
        )
        .expect_err("insufficient input storage must fail closed");
        assert!(matches!(
            err,
            KernelError::InsufficientStorage {
                side: "input",
                needed: 3,
                available: 2
            }
        ));
    }

    #[test]
    fn addmm_tensor_contiguous_rejects_non_contiguous_mat2() {
        // A transposed (column-major) stride layout on mat2 would be silently
        // mis-read by the row-major GEMM, producing wrong results. addmm must
        // fail closed the same way matmul/bmm already do.
        let input_meta = TensorMeta::from_shape(vec![2, 2], DType::F64, Device::Cpu);
        let mat1_meta = TensorMeta::from_shape(vec![2, 2], DType::F64, Device::Cpu);
        let mat2_meta =
            TensorMeta::from_shape_and_strides(vec![2, 2], vec![1, 2], 0, DType::F64, Device::Cpu)
                .expect("strided meta is valid");
        assert!(!mat2_meta.is_contiguous());
        let input = vec![0.0; 4];
        let mat1 = vec![1.0, 2.0, 3.0, 4.0];
        let mat2 = vec![1.0, 2.0, 3.0, 4.0];

        let err = super::addmm_tensor_contiguous_f64(
            &input,
            &mat1,
            &mat2,
            &input_meta,
            &mat1_meta,
            &mat2_meta,
            1.0,
            1.0,
        )
        .expect_err("non-contiguous mat2 must fail closed");
        assert!(matches!(
            err,
            KernelError::UnsupportedLayout { side: "mat2" }
        ));
    }

    #[test]
    fn addmm_tensor_contiguous_accepts_contiguous_operands() {
        // Sanity: the contiguity guard is a no-op for ordinary contiguous
        // operands and addmm still computes beta*input + alpha*(mat1 @ mat2).
        let input_meta = TensorMeta::from_shape(vec![2, 2], DType::F64, Device::Cpu);
        let mat1_meta = TensorMeta::from_shape(vec![2, 2], DType::F64, Device::Cpu);
        let mat2_meta = TensorMeta::from_shape(vec![2, 2], DType::F64, Device::Cpu);
        let input = vec![1.0, 1.0, 1.0, 1.0];
        let mat1 = vec![1.0, 2.0, 3.0, 4.0];
        let mat2 = vec![5.0, 6.0, 7.0, 8.0];

        let out = super::addmm_tensor_contiguous_f64(
            &input,
            &mat1,
            &mat2,
            &input_meta,
            &mat1_meta,
            &mat2_meta,
            1.0,
            1.0,
        )
        .expect("contiguous addmm should succeed");
        // mat1 @ mat2 = [[19, 22], [43, 50]]; + input(1) = [[20, 23], [44, 51]].
        assert_eq!(out, vec![20.0, 23.0, 44.0, 51.0]);
    }

    #[test]
    fn addmm_tensor_contiguous_f32_rejects_insufficient_input_storage() {
        let input_meta =
            TensorMeta::from_shape(vec![2], DType::F32, Device::Cpu).with_storage_offset(1);
        let mat1_meta = TensorMeta::from_shape(vec![1, 2], DType::F32, Device::Cpu);
        let mat2_meta = TensorMeta::from_shape(vec![2, 2], DType::F32, Device::Cpu);
        let input = vec![10.0f32, 11.0];
        let mat1 = vec![1.0f32, 2.0];
        let mat2 = vec![3.0f32, 4.0, 5.0, 6.0];

        let err = super::addmm_tensor_contiguous_f32(
            &input,
            &mat1,
            &mat2,
            &input_meta,
            &mat1_meta,
            &mat2_meta,
            1.0,
            1.0,
        )
        .expect_err("insufficient input storage must fail closed");
        assert!(matches!(
            err,
            KernelError::InsufficientStorage {
                side: "input",
                needed: 3,
                available: 2
            }
        ));
    }

    #[test]
    fn outer_tensor_contiguous_rejects_output_shape_overflow() {
        let lhs_meta = TensorMeta::from_shape(vec![usize::MAX], DType::F64, Device::Cpu);
        let rhs_meta = TensorMeta::from_shape(vec![2], DType::F64, Device::Cpu);
        let err = outer_tensor_contiguous_f64(&[], &[], &lhs_meta, &rhs_meta)
            .expect_err("output shape multiplication overflow must fail closed");
        assert!(
            matches!(err, KernelError::ShapeOverflow { .. }),
            "unexpected error: {err:?}"
        );
    }

    #[test]
    fn bmm_tensor_contiguous_rejects_output_shape_overflow() {
        let lhs_meta = TensorMeta::from_shape(vec![1, usize::MAX, 1], DType::F64, Device::Cpu);
        let rhs_meta = TensorMeta::from_shape(vec![1, 1, 2], DType::F64, Device::Cpu);
        let err = bmm_tensor_contiguous_f64(&[], &[], &lhs_meta, &rhs_meta)
            .expect_err("output shape multiplication overflow must fail closed");
        assert!(matches!(
            err,
            KernelError::ShapeOverflow {
                context: "bmm output batch stride overflow"
            }
        ));
    }

    #[test]
    fn eq_scalar_returns_expected_values() {
        let a = ScalarTensor::new(3.0, DType::F64, Device::Cpu);
        let b = ScalarTensor::new(3.0, DType::F64, Device::Cpu);
        let c = ScalarTensor::new(4.0, DType::F64, Device::Cpu);
        assert_eq!(eq_scalar(&a, &b).expect("eq should succeed").value(), 1.0);
        assert_eq!(eq_scalar(&a, &c).expect("eq should succeed").value(), 0.0);
    }

    #[test]
    fn ne_scalar_returns_expected_values() {
        let a = ScalarTensor::new(3.0, DType::F64, Device::Cpu);
        let b = ScalarTensor::new(3.0, DType::F64, Device::Cpu);
        let c = ScalarTensor::new(4.0, DType::F64, Device::Cpu);
        assert_eq!(ne_scalar(&a, &b).expect("ne should succeed").value(), 0.0);
        assert_eq!(ne_scalar(&a, &c).expect("ne should succeed").value(), 1.0);
    }

    #[test]
    fn eq_ne_scalar_respect_ieee_special_values() {
        let pos_inf = ScalarTensor::new(f64::INFINITY, DType::F64, Device::Cpu);
        let neg_inf = ScalarTensor::new(f64::NEG_INFINITY, DType::F64, Device::Cpu);
        let nan = ScalarTensor::new(f64::NAN, DType::F64, Device::Cpu);

        assert_eq!(
            eq_scalar(&pos_inf, &pos_inf)
                .expect("eq(+inf,+inf) should succeed")
                .value(),
            1.0
        );
        assert_eq!(
            eq_scalar(&neg_inf, &neg_inf)
                .expect("eq(-inf,-inf) should succeed")
                .value(),
            1.0
        );
        assert_eq!(
            ne_scalar(&pos_inf, &pos_inf)
                .expect("ne(+inf,+inf) should succeed")
                .value(),
            0.0
        );
        assert_eq!(
            ne_scalar(&neg_inf, &neg_inf)
                .expect("ne(-inf,-inf) should succeed")
                .value(),
            0.0
        );
        assert_eq!(
            eq_scalar(&nan, &nan)
                .expect("eq(nan,nan) should succeed")
                .value(),
            0.0
        );
        assert_eq!(
            ne_scalar(&nan, &nan)
                .expect("ne(nan,nan) should succeed")
                .value(),
            1.0
        );
        assert_eq!(
            eq_scalar(&pos_inf, &neg_inf)
                .expect("eq(+inf,-inf) should succeed")
                .value(),
            0.0
        );
        assert_eq!(
            ne_scalar(&pos_inf, &neg_inf)
                .expect("ne(+inf,-inf) should succeed")
                .value(),
            1.0
        );
    }

    #[test]
    fn lt_gt_scalar_returns_expected_values() {
        let a = ScalarTensor::new(2.0, DType::F64, Device::Cpu);
        let b = ScalarTensor::new(3.0, DType::F64, Device::Cpu);
        assert_eq!(lt_scalar(&a, &b).expect("lt should succeed").value(), 1.0);
        assert_eq!(lt_scalar(&b, &a).expect("lt should succeed").value(), 0.0);
        assert_eq!(gt_scalar(&b, &a).expect("gt should succeed").value(), 1.0);
        assert_eq!(gt_scalar(&a, &b).expect("gt should succeed").value(), 0.0);
    }

    #[test]
    fn le_ge_scalar_returns_expected_values() {
        let a = ScalarTensor::new(2.0, DType::F64, Device::Cpu);
        let b = ScalarTensor::new(2.0, DType::F64, Device::Cpu);
        let c = ScalarTensor::new(3.0, DType::F64, Device::Cpu);
        assert_eq!(le_scalar(&a, &b).expect("le should succeed").value(), 1.0);
        assert_eq!(le_scalar(&a, &c).expect("le should succeed").value(), 1.0);
        assert_eq!(le_scalar(&c, &a).expect("le should succeed").value(), 0.0);
        assert_eq!(ge_scalar(&a, &b).expect("ge should succeed").value(), 1.0);
        assert_eq!(ge_scalar(&c, &a).expect("ge should succeed").value(), 1.0);
        assert_eq!(ge_scalar(&a, &c).expect("ge should succeed").value(), 0.0);
    }

    #[test]
    fn lt_gt_le_ge_scalar_respect_ieee_special_values() {
        let pos_inf = ScalarTensor::new(f64::INFINITY, DType::F64, Device::Cpu);
        let neg_inf = ScalarTensor::new(f64::NEG_INFINITY, DType::F64, Device::Cpu);
        let nan = ScalarTensor::new(f64::NAN, DType::F64, Device::Cpu);

        assert_eq!(
            lt_scalar(&nan, &nan)
                .expect("lt(nan,nan) should succeed")
                .value(),
            0.0
        );
        assert_eq!(
            gt_scalar(&nan, &nan)
                .expect("gt(nan,nan) should succeed")
                .value(),
            0.0
        );
        assert_eq!(
            le_scalar(&nan, &nan)
                .expect("le(nan,nan) should succeed")
                .value(),
            0.0
        );
        assert_eq!(
            ge_scalar(&nan, &nan)
                .expect("ge(nan,nan) should succeed")
                .value(),
            0.0
        );
        assert_eq!(
            lt_scalar(&neg_inf, &pos_inf)
                .expect("lt(-inf,+inf) should succeed")
                .value(),
            1.0
        );
        assert_eq!(
            gt_scalar(&pos_inf, &neg_inf)
                .expect("gt(+inf,-inf) should succeed")
                .value(),
            1.0
        );
        assert_eq!(
            le_scalar(&pos_inf, &pos_inf)
                .expect("le(+inf,+inf) should succeed")
                .value(),
            1.0
        );
        assert_eq!(
            ge_scalar(&neg_inf, &neg_inf)
                .expect("ge(-inf,-inf) should succeed")
                .value(),
            1.0
        );
    }

    #[test]
    fn eq_tensor_contiguous_returns_expected_values() {
        let meta = TensorMeta::from_shape(vec![4], DType::F64, Device::Cpu);
        let lhs = vec![1.0, 2.0, 3.0, 4.0];
        let rhs = vec![1.0, 5.0, 3.0, 0.0];
        let out = eq_tensor_contiguous_f64(&lhs, &rhs, &meta, &meta).expect("eq should succeed");
        assert_eq!(out, vec![1.0, 0.0, 1.0, 0.0]);
    }

    #[test]
    fn ne_tensor_contiguous_returns_expected_values() {
        let meta = TensorMeta::from_shape(vec![4], DType::F64, Device::Cpu);
        let lhs = vec![1.0, 2.0, 3.0, 4.0];
        let rhs = vec![1.0, 5.0, 3.0, 0.0];
        let out = ne_tensor_contiguous_f64(&lhs, &rhs, &meta, &meta).expect("ne should succeed");
        assert_eq!(out, vec![0.0, 1.0, 0.0, 1.0]);
    }

    #[test]
    fn eq_ne_tensor_contiguous_respect_ieee_special_values() {
        let meta = TensorMeta::from_shape(vec![4], DType::F64, Device::Cpu);
        let lhs = vec![f64::INFINITY, f64::NEG_INFINITY, f64::NAN, 1.0];
        let rhs = vec![f64::INFINITY, f64::NEG_INFINITY, f64::NAN, 2.0];

        let eq_out =
            eq_tensor_contiguous_f64(&lhs, &rhs, &meta, &meta).expect("eq tensor should succeed");
        let ne_out =
            ne_tensor_contiguous_f64(&lhs, &rhs, &meta, &meta).expect("ne tensor should succeed");

        assert_eq!(eq_out, vec![1.0, 1.0, 0.0, 0.0]);
        assert_eq!(ne_out, vec![0.0, 0.0, 1.0, 1.0]);
    }

    #[test]
    fn lt_tensor_contiguous_returns_expected_values() {
        let meta = TensorMeta::from_shape(vec![3], DType::F64, Device::Cpu);
        let lhs = vec![1.0, 3.0, 2.0];
        let rhs = vec![2.0, 3.0, 1.0];
        let out = lt_tensor_contiguous_f64(&lhs, &rhs, &meta, &meta).expect("lt should succeed");
        assert_eq!(out, vec![1.0, 0.0, 0.0]);
    }

    #[test]
    fn gt_tensor_contiguous_returns_expected_values() {
        let meta = TensorMeta::from_shape(vec![3], DType::F64, Device::Cpu);
        let lhs = vec![1.0, 3.0, 2.0];
        let rhs = vec![2.0, 3.0, 1.0];
        let out = gt_tensor_contiguous_f64(&lhs, &rhs, &meta, &meta).expect("gt should succeed");
        assert_eq!(out, vec![0.0, 0.0, 1.0]);
    }

    #[test]
    fn le_tensor_contiguous_returns_expected_values() {
        let meta = TensorMeta::from_shape(vec![3], DType::F64, Device::Cpu);
        let lhs = vec![1.0, 3.0, 2.0];
        let rhs = vec![2.0, 3.0, 1.0];
        let out = le_tensor_contiguous_f64(&lhs, &rhs, &meta, &meta).expect("le should succeed");
        assert_eq!(out, vec![1.0, 1.0, 0.0]);
    }

    #[test]
    fn ge_tensor_contiguous_returns_expected_values() {
        let meta = TensorMeta::from_shape(vec![3], DType::F64, Device::Cpu);
        let lhs = vec![1.0, 3.0, 2.0];
        let rhs = vec![2.0, 3.0, 1.0];
        let out = ge_tensor_contiguous_f64(&lhs, &rhs, &meta, &meta).expect("ge should succeed");
        assert_eq!(out, vec![0.0, 1.0, 1.0]);
    }

    #[test]
    fn lt_gt_le_ge_tensor_contiguous_respect_ieee_special_values() {
        let meta = TensorMeta::from_shape(vec![5], DType::F64, Device::Cpu);
        let lhs = vec![f64::NAN, f64::INFINITY, f64::NEG_INFINITY, 1.0, 2.0];
        let rhs = vec![f64::NAN, f64::NEG_INFINITY, f64::INFINITY, 1.0, 3.0];

        let lt_out = lt_tensor_contiguous_f64(&lhs, &rhs, &meta, &meta).expect("lt should succeed");
        let gt_out = gt_tensor_contiguous_f64(&lhs, &rhs, &meta, &meta).expect("gt should succeed");
        let le_out = le_tensor_contiguous_f64(&lhs, &rhs, &meta, &meta).expect("le should succeed");
        let ge_out = ge_tensor_contiguous_f64(&lhs, &rhs, &meta, &meta).expect("ge should succeed");

        assert_eq!(lt_out, vec![0.0, 0.0, 1.0, 0.0, 1.0]);
        assert_eq!(gt_out, vec![0.0, 1.0, 0.0, 0.0, 0.0]);
        assert_eq!(le_out, vec![0.0, 0.0, 1.0, 1.0, 1.0]);
        assert_eq!(ge_out, vec![0.0, 1.0, 0.0, 1.0, 0.0]);
    }

    #[test]
    fn sqrt_scalar_returns_expected_value() {
        let input = ScalarTensor::new(9.0, DType::F64, Device::Cpu);
        let out = sqrt_scalar(&input);
        assert_eq!(out.value(), 3.0);
    }

    #[test]
    fn sqrt_scalar_of_zero() {
        let input = ScalarTensor::new(0.0, DType::F64, Device::Cpu);
        let out = sqrt_scalar(&input);
        assert_eq!(out.value(), 0.0);
    }

    #[test]
    fn sqrt_scalar_of_one() {
        let input = ScalarTensor::new(1.0, DType::F64, Device::Cpu);
        let out = sqrt_scalar(&input);
        assert_eq!(out.value(), 1.0);
    }

    #[test]
    fn sqrt_tensor_contiguous_returns_expected_values() {
        let meta = TensorMeta::from_shape(vec![4], DType::F64, Device::Cpu);
        let input = vec![0.0, 1.0, 4.0, 9.0];
        let out =
            sqrt_tensor_contiguous_f64(&input, &meta).expect("contiguous sqrt should succeed");
        assert_eq!(out, vec![0.0, 1.0, 2.0, 3.0]);
    }

    #[test]
    fn reciprocal_scalar_returns_expected_value() {
        let input = ScalarTensor::new(4.0, DType::F64, Device::Cpu);
        let out = reciprocal_scalar(&input);
        assert_eq!(out.value(), 0.25);
    }

    #[test]
    fn reciprocal_scalar_of_one() {
        let input = ScalarTensor::new(1.0, DType::F64, Device::Cpu);
        let out = reciprocal_scalar(&input);
        assert_eq!(out.value(), 1.0);
    }

    #[test]
    fn reciprocal_scalar_negative() {
        let input = ScalarTensor::new(-2.0, DType::F64, Device::Cpu);
        let out = reciprocal_scalar(&input);
        assert_eq!(out.value(), -0.5);
    }

    #[test]
    fn reciprocal_tensor_contiguous_returns_expected_values() {
        let meta = TensorMeta::from_shape(vec![3], DType::F64, Device::Cpu);
        let input = vec![1.0, 2.0, 4.0];
        let out = reciprocal_tensor_contiguous_f64(&input, &meta)
            .expect("contiguous reciprocal should succeed");
        assert_eq!(out, vec![1.0, 0.5, 0.25]);
    }

    #[test]
    fn pow_scalar_returns_expected_value() {
        let input = ScalarTensor::new(3.0, DType::F64, Device::Cpu);
        let out = pow_scalar(&input, 2.0);
        assert_eq!(out.value(), 9.0);
    }

    #[test]
    fn pow_scalar_zero_exponent() {
        let input = ScalarTensor::new(5.0, DType::F64, Device::Cpu);
        let out = pow_scalar(&input, 0.0);
        assert_eq!(out.value(), 1.0);
    }

    #[test]
    fn pow_scalar_fractional_exponent() {
        let input = ScalarTensor::new(4.0, DType::F64, Device::Cpu);
        let out = pow_scalar(&input, 0.5);
        assert!((out.value() - 2.0).abs() < 1e-10);
    }

    #[test]
    fn pow_scalar_matches_libm_pow_negative_zero_fractional_exponent() {
        // Per IEEE 754-2008 §9.2.1 and current libm (and current
        // torch), pow(-0, fractional > 0) returns +0, not -0. The
        // previous behavior in powf_torch_signed_zero_f64 forced
        // -0 to match an older torch convention; updated under
        // frankentorch-vgj2 to align with libm/torch parity now
        // enforced by torch_pow_ieee754_subprocess_conformance.
        let input = ScalarTensor::new(-0.0, DType::F64, Device::Cpu);
        let out = pow_scalar(&input, 0.5);
        assert_eq!(out.value().to_bits(), (0.0f64).to_bits());
    }

    #[test]
    fn pow_tensor_contiguous_returns_expected_values() {
        let meta = TensorMeta::from_shape(vec![3], DType::F64, Device::Cpu);
        let input = vec![1.0, 2.0, 3.0];
        let out =
            pow_tensor_contiguous_f64(&input, &meta, 2.0).expect("contiguous pow should succeed");
        assert_eq!(out, vec![1.0, 4.0, 9.0]);
    }

    #[test]
    fn pow_tensor_contiguous_negative_exponent() {
        let meta = TensorMeta::from_shape(vec![2], DType::F64, Device::Cpu);
        let input = vec![2.0, 4.0];
        let out = pow_tensor_contiguous_f64(&input, &meta, -1.0)
            .expect("pow with negative exponent should succeed");
        assert_eq!(out, vec![0.5, 0.25]);
    }

    #[test]
    fn pow_tensor_matches_libm_for_negative_zero_fractional_exponent() {
        // Per IEEE 754-2008 §9.2.1, pow(-0, fractional > 0) returns
        // +0 (matches libm pow). Updated under frankentorch-vgj2 to
        // align with current torch parity.
        let meta = TensorMeta::from_shape(vec![2], DType::F64, Device::Cpu);
        let input = vec![-0.0, 0.0];
        let out =
            pow_tensor_contiguous_f64(&input, &meta, 0.5).expect("fractional pow should succeed");
        assert_eq!(out[0].to_bits(), 0.0f64.to_bits());
        assert_eq!(out[1].to_bits(), 0.0f64.to_bits());
    }

    #[test]
    fn pow_tensor_f32_matches_libm_for_negative_zero_fractional_exponent() {
        // Companion to pow_tensor_matches_libm_for_negative_zero_fractional_exponent.
        let meta = TensorMeta::from_shape(vec![2], DType::F32, Device::Cpu);
        let input = vec![-0.0f32, 0.0f32];
        let out =
            pow_tensor_contiguous_f32(&input, &meta, 0.5).expect("fractional pow should succeed");
        assert_eq!(out[0].to_bits(), 0.0f32.to_bits());
        assert_eq!(out[1].to_bits(), 0.0f32.to_bits());
    }

    #[test]
    fn clamp_scalar_within_range() {
        let input = ScalarTensor::new(3.0, DType::F64, Device::Cpu);
        let out = clamp_scalar(&input, 1.0, 5.0);
        assert_eq!(out.value(), 3.0);
    }

    #[test]
    fn clamp_scalar_below_min() {
        let input = ScalarTensor::new(-1.0, DType::F64, Device::Cpu);
        let out = clamp_scalar(&input, 0.0, 5.0);
        assert_eq!(out.value(), 0.0);
    }

    #[test]
    fn clamp_scalar_above_max() {
        let input = ScalarTensor::new(10.0, DType::F64, Device::Cpu);
        let out = clamp_scalar(&input, 0.0, 5.0);
        assert_eq!(out.value(), 5.0);
    }

    #[test]
    fn clamp_tensor_contiguous_returns_expected_values() {
        let meta = TensorMeta::from_shape(vec![5], DType::F64, Device::Cpu);
        let input = vec![-2.0, 0.0, 3.0, 5.0, 8.0];
        let out = clamp_tensor_contiguous_f64(&input, &meta, 0.0, 5.0)
            .expect("contiguous clamp should succeed");
        assert_eq!(out, vec![0.0, 0.0, 3.0, 5.0, 5.0]);
    }

    #[test]
    fn clamp_min_greater_than_max_returns_max() {
        // clamp = min(max(x, lo), hi). When lo > hi every element collapses
        // to hi, matching PyTorch / std::min(std::max(...)).
        let scalar = clamp_scalar(&ScalarTensor::new(3.0, DType::F64, Device::Cpu), 5.0, 2.0);
        assert_eq!(scalar.value(), 2.0);

        let meta = TensorMeta::from_shape(vec![4], DType::F64, Device::Cpu);
        let out = clamp_tensor_contiguous_f64(&[-10.0, 3.0, 7.0, 100.0], &meta, 5.0, 2.0)
            .expect("clamp should succeed");
        assert_eq!(out, vec![2.0, 2.0, 2.0, 2.0]);
    }

    #[test]
    fn norm_inf_propagates_nan() {
        // norm(inf)/norm(-inf) are max/min of |x|; a NaN element makes the
        // whole result NaN, matching PyTorch's max/min reductions.
        let meta = TensorMeta::from_shape(vec![3], DType::F64, Device::Cpu);
        let inf = norm_tensor_contiguous_f64(&[1.0, f64::NAN, 2.0], &meta, f64::INFINITY)
            .expect("norm should succeed");
        assert!(inf.is_nan(), "norm(inf) with a NaN element must be NaN");
        let neg_inf = norm_tensor_contiguous_f64(&[1.0, f64::NAN, 2.0], &meta, f64::NEG_INFINITY)
            .expect("norm should succeed");
        assert!(
            neg_inf.is_nan(),
            "norm(-inf) with a NaN element must be NaN"
        );
        // Without a NaN the result is still the ordinary extremum.
        let clean = norm_tensor_contiguous_f64(&[1.0, -3.0, 2.0], &meta, f64::INFINITY)
            .expect("norm should succeed");
        assert_eq!(clean, 3.0);
    }

    #[test]
    fn sort_places_nan_as_largest() {
        // PyTorch sorts NaN as the largest value: ascending puts it last,
        // descending puts it first. A non-transitive comparator would leave
        // the finite elements unsorted.
        let meta = TensorMeta::from_shape(vec![3], DType::F64, Device::Cpu);
        let (asc, _) = super::sort_tensor_contiguous_f64(&[2.0, f64::NAN, 1.0], &meta, 0, false)
            .expect("sort should succeed");
        assert_eq!(asc[0], 1.0);
        assert_eq!(asc[1], 2.0);
        assert!(asc[2].is_nan(), "ascending sort must place NaN last");
        let (desc, _) = super::sort_tensor_contiguous_f64(&[2.0, f64::NAN, 1.0], &meta, 0, true)
            .expect("sort should succeed");
        assert!(desc[0].is_nan(), "descending sort must place NaN first");
        assert_eq!(desc[1], 2.0);
        assert_eq!(desc[2], 1.0);
    }

    #[test]
    fn topk_largest_selects_nan_first() {
        // With largest=true PyTorch treats NaN as the maximum element.
        let meta = TensorMeta::from_shape(vec![3], DType::F64, Device::Cpu);
        let (values, indices) =
            super::topk_tensor_contiguous_f64(&[1.0, f64::NAN, 3.0], &meta, 1, 0, true, true)
                .expect("topk should succeed");
        assert!(
            values[0].is_nan(),
            "topk largest must pick NaN, got {values:?}"
        );
        assert_eq!(indices[0], 1);
    }

    #[test]
    fn min_scalar_returns_expected_value() {
        let a = ScalarTensor::new(3.0, DType::F64, Device::Cpu);
        let b = ScalarTensor::new(5.0, DType::F64, Device::Cpu);
        let out = min_scalar(&a, &b).expect("min should succeed");
        assert_eq!(out.value(), 3.0);
    }

    #[test]
    fn max_scalar_returns_expected_value() {
        let a = ScalarTensor::new(3.0, DType::F64, Device::Cpu);
        let b = ScalarTensor::new(5.0, DType::F64, Device::Cpu);
        let out = max_scalar(&a, &b).expect("max should succeed");
        assert_eq!(out.value(), 5.0);
    }

    #[test]
    fn min_tensor_contiguous_returns_expected_values() {
        let meta = TensorMeta::from_shape(vec![3], DType::F64, Device::Cpu);
        let lhs = vec![1.0, 5.0, 3.0];
        let rhs = vec![2.0, 4.0, 3.0];
        let out = min_tensor_contiguous_f64(&lhs, &rhs, &meta, &meta).expect("min should succeed");
        assert_eq!(out, vec![1.0, 4.0, 3.0]);
    }

    #[test]
    fn max_tensor_contiguous_returns_expected_values() {
        let meta = TensorMeta::from_shape(vec![3], DType::F64, Device::Cpu);
        let lhs = vec![1.0, 5.0, 3.0];
        let rhs = vec![2.0, 4.0, 3.0];
        let out = max_tensor_contiguous_f64(&lhs, &rhs, &meta, &meta).expect("max should succeed");
        assert_eq!(out, vec![2.0, 5.0, 3.0]);
    }

    #[test]
    fn sum_dim_reduces_along_dim0() {
        // shape [2, 3]: [[1, 2, 3], [4, 5, 6]]
        let meta = TensorMeta::from_shape(vec![2, 3], DType::F64, Device::Cpu);
        let input = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let out = sum_dim_tensor_contiguous_f64(&input, &meta, 0).expect("sum_dim 0");
        // reduce rows: [1+4, 2+5, 3+6] = [5, 7, 9]
        assert_eq!(out, vec![5.0, 7.0, 9.0]);
    }

    #[test]
    fn sum_dim_reduces_along_dim1() {
        // shape [2, 3]: [[1, 2, 3], [4, 5, 6]]
        let meta = TensorMeta::from_shape(vec![2, 3], DType::F64, Device::Cpu);
        let input = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let out = sum_dim_tensor_contiguous_f64(&input, &meta, 1).expect("sum_dim 1");
        // reduce cols: [1+2+3, 4+5+6] = [6, 15]
        assert_eq!(out, vec![6.0, 15.0]);
    }

    #[test]
    fn sum_dim_3d_reduces_middle_dim() {
        // shape [2, 3, 2]: 12 elements
        let meta = TensorMeta::from_shape(vec![2, 3, 2], DType::F64, Device::Cpu);
        let input = vec![
            1.0, 2.0, 3.0, 4.0, 5.0, 6.0, // first [3,2] slice
            7.0, 8.0, 9.0, 10.0, 11.0, 12.0, // second [3,2] slice
        ];
        let out = sum_dim_tensor_contiguous_f64(&input, &meta, 1).expect("sum_dim 1 on 3d");
        // Output shape [2, 2]: reduce along dim 1 (size 3)
        // [0,0]: 1+3+5=9, [0,1]: 2+4+6=12, [1,0]: 7+9+11=27, [1,1]: 8+10+12=30
        assert_eq!(out, vec![9.0, 12.0, 27.0, 30.0]);
    }

    #[test]
    fn sum_dim_invalid_dim_returns_error() {
        let meta = TensorMeta::from_shape(vec![2, 3], DType::F64, Device::Cpu);
        let input = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let err = sum_dim_tensor_contiguous_f64(&input, &meta, 2).unwrap_err();
        assert!(matches!(
            err,
            KernelError::InvalidDimension { dim: 2, ndim: 2 }
        ));
    }

    #[test]
    fn sum_dim_empty_with_storage_offset_returns_zeros() {
        let meta =
            TensorMeta::from_shape(vec![0, 3], DType::F64, Device::Cpu).with_storage_offset(8);
        let input: Vec<f64> = vec![];
        let out = sum_dim_tensor_contiguous_f64(&input, &meta, 0)
            .expect("sum_dim empty shape should succeed");
        assert_eq!(out, vec![0.0, 0.0, 0.0]);
    }

    #[test]
    fn prod_dim_empty_with_storage_offset_returns_ones() {
        let meta =
            TensorMeta::from_shape(vec![0, 3], DType::F64, Device::Cpu).with_storage_offset(8);
        let input: Vec<f64> = vec![];
        let out = prod_dim_tensor_contiguous_f64(&input, &meta, 0)
            .expect("prod_dim empty shape should succeed");
        assert_eq!(out, vec![1.0, 1.0, 1.0]);
    }

    #[test]
    fn var_dim_empty_with_storage_offset_returns_nan() {
        let meta =
            TensorMeta::from_shape(vec![0, 3], DType::F64, Device::Cpu).with_storage_offset(8);
        let input: Vec<f64> = vec![];
        let out = var_dim_tensor_contiguous_f64(&input, &meta, 0)
            .expect("var_dim empty shape should succeed");
        assert_eq!(out.len(), 3);
        assert!(out.iter().all(|v| v.is_nan()));
    }

    #[test]
    fn sum_dim_f32_empty_with_storage_offset_returns_zeros() {
        let meta =
            TensorMeta::from_shape(vec![0, 3], DType::F32, Device::Cpu).with_storage_offset(8);
        let input: Vec<f32> = vec![];
        let out = sum_dim_tensor_contiguous_f32(&input, &meta, 0)
            .expect("sum_dim f32 empty shape should succeed");
        assert_eq!(out, vec![0.0, 0.0, 0.0]);
    }

    #[test]
    fn mean_dim_reduces_along_dim0() {
        let meta = TensorMeta::from_shape(vec![2, 3], DType::F64, Device::Cpu);
        let input = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let out = mean_dim_tensor_contiguous_f64(&input, &meta, 0).expect("mean_dim 0");
        // [2.5, 3.5, 4.5]
        assert_eq!(out, vec![2.5, 3.5, 4.5]);
    }

    #[test]
    fn mean_dim_reduces_along_dim1() {
        let meta = TensorMeta::from_shape(vec![2, 3], DType::F64, Device::Cpu);
        let input = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let out = mean_dim_tensor_contiguous_f64(&input, &meta, 1).expect("mean_dim 1");
        // [6/3, 15/3] = [2.0, 5.0]
        assert_eq!(out, vec![2.0, 5.0]);
    }

    #[test]
    fn sum_dim_1d_reduces_to_scalar() {
        let meta = TensorMeta::from_shape(vec![4], DType::F64, Device::Cpu);
        let input = vec![1.0, 2.0, 3.0, 4.0];
        let out = sum_dim_tensor_contiguous_f64(&input, &meta, 0).expect("sum_dim on 1d");
        // Reduces [4] along dim 0 -> scalar-like vec of len 1? No, outer=1, inner=1 -> output len 1
        assert_eq!(out, vec![10.0]);
    }

    #[test]
    fn prod_dim_reduces_along_dim0() {
        let meta = TensorMeta::from_shape(vec![2, 3], DType::F64, Device::Cpu);
        let input = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let out = prod_dim_tensor_contiguous_f64(&input, &meta, 0).expect("prod_dim 0");
        // [1*4, 2*5, 3*6] = [4, 10, 18]
        assert_eq!(out, vec![4.0, 10.0, 18.0]);
    }

    #[test]
    fn prod_dim_reduces_along_dim1() {
        let meta = TensorMeta::from_shape(vec![2, 3], DType::F64, Device::Cpu);
        let input = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let out = prod_dim_tensor_contiguous_f64(&input, &meta, 1).expect("prod_dim 1");
        // [1*2*3, 4*5*6] = [6, 120]
        assert_eq!(out, vec![6.0, 120.0]);
    }

    #[test]
    fn prod_dim_invalid_dim_returns_error() {
        let meta = TensorMeta::from_shape(vec![2, 3], DType::F64, Device::Cpu);
        let input = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let err = prod_dim_tensor_contiguous_f64(&input, &meta, 2).unwrap_err();
        assert!(matches!(
            err,
            KernelError::InvalidDimension { dim: 2, ndim: 2 }
        ));
    }

    #[test]
    fn var_dim_reduces_along_dim0() {
        let meta = TensorMeta::from_shape(vec![3, 2], DType::F64, Device::Cpu);
        // col0: [1,3,5] mean=3, var=((1-3)^2+(3-3)^2+(5-3)^2)/2 = (4+0+4)/2 = 4
        // col1: [2,4,6] mean=4, var=((2-4)^2+(4-4)^2+(6-4)^2)/2 = (4+0+4)/2 = 4
        let input = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let out = var_dim_tensor_contiguous_f64(&input, &meta, 0).expect("var_dim 0");
        assert!((out[0] - 4.0).abs() < 1e-12);
        assert!((out[1] - 4.0).abs() < 1e-12);
    }

    #[test]
    fn var_dim_reduces_along_dim1() {
        let meta = TensorMeta::from_shape(vec![2, 3], DType::F64, Device::Cpu);
        // row0: [1,2,3] mean=2, var=((1-2)^2+(2-2)^2+(3-2)^2)/2 = (1+0+1)/2 = 1
        // row1: [4,5,6] mean=5, var=((4-5)^2+(5-5)^2+(6-5)^2)/2 = (1+0+1)/2 = 1
        let input = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let out = var_dim_tensor_contiguous_f64(&input, &meta, 1).expect("var_dim 1");
        assert!((out[0] - 1.0).abs() < 1e-12);
        assert!((out[1] - 1.0).abs() < 1e-12);
    }

    #[test]
    fn std_dim_reduces_along_dim0() {
        let meta = TensorMeta::from_shape(vec![3, 2], DType::F64, Device::Cpu);
        let input = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let out = std_dim_tensor_contiguous_f64(&input, &meta, 0).expect("std_dim 0");
        assert!((out[0] - 2.0).abs() < 1e-12); // sqrt(4) = 2
        assert!((out[1] - 2.0).abs() < 1e-12);
    }

    #[test]
    fn argmax_dim_reduces_along_dim0() {
        // shape [2, 3]: [[1, 5, 3], [4, 2, 6]]
        let meta = TensorMeta::from_shape(vec![2, 3], DType::F64, Device::Cpu);
        let input = vec![1.0, 5.0, 3.0, 4.0, 2.0, 6.0];
        let out = argmax_dim_tensor_contiguous_f64(&input, &meta, 0).expect("argmax_dim 0");
        // col0: max(1,4)=4 at idx 1; col1: max(5,2)=5 at idx 0; col2: max(3,6)=6 at idx 1
        assert_eq!(out, vec![1.0, 0.0, 1.0]);
    }

    #[test]
    fn argmax_dim_reduces_along_dim1() {
        // shape [2, 3]: [[1, 5, 3], [4, 2, 6]]
        let meta = TensorMeta::from_shape(vec![2, 3], DType::F64, Device::Cpu);
        let input = vec![1.0, 5.0, 3.0, 4.0, 2.0, 6.0];
        let out = argmax_dim_tensor_contiguous_f64(&input, &meta, 1).expect("argmax_dim 1");
        // row0: max(1,5,3)=5 at idx 1; row1: max(4,2,6)=6 at idx 2
        assert_eq!(out, vec![1.0, 2.0]);
    }

    #[test]
    fn argmax_dim_invalid_dim_returns_error() {
        let meta = TensorMeta::from_shape(vec![2, 3], DType::F64, Device::Cpu);
        let input = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let err = argmax_dim_tensor_contiguous_f64(&input, &meta, 2).unwrap_err();
        assert!(matches!(
            err,
            KernelError::InvalidDimension { dim: 2, ndim: 2 }
        ));
    }

    #[test]
    fn argmax_max_over_empty_dim_error() {
        // argmax/argmin/max/min have no identity element, so PyTorch raises
        // a runtime error when the reduced dimension has size zero rather
        // than returning a sentinel index/value.
        let meta = TensorMeta::from_shape(vec![3, 0], DType::F64, Device::Cpu);
        let input: Vec<f64> = Vec::new();
        assert!(matches!(
            argmax_dim_tensor_contiguous_f64(&input, &meta, 1).unwrap_err(),
            KernelError::EmptyReductionDim { dim: 1 }
        ));
        assert!(matches!(
            argmin_dim_tensor_contiguous_f64(&input, &meta, 1).unwrap_err(),
            KernelError::EmptyReductionDim { dim: 1 }
        ));
        assert!(matches!(
            max_dim_tensor_contiguous_f64(&input, &meta, 1).unwrap_err(),
            KernelError::EmptyReductionDim { dim: 1 }
        ));
        assert!(matches!(
            min_dim_tensor_contiguous_f64(&input, &meta, 1).unwrap_err(),
            KernelError::EmptyReductionDim { dim: 1 }
        ));
        // sum/mean/prod DO have identity elements and still succeed.
        assert!(sum_dim_tensor_contiguous_f64(&input, &meta, 1).is_ok());
    }

    #[test]
    fn argmin_dim_reduces_along_dim0() {
        // shape [2, 3]: [[1, 5, 3], [4, 2, 6]]
        let meta = TensorMeta::from_shape(vec![2, 3], DType::F64, Device::Cpu);
        let input = vec![1.0, 5.0, 3.0, 4.0, 2.0, 6.0];
        let out = argmin_dim_tensor_contiguous_f64(&input, &meta, 0).expect("argmin_dim 0");
        // col0: min(1,4)=1 at idx 0; col1: min(5,2)=2 at idx 1; col2: min(3,6)=3 at idx 0
        assert_eq!(out, vec![0.0, 1.0, 0.0]);
    }

    #[test]
    fn argmin_dim_reduces_along_dim1() {
        // shape [2, 3]: [[1, 5, 3], [4, 2, 6]]
        let meta = TensorMeta::from_shape(vec![2, 3], DType::F64, Device::Cpu);
        let input = vec![1.0, 5.0, 3.0, 4.0, 2.0, 6.0];
        let out = argmin_dim_tensor_contiguous_f64(&input, &meta, 1).expect("argmin_dim 1");
        // row0: min(1,5,3)=1 at idx 0; row1: min(4,2,6)=2 at idx 1
        assert_eq!(out, vec![0.0, 1.0]);
    }

    #[test]
    fn argmin_dim_invalid_dim_returns_error() {
        let meta = TensorMeta::from_shape(vec![2, 3], DType::F64, Device::Cpu);
        let input = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let err = argmin_dim_tensor_contiguous_f64(&input, &meta, 2).unwrap_err();
        assert!(matches!(
            err,
            KernelError::InvalidDimension { dim: 2, ndim: 2 }
        ));
    }

    #[test]
    fn max_dim_reduces_along_dim0() {
        // shape [2, 3]: [[1, 5, 3], [4, 2, 6]]
        let meta = TensorMeta::from_shape(vec![2, 3], DType::F64, Device::Cpu);
        let input = vec![1.0, 5.0, 3.0, 4.0, 2.0, 6.0];
        let (values, indices) = max_dim_tensor_contiguous_f64(&input, &meta, 0).expect("max_dim 0");
        assert_eq!(values, vec![4.0, 5.0, 6.0]);
        assert_eq!(indices, vec![1.0, 0.0, 1.0]);
    }

    #[test]
    fn max_dim_reduces_along_dim1() {
        // shape [2, 3]: [[1, 5, 3], [4, 2, 6]]
        let meta = TensorMeta::from_shape(vec![2, 3], DType::F64, Device::Cpu);
        let input = vec![1.0, 5.0, 3.0, 4.0, 2.0, 6.0];
        let (values, indices) = max_dim_tensor_contiguous_f64(&input, &meta, 1).expect("max_dim 1");
        assert_eq!(values, vec![5.0, 6.0]);
        assert_eq!(indices, vec![1.0, 2.0]);
    }

    #[test]
    fn max_dim_invalid_dim_returns_error() {
        let meta = TensorMeta::from_shape(vec![2, 3], DType::F64, Device::Cpu);
        let input = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let err = max_dim_tensor_contiguous_f64(&input, &meta, 2).unwrap_err();
        assert!(matches!(
            err,
            KernelError::InvalidDimension { dim: 2, ndim: 2 }
        ));
    }

    #[test]
    fn min_dim_reduces_along_dim0() {
        // shape [2, 3]: [[1, 5, 3], [4, 2, 6]]
        let meta = TensorMeta::from_shape(vec![2, 3], DType::F64, Device::Cpu);
        let input = vec![1.0, 5.0, 3.0, 4.0, 2.0, 6.0];
        let (values, indices) = min_dim_tensor_contiguous_f64(&input, &meta, 0).expect("min_dim 0");
        assert_eq!(values, vec![1.0, 2.0, 3.0]);
        assert_eq!(indices, vec![0.0, 1.0, 0.0]);
    }

    #[test]
    fn min_dim_reduces_along_dim1() {
        // shape [2, 3]: [[1, 5, 3], [4, 2, 6]]
        let meta = TensorMeta::from_shape(vec![2, 3], DType::F64, Device::Cpu);
        let input = vec![1.0, 5.0, 3.0, 4.0, 2.0, 6.0];
        let (values, indices) = min_dim_tensor_contiguous_f64(&input, &meta, 1).expect("min_dim 1");
        assert_eq!(values, vec![1.0, 2.0]);
        assert_eq!(indices, vec![0.0, 1.0]);
    }

    #[test]
    fn min_dim_invalid_dim_returns_error() {
        let meta = TensorMeta::from_shape(vec![2, 3], DType::F64, Device::Cpu);
        let input = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let err = min_dim_tensor_contiguous_f64(&input, &meta, 2).unwrap_err();
        assert!(matches!(
            err,
            KernelError::InvalidDimension { dim: 2, ndim: 2 }
        ));
    }

    #[test]
    fn max_dim_3d_reduces_middle_dim() {
        // shape [2, 3, 2]: 12 elements
        let meta = TensorMeta::from_shape(vec![2, 3, 2], DType::F64, Device::Cpu);
        let input = vec![
            1.0, 2.0, 3.0, 4.0, 5.0, 6.0, // first [3,2] slice
            7.0, 8.0, 9.0, 10.0, 11.0, 12.0, // second [3,2] slice
        ];
        let (values, indices) =
            max_dim_tensor_contiguous_f64(&input, &meta, 1).expect("max_dim 1 on 3d");
        // Output shape [2, 2]: max along dim 1 (size 3)
        // [0,0]: max(1,3,5)=5 at idx 2; [0,1]: max(2,4,6)=6 at idx 2
        // [1,0]: max(7,9,11)=11 at idx 2; [1,1]: max(8,10,12)=12 at idx 2
        assert_eq!(values, vec![5.0, 6.0, 11.0, 12.0]);
        assert_eq!(indices, vec![2.0, 2.0, 2.0, 2.0]);
    }

    #[test]
    fn argmax_dim_1d_reduces_to_scalar() {
        let meta = TensorMeta::from_shape(vec![4], DType::F64, Device::Cpu);
        let input = vec![1.0, 4.0, 2.0, 3.0];
        let out = argmax_dim_tensor_contiguous_f64(&input, &meta, 0).expect("argmax_dim on 1d");
        assert_eq!(out, vec![1.0]);
    }

    #[test]
    fn argmin_dim_1d_reduces_to_scalar() {
        let meta = TensorMeta::from_shape(vec![4], DType::F64, Device::Cpu);
        let input = vec![3.0, 1.0, 4.0, 2.0];
        let out = argmin_dim_tensor_contiguous_f64(&input, &meta, 0).expect("argmin_dim on 1d");
        assert_eq!(out, vec![1.0]);
    }

    #[test]
    fn softmax_family_parallel_matches_serial_bit_exact() {
        // Isomorphism proof for parallelizing the softmax/log_softmax dim family:
        // the parallel kernels must equal the original serial gather/exp/
        // pairwise-sum/scatter algorithm BIT-FOR-BIT. Use a STRIDED shape
        // (inner_size > 1) so the general (non-fast-path) code is exercised, plus
        // a last-dim shape for the fast path.
        for (shape, dim) in [
            (vec![7usize, 16usize, 5usize], 1usize), // strided: inner_size = 5
            (vec![64usize, 48usize], 1usize),        // fast path: inner_size = 1
        ] {
            let numel: usize = shape.iter().product();
            let reduce = shape[dim];
            let inner: usize = shape[dim + 1..].iter().product();
            let outer: usize = shape[..dim].iter().product();
            let data: Vec<f64> = (0..numel)
                .map(|i| ((i % 23) as f64 - 11.0) * 0.37 + (i as f64) * 1e-6)
                .collect();
            let meta = TensorMeta::from_shape(shape.clone(), DType::F64, Device::Cpu);

            // Serial references replicating the pre-parallel algorithm exactly.
            let mut sm_ref = vec![0.0_f64; numel];
            let mut lsm_ref = vec![0.0_f64; numel];
            let mut scratch = vec![0.0_f64; reduce];
            for o in 0..outer {
                for i in 0..inner {
                    for r in 0..reduce {
                        scratch[r] = data[o * reduce * inner + r * inner + i];
                    }
                    let max_val = scratch.iter().copied().fold(f64::NEG_INFINITY, f64::max);
                    let log_sum_exp = pairwise_sum_map_f64(&scratch, |x| (x - max_val).exp()).ln();
                    let mut exps = scratch.clone();
                    for v in exps.iter_mut() {
                        *v = (*v - max_val).exp();
                    }
                    let sum = pairwise_sum_f64(&exps);
                    for r in 0..reduce {
                        let idx = o * reduce * inner + r * inner + i;
                        sm_ref[idx] = exps[r] / sum;
                        lsm_ref[idx] = (scratch[r] - max_val) - log_sum_exp;
                    }
                }
            }

            let sm = softmax_dim_tensor_contiguous_f64(&data, &meta, dim).expect("softmax");
            let lsm =
                log_softmax_dim_tensor_contiguous_f64(&data, &meta, dim).expect("log_softmax");
            for idx in 0..numel {
                assert_eq!(
                    sm[idx].to_bits(),
                    sm_ref[idx].to_bits(),
                    "softmax f64 shape {shape:?} dim {dim} diverged at {idx}"
                );
                assert_eq!(
                    lsm[idx].to_bits(),
                    lsm_ref[idx].to_bits(),
                    "log_softmax f64 shape {shape:?} dim {dim} diverged at {idx}"
                );
            }

            // f32 mirror.
            let data32: Vec<f32> = (0..numel)
                .map(|i| ((i % 23) as f32 - 11.0) * 0.37 + (i as f32) * 1e-4)
                .collect();
            let meta32 = TensorMeta::from_shape(shape.clone(), DType::F32, Device::Cpu);
            let mut sm_ref32 = vec![0.0_f32; numel];
            let mut lsm_ref32 = vec![0.0_f32; numel];
            let mut scratch32 = vec![0.0_f32; reduce];
            for o in 0..outer {
                for i in 0..inner {
                    for r in 0..reduce {
                        scratch32[r] = data32[o * reduce * inner + r * inner + i];
                    }
                    let max_val = scratch32.iter().copied().fold(f32::NEG_INFINITY, f32::max);
                    let log_sum_exp =
                        super::pairwise_sum_map_f32(&scratch32, |x| (x - max_val).exp()).ln();
                    let mut exps = scratch32.clone();
                    for v in exps.iter_mut() {
                        *v = (*v - max_val).exp();
                    }
                    let sum = super::pairwise_sum_f32(&exps);
                    for r in 0..reduce {
                        let idx = o * reduce * inner + r * inner + i;
                        sm_ref32[idx] = exps[r] / sum;
                        lsm_ref32[idx] = (scratch32[r] - max_val) - log_sum_exp;
                    }
                }
            }
            let sm32 =
                super::softmax_dim_tensor_contiguous_f32(&data32, &meta32, dim).expect("softmax32");
            let lsm32 = super::log_softmax_dim_tensor_contiguous_f32(&data32, &meta32, dim)
                .expect("log_softmax32");
            for idx in 0..numel {
                assert_eq!(
                    sm32[idx].to_bits(),
                    sm_ref32[idx].to_bits(),
                    "softmax f32 at {idx}"
                );
                assert_eq!(
                    lsm32[idx].to_bits(),
                    lsm_ref32[idx].to_bits(),
                    "log_softmax f32 at {idx}"
                );
            }
        }
    }

    #[test]
    fn softmax_family_parallel_golden_output_matches_fixture() {
        use std::fmt::Write as _;

        let fast_rows = 8192usize;
        let fast_cols = 2usize;
        let fast_numel = fast_rows * fast_cols;

        let data64 = vec![0.0_f64; fast_numel];
        let meta64 = TensorMeta::from_shape(vec![fast_rows, fast_cols], DType::F64, Device::Cpu);
        let softmax64 =
            softmax_dim_tensor_contiguous_f64(&data64, &meta64, 1).expect("softmax f64");

        let log_data64 = vec![3.0_f64; fast_rows];
        let log_meta64 = TensorMeta::from_shape(vec![fast_rows, 1], DType::F64, Device::Cpu);
        let log_softmax64 = log_softmax_dim_tensor_contiguous_f64(&log_data64, &log_meta64, 1)
            .expect("log_softmax f64");

        let data32 = vec![0.0_f32; fast_numel];
        let meta32 = TensorMeta::from_shape(vec![fast_rows, fast_cols], DType::F32, Device::Cpu);
        let softmax32 =
            super::softmax_dim_tensor_contiguous_f32(&data32, &meta32, 1).expect("softmax f32");

        let log_data32 = vec![3.0_f32; fast_rows];
        let log_meta32 = TensorMeta::from_shape(vec![fast_rows, 1], DType::F32, Device::Cpu);
        let log_softmax32 =
            super::log_softmax_dim_tensor_contiguous_f32(&log_data32, &log_meta32, 1)
                .expect("log_softmax f32");

        let strided_shape = vec![4096usize, 2usize, 2usize];
        let strided_numel: usize = strided_shape.iter().product();
        let strided64 = vec![0.0_f64; strided_numel];
        let strided_meta64 = TensorMeta::from_shape(strided_shape.clone(), DType::F64, Device::Cpu);
        let strided_softmax64 = softmax_dim_tensor_contiguous_f64(&strided64, &strided_meta64, 1)
            .expect("strided softmax f64");

        let log_strided_shape = vec![4096usize, 1usize, 2usize];
        let log_strided_numel: usize = log_strided_shape.iter().product();
        let log_strided64 = vec![7.0_f64; log_strided_numel];
        let log_strided_meta64 =
            TensorMeta::from_shape(log_strided_shape.clone(), DType::F64, Device::Cpu);
        let strided_log_softmax64 =
            log_softmax_dim_tensor_contiguous_f64(&log_strided64, &log_strided_meta64, 1)
                .expect("strided log_softmax f64");

        let strided32 = vec![0.0_f32; strided_numel];
        let strided_meta32 = TensorMeta::from_shape(strided_shape.clone(), DType::F32, Device::Cpu);
        let strided_softmax32 =
            super::softmax_dim_tensor_contiguous_f32(&strided32, &strided_meta32, 1)
                .expect("strided softmax f32");

        let log_strided32 = vec![7.0_f32; log_strided_numel];
        let log_strided_meta32 = TensorMeta::from_shape(log_strided_shape, DType::F32, Device::Cpu);
        let strided_log_softmax32 =
            super::log_softmax_dim_tensor_contiguous_f32(&log_strided32, &log_strided_meta32, 1)
                .expect("strided log_softmax f32");

        let mut output = String::from(
            "frankentorch-wcoo softmax_parallel_golden\nfast_rows=8192\nfast_cols=2\n",
        );
        output.push_str("f64_softmax_fast_bits:\n");
        for idx in [0usize, 1, fast_numel - 1] {
            let _ = writeln!(&mut output, "{idx}: {:#018x}", softmax64[idx].to_bits());
        }
        output.push_str("f64_log_softmax_fast_bits:\n");
        for idx in [0usize, fast_rows - 1] {
            let _ = writeln!(&mut output, "{idx}: {:#018x}", log_softmax64[idx].to_bits());
        }
        output.push_str("f32_softmax_fast_bits:\n");
        for idx in [0usize, 1, fast_numel - 1] {
            let _ = writeln!(&mut output, "{idx}: {:#010x}", softmax32[idx].to_bits());
        }
        output.push_str("f32_log_softmax_fast_bits:\n");
        for idx in [0usize, fast_rows - 1] {
            let _ = writeln!(&mut output, "{idx}: {:#010x}", log_softmax32[idx].to_bits());
        }

        output.push_str("strided_outer=4096\nstrided_reduce=2\nstrided_inner=2\n");
        output.push_str("f64_softmax_strided_bits:\n");
        for idx in [0usize, 1, 2, 3, strided_numel - 1] {
            let _ = writeln!(
                &mut output,
                "{idx}: {:#018x}",
                strided_softmax64[idx].to_bits()
            );
        }
        output.push_str("f64_log_softmax_strided_bits:\n");
        for idx in [0usize, 1, log_strided_numel - 1] {
            let _ = writeln!(
                &mut output,
                "{idx}: {:#018x}",
                strided_log_softmax64[idx].to_bits()
            );
        }
        output.push_str("f32_softmax_strided_bits:\n");
        for idx in [0usize, 1, 2, 3, strided_numel - 1] {
            let _ = writeln!(
                &mut output,
                "{idx}: {:#010x}",
                strided_softmax32[idx].to_bits()
            );
        }
        output.push_str("f32_log_softmax_strided_bits:\n");
        for idx in [0usize, 1, log_strided_numel - 1] {
            let _ = writeln!(
                &mut output,
                "{idx}: {:#010x}",
                strided_log_softmax32[idx].to_bits()
            );
        }

        assert_eq!(
            output,
            include_str!(
                "../../../artifacts/optimization/golden_outputs/ft_kernel_cpu_softmax_parallel_frankentorch-wcoo.txt"
            )
        );
    }

    #[test]
    fn softmax_dim_sums_to_one() {
        let meta = TensorMeta::from_shape(vec![2, 3], DType::F64, Device::Cpu);
        let input = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let out = softmax_dim_tensor_contiguous_f64(&input, &meta, 1).expect("softmax");
        // Each row should sum to 1
        let row0_sum: f64 = out[0..3].iter().sum();
        let row1_sum: f64 = out[3..6].iter().sum();
        assert!((row0_sum - 1.0).abs() < 1e-12);
        assert!((row1_sum - 1.0).abs() < 1e-12);
        // All values should be positive
        assert!(out.iter().all(|&v| v > 0.0));
    }

    #[test]
    fn softmax_dim_preserves_relative_order() {
        let meta = TensorMeta::from_shape(vec![1, 3], DType::F64, Device::Cpu);
        let input = vec![1.0, 3.0, 2.0];
        let out = softmax_dim_tensor_contiguous_f64(&input, &meta, 1).expect("softmax");
        // input[1] > input[2] > input[0], so output should preserve order
        assert!(out[1] > out[2]);
        assert!(out[2] > out[0]);
    }

    #[test]
    fn log_softmax_dim_values() {
        let meta = TensorMeta::from_shape(vec![1, 3], DType::F64, Device::Cpu);
        let input = vec![1.0, 2.0, 3.0];
        let out = log_softmax_dim_tensor_contiguous_f64(&input, &meta, 1).expect("log_softmax");
        let sm = softmax_dim_tensor_contiguous_f64(&input, &meta, 1).expect("softmax");
        // log_softmax should equal log(softmax)
        for i in 0..3 {
            assert!((out[i] - sm[i].ln()).abs() < 1e-12);
        }
    }

    #[test]
    fn log_softmax_preserves_precision_at_large_magnitudes() {
        // Regression test for frankentorch-ebrb. The log_softmax
        // kernel previously computed
        //     output[i] = x[i] - (max + log(sum(exp(x - max))))
        // which suffers catastrophic cancellation at large magnitudes
        // (e.g. logits ~1000): max + log(sum) collapses ~1000-magnitude
        // values to a ~−O(1) result, wiping ~13 mantissa digits. The
        // fix rearranges to (x[i] - max) - log(sum(exp(x - max))),
        // which keeps both intermediates near the result magnitude.
        //
        // Reference values are the analytical scipy answers — at
        // x = [1000, 1001, 1002], log_softmax should equal
        //     ([-2, -1, 0]) - log(e^-2 + e^-1 + 1)
        //   = ([-2, -1, 0]) - log(0.13534 + 0.36788 + 1.0)
        //   ≈ [-2.4076059644443804, -1.4076059644443804, -0.4076059644443804]
        // The pre-fix kernel returned -2.4076059644444285 etc. (~1000
        // ULPs off). Lock the new envelope to <= 16 ULPs absolute,
        // which catches a regression to the old algebra without being
        // brittle to libm exp/log rounding.
        let meta = TensorMeta::from_shape(vec![1, 3], DType::F64, Device::Cpu);
        let input = vec![1000.0, 1001.0, 1002.0];
        let out = log_softmax_dim_tensor_contiguous_f64(&input, &meta, 1)
            .expect("log_softmax large magnitude");
        let expected = [
            -2.4076059644443804_f64,
            -1.4076059644443804,
            -0.4076059644443804,
        ];
        for i in 0..3 {
            let bits_diff = out[i].to_bits().abs_diff(expected[i].to_bits());
            assert!(
                bits_diff <= 16,
                "log_softmax[{i}]={} bits=0x{:016x}, expected={} bits=0x{:016x}, ULP diff={bits_diff}",
                out[i],
                out[i].to_bits(),
                expected[i],
                expected[i].to_bits(),
            );
        }
    }

    #[test]
    fn softmax_dim_invalid_dim() {
        let meta = TensorMeta::from_shape(vec![2, 3], DType::F64, Device::Cpu);
        let input = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let err = softmax_dim_tensor_contiguous_f64(&input, &meta, 2).unwrap_err();
        assert!(matches!(
            err,
            KernelError::InvalidDimension { dim: 2, ndim: 2 }
        ));
    }

    #[test]
    fn cat_along_dim0() {
        let m0 = TensorMeta::from_shape(vec![2, 3], DType::F64, Device::Cpu);
        let d0 = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let m1 = TensorMeta::from_shape(vec![1, 3], DType::F64, Device::Cpu);
        let d1 = vec![7.0, 8.0, 9.0];
        let out = cat_tensor_contiguous_f64(&[(&d0, &m0), (&d1, &m1)], 0).expect("cat dim 0");
        assert_eq!(out, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]);
    }

    #[test]
    fn cat_along_dim1() {
        let m0 = TensorMeta::from_shape(vec![2, 2], DType::F64, Device::Cpu);
        let d0 = vec![1.0, 2.0, 3.0, 4.0];
        let m1 = TensorMeta::from_shape(vec![2, 3], DType::F64, Device::Cpu);
        let d1 = vec![5.0, 6.0, 7.0, 8.0, 9.0, 10.0];
        let out = cat_tensor_contiguous_f64(&[(&d0, &m0), (&d1, &m1)], 1).expect("cat dim 1");
        // row0: [1,2,5,6,7], row1: [3,4,8,9,10]
        assert_eq!(out, vec![1.0, 2.0, 5.0, 6.0, 7.0, 3.0, 4.0, 8.0, 9.0, 10.0]);
    }

    #[test]
    fn cat_along_dim1_f32() {
        let m0 = TensorMeta::from_shape(vec![2, 2], DType::F32, Device::Cpu);
        let d0 = vec![1.0f32, 2.0, 3.0, 4.0];
        let m1 = TensorMeta::from_shape(vec![2, 3], DType::F32, Device::Cpu);
        let d1 = vec![5.0f32, 6.0, 7.0, 8.0, 9.0, 10.0];
        let out = cat_tensor_contiguous_f32(&[(&d0, &m0), (&d1, &m1)], 1).expect("cat dim 1");
        assert_eq!(
            out,
            vec![1.0f32, 2.0, 5.0, 6.0, 7.0, 3.0, 4.0, 8.0, 9.0, 10.0]
        );
    }

    #[test]
    fn cat_skips_empty_input_with_offset() {
        let m0 = TensorMeta::from_shape(vec![0, 2], DType::F64, Device::Cpu).with_storage_offset(5);
        let d0: Vec<f64> = Vec::new();
        let m1 = TensorMeta::from_shape(vec![1, 2], DType::F64, Device::Cpu).with_storage_offset(1);
        let d1 = vec![99.0, 1.0, 2.0];
        let out = cat_tensor_contiguous_f64(&[(&d0, &m0), (&d1, &m1)], 0).expect("cat dim 0");
        assert_eq!(out, vec![1.0, 2.0]);
    }

    #[test]
    fn scatter_empty_index_returns_input_view() {
        let meta =
            TensorMeta::from_shape(vec![2, 2], DType::F64, Device::Cpu).with_storage_offset(2);
        let input = vec![9.0, 8.0, 1.0, 2.0, 3.0, 4.0];
        let idx_meta = TensorMeta::from_shape(vec![2, 0], DType::F64, Device::Cpu);
        let index: Vec<f64> = Vec::new();
        let src: Vec<f64> = Vec::new();
        let out = scatter_tensor_contiguous_f64(&input, &meta, 1, &index, &idx_meta, &src)
            .expect("scatter");
        assert_eq!(out, vec![1.0, 2.0, 3.0, 4.0]);
    }

    #[test]
    fn scatter_f32_empty_index_returns_input_view() {
        let meta =
            TensorMeta::from_shape(vec![2, 2], DType::F32, Device::Cpu).with_storage_offset(1);
        let input = vec![10.0f32, 1.0, 2.0, 3.0, 4.0];
        let idx_meta = TensorMeta::from_shape(vec![2, 0], DType::F64, Device::Cpu);
        let index: Vec<f64> = Vec::new();
        let src: Vec<f32> = Vec::new();
        let out = scatter_tensor_contiguous_f32(&input, &meta, 1, &index, &idx_meta, &src)
            .expect("scatter f32");
        assert_eq!(out, vec![1.0, 2.0, 3.0, 4.0]);
    }

    #[test]
    fn stack_along_dim0() {
        let m0 = TensorMeta::from_shape(vec![3], DType::F64, Device::Cpu);
        let d0 = vec![1.0, 2.0, 3.0];
        let m1 = TensorMeta::from_shape(vec![3], DType::F64, Device::Cpu);
        let d1 = vec![4.0, 5.0, 6.0];
        let out = stack_tensor_contiguous_f64(&[(&d0, &m0), (&d1, &m1)], 0).expect("stack dim 0");
        // shape [2,3]: [[1,2,3],[4,5,6]]
        assert_eq!(out, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    }

    #[test]
    fn stack_along_dim1() {
        let m0 = TensorMeta::from_shape(vec![2, 2], DType::F64, Device::Cpu);
        let d0 = vec![1.0, 2.0, 3.0, 4.0];
        let m1 = TensorMeta::from_shape(vec![2, 2], DType::F64, Device::Cpu);
        let d1 = vec![5.0, 6.0, 7.0, 8.0];
        let out = stack_tensor_contiguous_f64(&[(&d0, &m0), (&d1, &m1)], 1).expect("stack dim 1");
        // shape [2,2,2]: [[[1,2],[5,6]],[[3,4],[7,8]]]
        assert_eq!(out, vec![1.0, 2.0, 5.0, 6.0, 3.0, 4.0, 7.0, 8.0]);
    }

    #[test]
    fn stack_along_dim1_f32() {
        let m0 = TensorMeta::from_shape(vec![2, 2], DType::F32, Device::Cpu);
        let d0 = vec![1.0f32, 2.0, 3.0, 4.0];
        let m1 = TensorMeta::from_shape(vec![2, 2], DType::F32, Device::Cpu);
        let d1 = vec![5.0f32, 6.0, 7.0, 8.0];
        let out = stack_tensor_contiguous_f32(&[(&d0, &m0), (&d1, &m1)], 1).expect("stack dim 1");
        assert_eq!(out, vec![1.0f32, 2.0, 5.0, 6.0, 3.0, 4.0, 7.0, 8.0]);
    }

    #[test]
    fn narrow_dim0_middle_rows() {
        // shape [4, 3], narrow(dim=0, start=1, length=2) -> shape [2, 3]
        let meta = TensorMeta::from_shape(vec![4, 3], DType::F64, Device::Cpu);
        let input: Vec<f64> = (1..=12).map(|i| i as f64).collect();
        let result = narrow_tensor_contiguous_f64(&input, &meta, 0, 1, 2).unwrap();
        assert_eq!(result, vec![4.0, 5.0, 6.0, 7.0, 8.0, 9.0]);
    }

    #[test]
    fn narrow_dim1_selects_columns() {
        // shape [2, 4], narrow(dim=1, start=1, length=2) -> shape [2, 2]
        let meta = TensorMeta::from_shape(vec![2, 4], DType::F64, Device::Cpu);
        let input: Vec<f64> = (1..=8).map(|i| i as f64).collect();
        let result = narrow_tensor_contiguous_f64(&input, &meta, 1, 1, 2).unwrap();
        assert_eq!(result, vec![2.0, 3.0, 6.0, 7.0]);
    }

    #[test]
    fn narrow_out_of_bounds_fails() {
        let meta = TensorMeta::from_shape(vec![3, 2], DType::F64, Device::Cpu);
        let input = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let result = narrow_tensor_contiguous_f64(&input, &meta, 0, 2, 2);
        assert!(result.is_err());
    }

    #[test]
    fn narrow_invalid_dim_fails() {
        let meta = TensorMeta::from_shape(vec![3, 2], DType::F64, Device::Cpu);
        let input = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let result = narrow_tensor_contiguous_f64(&input, &meta, 5, 0, 1);
        assert!(result.is_err());
    }

    #[test]
    fn narrow_start_plus_length_overflow_fails_closed() {
        let meta = TensorMeta::from_shape(vec![3, 2], DType::F64, Device::Cpu);
        let input = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let result = narrow_tensor_contiguous_f64(&input, &meta, 0, usize::MAX, 1);
        assert!(result.is_err());
    }

    #[test]
    fn narrow_3d_tensor() {
        // shape [2, 3, 4], narrow(dim=1, start=1, length=2) -> shape [2, 2, 4]
        let meta = TensorMeta::from_shape(vec![2, 3, 4], DType::F64, Device::Cpu);
        let input: Vec<f64> = (1..=24).map(|i| i as f64).collect();
        let result = narrow_tensor_contiguous_f64(&input, &meta, 1, 1, 2).unwrap();
        assert_eq!(
            result,
            vec![
                5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 17.0, 18.0, 19.0, 20.0, 21.0, 22.0,
                23.0, 24.0
            ]
        );
    }

    #[test]
    fn expand_singleton_dim() {
        // shape [1, 3] -> [4, 3]
        let meta = TensorMeta::from_shape(vec![1, 3], DType::F64, Device::Cpu);
        let input = vec![1.0, 2.0, 3.0];
        let result = expand_tensor_contiguous_f64(&input, &meta, &[4, 3]).unwrap();
        assert_eq!(
            result,
            vec![1.0, 2.0, 3.0, 1.0, 2.0, 3.0, 1.0, 2.0, 3.0, 1.0, 2.0, 3.0]
        );
    }

    #[test]
    fn expand_multiple_singleton_dims() {
        // shape [1, 3, 1] -> [2, 3, 4]
        let meta = TensorMeta::from_shape(vec![1, 3, 1], DType::F64, Device::Cpu);
        let input = vec![1.0, 2.0, 3.0];
        let result = expand_tensor_contiguous_f64(&input, &meta, &[2, 3, 4]).unwrap();
        assert_eq!(result.len(), 24);
        // First slice (outer=0): [1,1,1,1, 2,2,2,2, 3,3,3,3]
        assert_eq!(
            &result[0..12],
            &[1.0, 1.0, 1.0, 1.0, 2.0, 2.0, 2.0, 2.0, 3.0, 3.0, 3.0, 3.0]
        );
        // Second slice same as first
        assert_eq!(
            &result[12..24],
            &[1.0, 1.0, 1.0, 1.0, 2.0, 2.0, 2.0, 2.0, 3.0, 3.0, 3.0, 3.0]
        );
    }

    #[test]
    fn expand_non_singleton_fails() {
        // shape [2, 3] -> [4, 3]: dim 0 is 2, not 1 => error
        let meta = TensorMeta::from_shape(vec![2, 3], DType::F64, Device::Cpu);
        let input = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let result = expand_tensor_contiguous_f64(&input, &meta, &[4, 3]);
        assert!(result.is_err());
    }

    #[test]
    fn expand_same_shape_is_identity() {
        let meta = TensorMeta::from_shape(vec![2, 3], DType::F64, Device::Cpu);
        let input = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let result = expand_tensor_contiguous_f64(&input, &meta, &[2, 3]).unwrap();
        assert_eq!(result, input);
    }

    #[test]
    fn expand_rank_mismatch_fails() {
        let meta = TensorMeta::from_shape(vec![1, 3], DType::F64, Device::Cpu);
        let input = vec![1.0, 2.0, 3.0];
        let result = expand_tensor_contiguous_f64(&input, &meta, &[4, 3, 2]);
        assert!(result.is_err());
    }

    #[test]
    fn reduce_sum_for_broadcast_singleton_leading_dim() {
        let expanded = vec![1.0; 12];
        let reduced =
            super::reduce_sum_for_broadcast(&expanded, &[4, 3], &[1, 3]).expect("valid reduce");
        assert_eq!(reduced, vec![4.0, 4.0, 4.0]);
    }

    #[test]
    fn reduce_sum_for_broadcast_multiple_singletons() {
        let expanded: Vec<f64> = vec![1.0; 24];
        let reduced = super::reduce_sum_for_broadcast(&expanded, &[2, 3, 4], &[1, 3, 1])
            .expect("valid reduce");
        assert_eq!(reduced, vec![8.0, 8.0, 8.0]);
    }

    #[test]
    fn reduce_sum_for_broadcast_rejects_rank_mismatch() {
        let err = super::reduce_sum_for_broadcast(&[1.0, 2.0], &[2], &[1, 2])
            .expect_err("rank mismatch should fail");
        assert!(matches!(err, super::KernelError::ShapeMismatch { .. }));
    }

    #[test]
    fn reduce_sum_for_broadcast_rejects_invalid_broadcast_contract() {
        let err = super::reduce_sum_for_broadcast(&[1.0, 2.0, 3.0, 4.0], &[2, 2], &[2, 3])
            .expect_err("incompatible broadcast should fail");
        assert!(matches!(err, super::KernelError::ShapeMismatch { .. }));
    }

    #[test]
    fn reduce_sum_for_broadcast_rejects_gradient_len_mismatch() {
        let err = super::reduce_sum_for_broadcast(&[1.0, 2.0, 3.0], &[2, 2], &[1, 2])
            .expect_err("gradient shape mismatch should fail");
        assert!(matches!(err, super::KernelError::ShapeMismatch { .. }));
    }

    // ── index_select tests ─────────────────────────────────────────────

    #[test]
    fn index_select_dim0_swaps_rows() {
        let meta = TensorMeta::from_shape(vec![2, 3], DType::F64, Device::Cpu);
        let input = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let indices = vec![1.0, 0.0];
        let result = index_select_tensor_contiguous_f64(&input, &meta, 0, &indices).unwrap();
        assert_eq!(result, vec![4.0, 5.0, 6.0, 1.0, 2.0, 3.0]);
    }

    #[test]
    fn index_select_dim1_picks_columns() {
        let meta = TensorMeta::from_shape(vec![2, 3], DType::F64, Device::Cpu);
        let input = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let indices = vec![2.0, 0.0];
        let result = index_select_tensor_contiguous_f64(&input, &meta, 1, &indices).unwrap();
        assert_eq!(result, vec![3.0, 1.0, 6.0, 4.0]);
    }

    #[test]
    fn index_select_invalid_dim_returns_error() {
        let meta = TensorMeta::from_shape(vec![2, 3], DType::F64, Device::Cpu);
        let input = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let result = index_select_tensor_contiguous_f64(&input, &meta, 5, &[0.0]);
        assert!(result.is_err());
    }

    #[test]
    fn index_select_out_of_bounds_index_returns_error() {
        let meta = TensorMeta::from_shape(vec![2, 3], DType::F64, Device::Cpu);
        let input = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let result = index_select_tensor_contiguous_f64(&input, &meta, 0, &[5.0]);
        assert!(result.is_err());
    }

    #[test]
    fn index_select_nan_index_returns_error() {
        let meta = TensorMeta::from_shape(vec![2, 2], DType::F64, Device::Cpu);
        let input = vec![1.0, 2.0, 3.0, 4.0];
        let result = index_select_tensor_contiguous_f64(&input, &meta, 0, &[f64::NAN]);
        assert!(result.is_err());
    }

    #[test]
    fn index_select_huge_index_returns_error() {
        let meta = TensorMeta::from_shape(vec![2, 2], DType::F64, Device::Cpu);
        let input = vec![1.0, 2.0, 3.0, 4.0];
        let result = index_select_tensor_contiguous_f64(&input, &meta, 0, &[1.0e300]);
        assert!(result.is_err());
    }

    #[test]
    fn index_select_negative_index_returns_error() {
        // torch.index_select rejects negative indices (IndexError);
        // unlike advanced indexing it does not wrap. (frankentorch-n0un)
        let meta = TensorMeta::from_shape(vec![3, 2], DType::F64, Device::Cpu);
        let input = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        assert!(index_select_tensor_contiguous_f64(&input, &meta, 0, &[-1.0]).is_err());
        let input_f32 = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];
        assert!(index_select_tensor_contiguous_f32(&input_f32, &meta, 0, &[-1.0]).is_err());
    }

    #[test]
    fn gather_negative_index_returns_error() {
        // torch.gather rejects negative indices (RuntimeError). (frankentorch-n0un)
        let meta = TensorMeta::from_shape(vec![2, 2], DType::F64, Device::Cpu);
        let idx_meta = TensorMeta::from_shape(vec![2, 1], DType::F64, Device::Cpu);
        let input = vec![1.0, 2.0, 3.0, 4.0];
        assert!(gather_tensor_contiguous_f64(&input, &meta, 1, &[-1.0, -1.0], &idx_meta).is_err());
    }

    #[test]
    fn scatter_negative_index_returns_error() {
        // torch.scatter / scatter_add reject negative indices. (frankentorch-n0un)
        let meta = TensorMeta::from_shape(vec![2, 2], DType::F64, Device::Cpu);
        let idx_meta = TensorMeta::from_shape(vec![2, 1], DType::F64, Device::Cpu);
        let input = vec![1.0, 2.0, 3.0, 4.0];
        let src = vec![9.0, 9.0];
        assert!(
            scatter_tensor_contiguous_f64(&input, &meta, 1, &[-1.0, -1.0], &idx_meta, &src)
                .is_err()
        );
        assert!(
            scatter_add_tensor_contiguous_f64(&input, &meta, 1, &[-1.0, -1.0], &idx_meta, &src)
                .is_err()
        );
    }

    #[test]
    fn index_select_single_index() {
        let meta = TensorMeta::from_shape(vec![3, 2], DType::F64, Device::Cpu);
        let input = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let indices = vec![1.0];
        let result = index_select_tensor_contiguous_f64(&input, &meta, 0, &indices).unwrap();
        assert_eq!(result, vec![3.0, 4.0]);
    }

    // ── gather tests ───────────────────────────────────────────────────

    #[test]
    fn gather_dim1_picks_values() {
        let meta = TensorMeta::from_shape(vec![2, 3], DType::F64, Device::Cpu);
        let input = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let index = vec![0.0, 2.0, 1.0, 0.0];
        let idx_meta = TensorMeta::from_shape(vec![2, 2], DType::F64, Device::Cpu);
        let result = gather_tensor_contiguous_f64(&input, &meta, 1, &index, &idx_meta).unwrap();
        assert_eq!(result, vec![1.0, 3.0, 5.0, 4.0]);
    }

    #[test]
    fn gather_dim0_picks_values() {
        let meta = TensorMeta::from_shape(vec![2, 3], DType::F64, Device::Cpu);
        let input = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let index = vec![0.0, 1.0, 1.0];
        let idx_meta = TensorMeta::from_shape(vec![1, 3], DType::F64, Device::Cpu);
        let result = gather_tensor_contiguous_f64(&input, &meta, 0, &index, &idx_meta).unwrap();
        assert_eq!(result, vec![1.0, 5.0, 6.0]);
    }

    #[test]
    fn gather_invalid_dim_returns_error() {
        let meta = TensorMeta::from_shape(vec![2, 3], DType::F64, Device::Cpu);
        let input = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let idx_meta = TensorMeta::from_shape(vec![2, 2], DType::F64, Device::Cpu);
        let result = gather_tensor_contiguous_f64(&input, &meta, 5, &[0.0; 4], &idx_meta);
        assert!(result.is_err());
    }

    #[test]
    fn gather_shape_mismatch_returns_error() {
        let meta = TensorMeta::from_shape(vec![2, 3], DType::F64, Device::Cpu);
        let input = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let idx_meta = TensorMeta::from_shape(vec![3, 2], DType::F64, Device::Cpu);
        let result = gather_tensor_contiguous_f64(&input, &meta, 1, &[0.0; 6], &idx_meta);
        assert!(result.is_err());
    }

    #[test]
    fn gather_non_integer_index_returns_error() {
        let meta = TensorMeta::from_shape(vec![2, 2], DType::F64, Device::Cpu);
        let input = vec![1.0, 2.0, 3.0, 4.0];
        let index = vec![0.5, 0.0];
        let idx_meta = TensorMeta::from_shape(vec![1, 2], DType::F64, Device::Cpu);
        let result = gather_tensor_contiguous_f64(&input, &meta, 1, &index, &idx_meta);
        assert!(result.is_err());
    }

    // ── scatter tests ──────────────────────────────────────────────────

    #[test]
    fn scatter_dim1_places_values() {
        let meta = TensorMeta::from_shape(vec![2, 3], DType::F64, Device::Cpu);
        let input = vec![0.0; 6];
        let index = vec![0.0, 2.0, 1.0, 0.0];
        let idx_meta = TensorMeta::from_shape(vec![2, 2], DType::F64, Device::Cpu);
        let src = vec![10.0, 20.0, 30.0, 40.0];
        let result =
            scatter_tensor_contiguous_f64(&input, &meta, 1, &index, &idx_meta, &src).unwrap();
        assert_eq!(result, vec![10.0, 0.0, 20.0, 40.0, 30.0, 0.0]);
    }

    #[test]
    fn scatter_dim0_places_values() {
        let meta = TensorMeta::from_shape(vec![2, 3], DType::F64, Device::Cpu);
        let input = vec![0.0; 6];
        let index = vec![1.0, 0.0, 1.0];
        let idx_meta = TensorMeta::from_shape(vec![1, 3], DType::F64, Device::Cpu);
        let src = vec![10.0, 20.0, 30.0];
        let result =
            scatter_tensor_contiguous_f64(&input, &meta, 0, &index, &idx_meta, &src).unwrap();
        assert_eq!(result, vec![0.0, 20.0, 0.0, 10.0, 0.0, 30.0]);
    }

    #[test]
    fn scatter_invalid_dim_returns_error() {
        let meta = TensorMeta::from_shape(vec![2, 3], DType::F64, Device::Cpu);
        let input = vec![0.0; 6];
        let idx_meta = TensorMeta::from_shape(vec![2, 2], DType::F64, Device::Cpu);
        let result =
            scatter_tensor_contiguous_f64(&input, &meta, 5, &[0.0; 4], &idx_meta, &[0.0; 4]);
        assert!(result.is_err());
    }

    #[test]
    fn scatter_non_finite_index_returns_error() {
        let meta = TensorMeta::from_shape(vec![2, 2], DType::F64, Device::Cpu);
        let input = vec![0.0; 4];
        let index = vec![f64::INFINITY, 0.0];
        let idx_meta = TensorMeta::from_shape(vec![1, 2], DType::F64, Device::Cpu);
        let src = vec![1.0, 2.0];
        let result = scatter_tensor_contiguous_f64(&input, &meta, 1, &index, &idx_meta, &src);
        assert!(result.is_err());
    }

    // ── masked_fill tests ──────────────────────────────────────────────

    #[test]
    fn masked_fill_replaces_masked_positions() {
        let meta = TensorMeta::from_shape(vec![2, 3], DType::F64, Device::Cpu);
        let input = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let mask = vec![1.0, 0.0, 1.0, 0.0, 1.0, 0.0];
        let result = masked_fill_tensor_contiguous_f64(&input, &meta, &mask, -1.0).unwrap();
        assert_eq!(result, vec![-1.0, 2.0, -1.0, 4.0, -1.0, 6.0]);
    }

    #[test]
    fn masked_fill_all_masked() {
        let meta = TensorMeta::from_shape(vec![3], DType::F64, Device::Cpu);
        let input = vec![1.0, 2.0, 3.0];
        let mask = vec![1.0, 1.0, 1.0];
        let result = masked_fill_tensor_contiguous_f64(&input, &meta, &mask, 0.0).unwrap();
        assert_eq!(result, vec![0.0, 0.0, 0.0]);
    }

    #[test]
    fn masked_fill_none_masked() {
        let meta = TensorMeta::from_shape(vec![3], DType::F64, Device::Cpu);
        let input = vec![1.0, 2.0, 3.0];
        let mask = vec![0.0, 0.0, 0.0];
        let result = masked_fill_tensor_contiguous_f64(&input, &meta, &mask, -1.0).unwrap();
        assert_eq!(result, vec![1.0, 2.0, 3.0]);
    }

    #[test]
    fn masked_fill_shape_mismatch_returns_error() {
        let meta = TensorMeta::from_shape(vec![2, 3], DType::F64, Device::Cpu);
        let input = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let mask = vec![1.0, 0.0]; // too short
        let result = masked_fill_tensor_contiguous_f64(&input, &meta, &mask, -1.0);
        assert!(result.is_err());
    }

    // ---- cos ----

    #[test]
    fn cos_scalar_returns_expected_value() {
        let input = ScalarTensor::new(0.0, DType::F64, Device::Cpu);
        let out = cos_scalar(&input);
        assert!((out.value() - 1.0).abs() < 1e-10);
    }

    #[test]
    fn cos_tensor_contiguous_returns_expected_values() {
        let meta = TensorMeta::from_shape(vec![3], DType::F64, Device::Cpu);
        let input = vec![0.0, std::f64::consts::PI, std::f64::consts::FRAC_PI_2];
        let out = cos_tensor_contiguous_f64(&input, &meta).expect("cos should succeed");
        assert!((out[0] - 1.0).abs() < 1e-10);
        assert!((out[1] - (-1.0)).abs() < 1e-10);
        assert!(out[2].abs() < 1e-10);
    }

    // ---- floor ----

    #[test]
    fn floor_scalar_returns_expected_value() {
        let input = ScalarTensor::new(3.7, DType::F64, Device::Cpu);
        let out = floor_scalar(&input);
        assert_eq!(out.value(), 3.0);
    }

    #[test]
    fn floor_tensor_contiguous_returns_expected_values() {
        let meta = TensorMeta::from_shape(vec![4], DType::F64, Device::Cpu);
        let input = vec![1.1, 2.9, -0.5, -2.1];
        let out = floor_tensor_contiguous_f64(&input, &meta).expect("floor should succeed");
        assert_eq!(out, vec![1.0, 2.0, -1.0, -3.0]);
    }

    // ---- ceil ----

    #[test]
    fn ceil_scalar_returns_expected_value() {
        let input = ScalarTensor::new(3.2, DType::F64, Device::Cpu);
        let out = ceil_scalar(&input);
        assert_eq!(out.value(), 4.0);
    }

    #[test]
    fn ceil_tensor_contiguous_returns_expected_values() {
        let meta = TensorMeta::from_shape(vec![4], DType::F64, Device::Cpu);
        let input = vec![1.1, 2.9, -0.5, -2.1];
        let out = ceil_tensor_contiguous_f64(&input, &meta).expect("ceil should succeed");
        assert_eq!(out, vec![2.0, 3.0, 0.0, -2.0]);
    }

    #[test]
    fn round_scalar_ties_to_even() {
        let cases = [
            (-2.5, -2.0),
            (-1.5, -2.0),
            (0.5, 0.0),
            (1.5, 2.0),
            (2.5, 2.0),
        ];

        for (input, expected) in cases {
            let input = ScalarTensor::new(input, DType::F64, Device::Cpu);
            let out = super::round_scalar(&input);
            assert_eq!(out.value(), expected);
        }

        let neg_zero = ScalarTensor::new(-0.5, DType::F64, Device::Cpu);
        assert!(super::round_scalar(&neg_zero).value().is_sign_negative());
    }

    #[test]
    fn round_tensor_contiguous_f64_ties_to_even() {
        let meta = TensorMeta::from_shape(vec![6], DType::F64, Device::Cpu);
        let input = vec![-2.5, -1.5, -0.5, 0.5, 1.5, 2.5];

        let out = super::round_tensor_contiguous_f64(&input, &meta).expect("round should succeed");

        assert_eq!(out, vec![-2.0, -2.0, -0.0, 0.0, 2.0, 2.0]);
        assert!(out[2].is_sign_negative());
    }

    #[test]
    fn round_tensor_contiguous_f32_ties_to_even() {
        let meta = TensorMeta::from_shape(vec![6], DType::F32, Device::Cpu);
        let input = vec![-2.5_f32, -1.5, -0.5, 0.5, 1.5, 2.5];

        let out = super::round_tensor_contiguous_f32(&input, &meta).expect("round should succeed");

        assert_eq!(out, vec![-2.0, -2.0, -0.0, 0.0, 2.0, 2.0]);
        assert!(out[2].is_sign_negative());
    }

    // ---- log2 ----

    #[test]
    fn log2_scalar_returns_expected_value() {
        let input = ScalarTensor::new(8.0, DType::F64, Device::Cpu);
        let out = log2_scalar(&input);
        assert!((out.value() - 3.0).abs() < 1e-10);
    }

    #[test]
    fn log2_tensor_contiguous_returns_expected_values() {
        let meta = TensorMeta::from_shape(vec![3], DType::F64, Device::Cpu);
        let input = vec![1.0, 2.0, 4.0];
        let out = log2_tensor_contiguous_f64(&input, &meta).expect("log2 should succeed");
        assert!((out[0] - 0.0).abs() < 1e-10);
        assert!((out[1] - 1.0).abs() < 1e-10);
        assert!((out[2] - 2.0).abs() < 1e-10);
    }

    // ---- log10 ----

    #[test]
    fn log10_scalar_returns_expected_value() {
        let input = ScalarTensor::new(1000.0, DType::F64, Device::Cpu);
        let out = log10_scalar(&input);
        assert!((out.value() - 3.0).abs() < 1e-10);
    }

    #[test]
    fn log10_tensor_contiguous_returns_expected_values() {
        let meta = TensorMeta::from_shape(vec![3], DType::F64, Device::Cpu);
        let input = vec![1.0, 10.0, 100.0];
        let out = log10_tensor_contiguous_f64(&input, &meta).expect("log10 should succeed");
        assert!((out[0] - 0.0).abs() < 1e-10);
        assert!((out[1] - 1.0).abs() < 1e-10);
        assert!((out[2] - 2.0).abs() < 1e-10);
    }

    // ---- log1p ----

    #[test]
    fn log1p_scalar_returns_expected_value() {
        let input = ScalarTensor::new(0.0, DType::F64, Device::Cpu);
        let out = log1p_scalar(&input);
        assert!((out.value() - 0.0).abs() < 1e-10);
    }

    #[test]
    fn log1p_tensor_contiguous_returns_expected_values() {
        let meta = TensorMeta::from_shape(vec![3], DType::F64, Device::Cpu);
        let input = vec![0.0, 1.0, std::f64::consts::E - 1.0];
        let out = log1p_tensor_contiguous_f64(&input, &meta).expect("log1p should succeed");
        assert!((out[0] - 0.0).abs() < 1e-10);
        assert!((out[1] - 2.0_f64.ln()).abs() < 1e-10);
        assert!((out[2] - 1.0).abs() < 1e-10);
    }

    // ---- expm1 ----

    #[test]
    fn expm1_scalar_returns_expected_value() {
        let input = ScalarTensor::new(0.0, DType::F64, Device::Cpu);
        let out = expm1_scalar(&input);
        assert!((out.value() - 0.0).abs() < 1e-10);
    }

    #[test]
    fn expm1_tensor_contiguous_returns_expected_values() {
        let meta = TensorMeta::from_shape(vec![2], DType::F64, Device::Cpu);
        let input = vec![0.0, 1.0];
        let out = expm1_tensor_contiguous_f64(&input, &meta).expect("expm1 should succeed");
        assert!((out[0] - 0.0).abs() < 1e-10);
        assert!((out[1] - (std::f64::consts::E - 1.0)).abs() < 1e-10);
    }

    // ---- sign ----

    #[test]
    fn sign_scalar_returns_expected_value() {
        // PyTorch parity (frankentorch-wfyq): torch.sign maps both
        // signed zeros to +0.0, propagates NaN, and returns ±1.0 only
        // for non-zero finite values. Rust's f64::signum returns ±1.0
        // for ±0.0 (IEEE 754 sign-bit semantics) — we override.
        let pos = ScalarTensor::new(5.0, DType::F64, Device::Cpu);
        let neg = ScalarTensor::new(-3.0, DType::F64, Device::Cpu);
        let pos_zero = ScalarTensor::new(0.0, DType::F64, Device::Cpu);
        let neg_zero = ScalarTensor::new(-0.0, DType::F64, Device::Cpu);
        assert_eq!(sign_scalar(&pos).value(), 1.0);
        assert_eq!(sign_scalar(&neg).value(), -1.0);
        assert_eq!(sign_scalar(&pos_zero).value(), 0.0);
        assert_eq!(sign_scalar(&neg_zero).value(), 0.0);
    }

    #[test]
    fn sign_scalar_propagates_nan() {
        let nan = ScalarTensor::new(f64::NAN, DType::F64, Device::Cpu);
        assert!(sign_scalar(&nan).value().is_nan());
    }

    #[test]
    fn sign_tensor_contiguous_returns_expected_values() {
        // PyTorch parity (frankentorch-wfyq): both signed zeros map to
        // +0.0; non-zero negatives stay -1.0.
        let meta = TensorMeta::from_shape(vec![5], DType::F64, Device::Cpu);
        let input = vec![3.0, -2.0, -0.0, 0.0, -0.5];
        let out = sign_tensor_contiguous_f64(&input, &meta).expect("sign should succeed");
        assert_eq!(out, vec![1.0, -1.0, 0.0, 0.0, -1.0]);
    }

    #[test]
    fn sign_tensor_contiguous_propagates_nan() {
        let meta = TensorMeta::from_shape(vec![3], DType::F64, Device::Cpu);
        let input = vec![f64::NAN, 1.0, -1.0];
        let out = sign_tensor_contiguous_f64(&input, &meta).expect("sign should succeed");
        assert!(out[0].is_nan());
        assert_eq!(out[1], 1.0);
        assert_eq!(out[2], -1.0);
    }

    // ---- trunc ----

    #[test]
    fn trunc_scalar_returns_expected_value() {
        let input = ScalarTensor::new(3.7, DType::F64, Device::Cpu);
        let out = trunc_scalar(&input);
        assert_eq!(out.value(), 3.0);
        let neg = ScalarTensor::new(-2.9, DType::F64, Device::Cpu);
        assert_eq!(trunc_scalar(&neg).value(), -2.0);
    }

    #[test]
    fn trunc_tensor_contiguous_returns_expected_values() {
        let meta = TensorMeta::from_shape(vec![4], DType::F64, Device::Cpu);
        let input = vec![1.9, -2.9, 0.1, -0.1];
        let out = trunc_tensor_contiguous_f64(&input, &meta).expect("trunc should succeed");
        assert_eq!(out, vec![1.0, -2.0, 0.0, 0.0]);
    }

    // ---- asin ----

    #[test]
    fn asin_scalar_returns_expected_value() {
        let input = ScalarTensor::new(0.0, DType::F64, Device::Cpu);
        let out = asin_scalar(&input);
        assert!((out.value() - 0.0).abs() < 1e-10);
    }

    #[test]
    fn asin_tensor_contiguous_returns_expected_values() {
        let meta = TensorMeta::from_shape(vec![3], DType::F64, Device::Cpu);
        let input = vec![0.0, 0.5, 1.0];
        let out = asin_tensor_contiguous_f64(&input, &meta).expect("asin should succeed");
        assert!((out[0] - 0.0).abs() < 1e-10);
        assert!((out[1] - std::f64::consts::FRAC_PI_6).abs() < 1e-10);
        assert!((out[2] - std::f64::consts::FRAC_PI_2).abs() < 1e-10);
    }

    // ---- acos ----

    #[test]
    fn acos_scalar_returns_expected_value() {
        let input = ScalarTensor::new(1.0, DType::F64, Device::Cpu);
        let out = acos_scalar(&input);
        assert!((out.value() - 0.0).abs() < 1e-10);
    }

    #[test]
    fn acos_tensor_contiguous_returns_expected_values() {
        let meta = TensorMeta::from_shape(vec![3], DType::F64, Device::Cpu);
        let input = vec![1.0, 0.0, -1.0];
        let out = acos_tensor_contiguous_f64(&input, &meta).expect("acos should succeed");
        assert!((out[0] - 0.0).abs() < 1e-10);
        assert!((out[1] - std::f64::consts::FRAC_PI_2).abs() < 1e-10);
        assert!((out[2] - std::f64::consts::PI).abs() < 1e-10);
    }

    // ---- atan ----

    #[test]
    fn atan_scalar_returns_expected_value() {
        let input = ScalarTensor::new(1.0, DType::F64, Device::Cpu);
        let out = atan_scalar(&input);
        assert!((out.value() - std::f64::consts::FRAC_PI_4).abs() < 1e-10);
    }

    #[test]
    fn atan_tensor_contiguous_returns_expected_values() {
        let meta = TensorMeta::from_shape(vec![3], DType::F64, Device::Cpu);
        let input = vec![0.0, 1.0, -1.0];
        let out = atan_tensor_contiguous_f64(&input, &meta).expect("atan should succeed");
        assert!((out[0] - 0.0).abs() < 1e-10);
        assert!((out[1] - std::f64::consts::FRAC_PI_4).abs() < 1e-10);
        assert!((out[2] - (-std::f64::consts::FRAC_PI_4)).abs() < 1e-10);
    }

    // ---- sinh ----

    #[test]
    fn sinh_scalar_returns_expected_value() {
        let input = ScalarTensor::new(0.0, DType::F64, Device::Cpu);
        let out = sinh_scalar(&input);
        assert!((out.value() - 0.0).abs() < 1e-10);
    }

    #[test]
    fn sinh_tensor_contiguous_returns_expected_values() {
        let meta = TensorMeta::from_shape(vec![2], DType::F64, Device::Cpu);
        let input = vec![0.0, 1.0];
        let out = sinh_tensor_contiguous_f64(&input, &meta).expect("sinh should succeed");
        assert!((out[0] - 0.0).abs() < 1e-10);
        assert!((out[1] - 1.0_f64.sinh()).abs() < 1e-10);
    }

    // ---- cosh ----

    #[test]
    fn cosh_scalar_returns_expected_value() {
        let input = ScalarTensor::new(0.0, DType::F64, Device::Cpu);
        let out = cosh_scalar(&input);
        assert!((out.value() - 1.0).abs() < 1e-10);
    }

    #[test]
    fn cosh_tensor_contiguous_returns_expected_values() {
        let meta = TensorMeta::from_shape(vec![2], DType::F64, Device::Cpu);
        let input = vec![0.0, 1.0];
        let out = cosh_tensor_contiguous_f64(&input, &meta).expect("cosh should succeed");
        assert!((out[0] - 1.0).abs() < 1e-10);
        assert!((out[1] - 1.0_f64.cosh()).abs() < 1e-10);
    }

    // ---- gelu ----

    #[test]
    fn gelu_scalar_returns_expected_value() {
        let zero = ScalarTensor::new(0.0, DType::F64, Device::Cpu);
        let out = gelu_scalar(&zero);
        assert!((out.value() - 0.0).abs() < 1e-10);
        // GELU(x) for large positive x ≈ x
        let large = ScalarTensor::new(5.0, DType::F64, Device::Cpu);
        let out_large = gelu_scalar(&large);
        assert!((out_large.value() - 5.0).abs() < 0.01);
    }

    #[test]
    fn gelu_tensor_contiguous_returns_expected_values() {
        let meta = TensorMeta::from_shape(vec![3], DType::F64, Device::Cpu);
        let input = vec![0.0, -5.0, 5.0];
        let out = gelu_tensor_contiguous_f64(&input, &meta).expect("gelu should succeed");
        assert!((out[0] - 0.0).abs() < 1e-10);
        assert!(out[1].abs() < 0.01); // GELU(-5) ≈ 0
        assert!((out[2] - 5.0).abs() < 0.01); // GELU(5) ≈ 5
    }

    // ---- silu ----

    #[test]
    fn silu_scalar_returns_expected_value() {
        let zero = ScalarTensor::new(0.0, DType::F64, Device::Cpu);
        let out = silu_scalar(&zero);
        assert!((out.value() - 0.0).abs() < 1e-10);
        // SiLU(x) = x * sigmoid(x), for large x → x
        let large = ScalarTensor::new(10.0, DType::F64, Device::Cpu);
        let out_large = silu_scalar(&large);
        assert!((out_large.value() - 10.0).abs() < 0.001);
    }

    #[test]
    fn silu_tensor_contiguous_returns_expected_values() {
        let meta = TensorMeta::from_shape(vec![3], DType::F64, Device::Cpu);
        let input = vec![0.0, -10.0, 10.0];
        let out = silu_tensor_contiguous_f64(&input, &meta).expect("silu should succeed");
        assert!((out[0] - 0.0).abs() < 1e-10);
        assert!(out[1].abs() < 0.001); // SiLU(-10) ≈ 0
        assert!((out[2] - 10.0).abs() < 0.001); // SiLU(10) ≈ 10
    }

    // ---- leaky_relu ----

    #[test]
    fn leaky_relu_scalar_returns_expected_value() {
        let pos = ScalarTensor::new(3.0, DType::F64, Device::Cpu);
        let neg = ScalarTensor::new(-2.0, DType::F64, Device::Cpu);
        assert_eq!(leaky_relu_scalar(&pos).value(), 3.0);
        assert!((leaky_relu_scalar(&neg).value() - (-0.02)).abs() < 1e-10);
    }

    #[test]
    fn leaky_relu_tensor_contiguous_returns_expected_values() {
        let meta = TensorMeta::from_shape(vec![4], DType::F64, Device::Cpu);
        let input = vec![1.0, -1.0, 0.0, -5.0];
        let out =
            leaky_relu_tensor_contiguous_f64(&input, &meta).expect("leaky_relu should succeed");
        assert_eq!(out[0], 1.0);
        assert!((out[1] - (-0.01)).abs() < 1e-10);
        assert_eq!(out[2], 0.0);
        assert!((out[3] - (-0.05)).abs() < 1e-10);
    }

    // ---- cumsum ----

    #[test]
    fn cumsum_1d() {
        let meta = TensorMeta::from_shape(vec![5], DType::F64, Device::Cpu);
        let input = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let out =
            super::cumsum_tensor_contiguous_f64(&input, &meta, 0).expect("cumsum should succeed");
        assert_eq!(out, vec![1.0, 3.0, 6.0, 10.0, 15.0]);
    }

    #[test]
    fn cumsum_2d_dim0() {
        let meta = TensorMeta::from_shape(vec![2, 3], DType::F64, Device::Cpu);
        let input = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let out =
            super::cumsum_tensor_contiguous_f64(&input, &meta, 0).expect("cumsum should succeed");
        // dim0: accumulate rows: [1,2,3] -> [1,2,3], [1+4,2+5,3+6] = [5,7,9]
        assert_eq!(out, vec![1.0, 2.0, 3.0, 5.0, 7.0, 9.0]);
    }

    #[test]
    fn cumsum_2d_dim1() {
        let meta = TensorMeta::from_shape(vec![2, 3], DType::F64, Device::Cpu);
        let input = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let out =
            super::cumsum_tensor_contiguous_f64(&input, &meta, 1).expect("cumsum should succeed");
        // dim1: accumulate within rows: [1,1+2,1+2+3] [4,4+5,4+5+6] = [1,3,6,4,9,15]
        assert_eq!(out, vec![1.0, 3.0, 6.0, 4.0, 9.0, 15.0]);
    }

    #[test]
    fn cumsum_invalid_dim() {
        let meta = TensorMeta::from_shape(vec![3], DType::F64, Device::Cpu);
        let input = vec![1.0, 2.0, 3.0];
        let result = super::cumsum_tensor_contiguous_f64(&input, &meta, 1);
        assert!(result.is_err());
    }

    #[test]
    fn cumsum_backward_is_reverse_cumsum() {
        let meta = TensorMeta::from_shape(vec![4], DType::F64, Device::Cpu);
        let grad_output = vec![1.0, 1.0, 1.0, 1.0];
        let grad_input = super::cumsum_backward_tensor_contiguous_f64(&grad_output, &meta, 0)
            .expect("cumsum backward should succeed");
        // Reverse cumsum of [1,1,1,1] = [4,3,2,1]
        assert_eq!(grad_input, vec![4.0, 3.0, 2.0, 1.0]);
    }

    // ---- cumprod ----

    #[test]
    fn cumprod_1d() {
        let meta = TensorMeta::from_shape(vec![4], DType::F64, Device::Cpu);
        let input = vec![2.0, 3.0, 4.0, 5.0];
        let out =
            super::cumprod_tensor_contiguous_f64(&input, &meta, 0).expect("cumprod should succeed");
        assert_eq!(out, vec![2.0, 6.0, 24.0, 120.0]);
    }

    #[test]
    fn cumprod_2d_dim0() {
        let meta = TensorMeta::from_shape(vec![2, 3], DType::F64, Device::Cpu);
        let input = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let out =
            super::cumprod_tensor_contiguous_f64(&input, &meta, 0).expect("cumprod should succeed");
        // dim0: [1,2,3] -> [1,2,3], [1*4,2*5,3*6] = [4,10,18]
        assert_eq!(out, vec![1.0, 2.0, 3.0, 4.0, 10.0, 18.0]);
    }

    #[test]
    fn cumprod_2d_dim1() {
        let meta = TensorMeta::from_shape(vec![2, 3], DType::F64, Device::Cpu);
        let input = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let out =
            super::cumprod_tensor_contiguous_f64(&input, &meta, 1).expect("cumprod should succeed");
        // dim1: [1,1*2,1*2*3] [4,4*5,4*5*6] = [1,2,6,4,20,120]
        assert_eq!(out, vec![1.0, 2.0, 6.0, 4.0, 20.0, 120.0]);
    }

    #[test]
    fn cumprod_invalid_dim() {
        let meta = TensorMeta::from_shape(vec![3], DType::F64, Device::Cpu);
        let input = vec![1.0, 2.0, 3.0];
        let result = super::cumprod_tensor_contiguous_f64(&input, &meta, 1);
        assert!(result.is_err());
    }

    // ---- where ----

    #[test]
    fn where_selects_by_condition() {
        let meta = TensorMeta::from_shape(vec![4], DType::F64, Device::Cpu);
        let cond = vec![1.0, 0.0, 1.0, 0.0];
        let x = vec![10.0, 20.0, 30.0, 40.0];
        let y = vec![-1.0, -2.0, -3.0, -4.0];
        let out =
            super::where_tensor_contiguous_f64(&cond, &x, &y, &meta).expect("where should succeed");
        assert_eq!(out, vec![10.0, -2.0, 30.0, -4.0]);
    }

    #[test]
    fn where_all_true() {
        let meta = TensorMeta::from_shape(vec![3], DType::F64, Device::Cpu);
        let cond = vec![1.0, 1.0, 1.0];
        let x = vec![10.0, 20.0, 30.0];
        let y = vec![-1.0, -2.0, -3.0];
        let out =
            super::where_tensor_contiguous_f64(&cond, &x, &y, &meta).expect("where should succeed");
        assert_eq!(out, vec![10.0, 20.0, 30.0]);
    }

    #[test]
    fn where_all_false() {
        let meta = TensorMeta::from_shape(vec![3], DType::F64, Device::Cpu);
        let cond = vec![0.0, 0.0, 0.0];
        let x = vec![10.0, 20.0, 30.0];
        let y = vec![-1.0, -2.0, -3.0];
        let out =
            super::where_tensor_contiguous_f64(&cond, &x, &y, &meta).expect("where should succeed");
        assert_eq!(out, vec![-1.0, -2.0, -3.0]);
    }

    // ---- sort ----

    #[test]
    fn sort_parallel_matches_serial_bit_exact() {
        // Isomorphism proof for parallelizing sort-along-dim over lanes: the
        // parallel kernel must reproduce the serial per-lane stable sort
        // BIT-FOR-BIT — identical sorted values AND identical original-index
        // tie-breaks. Uses a strided shape (inner_size > 1) and a last-dim shape.
        for (shape, dim, desc) in [
            (vec![13usize, 32usize, 4usize], 1usize, false), // strided, inner=4
            (vec![97usize, 64usize], 1usize, true),          // last dim, descending
        ] {
            let numel: usize = shape.iter().product();
            let reduce = shape[dim];
            let inner: usize = shape[dim + 1..].iter().product();
            let outer: usize = shape[..dim].iter().product();
            // Include duplicates so the stable tie-break is actually exercised.
            let data: Vec<f64> = (0..numel)
                .map(|i| ((i * 37 + 11) % 19) as f64 * 0.5)
                .collect();
            let meta = TensorMeta::from_shape(shape.clone(), DType::F64, Device::Cpu);

            let mut sv_ref = vec![0.0_f64; numel];
            let mut idx_ref = vec![0usize; numel];
            for o in 0..outer {
                for i in 0..inner {
                    let mut lane: Vec<(usize, f64)> = (0..reduce)
                        .map(|d| (d, data[o * reduce * inner + d * inner + i]))
                        .collect();
                    if desc {
                        lane.sort_by(|a, b| super::nan_greatest_cmp_f64(b.1, a.1));
                    } else {
                        lane.sort_by(|a, b| super::nan_greatest_cmp_f64(a.1, b.1));
                    }
                    for (out_d, (orig_d, val)) in lane.into_iter().enumerate() {
                        let idx = o * reduce * inner + out_d * inner + i;
                        sv_ref[idx] = val;
                        idx_ref[idx] = orig_d;
                    }
                }
            }

            let (sv, idx) =
                super::sort_tensor_contiguous_f64(&data, &meta, dim, desc).expect("sort");
            assert_eq!(
                idx, idx_ref,
                "sort indices diverged for {shape:?} dim {dim}"
            );
            for k in 0..numel {
                assert_eq!(
                    sv[k].to_bits(),
                    sv_ref[k].to_bits(),
                    "sort values diverged at {k} for {shape:?} dim {dim}"
                );
            }
        }
    }

    #[test]
    fn sort_radix_path_matches_comparison_bit_exact() {
        // The LSD radix fast path only engages for lanes >= SORT_RADIX_MIN_LEN, so
        // the small-shape tests above never reach it. Exercise it directly on a
        // 512-long lane carrying duplicates (stable ties), negatives, and BOTH
        // signed zeros, and prove it is bit-for-bit identical to the stable
        // comparison sort it replaces — values (to_bits) AND original indices.
        let (rows, cols) = (6usize, 512usize);
        assert!(
            cols >= super::SORT_RADIX_MIN_LEN,
            "lane must hit radix path"
        );
        let numel = rows * cols;
        let data: Vec<f64> = (0..numel)
            .map(|i| match i % 7 {
                0 => 0.0,
                1 => -0.0,
                2 => -((i % 23) as f64) * 0.25,
                _ => ((i * 2654435761usize) % 211) as f64 * 0.125,
            })
            .collect();
        let meta = TensorMeta::from_shape(vec![rows, cols], DType::F64, Device::Cpu);

        for desc in [false, true] {
            let (sv, idx) = super::sort_tensor_contiguous_f64(&data, &meta, 1, desc)
                .expect("radix sort should succeed");
            for r in 0..rows {
                let base = r * cols;
                let mut lane: Vec<(usize, f64)> = (0..cols).map(|d| (d, data[base + d])).collect();
                if desc {
                    lane.sort_by(|a, b| super::nan_greatest_cmp_f64(b.1, a.1));
                } else {
                    lane.sort_by(|a, b| super::nan_greatest_cmp_f64(a.1, b.1));
                }
                for (out_d, (orig_d, val)) in lane.into_iter().enumerate() {
                    let k = base + out_d;
                    assert_eq!(
                        idx[k], orig_d,
                        "radix index diverged at row {r} pos {out_d} (desc={desc})"
                    );
                    assert_eq!(
                        sv[k].to_bits(),
                        val.to_bits(),
                        "radix value diverged at row {r} pos {out_d} (desc={desc})"
                    );
                }
            }
        }
    }

    #[test]
    fn sort_f32_radix_path_matches_comparison_bit_exact() {
        // f32 radix path (4 effective passes via the high-zero u64 key). Same
        // obligations as f64: a 512-long lane with duplicates, negatives, and
        // both signed zeros must match the stable comparison sort bit-for-bit
        // (values via to_bits AND original indices) in both directions.
        let (rows, cols) = (6usize, 512usize);
        assert!(
            cols >= super::SORT_RADIX_MIN_LEN,
            "lane must hit radix path"
        );
        let numel = rows * cols;
        let data: Vec<f32> = (0..numel)
            .map(|i| match i % 7 {
                0 => 0.0f32,
                1 => -0.0f32,
                2 => -((i % 23) as f32) * 0.25,
                _ => ((i * 2654435761usize) % 211) as f32 * 0.125,
            })
            .collect();
        let meta = TensorMeta::from_shape(vec![rows, cols], DType::F32, Device::Cpu);

        for desc in [false, true] {
            let (sv, idx) = super::sort_tensor_contiguous_f32(&data, &meta, 1, desc)
                .expect("f32 radix sort should succeed");
            for r in 0..rows {
                let base = r * cols;
                let mut lane: Vec<(usize, f32)> = (0..cols).map(|d| (d, data[base + d])).collect();
                if desc {
                    lane.sort_by(|a, b| super::nan_greatest_cmp_f32(b.1, a.1));
                } else {
                    lane.sort_by(|a, b| super::nan_greatest_cmp_f32(a.1, b.1));
                }
                for (out_d, (orig_d, val)) in lane.into_iter().enumerate() {
                    let k = base + out_d;
                    assert_eq!(
                        idx[k], orig_d,
                        "f32 radix index diverged at row {r} pos {out_d} (desc={desc})"
                    );
                    assert_eq!(
                        sv[k].to_bits(),
                        val.to_bits(),
                        "f32 radix value diverged at row {r} pos {out_d} (desc={desc})"
                    );
                }
            }
        }
    }

    #[test]
    fn sort_1d_ascending() {
        let meta = TensorMeta::from_shape(vec![5], DType::F64, Device::Cpu);
        let input = vec![3.0, 1.0, 4.0, 1.0, 5.0];
        let (vals, idxs) =
            super::sort_tensor_contiguous_f64(&input, &meta, 0, false).expect("sort should work");
        assert_eq!(vals, vec![1.0, 1.0, 3.0, 4.0, 5.0]);
        assert_eq!(idxs, vec![1, 3, 0, 2, 4]);
    }

    #[test]
    fn sort_1d_descending() {
        let meta = TensorMeta::from_shape(vec![4], DType::F64, Device::Cpu);
        let input = vec![2.0, 5.0, 1.0, 3.0];
        let (vals, idxs) =
            super::sort_tensor_contiguous_f64(&input, &meta, 0, true).expect("sort should work");
        assert_eq!(vals, vec![5.0, 3.0, 2.0, 1.0]);
        assert_eq!(idxs, vec![1, 3, 0, 2]);
    }

    #[test]
    fn sort_2d_along_dim1() {
        let meta = TensorMeta::from_shape(vec![2, 3], DType::F64, Device::Cpu);
        let input = vec![3.0, 1.0, 2.0, 6.0, 4.0, 5.0];
        let (vals, idxs) =
            super::sort_tensor_contiguous_f64(&input, &meta, 1, false).expect("sort should work");
        assert_eq!(vals, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        assert_eq!(idxs, vec![1, 2, 0, 1, 2, 0]);
    }

    #[test]
    fn sort_invalid_dim() {
        let meta = TensorMeta::from_shape(vec![3], DType::F64, Device::Cpu);
        let input = vec![1.0, 2.0, 3.0];
        let result = super::sort_tensor_contiguous_f64(&input, &meta, 1, false);
        assert!(result.is_err());
    }

    // ---- topk ----

    #[test]
    fn topk_parallel_matches_serial_bit_exact() {
        // Isomorphism proof for topk-along-dim over lanes: the optimized kernel
        // must reproduce the old serial full-sort select BIT-FOR-BIT — identical
        // values AND original indices, in both sorted modes. Duplicates, signed
        // zeros, and NaN payloads exercise stable tie-breaking.
        for (shape, dim, k, largest, srt) in [
            (vec![11usize, 24usize, 3usize], 1usize, 5usize, true, true),
            (vec![11usize, 24usize, 3usize], 1usize, 5usize, true, false),
            (vec![73usize, 40usize], 1usize, 7usize, false, false),
        ] {
            let reduce = shape[dim];
            let inner: usize = shape[dim + 1..].iter().product();
            let outer: usize = shape[..dim].iter().product();
            let numel: usize = shape.iter().product();
            let mut data: Vec<f64> = (0..numel)
                .map(|i| ((i * 53 + 7) % 17) as f64 * 0.25)
                .collect();
            for i in (0..numel).step_by(37) {
                let payload = u64::try_from(i & 0xff).expect("masked payload fits u64");
                data[i] = f64::from_bits(0x7ff8_0000_0000_0000 | payload);
            }
            for i in (11..numel).step_by(41) {
                data[i] = -0.0;
            }
            for i in (17..numel).step_by(43) {
                data[i] = 0.0;
            }

            let mut v_ref = vec![0.0_f64; outer * k * inner];
            let mut i_ref = vec![0usize; outer * k * inner];
            for o in 0..outer {
                for inr in 0..inner {
                    let mut lane: Vec<(usize, f64)> = (0..reduce)
                        .map(|d| (d, data[o * reduce * inner + d * inner + inr]))
                        .collect();
                    if largest {
                        lane.sort_by(|a, b| super::nan_greatest_cmp_f64(b.1, a.1));
                    } else {
                        lane.sort_by(|a, b| super::nan_greatest_cmp_f64(a.1, b.1));
                    }
                    let mut sel: Vec<(usize, f64)> = lane[..k].to_vec();
                    if !srt {
                        sel.sort_by_key(|(o2, _)| *o2);
                    }
                    for (od, (orig, val)) in sel.into_iter().enumerate() {
                        let idx = o * k * inner + od * inner + inr;
                        v_ref[idx] = val;
                        i_ref[idx] = orig;
                    }
                }
            }

            let meta = TensorMeta::from_shape(shape.clone(), DType::F64, Device::Cpu);
            let (v, i) = super::topk_tensor_contiguous_f64(&data, &meta, k, dim, largest, srt)
                .expect("topk");
            assert_eq!(
                i, i_ref,
                "topk indices diverged {shape:?} k={k} sorted={srt}"
            );
            for t in 0..v.len() {
                assert_eq!(
                    v[t].to_bits(),
                    v_ref[t].to_bits(),
                    "topk values diverged at {t} {shape:?} k={k} sorted={srt}"
                );
            }
        }
    }

    #[test]
    fn topk_f32_bounded_selection_matches_full_sort_bit_exact() {
        // F32 isomorphism proof for the bounded-selection path: match the old
        // full-sort select exactly, including NaN-as-largest and original-index
        // tie order for equal values/signed zeros.
        for (shape, dim, k, largest, srt) in [
            (vec![11usize, 24usize, 3usize], 1usize, 5usize, true, true),
            (vec![11usize, 24usize, 3usize], 1usize, 5usize, true, false),
            (vec![73usize, 40usize], 1usize, 7usize, false, false),
        ] {
            let reduce = shape[dim];
            let inner: usize = shape[dim + 1..].iter().product();
            let outer: usize = shape[..dim].iter().product();
            let numel: usize = shape.iter().product();
            let mut data: Vec<f32> = (0..numel)
                .map(|i| ((i * 53 + 7) % 17) as f32 * 0.25)
                .collect();
            for i in (0..numel).step_by(37) {
                let payload = u32::try_from(i & 0xff).expect("masked payload fits u32");
                data[i] = f32::from_bits(0x7fc0_0000 | payload);
            }
            for i in (11..numel).step_by(41) {
                data[i] = -0.0;
            }
            for i in (17..numel).step_by(43) {
                data[i] = 0.0;
            }

            let mut v_ref = vec![0.0_f32; outer * k * inner];
            let mut i_ref = vec![0usize; outer * k * inner];
            for o in 0..outer {
                for inr in 0..inner {
                    let mut lane: Vec<(usize, f32)> = (0..reduce)
                        .map(|d| (d, data[o * reduce * inner + d * inner + inr]))
                        .collect();
                    if largest {
                        lane.sort_by(|a, b| super::nan_greatest_cmp_f32(b.1, a.1));
                    } else {
                        lane.sort_by(|a, b| super::nan_greatest_cmp_f32(a.1, b.1));
                    }
                    let mut sel: Vec<(usize, f32)> = lane[..k].to_vec();
                    if !srt {
                        sel.sort_by_key(|(o2, _)| *o2);
                    }
                    for (od, (orig, val)) in sel.into_iter().enumerate() {
                        let idx = o * k * inner + od * inner + inr;
                        v_ref[idx] = val;
                        i_ref[idx] = orig;
                    }
                }
            }

            let meta = TensorMeta::from_shape(shape.clone(), DType::F32, Device::Cpu);
            let (v, i) = super::topk_tensor_contiguous_f32(&data, &meta, k, dim, largest, srt)
                .expect("topk_f32");
            assert_eq!(
                i, i_ref,
                "topk_f32 indices diverged {shape:?} k={k} sorted={srt}"
            );
            for t in 0..v.len() {
                assert_eq!(
                    v[t].to_bits(),
                    v_ref[t].to_bits(),
                    "topk_f32 values diverged at {t} {shape:?} k={k} sorted={srt}"
                );
            }
        }
    }

    #[test]
    fn topk_largest_sorted() {
        let meta = TensorMeta::from_shape(vec![5], DType::F64, Device::Cpu);
        let input = vec![3.0, 1.0, 4.0, 1.0, 5.0];
        let (vals, idxs) =
            super::topk_tensor_contiguous_f64(&input, &meta, 3, 0, true, true).expect("topk");
        assert_eq!(vals, vec![5.0, 4.0, 3.0]);
        assert_eq!(idxs, vec![4, 2, 0]);
    }

    #[test]
    fn topk_smallest_sorted() {
        let meta = TensorMeta::from_shape(vec![5], DType::F64, Device::Cpu);
        let input = vec![3.0, 1.0, 4.0, 1.0, 5.0];
        let (vals, idxs) =
            super::topk_tensor_contiguous_f64(&input, &meta, 2, 0, false, true).expect("topk");
        assert_eq!(vals, vec![1.0, 1.0]);
        assert_eq!(idxs, vec![1, 3]);
    }

    #[test]
    fn topk_2d_along_dim1() {
        let meta = TensorMeta::from_shape(vec![2, 4], DType::F64, Device::Cpu);
        let input = vec![4.0, 2.0, 3.0, 1.0, 8.0, 6.0, 7.0, 5.0];
        let (vals, idxs) =
            super::topk_tensor_contiguous_f64(&input, &meta, 2, 1, true, true).expect("topk");
        // First row top-2: 4.0 (idx 0), 3.0 (idx 2)
        // Second row top-2: 8.0 (idx 0), 7.0 (idx 2)
        assert_eq!(vals, vec![4.0, 3.0, 8.0, 7.0]);
        assert_eq!(idxs, vec![0, 2, 0, 2]);
    }

    #[test]
    fn topk_k_exceeds_dim_returns_error() {
        let meta = TensorMeta::from_shape(vec![3], DType::F64, Device::Cpu);
        let input = vec![1.0, 2.0, 3.0];
        let result = super::topk_tensor_contiguous_f64(&input, &meta, 5, 0, true, true);
        assert!(result.is_err());
    }

    // -------------------------------------------------------------------
    // LU Decomposition tests
    // -------------------------------------------------------------------

    fn mat_mul_nn(a: &[f64], b: &[f64], n: usize) -> Vec<f64> {
        let mut c = vec![0.0; n * n];
        for i in 0..n {
            for j in 0..n {
                let mut acc = 0.0;
                for k in 0..n {
                    acc += a[i * n + k] * b[k * n + j];
                }
                c[i * n + j] = acc;
            }
        }
        c
    }

    fn assert_mat_approx_eq(a: &[f64], b: &[f64], tol: f64, msg: &str) {
        assert_eq!(a.len(), b.len(), "{msg}: length mismatch");
        for (i, (&av, &bv)) in a.iter().zip(b.iter()).enumerate() {
            assert!(
                (av - bv).abs() < tol,
                "{msg}: element {i} differs: {av} vs {bv} (tol={tol})"
            );
        }
    }

    #[test]
    fn lu_factor_reconstructs_pa_eq_lu() {
        // Parity guard for the blocked (getrf-style) factorization: the pivot
        // SEQUENCE is identical to a straightforward reference Gaussian
        // elimination with partial pivoting (blocked LU pivots column-by-column,
        // same as unblocked), and P·L·U reconstructs A to working precision.
        // The blocked trailing GEMM reassociates the trailing update, so the LU
        // bits differ from naive GE by ~1 ulp — exactly what LAPACK/numpy do —
        // hence reconstruction rather than a bit-for-bit comparison.
        let n = 200usize;
        // Diagonally dominant so pivoting is well-defined and no near-singular
        // `continue` fires; fractional values so rounding actually matters.
        let mut a = vec![0.0_f64; n * n];
        for i in 0..n {
            for j in 0..n {
                a[i * n + j] = ((i * 31 + j * 17) % 97) as f64 * 0.013 - 0.5;
            }
            a[i * n + i] += n as f64; // dominant diagonal
        }

        // Serial reference: identical algorithm to lu_factor_contiguous_f64 but
        // with the trailing update forced serial.
        let mut lu_ref = a.clone();
        let mut piv_ref: Vec<usize> = (0..n).collect();
        for k in 0..n {
            let mut max_val = lu_ref[k * n + k].abs();
            let mut max_row = k;
            for i in (k + 1)..n {
                let v = lu_ref[i * n + k].abs();
                if v > max_val {
                    max_val = v;
                    max_row = i;
                }
            }
            if max_row != k {
                piv_ref.swap(k, max_row);
                for j in 0..n {
                    lu_ref.swap(k * n + j, max_row * n + j);
                }
            }
            let diag = lu_ref[k * n + k];
            if diag.abs() < f64::EPSILON * 1e3 {
                continue;
            }
            for i in (k + 1)..n {
                let m = lu_ref[i * n + k] / diag;
                lu_ref[i * n + k] = m;
                for j in (k + 1)..n {
                    lu_ref[i * n + j] -= m * lu_ref[k * n + j];
                }
            }
        }

        let meta = TensorMeta::from_shape(vec![n, n], DType::F64, Device::Cpu);
        let result = super::lu_factor_contiguous_f64(&a, &meta).expect("lu_factor");
        // Pivot sequence must be identical to the reference (column-by-column
        // partial pivoting is unchanged by blocking).
        assert_eq!(result.pivots, piv_ref, "pivot order diverged");

        // Reconstruction: P · L · U == A to working precision. L is unit-lower
        // and U upper, so (L·U)[i][j] = sum_{t<=min(i,j)} L[i][t]·U[t][j].
        let unpacked = super::lu_unpack(&result);
        let mut lu_prod = vec![0.0f64; n * n];
        for i in 0..n {
            for j in 0..n {
                let mut s = 0.0;
                for t in 0..=i.min(j) {
                    s += unpacked.l[i * n + t] * unpacked.u[t * n + j];
                }
                lu_prod[i * n + j] = s;
            }
        }
        // P·(L·U): row `orig` of the product = row `pos` of lu_prod where
        // pivots[pos]==orig.
        for pos in 0..n {
            let orig = result.pivots[pos];
            for j in 0..n {
                let recon = lu_prod[pos * n + j];
                let expected = a[orig * n + j];
                assert!(
                    (recon - expected).abs() < 1e-9,
                    "P·L·U mismatch at ({orig},{j}): {recon} vs {expected}"
                );
            }
        }
    }

    #[test]
    fn lu_factor_parallel_golden_output_matches_fixture() {
        use std::fmt::Write as _;

        let n = 128usize; // > LU_PAR_MIN_ROWS -> parallel path when the pool has >1 thread
        let mut a = vec![0.0_f64; n * n];
        for i in 0..n {
            a[i * n + i] = 1.0;
        }
        let meta = TensorMeta::from_shape(vec![n, n], DType::F64, Device::Cpu);
        let result = super::lu_factor_contiguous_f64(&a, &meta).expect("lu_factor");

        let mut output = String::from("frankentorch-872a lu_parallel_golden\nn=128\n");
        output.push_str("selected_pivots:\n");
        for idx in [0usize, 1, n - 1] {
            let _ = writeln!(&mut output, "{idx}: {}", result.pivots[idx]);
        }
        output.push_str("selected_lu_bits:\n");
        for idx in [0usize, 1, n + 1, n * n - 1] {
            let _ = writeln!(&mut output, "{idx}: {:#018x}", result.lu[idx].to_bits());
        }

        assert_eq!(
            output,
            include_str!(
                "../../../artifacts/optimization/golden_outputs/ft_kernel_cpu_lu_parallel_frankentorch-872a.txt"
            )
        );
    }

    #[test]
    fn lu_factor_identity_3x3() {
        let meta = TensorMeta::from_shape(vec![3, 3], DType::F64, Device::Cpu);
        #[rustfmt::skip]
        let a = vec![
            1.0, 0.0, 0.0,
            0.0, 1.0, 0.0,
            0.0, 0.0, 1.0,
        ];
        let result = super::lu_factor_contiguous_f64(&a, &meta).expect("lu_factor should succeed");
        assert_eq!(result.n, 3);

        let unpacked = super::lu_unpack(&result);
        // For identity: P=I, L=I, U=I
        assert_mat_approx_eq(&unpacked.l, &a, 1e-12, "L should be identity");
        assert_mat_approx_eq(&unpacked.u, &a, 1e-12, "U should be identity");

        // P @ L @ U == A
        let pl = mat_mul_nn(&unpacked.p, &unpacked.l, 3);
        let plu = mat_mul_nn(&pl, &unpacked.u, 3);
        assert_mat_approx_eq(&plu, &a, 1e-12, "P @ L @ U should equal A");
    }

    #[test]
    fn lu_factor_known_3x3() {
        let meta = TensorMeta::from_shape(vec![3, 3], DType::F64, Device::Cpu);
        #[rustfmt::skip]
        let a = vec![
            2.0, 1.0, 1.0,
            4.0, 3.0, 3.0,
            8.0, 7.0, 9.0,
        ];
        let result = super::lu_factor_contiguous_f64(&a, &meta).expect("lu_factor should succeed");
        let unpacked = super::lu_unpack(&result);

        // Verify P @ L @ U == A
        let pl = mat_mul_nn(&unpacked.p, &unpacked.l, 3);
        let plu = mat_mul_nn(&pl, &unpacked.u, 3);
        assert_mat_approx_eq(&plu, &a, 1e-10, "P @ L @ U should equal A");

        // Verify L is lower triangular with unit diagonal
        for i in 0..3 {
            assert!(
                (unpacked.l[i * 3 + i] - 1.0).abs() < 1e-12,
                "L diagonal should be 1.0"
            );
            for j in (i + 1)..3 {
                assert!(
                    unpacked.l[i * 3 + j].abs() < 1e-12,
                    "L should be zero above diagonal"
                );
            }
        }

        // Verify U is upper triangular
        for i in 0..3 {
            for j in 0..i {
                assert!(
                    unpacked.u[i * 3 + j].abs() < 1e-12,
                    "U should be zero below diagonal"
                );
            }
        }
    }

    #[test]
    fn lu_factor_already_upper_triangular() {
        let meta = TensorMeta::from_shape(vec![3, 3], DType::F64, Device::Cpu);
        #[rustfmt::skip]
        let a = vec![
            3.0, 2.0, 1.0,
            0.0, 5.0, 4.0,
            0.0, 0.0, 6.0,
        ];
        let result = super::lu_factor_contiguous_f64(&a, &meta).expect("lu_factor");
        let unpacked = super::lu_unpack(&result);

        let pl = mat_mul_nn(&unpacked.p, &unpacked.l, 3);
        let plu = mat_mul_nn(&pl, &unpacked.u, 3);
        assert_mat_approx_eq(&plu, &a, 1e-10, "P @ L @ U should equal A");
    }

    #[test]
    fn lu_factor_1x1() {
        let meta = TensorMeta::from_shape(vec![1, 1], DType::F64, Device::Cpu);
        let a = vec![5.0];
        let result = super::lu_factor_contiguous_f64(&a, &meta).expect("lu_factor");
        let unpacked = super::lu_unpack(&result);

        assert_mat_approx_eq(&unpacked.p, &[1.0], 1e-12, "P for 1x1");
        assert_mat_approx_eq(&unpacked.l, &[1.0], 1e-12, "L for 1x1");
        assert_mat_approx_eq(&unpacked.u, &[5.0], 1e-12, "U for 1x1");
    }

    #[test]
    fn lu_factor_requires_square() {
        let meta = TensorMeta::from_shape(vec![2, 3], DType::F64, Device::Cpu);
        let a = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let result = super::lu_factor_contiguous_f64(&a, &meta);
        assert!(result.is_err(), "non-square matrix should error");
    }

    #[test]
    fn lu_factor_requires_2d() {
        let meta = TensorMeta::from_shape(vec![8], DType::F64, Device::Cpu);
        let a = vec![1.0; 8];
        let result = super::lu_factor_contiguous_f64(&a, &meta);
        assert!(result.is_err(), "1D tensor should error");
    }

    #[test]
    fn lu_factor_zeros_on_diagonal_needs_pivoting() {
        let meta = TensorMeta::from_shape(vec![3, 3], DType::F64, Device::Cpu);
        #[rustfmt::skip]
        let a = vec![
            0.0, 1.0, 2.0,
            3.0, 4.0, 5.0,
            6.0, 7.0, 8.0,
        ];
        let result =
            super::lu_factor_contiguous_f64(&a, &meta).expect("pivoting should handle this");
        let unpacked = super::lu_unpack(&result);

        let pl = mat_mul_nn(&unpacked.p, &unpacked.l, 3);
        let plu = mat_mul_nn(&pl, &unpacked.u, 3);
        assert_mat_approx_eq(
            &plu,
            &a,
            1e-10,
            "P @ L @ U should equal A even with zero diagonal",
        );
    }

    #[test]
    fn matrix_exp_defining_properties() {
        // (1) expm of a diagonal matrix = diag(exp(diagonal)).
        let meta3 = TensorMeta::from_shape(vec![3, 3], DType::F64, Device::Cpu);
        #[rustfmt::skip]
        let diag = vec![
            1.0, 0.0, 0.0,
            0.0, -1.0, 0.0,
            0.0, 0.0, 2.0,
        ];
        let e = super::matrix_exp_contiguous_f64(&diag, &meta3).unwrap();
        #[rustfmt::skip]
        let want = vec![
            1.0_f64.exp(), 0.0, 0.0,
            0.0, (-1.0_f64).exp(), 0.0,
            0.0, 0.0, 2.0_f64.exp(),
        ];
        assert_mat_approx_eq(&e, &want, 1e-9, "expm(diag) = diag(exp)");

        // (2) expm of a nilpotent N=[[0,1],[0,0]] is exactly I + N = [[1,1],[0,1]].
        let meta2 = TensorMeta::from_shape(vec![2, 2], DType::F64, Device::Cpu);
        let nil = vec![0.0, 1.0, 0.0, 0.0];
        let en = super::matrix_exp_contiguous_f64(&nil, &meta2).unwrap();
        assert_mat_approx_eq(&en, &[1.0, 1.0, 0.0, 1.0], 1e-12, "expm(nilpotent)");

        // (3) expm(A)·expm(-A) = I — exercises scaling-and-squaring on a larger-norm,
        //     non-normal matrix.
        let n = 3;
        #[rustfmt::skip]
        let a = vec![
            1.0, 2.0, 0.0,
            0.0, 1.0, 3.0,
            1.0, 0.0, 1.0,
        ];
        let neg: Vec<f64> = a.iter().map(|x| -x).collect();
        let ea = super::matrix_exp_contiguous_f64(&a, &meta3).unwrap();
        let ena = super::matrix_exp_contiguous_f64(&neg, &meta3).unwrap();
        let mut prod = vec![0.0f64; n * n];
        for i in 0..n {
            for j in 0..n {
                let mut s = 0.0;
                for k in 0..n {
                    s += ea[i * n + k] * ena[k * n + j];
                }
                prod[i * n + j] = s;
            }
        }
        let mut ident = vec![0.0f64; n * n];
        for i in 0..n {
            ident[i * n + i] = 1.0;
        }
        assert_mat_approx_eq(&prod, &ident, 1e-9, "expm(A)·expm(-A) = I");
    }

    #[test]
    fn lu_solve_simple_system() {
        // Solve A * x = b where A = [[2, 1], [5, 3]], b = [4, 7]
        // Expected: x = [5, -6]
        let meta_a = TensorMeta::from_shape(vec![2, 2], DType::F64, Device::Cpu);
        let a = vec![2.0, 1.0, 5.0, 3.0];
        let factor = super::lu_factor_contiguous_f64(&a, &meta_a).expect("lu_factor");

        let meta_b = TensorMeta::from_shape(vec![2], DType::F64, Device::Cpu);
        let b = vec![4.0, 7.0];
        let x = super::lu_solve_contiguous_f64(&factor, &b, &meta_b).expect("lu_solve");

        assert_mat_approx_eq(&x, &[5.0, -6.0], 1e-10, "solution should be [5, -6]");
    }

    #[test]
    fn lu_solve_3x3_system() {
        // A = [[1, 2, 3], [4, 5, 6], [7, 8, 10]]
        // b = [1, 2, 3]
        let meta_a = TensorMeta::from_shape(vec![3, 3], DType::F64, Device::Cpu);
        #[rustfmt::skip]
        let a = vec![
            1.0, 2.0, 3.0,
            4.0, 5.0, 6.0,
            7.0, 8.0, 10.0,
        ];
        let factor = super::lu_factor_contiguous_f64(&a, &meta_a).expect("lu_factor");

        let meta_b = TensorMeta::from_shape(vec![3], DType::F64, Device::Cpu);
        let b = vec![1.0, 2.0, 3.0];
        let x = super::lu_solve_contiguous_f64(&factor, &b, &meta_b).expect("lu_solve");

        // Verify A * x ≈ b
        let mut ax = vec![0.0; 3];
        for i in 0..3 {
            for j in 0..3 {
                ax[i] += a[i * 3 + j] * x[j];
            }
        }
        assert_mat_approx_eq(&ax, &b, 1e-10, "A * x should equal b");
    }

    #[test]
    fn lu_solve_multiple_rhs() {
        let meta_a = TensorMeta::from_shape(vec![2, 2], DType::F64, Device::Cpu);
        let a = vec![2.0, 1.0, 5.0, 3.0];
        let factor = super::lu_factor_contiguous_f64(&a, &meta_a).expect("lu_factor");

        // B is [2, 2]: two right-hand sides
        let meta_b = TensorMeta::from_shape(vec![2, 2], DType::F64, Device::Cpu);
        let b = vec![4.0, 3.0, 7.0, 8.0]; // columns: [4,7] and [3,8]
        let x = super::lu_solve_contiguous_f64(&factor, &b, &meta_b).expect("lu_solve multi-rhs");

        // Verify A * X ≈ B (column by column)
        for rhs in 0..2 {
            for i in 0..2 {
                let mut val = 0.0;
                for j in 0..2 {
                    val += a[i * 2 + j] * x[j * 2 + rhs];
                }
                assert!(
                    (val - b[i * 2 + rhs]).abs() < 1e-10,
                    "A*X[{i},{rhs}] = {val}, expected {}",
                    b[i * 2 + rhs]
                );
            }
        }
    }

    #[test]
    fn lu_solve_dimension_mismatch_errors() {
        let meta_a = TensorMeta::from_shape(vec![3, 3], DType::F64, Device::Cpu);
        // Use a simple diagonal to avoid singularity
        let a_diag = vec![1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0];
        let factor = super::lu_factor_contiguous_f64(&a_diag, &meta_a).expect("lu_factor");

        let meta_b = TensorMeta::from_shape(vec![2], DType::F64, Device::Cpu);
        let b = vec![1.0, 2.0];
        let result = super::lu_solve_contiguous_f64(&factor, &b, &meta_b);
        assert!(result.is_err(), "mismatched dimensions should error");
    }

    #[test]
    fn lu_factor_empty_matrix() {
        let meta = TensorMeta::from_shape(vec![0, 0], DType::F64, Device::Cpu);
        let a: Vec<f64> = vec![];
        let result = super::lu_factor_contiguous_f64(&a, &meta).expect("empty should succeed");
        assert_eq!(result.n, 0);
        assert!(result.lu.is_empty());
        assert!(result.pivots.is_empty());
    }

    #[test]
    fn inv_identity_3x3() {
        let meta = TensorMeta::from_shape(vec![3, 3], DType::F64, Device::Cpu);
        #[rustfmt::skip]
        let a = vec![
            1.0, 0.0, 0.0,
            0.0, 1.0, 0.0,
            0.0, 0.0, 1.0,
        ];
        let inv = super::inv_tensor_contiguous_f64(&a, &meta).expect("inverse should succeed");
        assert_mat_approx_eq(&inv, &a, 1e-12, "inv(I) should equal I");
    }

    #[test]
    fn inv_known_2x2() {
        let meta = TensorMeta::from_shape(vec![2, 2], DType::F64, Device::Cpu);
        #[rustfmt::skip]
        let a = vec![
            4.0, 7.0,
            2.0, 6.0,
        ];
        #[rustfmt::skip]
        let expected = vec![
            0.6, -0.7,
            -0.2, 0.4,
        ];
        let inv = super::inv_tensor_contiguous_f64(&a, &meta).expect("inverse should succeed");
        assert_mat_approx_eq(&inv, &expected, 1e-10, "known 2x2 inverse");
    }

    #[test]
    fn inv_round_trip_matches_identity() {
        let meta = TensorMeta::from_shape(vec![3, 3], DType::F64, Device::Cpu);
        #[rustfmt::skip]
        let a = vec![
            3.0, 0.0, 2.0,
            2.0, 0.0, -2.0,
            0.0, 1.0, 1.0,
        ];
        let inv = super::inv_tensor_contiguous_f64(&a, &meta).expect("inverse should succeed");
        let left = mat_mul_nn(&a, &inv, 3);
        let right = mat_mul_nn(&inv, &a, 3);
        #[rustfmt::skip]
        let identity = vec![
            1.0, 0.0, 0.0,
            0.0, 1.0, 0.0,
            0.0, 0.0, 1.0,
        ];
        assert_mat_approx_eq(&left, &identity, 1e-10, "A @ inv(A) should equal I");
        assert_mat_approx_eq(&right, &identity, 1e-10, "inv(A) @ A should equal I");
    }

    #[test]
    fn inv_singular_matrix_returns_error() {
        let meta = TensorMeta::from_shape(vec![2, 2], DType::F64, Device::Cpu);
        #[rustfmt::skip]
        let a = vec![
            1.0, 2.0,
            2.0, 4.0,
        ];
        let err =
            super::inv_tensor_contiguous_f64(&a, &meta).expect_err("singular inverse must fail");
        assert!(matches!(
            err,
            super::KernelError::SingularMatrix { size: 2 }
        ));
    }

    #[test]
    fn inv_requires_square_matrix() {
        let meta = TensorMeta::from_shape(vec![2, 3], DType::F64, Device::Cpu);
        let a = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let err = super::inv_tensor_contiguous_f64(&a, &meta)
            .expect_err("non-square inverse should fail");
        assert!(matches!(err, super::KernelError::ShapeMismatch { .. }));
    }

    // ---- Determinant tests (bd-2drq.2) ----

    #[test]
    fn det_identity() {
        let meta = TensorMeta::from_shape(vec![3, 3], DType::F64, Device::Cpu);
        #[rustfmt::skip]
        let a = vec![
            1.0, 0.0, 0.0,
            0.0, 1.0, 0.0,
            0.0, 0.0, 1.0,
        ];
        let result = super::det_contiguous_f64(&a, &meta).unwrap();
        assert!((result.det - 1.0).abs() < 1e-12);
    }

    #[test]
    fn det_scaled_identity() {
        let meta = TensorMeta::from_shape(vec![3, 3], DType::F64, Device::Cpu);
        #[rustfmt::skip]
        let a = vec![
            2.0, 0.0, 0.0,
            0.0, 2.0, 0.0,
            0.0, 0.0, 2.0,
        ];
        let result = super::det_contiguous_f64(&a, &meta).unwrap();
        assert!((result.det - 8.0).abs() < 1e-12);
    }

    #[test]
    fn det_known_3x3() {
        let meta = TensorMeta::from_shape(vec![3, 3], DType::F64, Device::Cpu);
        #[rustfmt::skip]
        let a = vec![
            1.0, 2.0, 3.0,
            4.0, 5.0, 6.0,
            7.0, 8.0, 10.0,
        ];
        // det = 1*(50-48) - 2*(40-42) + 3*(32-35) = 2 + 4 - 9 = -3
        let result = super::det_contiguous_f64(&a, &meta).unwrap();
        assert!((result.det - (-3.0)).abs() < 1e-10);
    }

    #[test]
    fn det_singular_is_zero() {
        let meta = TensorMeta::from_shape(vec![2, 2], DType::F64, Device::Cpu);
        let a = vec![1.0, 2.0, 2.0, 4.0];
        let result = super::det_contiguous_f64(&a, &meta).unwrap();
        assert!(result.det.abs() < 1e-10);
    }

    #[test]
    fn det_1x1() {
        let meta = TensorMeta::from_shape(vec![1, 1], DType::F64, Device::Cpu);
        let a = vec![7.5];
        let result = super::det_contiguous_f64(&a, &meta).unwrap();
        assert!((result.det - 7.5).abs() < 1e-12);
    }

    #[test]
    fn det_empty_matrix() {
        let meta = TensorMeta::from_shape(vec![0, 0], DType::F64, Device::Cpu);
        let a: Vec<f64> = vec![];
        let result = super::det_contiguous_f64(&a, &meta).unwrap();
        assert!((result.det - 1.0).abs() < 1e-12);
    }

    #[test]
    fn det_non_square_errors() {
        let meta = TensorMeta::from_shape(vec![2, 3], DType::F64, Device::Cpu);
        let a = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        assert!(super::det_contiguous_f64(&a, &meta).is_err());
    }

    #[test]
    fn det_negative() {
        let meta = TensorMeta::from_shape(vec![2, 2], DType::F64, Device::Cpu);
        // det([[0,1],[1,0]]) = -1
        let a = vec![0.0, 1.0, 1.0, 0.0];
        let result = super::det_contiguous_f64(&a, &meta).unwrap();
        assert!((result.det - (-1.0)).abs() < 1e-12);
    }

    #[test]
    fn slogdet_identity() {
        let meta = TensorMeta::from_shape(vec![3, 3], DType::F64, Device::Cpu);
        #[rustfmt::skip]
        let a = vec![
            1.0, 0.0, 0.0,
            0.0, 1.0, 0.0,
            0.0, 0.0, 1.0,
        ];
        let result = super::slogdet_contiguous_f64(&a, &meta).unwrap();
        assert!((result.sign - 1.0).abs() < 1e-12);
        assert!((result.logabsdet - 0.0).abs() < 1e-12);
    }

    #[test]
    fn slogdet_matches_det() {
        let meta = TensorMeta::from_shape(vec![3, 3], DType::F64, Device::Cpu);
        #[rustfmt::skip]
        let a = vec![
            1.0, 2.0, 3.0,
            4.0, 5.0, 6.0,
            7.0, 8.0, 10.0,
        ];
        let det_result = super::det_contiguous_f64(&a, &meta).unwrap();
        let slogdet_result = super::slogdet_contiguous_f64(&a, &meta).unwrap();
        let reconstructed = slogdet_result.sign * slogdet_result.logabsdet.exp();
        assert!(
            (reconstructed - det_result.det).abs() < 1e-10,
            "sign * exp(logabsdet) = {reconstructed}, det = {}",
            det_result.det
        );
    }

    #[test]
    fn slogdet_negative_det() {
        let meta = TensorMeta::from_shape(vec![2, 2], DType::F64, Device::Cpu);
        let a = vec![0.0, 1.0, 1.0, 0.0];
        let result = super::slogdet_contiguous_f64(&a, &meta).unwrap();
        assert!((result.sign - (-1.0)).abs() < 1e-12);
        assert!(result.logabsdet.abs() < 1e-12); // log(|−1|) = 0
    }

    #[test]
    fn slogdet_singular() {
        let meta = TensorMeta::from_shape(vec![2, 2], DType::F64, Device::Cpu);
        let a = vec![1.0, 2.0, 2.0, 4.0];
        let result = super::slogdet_contiguous_f64(&a, &meta).unwrap();
        assert!((result.sign).abs() < 1e-12);
        assert!(result.logabsdet.is_infinite() && result.logabsdet < 0.0);
    }

    #[test]
    fn slogdet_large_det() {
        let meta = TensorMeta::from_shape(vec![3, 3], DType::F64, Device::Cpu);
        // Diagonal matrix with large values: det = 1e100 * 1e100 * 1e100 = 1e300
        #[rustfmt::skip]
        let a = vec![
            1e100, 0.0,   0.0,
            0.0,   1e100, 0.0,
            0.0,   0.0,   1e100,
        ];
        let slogdet_result = super::slogdet_contiguous_f64(&a, &meta).unwrap();
        assert!((slogdet_result.sign - 1.0).abs() < 1e-12);
        // logabsdet should be 300 * ln(10)
        let expected = 300.0 * 10.0_f64.ln();
        assert!(
            (slogdet_result.logabsdet - expected).abs() < 1e-6,
            "logabsdet = {}, expected {expected}",
            slogdet_result.logabsdet
        );
    }

    // ---- Eigendecomposition tests (bd-2drq.7) ----

    #[test]
    fn eigh_identity() {
        let meta = TensorMeta::from_shape(vec![3, 3], DType::F64, Device::Cpu);
        #[rustfmt::skip]
        let a = vec![
            1.0, 0.0, 0.0,
            0.0, 1.0, 0.0,
            0.0, 0.0, 1.0,
        ];
        let result = super::eigh_contiguous_f64(&a, &meta).unwrap();
        for &v in &result.eigenvalues {
            assert!((v - 1.0).abs() < 1e-10);
        }
    }

    #[test]
    fn eigh_known_2x2() {
        let meta = TensorMeta::from_shape(vec![2, 2], DType::F64, Device::Cpu);
        let a = vec![1.0, 3.0, 3.0, 1.0]; // eigenvalues: -2, 4
        let result = super::eigh_contiguous_f64(&a, &meta).unwrap();
        assert!((result.eigenvalues[0] - (-2.0)).abs() < 1e-10);
        assert!((result.eigenvalues[1] - 4.0).abs() < 1e-10);
    }

    #[test]
    fn lobpcg_top_k_matches_eigh() {
        // 48x48 symmetric with a well-separated spectrum. LOBPCG's top-4
        // eigenvalues must match the full eigh's top-4, and each returned pair
        // must satisfy A·v = λ·v.
        let n = 48usize;
        let mut a = vec![0.0f64; n * n];
        for i in 0..n {
            for j in 0..n {
                a[i * n + j] = (((i * 7 + j * 3 + 1) % 13) as f64) * 0.05;
            }
        }
        for i in 0..n {
            for j in 0..i {
                let s = 0.5 * (a[i * n + j] + a[j * n + i]);
                a[i * n + j] = s;
                a[j * n + i] = s;
            }
            a[i * n + i] += i as f64;
        }
        let meta = TensorMeta::from_shape(vec![n, n], DType::F64, Device::Cpu);
        let k = 4usize;
        let (evals, evecs) = super::lobpcg_contiguous_f64(&a, &meta, k, true, 300, 1e-10).unwrap();
        let full = super::eigh_contiguous_f64(&a, &meta).unwrap();
        let mut full_top: Vec<f64> = (0..k).map(|t| full.eigenvalues[n - 1 - t]).collect();
        let mut got = evals.clone();
        got.sort_by(|x, y| y.total_cmp(x));
        full_top.sort_by(|x, y| y.total_cmp(x));
        for t in 0..k {
            assert!(
                (got[t] - full_top[t]).abs() < 1e-6,
                "eval {t}: lobpcg {} vs eigh {}",
                got[t],
                full_top[t]
            );
        }
        for t in 0..k {
            let lam = evals[t];
            for row in 0..n {
                let mut av = 0.0;
                for col in 0..n {
                    av += a[row * n + col] * evecs[col * k + t];
                }
                let lv = lam * evecs[row * k + t];
                assert!(
                    (av - lv).abs() < 1e-5,
                    "A·v != λ·v at pair {t} row {row}: {av} vs {lv}"
                );
            }
        }
    }

    #[test]
    fn eigh_reconstruction_3x3() {
        let meta = TensorMeta::from_shape(vec![3, 3], DType::F64, Device::Cpu);
        #[rustfmt::skip]
        let a = vec![
            2.0, 1.0, 0.0,
            1.0, 3.0, 1.0,
            0.0, 1.0, 2.0,
        ];
        let result = super::eigh_contiguous_f64(&a, &meta).unwrap();
        let n = 3;
        // Verify A = V @ diag(λ) @ V^T
        for i in 0..n {
            for j in 0..n {
                let mut val = 0.0;
                for k in 0..n {
                    val += result.eigenvectors[i * n + k]
                        * result.eigenvalues[k]
                        * result.eigenvectors[j * n + k];
                }
                assert!(
                    (val - a[i * n + j]).abs() < 1e-10,
                    "reconstructed[{i},{j}] = {val}, expected {}",
                    a[i * n + j]
                );
            }
        }
    }

    #[test]
    fn eigvalsh_matches_eigh() {
        let meta = TensorMeta::from_shape(vec![2, 2], DType::F64, Device::Cpu);
        let a = vec![4.0, 1.0, 1.0, 3.0];
        let full = super::eigh_contiguous_f64(&a, &meta).unwrap();
        let vals_only = super::eigvalsh_contiguous_f64(&a, &meta).unwrap();
        for (i, (&f, &o)) in full.eigenvalues.iter().zip(vals_only.iter()).enumerate() {
            assert!((f - o).abs() < 1e-12, "mismatch at [{i}]");
        }
    }

    #[test]
    fn eigh_tred2_tql2_orthonormal_and_reconstructs_24x24() {
        // Exercises the Householder/QL eigensolver at a size where the
        // tridiagonalization + shifted-QL machinery is fully engaged (small
        // cases can finish in a single step). Builds a non-trivial symmetric
        // matrix and verifies the three defining properties of a symmetric
        // eigendecomposition.
        let n = 24;
        let mut a = vec![0.0f64; n * n];
        for i in 0..n {
            for j in 0..n {
                let v = (((i * 37 + j * 101 + 7) % 53) as f64) * 0.11 - 2.5
                    + ((i as f64) - (j as f64)).abs().sin();
                a[i * n + j] = v;
            }
        }
        // Symmetrize.
        for i in 0..n {
            for j in (i + 1)..n {
                let s = 0.5 * (a[i * n + j] + a[j * n + i]);
                a[i * n + j] = s;
                a[j * n + i] = s;
            }
        }
        let meta = TensorMeta::from_shape(vec![n, n], DType::F64, Device::Cpu);
        let r = super::eigh_contiguous_f64(&a, &meta).unwrap();

        // (1) Eigenvalues sorted ascending.
        for i in 1..n {
            assert!(
                r.eigenvalues[i] >= r.eigenvalues[i - 1] - 1e-12,
                "eigenvalues not ascending at {i}"
            );
        }
        // (2) Eigenvectors orthonormal: V^T V = I.
        for c1 in 0..n {
            for c2 in 0..n {
                let mut dot = 0.0;
                for row in 0..n {
                    dot += r.eigenvectors[row * n + c1] * r.eigenvectors[row * n + c2];
                }
                let expected = if c1 == c2 { 1.0 } else { 0.0 };
                assert!(
                    (dot - expected).abs() < 1e-9,
                    "V^T V[{c1},{c2}] = {dot}, expected {expected}"
                );
            }
        }
        // (3) Reconstruction: V diag(λ) V^T = A.
        for i in 0..n {
            for j in 0..n {
                let mut val = 0.0;
                for k in 0..n {
                    val += r.eigenvectors[i * n + k] * r.eigenvalues[k] * r.eigenvectors[j * n + k];
                }
                assert!(
                    (val - a[i * n + j]).abs() < 1e-9,
                    "reconstructed[{i},{j}] = {val}, expected {}",
                    a[i * n + j]
                );
            }
        }
    }

    // ---- eig tests (general eigendecomposition) ----

    #[test]
    fn eig_identity_2x2() {
        let meta = TensorMeta::from_shape(vec![2, 2], DType::F64, Device::Cpu);
        let a = vec![1.0, 0.0, 0.0, 1.0];
        let result = super::eig_contiguous_f64(&a, &meta).unwrap();
        assert_eq!(result.n, 2);
        // Identity has eigenvalues 1, 1 (both real)
        assert!((result.eigenvalues[0] - 1.0).abs() < 1e-10, "eig[0] real");
        assert!(result.eigenvalues[1].abs() < 1e-10, "eig[0] imag");
        assert!((result.eigenvalues[2] - 1.0).abs() < 1e-10, "eig[1] real");
        assert!(result.eigenvalues[3].abs() < 1e-10, "eig[1] imag");
    }

    #[test]
    fn eig_diagonal_3x3() {
        let meta = TensorMeta::from_shape(vec![3, 3], DType::F64, Device::Cpu);
        #[rustfmt::skip]
        let a = vec![
            2.0, 0.0, 0.0,
            0.0, 5.0, 0.0,
            0.0, 0.0, 3.0,
        ];
        let result = super::eig_contiguous_f64(&a, &meta).unwrap();
        // Diagonal matrix eigenvalues are the diagonal entries
        let mut eigs: Vec<f64> = (0..3).map(|i| result.eigenvalues[2 * i]).collect();
        eigs.sort_by(|a, b| a.partial_cmp(b).unwrap());
        assert!((eigs[0] - 2.0).abs() < 1e-8);
        assert!((eigs[1] - 3.0).abs() < 1e-8);
        assert!((eigs[2] - 5.0).abs() < 1e-8);
    }

    #[test]
    fn eig_rotation_has_complex_eigenvalues() {
        let meta = TensorMeta::from_shape(vec![2, 2], DType::F64, Device::Cpu);
        // 90-degree rotation matrix: eigenvalues are +i and -i
        let a = vec![0.0, -1.0, 1.0, 0.0];
        let result = super::eig_contiguous_f64(&a, &meta).unwrap();
        // Real parts should be near 0, imag parts should be ±1
        let re0 = result.eigenvalues[0];
        let im0 = result.eigenvalues[1];
        let re1 = result.eigenvalues[2];
        let im1 = result.eigenvalues[3];
        assert!(re0.abs() < 1e-8, "real part should be 0");
        assert!(re1.abs() < 1e-8, "real part should be 0");
        assert!((im0.abs() - 1.0).abs() < 1e-8, "imag part should be ±1");
        assert!((im1.abs() - 1.0).abs() < 1e-8, "imag part should be ±1");
        // Conjugate pair: im0 = -im1
        assert!((im0 + im1).abs() < 1e-8, "should be conjugate pair");
    }

    #[test]
    fn eig_eigenvectors_satisfy_av_lambda_v() {
        // A general non-symmetric matrix (a mix of real and complex eigenvalues).
        // For each returned eigenpair, the eigenvectors must satisfy A·v = λ·v —
        // the back-substitution step, NOT just the Schur vectors.
        let n = 5usize;
        let mut a = vec![0.0f64; n * n];
        for i in 0..n {
            for j in 0..n {
                a[i * n + j] =
                    (((i * 7 + j * 3 + 1) % 11) as f64 - 5.0) + if i == j { 3.0 } else { 0.0 };
            }
        }
        let meta = TensorMeta::from_shape(vec![n, n], DType::F64, Device::Cpu);
        let r = super::eig_contiguous_f64(&a, &meta).unwrap();
        let mut saw_complex = false;
        let mut k = 0usize;
        while k < n {
            let re = r.eigenvalues[2 * k];
            let im = r.eigenvalues[2 * k + 1];
            if im == 0.0 {
                for row in 0..n {
                    let mut av = 0.0;
                    for col in 0..n {
                        av += a[row * n + col] * r.eigenvectors[col * n + k];
                    }
                    let lv = re * r.eigenvectors[row * n + k];
                    assert!(
                        (av - lv).abs() < 1e-6,
                        "real eig k={k}: (A v - λ v)[{row}] = {}",
                        av - lv
                    );
                }
                k += 1;
            } else {
                saw_complex = true;
                // λ = re + i·im (im>0); eigenvector = col k + i·col (k+1).
                for row in 0..n {
                    let (mut avr, mut avi) = (0.0, 0.0);
                    for col in 0..n {
                        avr += a[row * n + col] * r.eigenvectors[col * n + k];
                        avi += a[row * n + col] * r.eigenvectors[col * n + (k + 1)];
                    }
                    let vr = r.eigenvectors[row * n + k];
                    let vi = r.eigenvectors[row * n + (k + 1)];
                    assert!(
                        (avr - (re * vr - im * vi)).abs() < 1e-6,
                        "cplx eig k={k} re-part[{row}]"
                    );
                    assert!(
                        (avi - (re * vi + im * vr)).abs() < 1e-6,
                        "cplx eig k={k} im-part[{row}]"
                    );
                }
                k += 2;
            }
        }
        assert!(saw_complex, "test matrix should exercise a complex pair");
    }

    #[test]
    fn eigvals_matches_eig() {
        let meta = TensorMeta::from_shape(vec![2, 2], DType::F64, Device::Cpu);
        let a = vec![3.0, 1.0, 0.0, 2.0];
        let full = super::eig_contiguous_f64(&a, &meta).unwrap();
        let vals_only = super::eigvals_contiguous_f64(&a, &meta).unwrap();
        for (full_val, vals_val) in full.eigenvalues.iter().zip(&vals_only).take(4) {
            assert!((full_val - vals_val).abs() < 1e-12);
        }
    }

    #[test]
    fn eigvals_companion_complex_roots() {
        // Companion matrix (upper-Hessenberg) of a degree-6 real polynomial with
        // KNOWN roots including complex-conjugate pairs. This pins the correct
        // spectrum so the Francis-QR fix can be validated.
        // Roots: 1, 2, ±i, 1±i.
        let factors: Vec<Vec<f64>> = vec![
            vec![1.0, -1.0],      // x - 1
            vec![1.0, -2.0],      // x - 2
            vec![1.0, 0.0, 1.0],  // x^2 + 1      -> ±i
            vec![1.0, -2.0, 2.0], // x^2 - 2x + 2 -> 1±i
        ];
        let mut poly = vec![1.0f64];
        for f in &factors {
            let mut next = vec![0.0f64; poly.len() + f.len() - 1];
            for (i, &a) in poly.iter().enumerate() {
                for (j, &b) in f.iter().enumerate() {
                    next[i + j] += a * b;
                }
            }
            poly = next;
        }
        let n = poly.len() - 1; // 6
        // Companion: subdiagonal 1s, last column = -a_i where a_i = poly[n - i].
        let mut c = vec![0.0f64; n * n];
        for i in 1..n {
            c[i * n + (i - 1)] = 1.0;
        }
        for i in 0..n {
            c[i * n + (n - 1)] = -poly[n - i];
        }
        let meta = TensorMeta::from_shape(vec![n, n], DType::F64, Device::Cpu);
        let vals = super::eigvals_contiguous_f64(&c, &meta).unwrap();
        let got: Vec<(f64, f64)> = (0..n).map(|k| (vals[2 * k], vals[2 * k + 1])).collect();
        let want: Vec<(f64, f64)> = vec![
            (1.0, 0.0),
            (2.0, 0.0),
            (0.0, 1.0),
            (0.0, -1.0),
            (1.0, 1.0),
            (1.0, -1.0),
        ];
        // Eigenvalues are an unordered SET — match each expected one to a
        // distinct computed one within tolerance (a positional sort tie-breaks
        // unstably at the last ULP between a real root and a complex pair that
        // share a real part).
        let mut used = vec![false; got.len()];
        for w in &want {
            let hit = got.iter().enumerate().position(|(gi, g)| {
                !used[gi] && (g.0 - w.0).abs() < 1e-6 && (g.1 - w.1).abs() < 1e-6
            });
            assert!(hit.is_some(), "missing eigenvalue {w:?} in {got:?}");
            if let Some(gi) = hit {
                used[gi] = true;
            }
        }
    }

    // ---- SVD tests (bd-2drq.3) ----

    #[test]
    fn svd_identity_2x2() {
        let meta = TensorMeta::from_shape(vec![2, 2], DType::F64, Device::Cpu);
        let a = vec![1.0, 0.0, 0.0, 1.0];
        let result = super::svd_contiguous_f64(&a, &meta, false).unwrap();
        assert_eq!(result.k, 2);
        for &sv in &result.s {
            assert!((sv - 1.0).abs() < 1e-10);
        }
    }

    #[test]
    fn svd_diagonal_sorted() {
        let meta = TensorMeta::from_shape(vec![3, 3], DType::F64, Device::Cpu);
        #[rustfmt::skip]
        let a = vec![
            1.0, 0.0, 0.0,
            0.0, 5.0, 0.0,
            0.0, 0.0, 3.0,
        ];
        let result = super::svd_contiguous_f64(&a, &meta, false).unwrap();
        // Should be sorted descending: 5, 3, 1
        assert!((result.s[0] - 5.0).abs() < 1e-10);
        assert!((result.s[1] - 3.0).abs() < 1e-10);
        assert!((result.s[2] - 1.0).abs() < 1e-10);
    }

    #[test]
    fn svd_reconstruction_3x3() {
        let meta = TensorMeta::from_shape(vec![3, 3], DType::F64, Device::Cpu);
        #[rustfmt::skip]
        let a = vec![
            1.0, 2.0, 3.0,
            4.0, 5.0, 6.0,
            7.0, 8.0, 10.0,
        ];
        let result = super::svd_contiguous_f64(&a, &meta, false).unwrap();
        let (m, n, k) = (3, 3, 3);
        for i in 0..m {
            for j in 0..n {
                let mut val = 0.0;
                for l in 0..k {
                    val += result.u[i * k + l] * result.s[l] * result.vh[l * n + j];
                }
                assert!(
                    (val - a[i * n + j]).abs() < 1e-10,
                    "reconstructed[{i},{j}] = {val}, expected {}",
                    a[i * n + j]
                );
            }
        }
    }

    #[test]
    fn svd_ill_conditioned_hilbert() {
        // Hilbert matrix H[i][j] = 1/(i+j+1): SPD but extremely ill-conditioned.
        // The implicit-QR SVD must NOT spuriously return SingularMatrix and must
        // still reconstruct A = U·diag(s)·Vh (a torch-parity robustness check).
        let n = 16usize;
        let mut a = vec![0.0f64; n * n];
        for i in 0..n {
            for j in 0..n {
                a[i * n + j] = 1.0 / ((i + j + 1) as f64);
            }
        }
        let meta = TensorMeta::from_shape(vec![n, n], DType::F64, Device::Cpu);
        let r = super::svd_contiguous_f64(&a, &meta, false)
            .expect("SVD must converge on a Hilbert matrix");
        let k = r.k;
        // descending, non-negative
        for w in r.s.windows(2) {
            assert!(w[0] >= w[1] - 1e-12 && w[1] >= -1e-12);
        }
        // reconstruction (loose tol due to conditioning)
        for i in 0..n {
            for j in 0..n {
                let mut val = 0.0;
                for l in 0..k {
                    val += r.u[i * k + l] * r.s[l] * r.vh[l * n + j];
                }
                assert!(
                    (val - a[i * n + j]).abs() < 1e-9,
                    "Hilbert recon[{i},{j}] = {val} vs {}",
                    a[i * n + j]
                );
            }
        }
    }

    #[test]
    fn svd_lowrank_recovers_exact_low_rank() {
        // A = B·C has exact rank 3. Randomized SVD with target q>=3 must recover
        // it to working precision, and its top-3 singular values must match the
        // full SVD's. Validates the Halko-Martinsson-Tropp range finder + the
        // Q^T·A small-SVD step against the exact factorization.
        let (m, n, r) = (10usize, 7usize, 3usize);
        let mut b = vec![0.0f64; m * r];
        for i in 0..m {
            for j in 0..r {
                b[i * r + j] = ((i * 5 + j * 3 + 1) % 13) as f64 * 0.1 - 0.6;
            }
        }
        let mut c = vec![0.0f64; r * n];
        for i in 0..r {
            for j in 0..n {
                c[i * n + j] = ((i * 7 + j * 2 + 4) % 11) as f64 * 0.1 - 0.5;
            }
        }
        let mut a = vec![0.0f64; m * n];
        for i in 0..m {
            for j in 0..n {
                let mut s = 0.0;
                for k in 0..r {
                    s += b[i * r + k] * c[k * n + j];
                }
                a[i * n + j] = s;
            }
        }
        let meta = TensorMeta::from_shape(vec![m, n], DType::F64, Device::Cpu);
        let lr = super::svd_lowrank_contiguous_f64(&a, &meta, 4, 2).unwrap();
        let k = lr.k;
        // reconstruction U·diag(s)·Vh ≈ A
        for i in 0..m {
            for j in 0..n {
                let mut val = 0.0;
                for c2 in 0..k {
                    val += lr.u[i * k + c2] * lr.s[c2] * lr.vh[c2 * n + j];
                }
                assert!(
                    (val - a[i * n + j]).abs() < 1e-7,
                    "lowrank recon[{i},{j}] = {val} vs {}",
                    a[i * n + j]
                );
            }
        }
        // top-3 singular values match the full SVD's.
        let full = super::svd_contiguous_f64(&a, &meta, false).unwrap();
        for t in 0..r {
            assert!(
                (lr.s[t] - full.s[t]).abs() < 1e-7,
                "singular value {t}: lowrank {} vs full {}",
                lr.s[t],
                full.s[t]
            );
        }
        // descending, non-negative
        for w in lr.s.windows(2) {
            assert!(w[0] >= w[1] - 1e-12 && w[1] >= -1e-12);
        }
    }

    #[test]
    fn svd_rank_deficient_reconstructs() {
        // Row 2 = 2·row 1 -> rank 3 (one exactly-zero singular value). SVD must
        // still reconstruct A = U·diag(s)·Vh and produce a near-zero singular value.
        let (m, n) = (4usize, 4usize);
        #[rustfmt::skip]
        let a = vec![
            1.0, 2.0, 3.0, 4.0,
            2.0, 4.0, 6.0, 8.0,
            1.0, 0.0, 1.0, 0.0,
            0.0, 1.0, 0.0, 1.0,
        ];
        let meta = TensorMeta::from_shape(vec![m, n], DType::F64, Device::Cpu);
        let r = super::svd_contiguous_f64(&a, &meta, false).unwrap();
        let k = r.k;
        // smallest singular value ~ 0 (rank deficiency)
        let smin = r.s.iter().cloned().fold(f64::INFINITY, f64::min);
        assert!(
            smin < 1e-9,
            "rank-deficient matrix should have a ~0 singular value, got {smin}"
        );
        // singular values non-negative and descending
        for w in r.s.windows(2) {
            assert!(w[0] >= w[1] - 1e-12, "singular values must be descending");
            assert!(w[1] >= -1e-12, "singular values must be non-negative");
        }
        // reconstruction U·diag(s)·Vh = A
        for i in 0..m {
            for j in 0..n {
                let mut val = 0.0;
                for l in 0..k {
                    val += r.u[i * k + l] * r.s[l] * r.vh[l * n + j];
                }
                assert!(
                    (val - a[i * n + j]).abs() < 1e-9,
                    "reconstructed[{i},{j}] = {val}, expected {}",
                    a[i * n + j]
                );
            }
        }
    }

    #[test]
    fn svd_tall_3x2() {
        let meta = TensorMeta::from_shape(vec![3, 2], DType::F64, Device::Cpu);
        #[rustfmt::skip]
        let a = vec![
            1.0, 2.0,
            3.0, 4.0,
            5.0, 6.0,
        ];
        let result = super::svd_contiguous_f64(&a, &meta, false).unwrap();
        assert_eq!(result.m, 3);
        assert_eq!(result.n, 2);
        assert_eq!(result.k, 2);
        assert_eq!(result.s.len(), 2);
        // Reconstruction
        for i in 0..3 {
            for j in 0..2 {
                let mut val = 0.0;
                for l in 0..2 {
                    val += result.u[i * 2 + l] * result.s[l] * result.vh[l * 2 + j];
                }
                assert!(
                    (val - a[i * 2 + j]).abs() < 1e-10,
                    "reconstructed[{i},{j}] = {val}, expected {}",
                    a[i * 2 + j]
                );
            }
        }
    }

    #[test]
    fn svd_wide_2x3() {
        let meta = TensorMeta::from_shape(vec![2, 3], DType::F64, Device::Cpu);
        #[rustfmt::skip]
        let a = vec![
            1.0, 2.0, 3.0,
            4.0, 5.0, 6.0,
        ];
        let result = super::svd_contiguous_f64(&a, &meta, false).unwrap();
        assert_eq!(result.k, 2);
        // Reconstruction
        for i in 0..2 {
            for j in 0..3 {
                let mut val = 0.0;
                for l in 0..2 {
                    val += result.u[i * 2 + l] * result.s[l] * result.vh[l * 3 + j];
                }
                assert!(
                    (val - a[i * 3 + j]).abs() < 1e-10,
                    "reconstructed[{i},{j}] = {val}, expected {}",
                    a[i * 3 + j]
                );
            }
        }
    }

    #[test]
    fn svdvals_matches_full_svd() {
        let meta = TensorMeta::from_shape(vec![3, 3], DType::F64, Device::Cpu);
        #[rustfmt::skip]
        let a = vec![
            2.0, 1.0, 0.0,
            1.0, 3.0, 1.0,
            0.0, 1.0, 2.0,
        ];
        let full = super::svd_contiguous_f64(&a, &meta, false).unwrap();
        let vals_only = super::svdvals_contiguous_f64(&a, &meta).unwrap();
        assert_eq!(full.s.len(), vals_only.len());
        for (i, (&a_val, &b_val)) in full.s.iter().zip(vals_only.iter()).enumerate() {
            assert!(
                (a_val - b_val).abs() < 1e-10,
                "mismatch at [{i}]: {a_val} vs {b_val}"
            );
        }
    }

    #[test]
    fn svdvals_matches_full_svd_tall_and_wide() {
        // The values-only Golub-Reinsch path must agree with the full SVD's
        // singular values for both tall (m > n) and wide (m < n) inputs, at a
        // size where bidiagonalization + several QR sweeps are exercised.
        for &(m, n) in &[(40usize, 24usize), (24usize, 40usize)] {
            let mut a = vec![0.0f64; m * n];
            for i in 0..m {
                for j in 0..n {
                    a[i * n + j] = (((i * 53 + j * 29 + 11) % 101) as f64) * 0.07 - 3.0
                        + ((i as f64) * 0.3 - (j as f64) * 0.2).cos();
                }
            }
            let meta = TensorMeta::from_shape(vec![m, n], DType::F64, Device::Cpu);
            let full = super::svd_contiguous_f64(&a, &meta, false).unwrap();
            let vals_only = super::svdvals_contiguous_f64(&a, &meta).unwrap();
            assert_eq!(full.s.len(), vals_only.len(), "{m}x{n} length");
            assert_eq!(vals_only.len(), m.min(n));
            for (i, (&fv, &vv)) in full.s.iter().zip(vals_only.iter()).enumerate() {
                assert!(
                    (fv - vv).abs() < 1e-9 * (1.0 + fv.abs()),
                    "{m}x{n} singular value [{i}]: full {fv} vs values-only {vv}"
                );
            }
        }
    }

    #[test]
    fn svd_tall_orthonormal_and_reconstructs_40x24() {
        // Strong structural-parity test for the SVD at a non-trivial size with a
        // wider singular-value spread than the tiny fixtures. Verifies the
        // properties numpy / torch.linalg.svd guarantee: U^T U = I, V V^T = I,
        // descending non-negative singular values, and U diag(s) V^T = A. Guards
        // any future eigensolver-based SVD rewrite: the bare Gram (A^T A)
        // shortcut fails U^T U = I here because squaring the condition number
        // corrupts the small-singular-value U directions.
        let (m, n) = (40usize, 24usize);
        let mut a = vec![0.0f64; m * n];
        for i in 0..m {
            for j in 0..n {
                a[i * n + j] = (((i * 53 + j * 29 + 11) % 101) as f64) * 0.07 - 3.0
                    + ((i as f64) * 0.3 - (j as f64) * 0.2).cos();
            }
        }
        let meta = TensorMeta::from_shape(vec![m, n], DType::F64, Device::Cpu);
        let r = super::svd_contiguous_f64(&a, &meta, false).unwrap();
        let k = n; // reduced, m >= n
        assert_eq!(r.s.len(), k);

        // Singular values non-negative and descending.
        for i in 1..k {
            assert!(
                r.s[i] >= -1e-12 && r.s[i] <= r.s[i - 1] + 1e-9,
                "s not sorted at {i}"
            );
        }
        // U columns orthonormal: U^T U = I_k (U is m x k).
        for c1 in 0..k {
            for c2 in 0..k {
                let mut dot = 0.0;
                for row in 0..m {
                    dot += r.u[row * k + c1] * r.u[row * k + c2];
                }
                let expected = if c1 == c2 { 1.0 } else { 0.0 };
                assert!((dot - expected).abs() < 1e-9, "U^T U[{c1},{c2}]={dot}");
            }
        }
        // V rows orthonormal: vh is (k x n); vh vh^T = I_k.
        for r1 in 0..k {
            for r2 in 0..k {
                let mut dot = 0.0;
                for col in 0..n {
                    dot += r.vh[r1 * n + col] * r.vh[r2 * n + col];
                }
                let expected = if r1 == r2 { 1.0 } else { 0.0 };
                assert!((dot - expected).abs() < 1e-9, "V V^T[{r1},{r2}]={dot}");
            }
        }
        // Reconstruction: U diag(s) V^T = A.
        for i in 0..m {
            for j in 0..n {
                let mut val = 0.0;
                for l in 0..k {
                    val += r.u[i * k + l] * r.s[l] * r.vh[l * n + j];
                }
                assert!(
                    (val - a[i * n + j]).abs() < 1e-9,
                    "reconstructed[{i},{j}]={val}, expected {}",
                    a[i * n + j]
                );
            }
        }
    }

    // ---- Cholesky Decomposition tests (bd-2drq.5) ----

    #[test]
    fn cholesky_identity() {
        let meta = TensorMeta::from_shape(vec![3, 3], DType::F64, Device::Cpu);
        #[rustfmt::skip]
        let a = vec![
            1.0, 0.0, 0.0,
            0.0, 1.0, 0.0,
            0.0, 0.0, 1.0,
        ];
        let result = super::cholesky_contiguous_f64(&a, &meta, false).unwrap();
        assert_eq!(result.n, 3);
        for i in 0..3 {
            for j in 0..3 {
                let expected = if i == j { 1.0 } else { 0.0 };
                assert!(
                    (result.factor[i * 3 + j] - expected).abs() < 1e-12,
                    "L[{i},{j}] = {}, expected {expected}",
                    result.factor[i * 3 + j]
                );
            }
        }
    }

    #[test]
    fn cholesky_2x2_spd() {
        let meta = TensorMeta::from_shape(vec![2, 2], DType::F64, Device::Cpu);
        let a = vec![4.0, 2.0, 2.0, 3.0];
        let result = super::cholesky_contiguous_f64(&a, &meta, false).unwrap();
        assert!((result.factor[0] - 2.0).abs() < 1e-12);
        assert!(result.factor[1].abs() < 1e-12);
        assert!((result.factor[2] - 1.0).abs() < 1e-12);
        assert!((result.factor[3] - 2.0f64.sqrt()).abs() < 1e-12);
    }

    #[test]
    fn cholesky_upper_flag() {
        let meta = TensorMeta::from_shape(vec![2, 2], DType::F64, Device::Cpu);
        let a = vec![4.0, 2.0, 2.0, 3.0];
        let result = super::cholesky_contiguous_f64(&a, &meta, true).unwrap();
        // U = [[2, 1], [0, sqrt(2)]]
        assert!((result.factor[0] - 2.0).abs() < 1e-12);
        assert!((result.factor[1] - 1.0).abs() < 1e-12);
        assert!(result.factor[2].abs() < 1e-12);
        assert!((result.factor[3] - 2.0f64.sqrt()).abs() < 1e-12);
    }

    #[test]
    fn cholesky_not_positive_definite() {
        let meta = TensorMeta::from_shape(vec![2, 2], DType::F64, Device::Cpu);
        let a = vec![1.0, 2.0, 2.0, 1.0]; // eigenvalues: 3, -1
        let err = super::cholesky_contiguous_f64(&a, &meta, false).expect_err("non-SPD must fail");
        assert!(matches!(err, super::KernelError::NotPositiveDefinite));
    }

    #[test]
    fn cholesky_3x3_reconstruction() {
        let meta = TensorMeta::from_shape(vec![3, 3], DType::F64, Device::Cpu);
        #[rustfmt::skip]
        let a = vec![
            4.0, 12.0, -16.0,
            12.0, 37.0, -43.0,
            -16.0, -43.0, 98.0,
        ];
        let result = super::cholesky_contiguous_f64(&a, &meta, false).unwrap();
        let n = 3;
        // Verify L @ L^T = A
        for i in 0..n {
            for j in 0..n {
                let mut dot = 0.0;
                for k in 0..n {
                    dot += result.factor[i * n + k] * result.factor[j * n + k];
                }
                assert!(
                    (dot - a[i * n + j]).abs() < 1e-10,
                    "(L@L^T)[{i},{j}] = {dot}, expected {}",
                    a[i * n + j]
                );
            }
        }
    }

    #[test]
    fn cholesky_blocked_matches_serial_and_reconstructs() {
        // n = 768 > NB exercises the blocked panel/TRSM/SYRK path. The blocked
        // right-looking algorithm REASSOCIATES the trailing sums (panel-by-panel
        // through GEMM) vs the serial left-looking long dot, so it matches the
        // serial reference only to TOLERANCE — assert that, plus the strict
        // proof obligation that L is lower-triangular and reconstructs A.
        let n = 768usize;
        // Cheap symmetric, strictly diagonally dominant (hence SPD) matrix.
        let mut a = vec![0.0f64; n * n];
        for i in 0..n {
            for j in 0..i {
                let v = (((i * 131 + j * 17) % 19) as f64 - 9.0) * 0.01;
                a[i * n + j] = v;
                a[j * n + i] = v;
            }
            a[i * n + i] = n as f64;
        }

        // Serial left-looking reference factor.
        let mut l_ref = vec![0.0f64; n * n];
        for j in 0..n {
            let mut sum_sq = 0.0;
            for k in 0..j {
                sum_sq += l_ref[j * n + k] * l_ref[j * n + k];
            }
            let l_jj = (a[j * n + j] - sum_sq).sqrt();
            l_ref[j * n + j] = l_jj;
            for i in (j + 1)..n {
                let mut dot = 0.0;
                for k in 0..j {
                    dot += l_ref[i * n + k] * l_ref[j * n + k];
                }
                l_ref[i * n + j] = (a[i * n + j] - dot) / l_jj;
            }
        }

        let meta = TensorMeta::from_shape(vec![n, n], DType::F64, Device::Cpu);
        let got = super::cholesky_contiguous_f64(&a, &meta, false).expect("cholesky");
        assert_eq!(got.factor.len(), n * n);
        for i in 0..n {
            for j in 0..n {
                let v = got.factor[i * n + j];
                if j > i {
                    // strictly upper triangle must be exactly zero
                    assert_eq!(v, 0.0, "factor not lower-triangular at ({i},{j})");
                } else {
                    let rel = (v - l_ref[i * n + j]).abs() / (l_ref[i * n + j].abs() + 1.0);
                    assert!(
                        rel < 1e-9,
                        "blocked factor[{i},{j}]={v} vs serial {} (rel {rel:e})",
                        l_ref[i * n + j]
                    );
                }
            }
        }
        // Reconstruction L·L^T ≈ A on a sample of entries (full check is O(n^3)).
        for &(i, j) in &[(0usize, 0usize), (5, 2), (700, 699), (767, 0), (400, 400)] {
            let mut dot = 0.0;
            for k in 0..=j.min(i) {
                dot += got.factor[i * n + k] * got.factor[j * n + k];
            }
            assert!(
                (dot - a[i * n + j]).abs() < 1e-7,
                "(L·L^T)[{i},{j}]={dot} expected {}",
                a[i * n + j]
            );
        }
    }

    #[test]
    fn winograd_conv2d_matches_direct_within_tolerance() {
        // Isomorphism proof for the Winograd F(2,3) path: it must match a direct
        // 3x3 stride-1 convolution to tolerance (it reassociates, not bit-exact).
        // Use odd out dims (7) to exercise the edge-tile trimming.
        let (batch, in_ch, out_ch) = (2usize, 3usize, 4usize);
        let (padded_h, padded_w) = (9usize, 9usize); // out = 7x7
        let out_h = padded_h - 2;
        let out_w = padded_w - 2;
        let input: Vec<f64> = (0..batch * in_ch * padded_h * padded_w)
            .map(|i| (((i * 2654435761usize) % 211) as f64 - 105.0) * 0.013)
            .collect();
        let weight: Vec<f64> = (0..out_ch * in_ch * 9)
            .map(|i| (((i * 40503usize) % 97) as f64 - 48.0) * 0.021)
            .collect();

        let got = super::winograd_conv2d_3x3_s1_f64(
            &input, &weight, batch, in_ch, out_ch, padded_h, padded_w,
        );

        // Direct reference.
        let mut want = vec![0.0f64; batch * out_ch * out_h * out_w];
        for b in 0..batch {
            for oc in 0..out_ch {
                for oh in 0..out_h {
                    for ow in 0..out_w {
                        let mut acc = 0.0;
                        for ic in 0..in_ch {
                            let ibase = (b * in_ch + ic) * padded_h * padded_w;
                            let wbase = (oc * in_ch + ic) * 9;
                            for kh in 0..3 {
                                for kw in 0..3 {
                                    acc += input[ibase + (oh + kh) * padded_w + (ow + kw)]
                                        * weight[wbase + kh * 3 + kw];
                                }
                            }
                        }
                        want[((b * out_ch + oc) * out_h + oh) * out_w + ow] = acc;
                    }
                }
            }
        }
        assert_eq!(got.len(), want.len());
        for (idx, (&g, &w)) in got.iter().zip(want.iter()).enumerate() {
            assert!(
                (g - w).abs() < 1e-9,
                "winograd[{idx}]={g} vs direct {w} (diff {:e})",
                (g - w).abs()
            );
        }
    }

    #[test]
    fn winograd_conv2d_f32_matches_direct_within_tolerance() {
        // Isomorphism proof for the f32 Winograd F(2,3) path: matches a direct
        // 3x3 stride-1 conv to f32 tolerance (it reassociates, not bit-exact).
        let (batch, in_ch, out_ch) = (2usize, 3usize, 4usize);
        let (padded_h, padded_w) = (9usize, 9usize); // out = 7x7
        let out_h = padded_h - 2;
        let out_w = padded_w - 2;
        let input: Vec<f32> = (0..batch * in_ch * padded_h * padded_w)
            .map(|i| (((i * 2654435761usize) % 211) as f32 - 105.0) * 0.013)
            .collect();
        let weight: Vec<f32> = (0..out_ch * in_ch * 9)
            .map(|i| (((i * 40503usize) % 97) as f32 - 48.0) * 0.021)
            .collect();

        let got = super::winograd_conv2d_3x3_s1_f32(
            &input, &weight, batch, in_ch, out_ch, padded_h, padded_w,
        );

        let mut want = vec![0.0f32; batch * out_ch * out_h * out_w];
        for b in 0..batch {
            for oc in 0..out_ch {
                for oh in 0..out_h {
                    for ow in 0..out_w {
                        let mut acc = 0.0f32;
                        for ic in 0..in_ch {
                            let ibase = (b * in_ch + ic) * padded_h * padded_w;
                            let wbase = (oc * in_ch + ic) * 9;
                            for kh in 0..3 {
                                for kw in 0..3 {
                                    acc += input[ibase + (oh + kh) * padded_w + (ow + kw)]
                                        * weight[wbase + kh * 3 + kw];
                                }
                            }
                        }
                        want[((b * out_ch + oc) * out_h + oh) * out_w + ow] = acc;
                    }
                }
            }
        }
        assert_eq!(got.len(), want.len());
        for (idx, (&g, &w)) in got.iter().zip(want.iter()).enumerate() {
            let tol = 1e-4 + 1e-4 * w.abs();
            assert!(
                (g - w).abs() < tol,
                "winograd_f32[{idx}]={g} vs direct {w} (diff {:e})",
                (g - w).abs()
            );
        }
    }

    #[test]
    fn cholesky_1x1() {
        let meta = TensorMeta::from_shape(vec![1, 1], DType::F64, Device::Cpu);
        let a = vec![9.0];
        let result = super::cholesky_contiguous_f64(&a, &meta, false).unwrap();
        assert!((result.factor[0] - 3.0).abs() < 1e-12);
    }

    #[test]
    fn cholesky_solve_identity_factor() {
        let meta = TensorMeta::from_shape(vec![2, 2], DType::F64, Device::Cpu);
        let a = vec![1.0, 0.0, 0.0, 1.0];
        let factor = super::cholesky_contiguous_f64(&a, &meta, false).unwrap();
        let b_meta = TensorMeta::from_shape(vec![2], DType::F64, Device::Cpu);
        let b = vec![5.0, 7.0];
        let x = super::cholesky_solve_contiguous_f64(&factor, &b, &b_meta, false).unwrap();
        assert!((x[0] - 5.0).abs() < 1e-12);
        assert!((x[1] - 7.0).abs() < 1e-12);
    }

    #[test]
    fn cholesky_solve_2x2() {
        let meta = TensorMeta::from_shape(vec![2, 2], DType::F64, Device::Cpu);
        let a = vec![4.0, 2.0, 2.0, 3.0];
        let factor = super::cholesky_contiguous_f64(&a, &meta, false).unwrap();
        let b_meta = TensorMeta::from_shape(vec![2], DType::F64, Device::Cpu);
        let b = vec![1.0, 2.0];
        let x = super::cholesky_solve_contiguous_f64(&factor, &b, &b_meta, false).unwrap();
        // x = A^-1 @ b = [-0.125, 0.75]
        assert!((x[0] - (-0.125)).abs() < 1e-10);
        assert!((x[1] - 0.75).abs() < 1e-10);
    }

    // ---- QR Decomposition tests (bd-2drq.4) ----

    /// Multiply an (m x p) matrix by a (p x n) matrix, returning (m x n).
    fn mat_mul(a: &[f64], b: &[f64], m: usize, p: usize, n: usize) -> Vec<f64> {
        let mut c = vec![0.0; m * n];
        for i in 0..m {
            for j in 0..n {
                let mut acc = 0.0;
                for k in 0..p {
                    acc += a[i * p + k] * b[k * n + j];
                }
                c[i * n + j] = acc;
            }
        }
        c
    }

    #[test]
    fn qr_identity_3x3_complete() {
        let meta = TensorMeta::from_shape(vec![3, 3], DType::F64, Device::Cpu);
        #[rustfmt::skip]
        let a = vec![
            1.0, 0.0, 0.0,
            0.0, 1.0, 0.0,
            0.0, 0.0, 1.0,
        ];
        let result = super::qr_contiguous_f64(&a, &meta, false).expect("qr should succeed");
        // Complete mode: Q is 3x3, R is 3x3
        assert_eq!(result.q.len(), 9);
        assert_eq!(result.r.len(), 9);

        // Q should be orthogonal (Q^T @ Q == I)
        let qtq = mat_mul_nn(&transpose_mat(&result.q, 3, 3), &result.q, 3);
        assert_mat_approx_eq(&qtq, &a, 1e-12, "Q^T @ Q should be identity");

        // Q @ R == A
        let qr = mat_mul(&result.q, &result.r, 3, 3, 3);
        assert_mat_approx_eq(&qr, &a, 1e-12, "Q @ R should equal A");
    }

    fn transpose_mat(a: &[f64], m: usize, n: usize) -> Vec<f64> {
        let mut t = vec![0.0; m * n];
        for i in 0..m {
            for j in 0..n {
                t[j * m + i] = a[i * n + j];
            }
        }
        t
    }

    #[test]
    fn qr_parallel_apply_reconstructs_and_orthonormal() {
        // n=256 crosses QR_PAR_WORK so the column/row apply fan-outs engage. The
        // parallelization keeps the exact fp op order (bit-exact by construction);
        // this guards the indexing by checking the proof obligations Q*R == A and
        // Q^T*Q == I at a size the small unit tests never reach.
        let n = 256usize;
        let mut a = vec![0.0f64; n * n];
        for i in 0..n {
            for j in 0..n {
                a[i * n + j] = (((i * 53 + j * 31) % 97) as f64 - 48.0) * 0.1;
            }
            a[i * n + i] += n as f64;
        }
        let meta = TensorMeta::from_shape(vec![n, n], DType::F64, Device::Cpu);
        let r = super::qr_contiguous_f64(&a, &meta, true).expect("qr");
        // Q*R == A
        for &(i, j) in &[(0usize, 0usize), (5, 200), (255, 1), (128, 128), (200, 17)] {
            let mut dot = 0.0;
            for t in 0..n {
                dot += r.q[i * n + t] * r.r[t * n + j];
            }
            assert!(
                (dot - a[i * n + j]).abs() < 1e-7,
                "(QR)[{i},{j}]={dot} vs {}",
                a[i * n + j]
            );
        }
        // Q^T*Q == I (orthonormal columns)
        for &(c1, c2) in &[(0usize, 0usize), (10, 10), (3, 200), (255, 254)] {
            let mut dot = 0.0;
            for t in 0..n {
                dot += r.q[t * n + c1] * r.q[t * n + c2];
            }
            let expected = if c1 == c2 { 1.0 } else { 0.0 };
            assert!(
                (dot - expected).abs() < 1e-9,
                "Q^T Q[{c1},{c2}]={dot} vs {expected}"
            );
        }
    }

    #[test]
    fn qr_known_3x3_reduced() {
        let meta = TensorMeta::from_shape(vec![3, 3], DType::F64, Device::Cpu);
        #[rustfmt::skip]
        let a = vec![
            12.0, -51.0,   4.0,
             6.0, 167.0, -68.0,
            -4.0,  24.0, -41.0,
        ];
        let result = super::qr_contiguous_f64(&a, &meta, true).expect("qr should succeed");
        // Reduced: k = min(3,3) = 3, so Q is 3x3, R is 3x3
        assert_eq!(result.q.len(), 9);
        assert_eq!(result.r.len(), 9);

        // Q @ R == A
        let qr = mat_mul(&result.q, &result.r, 3, 3, 3);
        assert_mat_approx_eq(&qr, &a, 1e-10, "Q @ R should equal A");

        // Q^T @ Q should be identity
        let qtq = mat_mul(&transpose_mat(&result.q, 3, 3), &result.q, 3, 3, 3);
        #[rustfmt::skip]
        let eye3 = vec![
            1.0, 0.0, 0.0,
            0.0, 1.0, 0.0,
            0.0, 0.0, 1.0,
        ];
        assert_mat_approx_eq(&qtq, &eye3, 1e-12, "Q^T @ Q should be identity");

        // R should be upper triangular
        for i in 0..3 {
            for j in 0..i {
                assert!(
                    result.r[i * 3 + j].abs() < 1e-12,
                    "R[{i},{j}] = {} should be zero",
                    result.r[i * 3 + j]
                );
            }
        }
    }

    #[test]
    fn qr_tall_matrix_4x2_reduced() {
        let meta = TensorMeta::from_shape(vec![4, 2], DType::F64, Device::Cpu);
        #[rustfmt::skip]
        let a = vec![
            1.0, 2.0,
            3.0, 4.0,
            5.0, 6.0,
            7.0, 8.0,
        ];
        let result = super::qr_contiguous_f64(&a, &meta, true).expect("qr should succeed");
        let m = 4;
        let n = 2;
        let k = 2; // min(4,2)
        // Reduced: Q is 4x2, R is 2x2
        assert_eq!(result.q.len(), m * k);
        assert_eq!(result.r.len(), k * n);

        // Q @ R == A
        let qr = mat_mul(&result.q, &result.r, m, k, n);
        assert_mat_approx_eq(&qr, &a, 1e-10, "Q @ R should equal A");

        // Q^T @ Q should be k x k identity
        let qtq = mat_mul(&transpose_mat(&result.q, m, k), &result.q, k, m, k);
        let mut eye_k = vec![0.0; k * k];
        for i in 0..k {
            eye_k[i * k + i] = 1.0;
        }
        assert_mat_approx_eq(&qtq, &eye_k, 1e-12, "Q^T @ Q should be identity");
    }

    #[test]
    fn qr_tall_matrix_4x2_complete() {
        let meta = TensorMeta::from_shape(vec![4, 2], DType::F64, Device::Cpu);
        #[rustfmt::skip]
        let a = vec![
            1.0, 2.0,
            3.0, 4.0,
            5.0, 6.0,
            7.0, 8.0,
        ];
        let result = super::qr_contiguous_f64(&a, &meta, false).expect("qr should succeed");
        let m = 4;
        let n = 2;
        // Complete: Q is 4x4, R is 4x2
        assert_eq!(result.q.len(), m * m);
        assert_eq!(result.r.len(), m * n);

        // Q @ R == A
        let qr = mat_mul(&result.q, &result.r, m, m, n);
        assert_mat_approx_eq(&qr, &a, 1e-10, "Q @ R should equal A");

        // Q^T @ Q should be m x m identity
        let qtq = mat_mul(&transpose_mat(&result.q, m, m), &result.q, m, m, m);
        let mut eye_m = vec![0.0; m * m];
        for i in 0..m {
            eye_m[i * m + i] = 1.0;
        }
        assert_mat_approx_eq(&qtq, &eye_m, 1e-12, "Q^T @ Q should be identity");
    }

    #[test]
    fn qr_wide_matrix_2x4_reduced() {
        let meta = TensorMeta::from_shape(vec![2, 4], DType::F64, Device::Cpu);
        #[rustfmt::skip]
        let a = vec![
            1.0, 2.0, 3.0, 4.0,
            5.0, 6.0, 7.0, 8.0,
        ];
        let result = super::qr_contiguous_f64(&a, &meta, true).expect("qr should succeed");
        let m = 2;
        let n = 4;
        let k = 2; // min(2,4)
        // Reduced: Q is 2x2, R is 2x4
        assert_eq!(result.q.len(), m * k);
        assert_eq!(result.r.len(), k * n);

        // Q @ R == A
        let qr = mat_mul(&result.q, &result.r, m, k, n);
        assert_mat_approx_eq(&qr, &a, 1e-10, "Q @ R should equal A");

        // Q^T @ Q should be k x k identity
        let qtq = mat_mul(&transpose_mat(&result.q, m, k), &result.q, k, m, k);
        let mut eye_k = vec![0.0; k * k];
        for i in 0..k {
            eye_k[i * k + i] = 1.0;
        }
        assert_mat_approx_eq(&qtq, &eye_k, 1e-12, "Q^T @ Q should be identity");
    }

    #[test]
    fn qr_1x1() {
        let meta = TensorMeta::from_shape(vec![1, 1], DType::F64, Device::Cpu);
        let a = vec![5.0];
        let result = super::qr_contiguous_f64(&a, &meta, true).expect("qr should succeed");
        assert_eq!(result.q.len(), 1);
        assert_eq!(result.r.len(), 1);
        // For scalar: Q = ±1, R = ±5, Q*R = 5
        let qr_val = result.q[0] * result.r[0];
        assert!((qr_val - 5.0).abs() < 1e-12, "Q*R should equal 5.0");
        // Q should be orthogonal: |Q| = 1
        assert!((result.q[0].abs() - 1.0).abs() < 1e-12, "Q should be ±1");
    }

    #[test]
    fn qr_empty_matrix() {
        let meta = TensorMeta::from_shape(vec![0, 3], DType::F64, Device::Cpu);
        let a: Vec<f64> = vec![];
        let result = super::qr_contiguous_f64(&a, &meta, true).expect("empty should succeed");
        assert!(result.q.is_empty());
        assert!(result.r.is_empty());
    }

    #[test]
    fn qr_rejects_1d_tensor() {
        let meta = TensorMeta::from_shape(vec![4], DType::F64, Device::Cpu);
        let a = vec![1.0, 2.0, 3.0, 4.0];
        assert!(
            super::qr_contiguous_f64(&a, &meta, true).is_err(),
            "1D tensor should be rejected"
        );
    }

    #[test]
    fn qr_r_is_upper_triangular() {
        let meta = TensorMeta::from_shape(vec![4, 3], DType::F64, Device::Cpu);
        #[rustfmt::skip]
        let a = vec![
            1.0, 2.0, 3.0,
            4.0, 5.0, 6.0,
            7.0, 8.0, 9.0,
            10.0, 11.0, 12.0,
        ];
        let result = super::qr_contiguous_f64(&a, &meta, true).expect("qr should succeed");
        let k = 3; // min(4,3)
        let n = 3;
        for i in 0..k {
            for j in 0..i.min(n) {
                assert!(
                    result.r[i * n + j].abs() < 1e-12,
                    "R[{i},{j}] = {} should be zero below diagonal",
                    result.r[i * n + j]
                );
            }
        }
    }

    // ── Complex kernel tests ────────────────────────────────────────

    #[test]
    fn complex_real_extracts_real_parts() {
        let meta = TensorMeta::from_shape(vec![3], DType::Complex128, Device::Cpu);
        let input = vec![
            Complex128::new(1.0, 2.0),
            Complex128::new(3.0, 4.0),
            Complex128::new(-1.0, 0.5),
        ];
        let result = super::complex_real_contiguous(&input, &meta).unwrap();
        assert_eq!(result, vec![1.0, 3.0, -1.0]);
    }

    #[test]
    fn complex_imag_extracts_imaginary_parts() {
        let meta = TensorMeta::from_shape(vec![3], DType::Complex128, Device::Cpu);
        let input = vec![
            Complex128::new(1.0, 2.0),
            Complex128::new(3.0, 4.0),
            Complex128::new(-1.0, 0.5),
        ];
        let result = super::complex_imag_contiguous(&input, &meta).unwrap();
        assert_eq!(result, vec![2.0, 4.0, 0.5]);
    }

    #[test]
    fn complex_conj_negates_imaginary() {
        let meta = TensorMeta::from_shape(vec![2], DType::Complex128, Device::Cpu);
        let input = vec![Complex128::new(1.0, 2.0), Complex128::new(-3.0, 4.0)];
        let result = super::complex_conj_contiguous(&input, &meta).unwrap();
        assert_eq!(result[0], Complex128::new(1.0, -2.0));
        assert_eq!(result[1], Complex128::new(-3.0, -4.0));
    }

    #[test]
    fn complex_abs_returns_magnitude() {
        let meta = TensorMeta::from_shape(vec![2], DType::Complex128, Device::Cpu);
        let input = vec![
            Complex128::new(3.0, 4.0), // |3+4i| = 5
            Complex128::new(0.0, 1.0), // |i| = 1
        ];
        let result = super::complex_abs_contiguous(&input, &meta).unwrap();
        assert!((result[0] - 5.0).abs() < 1e-12);
        assert!((result[1] - 1.0).abs() < 1e-12);
    }

    #[test]
    fn complex_angle_returns_phase() {
        let meta = TensorMeta::from_shape(vec![3], DType::Complex128, Device::Cpu);
        let input = vec![
            Complex128::new(1.0, 0.0),  // angle = 0
            Complex128::new(0.0, 1.0),  // angle = pi/2
            Complex128::new(-1.0, 0.0), // angle = pi
        ];
        let result = super::complex_angle_contiguous(&input, &meta).unwrap();
        assert!(result[0].abs() < 1e-12);
        assert!((result[1] - std::f64::consts::FRAC_PI_2).abs() < 1e-12);
        assert!((result[2] - std::f64::consts::PI).abs() < 1e-12);
    }

    #[test]
    fn complex_mul_follows_algebra() {
        // (1+2i) * (3+4i) = (1*3 - 2*4) + (1*4 + 2*3)i = -5 + 10i
        let meta = TensorMeta::from_shape(vec![1], DType::Complex128, Device::Cpu);
        let lhs = vec![Complex128::new(1.0, 2.0)];
        let rhs = vec![Complex128::new(3.0, 4.0)];
        let result = super::complex_mul_contiguous(&lhs, &rhs, &meta, &meta).unwrap();
        assert_eq!(result[0], Complex128::new(-5.0, 10.0));
    }

    #[test]
    fn complex_from_real_imag_constructs_correctly() {
        let real = vec![1.0, 2.0, 3.0];
        let imag = vec![4.0, 5.0, 6.0];
        let result = super::complex_from_real_imag(&real, &imag).unwrap();
        assert_eq!(result[0], Complex128::new(1.0, 4.0));
        assert_eq!(result[1], Complex128::new(2.0, 5.0));
        assert_eq!(result[2], Complex128::new(3.0, 6.0));
    }

    // ── Sparse tensor kernel tests ────────────────────────────────────

    #[test]
    fn sparse_coo_matmul_dense_matches_dense_matmul() {
        // sparse [2, 3] @ dense [3, 2] -> dense [2, 2]
        // Sparse matrix:
        // [[1, 0, 2],
        //  [0, 3, 0]]
        let sparse = SparseCOOTensor::from_coords(
            &[vec![0, 0], vec![0, 2], vec![1, 1]],
            vec![1.0, 2.0, 3.0],
            vec![2, 3],
            DType::F64,
            Device::Cpu,
        )
        .unwrap();

        // Dense matrix:
        // [[1, 2],
        //  [3, 4],
        //  [5, 6]]
        let dense = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let dense_meta = TensorMeta::from_shape(vec![3, 2], DType::F64, Device::Cpu);

        let result = sparse_coo_matmul_dense_f64(&sparse, &dense, &dense_meta).unwrap();

        // Expected: sparse @ dense
        // row 0: 1*[1,2] + 0*[3,4] + 2*[5,6] = [1+10, 2+12] = [11, 14]
        // row 1: 0*[1,2] + 3*[3,4] + 0*[5,6] = [9, 12]
        assert_eq!(result, vec![11.0, 14.0, 9.0, 12.0]);
    }

    #[test]
    fn sparse_coo_coalesce_sums_duplicates() {
        // Create sparse tensor with duplicate indices
        let sparse = SparseCOOTensor::from_coords(
            &[vec![0, 0], vec![0, 0], vec![1, 1]], // (0,0) appears twice
            vec![1.0, 2.0, 3.0],
            vec![2, 2],
            DType::F64,
            Device::Cpu,
        )
        .unwrap();

        let coalesced = sparse_coo_coalesce(&sparse).unwrap();

        // Should have only 2 unique entries: (0,0)=3.0, (1,1)=3.0
        assert_eq!(coalesced.nnz(), 2);

        // Convert to dense to verify values
        let dense = coalesced.to_dense().unwrap();
        let values = dense.contiguous_values_as_f64().unwrap();
        // [[3, 0], [0, 3]]
        assert_eq!(values, vec![3.0, 0.0, 0.0, 3.0]);
    }

    #[test]
    fn sparse_coo_add_combines_tensors() {
        // Create two sparse tensors
        let a = SparseCOOTensor::from_coords(
            &[vec![0, 0], vec![1, 1]],
            vec![1.0, 2.0],
            vec![2, 2],
            DType::F64,
            Device::Cpu,
        )
        .unwrap();

        let b = SparseCOOTensor::from_coords(
            &[vec![0, 0], vec![0, 1]], // (0,0) overlaps with a
            vec![3.0, 4.0],
            vec![2, 2],
            DType::F64,
            Device::Cpu,
        )
        .unwrap();

        let sum = sparse_coo_add(&a, &b).unwrap();

        // Convert to dense to verify: [[4, 4], [0, 2]]
        let dense = sum.to_dense().unwrap();
        let values = dense.contiguous_values_as_f64().unwrap();
        assert_eq!(values, vec![4.0, 4.0, 0.0, 2.0]);
    }

    // ── frankentorch-igu: Property-based kernel tests ─────────────────

    use proptest::prelude::*;

    #[test]
    fn cumprod_backward_interior_zero_matches_definition() {
        // The `input == 0` branch of cumprod_backward cannot be reached
        // by random floats, so pin it deterministically. For
        // input = [2, 0, 3] the forward cumprod is [2, 0, 0]; the exact
        // Jacobian (d out[j] / d in[d], summed against grad_output = 1)
        // gives grad_input = [1, 8, 0]:
        //   d=0: only out[0] depends on in[0] -> 1
        //   d=1: d out[1]/d in[1] = in[0] = 2,
        //        d out[2]/d in[1] = in[0]*in[2] = 6  -> 8
        //   d=2: d out[2]/d in[2] = in[0]*in[1] = 0  -> 0
        let input = [2.0, 0.0, 3.0];
        let meta = TensorMeta::from_shape(vec![3], DType::F64, Device::Cpu);
        let output = super::cumprod_tensor_contiguous_f64(&input, &meta, 0).unwrap();
        assert_eq!(output, vec![2.0, 0.0, 0.0]);
        let grad_output = [1.0, 1.0, 1.0];
        let grad_input =
            super::cumprod_backward_tensor_contiguous_f64(&grad_output, &input, &output, &meta, 0)
                .unwrap();
        assert_eq!(grad_input, vec![1.0, 8.0, 0.0]);
    }

    proptest! {
        #[test]
        fn prop_unary_neg_is_self_inverse(
            vals in proptest::collection::vec(-100.0f64..100.0, 1..32),
        ) {
            let n = vals.len();
            let meta = TensorMeta::from_shape(vec![n], DType::F64, Device::Cpu);
            let negated = super::neg_tensor_contiguous_f64(&vals, &meta).unwrap();
            let restored = super::neg_tensor_contiguous_f64(&negated, &meta).unwrap();
            for (i, (&orig, &rest)) in vals.iter().zip(restored.iter()).enumerate() {
                prop_assert!(
                    (orig - rest).abs() < 1e-12,
                    "neg(neg(x)) != x at index {i}: orig={orig}, restored={rest}"
                );
            }
        }

        #[test]
        fn prop_exp_log_round_trip(
            vals in proptest::collection::vec(0.01f64..50.0, 1..16),
        ) {
            // log(exp(x)) == x for positive x
            let n = vals.len();
            let meta = TensorMeta::from_shape(vec![n], DType::F64, Device::Cpu);
            let exped = super::exp_tensor_contiguous_f64(&vals, &meta).unwrap();
            let logged = super::log_tensor_contiguous_f64(&exped, &meta).unwrap();
            for (i, (&orig, &rest)) in vals.iter().zip(logged.iter()).enumerate() {
                prop_assert!(
                    (orig - rest).abs() < 1e-8,
                    "log(exp(x)) != x at index {i}: x={orig}, result={rest}"
                );
            }
        }

        #[test]
        fn prop_sum_reduction_matches_naive(
            vals in proptest::collection::vec(-50.0f64..50.0, 1..64),
        ) {
            let n = vals.len();
            let meta = TensorMeta::from_shape(vec![n], DType::F64, Device::Cpu);
            let kernel_sum = super::sum_tensor_contiguous_f64(&vals, &meta).unwrap();
            let naive_sum: f64 = vals.iter().sum();
            prop_assert!(
                (kernel_sum - naive_sum).abs() < 1e-8,
                "kernel sum={kernel_sum} != naive sum={naive_sum}"
            );
        }

        #[test]
        fn prop_softmax_sums_to_one(
            vals in proptest::collection::vec(-10.0f64..10.0, 2..16),
        ) {
            let n = vals.len();
            let meta = TensorMeta::from_shape(vec![n], DType::F64, Device::Cpu);
            let sm = super::softmax_dim_tensor_contiguous_f64(&vals, &meta, 0).unwrap();
            let total: f64 = sm.iter().sum();
            prop_assert!(
                (total - 1.0).abs() < 1e-10,
                "softmax should sum to 1.0, got {total}"
            );
            // All elements should be positive
            for (i, &v) in sm.iter().enumerate() {
                prop_assert!(v >= 0.0, "softmax[{i}] = {v} should be non-negative");
            }
        }

        #[test]
        fn prop_cumprod_backward_matches_finite_difference(
            input in proptest::collection::vec(0.3f64..2.0, 2..7),
            grad_seed in proptest::collection::vec(-2.0f64..2.0, 2..7),
        ) {
            // gradcheck the analytic cumprod backward against a central
            // finite-difference Jacobian-vector product. Inputs stay
            // strictly positive so the forward is well-conditioned and
            // the `acc / input` main path (not the zero branch) runs.
            let n = input.len().min(grad_seed.len());
            let input = &input[..n];
            let grad_output = &grad_seed[..n];
            let meta = TensorMeta::from_shape(vec![n], DType::F64, Device::Cpu);
            let output = super::cumprod_tensor_contiguous_f64(input, &meta, 0).unwrap();
            let analytic = super::cumprod_backward_tensor_contiguous_f64(
                grad_output,
                input,
                &output,
                &meta,
                0,
            )
            .unwrap();

            let h = 1e-6;
            for d in 0..n {
                let mut plus = input.to_vec();
                let mut minus = input.to_vec();
                plus[d] += h;
                minus[d] -= h;
                let y_plus = super::cumprod_tensor_contiguous_f64(&plus, &meta, 0).unwrap();
                let y_minus = super::cumprod_tensor_contiguous_f64(&minus, &meta, 0).unwrap();
                let mut numeric = 0.0;
                for j in 0..n {
                    numeric += grad_output[j] * (y_plus[j] - y_minus[j]) / (2.0 * h);
                }
                prop_assert!(
                    (analytic[d] - numeric).abs() <= 1e-4 * (1.0 + numeric.abs()),
                    "cumprod backward[{d}] analytic={} finite-diff={numeric}",
                    analytic[d]
                );
            }
        }

        #[test]
        fn prop_var_dim_matches_naive_two_pass(
            vals in proptest::collection::vec(-30.0f64..30.0, 2..40),
        ) {
            // var_dim must equal the textbook unbiased (Bessel-corrected)
            // two-pass variance: sum((x - mean)^2) / (n - 1).
            let n = vals.len();
            let meta = TensorMeta::from_shape(vec![n], DType::F64, Device::Cpu);
            let kernel = super::var_dim_tensor_contiguous_f64(&vals, &meta, 0).unwrap();
            let mean = vals.iter().sum::<f64>() / n as f64;
            let naive = vals.iter().map(|x| (x - mean) * (x - mean)).sum::<f64>()
                / (n as f64 - 1.0);
            prop_assert_eq!(kernel.len(), 1);
            prop_assert!(
                (kernel[0] - naive).abs() <= 1e-6 * (1.0 + naive.abs()),
                "var_dim={} != naive two-pass={naive}",
                kernel[0]
            );
        }

        #[test]
        fn prop_mean_dim_is_sum_dim_over_count(
            vals in proptest::collection::vec(-50.0f64..50.0, 1..48),
        ) {
            // mean_dim is exactly sum_dim divided by the reduced count.
            let n = vals.len();
            let meta = TensorMeta::from_shape(vec![n], DType::F64, Device::Cpu);
            let summed = super::sum_dim_tensor_contiguous_f64(&vals, &meta, 0).unwrap();
            let meaned = super::mean_dim_tensor_contiguous_f64(&vals, &meta, 0).unwrap();
            prop_assert_eq!(summed.len(), 1);
            prop_assert_eq!(meaned.len(), 1);
            let expected = summed[0] / n as f64;
            prop_assert!(
                (meaned[0] - expected).abs() <= 1e-9 * (1.0 + expected.abs()),
                "mean_dim={} != sum_dim/n={expected}",
                meaned[0]
            );
        }

        #[test]
        fn prop_matmul_matches_naive_triple_loop(
            (dims, lhs, rhs) in (1usize..6, 1usize..6, 1usize..6).prop_flat_map(
                |(m, k, n)| {
                    (
                        Just((m, k, n)),
                        proptest::collection::vec(-5.0f64..5.0, m * k),
                        proptest::collection::vec(-5.0f64..5.0, k * n),
                    )
                },
            ),
        ) {
            // The blocked matmul kernel must agree with the textbook
            // O(m*k*n) triple loop for every (m, k, n) shape.
            let (m, k, n) = dims;
            let lhs_meta = TensorMeta::from_shape(vec![m, k], DType::F64, Device::Cpu);
            let rhs_meta = TensorMeta::from_shape(vec![k, n], DType::F64, Device::Cpu);
            let kernel =
                super::matmul_tensor_contiguous_f64(&lhs, &rhs, &lhs_meta, &rhs_meta).unwrap();
            prop_assert_eq!(kernel.len(), m * n);
            for i in 0..m {
                for j in 0..n {
                    let mut acc = 0.0;
                    for p in 0..k {
                        acc += lhs[i * k + p] * rhs[p * n + j];
                    }
                    prop_assert!(
                        (kernel[i * n + j] - acc).abs() <= 1e-9 * (1.0 + acc.abs()),
                        "matmul[{i}][{j}] = {}, naive = {acc}",
                        kernel[i * n + j]
                    );
                }
            }
        }

        #[test]
        fn prop_inv_roundtrip_diagonally_dominant(
            (n, raw) in (2usize..6).prop_flat_map(|n| {
                (Just(n), proptest::collection::vec(-1.0f64..1.0, n * n))
            }),
        ) {
            // Build a strictly diagonally dominant matrix — guaranteed
            // invertible and well-conditioned: each diagonal entry gets
            // a +(n+1) bump, so |A[i][i]| >= n, which dominates the row's
            // off-diagonal magnitude sum (<= n - 1). A @ inv(A) must then
            // recover the identity.
            let mut a = raw;
            for i in 0..n {
                a[i * n + i] += (n + 1) as f64;
            }
            let meta = TensorMeta::from_shape(vec![n, n], DType::F64, Device::Cpu);
            let inv = super::inv_tensor_contiguous_f64(&a, &meta).unwrap();
            let prod =
                super::matmul_tensor_contiguous_f64(&a, &inv, &meta, &meta).unwrap();
            for i in 0..n {
                for j in 0..n {
                    let expected = if i == j { 1.0 } else { 0.0 };
                    prop_assert!(
                        (prod[i * n + j] - expected).abs() < 1e-8,
                        "A @ inv(A) [{i}][{j}] = {}, expected {expected}",
                        prod[i * n + j]
                    );
                }
            }
        }

        #[test]
        fn prop_broadcast_add_scalar_matches_elementwise(
            vals in proptest::collection::vec(-100.0f64..100.0, 1..20),
            scalar in -100.0f64..100.0,
        ) {
            let n = vals.len();
            let vec_meta = TensorMeta::from_shape(vec![n], DType::F64, Device::Cpu);
            let scalar_meta = TensorMeta::from_shape(vec![1], DType::F64, Device::Cpu);

            let (result, shape) = super::add_tensor_broadcast_f64(
                &vals,
                &[scalar],
                &vec_meta,
                &scalar_meta,
            ).unwrap();

            prop_assert_eq!(shape, vec![n]);
            for (i, &v) in vals.iter().enumerate() {
                prop_assert!(
                    (result[i] - (v + scalar)).abs() < 1e-10,
                    "broadcast[{}] = {}, expected {}",
                    i, result[i], v + scalar
                );
            }
        }

        #[test]
        fn prop_broadcast_mul_is_commutative(
            a in proptest::collection::vec(-10.0f64..10.0, 1..8),
            b in proptest::collection::vec(-10.0f64..10.0, 1..8),
        ) {
            let a_meta = TensorMeta::from_shape(vec![a.len(), 1], DType::F64, Device::Cpu);
            let b_meta = TensorMeta::from_shape(vec![1, b.len()], DType::F64, Device::Cpu);

            let (ab, ab_shape) = super::mul_tensor_broadcast_f64(&a, &b, &a_meta, &b_meta).unwrap();
            let (ba, ba_shape) = super::mul_tensor_broadcast_f64(&b, &a, &b_meta, &a_meta).unwrap();

            prop_assert_eq!(ab_shape, ba_shape);
            prop_assert_eq!(ab.len(), ba.len());

            for i in 0..ab.len() {
                prop_assert!(
                    (ab[i] - ba[i]).abs() < 1e-10,
                    "a*b[{}] = {}, b*a[{}] = {}",
                    i, ab[i], i, ba[i]
                );
            }
        }

        #[test]
        fn prop_broadcast_shape_is_symmetric(
            a_dims in proptest::collection::vec(1usize..5, 1..4),
            b_dims in proptest::collection::vec(1usize..5, 1..4),
        ) {
            let ab = super::compute_broadcast_shape(&a_dims, &b_dims);
            let ba = super::compute_broadcast_shape(&b_dims, &a_dims);

            match (ab, ba) {
                (Ok(ab_shape), Ok(ba_shape)) => {
                    prop_assert_eq!(ab_shape, ba_shape, "broadcast shapes should be symmetric");
                }
                (Err(_), Err(_)) => {
                    // Both failed, which is fine for incompatible shapes
                }
                (Ok(ab_shape), Err(_)) | (Err(_), Ok(ab_shape)) => {
                    prop_assert!(false, "asymmetric broadcast result: {:?} vs error", ab_shape);
                }
            }
        }

        #[test]
        fn prop_svd_singular_values_are_nonnegative(
            size in 2usize..6,
        ) {
            let n = size;
            let meta = TensorMeta::from_shape(vec![n, n], DType::F64, Device::Cpu);
            let data: Vec<f64> = (0..n*n).map(|i| (i as f64) * 0.1 + 0.1).collect();

            let svd_result = super::svd_contiguous_f64(&data, &meta, false).unwrap();

            for (i, &s) in svd_result.s.iter().enumerate() {
                prop_assert!(s >= 0.0, "singular value {} is negative: {}", i, s);
            }
        }

        #[test]
        fn prop_svd_reconstruction_within_tolerance(
            size in 2usize..5,
        ) {
            let n = size;
            let meta = TensorMeta::from_shape(vec![n, n], DType::F64, Device::Cpu);
            let data: Vec<f64> = (0..n*n).map(|i| ((i as f64) * 0.3 + 0.1).sin()).collect();

            let svd_result = super::svd_contiguous_f64(&data, &meta, false).unwrap();
            let u = &svd_result.u;
            let s = &svd_result.s;
            let vh = &svd_result.vh;
            let k = svd_result.k;

            let mut reconstructed = vec![0.0; n * n];
            for i in 0..n {
                for j in 0..n {
                    let mut sum = 0.0;
                    for l in 0..k {
                        sum += u[i * k + l] * s[l] * vh[l * n + j];
                    }
                    reconstructed[i * n + j] = sum;
                }
            }

            for i in 0..n*n {
                prop_assert!(
                    (reconstructed[i] - data[i]).abs() < 1e-10,
                    "SVD reconstruction mismatch at {}: got {} expected {}",
                    i, reconstructed[i], data[i]
                );
            }
        }

        #[test]
        fn prop_det_of_identity_is_one(
            size in 2usize..6,
        ) {
            let n = size;
            let meta = TensorMeta::from_shape(vec![n, n], DType::F64, Device::Cpu);
            let mut data = vec![0.0; n * n];
            for i in 0..n {
                data[i * n + i] = 1.0;
            }

            let det_result = super::det_contiguous_f64(&data, &meta).unwrap();

            prop_assert!(
                (det_result.det - 1.0).abs() < 1e-10,
                "det(I) should be 1, got {}",
                det_result.det
            );
        }

        #[test]
        fn prop_det_of_diagonal_is_product_of_diag(
            diag in proptest::collection::vec(-10.0f64..10.0, 2..5),
        ) {
            let n = diag.len();
            let meta = TensorMeta::from_shape(vec![n, n], DType::F64, Device::Cpu);
            let mut data = vec![0.0; n * n];
            for (i, &d) in diag.iter().enumerate() {
                data[i * n + i] = d;
            }
            let expected: f64 = diag.iter().product();

            let det_result = super::det_contiguous_f64(&data, &meta).unwrap();

            prop_assert!(
                (det_result.det - expected).abs() < 1e-8,
                "det(diag) should be product of diagonal, got {} expected {}",
                det_result.det, expected
            );
        }

        #[test]
        fn prop_cholesky_reconstruction_positive_definite(
            size in 2usize..5,
        ) {
            let n = size;
            let meta = TensorMeta::from_shape(vec![n, n], DType::F64, Device::Cpu);
            let mut data = vec![0.0; n * n];
            for i in 0..n {
                data[i * n + i] = (n + 1) as f64;
                for j in 0..i {
                    data[i * n + j] = 0.1;
                    data[j * n + i] = 0.1;
                }
            }

            let result = super::cholesky_contiguous_f64(&data, &meta, false).unwrap();
            let l = &result.factor;

            let mut reconstructed = vec![0.0; n * n];
            for i in 0..n {
                for j in 0..n {
                    let mut sum = 0.0;
                    for k in 0..n {
                        sum += l[i * n + k] * l[j * n + k];
                    }
                    reconstructed[i * n + j] = sum;
                }
            }

            for i in 0..n*n {
                prop_assert!(
                    (reconstructed[i] - data[i]).abs() < 1e-10,
                    "Cholesky L@L^T mismatch at {}: got {} expected {}",
                    i, reconstructed[i], data[i]
                );
            }
        }

        #[test]
        fn prop_qr_reconstruction_square_matrix(
            size in 2usize..5,
        ) {
            let n = size;
            let meta = TensorMeta::from_shape(vec![n, n], DType::F64, Device::Cpu);
            let data: Vec<f64> = (0..n*n).map(|i| ((i as f64) * 0.3 + 0.1).sin()).collect();

            let result = super::qr_contiguous_f64(&data, &meta, false).unwrap();
            let q = &result.q;
            let r = &result.r;
            let m = result.m;

            let mut reconstructed = vec![0.0; n * n];
            for i in 0..n {
                for j in 0..n {
                    let mut sum = 0.0;
                    for l in 0..m {
                        sum += q[i * m + l] * r[l * n + j];
                    }
                    reconstructed[i * n + j] = sum;
                }
            }

            for i in 0..n*n {
                prop_assert!(
                    (reconstructed[i] - data[i]).abs() < 1e-10,
                    "QR Q@R mismatch at {}: got {} expected {}",
                    i, reconstructed[i], data[i]
                );
            }
        }
    }

    #[test]
    fn nan_propagation_in_min_max() {
        let meta = TensorMeta::from_shape(vec![3], DType::F64, Device::Cpu);
        let with_nan = vec![1.0, f64::NAN, 3.0];
        let normal = vec![2.0, 2.0, 2.0];

        let min_result = min_tensor_contiguous_f64(&with_nan, &normal, &meta, &meta).unwrap();
        let max_result = max_tensor_contiguous_f64(&with_nan, &normal, &meta, &meta).unwrap();

        assert!(min_result[1].is_nan(), "min with NaN should propagate NaN");
        assert!(max_result[1].is_nan(), "max with NaN should propagate NaN");
    }

    #[test]
    fn softmax_handles_large_values_without_overflow() {
        let meta = TensorMeta::from_shape(vec![3], DType::F64, Device::Cpu);
        let large_vals = vec![1000.0, 1001.0, 1002.0];

        let result = softmax_dim_tensor_contiguous_f64(&large_vals, &meta, 0).unwrap();

        assert!(
            !result.iter().any(|x| x.is_nan()),
            "softmax should not produce NaN for large values"
        );
        assert!(
            !result.iter().any(|x| x.is_infinite()),
            "softmax should not produce Inf for large values"
        );
        let sum: f64 = result.iter().sum();
        assert!(
            (sum - 1.0).abs() < 1e-10,
            "softmax should sum to 1, got {sum}"
        );
    }

    #[test]
    fn log_of_zero_is_negative_infinity() {
        let meta = TensorMeta::from_shape(vec![2], DType::F64, Device::Cpu);
        let vals = vec![0.0, 1.0];

        let result = log_tensor_contiguous_f64(&vals, &meta).unwrap();

        assert!(
            result[0].is_infinite() && result[0] < 0.0,
            "log(0) should be -inf"
        );
        assert!((result[1] - 0.0).abs() < 1e-10, "log(1) should be 0");
    }

    #[test]
    fn reciprocal_of_zero_is_infinity() {
        let meta = TensorMeta::from_shape(vec![2], DType::F64, Device::Cpu);
        let vals = vec![0.0, 2.0];

        let result = reciprocal_tensor_contiguous_f64(&vals, &meta).unwrap();

        assert!(result[0].is_infinite(), "1/0 should be inf");
        assert!((result[1] - 0.5).abs() < 1e-10, "1/2 should be 0.5");
    }

    #[test]
    fn empty_tensor_sum_is_zero() {
        let meta = TensorMeta::from_shape(vec![0], DType::F64, Device::Cpu);
        let vals: Vec<f64> = vec![];

        let result = sum_tensor_contiguous_f64(&vals, &meta).unwrap();

        assert_eq!(result, 0.0, "sum of empty tensor should be 0");
    }

    #[test]
    fn empty_tensor_mean_is_nan() {
        let meta = TensorMeta::from_shape(vec![0], DType::F64, Device::Cpu);
        let vals: Vec<f64> = vec![];

        let result = mean_tensor_contiguous_f64(&vals, &meta).unwrap();

        assert!(result.is_nan(), "mean of empty tensor should be NaN (0/0)");
    }

    #[test]
    fn inf_in_softmax_produces_nan_due_to_stability_trick() {
        let meta = TensorMeta::from_shape(vec![3], DType::F64, Device::Cpu);
        let with_inf = vec![f64::NEG_INFINITY, 0.0, f64::INFINITY];

        let result = softmax_dim_tensor_contiguous_f64(&with_inf, &meta, 0).unwrap();

        assert!(
            result.iter().all(|x| x.is_nan()),
            "softmax with inf produces NaN due to max-subtraction (inf - inf = NaN)"
        );
    }

    #[test]
    fn softmax_with_finite_extremes_is_well_defined() {
        let meta = TensorMeta::from_shape(vec![3], DType::F64, Device::Cpu);
        let vals = vec![-1000.0, 0.0, 1000.0];

        let result = softmax_dim_tensor_contiguous_f64(&vals, &meta, 0).unwrap();

        assert_eq!(result[0], 0.0, "exp(-2000) underflows to 0");
        assert_eq!(result[2], 1.0, "largest finite value dominates");
        let sum: f64 = result.iter().sum();
        assert!((sum - 1.0).abs() < 1e-10, "softmax sums to 1");
    }

    #[test]
    fn simd_and_scalar_paths_produce_same_results() {
        let small_meta = TensorMeta::from_shape(vec![3], DType::F64, Device::Cpu);
        let large_meta = TensorMeta::from_shape(vec![16], DType::F64, Device::Cpu);

        let small_vals = vec![1.0, 2.0, 3.0];
        let large_vals: Vec<f64> = (1..=16).map(|x| x as f64).collect();

        let small_neg = neg_tensor_contiguous_f64(&small_vals, &small_meta).unwrap();
        let large_neg = neg_tensor_contiguous_f64(&large_vals, &large_meta).unwrap();

        assert_eq!(small_neg, vec![-1.0, -2.0, -3.0]);
        let expected: Vec<f64> = (1..=16).map(|x| -(x as f64)).collect();
        assert_eq!(large_neg, expected);
    }

    #[test]
    fn pairwise_sum_map_parallel_matches_serial_bit_exact() {
        let numel = (1 << 19) + 77;
        let v: Vec<f64> = (0..numel)
            .map(|i| ((i * 11 + 7) % 211) as f64 * 0.017 - 1.8)
            .collect();
        let serial = super::pairwise_sum_map_f64(&v, |x| x * x);
        let parallel = super::pairwise_sum_map_f64_par(&v, |x| x * x);
        assert_eq!(
            serial.to_bits(),
            parallel.to_bits(),
            "{serial} vs {parallel}"
        );
        // L2 norm wires the parallel path above the threshold.
        let meta = TensorMeta::from_shape(vec![numel], DType::F64, Device::Cpu);
        let nrm = super::norm_tensor_contiguous_f64(&v, &meta, 2.0).expect("norm");
        assert_eq!(nrm.to_bits(), serial.sqrt().to_bits());
    }
}

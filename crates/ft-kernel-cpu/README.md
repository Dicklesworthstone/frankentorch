# frankentorch-kernel-cpu

CPU kernel implementations for [FrankenTorch](https://github.com/Dicklesworthstone/frankentorch),
a deterministic tensor computation and autograd engine written in memory-safe Rust
(`#![forbid(unsafe_code)]`) that reimplements core PyTorch semantics.

This crate contains the portable and SIMD-accelerated compute kernels the engine
dispatches to on CPU:

- Elementwise, reduction, and broadcasting kernels
- GEMM / matmul paths (via `matrixmultiply` plus tuned tiling), Winograd convolution,
  softmax, scan, sort/top-k, and linear algebra routines
- SIMD acceleration through the safe `wide` crate and data parallelism via `rayon`

Kernels are written for deterministic, PyTorch-parity numerics; correctness outranks
speed (validated against a PyTorch oracle in the main repository's conformance suite).

The Rust library name is `ft_kernel_cpu` (`use ft_kernel_cpu::...`), while the package
name on crates.io is `frankentorch-kernel-cpu`. It builds on
[`frankentorch-core`](https://crates.io/crates/frankentorch-core).

See the main repository for the full engine (dispatch, autograd, API):
https://github.com/Dicklesworthstone/frankentorch

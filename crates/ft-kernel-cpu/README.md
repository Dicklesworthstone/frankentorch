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

## Rust toolchains and SIMD features

Default builds support **stable Rust 1.95 or newer**. SIMD acceleration through
`wide` and the existing x86_64 AVX2/AVX-512-VNNI and aarch64 NEON implementations
remain available, with runtime CPU-feature detection where required. Stable
builds use `wide::f64x4` for the vectorized 2x2 average-pooling backward path,
including scalar tails and the same signed-zero behavior as the scalar kernel.

The optional `nightly-simd` Cargo feature enables the `std::simd` implementation
of that pooling path and requires a nightly compiler. It is disabled by default
and does not control the architecture-specific intrinsic kernels.

From a repository checkout, select the toolchain explicitly because the
repository's development toolchain is pinned to nightly:

```bash
cargo +stable test --locked -j2 -p frankentorch-kernel-cpu
cargo +nightly test --locked -j2 -p frankentorch-kernel-cpu --features nightly-simd
```

Use nightly with `--all-features`, which also enables `nightly-simd`.

See the main repository for the full engine (dispatch, autograd, API):
https://github.com/Dicklesworthstone/frankentorch

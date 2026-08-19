# frankentorch-core

Zero-dependency-core tensor foundation types for [FrankenTorch](https://github.com/Dicklesworthstone/frankentorch),
a deterministic tensor computation and autograd engine written in memory-safe Rust
(`#![forbid(unsafe_code)]`) that reimplements core PyTorch semantics.

This crate provides the shared vocabulary the rest of the FrankenTorch stack builds on:

- `DType` definitions and dtype promotion/conversion rules (including f16/bf16 via `half` and complex types via `num-complex`)
- `Shape` and `Stride` arithmetic, broadcasting, and contiguity checks
- `TensorMeta` descriptors used by the dispatcher and kernel crates
- Scalar semantics and the common error types used across the workspace

The Rust library name is `ft_core` (`use ft_core::...`), while the package name on
crates.io is `frankentorch-core`.

See the main repository for the full engine (dispatch, CPU kernels, autograd, API):
https://github.com/Dicklesworthstone/frankentorch

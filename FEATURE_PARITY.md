# FEATURE_PARITY

## Absolute Parity Doctrine

This matrix tracks execution progress, not allowable scope reduction. Release readiness requires complete feature/functionality overlap with upstream PyTorch for intended drop-in replacement operation.

## Status Legend

- `not_started`
- `in_progress`
- `parity_green`
- `parity_gap`

## Parity Matrix

| Feature Family | Status | Notes |
|---|---|---|
| Tensor core semantics | parity_green | DenseTensor with contiguous f64 storage, TensorMeta (shape/strides/offset), in-place mutation with version tracking |
| Dispatch key routing | parity_green | `FT-P2C-002` keyset model + strict/hardened mode split conformance |
| Autograd correctness | parity_green | `FT-P2C-004` dependency-driven scheduler + deterministic replay telemetry; full tensor backward |
| CPU kernel semantics | parity_green | Full unary/binary/reduction/comparison/shape/matmul kernels with broadcasting |
| Checkpoint compatibility | parity_green | `FT-P2C-006` typed checkpoint + fail-closed decode + RaptorQ sidecars |
| Broadcasting | parity_green | NumPy-style broadcasting for all binary tensor ops with gradient reduction |
| Random operations | parity_green | Deterministic xoshiro256++ PRNG: rand, randn, rand_like, randn_like |
| In-place operations | parity_green | add_, sub_, mul_, div_, zero_, fill_, mul_scalar_, add_scalar_ with version tracking |
| Loss functions | parity_green | mse_loss, l1_loss, bce_loss, smooth_l1_loss (composed from autograd ops) |
| Shape operations | parity_green | reshape, squeeze, unsqueeze, transpose, permute, cat, stack, flatten, unflatten, narrow, expand, split, chunk |
| Advanced reductions | parity_green | argmax, argmin, max_dim, min_dim (with backward), softmax, log_softmax, sum_dim, mean_dim, prod_dim, var_dim, std_dim |
| Neural network modules (ft-nn) | parity_green | Module trait, Linear (Kaiming init), ReLU, Sigmoid, Tanh, GELU, SiLU, Sequential, Dropout |
| Optimizers (ft-optim) | parity_green | Optimizer trait, SGD (momentum, weight_decay, nesterov), Adam (bias correction, weight_decay) |
| Advanced indexing | parity_green | index_select, gather, scatter, masked_fill (with backward for index_select, gather) |
| Full PyTorch drop-in surface | in_progress | aggregate parity-closure tracker; no intentional feature omissions permitted at release |

## Detailed Operation Coverage

### Unary Ops (30+)
neg, abs, exp, log, relu, sigmoid, tanh, sin, cos, tan, floor, ceil, round, log2, log10, log1p, expm1, sign, trunc, frac, asin, acos, atan, sinh, cosh, gelu, silu, leaky_relu, elu, sqrt, reciprocal, pow

### Binary Ops
add, sub, mul, div, min, max, clamp (all with broadcasting + backward)

### Comparison Ops
eq, ne, lt, gt, le, ge (all with broadcasting, return 0.0/1.0 masks)

### Reductions
sum, mean (global); sum_dim, mean_dim, prod_dim, var_dim, std_dim (per-dim); argmax, argmin, max_dim, min_dim (per-dim with indices)

### Normalization
softmax, log_softmax (per-dim, numerically stable)

### Shape Operations
reshape, view, squeeze, unsqueeze, transpose, permute, cat, stack, flatten, unflatten, narrow, expand, split, chunk

### Matrix Operations
matmul (with backward)

### Random Operations
rand (uniform [0,1)), randn (normal), rand_like, randn_like (deterministic xoshiro256++ PRNG)

### In-Place Operations
tensor_add_, tensor_sub_, tensor_mul_, tensor_div_, tensor_zero_, tensor_fill_, tensor_mul_scalar_, tensor_add_scalar_

### Loss Functions
mse_loss, l1_loss, bce_loss, smooth_l1_loss

### Advanced Indexing
index_select, gather, scatter, masked_fill

## Current Green Scope

- `crates/ft-conformance/fixtures/scalar_autograd_cases.json`
- `crates/ft-conformance/fixtures/dispatch_key_cases.json`
- `crates/ft-conformance/fixtures/autograd_scheduler_cases.json`
- `crates/ft-conformance/fixtures/serialization_cases.json`

Modes tested for all listed families: strict + hardened.
804 tests passing across workspace.

## Gap Policy

- Any `parity_gap` item must have explicit closure beads and dependencies.
- Temporary sequencing gaps are acceptable only with closure evidence plans.
- Release sign-off requires `Full PyTorch drop-in surface` to be `parity_green`.

## Required Evidence Per Feature Family

1. Differential fixture report.
2. Edge-case/adversarial test results.
3. Benchmark delta (when performance-sensitive).
4. Documented compatibility exceptions (if any).
5. RaptorQ sidecar + decode-proof chain for durable parity bundles.

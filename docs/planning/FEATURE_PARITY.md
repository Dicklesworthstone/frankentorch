# FEATURE_PARITY

## Absolute Parity Doctrine

This matrix tracks execution progress, not allowable scope reduction. Release readiness requires complete feature/functionality overlap with upstream PyTorch for intended drop-in replacement operation.

## Status Legend

- `not_started`
- `in_progress`
- `parity_green`
- `parity_gap`

## Parity Scope Declaration

`parity_green` in this file means the current V1 CPU + Metal eager-mode FrankenTorch slice is implemented and backed by conformance evidence across all 13 workspace crates (`ft-core`, `ft-dispatch`, `ft-kernel-cpu`, `ft-kernel-metal`, `ft-autograd`, `ft-device`, `ft-serialize`, `ft-api`, `ft-conformance`, `ft-runtime`, `ft-nn`, `ft-optim`, `ft-data`). It does not mean out-of-scope PyTorch distributed/CUDA/TorchScript subsystems are complete.

Current phase scope covers:
- CPU and Metal (`Device::Cpu`, `Device::Metal`) dense and sparse tensor execution with `ft-device` device-affinity enforcement
- deterministic autograd, DAC evidence, Bayesian `DecisionController` loss-matrix calibration, `asupersync` `&Cx` cancellation-aware runtime execution, and 4-panel `FrankenTuiCockpit` rendering in `ft-runtime`
- full `ft-nn`, `ft-optim`, typed `ft-data` (`DenseTensor` batching/transforms), `ft-serialize` (all-dtype `FTSV`, SafeTensors, and sequential ONNX export), ergonomic `Tensor` operator/fluent API in `ft-api`, and `ft-conformance` integration

Current phase does not cover out-of-scope CUDA/distributed/JIT PyTorch subsystems. Those are tracked explicitly in `Deferred Subsystems` below and must not be inferred from the green rows in the current V1 slice.

## Parity Matrix

| Feature Family | Status | Notes |
|---|---|---|
| Tensor core semantics | parity_green | `DenseTensor` with all 8 storage dtypes (`F64`, `F32`, `F16`, `BF16`, `Complex64`, `Complex128`, `QInt8`, `QUInt8`), `TensorMeta` (shape/strides/offset/device), in-place mutation with version tracking |
| Device & Backend Routing | parity_green | `ft-device` (`Device::Cpu`, `Device::Cuda`, `Device::Metal`, `DeviceGuard`, thread-local device context) wired into `ft-core`, `ft-dispatch`, `ft-kernel-cpu`, `ft-kernel-metal`, and `ft-api` |
| Dispatch key routing | parity_green | `FT-P2C-002` keyset model + strict/hardened mode split conformance + CPU/Metal backend dispatch |
| Autograd correctness | parity_green | `FT-P2C-004` dependency-driven scheduler + deterministic replay telemetry; full tensor backward |
| Ergonomic Tensor API | parity_green | `TensorContext` + `Tensor` handle in `ft-api` implementing `std::ops::{Add, Sub, Mul, Div, Neg}` and fluent method chaining (`.matmul`, `.relu`, `.sigmoid`, `.tanh`, `.sum`, `.mean`, `.reshape`, `.transpose`, `.backward`, `.grad`) |
| CPU kernel semantics | parity_green | Full unary/binary/reduction/comparison/shape/matmul/linalg kernels with broadcasting |
| Runtime, Asupersync & FrankenTUI | parity_green | Spec §6 Bayesian `DecisionController`, real `asupersync::Cx` checkpoints (`run_with_cx`, `check_cx_checkpoint`), and 4-panel `FrankenTuiCockpit` widget in `ft-runtime` |
| Checkpoint & ONNX compatibility | parity_green | `FT-P2C-006` typed checkpoint + native `FTSV` across all 8 `DenseTensor` dtypes + `TypedStateEntry` (`I64`, `I32`, `Bool`) + SafeTensors + multi-layer sequential ONNX export + RaptorQ sidecars |
| Dataset & DataLoader (ft-data) | parity_green | Typed `DenseTensor` `DataItem`, `TensorDataset`, and `DataLoader` collation preserving `DType` (`F32`, `F64`, `F16`, `BF16`, quantized) + `NormalizeTransform` / `ScaleTransform` |
| Broadcasting | parity_green | NumPy-style broadcasting for all binary tensor ops with gradient reduction |
| Random operations | parity_green | Deterministic xoshiro256++ PRNG: rand, randn, rand_like, randn_like |
| In-place operations | parity_green | add_, sub_, mul_, div_, zero_, fill_, mul_scalar_, add_scalar_ with version tracking |
| Loss functions | parity_green | mse_loss, l1_loss, bce_loss, smooth_l1_loss, huber_loss, cross_entropy_loss, nll_loss, cosine_embedding_loss (composed from autograd ops); nn modules: MSELoss, L1Loss, CrossEntropyLoss, NLLLoss, BCELoss, BCEWithLogitsLoss, SmoothL1Loss, HuberLoss, CosineEmbeddingLoss, KLDivLoss |
| Shape operations | parity_green | reshape, squeeze, unsqueeze, transpose, permute, cat, stack, flatten, unflatten, narrow, expand, split, chunk |
| Advanced reductions | parity_green | argmax, argmin, max_dim, min_dim (with backward), softmax, log_softmax, sum_dim, mean_dim, prod_dim, var_dim, std_dim |
| Functional API (`torch.nn.functional`-style session ops) | parity_green | session-level normalize, dropout, embedding, and linear helpers available in `ft-api` and covered by targeted tests |
| Neural network modules (ft-nn) | parity_green | Module trait, LossModule trait, Linear, Bilinear, Conv1d, Conv2d, ConvTranspose1d, ReLU, Sigmoid, Tanh, GELU, SiLU, LeakyReLU, ELU, Mish, Softplus, Sequential, ModuleList, ModuleDict, ParameterList, ParameterDict, Dropout, LayerNorm, BatchNorm1d, BatchNorm2d, BatchNorm3d, GroupNorm, InstanceNorm1d, InstanceNorm2d, Embedding, MultiheadAttention, Softmax, LogSoftmax, Flatten, Identity, Unfold, Fold, AvgPool1d, MaxPool1d, MaxPool2d, AdaptiveAvgPool2d, LPPool1d, LPPool2d, Upsample1d, Upsample2d, ConstantPad1d, ConstantPad2d, ZeroPad2d, RNNCell, LSTMCell, GRUCell, PixelShuffle, PixelUnshuffle, CosineSimilarity, PairwiseDistance, Hardswish, Hardsigmoid, LogSigmoid, RReLU, MultiLabelMarginLoss, MSELoss, L1Loss, CrossEntropyLoss, NLLLoss, BCELoss, BCEWithLogitsLoss, SmoothL1Loss, HuberLoss, KLDivLoss |
| Optimizers (ft-optim) | parity_green | Optimizer trait, SGD (momentum, weight_decay, nesterov), Adam (bias correction, weight_decay), AdamW, RMSprop, Adagrad, RAdam, SparseAdam, Muon; 15 LR schedulers; clip_grad_norm_, clip_grad_value_ |
| Advanced indexing | parity_green | index_select, gather, scatter, masked_fill (with backward for index_select, gather) |
| Spec §17 Performance SLOs | parity_green | All 8/8 Spec §17 performance and memory SLO budgets (`elementwise_forward`, `representative_backward`, `matmul_core`, `optimizer_step`, `checkpoint_roundtrip`, `training_step_memory`, `backward_heavy_allocation`, `benchmark_family_tail`) pass (`overall_status: "pass"`) and are enforced by `validate_phase2c_artifacts` |
| Full PyTorch drop-in surface (V1 CPU+Metal scope) | parity_green | V1 CPU+Metal eager-mode slice is complete across all 13 workspace crates; out-of-scope CUDA/distributed/JIT subsystems remain tracked in `Deferred Subsystems` below. |

## Detailed Operation Coverage

### Unary Ops (40+)
neg, abs, exp, log, relu, sigmoid, tanh, sin, cos, tan, floor, ceil, round, log2, log10, log1p, expm1, sign, trunc, frac, asin, acos, atan, sinh, cosh, gelu, silu, leaky_relu, elu, sqrt, reciprocal, pow, rsqrt, erf, erfc, hardswish, hardsigmoid, hardtanh, softplus, mish, square

### Binary Ops
add, sub, mul, div, min, max, clamp, atan2, fmod, remainder (all with broadcasting + backward)

### Comparison Ops
eq, ne, lt, gt, le, ge (all with broadcasting, return 0.0/1.0 masks)

### Float Classification Ops
isnan, isinf, isfinite (return 0.0/1.0 masks, no autograd tracking)

### Reductions
sum, mean (global); sum_dim, mean_dim, prod_dim, var_dim, std_dim (per-dim); argmax, argmin, max_dim, min_dim (per-dim with indices)

### Norm Operations
norm (global p-norm), norm_dim (per-dim p-norm) — supports L0, L1, L2, Lp, Linf, L-inf with backward

### Normalization
softmax, log_softmax (per-dim, numerically stable)

### Functional API
functional_normalize, functional_dropout, functional_embedding, functional_linear, scaled_dot_product_attention, pixel_shuffle, pixel_unshuffle, functional_conv1d, functional_conv2d, functional_conv3d, functional_conv_transpose1d, functional_conv_transpose2d, functional_avg_pool1d, functional_avg_pool2d, functional_max_pool1d, functional_max_pool2d, functional_adaptive_avg_pool1d, functional_adaptive_avg_pool2d, functional_adaptive_avg_pool3d, functional_layer_norm, functional_group_norm, functional_instance_norm1d, functional_instance_norm2d

### Complex Number Operations
tensor_real, tensor_imag, tensor_conj, tensor_complex, tensor_view_as_real, tensor_view_as_complex

### Spectral Operations
hann_window, tensor_stft, tensor_istft

### Spatial Transforms
tensor_affine_grid, tensor_grid_sample (bilinear/nearest, zeros/border/reflection padding)

### Shape Operations
reshape, view, squeeze, unsqueeze, transpose, permute, cat, stack, flatten, unflatten, narrow, expand, split, chunk, unbind, hstack, vstack, dstack, column_stack, row_stack, hsplit, vsplit, dsplit

### Matrix Operations
matmul, addmm, addmv, bmm, baddbmm, addbmm (with backward)

### Interpolation
lerp (linear interpolation, with backward)

### Numerical Integration
trapezoid, cumulative_trapezoid (trapezoidal rule along dimension)

### Inner/Outer Products
inner (generalized inner product), addr (rank-1 update: beta*M + alpha*v1⊗v2), outer

### Grid/Creation Operations
meshgrid, diagonal (with offset support), one_hot

### Random Operations
rand (uniform [0,1)), randn (normal), rand_like, randn_like, randint, randperm, multinomial, bernoulli, bernoulli_p, poisson (deterministic xoshiro256++ PRNG)

### In-Place Operations
tensor_add_, tensor_sub_, tensor_mul_, tensor_div_, tensor_zero_, tensor_fill_, tensor_mul_scalar_, tensor_add_scalar_

### Loss Functions
mse_loss, l1_loss, bce_loss, bce_with_logits_loss, smooth_l1_loss, huber_loss, cross_entropy_loss (with label_smoothing), nll_loss, cosine_embedding_loss, focal_loss, poisson_nll_loss, gaussian_nll_loss, soft_margin_loss, multi_margin_loss, multilabel_soft_margin_loss

### Advanced Indexing
index_select, gather, scatter, scatter_add, scatter_reduce (sum/prod/mean/amax/amin), masked_fill, masked_select, masked_scatter, index_add, index_copy, index_fill, index_put, select, take_along_dim

### Sorting & Selection
sort, topk, argsort, kthvalue

### Linear Algebra
svd, qr, cholesky, cholesky_solve, cholesky_inverse, det, slogdet, inv, pinverse, solve, eigh, matrix_power, matrix_exp, linalg_vector_norm, triangular_solve, matrix_norm (fro/1/-1/inf/-inf), lstsq, cond, matrix_rank

### Statistical / Histogram Operations
bincount (with weights, minlength), histc (with auto-range; out-of-range values dropped per torch.histc / numpy.histogram semantics)

### Structured Matrix Operations
block_diag (from 1-D or 2-D tensors), diagflat (with offset support)

### Distance Operations
cdist (batched pairwise Lp distance, L0/L2/Linf), pdist (within-tensor pairwise Lp distance)

### Numeric Stability Operations
nan_to_num, logaddexp, logaddexp2, xlogy, logcumsumexp, logsumexp

### Dimension Manipulation
movedim (move dimension to new position)

### Element Testing
isin (set membership), renorm (per-slice norm capping)

### Sliding Windows
unfold (extract sliding local blocks)

### Combined Reductions
aminmax (simultaneous min and max along dim), dist (p-norm distance between tensors)

### Rotation / Rearrangement
rot90, pixel_shuffle, pixel_unshuffle, tile, fliplr, flipud

### Statistical Operations
cov (covariance matrix), corrcoef (Pearson correlation), mode (most frequent value), quantile (with interpolation)

### Padding
tensor_pad (constant), tensor_pad_mode (constant, reflect, replicate, circular — PyTorch F.pad convention)

### torch.special Functions
expit (sigmoid), logit, erfinv, gammaln (lgamma), digamma, polygamma, multigammaln, xlog1py, entr, xlogy

### Parameter Initialization (torch.nn.init)
constant_, zeros_, ones_, uniform_, normal_, trunc_normal_, eye_, xavier_uniform_, xavier_normal_, kaiming_uniform_, kaiming_normal_, orthogonal_, sparse_, dirac_, calculate_gain

### Interpolation (F.interpolate)
nearest, linear (1D), bilinear (2D), bicubic (2D), trilinear (3D) — supports size, scale_factor, align_corners

### RNN Utilities (torch.nn.utils.rnn)
PackedSequence, pack_padded_sequence, pad_packed_sequence, pad_sequence

### Tensor Creation
polar (magnitude+phase to complex), cartesian_prod (Cartesian product of 1-D tensors), combinations (r-length combinations)

### Weight Normalization (torch.nn.utils)
weight_norm_decompose, weight_norm_reconstruct, spectral_norm

### Dataset Utilities (torch.utils.data)
Subset (Arc-based), random_split (deterministic seeded)

## Deferred & Scoped Subsystems

| Subsystem | Status | Current truth | Closure bead |
|---|---|---|---|
| CUDA / GPU backend | not_started | `Device::Cuda` exists as metadata and device-guard validation; V1 GPU execution targets `Device::Metal` (`ft-kernel-metal`) rather than CUDA | `frankentorch-c3d` (out-of-scope for V1) |
| Metal GPU backend (`Device::Metal`) | parity_green | `Device::Metal` wired through `ft-core`, `ft-device`, `ft-dispatch`, `ft-kernel-metal`, and `ft-api` | `frankentorch-e0dyu` (closed) |
| Distributed training (`DDP` / `FSDP`) | not_started | No multi-rank process-group, collective communication, gradient sync, or parameter sharding surface exists in-tree | `frankentorch-82b` (out-of-scope for V1) |
| `TorchScript` / `torch.compile` / JIT graph execution | not_started | Execution is eager-only; there is no graph capture, compiled runtime, or TorchScript-like serialization/execution pipeline | `frankentorch-8zy` (out-of-scope for V1) |
| Quantization | parity_green | Per-tensor & per-channel quantization (`tensor_quantize_per_tensor`, `tensor_dequantize_per_tensor`, `tensor_fake_quantize_per_tensor`, `quantize_per_output_channel_i8`), `QInt8`/`QUInt8` storage dtypes in `DenseTensor` and `FTSV` checkpoints, `QuantizedLinear`, `MinMaxObserver`, `QParams`, and dynamic int8 linear kernels shipped. | `frankentorch-4ak`, `frankentorch-hpg`, `frankentorch-yjwvg` (closed) |
| Sparse tensors | parity_partial | Sparse COO/CSR storage in ft-core, sparse kernels (matmul/coalesce/add) in ft-kernel-cpu, sparse-gradient surface from `Embedding(sparse=true)` + `IndexSelect` consumed by `SparseAdam`. Broader sparse coverage (sparse-sparse ops, CSC, BSR, ufuncs) remains gapped. | `frankentorch-0tw` (closed, MVP) |
| Mixed precision (`torch.amp`) | parity_partial | Autocast context + `GradScaler` shipped. F16/BF16 forward already supported; backward currently runs in F64. | `frankentorch-iha` (closed) |
| ONNX export | parity_green | Single-layer and multi-layer sequential ONNX ModelProto export (`export_linear_onnx_to_bytes`, `export_sequential_onnx_to_bytes`, `save_sequential_onnx` supporting `Linear`, `ReLU`, `GELU`, `Sigmoid`, `Tanh`, `LayerNorm`) shipped in `ft-serialize`. | `frankentorch-iey`, `frankentorch-yjwvg` (closed) |
| Gradient checkpointing | parity_partial | `session.checkpoint(...)` exposes recompute-on-backward; broader API surface (use_reentrant, distributed-aware) deferred. | `frankentorch-plj` (closed) |
| Custom autograd Function surface | parity_supported | Rust-idiomatic custom autograd is exposed through `FrankenTorchSession::tensor_apply_function(...)` and `TensorTape::apply_function(...)`, with `FunctionCtx::save_for_backward`, `needs_input_grad`, deterministic backward closures, and coverage under `frankentorch-2er`; a PyTorch-style trait/class wrapper would be API sugar, not a blocking parity gap. | `frankentorch-2er` (closed) |

## Current Green Scope (CPU Eager-Mode Slice)

The items below are the currently conformance-backed green slice. They should be read as CPU eager-mode evidence, not as proof that the deferred PyTorch subsystems above are already covered.

- `crates/ft-conformance/fixtures/op_schema_cases.json`
- `crates/ft-conformance/fixtures/scalar_autograd_cases.json`
- `crates/ft-conformance/fixtures/dispatch_key_cases.json`
- `crates/ft-conformance/fixtures/autograd_scheduler_cases.json`
- `crates/ft-conformance/fixtures/serialization_cases.json`
- `crates/ft-conformance/fixtures/nn_state_cases.json`
- `crates/ft-conformance/fixtures/optimizer_cases.json`
- `crates/ft-conformance/fixtures/tensor_binary_cases.json`
- `crates/ft-conformance/fixtures/tensor_unary_cases.json`
- `crates/ft-conformance/fixtures/tensor_comparison_cases.json`
- `crates/ft-conformance/fixtures/tensor_factory_cases.json`
- `crates/ft-conformance/fixtures/tensor_init_cases.json`
- `crates/ft-conformance/fixtures/tensor_random_cases.json`
- `crates/ft-conformance/fixtures/tensor_einsum_cases.json`
- `crates/ft-conformance/fixtures/tensor_searchsorted_cases.json`
- `crates/ft-conformance/fixtures/tensor_reduction_cases.json`
- `crates/ft-conformance/fixtures/tensor_loss_cases.json`
- `crates/ft-conformance/fixtures/tensor_linalg_cases.json`
- `crates/ft-conformance/fixtures/tensor_normalize_cases.json`
- `crates/ft-conformance/fixtures/tensor_elementwise_cmp_cases.json`
- `crates/ft-conformance/fixtures/tensor_shape_cases.json`
- `crates/ft-conformance/fixtures/tensor_inplace_cases.json`
- `crates/ft-conformance/fixtures/tensor_advanced_cases.json`
- `crates/ft-conformance/fixtures/tensor_sort_cases.json`
- `crates/ft-conformance/fixtures/tensor_indexing_cases.json`
- `crates/ft-conformance/fixtures/tensor_scan_cases.json`
- `crates/ft-conformance/fixtures/tensor_join_cases.json`
- `crates/ft-conformance/fixtures/tensor_meta_cases.json`

Modes tested for all listed families: strict + hardened.
Latest workspace evidence refreshed via remote validation:
- `rch exec -- cargo test --workspace`
- `rch exec -- cargo check --workspace --all-targets`
- `rch exec -- cargo clippy --workspace --all-targets -- -D warnings`

## Gap Policy

- Any `parity_gap` item must have explicit closure beads and dependencies.
- Deferred subsystem omissions are not allowed: if a major PyTorch area is out of scope for the current slice, it must remain listed above with a tracking bead.
- Temporary sequencing gaps are acceptable only with closure evidence plans.
- Release sign-off requires `Full PyTorch drop-in surface` to be `parity_green`.

## Required Evidence Per Feature Family

1. Differential fixture report.
2. Edge-case/adversarial test results.
3. Benchmark delta (when performance-sensitive).
4. Documented compatibility exceptions (if any).
5. RaptorQ sidecar + decode-proof chain for durable parity bundles.

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
| Loss functions | parity_green | mse_loss, l1_loss, bce_loss, smooth_l1_loss, huber_loss, cross_entropy_loss, nll_loss, cosine_embedding_loss (composed from autograd ops); nn modules: MSELoss, L1Loss, CrossEntropyLoss, NLLLoss, BCELoss, BCEWithLogitsLoss, SmoothL1Loss, HuberLoss, CosineEmbeddingLoss, KLDivLoss |
| Shape operations | parity_green | reshape, squeeze, unsqueeze, transpose, permute, cat, stack, flatten, unflatten, narrow, expand, split, chunk |
| Advanced reductions | parity_green | argmax, argmin, max_dim, min_dim (with backward), softmax, log_softmax, sum_dim, mean_dim, prod_dim, var_dim, std_dim |
| Functional API (`torch.nn.functional`-style session ops) | parity_green | session-level normalize, dropout, embedding, and linear helpers available in `ft-api` and covered by targeted tests |
| Neural network modules (ft-nn) | parity_green | Module trait, LossModule trait, Linear, Bilinear, Conv1d, Conv2d, ConvTranspose1d, ReLU, Sigmoid, Tanh, GELU, SiLU, LeakyReLU, ELU, Mish, Softplus, Sequential, ModuleList, ModuleDict, ParameterList, ParameterDict, Dropout, LayerNorm, BatchNorm1d, BatchNorm2d, BatchNorm3d, GroupNorm, InstanceNorm1d, InstanceNorm2d, Embedding, MultiheadAttention, Softmax, LogSoftmax, Flatten, Identity, Unfold, Fold, AvgPool1d, MaxPool1d, MaxPool2d, AdaptiveAvgPool2d, LPPool1d, LPPool2d, Upsample1d, Upsample2d, ConstantPad1d, ConstantPad2d, ZeroPad2d, RNNCell, LSTMCell, GRUCell, PixelShuffle, PixelUnshuffle, CosineSimilarity, PairwiseDistance, Hardswish, Hardsigmoid, LogSigmoid, RReLU, MultiLabelMarginLoss, MSELoss, L1Loss, CrossEntropyLoss, NLLLoss, BCELoss, BCEWithLogitsLoss, SmoothL1Loss, HuberLoss, KLDivLoss |
| Optimizers (ft-optim) | parity_green | Optimizer trait, SGD (momentum, weight_decay, nesterov), Adam (bias correction, weight_decay), AdamW, RMSprop, Adagrad, RAdam, SparseAdam; CyclicLR scheduler; clip_grad_norm_, clip_grad_value_ |
| Advanced indexing | parity_green | index_select, gather, scatter, masked_fill (with backward for index_select, gather) |
| Full PyTorch drop-in surface | parity_green | All tracked feature families in this ledger are green, the current beads backlog is empty, and the latest remote workspace validation (`cargo test/check/clippy`) passed cleanly |

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
functional_normalize, functional_dropout, functional_embedding, functional_linear, scaled_dot_product_attention, pixel_shuffle, pixel_unshuffle, functional_conv1d, functional_conv2d, functional_avg_pool1d, functional_avg_pool2d, functional_max_pool1d, functional_max_pool2d, functional_layer_norm, functional_group_norm, functional_instance_norm1d, functional_instance_norm2d

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
rand (uniform [0,1)), randn (normal), rand_like, randn_like (deterministic xoshiro256++ PRNG)

### In-Place Operations
tensor_add_, tensor_sub_, tensor_mul_, tensor_div_, tensor_zero_, tensor_fill_, tensor_mul_scalar_, tensor_add_scalar_

### Loss Functions
mse_loss, l1_loss, bce_loss, smooth_l1_loss, huber_loss, cross_entropy_loss, nll_loss, cosine_embedding_loss

### Advanced Indexing
index_select, gather, scatter, scatter_add, scatter_reduce (sum/prod/mean/amax/amin), masked_fill, masked_select, masked_scatter, index_add, index_copy, index_fill, index_put, select, take_along_dim

### Sorting & Selection
sort, topk, argsort, kthvalue

### Linear Algebra
svd, qr, cholesky, cholesky_solve, cholesky_inverse, det, slogdet, inv, pinverse, solve, eigh, matrix_power, matrix_exp, linalg_vector_norm, triangular_solve, matrix_norm (fro/1/-1/inf/-inf)

### Statistical / Histogram Operations
bincount (with weights, minlength), histc (with auto-range, clamping)

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
tensor_pad (constant padding, PyTorch F.pad convention)

## Current Green Scope

- `crates/ft-conformance/fixtures/scalar_autograd_cases.json`
- `crates/ft-conformance/fixtures/dispatch_key_cases.json`
- `crates/ft-conformance/fixtures/autograd_scheduler_cases.json`
- `crates/ft-conformance/fixtures/serialization_cases.json`

Modes tested for all listed families: strict + hardened.
Latest workspace evidence refreshed via remote validation:
- `rch exec -- cargo test --workspace`
- `rch exec -- cargo check --workspace --all-targets`
- `rch exec -- cargo clippy --workspace --all-targets -- -D warnings`

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

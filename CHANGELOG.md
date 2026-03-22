# Changelog

All notable changes to [FrankenTorch](https://github.com/Dicklesworthstone/frankentorch) are documented in this file.

FrankenTorch is a clean-room Rust reimplementation of PyTorch targeting complete drop-in replacement with semantic fidelity, mathematical rigor, operational safety, and profile-proven performance. The workspace enforces `unsafe_code = "forbid"` globally. There are no formal releases or tags yet; this changelog tracks development by capability milestones derived from the commit history.

Repository: <https://github.com/Dicklesworthstone/frankentorch>

---

## Milestone 7 -- Dtype Generalization, Data Pipeline, and SafeTensors (2026-03-12 .. 2026-03-14)

### Half-Precision and Complex Dtypes

- Add F16 and BF16 half-precision dtype support with full promotion rules ([`54b0df6`](https://github.com/Dicklesworthstone/frankentorch/commit/54b0df6930739486447d889acbc65b3335c6b9a9))
- Add Complex64 and Complex128 dtype support ([`5504783`](https://github.com/Dicklesworthstone/frankentorch/commit/55047839e164583570a6cfcf8f12e31ad28e0188))

### Dtype Promotion and Casting

- Implement dtype promotion rules and `to_dtype` casting across the full stack ([`ec90590`](https://github.com/Dicklesworthstone/frankentorch/commit/ec905904203ca50beebd0d3adb99f2b13ddd45de))
- Add `create_graph` flag for higher-order derivatives; make f32 a first-class dtype ([`76414dd`](https://github.com/Dicklesworthstone/frankentorch/commit/76414dd78b5aa0f5bf4c0df53d553045c8cebde5))

### Zero-Copy Storage

- Implement true zero-copy `tensor.view()` with `Arc`-backed shared storage ([`72e1e1a`](https://github.com/Dicklesworthstone/frankentorch/commit/72e1e1ac04085f6d59b17f9a08b3df4f3b8ae6af))
- Update tests for `Arc`-wrapped `TensorStorage` API and use `meta()` for shape queries ([`19e69c5`](https://github.com/Dicklesworthstone/frankentorch/commit/19e69c5ef96bfaa8160bea646ed97980797321a4))

### Tensor Factory and Manipulation Expansion

- Add `logspace`, `empty`, `tensordot`, `kron`, `cross`, `vecdot`, `diag_embed`, `unique`, and `unique_consecutive` factory/manipulation ops ([`6b4bf37`](https://github.com/Dicklesworthstone/frankentorch/commit/6b4bf37474900bfe499cbb16a012cc78ce265bed))
- Implement `einsum` with full index contraction, custom autograd functions, and `scatter_add` ([`13c4381`](https://github.com/Dicklesworthstone/frankentorch/commit/13c4381530a2f47254c79a727e2d05fc6021321c))

### Data Pipeline (ft-data crate)

- Add `Dataset` trait, `DataLoader` with batching and shuffling, `searchsorted`, `nonzero`, and `masked_select` ([`ca82f38`](https://github.com/Dicklesworthstone/frankentorch/commit/ca82f3839256b9d811ecce7456073e145a7e6abe))
- Fix DataLoader batch counting, CTCLoss bounds validation ([`b0112ec`](https://github.com/Dicklesworthstone/frankentorch/commit/b0112ecd9bfb9e3231fac160d4942a1be9fd71a1))
- Guard DataLoader against zero `batch_size` to prevent infinite loop ([`02a44bd`](https://github.com/Dicklesworthstone/frankentorch/commit/02a44bd93993d3d038c9b79c500391c81123395f))

### Loss Functions

- Add `CTCLoss` with log-space computation ([`ca82f38`](https://github.com/Dicklesworthstone/frankentorch/commit/ca82f3839256b9d811ecce7456073e145a7e6abe))

### Serialization

- Add SafeTensors format support and fix workspace clippy warnings ([`1446e17`](https://github.com/Dicklesworthstone/frankentorch/commit/1446e17ccf22a14dd224d7e57b923b7b54e00f56))
- State dict save/load for nn modules ([`ca82f38`](https://github.com/Dicklesworthstone/frankentorch/commit/ca82f3839256b9d811ecce7456073e145a7e6abe))

### Optimizer Expansion

- Add Adamax and Adadelta optimizers ([`e970d0d`](https://github.com/Dicklesworthstone/frankentorch/commit/e970d0d04520bc4c87ee682e4353a5aab84743d8))

### NN Module Cleanup

- Expose EmbeddingBag and MaxUnpool field accessors, remove `dead_code` suppression ([`a10dee7`](https://github.com/Dicklesworthstone/frankentorch/commit/a10dee7e5ca5938f50f71f83b3eeae2f88489893))
- Clippy compliance and dead-code cleanup across einsum and dropout ([`d7a39db`](https://github.com/Dicklesworthstone/frankentorch/commit/d7a39dbe7811c07231acc7ae9047f19327eec22d))
- Replace manual range comparison with `contains()` for clippy compliance ([`5b0f368`](https://github.com/Dicklesworthstone/frankentorch/commit/5b0f36819cf2296623051b78f268651eba1ce0c3))

### Conformance

- Add 11 tensor operation fixture suites and report types ([`30f642a`](https://github.com/Dicklesworthstone/frankentorch/commit/30f642a73b66f0789002d21863c2d77e64aaa198))
- Add 6 tensor operation conformance suites covering scan, join, sort, indexing, inplace, and advanced operations ([`ab4a38d`](https://github.com/Dicklesworthstone/frankentorch/commit/ab4a38de822e37af050619d4bc2250776daae272))
- Add gradient verification to normalize tests, dot/min/max binary ops, and einsum helper extraction ([`a87030d`](https://github.com/Dicklesworthstone/frankentorch/commit/a87030d7292a6d917e26b54b245eb77395291550))
- Add comprehensive pooling layer variant tests ([`4c9bdd4`](https://github.com/Dicklesworthstone/frankentorch/commit/4c9bdd4f416670469a7f59b493e949c7e6a97fbf))

---

## Milestone 6 -- Module System, LR Schedulers, and Recurrent Nets (2026-03-02 .. 2026-03-05)

### Module Trait System

- Add `Module` trait with `train()`/`eval()` mode propagation, dynamic parameter/buffer registration, and buffer collection APIs ([`9f16b57`](https://github.com/Dicklesworthstone/frankentorch/commit/9f16b571494afedb083be0fdd7d20bc23bdc829f))
- Add buffer registration to BatchNorm1d training with comprehensive module registration tests ([`bdf6e75`](https://github.com/Dicklesworthstone/frankentorch/commit/bdf6e756802dd384b04f3b56531e1bcf715da918))
- Add Module `state_dict` save/load with strict mode ([`429d813`](https://github.com/Dicklesworthstone/frankentorch/commit/429d813940050922da3fd0aad57c7d432666535a))

### Autograd Enhancements

- Implement persistent gradient accumulation across backward passes ([`def8be3`](https://github.com/Dicklesworthstone/frankentorch/commit/def8be3eb45270ab42acbf2926737c12d3bd071e))
- Add gradient clipping utilities (`clip_grad_norm_`, `clip_grad_value_`) and parameter vector conversion ([`2259b61`](https://github.com/Dicklesworthstone/frankentorch/commit/2259b61d1dff21568af1f7109013b1e1bc1f041f))
- Add tensor autograd introspection API and benchmark regression gate ([`b818ee9`](https://github.com/Dicklesworthstone/frankentorch/commit/b818ee9ba9e51a2e3e270f841261224b3a92ff0a))
- Extend autograd API test coverage with 291 additional test lines ([`f8d3c00`](https://github.com/Dicklesworthstone/frankentorch/commit/f8d3c00dd007d93de580e4e5f14c58fbad3190e3))

### L-BFGS Optimizer

- Implement L-BFGS optimizer with two-loop recursion and adversarial fuzz corpus for conformance ([`40bbf37`](https://github.com/Dicklesworthstone/frankentorch/commit/40bbf37eb769b50afcdf7519d5c094c4c64b8d0a))

### LR Scheduler Suite

- Add 6 LR schedulers (StepLR, MultiStepLR, ExponentialLR, CosineAnnealingLR, ReduceLROnPlateau, CyclicLR) with `step_with_metric` trait method ([`6f67577`](https://github.com/Dicklesworthstone/frankentorch/commit/6f675774fa1b6eefa94879a12024e81995222b6e))
- Add LambdaLR, SequentialLR, and ChainedScheduler LR schedulers ([`c1c1339`](https://github.com/Dicklesworthstone/frankentorch/commit/c1c133979761d5d7692d1cf2edccadccfb304cd9))
- Add OneCycleLR scheduler and optimizer momentum trait methods ([`41eb69b`](https://github.com/Dicklesworthstone/frankentorch/commit/41eb69b2248bd9373d19d87f5aa9b11ba406ba46))
- Add tensor comparison ops and dtype aliases ([`41eb69b`](https://github.com/Dicklesworthstone/frankentorch/commit/41eb69b2248bd9373d19d87f5aa9b11ba406ba46))

### Linear Algebra

- Add matrix inverse kernel and `SingularMatrix` error variant ([`9767539`](https://github.com/Dicklesworthstone/frankentorch/commit/9767539baa38f023765cb1541eadbd7c60d2494a))

### NN Module Expansion

- Add Conv3d, recurrent modules (RNN, LSTM, GRU), and expand nn module coverage ([`e012feb`](https://github.com/Dicklesworthstone/frankentorch/commit/e012febce325a3a68366132114bb5345cabf5356))

### Conformance

- Expand nn_state fixture coverage with register, export, and load edge cases ([`9ed4b38`](https://github.com/Dicklesworthstone/frankentorch/commit/9ed4b387c3c17ab94844823de24a33a58f3f5f6a))

---

## Milestone 5 -- NN Module Rewrite, f32 Support, and Attention (2026-02-25 .. 2026-02-26)

### Neural Network Modules

- Major neural network module rewrite with ~4,700 net new lines ([`0ca8d2c`](https://github.com/Dicklesworthstone/frankentorch/commit/0ca8d2c18f6bec3af4c82ddb6d5ceea4c47edbcf))
- Add MultiheadAttention, LayerNorm, BatchNorm1d, Conv1d, AvgPool1d, Embedding, and more nn modules ([`a0ce5f4`](https://github.com/Dicklesworthstone/frankentorch/commit/a0ce5f403a3a0eeeb45391bf36a8a02b48effcfa))
- Add similarity functions (cosine similarity, pairwise distance), expand nn modules, and rewrite optimizer suite ([`3a67f64`](https://github.com/Dicklesworthstone/frankentorch/commit/3a67f64082cecca85990e7b398ca20936a48ad0a))
- Add numerically-stable `BCEWithLogitsLoss` ([`569334b`](https://github.com/Dicklesworthstone/frankentorch/commit/569334be5df90a8b92ae8da295979bfbdac0845e))

### Autograd and Grad Context

- Add `retain_graph`, `no_grad`/`enable_grad` context managers, linalg kernels, and Module introspection ([`7f4529a`](https://github.com/Dicklesworthstone/frankentorch/commit/7f4529a24aade4201bb5d401eaa48238221e75a5))

### f32 Dtype Support

- Add f32 dtype support across kernel, dispatch, and autograd layers ([`370442e`](https://github.com/Dicklesworthstone/frankentorch/commit/370442e40af9a9d01cc4e3627c6ca800600b221f))
- Add f32 session API and fix autograd dispatch to properly convert f32 tensors ([`c355255`](https://github.com/Dicklesworthstone/frankentorch/commit/c3552558a4beea548a5f993acd4a70cf6255237c))
- Add dtype validation guards to `ensure_f32` and `ensure_f64` ([`11b1da6`](https://github.com/Dicklesworthstone/frankentorch/commit/11b1da6ee148dab801f4707415317952bb97bf7d))

### PyTorch-Compatible Tensor Repeat

- Support leading-dimension expansion in `tensor_repeat` (PyTorch-compatible behavior) ([`0bc2cdb`](https://github.com/Dicklesworthstone/frankentorch/commit/0bc2cdb087611d1601b81ee06ea820de4532b455))
- Add edge-case coverage for flip, repeat, roll, any/all, and median ([`618141c`](https://github.com/Dicklesworthstone/frankentorch/commit/618141c9627169fb7b69abfe2b1be63f4f69708d))

### Input Validation

- Validate `cosine_embedding_loss` target values and simplify `smooth_l1` ([`e82d5a9`](https://github.com/Dicklesworthstone/frankentorch/commit/e82d5a9b965c44257e1e2bf3540d4f4ba14e16d7))

### Code Quality

- Apply rustfmt + Clippy fixes across all crates for consistent style ([`43c7856`](https://github.com/Dicklesworthstone/frankentorch/commit/43c7856fe97b207f2491a4c54c167612ac732c28))

---

## Milestone 4 -- Safety Hardening and Overflow Protection (2026-02-22 .. 2026-02-23)

### Backward Pass Safety

- Replace panicking asserts with proper error returns, add scatter backward, and harden index validation ([`8f2af1c`](https://github.com/Dicklesworthstone/frankentorch/commit/8f2af1c5eb136b43ec9b828bc4cba33e18b0b9e3))
- Harden backward passes and `reduce_sum_for_broadcast` against invalid inputs ([`fa3692a`](https://github.com/Dicklesworthstone/frankentorch/commit/fa3692a90930966e034810eb6d9dfff8ed9b16e2))
- Further backward pass hardening and gradient coverage ([`5278c6e`](https://github.com/Dicklesworthstone/frankentorch/commit/5278c6eb78c966b4cc4a17aa74677ad55cbba49c))
- Add shape validation to `tensor_where`, harden backward passes with overflow checks ([`620b45c`](https://github.com/Dicklesworthstone/frankentorch/commit/620b45cb18cda18fc91443299665066aae5a2472))

### Overflow-Safe Arithmetic

- Add checked arithmetic for shape volumes across autograd, core, and serialize to prevent overflow ([`75c48db`](https://github.com/Dicklesworthstone/frankentorch/commit/75c48db448fc6b2dc615e4de8b7aa8c4d2c43207))
- Overflow-safe arithmetic hardening in autograd and CPU kernels ([`7b6fd42`](https://github.com/Dicklesworthstone/frankentorch/commit/7b6fd4228f31234b5f55d7b985c56e993ef6ec24))
- Safe f64-to-isize index conversion for `index_select` operations ([`5f560e0`](https://github.com/Dicklesworthstone/frankentorch/commit/5f560e0c861b824f73b2664e5fbae067b63f0dff))
- Hoist zero-dimension check before multiplication loop in `numel()` ([`abe40ca`](https://github.com/Dicklesworthstone/frankentorch/commit/abe40ca179264e57ff6b092ae6260c3fd246c2a1))

### Dispatch Layer Hardening

- Enforce dtype and device compatibility checks for binary and join operations ([`a63ceed`](https://github.com/Dicklesworthstone/frankentorch/commit/a63ceed61e26f22033682964d09d672ad485a802))

### NN and Optimizer Validation

- Validate `Linear` `in_features` and `Dropout` probability to fail closed on invalid inputs ([`c7722a6`](https://github.com/Dicklesworthstone/frankentorch/commit/c7722a6b512faf16f262cca3165d326faa918c3a))
- Validate optimizer hyperparameters; fix Adam/AdamW bias correction overflow ([`26b1a36`](https://github.com/Dicklesworthstone/frankentorch/commit/26b1a36a84586a92e227b5c8084f55f5184d9d1d))
- Harden SGD/Adam/AdamW with overflow-safe step counting, gradient shape validation, and state length checks ([`6b29394`](https://github.com/Dicklesworthstone/frankentorch/commit/6b2939483d830dda89873ab546da7e9df677dd77))

### Tensor Factory Hardening

- Harden tensor factories against shape overflow, strengthen in-place op validation, add evidence recording ([`17364cc`](https://github.com/Dicklesworthstone/frankentorch/commit/17364cc98a1be08c6316cb477962cca5455907d3))

### Bug Fixes

- Pass correct input node ID to `ensure_tensor_len` validation ([`dc15c29`](https://github.com/Dicklesworthstone/frankentorch/commit/dc15c29290cedb6a2bdf3711148d0b22fd885cc7))
- Apply storage offset to mask in `masked_fill` operation ([`33e8544`](https://github.com/Dicklesworthstone/frankentorch/commit/33e8544ff730cef578d55222eaae2a42e00c4c0e))
- Handle zero-step edge case in `linspace` ([`b2ecc40`](https://github.com/Dicklesworthstone/frankentorch/commit/b2ecc4094b3e449a4eb1c1ba64958212803fcc59))
- Restrict commutative metamorphic test to actually commutative ops ([`c840239`](https://github.com/Dicklesworthstone/frankentorch/commit/c8402394f4a31cd979e0243a8968c4af604caeb7))
- Include `contract_ok` in pass check, respect dtype/device in dispatch cases ([`80ace83`](https://github.com/Dicklesworthstone/frankentorch/commit/80ace83189160ebbb874269bbd5eb82829ad9f64))

### Refactoring

- Convert `diag`/`triu`/`tril`/`masked_fill` from eager to graph-traced ops ([`12d8fe7`](https://github.com/Dicklesworthstone/frankentorch/commit/12d8fe714a4b3c05252bba6b768a37d98d218b40))
- Normalize floating-point tolerances to 1e-12 across ft-api and ft-autograd tests ([`0878296`](https://github.com/Dicklesworthstone/frankentorch/commit/0878296557fa21bcd0bf592803683f3792b99a60))
- Swap `assert_eq!` argument order to `actual, expected` in conformance tests ([`ff592f3`](https://github.com/Dicklesworthstone/frankentorch/commit/ff592f3fe142fe5e24f05c2abd86beae6e7e7348))
- Improve formatting and simplify test mutation patterns in ft-autograd ([`24aee35`](https://github.com/Dicklesworthstone/frankentorch/commit/24aee3554b15cfdbc9f09a7adab02474f5e3efde))
- Normalize assert match formatting in `reduce_sum_for_broadcast` tests ([`2a8766a`](https://github.com/Dicklesworthstone/frankentorch/commit/2a8766a88123e1aca0e80416034f07ae6fef9ccf))
- Update device-mismatch assertions and normalize formatting in conformance ([`c3d17d4`](https://github.com/Dicklesworthstone/frankentorch/commit/c3d17d4078c9d18bc55d96efc8f4ccd061f099f5))

---

## Milestone 3 -- NN/Optimizer Crates, Operator Porting, and Tensor Operations (2026-02-19 .. 2026-02-21)

### ft-nn and ft-optim Crates (new)

- Add `ft-nn` and `ft-optim` crates; major API, autograd, and kernel expansion ([`47eaf51`](https://github.com/Dicklesworthstone/frankentorch/commit/47eaf510f049133377b8d1a6ec05202848947a1f))
- Expand optimizer, nn, and kernel subsystems with conformance coverage ([`708debe`](https://github.com/Dicklesworthstone/frankentorch/commit/708debead6d12c53586fbef354e23f6eb7eff0c3))
- Implement AdamW optimizer with decoupled weight decay ([`46aa9a4`](https://github.com/Dicklesworthstone/frankentorch/commit/46aa9a4c4f5e9bc410b78761bb6eb78f0468ad8a))

### Operator Porting (kernel-cpu + dispatch + autograd)

- Port matmul operator family across the full runtime stack ([`223b210`](https://github.com/Dicklesworthstone/frankentorch/commit/223b210a9b6ba749c7fb85f9722eace8e97a379d))
- Port activation functions (relu, sigmoid, tanh, etc.) and comparison ops across the runtime stack ([`d4256ef`](https://github.com/Dicklesworthstone/frankentorch/commit/d4256ef2d2206ba53887d18038c58ce91e066136))
- Port pow/sqrt/reciprocal and clamp/min/max ops across the runtime stack ([`c46e69c`](https://github.com/Dicklesworthstone/frankentorch/commit/c46e69c7aabcca989c5903649d0b20bb3b44f133))
- Add dimension-aware reduction ops (`sum_dim`, `mean_dim`) across the runtime stack ([`14f02f6`](https://github.com/Dicklesworthstone/frankentorch/commit/14f02f66be252917c62572ead6d62cf9740419cf))
- Massive expansion of unary operations with full backward pass support ([`5e4c518`](https://github.com/Dicklesworthstone/frankentorch/commit/5e4c51853f62989e47e6df9adbf2f309d735d486))
- Extend API surface and autograd tape for additional ops ([`47b6cdd`](https://github.com/Dicklesworthstone/frankentorch/commit/47b6cddeab76c066a9f90d7a8421b51cd98882a8))
- Update CPU kernel implementation ([`900741b`](https://github.com/Dicklesworthstone/frankentorch/commit/900741b11b19c6061eb02334f486d9ed64b417c4))

### Tensor Shape Operations

- Implement `cat`/`stack`/`reshape`/`view`/`squeeze`/`unsqueeze` with full backward pass ([`d2914d7`](https://github.com/Dicklesworthstone/frankentorch/commit/d2914d7596d58acad87fc30e551ddc596f4f888e))
- Add join dispatch layer for `cat`/`stack` tensor operations ([`43544e4`](https://github.com/Dicklesworthstone/frankentorch/commit/43544e4b3f4e9b6679633dbbf01ecadec525e536))

### Scan and Sorting Operations

- Add `cumsum` and `cumprod` scan operations with full autograd support ([`7894406`](https://github.com/Dicklesworthstone/frankentorch/commit/78944066834a45a29a6913e113aa1010876fe215))
- Add `sort`, `topk`, `where`, linear algebra ops, tensor manipulation, and loss functions ([`edf473a`](https://github.com/Dicklesworthstone/frankentorch/commit/edf473ab4ff99f1c6e0bcb14aa94754c05fb1d25))

### IEEE Floating-Point Correctness

- Fix inf/nan equality semantics in CPU kernels ([`b6d0d4b`](https://github.com/Dicklesworthstone/frankentorch/commit/b6d0d4b32174d14aa6317a90cf25f67983722a67))
- Add IEEE special-value comparison dispatch tests ([`a145134`](https://github.com/Dicklesworthstone/frankentorch/commit/a14513403779a6b367d1efe3294b8221d53d64f8))
- Align comparison dispatch kernel metadata with key ([`d5ccd18`](https://github.com/Dicklesworthstone/frankentorch/commit/d5ccd1857a94e7357efeba48f62c3d0c03ce3291))
- Add IEEE comparison regressions ([`08fa018`](https://github.com/Dicklesworthstone/frankentorch/commit/08fa01839431b6045f7e52143e130cb125a6b2e3))
- Add NaN ordering comparison regressions ([`6a620af`](https://github.com/Dicklesworthstone/frankentorch/commit/6a620afbd0606d9eb5ee6350210571717f0488a8))

### Bug Fixes

- Fix autograd gradient correctness, tensor storage safety, kernel offset handling, and nn module PyTorch conformance ([`d5e8802`](https://github.com/Dicklesworthstone/frankentorch/commit/d5e880241d55f72ec2d9e0468a1dd7424620a757))
- Fix `rand_like`/`randn_like` to use source tensor shape; add negative index support in gather backward ([`38d2ed4`](https://github.com/Dicklesworthstone/frankentorch/commit/38d2ed4c4f55171481be93c9a6cc2b3e5d24dcb3))
- Propagate NaN through activation gradients; use deterministic FNV-1a hasher for tensor fingerprints ([`0dcce83`](https://github.com/Dicklesworthstone/frankentorch/commit/0dcce83f2c782619fb7daff01a3b9c0ec9a1cc3e))
- Fix `expand` backward numel to allow zero-sized tensors instead of clamping to 1 ([`b8e73a8`](https://github.com/Dicklesworthstone/frankentorch/commit/b8e73a8ee5c5b25efaa94198cc98f70b539df05f))

### Branding

- Add GitHub social preview image (1280x640) ([`87326d9`](https://github.com/Dicklesworthstone/frankentorch/commit/87326d9c7ceae95b6c23e28afca14c7c334aa7a2))
- Optimize frankentorch illustration WebP asset ([`8059699`](https://github.com/Dicklesworthstone/frankentorch/commit/8059699933c3fa6aec64980a1226032201c7437b))

---

## Milestone 2 -- Autograd Safety, Contiguous f64 Ops, and Durability Evidence (2026-02-17 .. 2026-02-18)

### Contiguous f64 Tensor Operations

- Add contiguous f64 elementwise tensor ops and enrich `KernelError` taxonomy ([`6b76222`](https://github.com/Dicklesworthstone/frankentorch/commit/6b76222906947ce42598af44426587738cb37d6a))
- Add contiguous f64 tensor binary dispatch path with comprehensive tests ([`da56092`](https://github.com/Dicklesworthstone/frankentorch/commit/da560922588977d5ec006627f856ba96d271cdd2))
- Enhance dispatch key routing, autograd scalar cases, and CPU kernel implementations ([`e46b4da`](https://github.com/Dicklesworthstone/frankentorch/commit/e46b4da07b8bccfcedf80687e6cf187d6871c220))

### Legacy Oracle Safety

- Add stream-size bounds checking and structured error formatting for legacy oracle ([`554c732`](https://github.com/Dicklesworthstone/frankentorch/commit/554c732299527ba8d9492df2ed0cd15f9705f339))
- Replace `wait_with_output` with threaded stream capture, timeout polling, and child reaping ([`222bd49`](https://github.com/Dicklesworthstone/frankentorch/commit/222bd49efa24997b38d2eb3b0e00c081d52b4484))
- Add fail-closed boundary tests for legacy oracle script output safety ([`de642ec`](https://github.com/Dicklesworthstone/frankentorch/commit/de642ec7116b90d7ee3a22bb814839e75225ce5f))

### Autograd Fail-Closed Hardening

- Make `TensorMeta::numel` overflow-safe ([`3ecc6d8`](https://github.com/Dicklesworthstone/frankentorch/commit/3ecc6d87f45057fe1e463043d8bfdec0c8b867d1))
- Validate strided storage span in `DenseTensor::from_storage` ([`1d1b637`](https://github.com/Dicklesworthstone/frankentorch/commit/1d1b6373f9251ae779a44ba836b08c2db9662b11))
- Fail closed on tensor values for non-contiguous layouts ([`84017ec`](https://github.com/Dicklesworthstone/frankentorch/commit/84017ecad5932646eb43f115f7765e01b2aa4298))
- Fail closed on backward mul/div reads ([`bde06ac`](https://github.com/Dicklesworthstone/frankentorch/commit/bde06acbaef7830fbd82bc7244e7d68443121744))
- Preserve initial backward dependency snapshots ([`69e1787`](https://github.com/Dicklesworthstone/frankentorch/commit/69e17874c4de8768c7ce08a531477edad2ab7cd6))
- Fail closed on non-grad roots ([`c9b04a9`](https://github.com/Dicklesworthstone/frankentorch/commit/c9b04a9fcf9f05cde52b94f5acf5c096e9b483b8))
- Normalize TensorTape binary output storage metadata ([`3a507b0`](https://github.com/Dicklesworthstone/frankentorch/commit/3a507b0c9e86aba754d54a6e2ebcd9a3de2ff109))

### Runtime Durability Evidence

- Record runtime durability evidence for decode failures ([`707c5c6`](https://github.com/Dicklesworthstone/frankentorch/commit/707c5c659fa51b675ceee3f27966d15f089e1257))
- Wire runtime durability evidence into serialization logs ([`45516a0`](https://github.com/Dicklesworthstone/frankentorch/commit/45516a0ec6176f244855c529977b965df5e49943))
- Emit runtime evidence in conformance forensic logs ([`e173fe1`](https://github.com/Dicklesworthstone/frankentorch/commit/e173fe1a08ab43d6b737803839cab6b39b973b4b))
- Add conformance smoke for durability decode evidence ([`3e57a34`](https://github.com/Dicklesworthstone/frankentorch/commit/3e57a345750f861c0cdf5b826372e132c6877869))

### Conformance and Testing

- Add tensor binary fixture-driven conformance suite ([`40d1829`](https://github.com/Dicklesworthstone/frankentorch/commit/40d182904a231741f285950cd3c3b543bea553b1))
- Add fail-closed tensor device mismatch regression tests ([`5a4dd98`](https://github.com/Dicklesworthstone/frankentorch/commit/5a4dd98a938c7290bd470e2b1f6ee327c982b799))
- Add mul/div tensor API smoke and backward coverage ([`c44b5a8`](https://github.com/Dicklesworthstone/frankentorch/commit/c44b5a8e90fcf7447df68352ad23b26b7d1ce275))
- Cover tensor hardened reentrant fallback evidence ([`6061fa0`](https://github.com/Dicklesworthstone/frankentorch/commit/6061fa044877d945455f33dc1f86557fc90e4e4a))
- Fix non-contiguous add fixture precondition ([`6f119ad`](https://github.com/Dicklesworthstone/frankentorch/commit/6f119ad01dde27d0c1c1ee73e7bc35be2d948228))
- Fix non-contiguous smoke fixture precondition ([`67e9417`](https://github.com/Dicklesworthstone/frankentorch/commit/67e94175abb2820f368e19ac4d6d6427ac5bfe43))

### Schema Registry Dispatch

- Add schema-registry dispatch path for FT-P2C-003 ([`7ae63cc`](https://github.com/Dicklesworthstone/frankentorch/commit/7ae63ccf5a8c95f891f21f98b60fb82a74b88894))

### Cross-Crate API Updates

- Update workflow corpus, gap ledger, serialization fixtures, and cross-crate APIs ([`7d325f3`](https://github.com/Dicklesworthstone/frankentorch/commit/7d325f3638610f1e7a745fb4f1e261832e9d8f39))
- Update ft-core lib implementation and granular execution tracking ([`2740a6e`](https://github.com/Dicklesworthstone/frankentorch/commit/2740a6eebddc7d8e0fbfc2617837dd95a32897af))

### Licensing

- Adopt MIT + OpenAI/Anthropic rider across workspace metadata ([`96b591d`](https://github.com/Dicklesworthstone/frankentorch/commit/96b591d8441e11c76fe9d6492f55a0889fbf3d9b))

---

## Milestone 1 -- Phase2c Conformance Framework and CI Readiness (2026-02-13 .. 2026-02-16)

### Project Bootstrap

- Bootstrap FrankenTorch clean-room Cargo workspace with initial crate map: `ft-core`, `ft-device`, `ft-dispatch`, `ft-kernel-cpu`, `ft-runtime`, `ft-serialize`, `ft-conformance`, `ft-api`, `ft-autograd` ([`ae6ee9f`](https://github.com/Dicklesworthstone/frankentorch/commit/ae6ee9f035f224727b89529923c019dbf9b60861))
- Add WebP hero image to README and tighten ignore rules ([`3e29e97`](https://github.com/Dicklesworthstone/frankentorch/commit/3e29e97944e75b462d95c69be26743e696b4091d))

### Conformance Framework (ft-conformance)

- Add artifact validator and hardened conformance framework ([`9546c39`](https://github.com/Dicklesworthstone/frankentorch/commit/9546c394384273ccc0f8de850b24c96d3d2a74b3))
- Add structured logging, e2e forensics emitter, and parallel packet validation ([`a3c1837`](https://github.com/Dicklesworthstone/frankentorch/commit/a3c1837164d37068af97a5fbf1b37f431ac869f9))
- Add tensor-meta differential conformance harness and full-parity doctrine ([`abf5b40`](https://github.com/Dicklesworthstone/frankentorch/commit/abf5b40fa58c69693a5f3aef928eac7c41e71f9c))
- Add reliability gates, failure forensics, and user workflow scenario corpus ([`1ced638`](https://github.com/Dicklesworthstone/frankentorch/commit/1ced638c51c69227a441bb6956880eb1a3b2f494))
- Add metamorphic and adversarial dispatch key checks ([`4321806`](https://github.com/Dicklesworthstone/frankentorch/commit/432180605ff4925273cf933e5276f3f653f8c3d6))
- Expand conformance harness with tensor meta fixture expansion ([`efb84bf`](https://github.com/Dicklesworthstone/frankentorch/commit/efb84bfb34df7ca0dbaca0f8636e4293401d217e))
- Expand conformance harness with additional test coverage ([`e802eee`](https://github.com/Dicklesworthstone/frankentorch/commit/e802eeeac80d5154d554b97df20987407a1e5f24))
- Hoist fixture I/O out of mode loop and extract `_with_fixture` helpers ([`dc3f2c0`](https://github.com/Dicklesworthstone/frankentorch/commit/dc3f2c0faf3f8ddddeab3401aad8c6a23710d2a8))
- FT-P2C-005 differential report filtering and forensics evidence ([`7e7973f`](https://github.com/Dicklesworthstone/frankentorch/commit/7e7973fd52e7d31bccfb1995bea878e5c51bd714))
- Expand conformance fixtures, reliability budget, and kernel dispatch hardening ([`a170364`](https://github.com/Dicklesworthstone/frankentorch/commit/a1703642b343f9f1595ffb8a7af45d0c4379409a))
- Add metamorphic, adversarial, and e2e forensics validation for FT-P2C-004 ([`2ac255e`](https://github.com/Dicklesworthstone/frankentorch/commit/2ac255ea195a45ca5b7aa18956adb03bbc2f013d))

### RaptorQ Durability Pipeline

- Add phase2c artifact validation, RaptorQ durability pipeline, and perf rebaseline ([`50559cd`](https://github.com/Dicklesworthstone/frankentorch/commit/50559cdf9d4fd57cbd93ad5755f087e00cc6f40c))
- Ensure RaptorQ repair count meets decoder `k_prime` minimum ([`5c988e3`](https://github.com/Dicklesworthstone/frankentorch/commit/5c988e3c9a67483b387e46924da1f2e39cf9eb2a))

### Dispatch Key Resolution

- Add property-based test suite for dispatch keyset resolution and mode validation ([`637d25a`](https://github.com/Dicklesworthstone/frankentorch/commit/637d25af525c3b88d4345a8d0a99cc3784d34339))
- Expand FT-P2C-002/003 conformance and dispatch key resolution ([`81200e4`](https://github.com/Dicklesworthstone/frankentorch/commit/81200e42a9760af161c9c7d43aae8b1003b1d53f))

### Device Guards and NN State Conformance

- Expand P2C-007 device guard scenarios and conformance coverage ([`3ac1fb2`](https://github.com/Dicklesworthstone/frankentorch/commit/3ac1fb220bdfdb097d2ff0cf4d30631508f62f97))
- Expand P2C-007 device guards and P2C-008 NN state conformance ([`0c6bf15`](https://github.com/Dicklesworthstone/frankentorch/commit/0c6bf1578495a01555d7272113b11bf1c9b5fa03))
- Extend differential engine and tests for FT-P2C-008 NN state ([`b367fea`](https://github.com/Dicklesworthstone/frankentorch/commit/b367fea98081c8253ca5feba5a7d34f84f116448))

### Serialization

- Add serialization differential checks, property tests, sidecar retry, and autograd allocation optimization ([`55fcd3b`](https://github.com/Dicklesworthstone/frankentorch/commit/55fcd3b9737ae60cac5dce2e0ce82b3466e49ab0))

### Autograd

- Expand FT-P2C-004 autograd scheduler and conformance infrastructure ([`49c3f16`](https://github.com/Dicklesworthstone/frankentorch/commit/49c3f16b9eb953e7001bc03dffa7195375e0262c))
- Continue autograd module implementation ([`21c34dd`](https://github.com/Dicklesworthstone/frankentorch/commit/21c34dd9be8ade8265e44c0e5c89d6ab3ff10943))

### CI and Readiness

- Add CI workflow, gate-window e2e assembly, and readiness drill signoff ([`39f25c1`](https://github.com/Dicklesworthstone/frankentorch/commit/39f25c1c7ed92be1cdc2b6c71514f46fb33cc437))
- Make ft-conformance CI tests oracle-portable ([`9c11249`](https://github.com/Dicklesworthstone/frankentorch/commit/9c112492bdd30649a60a3ef7a058865d1cc2275f))
- Make smoke integration test oracle-portable ([`ec19d62`](https://github.com/Dicklesworthstone/frankentorch/commit/ec19d6232859e6f9b4430d746b39e032ffa0d4a3))
- Fix G4 duplicate-mode forensics parse in e2e logs ([`314f99e`](https://github.com/Dicklesworthstone/frankentorch/commit/314f99e09b168751704450e0e0a1756bee104b14))
- Add durability linkage and fix CI ftui path ([`c5c1589`](https://github.com/Dicklesworthstone/frankentorch/commit/c5c1589a4508eb7907d9208ef5f5b827dc88e77e))
- Phase2c readiness sign-off marked READY with G8 CI evidence captured ([`70c4316`](https://github.com/Dicklesworthstone/frankentorch/commit/70c431672b80504c9c424ce338c48cb2ef4076c3))
- Update readiness sign-off with CI dispatch queue status and remediated run evidence through G7 ([`0d3a90c`](https://github.com/Dicklesworthstone/frankentorch/commit/0d3a90ccce5298804c287ac6d71ac95144cd03eb))

### Phase2c Packet Artifacts

- FT-P2C-001 contract table, fixture manifest, and legacy anchor map ([`24b3d30`](https://github.com/Dicklesworthstone/frankentorch/commit/24b3d30cc3d908e5c6b783ae1f4f4daea16c9f5f))
- FT-P2C-002 threat model and contract spec, perf baselines ([`f9860d8`](https://github.com/Dicklesworthstone/frankentorch/commit/f9860d8228d8df73d3f0e91466670e0476f52946))
- FT-P2C-002 conformance artifacts and dependency refresh ([`cc6008d`](https://github.com/Dicklesworthstone/frankentorch/commit/cc6008d81aeb10d6f0285e96e3ce2ca5e473736a))
- FT-P2C-004 fixture manifest, risk note, optimization delta, and isomorphism artifacts ([`3edef69`](https://github.com/Dicklesworthstone/frankentorch/commit/3edef699ccaac59e552a7224fd8856e040f8ad7c), [`e13d2f4`](https://github.com/Dicklesworthstone/frankentorch/commit/e13d2f4c5ed29bce32f07fd04841384e9db77ae5))
- FT-P2C-005 optimization profiling, behavior ledger, contracts, threat model, and parity evidence ([`6043af2`](https://github.com/Dicklesworthstone/frankentorch/commit/6043af2f5d4ec75f697dac87a3547438a1422fca), [`7c941e2`](https://github.com/Dicklesworthstone/frankentorch/commit/7c941e264a0ff8e8518a802502d2f8b5e5eb39b0), [`3392e54`](https://github.com/Dicklesworthstone/frankentorch/commit/3392e5451661230198e0cc2173b756ae87bfcad5), [`a544ec9`](https://github.com/Dicklesworthstone/frankentorch/commit/a544ec973679109e7ff44cb53c8f1692e35b923f), [`2089b76`](https://github.com/Dicklesworthstone/frankentorch/commit/2089b7694abdca5c4f2ae5aca704776760455353))
- FT-P2C-006 serialization packet with full validation artifact set, e2e forensics, crash triage, failure index, and security threat matrix ([`13935e4`](https://github.com/Dicklesworthstone/frankentorch/commit/13935e4cdb23c9dd66bc4904341eb8bfbbdd77f8), [`2612db5`](https://github.com/Dicklesworthstone/frankentorch/commit/2612db55066b7a447c69e25d14ca455b723c73e7), [`c3443ff`](https://github.com/Dicklesworthstone/frankentorch/commit/c3443ff5dbd6fad09f9b5f27a84394ab49a11e1d))
- FT-P2C-008 NN state contract investigation and reliability/security policy artifacts ([`1fc89fc`](https://github.com/Dicklesworthstone/frankentorch/commit/1fc89fc0a252df4f49d49234f587c0d0782f54fb), [`f0f66ef`](https://github.com/Dicklesworthstone/frankentorch/commit/f0f66efc6faf09c96917e25ac7de1c3f6aaeda63))

### Documentation

- Expand PyTorch structure analysis with detailed module hierarchy and API surface mapping ([`764dbfd`](https://github.com/Dicklesworthstone/frankentorch/commit/764dbfd65b4309426c11d6b225c19d27b1a8fdd4))
- Expand and deepen legacy analysis and PyTorch structure documentation ([`0cf4734`](https://github.com/Dicklesworthstone/frankentorch/commit/0cf47349bcead1a723b85e0db893dfaf67a52203), [`8bb312a`](https://github.com/Dicklesworthstone/frankentorch/commit/8bb312a4f6fd7866a170b4fadd30db9bc8dece2d))
- Add phase2c logging contract, method stack report, and optimization benchmark ([`e55005d`](https://github.com/Dicklesworthstone/frankentorch/commit/e55005d656166266f924a4ffc01fe91c375e1ecd))
- Update README with conformance/forensics usage and mark logging/optimization tasks complete ([`afef3f3`](https://github.com/Dicklesworthstone/frankentorch/commit/afef3f3c297ae016a726e2ca60e157b89227e204))

---

## Crate Inventory

| Crate | Purpose | Lines |
|-------|---------|------:|
| `ft-autograd` | Gradient tape, backward passes, DAC (Deterministic Autograd Contract) | 18,609 |
| `ft-nn` | Neural network modules (Linear, Conv1d/3d, RNN/LSTM/GRU, MultiheadAttention, Norm, Pooling, Embedding, Loss) | 17,696 |
| `ft-api` | Public tensor API, factory ops, manipulation, einsum, dtype promotion | 16,609 |
| `ft-conformance` | Differential conformance harness, forensics, RaptorQ durability, reliability gates | 12,171 |
| `ft-kernel-cpu` | CPU kernel implementations (elementwise, reduce, matmul, linalg, scan) | 9,991 |
| `ft-dispatch` | Dispatch key resolution, dtype/device routing, schema registry | 8,092 |
| `ft-optim` | Optimizers (SGD, Adam, AdamW, Adamax, Adadelta, L-BFGS) and LR schedulers (10 scheduler types) | 8,071 |
| `ft-core` | Tensor metadata, DType (f16/bf16/f32/f64/complex64/complex128), storage, `Arc`-backed views | 2,945 |
| `ft-serialize` | Checkpoint save/load, SafeTensors format, RaptorQ sidecars, state dict | 2,089 |
| `ft-data` | Dataset trait, DataLoader with batching/shuffling | 1,092 |
| `ft-runtime` | Runtime orchestration | 345 |
| `ft-device` | Device abstraction (CPU, CUDA placeholder) | 307 |

Total: ~98,000 lines of Rust across 187 commits (2026-02-13 to 2026-03-14).

---

## Commit Reference

Every commit hash in this changelog links to `https://github.com/Dicklesworthstone/frankentorch/commit/<HASH>`. The full linear history can be viewed at <https://github.com/Dicklesworthstone/frankentorch/commits/main>.

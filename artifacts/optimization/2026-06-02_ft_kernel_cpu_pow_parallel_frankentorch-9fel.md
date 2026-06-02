# ft-kernel-cpu Pow Parallel Pass 28

- Bead: `frankentorch-9fel`
- Umbrella: `frankentorch-kgs4`
- Skills: `/profiling-software-performance`, `/extreme-software-optimization`, `/alien-graveyard`, `/alien-artifact-coding`
- Crate: `ft-kernel-cpu`
- Target benchmark: `elementwise_bench::{pow_f64_1m_exp2.5,pow_f32_1m_exp2.5}`

## Profile Target

`pow_tensor_contiguous_f64` and `pow_tensor_contiguous_f32` were residual
elementwise hot paths: both used a serial `window.iter().map(powf).collect()`
for large contiguous tensors while neighboring unary f64 kernels already used
the crate's large-input parallel threshold.

Same-worker baseline with the parallel implementation forced to one Rayon
thread:

```text
worker: vmi1227854
command: rch exec -- env RAYON_NUM_THREADS=1 cargo bench -p ft-kernel-cpu --bench elementwise_bench -- --warm-up-time 1 --measurement-time 5 --sample-size 20
pow_f64_1m_exp2.5: [17.167 ms 20.086 ms 24.886 ms]
pow_f32_1m_exp2.5: [7.5569 ms 8.3317 ms 9.2016 ms]
```

## Alien Recommendation Card

Change: use the existing large-input `PARALLEL_THRESHOLD` and Rayon `par_iter`
for compute-bound powf maps.

Mapped graveyard sections:

- `alien_cs_graveyard.md` section 8.2: vectorized execution and morsel-driven
  parallelism, especially amortizing scheduling overhead over cache-sized
  batches.
- `high_level_summary...md` FrankenSuite matrix: cache-sized batch execution is
  a reusable throughput primitive.
- `alien_cs_graveyard.md` appendix warning: constants and cache behavior can
  erase theoretical wins, so the same-worker Criterion after run is the
  acceptance gate.

EV score: Impact 2 * Confidence 3 * Reuse 2 / Effort 1 / AdoptionFriction 1 =
12.0.

Priority tier: A for large tensors only. The threshold preserves the scalar path
for small inputs where Rayon scheduling would dominate.

Adoption wedge: reuse the crate's existing Rayon dependency, threshold constant,
and pure map shape.

Fallback trigger: revert the lever if bit-exact pow tests fail, golden checksum
changes unexpectedly, or same-worker p50 score drops below 2.0.

## Alien Artifact Proof

Selected family: certified rewrite of a pure elementwise map.

Proof obligations:

- Ordering: output collection preserves input index order for Rayon slices.
- Tie-breaking: no comparisons or tie-breakers are introduced.
- Floating point: each element still executes exactly one
  `powf_torch_signed_zero_*` call with the same input and exponent; no
  accumulation or cross-element arithmetic exists.
- RNG: not involved.
- Errors: layout/storage validation remains before the branch and is unchanged.
- Threshold: tensors below `PARALLEL_THRESHOLD` keep the serial path exactly.
- Golden output: `ft_kernel_cpu_pow_parallel_frankentorch-9fel.txt` records
  threshold-crossing f64/f32 sample bit patterns with sha256
  `f79b98f37b9c3157d6d38089e352f510b2dc42b07a74252b2b9b23f2e17e6869`.

## One Lever

For `numel >= PARALLEL_THRESHOLD`, replace serial `iter().map(...).collect()`
with `par_iter().map(...).collect()` in both contiguous f64 and f32 pow kernels.

No public API, dtype handling, storage offset validation, exponent semantics, or
signed-zero repair logic changed.

## Result

After:

```text
worker: vmi1227854
command: rch exec -- cargo bench -p ft-kernel-cpu --bench elementwise_bench -- --warm-up-time 1 --measurement-time 5 --sample-size 20
pow_f64_1m_exp2.5: [6.5456 ms 7.8146 ms 9.3768 ms]
pow_f32_1m_exp2.5: [3.3945 ms 3.9928 ms 4.6056 ms]
```

Delta:

- f64 p50: `20.086 ms -> 7.8146 ms`, about 61.1 percent faster, 2.57x by p50.
- f32 p50: `8.3317 ms -> 3.9928 ms`, about 52.1 percent faster, 2.09x by p50.
- confidence: high because baseline and after ran on the same RCH worker.
- score: Impact 2 * Confidence 3 / Effort 1 = 6.0.
- decision: keep.

## Gates

- `rch exec -- cargo test -p ft-kernel-cpu pow_parallel_matches_elementwise_bit_exact -- --nocapture` passed on worker `vmi1153651`: 1/1, 370 filtered.
- `rch exec -- cargo test -p ft-kernel-cpu --lib` passed 372/372. RCH fell back local because workers were saturated; command remained crate-scoped.
- `rch exec -- cargo check -p ft-kernel-cpu --all-targets` passed on worker `vmi1153651` after one stale moving-worktree failure.
- `rch exec -- cargo clippy -p ft-kernel-cpu --all-targets --no-deps -- -D warnings` passed on worker `vmi1149989` after stale moving-worktree failures in a separate softmax pass were corrected by the other live edits.
- `rch exec -- cargo fmt -p ft-kernel-cpu --check` passed; RCH classed fmt as a non-compilation command.
- `sha256sum -c artifacts/optimization/golden_checksums.txt --ignore-missing` passed.
- `git diff --check` passed.
- `ubs artifacts/optimization/2026-06-02_ft_kernel_cpu_pow_parallel_frankentorch-9fel.md artifacts/optimization/golden_outputs/ft_kernel_cpu_pow_parallel_frankentorch-9fel.txt artifacts/optimization/golden_checksums.txt crates/ft-kernel-cpu/benches/elementwise_bench.rs .skill-loop-progress.md` exited 0. UBS reported bench-only `expect()` warnings in the Criterion harness; its own formatting, clippy, cargo check, tests-build, cargo-audit, and cargo-deny sections were clean.

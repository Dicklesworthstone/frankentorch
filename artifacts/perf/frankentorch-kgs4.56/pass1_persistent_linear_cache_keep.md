# frankentorch-kgs4.56 pass 1: persistent packed f64 Linear weight cache

## Lever

Cache the packed f64 Linear weight transpose in `FrankenTorchSession` for the
no-grad f64 `functional_linear` path. The cache key is:

- backing storage id
- storage offset
- weight strides
- tensor version
- input/output feature dimensions

This keeps the optimization tied to the exact logical layout and invalidates on
in-place weight updates via the existing tensor version.

## Profile-backed target

The target came from `frankentorch-kgs4.56`, a follow-up to the rejected
per-call `dgemm_bt` packing lever. The profiler-evident hotspot is repeated
f64 Linear-style `dgemm_bt` where per-call weight transpose/packing overhead
remains visible after the rejected non-persistent approach.

## Same-worker benchmark

Worker: `vmi1227854`

Command:

```text
RCH_REQUIRE_REMOTE=1 RCH_WORKER=vmi1227854 rch exec -- cargo bench -j 1 -p ft-api --bench ops_bench -- linear_forward/hidden/1024 --warm-up-time 1 --measurement-time 5 --sample-size 20
```

Baseline:

```text
linear_forward/hidden/1024
time:  [819.90 us 897.68 us 997.78 us]
thrpt: [32.841 Melem/s 36.503 Melem/s 39.966 Melem/s]
```

After:

```text
linear_forward/hidden/1024
time:  [500.78 us 536.84 us 586.13 us]
thrpt: [55.906 Melem/s 61.038 Melem/s 65.434 Melem/s]
```

Median speedup: `897.68 / 536.84 = 1.672x`.
Conservative interval speedup using baseline lower / after upper:
`819.90 / 586.13 = 1.399x`.

## Behavior proof

- Ordering: the packed path calls the same row-major `gemm::dgemm` kernel over
  the same copied f64 values, and applies bias with the same row-major
  element order.
- Tie-breaking: Linear has no comparison or tie behavior.
- Floating point: packing is pure f64 bit-copy. The focused tests compare every
  f64 output bit against `linear_tensor_f64` before and after cache
  invalidation.
- RNG: no RNG state is read or mutated.
- Autograd: the lever only routes the no-grad f64 path. The grad-enabled Linear
  path remains unchanged, preserving backward graph construction and gradients.
- Aliasing/versioning: the cache key includes storage id, storage offset,
  strides, dimensions, and tensor version. The focused ft-api test updates the
  weight tensor and verifies the cache is replaced and outputs remain bit-exact.

Golden output:

```text
sha256(pass1_linear_cache_golden.txt) = b7696b25ebdfed88c226520b264c7b80695345515178db592b7c4bd552e4ccea
```

## Validation

Passed:

- `RCH_REQUIRE_REMOTE=1 RCH_WORKER=vmi1227854 rch exec -- cargo test -j 1 -p ft-kernel-cpu packed_linear_weight_f64_matches_linear_tensor_bit_exact -- --nocapture`
- `RCH_REQUIRE_REMOTE=1 RCH_WORKER=vmi1227854 rch exec -- cargo test -j 1 -p ft-api functional_linear_f64_packed_cache_matches_reference_and_invalidates_version -- --nocapture`
- `RCH_REQUIRE_REMOTE=1 rch exec -- cargo check -j 1 -p ft-kernel-cpu --all-targets` exited 0
- `RCH_REQUIRE_REMOTE=1 rch exec -- cargo clippy -j 1 -p ft-kernel-cpu --lib -- -D warnings`
- `RCH_REQUIRE_REMOTE=1 rch exec -- cargo check -j 1 -p ft-api --lib --tests`
- `git diff --check`

Known unrelated gate debt found during validation:

- `cargo fmt --package ft-api --package ft-kernel-cpu --check` reports broad
  pre-existing formatting drift across examples/tests and existing `src/lib.rs`
  regions.
- `cargo check -p ft-kernel-cpu --all-targets` exits 0 but reports existing
  warnings in `crates/ft-kernel-cpu/examples/gemm_golden.rs`.
- `cargo clippy -p ft-kernel-cpu --lib --tests -- -D warnings` is blocked by
  the existing `items_after_test_module` lint at
  `crates/ft-kernel-cpu/src/lib.rs:418`.
- `cargo clippy -p ft-api --lib --tests -- -D warnings` reports 256 existing
  lints across `crates/ft-api/src/lib.rs`.
- `ubs crates/ft-api/src/lib.rs crates/ft-kernel-cpu/src/lib.rs` produced no
  findings before hanging for several minutes in the Rust scanner; the two-file
  scanner process group was terminated and this UBS run is inconclusive.
- The pre-commit UBS hook retried the staged 15-file scan with its extended
  large-file timeout and exited with `ubs: timeout on large file scan`, again
  without actionable findings.

Follow-up bead filed: `frankentorch-le0b3`.

## Score

Impact `3.5` x Confidence `4.0` / Effort `2.0` = `7.0`.

Keep: score is above the `2.0` threshold and the same-worker benchmark shows a
clear win on the profiled repeated-Linear target.

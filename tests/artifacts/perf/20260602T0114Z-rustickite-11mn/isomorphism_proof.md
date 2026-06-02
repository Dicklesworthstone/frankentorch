# frankentorch-11mn Isomorphism Proof

## Lever

`tensor_unfold` converts the precomputed gather map from `Vec<usize>` into a
shared immutable `Arc<[usize]>` and captures clones of that shared slice in the
forward and backward closures.

## Preserved Semantics

- Output ordering: unchanged. The gather table is built in the same flat output
  order with the same coordinate decode and `in_flat` formula.
- Tie-breaking: not applicable. No comparisons or ordering decisions changed.
- Floating-point behavior: unchanged. The forward still copies `vals[k]`; the
  backward still accumulates `grad_in[in_flat] += g[flat_out]` in the same
  iteration order.
- RNG behavior: unchanged. No random state is read or advanced.
- Diagnostics and shape behavior: unchanged. Validation, overflow checks, output
  shape construction, and error strings are untouched.
- Aliasing and mutation: the gather map is immutable after construction. Sharing
  removes a duplicate allocation/copy but does not introduce mutable aliasing.

## Golden Output

`sha256sum -c tests/artifacts/perf/20260602T0114Z-rustickite-11mn/golden_checksums.txt`
passed for `dbd74c3723fcd8628e7c91b58b3676e476ea7858ed9b2bb7d6735690fa4e04db`.

## Validation

- `sha256sum -c tests/artifacts/perf/20260602T0114Z-rustickite-11mn/golden_checksums.txt`: passed.
- `git diff --check -- crates/ft-api/src/lib.rs tests/artifacts/perf/20260602T0114Z-rustickite-11mn .skill-loop-progress.md`: passed.
- `rch exec -- cargo test -p ft-api unfold`: 4 passed.
- `rch exec -- cargo test -p ft-api functional_conv2d_with_bias`: 1 passed.
- `rch exec -- cargo check -p ft-api --all-targets`: passed with the existing ft-api warning set.
- `rch exec -- cargo bench -p ft-api --bench ops_bench -- conv2d/hw/32 --warm-up-time 1 --measurement-time 5 --sample-size 20`: p50 206.43 ms after same-worker control baseline p50 216.47 ms.
- `rch exec -- cargo clippy -p ft-api --all-targets -- -D warnings`: failed on the existing ft-api lint backlog (89 errors; first errors are unrelated unused variables and broad clippy modernization findings).
- `rch exec -- cargo fmt -p ft-api --check`: failed on existing broad formatting drift in `crates/ft-api/src/lib.rs` and `crates/ft-api/benches/ops_bench.rs`.
- `ubs crates/ft-api/src/lib.rs`: failed on existing monolith-wide findings (311 critical comparison-pattern findings, 16100 warnings, 1721 info items); UBS reported no unsafe blocks and no new finding tied to the Arc gather sharing line.

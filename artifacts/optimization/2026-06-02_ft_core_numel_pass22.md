# Pass 22: ft-core TensorMeta numel

- Bead: `frankentorch-xci9`
- Skill loop: `/profiling-software-performance` plus `/extreme-software-optimization`
- Crate: `ft-core`
- Target benchmark: `tensor_meta/numel_rank8_repeated_65536`

## Profile Target

Fresh profiling after the ready perf queue was drained selected an uncovered crate:
`ft-core`. The representative metadata path repeatedly calls
`TensorMeta::numel` on a high-rank contiguous tensor shape. Before the
optimization, the non-scalar non-zero path scanned the shape twice:

1. `any(|dim| dim == 0)` to detect zero dimensions.
2. A checked multiplication loop to compute the product.

Baseline via rch Criterion:

```text
worker: vmi1153651
command: rch exec -- cargo bench -p ft-core --bench core_bench -- tensor_meta/numel_rank8_repeated_65536 --warm-up-time 1 --measurement-time 5 --sample-size 20
time: [1.0513 ms 1.0950 ms 1.1492 ms]
```

## One Lever

Fold zero-dimension detection into the checked multiplication loop:

- scalar shape remains a special case returning `1`
- the first zero dimension still returns `0`
- checked multiplication still returns `usize::MAX` on overflow
- no dtype/device/fingerprint fields are changed

## Isomorphism Proof

- Ordering: shape iteration order is unchanged and no collection order changes.
- Tie-breaking: no comparisons or tie-breakers are introduced.
- Floating point: the changed path performs no floating-point arithmetic.
- RNG: the changed path uses no RNG and does not touch caller-visible RNG state.
- Overflow: multiplication still uses `checked_mul`; overflow still saturates to
  `usize::MAX`.
- Zero dimensions: the first zero still short-circuits before later dimensions,
  including oversized later dimensions.
- Golden output:

```text
sha256: fc134f8a2fbb18b29efcf4f4d8c09d3e78c8a0b4375fc586fb9035a658bff864
file: artifacts/optimization/golden_outputs/ft_core_numel_pass22.txt
```

## Result

After via rch Criterion:

```text
worker: vmi1153651
command: rch exec -- cargo bench -p ft-core --bench core_bench -- tensor_meta/numel_rank8_repeated_65536 --warm-up-time 1 --measurement-time 5 --sample-size 20
time: [536.49 us 554.24 us 576.07 us]
```

Delta:

- p50: `1.0950 ms -> 554.24 us`
- improvement: about 49.4 percent faster
- confidence: same-worker benchmark comparison
- score: impact 2 x confidence 3 / effort 1 = 6.0
- decision: keep

## Gates

- `rch exec -- cargo test -p ft-core tensor_meta_numel_golden_summary_matches_fixture -- --nocapture` passed before and after
- `rch exec -- cargo bench -p ft-core --bench core_bench -- tensor_meta/numel_rank8_repeated_65536 --warm-up-time 1 --measurement-time 5 --sample-size 20` passed before and after
- `rch exec -- cargo check -p ft-core --all-targets` passed
- `rch exec -- cargo clippy -p ft-core --all-targets --no-deps -- -D warnings` passed
- `cargo fmt -p ft-core --check` passed
- `sha256sum -c artifacts/optimization/golden_checksums.txt --ignore-missing` passed
- `git diff --check` passed
- `ubs crates/ft-core/src/lib.rs crates/ft-core/benches/core_bench.rs crates/ft-core/Cargo.toml` passed with 0 critical findings

## Landing Note

The pass files landed in commit `ff8bf16` through a shared-index race while
validation was still running. This artifact note records the evidence correction
and keeps the performance numbers explicit in the follow-up commit message.

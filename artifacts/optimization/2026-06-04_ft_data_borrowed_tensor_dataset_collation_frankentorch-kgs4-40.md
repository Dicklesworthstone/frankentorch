# ft-data borrowed TensorDataset dataloader collation

- Bead: `frankentorch-kgs4.40`
- Date: 2026-06-04
- Agent: CobaltForge
- Skills: `/repeatedly-apply-skill`, `/extreme-software-optimization`, `/alien-graveyard`, `/alien-artifact-coding`
- Target: `dataloader/epoch_2048x256_batch128`

## Profile-backed target

The targeted Criterion benchmark repeatedly cloned `TensorDataset` samples into a
temporary `Vec<DataItem>` before collating them back into tensors. The lever attacks
that allocation/copy surface without changing sampler order or tensor arithmetic.

## One lever

Add optional `Dataset::collate_indices` and implement it for `TensorDataset`.
`DataLoader::next_batch` validates sampler indices, advances the cursor at the same
point as the prior path, then calls the hook. Datasets without a hook keep the prior
clone-then-`collate` fallback.

## Baseline

```text
RCH_REQUIRE_REMOTE=1 rch exec -- env CARGO_TARGET_DIR=/data/tmp/frankentorch-cobaltforge-ftdata-pass55-baseline cargo bench -p ft-data --bench dataloader_bench -- dataloader/epoch_2048x256_batch128 --warm-up-time 1 --measurement-time 5 --sample-size 20
```

Worker: `ts2`

```text
dataloader/epoch_2048x256_batch128
time: [4.4674 ms 4.5603 ms 4.6718 ms]
```

## After

```text
RCH_REQUIRE_REMOTE=1 rch exec -- env CARGO_TARGET_DIR=/data/tmp/frankentorch-cobaltforge-ftdata-pass55-after cargo bench -p ft-data --bench dataloader_bench -- dataloader/epoch_2048x256_batch128 --warm-up-time 1 --measurement-time 5 --sample-size 20
```

Worker: `ts2`

```text
dataloader/epoch_2048x256_batch128
time: [2.5823 ms 2.7636 ms 2.9149 ms]
```

- p50: `4.5603 ms -> 2.7636 ms`
- Speedup: `1.65x`
- Time reduction: `39.4%`
- Score: `impact 5 * confidence 4 / effort 2 = 10.0`

## Isomorphism proof

- Sample index order is unchanged: the same `indices[batch_start..batch_end]` slice drives the fast hook and fallback.
- Cursor advancement is unchanged relative to errors after index validation: the cursor advances before collation as before.
- Tensor order is unchanged: tensor names are iterated from the first sample in the same order as `collate`.
- Batch value order is unchanged: values append sample-major for each tensor via `extend_from_slice`, matching the old cloned traversal.
- Validation is unchanged: out-of-range indices, inconsistent tensor counts, name drift, shape drift, values/shape mismatch, and overflow fail closed with the same `AutogradError` family.
- Floating point is unchanged: no arithmetic was introduced or reordered.
- RNG is unchanged: no sampler or seed path changed.

## Proof and gates

```text
RCH_REQUIRE_REMOTE=1 rch exec -- env CARGO_TARGET_DIR=/data/tmp/frankentorch-cobaltforge-ftdata-pass55-tests cargo test -p ft-data dataloader -- --nocapture
```

Result: passed on `ts2`, 22 dataloader tests.

```text
RCH_REQUIRE_REMOTE=1 rch exec -- env CARGO_TARGET_DIR=/data/tmp/frankentorch-cobaltforge-ftdata-pass55-gates cargo check -p ft-data --all-targets
```

Result: passed on `ts2`.

```text
RCH_REQUIRE_REMOTE=1 rch exec -- env CARGO_TARGET_DIR=/data/tmp/frankentorch-cobaltforge-ftdata-pass55-gates cargo clippy -p ft-data --all-targets --no-deps -- -D warnings
```

Result: passed on `ts2`.

```text
cargo fmt -p ft-data --check
```

Result: passed locally after `rch` refused remote fmt as a non-compilation command.

```text
sha256sum -c artifacts/optimization/golden_checksums.txt --ignore-missing
```

Result: passed, including retained f64 serialization evidence fixture from the rejected preceding ft-serialize save-path pass.

```text
ubs crates/ft-data/src/lib.rs crates/ft-serialize/src/lib.rs crates/ft-serialize/benches/serialize_bench.rs
```

Result: exited nonzero on pre-existing heuristic findings. The two reported critical
items are false positives on non-secret numeric comparisons: `NormalizeTransform`
channel-count/shape validation from 2026-04-28 and a single-weight sampler test
asserting `i == 0` from 2026-04-06. Clippy, check, fmt, and targeted tests passed.

# frankentorch-98pz: packed TensorDataset columnar collation

## Target

- Bead: `frankentorch-98pz`
- Crate: `ft-data`
- Profile-backed hotspot: `dataloader/epoch_2048x256_batch128`
- Lever: build an optional packed, column-major `TensorDataset` cache for homogeneous valid samples. Contiguous batches copy one contiguous value slice per tensor column; non-contiguous batches copy per requested index in sampler order. Heterogeneous or malformed datasets fall back to the original item-walk validator.

## Benchmark

Command:

```text
RCH_REQUIRE_REMOTE=1 rch exec -- cargo bench -p ft-data --bench dataloader_bench -- --warm-up-time 1 --measurement-time 3 --sample-size 10
```

Baseline, worker `ts1`, before edit:

```text
dataloader/epoch_2048x256_batch128                    [489.18 us 640.51 us 732.72 us]
sampler/without_replacement_size4096_samples66560     [135.10 us 136.31 us 137.31 us]
```

After, worker `ts1`, final source:

```text
dataloader/epoch_2048x256_batch128                    [311.18 us 398.24 us 445.32 us]
sampler/without_replacement_size4096_samples66560     [135.46 us 136.48 us 137.82 us]
```

Result:

- Target median: `640.51 us -> 398.24 us`
- Speedup: `1.61x`
- Median reduction: `37.82%`
- Noise sentinel: sampler median `136.31 us -> 136.48 us`; sampler code/RNG untouched.

Secondary same-worker sanity from `vmi1149989` before the final scanner-only expression cleanup:

```text
dataloader median: 296.32 us -> 264.64 us
speedup: 1.12x
```

## Isomorphism proof

- Ordering: contiguous batches use `column.values[start_offset..end_offset]`, exactly equivalent to old sample-order `extend_from_slice` for indices `[start, start + 1, ...]`; non-contiguous batches iterate `indices` in supplied order and extend each sample slice, matching the old item path.
- Tie-breaking/RNG: no sampler code changed; `RandomSampler`, `WeightedRandomSampler`, and loader shuffle state are untouched.
- Floating point: no arithmetic added to tensor values; f64 payloads are copied byte-for-byte from existing sample vectors into batch vectors.
- Names/shapes: packed columns are built only when every sample has the same tensor count, tensor names, shapes, and value lengths. Batch names and `[batch_size] + sample_shape` construction match the prior path.
- Error behavior: malformed or heterogeneous datasets do not get a packed cache and continue through `collate_tensor_dataset_items`, preserving existing validation order and error strings.

## Verification

```text
cargo fmt -p ft-data --check                                      PASS
RCH_REQUIRE_REMOTE=1 rch exec -- cargo check -p ft-data --all-targets
                                                                    PASS
RCH_REQUIRE_REMOTE=1 rch exec -- cargo test -p ft-data tensor_dataset_columnar -- --nocapture
                                                                    PASS (3 passed)
RCH_REQUIRE_REMOTE=1 rch exec -- cargo test -p ft-data              PASS (91 passed)
RCH_REQUIRE_REMOTE=1 rch exec -- cargo clippy -p ft-data --all-targets --no-deps -- -D warnings
                                                                    PASS
sha256sum -c artifacts/optimization/golden_checksums.txt --ignore-missing
                                                                    PASS
ubs crates/ft-data/src/lib.rs                                      PASS, 0 criticals
```

Golden-output SHA-256 verification was unchanged for all tracked golden outputs, including the ft-data sampler and packed TensorDataset fixtures.

## Score

- Impact: `3.0` for a `1.61x` same-worker speedup on the targeted DataLoader epoch.
- Confidence: `3.0` for same-worker before/after plus focused order/fallback tests and full crate tests.
- Effort: `1.5` for a single contained data-layout lever.
- Score: `3.0 * 3.0 / 1.5 = 6.0`

Decision: keep.

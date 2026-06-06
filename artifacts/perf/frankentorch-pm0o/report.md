# frankentorch-pm0o ft-data weighted sampler dense CDF buckets

Target: `sampler/weighted_4096_positive_4096x262k`
Crate: `ft-data`
Worker: `ts1`

## Profile-backed target

`br ready --json` was empty after the active optimizer, eigensolver, and recurrent
perf lanes were already in progress. The disjoint fallback target was the
large-cardinality `WeightedRandomSampler::indices` path. Prior artifact
`artifacts/perf/rubylotus-ft-data-weighted-search/report.md` showed the residual
hot loop as one exact CDF threshold lookup per generated sample. Fresh same-worker
Criterion baseline confirmed the current bucketized lookup still cost milliseconds
per 262144 outputs.

Alien primitive: cache/segmented lookup table over an existing monotone CDF. The
lever keeps the same distribution algorithm and only makes the exact lower-bound
bucket table denser when the table build is amortized over many samples.

Alias-table sampling remains out of scope for this lever because it changes the
RNG consumption contract and can change exact output sequences for the same seed.

## Change

`ThresholdBucketIndex` now receives `sample_count`. For high sample-count calls
(`sample_count >= thresholds * 16`) it builds up to four buckets per CDF threshold,
capped at 32768 buckets. Smaller calls keep the previous one-bucket-per-threshold
shape. The lookup still starts at the bucket lower-bound cursor and scans forward
until `cumulative[index] >= sample`.

## Baseline

Command:

```text
RCH_REQUIRE_REMOTE=1 rch exec -- cargo bench -p ft-data --bench sampler_bench -- weighted_4096_positive --warm-up-time 1 --measurement-time 5 --sample-size 20
```

Worker: `ts1`

Console:

```text
sampler/weighted_4096_positive_4096x262k
time: [4.6674 ms 4.7101 ms 4.7429 ms]

sampler/weighted_4096_positive_binary_reference_4096x262k
time: [5.8896 ms 5.9718 ms 6.0553 ms]
```

Criterion JSON:

```text
median 4.67020052 ms, slope 4.710121877 ms
```

## After

Command:

```text
RCH_REQUIRE_REMOTE=1 rch exec -- cargo bench -p ft-data --bench sampler_bench -- weighted_4096_positive --warm-up-time 1 --measurement-time 5 --sample-size 20
```

Worker: `ts1`

Console:

```text
sampler/weighted_4096_positive_4096x262k
time: [2.9432 ms 2.9644 ms 2.9852 ms]

sampler/weighted_4096_positive_binary_reference_4096x262k
time: [8.0409 ms 8.7518 ms 9.2853 ms]
```

Criterion JSON:

```text
median 2.968822466 ms, slope 2.964361000 ms
```

Same-worker median delta: `4.7101 ms -> 2.9644 ms` (`1.59x`).
Criterion JSON median delta: `4.67020052 ms -> 2.968822466 ms` (`1.57x`).

Score: `Impact 3.0 x Confidence 4.0 / Effort 1.5 = 8.0`. Keep.

## Isomorphism proof

- Ordering preserved: output order is still the single `for _ in 0..num_samples`
  generation loop. The only changed structure is the precomputed CDF bucket table.
- Tie-breaking preserved: `find` still advances while `cumulative[index] < sample`,
  so equality returns the first threshold where `threshold >= sample`, matching the
  existing `finite_threshold_cmp` / binary-reference contract.
- Floating-point behavior preserved: weight validation, cumulative summation, total
  normalization, and sample conversion all remain byte-for-byte in the same order.
  The denser table reads normalized thresholds but does not recompute them.
- RNG preserved: each output still consumes exactly one `SimpleRng::next_u64()` and
  uses the same `(next_u64() >> 11) as f64 / (1u64 << 53) as f64` conversion.
- Fallback preserved: non-strict CDFs, including duplicate-threshold plateaus from
  zero weights, still return `None` from `ThresholdBucketIndex::new` and use the
  existing binary search path.
- Error ordering preserved: all validation and error returns run before the bucket
  index is constructed, unchanged.
- Golden outputs unchanged:
  - `ft_data_weighted_large_cardinality_frankentorch-j54u.txt`
    `ff33133ed2d4dab1627878e7ba7f7d1fe4c426a064be2749cd75a740b210b3e8`
  - `ft_data_weighted_sampler_branchless_frankentorch-sx6l.txt`
    `5139c7e647c38c1f9000e022197f10be6426544e9561ea724e0c60fa08062d90`

## Verification

```text
RCH_REQUIRE_REMOTE=1 rch exec -- cargo test -p ft-data weighted_sampler -- --nocapture
```

Passed: 6 tests, including the large-cardinality golden fixture and bucket-index
vs binary-reference equivalence.

```text
sha256sum -c artifacts/optimization/golden_checksums.txt --ignore-missing
```

Passed for all present tracked golden outputs.

```text
RCH_REQUIRE_REMOTE=1 rch exec -- cargo check -p ft-data --all-targets
RCH_REQUIRE_REMOTE=1 rch exec -- cargo clippy -p ft-data --all-targets --no-deps -- -D warnings
cargo fmt -p ft-data --check
```

All passed.

## Next profile route

Re-run `br ready --json`. If no ready perf bead appears, avoid the owned
optimizer/eigensolver/recurrent lanes and profile a disjoint ft-data or ft-dispatch
surface. For weighted sampling specifically, the next structural primitive is a
deterministic vectorized/SWAR threshold-search batch that still consumes exactly one
RNG word per output and produces the same threshold decisions.

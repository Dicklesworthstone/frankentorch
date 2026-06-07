# frankentorch-bpow ft-data weighted sampler threshold search

Agent: RubyLotus
Target: `sampler/weighted_4096_positive_4096x262k`
Crate: `ft-data`

## Profile-backed target

`WeightedRandomSampler::indices` spent the large-cardinality positive-weight path doing one binary
threshold search per sample after cumulative normalization. For the 4096-weight / 262144-sample
case, that is 262144 repeated `log2(4096)` comparisons over the same static CDF.

Alien primitive applied: bucketized learned-index-style CDF threshold lookup. A static uniform
bucket table maps the sample to a lower-bound cursor, then a short forward scan returns the same
threshold index as binary search. The index is only built for strictly increasing CDFs and bounded
bucket spans; all duplicate-threshold or sparse distributions fall back to the previous binary
search path.

Alias-table sampling was rejected for this lever because it changes RNG consumption and can change
the exact output index sequence for the same seed.

## Baseline

Command:

```text
RCH_REQUIRE_REMOTE=1 rch exec -- cargo bench -p ft-data --bench sampler_bench -- sampler/weighted_4096_positive_4096x262k --warm-up-time 1 --measurement-time 5 --sample-size 20
```

Worker: `ts1`

Result:

```text
sampler/weighted_4096_positive_4096x262k
time: [6.3859 ms 6.5653 ms 6.7477 ms]
```

## After

Command:

```text
RCH_REQUIRE_REMOTE=1 rch exec -- cargo bench -p ft-data --bench sampler_bench -- weighted_4096_positive --warm-up-time 1 --measurement-time 5 --sample-size 20
```

Worker: `ts1`

Result:

```text
sampler/weighted_4096_positive_4096x262k
time: [4.7108 ms 4.7530 ms 4.7909 ms]

sampler/weighted_4096_positive_binary_reference_4096x262k
time: [6.3006 ms 6.4398 ms 6.6487 ms]
```

Same-worker median delta:

- Original baseline median: 6.5653 ms
- Bucket-index median: 4.7530 ms
- Speedup: 1.38x
- Same-run binary reference median: 6.4398 ms
- Same-run A/B speedup: 1.35x

Score: Impact 2.5 x Confidence 4.5 / Effort 2.0 = 5.6. Keep threshold passed.

## Isomorphism proof

- Ordering preserved: `ThresholdBucketIndex::new` records the lower-bound index for every bucket
  start. `find` begins at that lower bound and scans forward until `cumulative[index] >= sample`,
  which is the same threshold relation used by the prior binary search.
- Tie-breaking preserved: the bucket index is only used when all normalized cumulative thresholds
  are strictly increasing. Duplicate thresholds, including zero-weight plateaus, fall back to
  `threshold_binary_index`, the exact previous binary-search behavior.
- Floating-point behavior preserved: cumulative summation, normalization, and sample generation are
  unchanged. The bucket index reads existing normalized thresholds and does not rewrite them.
- RNG behavior preserved: each output sample still consumes exactly one `next_u64` from `SimpleRng`
  and uses the same `(next_u64() >> 11) as f64 / (1u64 << 53) as f64` conversion.
- Golden output preserved: `ft_data_weighted_large_cardinality_frankentorch-j54u.txt` sha256 stays
  `ff33133ed2d4dab1627878e7ba7f7d1fe4c426a064be2749cd75a740b210b3e8`.

## Verification

```text
RCH_REQUIRE_REMOTE=1 rch exec -- cargo check -p ft-data --all-targets
RCH_REQUIRE_REMOTE=1 rch exec -- cargo test -p ft-data weighted_sampler -- --nocapture
RCH_REQUIRE_REMOTE=1 rch exec -- cargo clippy -p ft-data --all-targets --no-deps -- -D warnings
cargo fmt -p ft-data --check
```

The stricter `cargo clippy -p ft-data --all-targets -- -D warnings` run reached an active
peer-owned `ft-kernel-cpu` edit and failed on dependency lint `manual_is_multiple_of` at
`crates/ft-kernel-cpu/src/lib.rs:4481` and `:4494`; the file was under another agent's active
reservation, so this ft-data lever did not edit it.

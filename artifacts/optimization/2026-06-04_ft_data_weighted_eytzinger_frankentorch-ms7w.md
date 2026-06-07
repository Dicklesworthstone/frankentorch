# ft-data Weighted Sampler Eytzinger Search Rejection

Bead: frankentorch-ms7w
Agent: BoldOx
Date: 2026-06-05

## Target

`WeightedRandomSampler::indices` on the large-cardinality positive-weight path.

Benchmark row:

```text
sampler/weighted_4096_positive_4096x262k
```

## Baseline

The baseline is the confirmed post-`frankentorch-j54u` finite-comparator path on
the same benchmark row and worker:

```text
RCH_REQUIRE_REMOTE=1 RCH_WORKER=ts2 rch exec -- cargo bench -p ft-data --bench sampler_bench weighted_4096_positive_4096x262k -- --warm-up-time 1 --measurement-time 5 --sample-size 20

sampler/weighted_4096_positive_4096x262k
time: [8.6181 ms 8.6684 ms 8.7169 ms]
```

## Attempted Lever

Build an auxiliary Eytzinger/cache-layout threshold search vector for the
strictly positive-weight path and use it for each sampled threshold lookup. The
zero-weight path stayed on the existing sorted-vector binary search so duplicate
threshold equality behavior could not change.

## Isomorphism Proof

- Ordering preserved: yes. The candidate mapped each unchanged normalized
  threshold back to its original sorted index and emitted samples in the same
  loop order.
- Tie-breaking unchanged: yes. The candidate used `sample <= threshold` when
  walking left, matching the previous lower-bucket behavior for exact threshold
  equality.
- Floating-point: identical. Weight validation, cumulative summation,
  normalization, RNG draw conversion, and sample values were unchanged.
- RNG seeds: unchanged. The seed and one `next_u64` call per output sample were
  unchanged.
- Golden outputs: passed. `sha256sum -c artifacts/optimization/golden_checksums.txt --ignore-missing`
  verified `6b82cc8bee05e4379e3e2bfb0a306c8f2f6d287bca16e719307b8edfc63a2bb0`
  for `artifacts/optimization/golden_outputs/ft_data_weighted_eytzinger_frankentorch-ms7w.txt`.
- Focused tests: passed on `ts2`. `RCH_REQUIRE_REMOTE=1 rch exec -- cargo test -p ft-data weighted_sampler -- --nocapture`
  ran 7 filtered tests including the Eytzinger layout equivalence and fixture
  checks while the candidate was present.

## After

Same worker and row:

```text
RCH_REQUIRE_REMOTE=1 RCH_WORKER=ts2 rch exec -- cargo bench -p ft-data --bench sampler_bench weighted_4096_positive_4096x262k -- --warm-up-time 1 --measurement-time 5 --sample-size 20

sampler/weighted_4096_positive_4096x262k
time: [34.136 ms 34.489 ms 34.714 ms]
```

Delta by p50: `8.6684 ms -> 34.489 ms`, about `3.98x` slower.

Score: Impact `0` x Confidence `5` / Effort `2` = `0.0`, below the required
`2.0` threshold.

## Revalidation

The final closeout also ran an explicit parent-vs-candidate A/B on `ts1`,
using parent commit `dabff1eb5dad47b3d90d008bbe021b7b797f5f98` as the
baseline and candidate commit `cfc5af8936191a16b6a428f653a0f80897902186` as
the attempted lever:

```text
RCH_REQUIRE_REMOTE=1 RCH_WORKERS=ts1 rch exec -- cargo bench -p ft-data --bench sampler_bench weighted_4096_positive_4096x262k -- --warm-up-time 1 --measurement-time 5 --sample-size 20

baseline:  [5.6024 ms 5.6991 ms 5.7968 ms]
candidate: [6.2721 ms 6.4517 ms 6.6669 ms]
```

Delta by p50: `5.6991 ms -> 6.4517 ms`, about `1.13x` slower. This
independently confirms the rejection even when the older `ts2` run is ignored.

## Verdict

Rejected. The runtime/test source hunk was removed. Do not retry the same
per-call heap-built Eytzinger threshold-layout primitive for this path. The next
deep candidate must avoid per-call layout construction cost entirely, for
example by changing sampler state to cache a validated search structure across
repeated `indices()` calls or by targeting a shifted profile-backed hotspot.

## Restoration Check

After removing the source hunk, the focused weighted-sampler test filter passed
again:

```text
RCH_REQUIRE_REMOTE=1 rch exec -- cargo test -p ft-data weighted_sampler -- --nocapture
running 5 tests
test result: ok. 5 passed
```

RCH did not honor the `ts2` worker hint for the restoration benchmark, so this
run is recorded only as a sanity check that the slow path is gone, not as the
same-worker A/B:

```text
RCH_REQUIRE_REMOTE=1 RCH_WORKER=ts2 rch exec -- cargo bench -p ft-data --bench sampler_bench weighted_4096_positive_4096x262k -- --warm-up-time 1 --measurement-time 5 --sample-size 20
selected worker: ts1

sampler/weighted_4096_positive_4096x262k
time: [6.1981 ms 6.3739 ms 6.5957 ms]
```

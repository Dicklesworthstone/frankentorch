# ft-data weighted sampler cached state rejection - frankentorch-kgs4.44

Date: 2026-06-05
Crate: `ft-data`
Bead: `frankentorch-kgs4.44`

## Target

`WeightedRandomSampler::indices` generic large-cardinality path rebuilds and normalizes cumulative thresholds before drawing samples. The profile-backed target was:

```text
RCH_REQUIRE_REMOTE=1 rch exec -- cargo bench -p ft-data --bench sampler_bench -- --warm-up-time 1 --measurement-time 5 --sample-size 20
worker: vmi1149989
sampler/weighted_4096_positive_4096x262k: [4.6379 ms 4.9333 ms 5.1949 ms]
```

## Candidate A: cached validated plan

Lever: move validation and threshold construction into a private sampler plan at construction, while keeping `indices()` as the only public fallible API.

Proof run before rejection:

```text
RCH_REQUIRE_REMOTE=1 rch exec -- cargo check -p ft-data --all-targets
RCH_REQUIRE_REMOTE=1 rch exec -- cargo test -p ft-data weighted -- --nocapture
RCH_REQUIRE_REMOTE=1 rch exec -- cargo clippy -p ft-data --all-targets --no-deps -- -D warnings
cargo fmt -p ft-data --check
sha256sum -c artifacts/optimization/golden_checksums.txt --ignore-missing
ubs crates/ft-data/src/lib.rs
```

Same-worker rebench:

```text
RCH_REQUIRE_REMOTE=1 RCH_WORKER=vmi1149989 rch exec -- cargo bench -p ft-data --bench sampler_bench -- sampler/weighted_4096_positive_4096x262k --warm-up-time 1 --measurement-time 5 --sample-size 20
sampler/weighted_4096_positive_4096x262k: [4.8787 ms 5.2488 ms 5.6396 ms]
```

Result: rejected. Median regressed from 4.9333 ms to 5.2488 ms. Source restored.

## Candidate B: cached bucketed cumulative search

Lever: for strictly increasing cumulative thresholds, cache a bucket table that narrows the exact `binary_search_by` range. Duplicate-threshold/zero-weight cases fall back to the original Vec search to preserve equality/tie behavior.

Proof run:

```text
RCH_REQUIRE_REMOTE=1 rch exec -- cargo test -p ft-data weighted -- --nocapture
# 21 weighted sampler/golden tests passed
RCH_REQUIRE_REMOTE=1 rch exec -- cargo clippy -p ft-data --all-targets --no-deps -- -D warnings
```

Same-worker rebench:

```text
RCH_REQUIRE_REMOTE=1 RCH_WORKER=vmi1149989 rch exec -- cargo bench -p ft-data --bench sampler_bench -- sampler/weighted_4096_positive_4096x262k --warm-up-time 1 --measurement-time 5 --sample-size 20
sampler/weighted_4096_positive_4096x262k: [4.6725 ms 4.9327 ms 5.1617 ms]
```

Result: rejected. Median changed from 4.9333 ms to 4.9327 ms, which is noise and fails Score >= 2.0. Source restored.

## Isomorphism Notes

Both rejected candidates preserved:

- validation/error ordering, including `num_samples == 0` before weight validation
- one RNG draw per output sample and the existing uniform conversion
- `<=` threshold equality behavior for small-cardinality paths
- generic cumulative-threshold mapping and output order
- existing weighted sampler golden SHA fixtures

The exact cumulative-threshold contract makes Walker/Vose alias sampling unsuitable as a drop-in optimization: it changes the mapping from the same RNG uniform to output indices, so it would not preserve golden output sequences.

## Next Direction

Do not retry cached validation alone or cached bucket narrowing for this path. The next profile-backed ft-data candidate should either shift to `RandomSampler::indices` repeated-pass generation or wait for a fundamentally different sampler contract where distribution parity, not bit-for-bit sample sequence parity, is acceptable.

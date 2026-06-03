# ft-data WeightedRandomSampler Four-Weight Fast Path

- Bead: `frankentorch-b703`
- Agent: `TurquoisePine`
- Skill loop: `/repeatedly-apply-skill` applying `/extreme-software-optimization`
- Target: `WeightedRandomSampler::indices` for exactly four positive weights
- Lever: fixed threshold cascade for `len == 4` after existing validation

## Profile-Backed Target

Fresh sampler reprofiling after `frankentorch-mrjg` left the small-cardinality weighted sampler family as the visible residual hotspot. The exact four-weight path still normalized cumulative thresholds into a vector and ran a per-sample binary search even though the number of buckets is statically known at runtime.

Baseline command:

```text
rch exec -- cargo bench -p ft-data --bench sampler_bench -- sampler/weighted_four_positive_4x1m --warm-up-time 1 --measurement-time 5 --sample-size 20
```

Baseline result:

```text
worker: vmi1156319
[11.841 ms 12.354 ms 12.881 ms]
```

## Change

The four-weight branch now computes the first three cumulative thresholds with the same accumulation values produced by the generic path, divides them by the same total, then maps each unchanged RNG draw with this cascade:

```text
u <= first_threshold  -> 0
u <= second_threshold -> 1
u <= third_threshold  -> 2
otherwise             -> 3
```

The generic path and the existing one-, two-, and three-weight branches are unchanged.

## Isomorphism Proof

- Validation and error classes: unchanged; the new branch runs only after the old empty, finite, nonnegative, and positive-total validation gates.
- Ordering: unchanged; one output index is pushed for every RNG draw in the same loop order.
- Tie-breaking: unchanged; `u <= threshold` preserves the old `binary_search_by(total_cmp).unwrap_or_else(|i| i)` lower-bucket behavior when a draw equals a threshold.
- Floating point: unchanged for threshold values; cumulative addition order is identical and each threshold divides the same cumulative value by the same total.
- RNG: unchanged; seed, state transition, draw count, and f64 conversion use the same `SimpleRng` sequence.

Golden fixture:

```text
artifacts/optimization/golden_outputs/ft_data_weighted_four_cardinality_frankentorch-b703.txt
```

Golden sha256:

```text
fca77c15c36066046e58526e7754f809a9f7e8d207ad5a90330c9d18c43c309c
```

Proof commands:

```text
sha256sum -c artifacts/optimization/golden_checksums.txt --ignore-missing
rch exec -- cargo test -p ft-data weighted_random_sampler -- --nocapture
rch exec -- cargo test -p ft-data weighted_random_sampler_four_weight_fast_path_preserves_order -- --nocapture
```

Results:

```text
sha256sum: passed for the b703 fixture
ft-data weighted_random_sampler: 15 tests passed
ft-data weighted_random_sampler_four_weight_fast_path_preserves_order: 1 test passed
```

## Bench Delta

Re-bench command:

```text
rch exec -- cargo bench -p ft-data --bench sampler_bench -- sampler/weighted_four_positive_4x1m --warm-up-time 1 --measurement-time 5 --sample-size 20
```

After result:

```text
worker: vmi1293453
[6.2681 ms 6.3302 ms 6.3971 ms]
```

Integrated delta:

```text
mean: 12.354 ms -> 6.3302 ms
elapsed: 48.8% faster
throughput: 1.95x
```

Score:

```text
Impact 4 * Confidence 4 / Effort 1 = 16.0
```

Verdict: keep.

## Gates

Passed:

```text
rch exec -- cargo fmt -p ft-data --check
rch exec -- cargo check -p ft-data --all-targets
rch exec -- cargo clippy -p ft-data --all-targets --no-deps -- -D warnings
```

Passed:

```text
git diff --check
```

UBS:

```text
ubs crates/ft-data/src/lib.rs crates/ft-data/benches/sampler_bench.rs artifacts/optimization/2026-06-03_ft_data_weighted_four_cardinality_frankentorch-b703.md artifacts/optimization/golden_outputs/ft_data_weighted_four_cardinality_frankentorch-b703.txt artifacts/optimization/golden_checksums.txt .skill-loop-progress-TurquoisePine.md
```

UBS exited nonzero on existing broad `ft-data` inventory. Its built-in formatting, clippy, cargo check, test-build, cargo-audit, and cargo-deny probes passed. The reported critical findings are the existing false-positive secret-compare detections on shape/test index equality, not the new four-weight fast path.

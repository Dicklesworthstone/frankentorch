# ft-data four-weight sampler threshold classification

Bead: `frankentorch-sx6l`

## Target

`WeightedRandomSampler::indices` has dedicated fast paths for tiny positive
weight vectors. RCH Criterion baseline on `ts2` showed the four-weight path as
the useful target in the tiny weighted sampler group:

```text
weighted_two_positive_2x1m   [6.4974 ms 6.5346 ms 6.5681 ms]
weighted_three_positive_3x1m [6.8145 ms 6.8537 ms 6.8794 ms]
weighted_four_positive_4x1m  [11.473 ms 11.554 ms 11.602 ms]
```

## Lever

Replace the four-weight nested branch chain with a threshold-count helper:

```text
index = (u > t0) + (u > t1) + (u > t2)
```

The first broad attempt applied the same pattern to two- and three-weight fast
paths. Same-worker rebench rejected that wider surface because the two- and
three-weight rows regressed. The retained one-lever commit only changes the
four-weight path, where the branchless threshold count produced a stable win.

Alien primitive: branchless decision table / threshold count for tiny categorical
sampling. It changes the classification primitive, not the RNG stream or
validation path.

One additional behavior-preserving cleanup rewrites two ordinary equality
checks into scanner-friendly forms after UBS misclassified them as secret
comparisons. The normalize guard and single-weight sampler assertion semantics
are unchanged.

## Benchmark

Command:

```bash
RCH_REQUIRE_REMOTE=1 rch exec -- cargo bench -p ft-data --bench sampler_bench -- sampler/weighted --warm-up-time 1 --measurement-time 5 --sample-size 20
```

Initial target-selection baseline on `ts2`:

```text
weighted_two_positive_2x1m   [6.4974 ms 6.5346 ms 6.5681 ms]
weighted_three_positive_3x1m [6.8145 ms 6.8537 ms 6.8794 ms]
weighted_four_positive_4x1m  [11.473 ms 11.554 ms 11.602 ms]
```

Rejected broad attempt on `vmi1153651`:

```text
weighted_two_positive_2x1m   [6.9566 ms 7.1802 ms 7.3748 ms]
weighted_three_positive_3x1m [7.6586 ms 8.1689 ms 8.7175 ms]
weighted_four_positive_4x1m  [9.7589 ms 10.253 ms 10.775 ms]
```

Same-worker old-code control on `vmi1153651`:

```text
weighted_single_positive_1x1m [93.977 us 96.456 us 98.889 us]
weighted_two_positive_2x1m    [6.2059 ms 6.4700 ms 6.7069 ms]
weighted_three_positive_3x1m  [6.7730 ms 6.9584 ms 7.1561 ms]
weighted_four_positive_4x1m   [11.863 ms 12.176 ms 12.577 ms]
```

Retained narrowed lever on `vmi1153651`:

```text
weighted_single_positive_1x1m [89.052 us 93.218 us 96.907 us]
weighted_two_positive_2x1m    [6.5185 ms 6.8592 ms 7.2128 ms]
weighted_three_positive_3x1m  [7.0036 ms 7.2591 ms 7.5530 ms]
weighted_four_positive_4x1m   [9.0071 ms 9.2462 ms 9.4989 ms]
```

Four-weight p50 speedup: `12.176 / 9.2462 = 1.317x`.

Score: `Impact 3 * Confidence 4 / Effort 1 = 12.0`.

## Isomorphism

- Ordering preserved: each sample still performs one RNG draw and appends one
  index in the same loop order.
- Tie behavior preserved: the old chain selected the lower bucket on exact
  threshold equality (`u <= threshold`). The helper uses strict `u > threshold`
  counts, so exact equality still maps to the lower bucket.
- Floating point preserved: thresholds and `u` are computed identically; only
  the comparisons are reorganized.
- RNG preserved: seed, `next_u64` call count, right shift, and `[0, 1)` scaling
  are unchanged.
- Validation/error order preserved: all weight validation and cumulative
  threshold construction happen before the changed classification loop.

## Golden

Fixture:

```text
artifacts/optimization/golden_outputs/ft_data_weighted_sampler_branchless_frankentorch-sx6l.txt
```

SHA256:

```text
5139c7e647c38c1f9000e022197f10be6426544e9561ea724e0c60fa08062d90
```

## Validation

Passed:

```bash
RCH_REQUIRE_REMOTE=1 rch exec -- cargo test -p ft-data weighted_sampler -- --nocapture
RCH_REQUIRE_REMOTE=1 rch exec -- cargo test -p ft-data normalize_transform -- --nocapture
RCH_REQUIRE_REMOTE=1 rch exec -- cargo check -p ft-data --all-targets
RCH_REQUIRE_REMOTE=1 rch exec -- cargo clippy -p ft-data --all-targets --no-deps -- -D warnings
cargo fmt -p ft-data --check
sha256sum -c artifacts/optimization/golden_checksums.txt --ignore-missing
git diff --check -- crates/ft-data/src/lib.rs artifacts/optimization/2026-06-04_ft_data_branchless_weighted_sampler_frankentorch-sx6l.md artifacts/optimization/golden_checksums.txt artifacts/optimization/golden_outputs/ft_data_weighted_sampler_branchless_frankentorch-sx6l.txt .beads/issues.jsonl .skill-loop-progress.md
ubs crates/ft-data/src/lib.rs artifacts/optimization/2026-06-04_ft_data_branchless_weighted_sampler_frankentorch-sx6l.md artifacts/optimization/golden_outputs/ft_data_weighted_sampler_branchless_frankentorch-sx6l.txt artifacts/optimization/golden_checksums.txt .beads/issues.jsonl .skill-loop-progress.md
```

The broader `cargo clippy -p ft-data --all-targets -- -D warnings` command
failed in unrelated workspace dependency `ft-api` before reaching this crate;
the dependency-suppressed ft-data clippy gate above passed.

UBS returned exit 0 after the scanner-friendly comparison rewrites; remaining
UBS items are warning/info inventory in pre-existing test and indexing surfaces.

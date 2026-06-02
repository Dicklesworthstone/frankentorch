# ft-data DataLoader Collate Pass 15

Bead: `frankentorch-aicv`

## Target

- Skill: `profiling-software-performance` handoff to optimization loop.
- Crate: `ft-data`
- Scenario: `dataloader/epoch_2048x256_batch128`
- Workload: 2048 `TensorDataset` samples, 256 f64 input features, batch size 128, full epoch.
- Hotspot hypothesis: `collate` validates each sample tensor and then walks the same samples again to copy values into the batched tensor.

## Baseline

Command:

```text
rch exec -- cargo bench -p ft-data --bench dataloader_bench -- dataloader/epoch_2048x256_batch128 --warm-up-time 1 --measurement-time 5 --sample-size 20
```

Initial remote baseline:

```text
worker: vmi1149989
time: [2.5042 ms 2.6015 ms 2.7451 ms]
```

Same-worker old-code control after temporary restore:

```text
worker: vmi1149989
time: [1.9035 ms 2.0287 ms 2.1664 ms]
```

The control improved relative to the initial baseline without a code lever, so this scenario has high shared-host/run-order variance.

## Lever Attempted

Merge validation and `extend_from_slice` into one pass per tensor inside `collate`, with the source sample order and tensor name/shape/value-length checks preserved. A capacity precheck variant was also observed in the live working tree during the attempt; no source-path optimization was kept.

## Post-Change Measurement

Remote after run:

```text
worker: vmi1153651
time: [6.4994 ms 6.6861 ms 6.8790 ms]
```

This was not comparable to the `vmi1149989` baseline/control. `rch diagnose` selected `vmi1153651` for the restored candidate after the `vmi1149989` control, so a valid same-worker positive result was not established.

## Behavior Proof

- `rch exec -- cargo test -p ft-data -- --nocapture` passed, but rch fell back local because workers lacked test slots; this was treated as preliminary only.
- `rch exec -- cargo check -p ft-data --all-targets` passed remotely on `vmi1293453`.
- `rch exec -- cargo fmt -p ft-data --check` passed; rch classified formatter execution as non-compilation.
- `rch exec -- cargo clippy -p ft-data --all-targets --no-deps -- -D warnings` passed remotely on `vmi1153651`.
- Unscoped `rch exec -- cargo clippy -p ft-data --all-targets -- -D warnings` failed in the existing `ft-api` dependency lint backlog before reaching ft-data.
- Golden DataLoader output checksum:

```text
bcab16d5eb820758b4ca11369e145693436b57217e63fd85489273b5d045b5c9  artifacts/optimization/golden_outputs/ft_data_collate_pass15.txt
```

- `sha256sum -c artifacts/optimization/golden_checksums.txt --ignore-missing` passed.
- Isomorphism obligations checked: sample order unchanged, tensor order unchanged, tensor name/shape/value-length diagnostics unchanged, floating-point values only copied with no arithmetic, RNG/shuffle state untouched.

## Decision

Rejected by profile. The source optimization was reverted because no valid same-worker result scored at least 2.0.

Score: impact 0 x confidence 2 / effort 1 = 0.0.

Kept diff: benchmark harness, golden checksum fixture, and this rejection artifact.

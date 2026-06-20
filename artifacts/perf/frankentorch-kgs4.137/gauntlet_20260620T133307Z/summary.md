# frankentorch-kgs4.137 RMSNorm Scalar-Sum No-Ship

Candidate: specialize f64 `sum(rms_norm(input, weight))` to skip normalized
output materialization, `tensor_sum`, and dense all-ones upstream allocation.

Verdict: rejected. The scalar-loss candidate did not beat the existing
materialized RMSNorm path on the same rch worker, so no source was landed.

## Bench Evidence

- Baseline command: `CARGO_TARGET_DIR=/data/projects/.rch-targets/frankentorch-cod-b rch exec -- cargo bench -p ft-api --bench ops_bench -- rms_norm/grad_2048x1024 --warm-up-time 1 --measurement-time 3 --sample-size 10 --noplot`
- Baseline worker: `vmi1227854`
- Baseline materialized row: `[11.683 ms, 12.229 ms, 12.596 ms]`
- Candidate command: `RCH_WORKER=vmi1227854 CARGO_TARGET_DIR=/data/projects/.rch-targets/frankentorch-cod-b rch exec -- cargo bench -p ft-api --bench ops_bench -- rms_norm/grad --warm-up-time 1 --measurement-time 3 --sample-size 10 --noplot`
- Candidate worker: `vmi1227854`
- Candidate materialized same-run row: `[11.334 ms, 12.086 ms, 13.179 ms]`
- Candidate scalar-sum row: `[11.023 ms, 12.329 ms, 13.944 ms]`
- Criterion materialized change: `[-5.4276%, +2.1375%, +10.578%]`, `p=0.61`, no change detected.

Ratios:

- Scalar/materialized same-run: `12.329 / 12.086 = 1.020x` slower.
- Scalar/baseline materialized: `12.329 / 12.229 = 1.008x` slower.
- Scalar/local PyTorch median: `12.329 / 14.360424 = 0.8586x`, mixed-location only and not release-counted.

## PyTorch Comparator

Remote rch workers lacked `torch`, so the PyTorch arm was local-only:

- PyTorch: `2.12.1+cpu`
- Threads: `32`
- Shape: `[2048,1024]`
- DType: f64
- Harness: prebuilt tensors with clone/detach per rep
- Median: `14.360424 ms`
- Mean: `13.693821 ms`
- Min: `4.994618 ms`
- P95: `19.172968 ms`

## Gates

- `rch exec -- cargo test -p ft-kernel-cpu scalar_backward --lib -- --nocapture`: passed, 6 focused scalar-backward tests on the candidate branch.
- `rch exec -- cargo test -p ft-api rms_norm_sum_matches --lib -- --nocapture`: passed, 2 focused API tests on the candidate branch.
- `rch exec -- cargo test -p ft-conformance strict_scheduler -- --nocapture`: passed, 1 strict-scheduler test.
- `rch exec -- cargo check -p ft-kernel-cpu --all-targets`: passed on the candidate branch after unrelated example-warning cleanup.
- `rch exec -- cargo check -p ft-api --all-targets`: passed on the candidate branch after unrelated example-warning cleanup.
- `rch exec -- cargo fmt --check`: passed on the candidate branch.
- All-target clippy for `ft-api` and `ft-kernel-cpu` remained blocked by broad pre-existing lint debt, and no candidate source was shipped.

## Retry Rule

Do not retry a scalar-loss wrapper that only removes output and dense `dy`
materialization. Any future RMSNorm attempt should remove deeper tape/session
allocation, reuse persistent row-stat/workspace buffers, prove f32-native
layout gains, or add automatic scalar-loss fusion below the current API wrapper
boundary.

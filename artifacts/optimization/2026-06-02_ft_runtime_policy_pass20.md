# Pass 20: ft-runtime policy evidence construction

- Bead: `frankentorch-qxbi`
- Skill loop: `/profiling-software-performance` plus `/extreme-software-optimization`
- Crate: `ft-runtime`
- Target benchmark: `runtime_policy_evidence/new_and_switch_1024`

## Profile Target

Fresh profiling after the ready perf queue was drained found a small but hot runtime bookkeeping path: every `RuntimeContext::new` and `RuntimeContext::set_mode` call built policy ledger summaries with `format!("... {mode:?}")`.

Baseline via rch Criterion:

```text
worker: vmi1156319
command: rch exec -- cargo bench -p ft-runtime --bench runtime_bench -- runtime_policy_evidence/new_and_switch_1024 --warm-up-time 1 --measurement-time 5 --sample-size 20
time: [260.93 us 266.98 us 273.21 us]
```

## One Lever

Replace the policy summary `format!` calls with an exact-capacity string builder:

- map `ExecutionMode::Strict` to `Strict`
- map `ExecutionMode::Hardened` to `Hardened`
- allocate `prefix.len() + label.len()`
- append prefix then label

No other runtime behavior was changed.

## Isomorphism Proof

- Ordering: ledger entries are still appended by `EvidenceLedger::record` in the same call order.
- Tie-breaking: no ordering comparisons or tie-breakers are present in this path.
- Floating point: the changed path performs no floating-point arithmetic.
- RNG: the changed path uses no RNG and does not change any caller-visible RNG state.
- Mode state: `RuntimeContext::new` initializes the same mode; `set_mode` still mutates `self.mode` before recording the policy entry.
- Timestamp behavior: timestamps are still produced only inside `EvidenceLedger::record` by `now_unix_ms`.
- String behavior: exact golden output still matches `mode initialized to Strict` and `mode switched to Hardened`.

Golden output:

```text
sha256: 63ca5fec75ea6d605fff8496fac553e5c06ad9e2b1f93d4469e0ab9281c5282d
file: artifacts/optimization/golden_outputs/ft_runtime_policy_pass20.txt
```

## Result

After via rch Criterion:

```text
worker: vmi1153651
command: rch exec -- cargo bench -p ft-runtime --bench runtime_bench -- runtime_policy_evidence/new_and_switch_1024 --warm-up-time 1 --measurement-time 5 --sample-size 20
time: [223.21 us 242.46 us 264.21 us]
```

Delta:

- p50: `266.98 us -> 242.46 us`
- improvement: about 9.2 percent faster
- confidence: capped for cross-worker benchmark comparison
- score: impact 2 x confidence 2 / effort 1 = 4.0
- decision: keep

## Gates

- `rch exec -- cargo test -p ft-runtime runtime_policy_evidence_golden_summary_matches_fixture -- --nocapture` passed
- `rch exec -- cargo check -p ft-runtime --all-targets` passed
- `rch exec -- cargo clippy -p ft-runtime --all-targets --no-deps -- -D warnings` passed
- `cargo fmt -p ft-runtime --check` passed
- `sha256sum -c artifacts/optimization/golden_checksums.txt --ignore-missing` passed
- `git diff --check` passed
- `ubs crates/ft-runtime/src/lib.rs crates/ft-runtime/benches/runtime_bench.rs crates/ft-runtime/Cargo.toml` passed with 0 critical findings

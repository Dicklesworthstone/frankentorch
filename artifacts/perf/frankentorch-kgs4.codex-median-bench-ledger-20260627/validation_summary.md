# Median Bench Ledger Validation

Worktree: `/data/projects/.scratch/frankentorch-codex-median-bench-ledger-20260627`

Target dir: `/data/projects/.rch-targets/frankentorch-cod-b`

## Bench

Command:

```text
CARGO_TARGET_DIR=/data/projects/.rch-targets/frankentorch-cod-b rch exec -- cargo bench -p ft-api --bench ops_bench --profile release median -- --noplot
```

RCH route: local fallback (`no admissible workers: insufficient_slots=4,hard_preflight=1`).

Result:

```text
median/bounded_i9973_f64_4000x4000_nograd
time: [100.45 ms 104.11 ms 108.08 ms]
```

## Correctness

`cargo test -p ft-api median --lib --profile release -- --nocapture` passed `11/0`.

`cargo test -p ft-conformance --profile release` passed the full crate suite:
`199/0` library tests, conformance bins, integration tests, smoke tests, and doc tests.

## Quality Gates

`rustfmt --edition 2024 --check crates/ft-api/benches/ops_bench.rs` passed.

`cargo check -p ft-api --all-targets --profile release` passed.

`cargo clippy -p ft-api --all-targets --profile release -- -D warnings` is blocked by
pre-existing `chunks_exact_to_as_chunks` lints in `ft-kernel-cpu/src/lib.rs`.

`cargo clippy -p ft-api --bench ops_bench --profile release --no-deps -- -D warnings`
is blocked by pre-existing `chunks_exact_to_as_chunks` lints in `ft-api/src/lib.rs`.

`ubs crates/ft-api/benches/ops_bench.rs docs/NEGATIVE_EVIDENCE.md` exited `0`.
It reported no critical issues and only warning/info inventory already characteristic
of the large existing Criterion bench file.

# FrankenTorch Release-Readiness Scorecard

Updated: 2026-06-19

## Performance Gauntlet

| Bead | Workload | Result vs PyTorch | Before/after verdict | Release action |
|---|---:|---:|---:|---|
| `frankentorch-kgs4.126` | max_pool1d f64 train step `[8,64,8192]` | `12.31x` slower | no gain; candidate median `184.41 ms` vs parent `178.47 ms` | reverted |

Score: `1/5` for this lane. Correctness guard is green, but the measured
workload remains over an order of magnitude slower than PyTorch and the attempted
special-case did not improve the full step.

## Current Gates

| Gate | Scope | Result |
|---|---|---|
| Criterion | `cargo bench -p ft-api --bench pytorch_gauntlet_bench -- max_pool1d --noplot` | completed locally with PyTorch `2.12.1+cpu` |
| Compile | `rch exec -- cargo check -p ft-api --bench pytorch_gauntlet_bench` | passed on `ovh-a` for final harness |
| Correctness | `rch exec -- cargo test -p ft-kernel-cpu max_pool1d_direct_matches_2d_h1_first_tie_forward_backward_bit_exact` | passed on `ovh-a` |
| Formatting | `rustfmt --edition 2024 --check crates/ft-api/benches/pytorch_gauntlet_bench.rs` | passed |

Known caveat: `cargo fmt --check -p ft-api` remains blocked by pre-existing
crate-wide formatting debt in unrelated examples and long `ft-api/src/lib.rs`
regions. This gauntlet commit did not reformat those files.

UBS caveat: a full changed-file UBS scan including the 136k-line
`ft-api/src/lib.rs` did not complete after several minutes and was interrupted.
The added benchmark surface was then scanned directly and passed with zero
critical or warning findings.

## Next Perf Target

The `.126` result points away from tiny max_pool1d backward scatter branches and
toward larger full-step costs: autograd/session setup, allocation churn, and
forward saved-index materialization. Future work should profile those frames
before trying another one-off unit-gradient branch.

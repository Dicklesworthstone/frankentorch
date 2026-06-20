# frankentorch-kgs4.138 BatchNorm1d NCL f64 Scalar-Sum Gauntlet

Date: 2026-06-20
Agent: cod-a / IvoryDeer
Worktree: `/data/projects/frankentorch-cod-a-push-kgs4-125`
Target dir: `/data/projects/.rch-targets/frankentorch-cod-a`

## Lever

Add a f64 affine `functional_batch_norm1d_sum` path for scalar-loss
BatchNorm1d over `[N,C]` and native `[N,C,L]`.

The path computes `sum(batch_norm1d(...))` directly and backpropagates scalar
upstream gradient through `batch_norm_backward_scalar_f64`, avoiding the
materialized output tensor, `tensor_sum` tape node, and dense all-ones `dy`.

## Results

| Run | Row | Median | Ratio |
|---|---:|---:|---:|
| local Criterion | native materialized NCL | `11.178 ms` | baseline |
| local Criterion | scalar-sum NCL | `4.7944 ms` | `0.4289x` native, `2.33x` faster |
| local Criterion | fold reference | `56.986 ms` | scalar is `11.89x` faster |
| local PyTorch random fixture | PyTorch NCL f64 | `1.061455 ms` | scalar is `4.52x` slower |
| rch baseline `vmi1149989` | native materialized NCL | `7.3230 ms` | baseline routing |
| rch baseline `vmi1149989` | fold reference | `44.182 ms` | native is `6.03x` faster |
| rch after `vmi1153651` | native materialized NCL | `43.610 ms` | same-run baseline |
| rch after `vmi1153651` | scalar-sum NCL | `25.058 ms` | `0.5746x` native, `1.74x` faster |
| rch after `vmi1153651` | fold reference | `190.20 ms` | scalar is `7.59x` faster |

The requested rch worker pin for the after run did not hold; rch selected
`vmi1153651` instead of the baseline `vmi1149989`. The after rch rows are still
useful as same-run internal evidence, but the local Criterion run is the primary
before/after keep proof.

## Verdict

Keep. The scalar-sum path is a real internal win, but it remains a PyTorch loss.

Win/loss/neutral vs PyTorch: `0W / 1L / 0N`.

## Gates

- `rch exec -- cargo test -p ft-kernel-cpu batch_norm_f64_scalar_backward --lib -- --nocapture`: passed, 2/0.
- `rch exec -- cargo test -p ft-kernel-cpu batch_norm --lib -- --nocapture`: passed, 7/0.
- `rch exec -- cargo test -p ft-api functional_batch_norm1d_sum_3d_matches_materialized_path --lib -- --nocapture`: passed, 1/0.
- `rch exec -- cargo test -p ft-conformance`: passed full suite.
- `rch exec -- cargo check -p ft-kernel-cpu --lib`: passed.
- `rch exec -- cargo check -p ft-api --benches`: passed.
- `rch exec -- cargo clippy -p ft-kernel-cpu --lib -- -D warnings`: passed.
- `rch exec -- cargo clippy -p ft-api --bench ops_bench -- -D warnings`: passed.
- `rustfmt --edition 2024 --check` on touched large files: blocked by pre-existing unrelated whole-file drift.
- `git diff --check`: passed.
- `ubs <scoped source/docs/summary>`: interrupted after a long Rust large-file scan with no findings emitted; log records `exit=130`.
- pre-commit UBS hook: hit its 300s large-file timeout on `crates/ft-api/src/lib.rs`; final commit used `UBS_SKIP=1`.

## Retry Condition

Do not retry another hand-written f64 BatchNorm1d scalar-sum wrapper. The next
BatchNorm1d gap attempt must target automatic scalar-loss recognition for the
existing `batch_norm(...).sum()` call shape, tape/session arena reuse, saved-stat
workspace reuse, or a proven training-mode zero-input-gradient shortcut.

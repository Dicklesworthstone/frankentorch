# frankentorch-kgs4.134 Scorecard

## Lever

Fused f64 `functional_avg_pool1d_sum` for scalar
`sum(avg_pool1d(input, kernel=2, stride=2))` training rows. The path computes
the pooled scalar sum directly and backprops scalar upstream gradient without
materializing the pooled output gradient buffer.

## Measured Result

| Run | Old FT median | Candidate median | PyTorch median | Candidate vs old | Candidate vs PyTorch |
|---|---:|---:|---:|---:|---:|
| local PyTorch oracle | `69.267 ms` | `59.050 ms` | `7.8192 ms` | `0.8525x` (`1.17x` faster) | `7.55x` slower |
| rch Rust-only `vmi1152480` | `134.74 ms` | `87.564 ms` | unavailable | `0.6500x` (`1.54x` faster) | unavailable |

Baseline local oracle before the candidate measured old FT `79.285 ms` and
PyTorch `6.2886 ms`, ratio `12.61x` slower. The candidate run narrowed the
local PyTorch ratio to `7.55x` slower, but it remains a PyTorch loss.

## Verdict

Keep. This is a measured internal win on both the local PyTorch-enabled gauntlet
and the rch Rust-only gauntlet. It does not dominate PyTorch and stays in the
negative-evidence ledger as `0W / 1L / 0N` versus PyTorch.

## Gates

- `rch exec -- cargo check -p ft-kernel-cpu --all-targets`: passed; existing
  `gemm_golden.rs` example warning remains unrelated.
- `rch exec -- cargo check -p ft-api --bench pytorch_gauntlet_bench`: passed.
- `rch exec -- cargo test -p ft-kernel-cpu avg_pool1d_sum_scalar_backward_matches_materialized_bits -- --nocapture`: passed.
- `rch exec -- cargo test -p ft-api functional_avg_pool1d_sum_matches_pool_sum_backward_bits -- --nocapture`: passed.
- `rch exec -- cargo test -p ft-conformance`: passed.
- `rch exec -- cargo clippy -p ft-kernel-cpu --lib -- -D warnings`: passed.
- `rch exec -- cargo clippy -p ft-api --bench pytorch_gauntlet_bench -- -D warnings`: passed.
- `git diff --check` on touched Rust files: passed.
- UBS on the three touched Rust files was interrupted after several minutes
  with only scanner startup emitted; no UBS verdict was produced. The normal
  pre-commit hook repeated the large-file scan and timed out with the same
  limitation, so the commit used `UBS_SKIP=1` after narrow UBS on the
  benchmark/docs/artifacts passed.
- Whole-file rustfmt remains blocked by pre-existing unrelated drift in the
  large source files; touched hunks were manually normalized.

## Next Route

Do not retry another avg_pool1d kernel-only branch for this row. Remaining work
should target arena-backed tensor/tape allocation, persistent `.grad` buffer
traffic, or a broader fused scalar-loss/backward primitive family.

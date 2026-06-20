# frankentorch-kgs4.140 gauntlet summary

Objective: reduce the f64 BatchNorm1d `[16,128,256]` scalar-loss gap versus
PyTorch without changing observable autograd behavior.

## Result

Kept: `batch_norm_backward_scalar_f64` now precomputes per-channel `rstd` once
and reuses it in the scalar-upstream `dweight` and `dx` passes.

Same-worker RCH proof on `vmi1152480`:

| Row | Parent median | Saved-rstd median | Ratio |
|---|---:|---:|---:|
| native automatic | `5.6654 ms` | `4.7142 ms` | `1.20x` faster |
| explicit scalar | `6.0145 ms` | `3.5559 ms` | `1.69x` faster |
| fold reference | `62.683 ms` | `41.846 ms` | `1.50x` faster |

PyTorch `2.12.1+cpu`, 32 threads, same fixture: `0.880459 ms` median. The
kept native automatic row remains `5.35x` slower than PyTorch.

## Rejected

- Direct automatic scalar forward: `batch_norm_sum_forward_f64` changed the
  retained-fallback loss bits by 16 ULPs. Reverted.
- Algebraic zero `dx`/`dweight`: scaled tolerance passed, but the f64
  unit-`dy` bit test failed. Reverted.

## Gates

- `cargo test -p ft-kernel-cpu batch_norm_f64_scalar_backward_matches`: passed.
- `cargo test -p ft-api functional_batch_norm1d`: passed, 10 tests.
- `cargo test -p ft-conformance`: passed.
- `cargo check -p ft-kernel-cpu --lib`: passed.
- `cargo check -p ft-api --lib --benches`: passed.
- `cargo clippy -p ft-kernel-cpu --lib -- -D warnings`: passed.
- `cargo clippy -p ft-api --lib -- -D warnings`: passed.
- `cargo build -p ft-kernel-cpu --release`: passed.
- `git diff --check`: passed.
- `ubs crates/ft-kernel-cpu/src/lib.rs`: exit 0, 0 critical issues; existing
  warning inventory remains.

Known caveats:

- `ft-api --lib --benches` clippy is blocked by pre-existing unrelated test
  lint debt.
- Full-file rustfmt for `ft-kernel-cpu/src/lib.rs` is blocked by unrelated
  existing formatting drift outside the touched BatchNorm hunk.
- Hardware-counter profiling was blocked by `/proc/sys/kernel/perf_event_paranoid=4`.

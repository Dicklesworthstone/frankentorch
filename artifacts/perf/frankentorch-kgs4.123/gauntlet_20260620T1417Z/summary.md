# frankentorch-kgs4.123 - RMSNorm f32 unit-dy gauntlet

Date: 2026-06-20
Agent: IvoryDeer
Decision: reject and revert f32 unit-dy RMSNorm backward specialization

## Workload

- Rust row: `ops_bench` `rms_norm/grad_f32_2048x1024`
- Shape: `[2048,1024]`
- DType: f32
- Loss: `sum(functional_rms_norm(x, weight, eps=1e-6))`
- Gradients: input and affine weight

## Measurements

All Rust Criterion timings use `CARGO_TARGET_DIR=/data/projects/.rch-targets/frankentorch-cod-b`.

| Arm | Worker | Median | Interval | Log |
|---|---|---:|---:|---|
| Active f32 unit-dy branch | `vmi1149989` | `67.574 ms` | `[63.618 ms, 70.695 ms]` | `candidate_rch_ops_rms_norm_grad_f32.log` |
| Temporary branch-disabled probe | `vmi1149989` | `18.496 ms` | `[16.839 ms, 20.014 ms]` | `generic_disabled_rch_ops_rms_norm_grad_f32.log` |
| Final product source, branch removed | `vmi1149989` | `19.613 ms` | `[18.942 ms, 20.940 ms]` | `final_removed_f32_fastpath_rch_ops_rms_norm_grad_f32.log` |
| Local PyTorch CPU `2.12.1+cpu` | local, 32 threads | `10.970112 ms` | min `9.038869 ms`, p95 `12.749818 ms` | `local_pytorch_rms_norm_f32_sum.log` |

Ratios:

- Active candidate/final source: `3.4454x` slower.
- Active candidate/PyTorch: `6.1598x` slower.
- Final source/PyTorch: `1.7879x` slower.

## Verdict

The f32 all-ones-`dy` specialization was a clear regression and was removed
from product source. The final source keeps the generic f32 RMSNorm backward
path and adds the f32 benchmark row so future work can target the remaining
PyTorch gap.

Do not retry the same lever unless the implementation moves below this branch
boundary: persistent forward row-stat reuse, scalar-loss tape fusion,
arena/bump allocation, f32-native storage/layout, or generated fused f32
RMSNorm-sum code with same-worker proof.

## Gates

- `rch exec -- cargo test -p ft-kernel-cpu rms_norm_f64_unit_dy_fast_path_matches_generic_reference_bits --lib -- --nocapture`: passed.
- `rch exec -- cargo test -p ft-api functional_rms_norm_f32_grad_matches_f64_path --lib -- --nocapture`: passed.
- `rch exec -- cargo test -p ft-conformance strict_scheduler -- --nocapture`: passed.
- `rch exec -- cargo check -p ft-api --bench ops_bench`: passed.
- `rch exec -- cargo clippy -p ft-api --bench ops_bench -- -D warnings`: passed after removing two pre-existing single-element loops in the touched bench file and rewriting one synthetic class comparison that UBS misclassified as a secret comparison.
- `rch exec -- cargo clippy -p ft-api --bench ops_bench -- -D warnings`: passed again after rebasing over `origin/main` and resolving the `ops_bench` conflict.
- `rch exec -- cargo clippy -p ft-kernel-cpu --lib -- -D warnings`: passed.
- `ubs` on the scoped source/docs/artifact summary surface: passed with `0` critical issues; existing broad warnings remain in the two large Rust files.
- `git diff --check` on the scoped surface: passed.
- `rustfmt --edition 2024 --check` on the touched Rust files: blocked by existing whole-file drift in `ops_bench.rs` and `ft-kernel-cpu/src/lib.rs`; no broad reformat was applied.

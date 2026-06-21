# frankentorch-kgs4.148 scorecard

Assignee: cod-a
Date: 2026-06-21
Status: rejected, no source kept

## Lever

Prototype a direct f64 5-D `sum(functional_conv3d(...))` path that bypasses
the dense output-gradient buffer by using a scalar-loss Conv3d backward helper.

## Result

| Row | Median |
| --- | ---: |
| `frankentorch_kgs4_119` current materialized sum | `19.413 ms` |
| `frankentorch_kgs4_148_fused_sum_loss` prototype | `20.604 ms` |
| PyTorch sidecar | `7.741739 ms/iter` |

Ratios:

- Candidate vs current: `1.061x` slower.
- Candidate vs PyTorch: `2.66x` slower.
- Current vs PyTorch: `2.51x` slower.
- Score: `0W / 1L / 0N`.

## Verification

- Prototype equivalence test passed before rejection:
  `cargo test -p ft-api --lib functional_conv3d_sum_matches_conv3d_sum_backward_bits --release -- --nocapture`.
- Candidate benchmark:
  `rch exec -- cargo bench --profile release -p ft-api --bench pytorch_gauntlet_bench -- gauntlet_conv3d_grad/frankentorch --noplot`.
- PyTorch sidecar:
  `FT_GAUNTLET_ITERS=40 FT_TORCH_THREADS=32 FT_TORCH_INTEROP_THREADS=32 /data/projects/.venvs/frankentorch-pytorch-cpu/bin/python crates/ft-api/benches/pytorch_conv3d_grad.py`.
- Post-rejection bench target check passed:
  `rch exec -- cargo check --release -p ft-api --bench pytorch_gauntlet_bench`.
- Post-rejection conformance passed:
  `rch exec -- cargo test --profile release -p ft-conformance`.

## Decision

Do not retry the scalar-loss wrapper family for Conv3d. The remaining gap needs
kernel/scheduler/workspace or oneDNN-class algorithm work, not another API-level
shortcut around dense output gradients.

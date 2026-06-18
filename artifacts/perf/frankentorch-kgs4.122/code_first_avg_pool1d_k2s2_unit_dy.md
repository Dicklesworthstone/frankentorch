# frankentorch-kgs4.122 - avg_pool1d k2s2 unit-dy constant-fill backward

## Target

- Bead: `frankentorch-kgs4.122`
- Assignee: `cod-b`
- Crate: `ft-kernel-cpu`
- Hot row: `conv1d_family/avg_pool1d_grad`
- Source profile: `artifacts/perf/frankentorch-ftapi-train-reprofile-20260616/baseline_train_hotspots.log`
- Baseline signal: profile row median `282.43 ms` before the direct f64 avg_pool1d keep; the training workload is a sum-loss backward path, so upstream `dout` is exact all-ones.

## Lever

Collapse the exact `avg_pool1d_backward_f64` case `kernel=2`, `stride=2`, no uncovered tail, exact all-ones `dout` to a single constant `0.5` fill over `din`.

This is narrower than the earlier direct avg_pool1d route: it only removes the per-output gradient read/divide/add loop for the sum-loss no-overlap case. The generic path is unchanged for non-unit gradients, overlapping windows, odd tails, and short `dout` slices.

## Correctness Guard

- Added `avg_pool1d_k2s2_unit_dy_backward_matches_generic_bits`.
- The guard builds an explicit generic reference and compares every output bit.
- It also verifies short-`dout` calls still panic instead of being silently accepted by the fast path.

## Negative-Evidence Ledger

| Attempt | Result | Evidence | Follow-up rule |
| --- | --- | --- | --- |
| Direct f64 avg_pool1d forward/backward route | Helped; kept in `frankentorch-3b7mi` | `artifacts/perf/frankentorch-3b7mi/closeout_avg_pool1d_keep.md` | Do not repeat direct-rank-4 bypass work. |
| k2/s2 unit-dy constant-fill collapse | Code-first crate check passed; batch Criterion/conformance pending | `CARGO_TARGET_DIR=/data/projects/.rch-targets/frankentorch-cod-b cargo check -p ft-kernel-cpu` passed locally | If batch Criterion or conformance rejects it, do not retry avg_pool1d constant-gradient fill variants without a deeper autograd/profile trace proving this kernel still dominates. |

## Batch Status

- Code-first guard committed immediately per campaign.
- Local validation: `CARGO_TARGET_DIR=/data/projects/.rch-targets/frankentorch-cod-b cargo check -p ft-kernel-cpu` passed.
- Criterion comparison against the legacy original and full conformance are pending in the batch runner.

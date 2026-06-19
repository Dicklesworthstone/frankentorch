# frankentorch-kgs4.127 - max_pool3d saved-index backward sidecar

Agent: IvoryDeer / cod-b
Date: 2026-06-19
Status: measured keep vs parent; PyTorch-loss row for release readiness

Note: the original commit used `frankentorch-kgs4.127` in its subject while the
live Beads entry for this max_pool3d lane is `frankentorch-kgs4.117`.

## Workload Trigger

Source profile:

- `artifacts/perf/frankentorch-next-reprofile-20260617/current_top_train_reprofile.log`
- Hot row: `max_pool3d/grad`
- Current timing in that route: `[35.230 ms 36.089 ms 36.992 ms]`
- Benchmark shape: `[N,C,D,H,W]=[2,32,16,32,32]`, `kernel=2x2x2`, `stride=2x2x2`, followed by `tensor_sum(out)` and backward.

The current f64 API path saved the full input tensor in the autograd context and
rescanned windows in backward.

## Lever

Add max-pool3d saved-index sidecars:

1. `max_pool3d_forward_with_indices_f64`: returns output plus plane-local
   first-argmax offsets.
2. `max_pool3d_backward_from_indices_f64`: scatters `dout` through saved
   offsets.
3. `functional_max_pool3d`: saves the compact arg-offset sidecar instead of the
   full `[N,C,D,H,W]` input for f64 grad.

This is distinct from `frankentorch-kgs4.117`, which only unrolled the 2x2x2
rescan loop inside the old input-rescan backward.

Alien mapping:

- Sidecar evidence / memoized search result: move argmax selection to forward
  once, then reuse it in backward.
- Cache-aware context layout: replace a full input save with output-sized
  offsets.
- Communication avoidance: remove backward input-window rescans from the f64 API
  route.

## Correctness Guard

Added `max_pool3d_indices_scatter_matches_rescan_first_tie_bits`.

The guard builds tie-heavy overlapping 3-D windows, verifies sidecar forward
values against the existing forward, and verifies sidecar scatter gradients
bit-for-bit against the existing `max_pool3d_backward_f64` rescan path.

## Negative-Evidence Ledger

| Attempt | Evidence | Decision |
| --- | --- | --- |
| MaxPool3d 2x2x2 stride2 unrolled rescan | `artifacts/perf/frankentorch-kgs4.117/code_first_max_pool3d_2x2x2s2_bwd.md`; superseded by this sidecar measurement. | Do not repeat unroll-only work; this pass removes the rescan/input save. |
| MaxPool2d borrowed-input tape plumbing | `artifacts/perf/frankentorch-pool2d-borrowed-max/report.md`; same-worker median regressed `99.832 ms -> 108.28 ms`. | Do not retry borrowed-input-only pool plumbing. |
| MaxPool1d direct saved-index route | `artifacts/perf/frankentorch-kgs4.109/closeout_direct_max_pool1d_keep.md`; kept. | Positive adjacent sidecar evidence; this pass extends the sidecar pattern to 3-D. |

If batch Criterion or conformance rejects this patch, do not retry max-pool3d
sidecar saves without a focused profile showing context memory or backward
window rescans still dominate.

## Verification

Measured under `artifacts/perf/frankentorch-kgs4.117/gauntlet_20260619T0320Z/`.

- Parent-before-sidecar `c79d3a23`, local Criterion:
  FrankenTorch median `20.585 ms`; PyTorch median `2.1381 ms`; ratio
  `9.63x` slower than PyTorch.
- Current post-lint tree, local Criterion:
  FrankenTorch median `15.794 ms`; PyTorch median `1.6228 ms`; ratio
  `9.73x` slower than PyTorch.
- Internal verdict: keep; `1.30x` speedup vs parent on the same deterministic
  workload.
- Release verdict: PyTorch-loss row; not performance-dominant against the
  original.
- rch row: current FrankenTorch median `28.124 ms` on `hz2`; the PyTorch arm
  failed there because `torch` was not installed on the remote worker.

Post-measure gates:

```bash
CARGO_TARGET_DIR=/data/projects/.rch-targets/frankentorch-cod-b rch exec -- cargo check -p ft-api --bench pytorch_gauntlet_bench
CARGO_TARGET_DIR=/data/projects/.rch-targets/frankentorch-cod-b rch exec -- cargo test -p ft-kernel-cpu max_pool3d_indices_scatter_matches_rescan_first_tie_bits
CARGO_TARGET_DIR=/data/projects/.rch-targets/frankentorch-cod-b rch exec -- cargo test -p ft-api functional_max_pool3d_grad_matches_finite_diff
CARGO_TARGET_DIR=/data/projects/.rch-targets/frankentorch-cod-b rch exec -- cargo clippy -p ft-api --bench pytorch_gauntlet_bench -- -D warnings
```

Result: PASS on 2026-06-19 after clippy-only lint fixes in `ft-kernel-cpu`.

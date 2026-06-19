# frankentorch-kgs4.126 - max_pool1d unit-dout scatter

Agent: IvoryDeer / cod-b
Date: 2026-06-19
Status: superseded by measured gauntlet reject

## Gauntlet Verdict

Measured on 2026-06-19 and reverted.

- Candidate median: `184.41 ms`
- Parent-before-lever median: `178.47 ms`
- PyTorch median in candidate run: `14.984 ms`
- Ratio vs PyTorch: `12.31x` slower
- Verdict: no statistically significant full-workload gain; point estimate was
  `1.033x` slower than parent.

See `artifacts/perf/frankentorch-kgs4.126/gauntlet_20260619T0113Z/summary.md`
and `docs/NEGATIVE_EVIDENCE.md`.

## Workload Trigger

Source profile:

- `artifacts/perf/frankentorch-ftapi-train-reprofile-20260616/baseline_train_hotspots.log`
- Hot row: `conv1d_family/max_pool1d_grad`
- Source-profile median before the direct max-pool1d keep: `306.70 ms`
- Benchmark shape: `[N,C,L]=[8,64,8192]`, `kernel=2`, `stride=2`, followed by `tensor_sum(out)` and backward.

`tensor_sum(out)` makes the upstream pool gradient exact all-ones.

## Lever

The kept direct f64 max-pool1d route already saves first-argmax offsets during
forward. Its backward still called the generic index scatter, reading `dout` for
every output position even when `dout` is known to be all `+1.0`.

This patch detects exact all-ones `dout` in the f64
`functional_max_pool1d` backward closure and scatters constant `+1.0` through
the saved arg offsets. The generic fallback remains unchanged for non-unit
gradients.

Alien mapping:

- Branch-specialized hot path: collapse a runtime invariant produced by sum-loss
  training traces.
- Cache-aware scatter: remove the contiguous `dout` read stream and keep the
  saved-index access pattern unchanged.
- Behavior-preserving specialization: first-tie argmax offsets and scatter order
  are unchanged.

## Correctness Guard

Added `max_pool1d_unit_dout_indices_match_generic_scatter_bits`.

The guard builds tie-heavy overlapping windows, captures saved arg offsets with
`max_pool1d_forward_with_indices_f64`, and compares the new constant-scatter
helper bit-for-bit against `max_pool1d_backward_from_indices_f64` with an
explicit all-ones `dout`.

## Negative-Evidence Ledger

| Attempt | Evidence | Decision |
| --- | --- | --- |
| Direct f64 max-pool1d route | `artifacts/perf/frankentorch-kgs4.109/closeout_direct_max_pool1d_keep.md`; kept, `420.32 ms -> 263.76 ms` local median. | Build on it; do not repeat rank-4 reshape bypass work. |
| max_pool2d borrowed-input-only tape route | `artifacts/perf/frankentorch-pool2d-borrowed-max/report.md`; same-worker median regressed `99.832 ms -> 108.28 ms`. | Do not retry borrowed-input-only pool plumbing. |
| avg_pool1d k2s2 unit-dy constant fill | `artifacts/perf/frankentorch-kgs4.122/code_first_avg_pool1d_k2s2_unit_dy.md`; code-first pending. | Adjacent positive pattern only; this pass is max-pool saved-index scatter. |

If batch Criterion or conformance rejects this patch, do not retry max-pool1d
unit-dout scatter variants without a new profile showing generic `dout` reads
still dominate after the direct route.

## Verification

Required local-only gate:

```bash
CARGO_TARGET_DIR=/data/projects/.rch-targets/frankentorch-cod-b cargo check -p ft-api
```

Result: PASS on 2026-06-19.

Not run by instruction: tests, rch, clippy, fmt, Criterion/conformance batch.

# frankentorch-kgs4.124 - SmoothL1 reduced grad custom op

## Campaign

- Agent: `cod-b` / `IvoryDeer`
- Mode: code-first, batch-test pending
- Bead: `frankentorch-kgs4.124`
- Scope: `ft-api` + `ft-kernel-cpu`
- Benchmark target: `smooth_l1/grad_8m` in `crates/ft-api/benches/ops_bench.rs`

## Profile evidence used

Existing checked-in routing evidence names loss functions as a major non-linalg
gap:

- `artifacts/perf/loss-fusion/ops_bench_hotspots.txt`
- `.beads/issues.jsonl` entry `frankentorch-3jmy3`

Relevant prior row:

- `smooth_l1/grad_8m`: `1.2587 s`
- `smooth_l1/nograd_8m`: `94.701 ms`

This pass does not claim a measured win. It prepares a single code-first lever
for the required batch Criterion/conformance pass.

## Lever

Existing f64 SmoothL1 already has a no-grad `mean`/`sum` fast path that computes
the reduced scalar directly. The grad path still materialized the full
per-element SmoothL1 tensor and then applied `tensor_mean` or `tensor_sum`.

This change adds:

1. `smooth_l1_backward_reduced_f64(scale, x, target, beta)` in `ft-kernel-cpu`.
2. A same-shape f64 `tensor_smooth_l1_loss(..., "mean"|"sum", beta)` grad route
   in `ft-api` that returns the reduced scalar directly.
3. A backward closure that maps scalar upstream gradient to the exact uniform
   per-element gradient used by the old materialized reduction path.

The `"none"` path and non-f64 fallback remain unchanged.

## Alien mapping

- Graveyard §6.5 Polyhedral/locality optimization: collapse affine elementwise
  loop plus reduction into one direct reduced dataflow.
- Graveyard §9.6 Communication-avoiding algorithms: reduce memory traffic by
  avoiding a full intermediate tensor and an extra reduction pass.
- Alien-artifact proof family: certified rewrite of a loss graph into an
  equivalent scalar custom op with explicit backward witness.

EV score:

- Impact: 4 - `smooth_l1/grad_8m` is a checked-in hot row.
- Confidence: 3 - no-grad reduction is already a kept pattern; grad scalar
  route is simpler than rejected gaussian NLL reduced grad.
- Reuse: 3 - same reduced-backward pattern applies to other simple losses if
  batch evidence keeps it.
- Effort: 2
- Adoption friction: 2
- EV = `(4 * 3 * 3) / (2 * 2) = 9.0`

## Behavior proof obligations

- Forward scalar value must match the old materialized per-element path followed
  by `tensor_mean`/`tensor_sum`.
- Input and target gradients must match the old path bit-for-bit for uniform
  upstream gradients.
- Boundary behavior remains `abs(d) < beta`; the kink at `abs(d) == beta` still
  takes the linear branch.
- Empty `mean` preserves `NaN` forward behavior and returns empty gradients.
- DType, shape, RNG behavior, and error behavior remain unchanged.

## Guards added

- `ft-kernel-cpu`: `smooth_l1_backward_reduced_f64_matches_uniform_dloss_bits`
- `ft-api`: `smooth_l1_loss_reduced_grad_matches_materialized_reference_bits`

These were added for the batch suite. Per campaign instruction, this slice only
ran local `cargo check -p ft-api`.

## Negative-evidence ledger

| Candidate | Prior evidence | Verdict | Retry condition |
|---|---|---|---|
| f32 SmoothL1 no-grad fast path | `frankentorch-cs2d` failed before stable after-benchmark because same-shape f32 path hit mixed f32/f64 comparison constants; scratch kernel proof alone was insufficient. | Do not retry here. | Only retry after a new bead fixes the f32 SmoothL1 dispatch/constants path and produces a current f32 criterion baseline. |
| Gaussian NLL reduced grad | `frankentorch-fdn1v` preserved bits but regressed `gaussian_nll/grad_8m` from `829.27 ms` to `1.0274 s`. | Do not generalize blindly. | Only retry if a new profile shows a different dominant cause than scalar reduction materialization. |
| SmoothL1 no-grad f64 pairwise map-reduce | `frankentorch-lonz` kept this route and it is already in `origin/main`. | Already shipped. | No retry; build on it only through a distinct grad-path lever. |

## Verification this slice

Passed:

```bash
CARGO_TARGET_DIR=/data/projects/.rch-targets/frankentorch-cod-b cargo check -p ft-api
```

Result: `Finished dev profile` for `ft-api` after checking `ft-core`,
`ft-kernel-cpu`, `ft-runtime`, `ft-dispatch`, and `ft-autograd`.

Not run by instruction:

- Criterion
- conformance
- tests
- clippy
- fmt
- rch

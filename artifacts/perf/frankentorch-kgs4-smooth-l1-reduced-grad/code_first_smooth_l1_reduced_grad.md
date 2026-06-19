# frankentorch-kgs4.124: SmoothL1 Direct Reduced Grad

## Lever

Direct f64 same-shape `tensor_smooth_l1_loss(..., "mean" | "sum", beta)` through a scalar custom autograd op when either input requires grad.

The previous grad path built a full per-element SmoothL1 tensor, then added a `tensor_mean` or `tensor_sum` node. Backward therefore materialized a uniform `dloss` vector before calling `smooth_l1_backward_f64`. This pass computes the scalar reduced value directly via the existing fused reducer and backpropagates the scalar upstream scale through `smooth_l1_backward_reduced_f64`.

## Guards

- Kernel guard: `smooth_l1_reduced_backward_f64_matches_uniform_dloss_bits` compares reduced scalar backward to the old materialized uniform-`dloss` helper bit-for-bit.
- API guard: `smooth_l1_loss_reduced_grad_matches_none_then_reduce_bits` compares direct reduced autograd against `reduction="none"` followed by the existing reduction graph for output, input grad, and target grad bits.

## Negative-Evidence Ledger

| Attempt | Evidence | Outcome | Retry rule |
| --- | --- | --- | --- |
| f32 SmoothL1 no-grad fused path | `artifacts/perf/frankentorch-cs2d/rejected_f32_smooth_l1_fast_path.md` | Rejected/no stable after benchmark; mixed f32/f64 constants and artifact churn blocked a keep. | Do not retry f32 no-grad SmoothL1 without a fresh dtype audit and same-worker A/B. |
| f64 SmoothL1 no-grad pairwise reducer | `frankentorch-lonz`, `artifacts/perf/frankentorch-ruby-smoothl1-f64-reduction/report.md` | Kept; baseline 136.80 ms -> 97.302 ms. | Do not rework the no-grad reducer family; this bead is grad-only. |
| direct reduced Gaussian NLL grad | `frankentorch-fdn1v` | Rejected; same-worker median regressed 829.27 ms -> 1.0274 s. | Do not generalize this SmoothL1 lever to Gaussian NLL without new profile proof. |
| SmoothL1 direct reduced grad | This bead | Code-first pending batch Criterion/conformance. | If rejected, route to retained-graph/tape compaction or loss-kernel branch/SIMD work, not another scalar-reduction wrapper. |

## Proof Status

Code-first local gate required by campaign: `CARGO_TARGET_DIR=/data/projects/.rch-targets/frankentorch-cod-a cargo check -p ft-api` and `cargo check -p ft-kernel-cpu` only. Criterion/conformance batch is intentionally pending.

# frankentorch-kgs4.123 - f32 RMSNorm unit-dy backward fast path

Assignee: cod-b
Agent Mail: IvoryDeer
Status: code-first batch-test pending; no speedup claimed yet.

## Profile target

- Current train reprofiles keep normalization backward in the hot set.
- `rms_norm/grad_2048x1024` is around 150 ms in
  `artifacts/perf/frankentorch-next-reprofile-20260617/current_top_train_reprofile.log`.
- The existing Criterion training row computes `loss = tensor_sum(rms_norm(...))`,
  so the upstream gradient entering RMSNorm backward is exactly all `+1.0`.
- f64 already has this constant-gradient branch; f32 is the dominant ML dtype and
  still used the generic `dy` stream before this pass.

## Lever

Specialize `rms_norm_backward_f32` when all of these hold:

- every `dy` bit is exactly `1.0f32`,
- `x` values are finite,
- optional `weight` values are finite.

The fast path stages each row's `rstd` once, removes `dy` loads and multiplies
from `dx`, and computes `dweight` as `sum(x * rstd)` in the same row-major order.
All non-unit or non-finite cases fall back to the generic f32 implementation.

## Alien mapping

- Graveyard primitive: guarded hot-path partial evaluation for stable workload
  state.
- Artifact-coding family: certified rewrite of a linear cotangent flow under a
  precise guard.
- Cache lever: remove one full f32 gradient stream and one multiply stream from
  the measured sum-loss training path.
- EV sketch: Impact 2, Confidence 4, Effort 1 -> Score 8.0 before benchmark;
  actual keep decision is pending Criterion and conformance.

## Behavior preservation

- Ordering preserved: yes. Per row, `ss`, `c`, `dx`, and `dweight` accumulation
  retain the generic loop order.
- Floating-point preserved: yes for the guarded finite `dy == 1.0` case;
  multiplying by exact one is removed without reassociation.
- Error/fallback behavior: non-one `dy`, NaN, inf, and non-finite weight inputs
  use the generic implementation.
- RNG/tie behavior: none.

## Guard

Added `rms_norm_f32_unit_dy_fast_path_matches_generic_reference_bits` in
`crates/ft-kernel-cpu/src/lib.rs`.

The test compares public `rms_norm_backward_f32` against the private generic
f32 formula bit-for-bit on finite unit-dy inputs with learned weight.

## Negative-evidence ledger

| Attempt | Evidence | Decision |
| --- | --- | --- |
| RMSNorm forward-saved rstd sidecar | `artifacts/perf/frankentorch-fad7c/report.md` and `artifacts/perf/frankentorch-kgs4.120/rms_norm_unit_dy_ledger.md` record overlapping/regressed saved-stat evidence. | Do not retry forward sidecars. |
| RMSNorm f64 stat-staging unit-dy branch | `artifacts/perf/frankentorch-t89dc/closeout_rms_norm_unit_dy_reject.md` rejected an ambiguous f64 route. | This pass is f32 only and mirrors the already-landed exact finite f64 branch shape. |
| Saved-stat normalization routes | `frankentorch-2rsa6` rejected saved-stat-only GroupNorm. | Do not route this bead through API saved-stat plumbing. |

If batch Criterion overlaps or regresses, reject this branch and route away from
RMSNorm unit-dy micro-specialization unless a later profile shows f32 RMSNorm
backward still top-5 and the miss is attributable to the remaining generic f32
dy stream.

## Verification

Required by this code-first pass:

```bash
CARGO_TARGET_DIR=/data/projects/.rch-targets/frankentorch-cod-b cargo check -p ft-kernel-cpu
```

Pending batch gates: focused Criterion f32 RMSNorm backward row, broad train
profile, and ft-conformance coverage.

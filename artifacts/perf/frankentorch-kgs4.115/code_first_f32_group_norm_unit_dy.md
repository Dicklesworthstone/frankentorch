# frankentorch-kgs4.115 - f32 GroupNorm unit-dy backward fast path

## Lever

Mirror the measured f64 all-ones-`dy` GroupNorm backward branch into
`group_norm_backward_f32`. The branch stages per-`(batch, group)` `(mean, rstd)`
once, reuses those stats for `dx`, and keeps affine `dweight`/`dbias` in the
same deterministic serial group-major reduction order as the generic f32 path.

This is a code-first candidate. It is not a kept win until a focused
batch-test pass measures the f32 Criterion target against the legacy/original
baseline.

## Benchmark Target

- Future focused gate: `group_norm/grad_f32_32x256x28x28`.
- Existing routing context: decontaminated norm-gradient notes show
  `group_norm/grad_32x256x28x28` remains a realistic training hotspot after
  removing in-loop RNG.

## Correctness Guard

- Added `group_norm_f32_unit_dy_matches_general_reference_bits`.
- The guard computes expected `dx`, `dweight`, and `dbias` with the previous
  generic f32 GroupNorm backward formula.
- Outputs are compared bit-for-bit.

## Negative-Evidence Ledger

| Attempt | Evidence | Decision |
| --- | --- | --- |
| f64 GroupNorm all-ones dy branch | `artifacts/perf/frankentorch-16m8a/closeout_group_norm_unit_dy_keep.md` measured a modest but real 1.0411x keep. | Use as adjacent positive evidence only; this bead targets the previously unchanged f32 primitive. |
| f64 GroupNorm saved-stat-only route | `frankentorch-2rsa6` closed rejected; saved-stat-only normalization work did not clear gates. | Avoid API saved-stat reroutes here. |
| f32 GroupNorm no-grad fusion | `frankentorch-r2hi` closed keep for f32 no-grad forward/upcast repair. | Do not repeat no-grad forward fusion; this is backward-only. |
| BatchNorm f64 spatial-1 row-major/stat route | `artifacts/perf/frankentorch-kgs4.110/closeout_batch_norm_spatial1_rejected.md` rejected that normalization micro-family. | Do not generalize row-major stats rewrites without fresh focused proof. |

## Status

- Code-first batch-test pending.
- No speedup claimed yet.
- Bead remains `in_progress` for focused benchmark/conformance batch follow-up.

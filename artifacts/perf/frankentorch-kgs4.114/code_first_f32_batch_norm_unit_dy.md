# frankentorch-kgs4.114 - f32 BatchNorm unit-dy backward fast path

Assignee: cod-b

## Lever

Specialize `batch_norm_backward_f32` for the common all-ones upstream-gradient
case. This removes the per-element `dy * xhat` multiply, replaces the `dbias`
reduction with the known reduction cardinality, precomputes per-channel `rstd`,
and uses a row-major spatial-1 dx pass.

This is a code-first candidate under the no-gaps perf epic. It is not yet a
kept performance win until the batch-test pass runs the focused criterion gate
against the legacy/original baseline.

## Benchmark Target

- Current routing profile:
  `artifacts/perf/frankentorch-next-reprofile-20260617/current_top_train_reprofile.log`
  reports `batch_norm/grad_1d_8192x1024` at `[678.28 ms 693.66 ms 717.16 ms]`
  and `batch_norm/grad_train_32x256x28x28` at `[503.95 ms 513.09 ms 520.41 ms]`.
- Future focused gate: `batch_norm/grad_f32_train_32x256x28x28`
- Future focused gate: f32 spatial-1 BatchNorm backward microbench
- Existing routing context: decontaminated norm-gradient notes show BatchNorm
  backward remains a realistic training-trace hotspot after removing in-loop RNG.

## Correctness Guard

- Added `batch_norm_f32_unit_dy_matches_general_reference_bits`.
- The guard checks both `spatial == 1` and `spatial > 1` layouts.
- The expected outputs are computed with the previous general BatchNorm backward
  formula and compared bit-for-bit for `dx`, `dweight`, and `dbias`.

## Negative-Evidence Ledger

| Attempt | Evidence | Decision |
| --- | --- | --- |
| f64 BatchNorm1d all-ones dy specialization | `artifacts/perf/frankentorch-etebu/closeout_batch_norm1d_unit_dy_reject.md` measured only ~1.0115x/no significant win. | Do not retry f64 spatial-1 all-ones micro-specialization as a keep claim. |
| f64 spatial-1 row-major/channel-chunk reduction | `artifacts/perf/frankentorch-kgs4.110/closeout_batch_norm_spatial1_rejected.md` rejected the row-major stats/reduction route. | Do not repeat f64 row-major stats/reduction changes. |
| f64 GroupNorm saved-stat-only route | `frankentorch-2rsa6` closed rejected; saved-stat-only normalization paths did not clear proof/perf gates. | Do not route this bead through saved-stat API changes. |
| f64 BatchNorm2d all-ones dy primitive | `frankentorch-6olvt` closed as a modest keep. | Adjacent positive evidence only; this bead targets the missing f32 primitive, not a repeat of the f64 code path. |

## Alien/Optimization Mapping

- Graveyard primitive: guarded hot-path specialization / partial evaluation for
  stable workload state (`dy == 1`) with a generic fallback for every other dy.
- Cache lever: precompute the per-channel `rstd` vector and avoid one full
  `dbias` reduction stream when the reduction cardinality is known exactly.
- Expected-loss rule: false positive is disallowed by exact `to_bits()` dy
  detection; if the branch does not clear same-worker benchmark gates, revert
  this commit or leave it behind a future evidence-backed retargeting bead.

## Status

- Code-first batch-test pending.
- No speedup claimed yet.
- Validation run completed: `CARGO_TARGET_DIR=/data/projects/.rch-targets/frankentorch-cod-b cargo check -p ft-kernel-cpu` passed on 2026-06-18.
- Not run by instruction: tests, Criterion, conformance, clippy, fmt, rch.
- Bead remains `in_progress` for focused benchmark/conformance batch follow-up.

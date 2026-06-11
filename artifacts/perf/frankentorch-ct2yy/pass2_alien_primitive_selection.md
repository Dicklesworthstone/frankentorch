# frankentorch-ct2yy pass 2 alien primitive selection

Date: 2026-06-11
Agent: Codex
Git HEAD inspected: `fe118805eb3c8b4e3332c27c88fb430b9722ac15`

## Mission result

No implementation in this pass. The correct pass 3 lever is an instrumentation-only
stage split for QR, not a kernel rewrite. The recursive/communication-avoiding QR
family is the right alien primitive family, but implementation is not yet
profile-backed because there is no panel-vs-trailing split row.

Current strongest baseline evidence comes from the pass 1 handoff:

| Criterion row | Worker | Estimate |
| --- | --- | ---: |
| `qr_f64_512x512` | `vmi1227854` | `[63.562 ms 64.514 ms 65.435 ms]` |
| `qr_f64_tall_2048x128` | `vmi1227854` | `[42.719 ms 46.387 ms 49.923 ms]` |

The existing `pass1_baseline_profile.md` contains an older local-fallback baseline
and correctly marks it as routing evidence only. Do not use that fallback row as a
keep/reject proof.

Audit note (2026-06-11T04:55Z): this directory currently contains the
local-fallback pass-1 logs, but no raw remote `vmi1227854` Criterion log for the
two rows above. Treat the `vmi1227854` figures in this artifact as reported
handoff evidence until the raw log is recovered or regenerated. Pass 3 remains
instrumentation-only and must produce its own same-worker remote proof before it
authorizes implementation.

## Source lineage

- Bead `frankentorch-ct2yy`: NB=32 -> 64 regressed `qr_512` and tall QR; bead
  notes name scalar Householder panel factorization as the residual, not GEMM
  skinniness.
- Current code: `qr_contiguous_f64` routes large `m >= 128 && k >= 16` matrices
  through `qr_householder_panel_blocked`.
- Current QR panel code: `NB=32`; scalar Householder panel work is the loops at
  `crates/ft-kernel-cpu/src/lib.rs:14491`; trailing R update uses three
  `gemm::dgemm` calls at `crates/ft-kernel-cpu/src/lib.rs:14571`; reverse Q
  formation uses three `gemm::dgemm` calls at
  `crates/ft-kernel-cpu/src/lib.rs:14611`.
- Bench rows: `qr_f64_512x512` and `qr_f64_tall_2048x128` live in
  `crates/ft-kernel-cpu/benches/linalg_bench.rs:113`.
- Alien graveyard source: `alien_cs_graveyard.md` section 9.6
  Communication-Avoiding Algorithms maps directly to CA-QR/TSQR, tree-structured
  local QR, Householder replay, data-movement lower bounds, and stability
  certificates.
- Alien artifact source: `34-NUMERICAL-LINEAR-ALGEBRA.md` selects QR for
  orthogonal basis/least squares, requires matrix characterization,
  factorization accuracy, orthogonality checks, condition monitoring, and
  decomposition records.
- FrankenSuite methodology source: performance claims require controlled
  benchmark environment, median/tail metrics, practical effect size, golden
  outputs plus invariants, and explicit profile-first opportunity gates.

## Recommendation contract

Change:

Add a temporary QR stage-split probe for `qr_householder_panel_blocked` that
separately times:

1. scalar panel factorization plus `T` build,
2. trailing `R` compact-WY GEMMs,
3. reverse `dorgqr` compact-WY GEMMs,
4. final copy/zeroing overhead.

Hotspot evidence:

- Same-worker `vmi1227854` Criterion rows show the QR path is still material.
- NB=64 is already a negative result, which points away from simple block-size
  tuning.
- Current code shows trailing `R` and reverse `Q` are already GEMM-backed; the
  remaining non-GEMM body is panel factorization plus `T` construction.

Mapped graveyard sections:

- `alien_cs_graveyard.md` section 9.6: CA-QR/TSQR tree structure and Householder
  replay. This maps to recursive/tournament panel QR, not another scalar NB
  tweak.
- `34-NUMERICAL-LINEAR-ALGEBRA.md` method table: QR is the selected direct method
  for orthogonal bases and least-squares shape workloads.
- `34-NUMERICAL-LINEAR-ALGEBRA.md` artifact and verification sections:
  decomposition record, factorization accuracy, orthogonality, residual, and
  condition-number checks.

EV score:

Instrumentation-only split probe has EV 40.0
`Impact 2 * Confidence 5 * Reuse 4 / (Effort 1 * Friction 1)`.

Priority tier:

S for pass 3 because it is the minimum profile evidence needed before touching
QR numerics.

Adoption wedge:

Bench/probe only. Do not route production QR differently in pass 3.

Budgeted mode:

Run only crate-scoped `ft-kernel-cpu` Criterion rows on one selected worker.
Default target worker: `vmi1227854`; if `rch` cannot select a remote worker,
record fallback as routing evidence only and do not implement a kernel change.

Expected-loss model:

States:

- `panel_dominant`: panel plus `T` >= 35% of total on at least one primary row
  and Amdahl model predicts >= 8% end-to-end improvement from a recursive panel.
- `gemm_or_q_dominant`: trailing `R`, reverse `Q`, allocation/copy, or scheduler
  overhead dominates.
- `noisy_or_unavailable`: worker mismatch, local fallback, or Criterion outliers
  make the split inconclusive.

Actions:

- `A_probe`: add only stage timing.
- `A_recursive_panel`: implement recursive panel QR in a later pass.
- `A_tsqr_caqr`: route to tile/TSQR/CAQR if tall-row evidence dominates.
- `A_reject_qr_panel`: stop this lever family and reprofile another QR primitive.

Loss:

- High loss for implementing recursive/CAQR without a stage split because it can
  change floating-point association and miss the real bottleneck.
- Medium loss for delaying implementation one pass.
- High loss for another NB-size attempt because the bead already has negative
  evidence.

Calibration and fallback trigger:

If same-worker split evidence is unavailable, pass 3 closes as "routing only" and
does not authorize implementation. If split evidence shows panel plus `T` < 25%
on both primary rows, reject recursive panel QR and route to the measured dominant
stage instead.

Isomorphism proof plan:

Pass 3 is instrumentation-only, so QR output must be bit-identical and golden
sha256 must match pre-probe output. Later recursive/CAQR work must preserve
documented QR semantics through deterministic Householder order, fixed tree
shape, no pivoting, unchanged shape/error behavior, and reconstruction plus
orthogonality proofs. If strict Q/R byte-for-byte preservation is required for
the implemented kernel, recursive/CAQR is reject-by-construction unless the new
path can match the current golden Q/R sha256.

p50/p95/p99 before/after target:

Pass 3 has no speed target. It must report total and per-stage medians, tails
where available, and stage percentages for `512x512` and `2048x128` on the same
worker. A later implementation must improve the primary row by >= 3% with 95%
confidence excluding zero and no p95/p99 regression; prefer >= 8% because QR
float-association risk is nontrivial.

Primary failure risk and countermeasure:

Risk: recursive/tournament QR changes floating-point association and may change
Q/R bit patterns.

Countermeasure: deterministic fixed tree, no pivoting, current sign/tiny rules,
golden sha256 for unchanged/probe passes, reconstruction/orthogonality/residual
ledger for tolerance-contract passes, and immediate fallback to current
compact-WY path if any invariant fails.

Baseline comparator:

Current compact-WY QR at `NB=32`.

Rollback:

Revert the single pass commit. Pass 3 should be isolated to probe code and
artifacts.

## EV and score table

EV uses alien-graveyard scoring:
`Impact * Confidence * Reuse / (Effort * AdoptionFriction)`.

ESO score uses optimization scoring:
`Impact * Confidence / Effort`.

| Candidate | Profile-backed path | Impact | Confidence | Reuse | Effort | Friction | EV | ESO score | Ranking |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| Instrumentation-only split probe | Directly measures the missing panel/T vs trailing/Q split on existing QR bench rows. | 2 | 5 | 4 | 1 | 1 | 40.0 | 10.0 | Pass 3 gate |
| Recursive/block-recursive panel QR | Conditional on split proving panel/T share. Maps to cache-oblivious blocked QR and keeps current compact-WY outer interface. | 4 | 3 | 3 | 5 | 2 | 3.6 | 2.4 | Best implementation candidate after probe |
| TSQR/CAQR/tournament panel | Strong graveyard lineage for tall QR; profile path exists through `2048x128`, but square QR benefit is less certain and proof burden is higher. | 5 | 2 | 3 | 5 | 3 | 2.0 | 2.0 | Conditional reroute if tall/panel tree dominates |
| Simple NB tuning | Already profile-negative: NB=64 regressed both square and tall rows. | 1 | 1 | 1 | 1 | 1 | 1.0 | 1.0 | Rejected |

## Proof obligations for QR preservation

1. Shape and error behavior:
   - Empty inputs return the same zero-shaped `Q`/`R`.
   - Reduced mode returns `Q` as `m x k` and `R` as `k x n`.
   - Full mode returns `Q` as `m x m` and `R` as `m x n`.
   - Non-2D and storage/layout error classes stay unchanged.

2. Householder contract:
   - Same no-pivoting policy.
   - Same sign convention: `sign = if v0 >= 0.0 { 1.0 } else { -1.0 }`.
   - Same tiny threshold behavior.
   - Same below-diagonal zeroing of `R`.

3. Numerical contract:
   - For instrumentation-only pass, Q/R byte sha256 must match before/after.
   - For any recursive/CAQR implementation, record whether it is bit-exact or
     tolerance-contract. If tolerance-contract, require:
     `||A - Q R||_F / ||A||_F` within existing QR tolerance,
     `||Q^T Q - I||_F` within existing QR tolerance, and no NaN/Inf behavior
     change on adversarial fixtures.
   - Condition-number report or at least a fixed ill-conditioned fixture set must
     be included because QR error budgets are condition-sensitive.

4. Determinism contract:
   - No RNG.
   - Fixed tree split order.
   - If row parallelism is introduced, reductions must have deterministic chunk
     boundaries and deterministic accumulation order.
   - No tie-breaking or pivot order changes.

5. Golden artifacts:
   - Pre-change Q/R sha256 for `qr_f64_512x512` and
     `qr_f64_tall_2048x128`.
   - Post-change sha256 check for pass 3 instrumentation.
   - Reconstruction/orthogonality ledger plus sha256 of the report for any later
     tolerance-contract implementation.

## Exact pass 3 target

One lever:

Add a crate-scoped QR stage-split Criterion/probe path for `frankentorch-ct2yy`
only. It should measure the existing algorithm without changing QR output.

Suggested rows:

- `qr_f64_512x512_stage_split`
- `qr_f64_tall_2048x128_stage_split`

Required same-worker command pattern:

```bash
RCH_REQUIRE_REMOTE=1 RCH_WORKER=vmi1227854 rch exec -v -- cargo bench -j 1 -p ft-kernel-cpu --bench linalg_bench -- qr_f64_512x512_stage_split --warm-up-time 1 --measurement-time 5 --sample-size 20
RCH_REQUIRE_REMOTE=1 RCH_WORKER=vmi1227854 rch exec -v -- cargo bench -j 1 -p ft-kernel-cpu --bench linalg_bench -- qr_f64_tall_2048x128_stage_split --warm-up-time 1 --measurement-time 5 --sample-size 20
```

Acceptance trigger for implementation in pass 4:

- Same remote worker for baseline and split.
- Panel factorization plus `T` build is >= 35% of total on at least one primary
  row, or >= 25% on both rows with low outlier noise.
- Amdahl model predicts >= 8% end-to-end improvement from a plausible recursive
  panel speedup.
- Golden Q/R sha256 unchanged by the probe.

Rejection/reroute trigger:

- Panel plus `T` < 25% on both rows: reject recursive panel QR.
- Reverse `dorgqr` dominates tall QR: route to Q-formation or TSQR replay.
- Trailing GEMMs dominate: route to GEMM packing/microkernel work, not QR panel.
- Worker mismatch or local fallback: record routing evidence only and rerun.

## Non-overlap note

This pass does not touch geev/AED/Hessenberg work (`qglh3`/`fql10`). Pass 3
should reserve only the QR bench/probe surface and the QR-local region around
`qr_householder_panel_blocked`; it should not edit eig/eigvals code paths.

## Files changed in pass 2

- `artifacts/perf/frankentorch-ct2yy/pass2_alien_primitive_selection.md`

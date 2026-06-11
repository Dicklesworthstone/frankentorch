# frankentorch-qglh3 pass 2 AED contract

Date: 2026-06-11

## Hotspot evidence

Current same-worker baseline on `vmi1227854`:

| Row | Median |
| --- | ---: |
| `eigvals_f64_256x256` | `27.445 ms` |
| `eig_f64_256x256` | `49.892 ms` |

The current `l9xod` profiling commit rules out the adjacent micro-levers:
parallel eigenvector back-substitution regresses, and column-range trimming is
effectively a no-op on dense spectra because `l` stays near zero. The remaining
shared wall is the serial single-bulge Francis QR phase.

## Mapped sources

- Graveyard canonical `§9.6 Communication-Avoiding Algorithms`: reduce data
  movement and batch dense linear-algebra operations into BLAS-3 kernels.
- FrankenSuite summary methodology: profile-first optimization and opportunity
  matrix gates.
- Alien-artifact numerical linear algebra: decompose the matrix method, attach
  factorization/reconstruction checks, and keep explicit tolerance/conditioning
  ledgers.
- Local policy `frankentorch-qgce4`: dense eigensolver vector outputs can use
  the same reconstruction + orthogonality + value-tolerance proof standard as
  blocked LU/Cholesky/QR/Hessenberg/SVD.

## Candidate matrix

| Candidate | Impact | Confidence | Effort | Score | Decision |
| --- | ---: | ---: | ---: | ---: | --- |
| AED trailing-window probe + shift-list artifact | 3 | 4 | 2 | 6.0 | Select |
| Full AED deflation/reorder in public `eig_francis_schur` | 5 | 2 | 5 | 2.0 | Too much for one pass |
| Multishift small-bulge BLAS-3 sweep | 5 | 2 | 5 | 2.0 | Depends on AED shift list |
| Backsub/column-range/live-window micro-tweaks | 0 | 1 | 1 | 0.0 | Rejected by evidence |

## Selected pass-3 lever

Add a bounded AED trailing-window probe that can Schur-factor a bounded trailing
Hessenberg window using the existing `eig_francis_schur` helper. The first
admissible wiring is values-only suffix deflation in `eigvals`; q_acc
back-transform and deterministic shift-list handoff remain separate follow-up
work for the full qglh3/fql10-C route. If the helper cannot pass focused tests,
golden digests, and same-worker Criterion, reject it without wiring.

## Recommendation contract

Change:
Introduce a trailing-window AED probe/helper that copies a `[kw..=en]`
Hessenberg window, recursively Schur-factors it with `eig_francis_schur`, and
only deflates a suffix when the spike-vector bound is conservatively small.

Hotspot evidence:
`eigvals_f64_256x256` median `27.445 ms` and full `eig_f64_256x256` median
`49.892 ms` on `vmi1227854`; current comments identify serial Francis QR as the
shared floor.

Fallback trigger:
Any public values-only wiring must fall back to the current double-shift QR unless
the spike-vector test proves safe deflation. The `want_vectors` path remains
unchanged until a later pass proves q_acc window back-transform and AED/multishift
handoff.

Isomorphism proof plan:
Ordering/tie behavior must preserve the current public interleaved `(re, im)`
order. RNG remains absent. Golden digests must still match for the strict current
fixtures; the pass-3 helper also needs focused tests that compare public
`eigvals` and `eig` consistency within the qgce4 tolerance budget.

p50/p95/p99 target:
Pass 3 is accepted only if it shows at least a `2.0` score against the
same-worker `27.445 ms` `eigvals` baseline while preserving the strict golden
digests.

Primary risk + countermeasure:
Risk is silently changing eigenvalue order or accepting premature deflation.
Countermeasure is no public wiring in the scaffold pass, explicit deflation
threshold fields, and strict fallback to the existing double-shift path.

Repro artifacts:
`pass1_baseline_profile.md`, raw pass-1 logs, this contract, focused helper tests,
and post-change `eigvals_golden` output.

Rollback:
Remove the helper/tests if it fails focused proof, golden, or same-worker
Criterion. Do not touch `ft-api`; full q_acc dispatch remains a later pass.

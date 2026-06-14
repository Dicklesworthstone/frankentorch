# Non-symmetric eig phase profile + the next primitive to attack (2026-06-14, BlackThrush)

## Why this note
The dense eigensolver/SVD parallelization vein (deferred whole-stream replay /
independent-trailing-update) has been HARVESTED for its clean Score>=2.0 wins:
- SVD Golub-Reinsch bidiagonalization + U/V back-transform parallel (kgs4.72, 2.2-3.1x)
- eigh tql2 eigenvector QL sweep deferred-replay, f64 + f32 (kgs4.73/74, 3.1-4.6x)
- (rejected: eigh tred2 REDUCE — sequential reflector loop is fork/join-bound, see
  eigh_tred2_reduce_parallel_rejected_2026-06-14.md)

This note records WHY the obvious remaining candidates are NOT Score>=2.0 levers, so
the next agent attacks the right (harder) primitive instead of re-profiling.

## Non-symmetric eig phase profile (f64, with eigenvectors, 10 threads)
    n=512:  hess 174ms | francis 247ms | backsub  30ms | total  454ms
    n=768:  hess 549ms | francis 761ms | backsub  68ms | total 1381ms
=> Hessenberg ~38-40%, Francis QR ~54-55%, eigenvector back-substitution ~5-7%.

### eig_backsub_eigenvectors (hqr2) — NOT worth parallelizing
Each eigenvector (Schur column) IS an independent back-solve and CAN be parallelized
over eigenvalues with one fork/join (write into a separate buffer reading the intact
Schur form, then the existing q_acc GEMM). BUT it is only 5-7% of eig, so even an 8x
phase speedup is ~1.06x on the op. Skip it.

### Francis QR (~55%) — q_acc accumulation already deferred-replay parallel (9y5bi);
the remaining serial cost is the bulge-chase recurrence on H (sequential bulges).

### Hessenberg reduce (~38%) — BLAS-2 per-reflector wall; the subtract-fuse was
already measured to regress (it is a matvec wall, not a subtract wall).

## NEXT PRIMITIVE (the alien-artifact the no-ceiling addendum demands)
Replace the BLAS-2 reductions with BLOCKED two-sided panel reductions — the genuine
algorithmic lever, multi-hour:
- **blocked dgehrd** (Hessenberg): LAPACK dlahr2 panel — accumulate NB reflectors
  with a compact-WY T and apply the trailing two-sided update as BLAS-3 GEMMs. The
  current hessenberg_reduce_blocked already does a partial version but its panel is
  still a per-reflector matvec (Y = H0·v_c, sequential within the panel). The real
  dlahr2 keeps the panel BLAS-2 but is structured so the *trailing* update is one big
  GEMM; the win is making the panel matvecs themselves a GEMV-batched/parallel form.
- **blocked dgebrd** (SVD bidiagonalization) and **blocked dsytrd** (eigh
  tridiagonalization): same compact-WY two-sided story under the qgce4 tolerance
  policy. These REPLACE the O(n^3) BLAS-2 reductions with BLAS-3, the only remaining
  structural lever for the eig/SVD reduction phases. Target: 2-4x on the reduction.

Everything cheaper than that has been done or proven fork/join-/BLAS-2-bound.

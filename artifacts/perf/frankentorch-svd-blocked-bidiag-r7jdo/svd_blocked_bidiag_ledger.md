# frankentorch-svd-blocked-bidiag-r7jdo - blocked bidiagonalization for SVD

## Claim

Shipped. `golub_reinsch_svd_impl` now reduces to bidiagonal form via a blocked
`dgebrd` path instead of the in-place Numerical-Recipes sweep. Measured
`2.9-4.6x` end-to-end on the intra-repo A/B at `N >= 256`.

This is a DEFENSIVE win, per the bead's `no-win` label. FT `svd` remains slower
than LAPACK at every size measured; the value is retiring a catastrophic gap and
making `N >= 512` usable. Commits `d4efd4a2`, `3dda5575`.

## Baseline correction: the 189x row did not reproduce

The bead description carries `MEASURED (ledger 21bl): FT linalg svd N=256 =
1874ms vs PyTorch(LAPACK) 9.9ms = 189x SLOWER (N=512+ timed out)`.

**That row was contention-inflated and is corrected here.** Re-measuring the
unchanged incumbent on a quiet worker:

| | 189x row | quiet-host re-measure (incumbent) |
|---|---|---|
| FT `svd` N=256 reduced | 1874 ms | **115-154 ms** |
| PyTorch N=256 reduced | 9.9 ms | **10.29 ms** |
| ratio | 189x | **11-15x** |
| FT `svd` N=512 | "timed out" | **835-1418 ms** (completes) |

Read this plainly:

- **The PyTorch side never moved.** 9.9 ms then, 10.29 ms now — LAPACK is doing
  exactly what it always did. There is no upstream regression here, and none
  should be inferred from the corrected ratio.
- **The FT side was mismeasured, not regressed-then-fixed.** The 12x spread
  between 1874 ms and 115-154 ms is worker contention on the shared fleet, the
  same confound already recorded against `frankentorch-1q8x` and
  `frankentorch-66pe`. The incumbent code at the time of the 189x row and the
  incumbent code re-measured here are the same algorithm.
- **`N=512` never actually "timed out"** as a property of the code; it did not
  finish inside the harness window on a contended worker. It completes in
  ~0.8-1.4 s on a quiet one, pre-change.

So the honest framing of this bead's opportunity was never "189x". It was
"11-15x, and growing with N". The work below is scored against the quiet-host
incumbent, not against the 189x row.

## Profile target

Phase split of the incumbent `golub_reinsch_svd_impl`, square f64, 16 threads:

| phase | N=256 | N=512 |
|---|---|---|
| reduce (NR Householder) | ~40 ms | ~271 ms |
| V back-accumulation | ~28 ms | ~202 ms |
| QR sweep | ~3.5 ms | ~14 ms |
| rotation replay | ~2.2 ms | ~20 ms |

Reduce + back-accumulation is `~92%` of the N=512 cost. That is the target.

## Lever

Blocked `dgebrd` panel (`dlabrd` + two accumulate-GEMMs) replacing the BLAS-2
reduction, plus `dorgbr`-shaped materialisation of `Q`/`P` replacing the
back-accumulation.

Two things had to be fixed first, or the wiring would have REGRESSED — see the
negative-evidence section.

## Intra-repo A/B — NOT head-to-head

**These numbers compare FrankenTorch against FrankenTorch.** They are not a
comparison against PyTorch and must never be quoted as one. Both arms are the
same ELF on the same remote worker under the same load: the probe
(`examples/svd_bidiag_phase_probe`) re-executes itself once per arm, because the
reduction choice is latched once per process. `FT_SVD_FORCE_NR=1` selects the
incumbent. Every child runs an A/A null gate. Two interleaved rounds:

| shape | incumbent (NR) | blocked (dgebrd) | ratio |
|---|---|---|---|
| N=256 reduced | 154 / 115 ms | 47 / 45 ms | ~2.9x |
| N=512 reduced | 835 / 1418 ms | 236 / 311 ms | ~3.4-4.6x |
| N=512 full | 1366 / 1744 ms | 348 / 382 ms | ~3.9-4.6x |
| N=512 svdvals | 383 / 769 ms | 132 / 124 ms | ~2.9-6.2x |

A/A skew ran 3-27%, with one 54% outlier at N=128 where both arms are
single-digit ms and the probe does not discriminate. The worker is contended;
the A/B ratio at `N >= 256` sits far outside that band, which is the only reason
these are claimable at all.

Component measurements behind the total, same conditions:

| component | before | after |
|---|---|---|
| `bidiag_form_q` N=512 | 1068 ms | 51 ms |
| `bidiag_form_p` N=512 | 769 ms | 40 ms |
| blocked reduce N=512 | 287 ms | ~125 ms |

Panel width `nb = 16`, chosen by measurement over 32 and 64 at both sizes.

## Head-to-head standing vs PyTorch/LAPACK

**This is the honest h2h number, and it is a loss at every size.** Mixed-location
ratio (FT on the remote worker, PyTorch CPU on the local host, both 8 threads,
warm — torch warmed 5 iterations before timing, per the cold-read confound):

| shape | FT (blocked) | PyTorch | standing |
|---|---|---|---|
| N=256 reduced | ~46 ms | 10.29 ms | **4.5x slower** |
| N=512 reduced | ~270 ms | 51.45 ms | **5.2x slower** |
| N=512 full | ~365 ms | 50.80 ms | **7.2x slower** |
| N=512 svdvals | ~128 ms | 22.91 ms | **5.6x slower** |

Prior standing at N=256 reduced was 11-15x slower (quiet-host incumbent, above).
So the bead moved `svd` from roughly 12x to roughly 4.5x at N=256 — real, and
still a loss. No vs-PyTorch win is claimed for this bead and none is expected;
LAPACK remains the gold standard here.

## Negative-evidence ledger

- **The wiring as originally specified would have regressed.** Timing the phases
  before writing it: at N=512 the incumbent `reduce + vaccum` was 473 ms, while
  blocked `reduce + form_p` would have been 1056 ms. Do not wire a blocked
  reduction without first checking what the back-transform costs.
- **`bidiag_form_q`/`form_p` were the larger of the two blockers.** Serial,
  strided by `lda`, and updating all `n` columns when only columns `>= i` can be
  nonzero — `Q`/`P` are built from the identity walking `i` downwards, so every
  column below `i` is still `e_c` with its lone 1 above row `i`. Fixed: ~19x.
- **BLAS-3 alone bought nothing.** With `dlabrd_panel_f64`'s two
  `O(m_sub * n_sub)` matvecs still serial, the blocked reduce merely MATCHED the
  already-rayon-parallel NR reduce at N=512 (287 vs 271 ms). The panel, not the
  GEMM, was the wall. Anyone repeating this on `dsytrd` (`frankentorch-t0b4l`,
  the same scalar-reduction wall for `eigh`) should expect the same and
  parallelize the panel matvecs as part of the lever, not after it.
- **Replay row-block: larger blocks REFUTED.** With the reduction blocked, the
  rotation replay became the largest single phase, so its row-block width — 8,
  a value the source comment says was carried over from the eigh QL replay and
  never re-measured for SVD — was swept in one process via
  `set_svd_qr_replay_block_override` (`examples/svd_replay_block_ab`, 16
  threads, anchor `block=1`):

  | n | mode | b=2 | b=4 | b=8 | b=16 | b=32 | b=64 | b=128 | b=256 |
  |---|---|---|---|---|---|---|---|---|---|
  | 512 | reduced | 1.09x | 1.25x | **1.31x** | 0.92x | 0.86x | 0.76x | 0.55x | 0.35x |
  | 512 | full | 1.18x | 1.33x | **1.41x** | 0.96x | 0.80x | 0.63x | 0.44x | 0.27x |
  | 1024 | reduced | 1.49x | **1.80x** | 1.62x | 1.24x | 1.20x | 1.18x | 0.78x | 0.54x |
  | 1024 | full | 1.29x | **1.50x** | 1.46x | 0.86x | 0.85x | 0.79x | 0.53x | 0.32x |
  | 2048 | reduced | 1.40x | **1.76x** | 1.62x | 0.86x | 0.77x | 0.91x | 0.91x | 0.51x |
  | 2048 | full | 1.80x | **2.29x** | 2.00x | 0.94x | 1.09x | 1.08x | 1.10x | 0.62x |

  The hypothesis was that wider blocks would pay by re-reading the multi-MB op
  stream fewer times. It is wrong, and sharply so — everything at `b >= 16`
  regresses, down to `0.27x` at `b=256`. The op stream is not the binding
  constraint; block width past ~8 costs more in lost parallelism and cache
  residency than it saves in streaming.

  The incumbent `b=8` is optimal at `n=512`. `b=4` is better at `n >= 1024`
  (1.09-1.15x over `b=8`), which is a real but sub-threshold effect: below the
  `Score >= 2.0` bar, so NO code change was made and the shipped default stays
  `8`. **Retry predicate:** revisit only if the replay is restructured so the op
  stream is traversed differently (e.g. a compressed op encoding, or an
  `n`-dependent block), not by re-running this sweep — the sweep is done and the
  answer will not change.

## Method notes

- Both prologues feed one shared `svd_bidiag_qr_f64`; the QR recurrence reads
  only `w`/`rv1`, so it is indifferent to which reduction produced them.
- The `~105`-line NR reduction that was duplicated verbatim between
  `golub_reinsch_svd_impl` and `golub_reinsch_singular_values` is now one
  `svd_nr_reduce_f64`. That de-duplication is load-bearing, not tidying:
  `svd_tall` documents its singular values as BIT-IDENTICAL to
  `svdvals_contiguous_f64`, so both entry points had to take the same branch.
  Wiring only the full SVD would have broken that contract silently, since the
  two spectra still agree to working precision. Pinned by a `to_bits()` test.
- Index convention is the one thing that fails silently: LAPACK holds the
  superdiagonal as `e[i]` at `(i, i+1)`, NR as `rv1[i]` at `(i-1, i)` with
  `rv1[0]` unused. A wrong shift still yields a plausible descending spectrum.
  Isolated in `svd_bidiag_to_nr_indexing`.
- Singular VECTOR outputs are compared by reconstruction/orthogonality within
  `1e-9` per the ratified tolerance-parity policy `frankentorch-qgce4`, which is
  what makes a rounding-reordering rewrite admissible at all.

## Gates

`ft-kernel-cpu --lib` 592 passed / 0 failed (3 new tests). `ft-conformance` all
green. `ft-api --lib` 2484 passed / 0 failed. `ft-nn --lib` 778 passed / 0
failed. `cargo check --workspace --all-targets` clean. `cargo fmt` clean.
`clippy -D warnings` zero findings in the added ranges; the 8 residual crate
findings each `git blame` to other commits.

## Next target

With the reduction blocked, the bidiagonal-QR Givens replay is the dominant
phase (N=512: 68-150 ms of the ~300-380 ms total) and the block-width lever on
it is now closed by the sweep above. A further win there needs the algorithm
changed, not tuned — LAPACK's answer is divide-and-conquer `dbdsdc`, a
multi-session rewrite. Separately, `dlabrd` steps (1)/(14)/(16) remain serial at
`O(m_sub * nb)`, a deliberately-left minor term.

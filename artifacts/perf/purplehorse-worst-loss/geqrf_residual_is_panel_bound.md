# `geqrf`'s residual 14.743x is PANEL-bound, not GEMM-bound — and three protocols gave three different "best nb"

**Result: at the shipped `nb = 32`, the recursive BLAS-2 panel factorization is ~55% of
blocked `geqrf` and the trailing GEMM update is ~38%. Blocking has already done its job;
the remaining gap needs a faster PANEL algorithm, not better GEMM scheduling and not
another constant. `nb = 16` measures 1.057x faster than the shipped 32 on min-of-7 — real
but small, and NOT shipped on this evidence.**

Probe: `geqrf_stage_attribution_and_nb_ladder` (ft-kernel-cpu), n=512, min-of-7 with the nb
ladder interleaved inside the round loop.

| nb | min-of-7 wall | panel+T | trailing_R | reverse_Q |
|---|---|---|---|---|
| 8 | 24.545 ms | 31.9% (7.828 ms) | 60.6% (14.869 ms) | 0.0% |
| **16** | **23.440 ms** | 43.2% (10.121 ms) | 48.8% (11.438 ms) | 0.0% |
| 32 (shipped) | 24.775 ms | 54.8% (13.570 ms) | 37.6% (9.308 ms) | 0.0% |
| 64 | 30.696 ms | 68.7% (21.083 ms) | 25.6% (7.873 ms) | 0.0% |

`reverse_Q = 0.0%` confirms `geqrf` correctly skips the Q build.

## The finding

`geqrf` at n=1024 is 130.383 ms against torch's 9.835 ms — ~11 GFLOPS vs ~143. That gap is
far too large for panel-width tuning, and the attribution says why: **the panel is the
wall.** `trailing_R` is already down to 37.6%, which is blocking working as intended and is
exactly why the re-route delivered 21.2x / 36.3x. What remains is BLAS-2 panel work that
blocking cannot touch.

This puts `geqrf`'s residual in the same category as `eigh`'s: **algorithmic**, not a
dispatch defect. The Householder family started as three private per-reflector loops that
never reached an existing kernel — that part is fixed and measured. The remainder is a
different kind of problem.

The share structure is monotone and reproducible across every protocol run:
`panel+T` 31.9 -> 43.2 -> 54.8 -> 68.7%, `trailing_R` 60.6 -> 48.8 -> 37.6 -> 25.6%.

## THREE PROTOCOLS, THREE WINNERS — the estimator was the instrument

| protocol | "best" nb |
|---|---|
| single shot per nb, run 1 | **32** |
| single shot per nb, run 2 | **8** |
| min-of-7, interleaved | **16** |

`nb = 16` was the WORST entry in both single-shot runs (34.249 ms, 52.552 ms) and the BEST
on min-of-7 (23.440 ms). Those two runs were contended — both stages inflated together by
~1.5x — and I read them as signal.

**I declared "nb=32 is already correctly tuned, the tuning lever is refuted" on a single
shot per nb.** That was wrong twice over: wrong conclusion, and wrong method. Every
vs-torch row in this campaign is min-of-N, paired, interleaved, gated on an A/A null — and
I skipped all of it here because the comparison was FT-vs-FT, as though internal
comparisons were exempt from the noise that governs external ones. A 1.06x effect needs the
same machinery as a 1.06x ratio.

The correction after run 2 ("min-of-2 favours nb=8") was also wrong. Only the interleaved
min-of-7 is trustworthy, and it names a third answer.

## Why nb=16 is NOT being shipped

* 1.057x is below the bar this campaign holds for a landed change.
* It is FT-vs-FT on a shared remote worker, not in situ against a live incumbent — and a
  standalone ladder has INVERTED in situ before in this codebase (a predicted 5.7x win
  measured as a 1.118x regression).
* `nb` is shared with `tensor_linalg_qr`, so the change moves an op that is not the one
  under study.

Filed as an observation with its measurement protocol, not as a pending change.

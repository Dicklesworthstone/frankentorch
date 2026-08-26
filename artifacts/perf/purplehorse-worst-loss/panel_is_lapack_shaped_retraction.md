# RETRACTION: the recursive panel IS BLAS-3. What is actually true about the remaining gap.

## The retraction

I reported, from a grep, that the recursive panel's combine step
(`qr_apply_panel_block_reflector_f64`) had "no `dgemm` at all — hand-written nested loops",
and called it a concrete lever on the worst loss.

**That is false.** My grep window was `+45` lines and the `gemm::dgemm` calls begin at about
`+47`. The nested loops I saw are STAGING — building `vblock`, `vt` and `panel_cols` — which
precede the GEMM calls. Reading the whole function shows the combine is genuine BLAS-3.

This is the second time in this campaign that a truncated view produced a confident wrong
claim about absence (the first nearly reported "LLVM won't vectorize the row-dot" on code
that is hand-vectorized with `wide::f64x4`). **Absence is the one thing a partial read
cannot establish.**

## What is actually true

**The panel is LAPACK-shaped.** `qr_factor_panel_recursive_f64` implements `dgeqrt3`: split
at mid, recurse on the left half, apply that half's block reflector to the right half via
GEMM, recurse on the right. `LEAF = 8`. There is no structural defect to fix here.

**`geqrf`'s residual is panel-bound** — measured, reproducible across protocols:

| nb | min-of-7 wall | panel+T | trailing_R |
|---|---|---|---|
| 8 | 24.545 ms | 31.9% | 60.6% |
| 16 | 23.440 ms | 43.2% | 48.8% |
| 32 (shipped) | 24.775 ms | 54.8% | 37.6% |
| 64 | 30.696 ms | 68.7% | 25.6% |

**`slogdet` IS bare getrf** — same process, min-of-9, interleaved, both warmed:

```
LU_ATTRIB n=512  lu_factor=9.798ms  slogdet=9.692ms  ratio=0.9892x
```

so the LU family's gap is its panel too, and there is no slogdet-specific lever.

## One architectural observation, NOT yet a measured cause

`gemm::dgemm(m, k, n, a, b, c)` takes **no leading-dimension parameter**, so operands must
be packed contiguous. Every blocked update therefore stages submatrices into packed buffers
and copies results back — the trailing update copies `rt` (m x nt) per panel, and each
panel combine copies `panel_cols` plus a `vt` transpose per recursion level.

A strided GEMM (an `ld` parameter) would remove those copies. **I have NOT measured what
share they represent**, and a rough estimate puts the trailing copies at only a few ms of a
24 ms lane — meaningful but not obviously dominant. Calling this THE bottleneck would be
exactly the kind of unmeasured assertion the retraction above was.

## Where this leaves the worst loss

`geqrf` 14.743x (n=1024) and `slogdet` 14.715x (n=512) are both blocked factorizations whose
residual is panel-bound, with a correctly-shaped recursive panel. No cheap lever remains:
the candidates are a strided GEMM (architectural, touches every blocked kernel) or a faster
panel leaf. Both are substantial, and neither should be started on an unmeasured hypothesis.

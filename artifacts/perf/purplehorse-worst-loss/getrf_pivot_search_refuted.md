# REFUTED: getrf's panel cost is NOT the strided pivot search (5.3-8.1% of panel)

**Bead frankentorch-e1isq.** Measured through the harness lane (`tensor_linalg_slogdet`) at
the production 8-thread width.

```
n= 512  pivot_search 0.42ms (3.5% of wall,  5.3% of panel)  row_swap 0.01ms (0.1%)
        wall 12.06ms  panel 65.8%  solve 8.1%  trailing 9.5%  UNACCOUNTED 16.6%
n=1024  pivot_search 1.84ms (5.2% of wall,  8.1% of panel)  row_swap 0.02ms (0.1%)
        wall 35.25ms  panel 64.0%  solve 11.0% trailing 13.1% UNACCOUNTED 11.8%
```

## The hypothesis, and why it was wrong

getrf's pivot search is `lu[i * n + k]` — a strided column scan in ROW-MAJOR storage. At
n=1024 consecutive elements are 8 KB apart, so I argued it was a cache (and TLB) miss per
element, ~n^2/2 ~ 524K of them, which at DRAM latency is ~52 ms — the same order as the
whole panel. LAPACK is column-major and gets that scan contiguous, so this looked like a
storage-layout articulation point that no amount of recursion could fix.

**It is 5.3-8.1% of the panel.** The miss-per-element model overestimated by an order of
magnitude: a constant-stride scan is exactly what a hardware prefetcher handles well, and
the reasoning ignored that.

This kills the packed-column-major-panel lever BEFORE it was built. That is the whole point
of measuring the mechanism rather than shipping on the arithmetic.

## What that leaves

~92% of the panel is the elimination work itself, which the recursive `dgetrf2`
(78cf5eea, `LEAF = 16`, `NB = 128`) DOES convert to BLAS-3 — leaf work is only ~12% of panel
flops at those widths. And it produced NO vs-torch gain (cc270881: 11.244x -> 11.445x,
certified).

The panel does ~134 MFLOP in 22.58 ms at n=1024, i.e. **~6 GFLOPS** — poor for BLAS-3. The
leading remaining explanation is that the recursion's combines are too narrow to reach the
microkernel's efficient stride. That is the same width-amortisation effect that already
killed three levers in this campaign:

  * `dgemm_sub_into` in the geqrf panel combine — 1.059x SLOWER (N <= 32 columns, barely one
    parallel window)
  * eigh backtransform update loop — 1.3x SLOWER (~640 dispatches over shrinking row sets)
  * eigh reduction ungated at n=256 — 2.35x SLOWER

## Also worth noting

UNACCOUNTED (11.8-16.6%) now exceeds pivot+swap by 2-3x, so there is real cost outside all
four instrumented regions — API/tape work around the kernel, not the factorisation.

## Status of the bead

getrf/slogdet remains the worst measured ratio (11.445x @ n=1024 CERTIFIED, 14.4x @ n=512).
Three structural hypotheses are now measured shut: the trailing update (already BLAS-3,
4.6-13.1%), the recursive panel (no gain), and the pivot search (5-8% of panel). The next
candidate is combine width, not storage layout.

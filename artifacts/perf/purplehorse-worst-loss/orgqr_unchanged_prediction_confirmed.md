# `orgqr` after the geqrf re-route: prediction CONFIRMED (unchanged), and the ratio moved anyway

**Result: `orgqr` n=512 reads 143.860x / 142.336x against a banked 125.2x. Read naively
that is a 1.15x REGRESSION. It is not one — torch is 1.21x faster in this window, and our
absolute time is unchanged. A/A null 1.000 / 1.004 (CERTIFIED), parity 8.96e-13 MATCH.
ELF `00c39a9e2c6236fa1888fa06a131613da61ba56f5ecf73df054b7962f80a7d79`.**

## The prediction

Stated before measuring, when reporting the geqrf win:

> `orgqr` and `ormqr` consume these same packed reflectors and were NOT touched. Whether
> they move is a prediction to measure, not to assume.
> My prediction: they are UNCHANGED — the harness runs the producing geqrf OUTSIDE their
> timer, and their cost is their own private `apply_reflector_left/right` loops.

## The row, and why the ratio is misleading

| n | FT min (arm0/arm1) | PT min | ratio | banked ratio | A/A null | parity |
|---|---|---|---|---|---|---|
| 512 | 798.185 / 797.927 ms | 4.872 ms | 143.860 / 142.336x | 125.2x | 1.000 / 1.004 | 8.96e-13 |

`incumbent_check n=512 PT_min=4.872ms banked~5.899ms -> ok`

**Torch is 1.21x FASTER in this window than when the baseline was banked.** With an
unchanged numerator, a 1.21x faster denominator predicts `125.2 * 1.21 = 151x`; measured
143.9x. Our own absolute went 738 ms (implied at banking) -> 798 ms, i.e. 1.08x, which is
window noise.

**So the numerator did not move and the prediction holds.** Reporting "orgqr regressed
1.15x" would have been reading a ratio without asking which end moved — the exact error
this campaign has been built to avoid, and it would have manufactured a regression out of
a faster incumbent.

## Instrument note

`incumbent_check` returned **ok** on a 21% incumbent shift. Its tolerance is wide by
design — it exists to catch a grossly scaled incumbent (the failure an A/A null is
structurally blind to), not drift of this size. A row still needs its absolute times read
before any cross-run claim.

## Consequence

The model is confirmed: `orgqr`'s cost is its own private per-reflector BLAS-2 loops, which
the geqrf re-route did not touch. **`orgqr` is now the largest remaining single-matrix
defect** (125-144x at n=512, 312-317x at n=1024).

The lever is scoped and reuses machinery that already exists:
  1. unpack packed A + tau into an m x k vmat with explicit unit diagonal and zeros above
     — O(m*k);
  2. build the compact-WY T per 32-column panel with `qr_build_compact_wy_t_f64`;
  3. run the reverse dorgqr that is already the second half of
     `qr_householder_panel_blocked_profiled`.

Step 2 is safe because `qr_build_compact_wy_t_f64` is CONVENTION-GENERIC: it uses only the
dlarft recurrence `T[i][c] = -tau_c * (v_i . v_c)` and never assumes `v_j = 1` or
`tau = 2/||v||^2`, so it accepts LAPACK-convention `(u, tau_L)` directly. Verified by
reading it, not assumed.

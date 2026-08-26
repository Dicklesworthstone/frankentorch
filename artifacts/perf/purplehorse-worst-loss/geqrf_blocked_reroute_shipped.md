# `geqrf` re-routed to the blocked kernel: 227.6x -> 10.755x and 535.2x -> 14.743x

**Result: the campaign's largest single-matrix defect is fixed. 21.2x improvement at
n=512 (CERTIFIED) and 36.3x at n=1024 (measured). Parity MATCH at both sizes.
ELF `00c39a9e2c6236fa1888fa06a131613da61ba56f5ecf73df054b7962f80a7d79`.**

| n | before | after (arm0/arm1) | FT abs | PT min | A/A null | parity | status |
|---|---|---|---|---|---|---|---|
| 512 | 227.6x | **10.755 / 10.902x** | 25.490 ms | 2.156 ms | 1.000 / 0.985 | 3.85e-13 | **CERTIFIED** |
| 1024 | 535.2x | **14.743 / 15.259x** | 130.383 ms | 9.835 ms | 1.000 / **0.971** | 7.73e-13 | measured |

`incumbent_check` PASSED at both sizes (PT 2.156 vs banked ~2.201; 9.835 vs banked ~9.468),
which is what licenses comparing against the banked baseline at all. An A/A null cannot
detect a uniformly scaled incumbent, so without this the before/after would be a guess.

## The internal consistency check, which is stronger than the ratio

At n=512 our `geqrf` is now **25.490 ms** against our own `qr` at **40.707 ms**. `geqrf`
does strictly LESS work than `qr` (no Q is formed), so it must be faster. Before this
change `geqrf`+`orgqr` took 1337.4 ms — 32.9x SLOWER than the op doing more work. **The
ordering is now correct**, and that is evidence independent of any torch comparison that
the factorisation is real rather than merely fast.

Our ratio (10.755x) remains above `qr`'s (5.50x) only because torch's own `geqrf` is ~3.4x
faster than torch's `qr` (2.156 ms vs ~7.40 ms implied). The denominator moved, not our
numerator.

## What the fix actually was — NOT what the bead said

The bead described "a re-route plus an exposure: the blocked kernel already computes V and
tau then discards them". **Both halves of that were wrong**, and either would have shipped
silently incorrect reflectors:

1. **tau does not survive at all.** `qr_householder_panel_blocked_profiled` did
   `panels.push((nb, vmat, tmat))` — it keeps V and the compact-WY T, and drops the tau
   vector the moment T has been built from it. Retaining tau is the one real change the
   kernel needed.

2. **The conventions differ.** The blocked kernel stores `v_j = alpha + s*||x||` explicitly
   with `tau_b = 2/||v||^2`; LAPACK implies `v_j = 1` with `tau_L = (beta-alpha)/beta`.
   They describe the same reflector under

       u = v / v_j,   tau_L = tau_b * v_j^2

   because `H = I - tau_b*v*v^T` with `v = v_j*u` gives `H = I - (tau_b*v_j^2)*u*u^T`, and
   `||v||^2 = 2*||x||*v_j` makes that scalar `v_j/||x||` = `tau_L`. The vectors coincide
   term for term since `v_i/v_j = a[i][j]/(alpha - beta)` is LAPACK's own expression.

Had I trusted the bead's summary, `orgqr`/`ormqr` would have consumed reflectors in the
wrong convention and returned a wrong Q — fast and wrong.

## Not bit-identical, and the reason is specific

The two paths accumulate the column norm in DIFFERENT ORDERS: the blocked leaf sums
`alpha^2` FIRST (`nrm2` runs over the whole column), `dlarfg` adds it LAST. So `||x||` can
differ in the last ulp, and with it `v_j`, `tau` and every `v_i`.

This is acceptable because the bar was established BEFORE the fix, not after: the existing
geqrf goldens assert `1e-11` against torch and are 4x3, so they stay on the naive path
below the `m >= 128` gate and are unaffected.

## The test that carries the weight

`geqrf_blocked_f64_reflectors_are_lapack_convention` (ft-kernel-cpu) rebuilds Q FROM THE
RETURNED REFLECTORS — `v_j = 1` implied, `v_i` read from below the diagonal — and asserts
`Q R == A` at 160x32. 1 passed, 0 failed.

An R-only comparison could not have validated this: R falls out of the forward pass
untouched by the convention conversion, so it would look correct even with `tau_L`
completely wrong. Only reconstructing Q exercises the conversion.

**A test I wrote earlier LOST its power with this commit.**
`geqrf_and_qr_agree_on_r_above_the_blocked_gate` was naive-geqrf vs blocked-qr when
written; after the re-route both entry points call the same kernel, so it is now
blocked-vs-blocked and compares one implementation with itself. It stays as a packing
regression guard. Flagged in the commit message so the next reader does not mistake it for
convention coverage.

## Route proof standard, stated honestly

The `qr` route at 160x32 was established by reading the call chain to the
`m >= 128 && k >= 16` gate and checking the arithmetic — NOT by a sentinel poison-return.
That is weaker than the standard applied elsewhere in this campaign. The measurement itself
is now the strong evidence: a 21-36x shift in ratio and a 52x drop in absolute time cannot
happen without the route changing.

## Remaining

`orgqr` (125.2x / 312-317x) and `ormqr` (~253x / 502-507x) consume these same packed
reflectors and were NOT touched. Whether they move is a prediction to measure, not to
assume.

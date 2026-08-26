# `orgqr` blocked: 143.9x -> 4.809x, and 30.9x faster in ABSOLUTE time. Both sizes CERTIFIED.

**Result: `orgqr` n=512 goes 798.185 ms -> 25.840 ms (30.9x) and its ratio 143.9x ->
4.809x/5.009x; n=1024 -> 6.492x/6.465x. All four A/A nulls in band, so BOTH sizes are
CERTIFIED. Parity 8.98e-13 and 2.96e-13 MATCH.
ELF `2a1cf8a07b4e1daba9b1241e5a70950d338b5dbd0395efe995d4898386ae1400`.**

```
FT_OP=orgqr FT_ROUNDS=25 RAYON_NUM_THREADS=8 FT_GATE_SIZES="512,1024" \
FT_GATE_VALUES="262144,262144" PYTORCH_PYTHON=/data/tmp/torchvenv-2121/bin/python
```

| n | FT min (arm0/arm1) | PT min | ratio | banked | A/A null | parity |
|---|---|---|---|---|---|---|
| 512 | 25.840 / 25.774 ms | 4.459 ms | **4.809 / 5.009x** | 125.2x / 143.9x | 1.000 / 0.989 | 8.98e-13 |
| 1024 | 129.296 / 106.954 ms | 20.933 ms | **6.492 / 6.465x** | 312-317x | 1.000 / 1.002 | 2.96e-13 |

## The absolute time is the claim, not the ratio

798.185 ms -> 25.840 ms at n=512, same host, same harness, ELF sha recorded on both runs.
A 30.9x absolute drop cannot be produced by incumbent drift: torch moved 1.09x between the
two windows (4.872 -> 4.459 ms), which is two orders of magnitude too small to matter here.

This is deliberately how it is reported. The PREVIOUS orgqr row in this campaign looked
like a 1.15x REGRESSION (125.2x banked -> 143.9x measured) purely because torch was 1.21x
faster in that window; the numerator had not moved at all. Reading the ratio without asking
which end moved would have manufactured a regression then, and would overstate the win now.

## What changed

`tensor_householder_product` applied ONE reflector per pass via `apply_reflector_left` — a
rank-1 update walking `a[i * n + kk]` with stride n, one cache line per useful element, once
per reflector. `orgqr_blocked_f64` collapses each 32-reflector panel into a single
`I - V T Vᵀ` applied as three GEMMs, so the work becomes BLAS-3.

It reuses `qr_build_compact_wy_t_f64` UNCHANGED. That was safe because the builder is
convention-generic — pure dlarft recurrence `T[i][c] = -tau_c * (v_i . v_c)`, with no
`v_j = 1` or `tau = 2/||v||^2` assumption — so it accepts LAPACK-convention reflectors
directly once the unit diagonal is made explicit in `vmat`. Verified by reading it, which
is the same check that caught the geqrf convention mismatch before it cost a build.

## Two reservations I stated before measuring, both WRONG

Recorded here because they were published in advance, not reconstructed after:

1. *"`Q` is m x n, so the block GEMMs are m x nb x n; a narrow Q would degenerate toward
   BLAS-2."* It did not bite at n=512/1024.
2. *"`orgqr_blocked_f64` materialises an m x nb vmat per panel with an explicit unit
   diagonal, which geqrf got free from its forward pass — O(m*k) of copying the naive path
   never does."* Also did not bite.

Both were real risks and neither materialised. Stating them beat assuming geqrf's 21x would
simply transfer — the shape argument for orgqr genuinely differs from geqrf's.

## Internal consistency

Our `geqrf` + `orgqr` is now 25.490 + 25.840 = 51.33 ms against our own `qr` at 40.707 ms,
a 1.26x gap — about right for a two-step that redoes work the fused path shares (orgqr
re-derives V from the packed form that geqrf just wrote).

**That same comparison was 32.9x this morning** (1337.4 ms vs 40.707 ms). The whole
Householder family defect was visible from this one internal relationship, with no
reference implementation involved.

## Coverage

Correct at BOTH levels, and the second was a gap I nearly shipped without:

* kernel — `orgqr_blocked_f64_rebuilds_q_from_packed_reflectors` checks `Qᵀ Q == I` AND
  `Q R == A`. Both needed: orthonormality alone passes for reflectors applied in the WRONG
  ORDER, which is exactly the error blocking introduces.
* ft-api routing — `routed_orgqr_matches_qr_q_above_the_blocked_gate`. The two existing
  `householder_product` goldens are 4x3, BELOW the gate, and reported "2 passed" against
  the routed build while exercising none of it.

## Family standing

| op | before | after |
|---|---|---|
| `geqrf` | 227.6x / 535.2x | **10.755x / 14.743x** (validated) |
| `orgqr` | 125-144x / 312-317x | **4.809x / 6.492x** (CERTIFIED both sizes) |
| `ormqr` | ~253x / 502-507x | untouched — now the worst remaining |

`ormqr`'s block structure is already derived on the bead (four cases; panel order REVERSE
iff `left != transpose`, T transposed iff `transpose`), cross-checked against the shipping
code's own `forward = (transpose == left)` flag. This measurement is what justifies writing
it: the identical transformation just paid 31x on a simpler shape.

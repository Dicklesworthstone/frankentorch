# `ormqr` blocked: ~253x -> 8.696x and 502-507x -> 11.176x, ~58x faster in absolute time. Both sizes CERTIFIED.

**Result: parity RESTORED (7.20e-13 and 1.54e-12 MATCH) and both sizes certified. The
n=1024 parity residual is bit-identical to the pre-fix naive baseline's 1.54e-12, which is
the end-to-end confirmation that the blocked QR leaf fix (`380699a2`) was the cause.
ELF `dee6f0fd41a452d5c6c8e89281ef34fa296544f272ce055b5946f42383a6048e`.**

```
FT_OP=ormqr FT_ROUNDS=25 RAYON_NUM_THREADS=8 FT_GATE_SIZES="512,1024" \
FT_GATE_VALUES="262144,262144" PYTORCH_PYTHON=/data/tmp/torchvenv-2121/bin/python
```

| n | FT min (arm0/arm1) | PT min | ratio | banked | A/A null | parity |
|---|---|---|---|---|---|---|
| 512 | 26.079 / 26.000 ms | 2.779 ms | **8.696 / 8.443x** | ~253x | 1.000 / 0.999 | 7.20e-13 MATCH |
| 1024 | 124.749 / 134.093 ms | 13.552 ms | **11.176 / 10.175x** | 502-507x | 1.000 / 1.002 | 1.54e-12 MATCH |

`incumbent_check` passed at both sizes (2.779 vs banked ~2.687; 13.552 vs ~14.299).

## Absolute time, with its caveat stated

n=1024: **7198.162 ms -> 124.749 ms, ~58x**. Torch moved 1.06x between the two runs, so no
incumbent-drift story explains it.

**But this is not a clean A/B and I am not quoting it to three digits.** The pre-fix
baseline was measured against a `geqrf` that carried the spurious `tau = 2` reflector, so
the two runs' INPUTS differed. The magnitude is sound; the precision is not.

## The parity number is the real result

The previous attempt at this row read **6.13e-4 / 1.53e-3 MISMATCH** with timings that
looked excellent (8.060x / 10.452x). Those timings were void — the values were wrong. That
mismatch is what exposed a pre-existing row-negation bug in the blocked QR leaf, which
`tensor_linalg_qr` had been carrying for every SQUARE input at `m >= 128`.

Now: **1.54e-12 at n=1024, the same residual the per-reflector path produced before any of
this work.** Not merely "within tolerance" — the identical value, which is what end-to-end
correctness through a sign-sensitive lane looks like.

## Householder family, complete

| op | before | after | speedup |
|---|---|---|---|
| `geqrf` | 227.6x / 535.2x | **10.755x / 14.743x** | 21.2x / 36.3x |
| `orgqr` | 125-144x / 312-317x | **4.809x / 6.492x** | 30.9x absolute |
| `ormqr` | ~253x / 502-507x | **8.696x / 11.176x** | ~58x absolute |

All three were ONE defect: private per-reflector BLAS-2 loops in `ft-api` that never
reached the blocked compact-WY kernel `tensor_linalg_qr` already used. All three now share
`householder_panels_from_packed_f64` and apply one `I - V T Vᵀ` per 32-reflector panel as
three GEMMs.

Internal consistency, which needs no torch: `geqrf` + `orgqr` is now 51.33 ms against our
own `qr` at 40.707 ms (1.26x, the expected cost of a two-step redoing shared work). That
same comparison was **32.9x** at the start of this campaign.

## What the campaign cost and returned

The `ormqr` re-route shipped broken (`984cb985`) because its tests could not fail the way
the code could break: all fixtures TALL (no column with an empty below-diagonal), two
SINGLE-PANEL (N=32 against nb_block=32, where REVERSE and FORWARD traversal are identical).
That regression is what surfaced the older leaf bug. The trade was worth making, but the
fixtures should have been built that way first — checklist now in
`blocked_qr_leaf_row_negation_bug.md`.

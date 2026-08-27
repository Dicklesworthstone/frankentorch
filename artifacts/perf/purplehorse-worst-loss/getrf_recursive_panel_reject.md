# REJECT: recursive getrf panel + NB=128 give NO vs-torch gain — and my FT-vs-FT probe was measuring a different lane

**Bead frankentorch-e1isq.** Measured live vs PyTorch 2.12.1+cpu, co-process in the same
invocation.

```
FT_OP=slogdet FT_ROUNDS=64 RAYON_NUM_THREADS=8 FT_GATE_SIZES="512,1024" \
FT_GATE_VALUES="262144,262144" PYTORCH_PYTHON=/data/tmp/torchvenv-2121/bin/python
in-process ELF sha256 = dd5609c32a890c2c2a75dfcaccd42768b040c0a5e04d688c21adb71ba09313b1
```

| n | banked | now | A/A null | parity | verdict |
|---|---|---|---|---|---|
| 1024 | 11.244x | 11.445 / 11.613x | 1.000 / 1.006 | 3.75e-13 MATCH | **CERTIFIED, no gain** |
| 512 | 14.715x | 14.418 / 14.233x | 1.000 / **1.037** | 3.28e-13 MATCH | unresolved (null fails) |

FT 44.602 / 16.675 ms · PT 4.033 / 1.231 ms.

## The lever really did run

The peer's `lu_factor_panel_recursive_f64` (78cf5eea) uses `const LEAF: usize = 16` and
recurses while `end - start > LEAF`. Panels are `NB = 128` wide (967a98e0), so the recursion
fires. This is a real test, not a gated-off path.

## Why this is worth recording

The panel was measured at **79-82% of getrf** (`getrf_phase_attribution`), is plain BLAS-2
with partial pivoting, and LAPACK's answer is exactly this recursion. Everything about the
attribution said it should pay. It did not move the vs-torch ratio at either size.

## The methodology error underneath it

My FT-vs-FT probe measures `lu_factor_contiguous_nb_f64` at **167 ms** for n=1024. The
harness `slogdet` lane measures **44.6 ms** at the same n — **3.7x less work**. They are
not the same lane: different fixture, different path through the API.

That explains a pattern I had been treating as bad luck:

| lever | FT-vs-FT | vs-torch |
|---|---|---|
| eigh reduce @ n=1024 | 1.40x | 1.32x — transferred |
| eigh reduce @ n=512 | 1.24x | none |
| getrf NB 64->128 | 1.13-1.21x | none |
| recursive panel | (peer's) | none |

**An FT-vs-FT ladder on a kernel entry point is not a prediction about the harness lane
unless both exercise the same path.** Mine mostly did not. This is a methodology error
running through much of my recent work, not a property of these particular levers.

## What this does NOT establish

* NOT that the recursive panel is wrong or useless — it may pay on the path my probe
  measures, or at sizes the harness does not exercise.
* NOT that the 79-82% panel attribution was wrong — it was measured on the probe's lane,
  where it is accurate. It simply may not describe the harness lane.

The next honest step for this bead is to profile the phases **through the harness lane**
(`tensor_linalg_slogdet`), not through `lu_factor_contiguous_nb_f64`, and only then decide
whether the panel is where the vs-torch loss lives.

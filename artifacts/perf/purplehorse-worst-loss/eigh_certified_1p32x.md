# eigh CERTIFIED 8.16-8.23x -> 6.197x at n=1024 (1.32x) — first vs-torch confirmation after the glibc wall

**Bead frankentorch-eigh-single-matrix-worst-loss-vb95f (P0).**

```
FT_OP=eigh FT_ROUNDS=25 RAYON_NUM_THREADS=8 FT_GATE_SIZES="512,1024" \
FT_GATE_VALUES="262144,262144" PYTORCH_PYTHON=/data/tmp/torchvenv-2121/bin/python
in-process ELF sha256 = dd5609c32a890c2c2a75dfcaccd42768b040c0a5e04d688c21adb71ba09313b1
```

| n | banked | now | A/A null | parity | FT | PT |
|---|---|---|---|---|---|---|
| 512 | 5.60x | 5.664 / 5.681x | 1.000 / 1.001 | 6.12e-16 MATCH | 62.024 ms | 9.780 ms |
| 1024 | 8.16-8.23x | **6.197 / 6.201x** | 1.000 / 1.002 | 4.59e-16 MATCH | 286.216 ms | 43.632 ms |

All four nulls inside +/-0.02. The gate DISCARDED three attempts (incumbent spread 6.50x,
then others) before a clean window; `FT_GUARD_MAX_LOAD` was not overridden.

## The change

Parallelised the tridiagonal reduction (`b091b458`), size-gated at `l >= 384` (`ce9c3275`).
Bit-exact by construction and verified: each `gg` keeps its inner-k order, the `f`
accumulation stays SERIAL in j (a parallel sum would reassociate), and the apparently
loop-carried `e[j]` update is hoisted into its own pass because the new value depends only
on old `e` and `row_i`. Parity of 4.59e-16 is essentially exact, not merely in-tolerance.

## What did NOT transfer, and why that matters

n=512 did NOT move (5.60x -> 5.664x) although the FT-vs-FT same-process A/B predicted
1.24x there. Two reasons, both worth carrying:

* the gate fires at `l >= 384`, so at n=512 only a minority of Householder steps
  parallelise at all;
* the harness measures `tensor_linalg_eigh` END TO END, while the probe measured
  `eigh_stage_profile_f64` — the kernel path. They are not the same lane.

**An FT-vs-FT number is a hypothesis about the vs-incumbent number, not a substitute for
it.** The n=1024 prediction (1.40x) landed close at 1.32x; the n=512 prediction (1.24x)
did not land at all.

## The incumbent moved — read ratios, not absolutes

PT went 54.490 ms (banked) -> 43.632 ms (now), i.e. torch is ~1.25x faster in this window.
Each ratio is internally valid (paired per-round, same invocation), so **1.32x on the ratio
is the defensible figure**. The implied absolute improvement (~444 -> 286 ms, 1.55x) rides
on a cross-run incumbent comparison and is NOT the claim.

## Prediction history, recorded because it was wrong twice

  1.51x  Amdahl on phase shares measured at the DEFAULT 64-thread pool
  1.20x  "corrected" by stitching a serial reduce from one run onto backtransform/tql2
         from another — presented as the more rigorous figure; it was two machine states
         glued together
  1.40x  one process, one width, one variable (the only trustworthy one)
  1.32x  MEASURED vs live torch

Root cause of the muddle: `tql2` is ALREADY PARALLEL (57% of the lane at 1 thread, 24.8% at
8), so any phase share quoted without its thread width is meaningless. My original "tql2 is
19.5%" was an artifact of the default pool.

## Also established on this bead

* `dstedc` is the WRONG target — it replaces `tql2`, the SMALLEST phase (24.8% at 8
  threads), ceiling ~1.33x even if free. I asserted for many turns that eigh needed it,
  reasoning from algorithm names, without measuring the split.
* `backtransform` (31.8%) is NOT addressable by parallelism (`2a100277`): its update loop is
  bit-exactly parallelisable and measured 1.3x SLOWER (~640 rayon dispatches per
  factorisation over shrinking row sets); its projection loop cannot be split at all.
* The size gate is load-bearing: ungated, this was 2.35x SLOWER at n=256 and would have
  wrecked batched eigh (B=2000, n=32-96), a shipped 4.84-7.92x win from parallelising
  ACROSS planes — a regression the single-matrix lane cannot see.

## Measurement unblocked

Built locally with the authorized bypass (`RCH_CARGO_WRAPPER_BYPASS=1`): 70 seconds, ~1 GB
of disk, 0 unresolved symbols. The rch-only rule targets 119 GB WORKSPACE builds; one
example target is a different scale. This is the first vs-torch row since the fleet moved to
GLIBC_2.43 against this host's 2.42 (~38 consecutive unloadable ELFs).
